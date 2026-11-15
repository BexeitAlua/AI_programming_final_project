"""Offline evaluation protocol shared by all recommenders.

* split 'val'  : fit on train,       evaluate on val   (model selection / tuning)
* split 'test' : fit on train + val, evaluate on test  (final numbers)
* relevant items  = target-period reviews with is_recommended = True
* evaluated users = users with history and >= 1 relevant item in the target period
* games from the user's history (liked or not) are excluded from recommendations
* 'warm' game = present in history, 'cold' game = new. ID-based models cannot recommend cold
  games, `recall_ceiling` is the best recall they could reach.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from src.preprocessing import temporal_split
from src.ranker import RankerRecommender
from src.recommenders import ALSRecommender, BaseRecommender, ItemCFRecommender, PopularityRecommender
from src.two_towers import TwoTowerRecommender
from src.utils import PROCESSED_DIR, RESULTS_DIR, save_json

METRICS_DIR = RESULTS_DIR / "metrics"
DEFAULT_KS = (10, 20, 100)


def load_split(split: str) -> tuple:
    """(history, target) for a protocol stage, built from the experiment sample."""
    sample = pd.read_parquet(PROCESSED_DIR / "sample_reviews.parquet")
    train, val, test = temporal_split(sample)
    if split == "val":
        return train, val
    if split == "test":
        return pd.concat([train, val], ignore_index=True), test
    raise ValueError(f"split must be 'val' or 'test', got {split!r}")


@dataclass
class EvalData:
    user_ids: np.ndarray   # evaluated users, sorted
    relevant: list         # per user (aligned with user_ids): relevant app_ids
    seen: dict             # user_id -> app_ids from history, excluded from recommendations
    catalog: np.ndarray    # sorted app_ids present in history ('warm' games)


def build_eval_data(history: pd.DataFrame, target: pd.DataFrame) -> EvalData:
    pos = target[target["is_recommended"]]
    users = np.intersect1d(history["user_id"].unique(), pos["user_id"].unique())
    relevant = pos[pos["user_id"].isin(users)].groupby("user_id")["app_id"].unique()
    hist = history[history["user_id"].isin(users)]
    seen = hist.groupby("user_id")["app_id"].unique().to_dict()
    return EvalData(
        user_ids=users,
        relevant=[np.asarray(a) for a in relevant.reindex(users)],
        seen=seen,
        catalog=np.sort(history["app_id"].unique()),
    )


def evaluate(recs: np.ndarray, data: EvalData, ks: tuple = DEFAULT_KS) -> dict:
    """Recall@K, NDCG@K, HitRate@K, Coverage@K and recall on warm / cold games.

    `recs` is [n_users, >= max(ks)] app_ids in rank order, rows aligned with data.user_ids,
    -1 for empty slots, no duplicates within a row.
    """
    n_users, kmax = len(data.user_ids), max(ks)
    if recs.ndim != 2 or recs.shape[0] != n_users or recs.shape[1] < kmax:
        raise ValueError(f"recs must have shape ({n_users}, >={kmax}), got {recs.shape}")

    catalog = set(data.catalog.tolist())
    relevant_sets = [set(r.tolist()) for r in data.relevant]
    discount = 1 / np.log2(np.arange(2, kmax + 2))

    n_relevant = sum(len(r) for r in relevant_sets)
    n_cold = sum(len(r - catalog) for r in relevant_sets)
    result = {
        "n_users": n_users,
        "n_relevant": n_relevant,
        "cold_share": n_cold / n_relevant,
        "recall_ceiling": np.mean([len(r & catalog) / len(r) for r in relevant_sets]),
    }

    for k in ks:
        recall, ndcg, hit_rate, recall_warm, recall_cold = [], [], [], [], []
        for relevant, row in zip(relevant_sets, recs[:, :k].tolist()):
            hits = [app in relevant for app in row]
            ideal_dcg = discount[: min(len(relevant), k)].sum()
            recall.append(sum(hits) / len(relevant))
            ndcg.append(np.dot(hits, discount[:k]) / ideal_dcg)
            hit_rate.append(any(hits))

            warm, cold = relevant & catalog, relevant - catalog
            if warm:
                recall_warm.append(len(warm.intersection(row)) / len(warm))
            if cold:
                recall_cold.append(len(cold.intersection(row)) / len(cold))

        top = recs[:, :k]
        result[f"recall@{k}"] = np.mean(recall)
        result[f"ndcg@{k}"] = np.mean(ndcg)
        result[f"hit_rate@{k}"] = np.mean(hit_rate)
        result[f"recall_warm@{k}"] = np.mean(recall_warm)
        result[f"recall_cold@{k}"] = np.mean(recall_cold) if recall_cold else float("nan")
        result[f"coverage@{k}"] = len(np.unique(top[top >= 0])) / len(catalog)

    return {key: value if isinstance(value, int) else round(float(value), 5)
            for key, value in result.items()}


def evaluate_model(model: BaseRecommender, data: EvalData, ks: tuple = DEFAULT_KS,
                   batch_size: int = 4096, fallback: BaseRecommender | None = None) -> dict:
    """Recommend in user batches and evaluate.

    Users the model returns nothing for (first slot is -1) get `fallback` recommendations,
    so they stay in the evaluation instead of silently inflating the metrics.
    """
    kmax = max(ks)
    recs = np.vstack([model.recommend(data.user_ids[s:s + batch_size], data.seen, kmax)
                      for s in range(0, len(data.user_ids), batch_size)])
    missing = recs[:, 0] < 0
    if fallback is not None and missing.any():
        recs[missing] = fallback.recommend(data.user_ids[missing], data.seen, kmax)
    metrics = evaluate(recs, data, ks)
    metrics["users_without_recs"] = int(missing.sum())
    metrics["fallback"] = fallback.name if fallback is not None and missing.any() else None
    return metrics


def save_metrics(metrics: dict, model_name: str, split: str) -> None:
    save_json({"model": model_name, "split": split, **metrics},
              METRICS_DIR / f"{split}_{model_name}.json")


def metrics_table(split: str | None = None) -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(METRICS_DIR.glob("*.json"))]
    table = pd.DataFrame(rows)
    if split is not None and len(table):
        table = table[table["split"] == split]
    return table.reset_index(drop=True)


SHOW_COLS = ["model", "split", "recall@10", "ndcg@10", "hit_rate@10", "recall@100",
             "recall_warm@10", "recall_cold@100", "coverage@10", "recall_ceiling"]


def run_popularity(windows: tuple = (None, 30, 90, 180, 365)) -> pd.DataFrame:
    """Tune the popularity window on val (by NDCG@10), then report all-time and best window on test."""
    history, target = load_split("val")
    data = build_eval_data(history, target)
    val_scores = {}
    for window in windows:
        model = PopularityRecommender(window).fit(history)
        metrics = evaluate_model(model, data)
        save_metrics(metrics, model.name, "val")
        val_scores[window] = metrics["ndcg@10"]
    best = max(val_scores, key=val_scores.get)
    print(f"best popularity window on val: {best} days" if best else "best window on val: all-time")

    history, target = load_split("test")
    data = build_eval_data(history, target)
    test_windows = [None] if best is None else [None, best]
    for window in test_windows:
        model = PopularityRecommender(window).fit(history)
        save_metrics(evaluate_model(model, data), model.name, "test")
    return metrics_table()


def run_als(factors=(32, 64, 128), regularizations=(0.01, 0.1, 1.0), alphas=(1, 10, 40)) -> pd.DataFrame:
    """Grid search on val (by NDCG@10), then refit the best ALS on train + val and report test.

    Users without positive reviews in history get all-time popularity.
    """
    history, target = load_split("val")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    grid = []
    for f, reg, alpha in product(factors, regularizations, alphas):
        model = ALSRecommender(f, reg, alpha).fit(history)
        metrics = evaluate_model(model, data, fallback=fallback)
        grid.append({**model.params, **metrics})
        print(f"factors={f:4d} reg={reg:<5} alpha={alpha:<3} ndcg@10={metrics['ndcg@10']:.4f} "
              f"recall@10={metrics['recall@10']:.4f}")
    pd.DataFrame(grid).to_csv(METRICS_DIR / "als_grid_val.csv", index=False)

    best = max(grid, key=lambda row: row["ndcg@10"])
    save_metrics(best, "als", "val")
    params = {name: best[name] for name in ("factors", "regularization", "alpha", "iterations")}
    print(f"best ALS on val: {params}")

    history, target = load_split("test")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    model = ALSRecommender(**params).fit(history)
    save_metrics({**params, **evaluate_model(model, data, fallback=fallback)}, "als", "test")
    return metrics_table()


def run_item_cf(neighbors: tuple = (10, 20, 50, 100, 200)) -> pd.DataFrame:
    """Tune the number of neighbours per game on val (by NDCG@10), then refit on train + val and report test.

    Users without positive reviews in history get all-time popularity.
    """
    history, target = load_split("val")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    grid = []
    for n in neighbors:
        model = ItemCFRecommender(n_neighbors=n).fit(history)
        metrics = evaluate_model(model, data, fallback=fallback)
        grid.append({**model.params, **metrics})
        print(f"n_neighbors={n:4d} ndcg@10={metrics['ndcg@10']:.4f} recall@10={metrics['recall@10']:.4f}")
    pd.DataFrame(grid).to_csv(METRICS_DIR / "item_cf_grid_val.csv", index=False)

    best = max(grid, key=lambda row: row["ndcg@10"])
    save_metrics(best, "item_cf", "val")
    print(f"best Item-CF on val: n_neighbors={best['n_neighbors']}")

    history, target = load_split("test")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    model = ItemCFRecommender(n_neighbors=best["n_neighbors"]).fit(history)
    save_metrics({**model.params, **evaluate_model(model, data, fallback=fallback)}, "item_cf", "test")
    return metrics_table()


def run_two_tower(variant: str = "id", use_review_stats: bool = True, max_epochs: int = 40) -> pd.DataFrame:
    """Train on train epoch by epoch and pick the epoch with the best NDCG@10 on val,
    then refit on train + val for that many epochs and report test.

    variant: "id", "content" or "hybrid" (see TwoTowerRecommender); use_review_stats=False drops the review
    count and positive share from the game features (ablation). Users without history get all-time popularity.
    """
    history, target = load_split("val")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    model = TwoTowerRecommender(variant, use_review_stats=use_review_stats)
    model.prepare(history)
    curve = []
    for epoch in range(1, max_epochs + 1):
        loss = model.train_epoch()
        metrics = evaluate_model(model, data, fallback=fallback)
        curve.append({"epoch": epoch, "loss": round(loss, 5), **metrics})
        print(f"epoch={epoch:2d} loss={loss:.4f} ndcg@10={metrics['ndcg@10']:.4f} "
              f"recall@10={metrics['recall@10']:.4f} recall@100={metrics['recall@100']:.4f}")
    pd.DataFrame(curve).to_csv(METRICS_DIR / f"{model.name}_epochs_val.csv", index=False)

    best = max(curve, key=lambda row: row["ndcg@10"])
    save_metrics(best, model.name, "val")
    print(f"best epoch on val: {best['epoch']}")

    history, target = load_split("test")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    model = TwoTowerRecommender(variant, use_review_stats=use_review_stats, epochs=best["epoch"]).fit(history)
    save_metrics({**model.params, **evaluate_model(model, data, fallback=fallback)}, model.name, "test")
    return metrics_table()


def run_ranker(n_candidates: int = 200) -> pd.DataFrame:
    """Two-Tower (content, best epoch on val) candidates re-ranked by LightGBM LambdaRank.

    val : generators fit on train, ranker trained on val likes of 80% of users; the other 20% are used
          for early stopping and for the comparison with the plain Two-Tower order of the same candidates.
    test: generators refit on train + val, the same ranker scores their candidates.
    """
    two_tower = json.loads((METRICS_DIR / "val_two_tower_content.json").read_text())
    als = json.loads((METRICS_DIR / "val_als.json").read_text())
    model = RankerRecommender(
        generator_params={"variant": "content", "epochs": two_tower["epoch"]},
        als_params={name: als[name] for name in ("factors", "regularization", "alpha", "iterations")},
        n_candidates=n_candidates)

    history, target = load_split("val")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    model.fit_generators(history)
    holdout = model.train_ranker(data)
    holdout_data = EvalData(user_ids=data.user_ids[holdout], relevant=[data.relevant[i] for i in holdout],
                            seen=data.seen, catalog=data.catalog)
    rows = []
    for name, recommender in (("two_tower_content", model.generator), (model.name, model)):
        metrics = evaluate_model(recommender, holdout_data, fallback=fallback)
        rows.append({"model": name, **metrics})
        print(f"val holdout {name:18s} ndcg@10={metrics['ndcg@10']:.4f} recall@10={metrics['recall@10']:.4f} "
              f"recall@100={metrics['recall@100']:.4f}")
    pd.DataFrame(rows).to_csv(METRICS_DIR / "ranker_holdout_val.csv", index=False)
    save_metrics(rows[-1], model.name, "val")
    model.feature_importance().rename("gain_share").to_csv(METRICS_DIR / "ranker_feature_importance.csv")

    history, target = load_split("test")
    data = build_eval_data(history, target)
    fallback = PopularityRecommender().fit(history)
    model.fit_generators(history)
    metrics = evaluate_model(model, data, fallback=fallback)
    save_metrics({"n_candidates": n_candidates, "generator_epochs": two_tower["epoch"],
                  "ranker_trees": model.ranker.best_iteration_, **metrics}, model.name, "test")
    return metrics_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommenders on the temporal split")
    parser.add_argument("--model", choices=["popularity", "item_cf", "als", "two_tower", "ranker"], default="popularity")
    parser.add_argument("--variant", choices=["id", "content", "hybrid"], default="id",
                        help="Two-Tower item features")
    parser.add_argument("--no-review-stats", action="store_true",
                        help="Two-Tower content/hybrid without review count and positive share")
    args = parser.parse_args()
    runners = {"popularity": run_popularity, "item_cf": run_item_cf, "als": run_als, "ranker": run_ranker,
               "two_tower": lambda: run_two_tower(args.variant, not args.no_review_stats)}
    table = runners[args.model]()
    pd.set_option("display.width", 200)
    print(table[[c for c in SHOW_COLS if c in table.columns]].to_string(index=False))
