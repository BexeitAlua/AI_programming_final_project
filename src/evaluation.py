"""Offline evaluation protocol shared by all recommenders.

Protocol
--------
* split 'val'  : fit on train,       evaluate on val   (model selection / tuning)
* split 'test' : fit on train + val, evaluate on test  (final numbers)
* relevant items  = target-period reviews with is_recommended = True
* evaluated users = users with history and >= 1 relevant item in the target period
* every game from the user's history (positive or negative) is excluded from recommendations
* metrics are also split by game type: 'warm' = game present in history, 'cold' = new game.
  ID-based models cannot recommend cold games, `recall_ceiling` shows the resulting upper bound.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing import temporal_split
from src.recommenders import BaseRecommender, PopularityRecommender
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
    """Recall@K, NDCG@K, HitRate@K, Coverage@K (+ warm/cold recall).

    `recs` is [n_users, >= max(ks)] app_ids in rank order, rows aligned with data.user_ids,
    -1 for empty slots, no duplicates within a row.
    """
    n, kmax = len(data.user_ids), max(ks)
    if recs.ndim != 2 or recs.shape[0] != n or recs.shape[1] < kmax:
        raise ValueError(f"recs must have shape ({n}, >={kmax}), got {recs.shape}")

    rel = pd.DataFrame({
        "row": np.repeat(np.arange(n), [len(r) for r in data.relevant]),
        "app_id": np.concatenate(data.relevant).astype(np.int64),
    })
    rel["warm"] = np.isin(rel["app_id"].to_numpy(), data.catalog).astype(float)
    n_rel = np.bincount(rel["row"], minlength=n)
    n_warm = np.bincount(rel["row"], weights=rel["warm"], minlength=n)
    n_cold = n_rel - n_warm

    ranked = pd.DataFrame({
        "row": np.repeat(np.arange(n), kmax),
        "rank": np.tile(np.arange(kmax), n),
        "app_id": recs[:, :kmax].astype(np.int64).ravel(),
    })
    hits = ranked.merge(rel, on=["row", "app_id"])  # one line per hit, with its rank

    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    idcg_cum = np.cumsum(disc)
    has_warm, has_cold = n_warm > 0, n_cold > 0

    out = {
        "n_users": int(n),
        "n_relevant": int(n_rel.sum()),
        "cold_share": float(n_cold.sum() / n_rel.sum()),
        "recall_ceiling": float(np.mean(n_warm / n_rel)),
    }
    for k in ks:
        h = hits[hits["rank"] < k]
        n_hits = np.bincount(h["row"], minlength=n)
        dcg = np.bincount(h["row"], weights=disc[h["rank"]], minlength=n)
        idcg = idcg_cum[np.minimum(n_rel, k) - 1]
        hits_warm = np.bincount(h["row"], weights=h["warm"], minlength=n)
        hits_cold = n_hits - hits_warm
        top = recs[:, :k]
        out[f"recall@{k}"] = float(np.mean(n_hits / n_rel))
        out[f"ndcg@{k}"] = float(np.mean(dcg / idcg))
        out[f"hit_rate@{k}"] = float(np.mean(n_hits > 0))
        out[f"recall_warm@{k}"] = float(np.mean(hits_warm[has_warm] / n_warm[has_warm]))
        out[f"recall_cold@{k}"] = (float(np.mean(hits_cold[has_cold] / n_cold[has_cold]))
                                   if has_cold.any() else float("nan"))
        out[f"coverage@{k}"] = float(len(np.unique(top[top >= 0])) / len(data.catalog))
    return {key: round(v, 5) if isinstance(v, float) else v for key, v in out.items()}


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
             "recall_warm@10", "coverage@10", "recall_ceiling"]


def run_popularity(windows: tuple = (None, 30, 90, 180, 365), ks: tuple = DEFAULT_KS) -> pd.DataFrame:
    """Tune the popularity window on val (by NDCG@10), then report all-time and best window on test."""
    history, target = load_split("val")
    data = build_eval_data(history, target)
    val_scores = {}
    for window in windows:
        model = PopularityRecommender(window).fit(history)
        metrics = evaluate_model(model, data, ks)
        save_metrics(metrics, model.name, "val")
        val_scores[window] = metrics["ndcg@10"]
    best = max(val_scores, key=val_scores.get)
    print(f"best popularity window on val: {best} days" if best else "best window on val: all-time")

    history, target = load_split("test")
    data = build_eval_data(history, target)
    for window in dict.fromkeys((None, best)):
        model = PopularityRecommender(window).fit(history)
        save_metrics(evaluate_model(model, data, ks), model.name, "test")
    return metrics_table()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommenders on the temporal split")
    parser.add_argument("--model", choices=["popularity"], default="popularity")
    args = parser.parse_args()
    if args.model == "popularity":
        table = run_popularity()
    pd.set_option("display.width", 200)
    print(table[[c for c in SHOW_COLS if c in table.columns]].to_string(index=False))
