from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.evaluation import METRICS_DIR, EvalData, build_eval_data, load_split, recommend_all
from src.recommenders import ALSRecommender, ItemCFRecommender, PopularityRecommender
from src.utils import PROCESSED_DIR, RESULTS_DIR, save_json

OUT_DIR = RESULTS_DIR / "error_analysis"
K = 10
K_DEEP = 100
SEGMENTS = [(1, 2, "1-2"), (3, 9, "3-9"), (10, 24, "10-24"), (25, np.inf, "25+")]
HEAD_SHARE, MID_SHARE = 0.01, 0.10   # head = top 1% most liked games of the history, mid = next 9%, tail = rest
NAMES = {"popularity_all": "Popularity", "item_cf": "Item-CF", "als": "ALS", "two_tower": "Two-Tower (id)",
         "two_tower_content": "Two-Tower (content)", "two_tower_hybrid": "Two-Tower (hybrid)"}


# --------------------------------------------------------------------------- models
def _val_params(model: str, names: tuple, defaults: dict) -> dict:
    path = METRICS_DIR / f"val_{model}.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {name: saved.get(name, defaults[name]) for name in names}


def build_model(name: str):
    """Recommender with the hyper-parameters chosen on validation."""
    if name == "popularity_all":
        return PopularityRecommender()
    if name == "item_cf":
        return ItemCFRecommender(**_val_params("item_cf", ("n_neighbors",), {"n_neighbors": 50}))
    if name == "als":
        return ALSRecommender(**_val_params("als", ("factors", "regularization", "alpha", "iterations"),
                                            {"factors": 64, "regularization": 0.01, "alpha": 1.0, "iterations": 15}))
    if name.startswith("two_tower"):
        from src.two_towers import TwoTowerRecommender   # torch is imported only when needed
        variant = {"two_tower": "id", "two_tower_content": "content", "two_tower_hybrid": "hybrid"}[name]
        epochs = _val_params(name, ("epoch",), {"epoch": 10})["epoch"]
        return TwoTowerRecommender(variant, epochs=epochs)
    raise ValueError(f"unknown model {name!r}; choose from {list(NAMES)}")


# --------------------------------------------------------------------------- per-user metrics
def per_user_metrics(recs: np.ndarray, data: EvalData, k: int = K) -> pd.DataFrame:
    """Recall@k, NDCG@k and hit@k for every evaluated user (rows aligned with data.user_ids)."""
    discount = 1 / np.log2(np.arange(2, k + 2))
    rows = []
    for user, relevant, row in zip(data.user_ids, data.relevant, recs[:, :k].tolist()):
        relevant = set(relevant.tolist())
        hits = np.array([app in relevant for app in row], dtype=float)
        ideal = discount[: min(len(relevant), k)].sum()
        rows.append({"user_id": user, "history_len": len(data.seen.get(user, ())),
                     "n_relevant": len(relevant), "recall": hits.sum() / len(relevant),
                     "ndcg": float(hits @ discount[: len(hits)]) / ideal, "hit": float(hits.any())})
    return pd.DataFrame(rows)


def segment_of(history_len: int) -> str:
    for low, high, label in SEGMENTS:
        if low <= history_len <= high:
            return label
    return "0"


def segment_table(users: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per history-length segment."""
    users = users.assign(segment=users["history_len"].map(segment_of))
    order = [label for _, _, label in SEGMENTS]
    table = users.groupby("segment").agg(n_users=("user_id", "size"), recall=("recall", "mean"),
                                         ndcg=("ndcg", "mean"), hit_rate=("hit", "mean"))
    return table.reindex([s for s in order if s in table.index]).reset_index()


# --------------------------------------------------------------------------- game groups
def game_groups(history: pd.DataFrame) -> pd.Series:
    """app_id -> 'head' / 'mid' / 'tail' by number of positive reviews in the history (cold games are absent)."""
    likes = history[history["is_recommended"]].groupby("app_id").size()
    every_game = pd.Series(0, index=pd.Index(history["app_id"].unique(), name="app_id"))
    counts = likes.reindex(every_game.index, fill_value=0).sort_values(ascending=False, kind="stable")
    n_head = max(1, int(round(len(counts) * HEAD_SHARE)))
    n_mid = max(1, int(round(len(counts) * MID_SHARE))) - n_head
    labels = np.array(["tail"] * len(counts), dtype=object)
    labels[:n_head] = "head"
    labels[n_head:n_head + n_mid] = "mid"
    return pd.Series(labels, index=counts.index)


def group_recall(recs: np.ndarray, data: EvalData, groups: pd.Series, ks: tuple = (K, K_DEEP)) -> pd.DataFrame:
    """Micro-averaged recall@k per game group: hits on relevant games of the group / relevant games of the group."""
    group_of = groups.to_dict()
    rows = []
    for k in ks:
        hits, totals = {}, {}
        for relevant, row in zip(data.relevant, recs[:, :k].tolist()):
            top = set(row)
            for app in relevant.tolist():
                g = group_of.get(app, "cold")
                totals[g] = totals.get(g, 0) + 1
                hits[g] = hits.get(g, 0) + (app in top)
        for g in ("head", "mid", "tail", "cold"):
            if totals.get(g):
                rows.append({"k": k, "group": g, "relevant": totals[g], "hits": hits.get(g, 0),
                             "recall": hits.get(g, 0) / totals[g]})
    return pd.DataFrame(rows)


def popularity_bias(recs: np.ndarray, groups: pd.Series, k: int = K) -> dict:
    """Share of the recommended top-k slots taken by head / mid / tail games."""
    top = recs[:, :k].ravel()
    top = top[top >= 0]
    labels = groups.reindex(top).fillna("cold").to_numpy()
    return {g: float((labels == g).mean()) for g in ("head", "mid", "tail")}


# --------------------------------------------------------------------------- miss reasons
def miss_reasons(recs: np.ndarray, data: EvalData) -> dict:
    """Share of relevant games: hit@10, rank 11-100, known but beyond 100, cold (never seen in the history)."""
    catalog = set(data.catalog.tolist())
    counts = {"hit_top10": 0, "rank_11_100": 0, "beyond_100": 0, "cold_game": 0}
    for relevant, row in zip(data.relevant, recs[:, :K_DEEP].tolist()):
        rank = {app: i for i, app in enumerate(row)}
        for app in relevant.tolist():
            if app not in catalog:
                counts["cold_game"] += 1
            elif rank.get(app, K_DEEP) < K:
                counts["hit_top10"] += 1
            elif app in rank:
                counts["rank_11_100"] += 1
            else:
                counts["beyond_100"] += 1
    total = sum(counts.values())
    return {name: n / total for name, n in counts.items()}


# --------------------------------------------------------------------------- examples
def pick_examples(users: pd.DataFrame, n: int = 3) -> list:
    """Users with zero hits@10, one from each of the first segments that has such users (deterministic)."""
    missed = users[users["hit"] == 0].assign(segment=users["history_len"].map(segment_of))
    chosen = []
    for _, _, label in SEGMENTS:
        candidates = missed[(missed["segment"] == label) & (missed["n_relevant"] >= 2)].sort_values("user_id")
        if len(candidates):
            chosen.append(int(candidates["user_id"].iloc[len(candidates) // 2]))
        if len(chosen) == n:
            break
    return chosen


def describe_example(user: int, history: pd.DataFrame, target: pd.DataFrame, rec_row: np.ndarray,
                     data: EvalData, groups: pd.Series, titles: pd.Series, model_label: str) -> str:
    title = lambda a: titles.get(a, f"app {a}")
    catalog = set(data.catalog.tolist())
    hist = history[history["user_id"] == user].sort_values("date")
    liked_next = target[(target["user_id"] == user) & target["is_recommended"]]["app_id"].tolist()
    rank = {app: i + 1 for i, app in enumerate(rec_row.tolist())}
    group_of = groups.to_dict()

    lines = [f"### User {user} ({len(hist)} games in history, {model_label})", "",
             "**History (last 8, oldest first):** " + "; ".join(
                 f"{title(r.app_id)} ({'+' if r.is_recommended else '-'}, {r.hours:.0f} h)" for r in hist.tail(8).itertuples()),
             "", "**Liked in the test period:**", ""]
    for app in liked_next:
        if app not in catalog:
            why = "new game, never seen in the history (cold start)"
        elif app in rank:
            why = f"ranked {rank[app]}"
        else:
            why = f"not in the top-{K_DEEP}"
        lines.append(f"- {title(app)} [{group_of.get(app, 'cold')}]: {why}")
    lines += ["", f"**Top-{K} recommended:** " + "; ".join(
        f"{title(a)} [{group_of.get(a, 'cold')}]" for a in rec_row[:K].tolist() if a >= 0), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- run
def run(models: list, split: str = "test") -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history, target = load_split(split)
    data = build_eval_data(history, target)
    groups = game_groups(history)
    fallback = PopularityRecommender().fit(history)
    titles = pd.read_parquet(PROCESSED_DIR / "games.parquet", columns=["app_id", "title"]).set_index("app_id")["title"]

    segments, group_rows, bias_rows, reason_rows, per_model_users, all_recs = [], [], [], [], {}, {}
    for name in models:
        model = build_model(name)
        print(f"{NAMES.get(name, name)}: fitting on {split} history ...")
        model.fit(history)
        recs, _ = recommend_all(model, data, K_DEEP, fallback=fallback)
        users = per_user_metrics(recs, data)
        label = NAMES.get(name, name)
        per_model_users[name], all_recs[name] = users, recs
        segments.append(segment_table(users).assign(model=label))
        group_rows.append(group_recall(recs, data, groups).assign(model=label))
        bias_rows.append({"model": label, **popularity_bias(recs, groups)})
        reason_rows.append({"model": label, **miss_reasons(recs, data)})

    segments = pd.concat(segments, ignore_index=True)
    game_table = pd.concat(group_rows, ignore_index=True)
    bias = pd.DataFrame(bias_rows)
    reasons = pd.DataFrame(reason_rows)
    segments.to_csv(OUT_DIR / "segments.csv", index=False)
    game_table.to_csv(OUT_DIR / "game_groups.csv", index=False)
    bias.to_csv(OUT_DIR / "popularity_bias.csv", index=False)
    reasons.to_csv(OUT_DIR / "miss_reasons.csv", index=False)

    best = max(models, key=lambda m: per_model_users[m]["ndcg"].mean())
    examples = pick_examples(per_model_users[best])
    rows_of = {u: i for i, u in enumerate(data.user_ids)}
    text = [f"# Error examples ({split}, model: {NAMES.get(best, best)})", "",
            "Groups: head = top 1% most liked games of the history, mid = next 9%, tail = the rest, "
            "cold = not in the history.", ""]
    text += [describe_example(u, history, target, all_recs[best][rows_of[u]], data, groups, titles,
                              NAMES.get(best, best)) for u in examples]
    (OUT_DIR / "examples.md").write_text("\n".join(text), encoding="utf-8")

    plot_segments(segments)
    summary = {"split": split, "models": models, "n_users": int(len(data.user_ids)),
               "group_sizes": groups.value_counts().to_dict(), "best_model": best, "example_users": examples}
    save_json(summary, OUT_DIR / "summary.json")
    return {"segments": segments, "game_groups": game_table, "popularity_bias": bias, "miss_reasons": reasons}


def plot_segments(segments: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = segments["segment"].drop_duplicates().tolist()
    models = segments["model"].drop_duplicates().tolist()
    colors = ["#9aa4b2", "#1b8a7a", "#1b5e8a", "#c77c2e", "#7a4fb3", "#c0392b"]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    for i, model in enumerate(models):
        values = segments[segments["model"] == model].set_index("segment").reindex(labels)["recall"]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width * 0.92, label=model, color=colors[i % len(colors)])
    counts = segments.drop_duplicates("segment").set_index("segment").reindex(labels)["n_users"]
    ax.set_xticks(x, [f"{s}\n({n:,} users)" for s, n in zip(labels, counts)])
    ax.set_xlabel("Games in the user's history")
    ax.set_ylabel(f"Recall@{K}")
    ax.set_title(f"Recall@{K} by length of user history (test period)", loc="left")
    ax.grid(axis="y", color="#e3e6ea")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "figures" / "error_by_segment.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error analysis of the recommenders")
    parser.add_argument("--models", default="popularity_all,item_cf,als",
                        help=f"comma-separated, from: {', '.join(NAMES)}")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    result = run([m.strip() for m in args.models.split(",") if m.strip()], args.split)
    pd.set_option("display.width", 200)
    for title, table in result.items():
        print(f"\n== {title}")
        print(table.round(4).to_string(index=False))
    print(f"\nSaved to {OUT_DIR} and results/figures/error_by_segment.png")
