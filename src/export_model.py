"""Train the final models on all data (up to 2022-12-31) and export what the FastAPI service needs.

models/items.parquet  one row per catalog game: Qdrant payload + vectors two_tower (64), als (64, empty for
                      games without positive reviews in the sample) and text (384)
models/user_side.npz  user tower weights (liked embedding, user MLP) and ALS Y^T Y to fold in new users
models/ranker.txt     LightGBM LambdaRank
models/manifest.json  parameters, ranker feature order, popularity fallback and a control example

The ranker is trained one period behind, like in the evaluation: generators fit on train + val, labels are
the 2022 likes. Two-Tower and ALS are then refit on all data for serving. `serve_reference` is the request
path of the service written in numpy; FastAPI + Qdrant must return the same answer for the control example.

Run: uv run python -m src.export_model
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src.evaluation import METRICS_DIR, build_eval_data, load_split
from src.preprocessing import build_user_item_matrix
from src.ranker import FEATURES, RankerRecommender
from src.recommenders import PopularityRecommender
from src.utils import PROCESSED_DIR

MODELS_DIR = Path("models")
N_CANDIDATES = 200
FALLBACK_SIZE = 100
GAME_FEATURES = ["log_reviews", "log_recent_reviews", "positive_share", "log_price", "log_age_days"]


def catalog_arrays(items: pd.DataFrame) -> dict:
    """What the service reads from Qdrant, as arrays aligned with the rows of items.parquet."""
    has_als = items["als"].notna().to_numpy()
    als = np.full((len(items), len(items.loc[has_als, "als"].iloc[0])), np.nan, dtype=np.float32)
    als[has_als] = np.stack(items.loc[has_als, "als"])
    return {
        "app_ids": items["app_id"].to_numpy(),
        "row_of": {int(app_id): row for row, app_id in enumerate(items["app_id"])},
        "two_tower": np.stack(items["two_tower"]).astype(np.float32),
        "als": als,
        "has_als": has_als,
        "text": np.stack(items["text"]).astype(np.float32),
        "tags": [set(tags) for tags in items["tags"]],
        "game_features": items[GAME_FEATURES].to_numpy(np.float32),
    }


def serve_reference(request: pd.DataFrame, cat: dict, side: dict, ranker: lgb.Booster, manifest: dict,
                    top_k: int) -> list:
    """Request path of the service. `request`: app_id, playtime (hours), is_recommended; oldest first."""
    seen = set(request["app_id"].astype(int))
    known = request[request["app_id"].isin(cat["row_of"])].drop_duplicates("app_id", keep="last")
    if known.empty:
        return [a for a in manifest["fallback_app_ids"] if a not in seen][:top_k]
    rows = np.array([cat["row_of"][int(a)] for a in known["app_id"]])
    weights = 1 + np.log1p(known["playtime"].to_numpy(np.float64).clip(min=0))
    liked = known["is_recommended"].to_numpy(bool)

    # user tower on the last max_len games
    last = slice(-manifest["max_len"], None)
    h = cat["two_tower"][rows[last]] + side["liked_embedding"][liked[last].astype(int)]
    u = (h * weights[last, None]).sum(0) / max(weights[last].sum(), 1e-6)
    u = np.maximum(u @ side["mlp1_weight"].T + side["mlp1_bias"], 0) @ side["mlp2_weight"].T + side["mlp2_bias"]
    u = u / max(np.linalg.norm(u), 1e-12)

    # candidates: exact dot-product search without the games of the request
    scores = cat["two_tower"] @ u
    scores[[cat["row_of"][a] for a in seen if a in cat["row_of"]]] = -np.inf
    candidates = np.argsort(-scores, kind="stable")[:manifest["n_candidates"]]

    # ALS factor of the new user from its liked games (one least squares solve)
    als_score = np.full(len(candidates), np.nan)
    with_als = liked & cat["has_als"][rows]
    if with_als.any():
        Y = cat["als"][rows[with_als]].astype(np.float64)
        c = manifest["als"]["alpha"] * weights[with_als]
        A = side["als_yty"] + manifest["als"]["regularization"] * np.eye(Y.shape[1]) + (Y.T * (c - 1)) @ Y
        x = np.linalg.solve(A, (c[:, None] * Y).sum(0))
        als_score = cat["als"][candidates] @ x

    tag_overlap = np.full(len(candidates), np.nan)
    text_similarity = np.full(len(candidates), np.nan)
    if liked.any():
        profile = cat["text"][rows[liked]].mean(0)
        text_similarity = cat["text"][candidates] @ (profile / np.linalg.norm(profile))
        liked_tags = set().union(*(cat["tags"][r] for r in rows[liked]))
        tag_overlap = np.array([len(cat["tags"][r] & liked_tags) / max(len(cat["tags"][r]), 1) for r in candidates])

    features = pd.DataFrame(cat["game_features"][candidates], columns=GAME_FEATURES)
    features["tt_score"] = scores[candidates]
    features["tt_rank"] = np.arange(len(candidates))
    features["als_score"] = als_score
    features["tag_overlap"] = tag_overlap
    features["text_similarity"] = text_similarity
    features["log_history_len"] = np.log1p(len(known))
    order = np.argsort(-ranker.predict(features[FEATURES]), kind="stable")[:top_k]
    return cat["app_ids"][candidates[order]].tolist()


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    train, val = load_split("val")
    _, test = load_split("test")
    train_val = pd.concat([train, val], ignore_index=True)
    history = pd.concat([train, val, test], ignore_index=True)
    two_tower = json.loads((METRICS_DIR / "val_two_tower_content.json").read_text())
    als = json.loads((METRICS_DIR / "val_als.json").read_text())
    als_params = {name: als[name] for name in ("factors", "regularization", "alpha", "iterations")}
    model = RankerRecommender({"variant": "content", "epochs": two_tower["epoch"]}, als_params, N_CANDIDATES)

    print("ranker: generators on train + val, labels from 2022")
    model.fit_generators(train_val)
    model.train_ranker(build_eval_data(train_val, test))
    model.ranker.booster_.save_model(MODELS_DIR / "ranker.txt", num_iteration=model.ranker.best_iteration_)

    print("serving models: Two-Tower and ALS on all data")
    model.fit_generators(history)
    generator, als_model = model.generator, model.als
    generator.model.eval()
    with torch.no_grad():
        all_items = torch.arange(len(generator.item_ids), device=generator.device)
        item_vectors = generator.model.item_tower(all_items).cpu().numpy()[2:]
    catalog = generator.item_ids[2:].numpy()

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet").set_index("app_id").reindex(catalog)
    meta = pd.read_parquet(PROCESSED_DIR / "games_metadata.parquet").set_index("app_id").reindex(catalog)
    game_rows = np.searchsorted(model.app_ids, catalog)
    als_vectors = pd.Series(list(als_model.model.item_factors), index=als_model.item_ids).reindex(catalog)
    items = pd.DataFrame({
        "app_id": catalog,
        "title": games["title"].to_numpy(),
        "price": games["price_final"].to_numpy(),
        "tags": [list(t) if isinstance(t, np.ndarray) else [] for t in meta["tags"]],
        "date_release": games["date_release"].dt.strftime("%Y-%m-%d").to_numpy(),
        "log_reviews": model.log_reviews[game_rows],
        "log_recent_reviews": model.log_recent_reviews[game_rows],
        "positive_share": model.positive_share[game_rows],
        "log_price": model.log_price[game_rows],
        "log_age_days": model.log_age_days[game_rows],
        "two_tower": list(item_vectors),
        "als": [None if v is None or (np.isscalar(v) and np.isnan(v)) else v for v in als_vectors.to_numpy()],
        "text": list(model.text[game_rows]),
    })
    items.to_parquet(MODELS_DIR / "items.parquet", index=False)

    user_mlp = generator.model.user_mlp
    item_factors = als_model.model.item_factors.astype(np.float64)
    side = {
        "liked_embedding": generator.model.liked_embedding.weight,
        "mlp1_weight": user_mlp[0].weight, "mlp1_bias": user_mlp[0].bias,
        "mlp2_weight": user_mlp[2].weight, "mlp2_bias": user_mlp[2].bias,
    }
    side = {name: value.detach().cpu().numpy() for name, value in side.items()}
    side["als_yty"] = item_factors.T @ item_factors
    np.savez(MODELS_DIR / "user_side.npz", **side)

    popularity = PopularityRecommender().fit(history)
    manifest = {
        "data_end": str(history["date"].max().date()),
        "max_len": generator.params["max_len"],
        "dim": generator.params["dim"],
        "n_candidates": N_CANDIDATES,
        "ranker_features": FEATURES,
        "two_tower": generator.params,
        "als": als_params,
        "fallback_app_ids": popularity.ranking[:FALLBACK_SIZE].tolist(),
        "qdrant": {"collection": "games", "distance": "Dot",
                   "vectors": {"two_tower": items["two_tower"].iloc[0].shape[0], "als": als_params["factors"],
                               "text": model.text.shape[1]}},
    }

    # check 1: ALS fold-in equals implicit's own recalculation for users of the training matrix
    matrix, user_index, _ = build_user_item_matrix(history)
    rng = np.random.default_rng(0)
    worst = 0.0
    for row in rng.choice(matrix.shape[0], 20, replace=False):
        Y, c = item_factors[matrix[row].indices], als_params["alpha"] * matrix[row].data
        A = side["als_yty"] + als_params["regularization"] * np.eye(Y.shape[1]) + (Y.T * (c - 1)) @ Y
        x = np.linalg.solve(A, (c[:, None] * Y).sum(0))
        worst = max(worst, np.abs(x - als_model.model.recalculate_user(row, matrix[row])).max())
    print(f"ALS fold-in vs implicit.recalculate_user: max abs diff {worst:.2e}")

    # check 2: the numpy request path agrees with the offline recommender
    ranker = lgb.Booster(model_file=str(MODELS_DIR / "ranker.txt"))
    cat = catalog_arrays(pd.read_parquet(MODELS_DIR / "items.parquet"))
    ordered = history.sort_values(["user_id", "date", "review_id"])
    by_user = {u: g for u, g in ordered.groupby("user_id")}
    users = rng.choice(np.array(list(generator.histories)), 300, replace=False)
    offline = model.recommend(users, {u: by_user[u]["app_id"].to_numpy() for u in users}, 10)
    overlaps = []
    for user, expected in zip(users, offline):
        request = by_user[user].rename(columns={"hours": "playtime"})[["app_id", "playtime", "is_recommended"]]
        got = serve_reference(request, cat, side, ranker, manifest, 10)
        overlaps.append(len(set(got) & set(expected)) / 10)
    manifest["parity_top10_overlap"] = round(float(np.mean(overlaps)), 4)
    print(f"numpy request path vs offline recommender: top-10 overlap {np.mean(overlaps):.3f} "
          "(differences come from ALS: offline uses the stored user factor, the service folds the user in)")

    lengths = ordered.groupby("user_id").size()
    control_user = int(lengths[lengths.between(15, 25)].index[0])
    request = by_user[control_user].rename(columns={"hours": "playtime"})[["app_id", "playtime", "is_recommended"]]
    request = request.assign(playtime=request["playtime"].round(1))
    expected = serve_reference(request, cat, side, ranker, manifest, 10)
    titles = dict(zip(items["app_id"], items["title"]))
    manifest["control_example"] = {
        "request": {"history": [{"app_id": int(r.app_id), "playtime": float(r.playtime),
                                 "is_recommended": bool(r.is_recommended)} for r in request.itertuples()],
                    "top_k": 10},
        "expected": [{"app_id": int(a), "title": titles[a]} for a in expected],
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in sorted(MODELS_DIR.iterdir()):
        print(f"{path}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
