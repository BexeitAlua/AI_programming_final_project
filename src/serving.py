"""Request path of the recommendation service: Two-Tower candidates from Qdrant re-ranked by LightGBM.

Standalone on purpose (numpy, pandas, lightgbm, qdrant-client only), so the FastAPI image needs just this file
and models/. The collection must be loaded first: `python -m src.load_qdrant`.

    recommender = Recommender("models", "http://localhost:6333")          # once, at startup
    recommender.recommend([{"app_id": 730, "playtime": 812.5, "is_recommended": True}], top_k=10)
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models


class Recommender:
    def __init__(self, models_dir="models", qdrant_url="http://localhost:6333", api_key=None, collection="games"):
        models_dir = Path(models_dir)
        self.manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
        self.side = dict(np.load(models_dir / "user_side.npz"))
        self.ranker = lgb.Booster(model_file=str(models_dir / "ranker.txt"))
        self.client = QdrantClient(url=qdrant_url, api_key=api_key or None)
        self.collection = collection

    def recommend(self, history: list, top_k: int = 10) -> list:
        """history: [{"app_id", "playtime" (hours), "is_recommended"}], oldest first.

        Returns [{"app_id", "title", "score", "price", "tags"}]; score is None for the popularity fallback,
        used when none of the games in `history` is in the catalog.
        """
        request = pd.DataFrame(history, columns=["app_id", "playtime", "is_recommended"])
        request = request[request["app_id"] > 0].drop_duplicates("app_id", keep="last")
        ids = request["app_id"].astype(int).tolist()
        records = {r.id: r for r in self.client.retrieve(self.collection, ids=ids, with_vectors=True,
                                                         with_payload=["tags"])} if ids else {}
        known = request[request["app_id"].isin(list(records))]
        if known.empty:
            return self.fallback(set(ids), top_k)
        weights = 1 + np.log1p(known["playtime"].to_numpy(float).clip(min=0))
        liked = known["is_recommended"].to_numpy(bool)
        games = [records[a] for a in known["app_id"]]

        # user tower on the last max_len games
        last = slice(-self.manifest["max_len"], None)
        h = np.array([g.vector["two_tower"] for g in games])[last] + self.side["liked_embedding"][liked[last].astype(int)]
        u = (h * weights[last, None]).sum(0) / max(weights[last].sum(), 1e-6)
        u = np.maximum(u @ self.side["mlp1_weight"].T + self.side["mlp1_bias"], 0)
        u = u @ self.side["mlp2_weight"].T + self.side["mlp2_bias"]
        u = u / max(np.linalg.norm(u), 1e-12)

        points = self.client.query_points(
            self.collection, query=u.tolist(), using="two_tower", limit=self.manifest["n_candidates"],
            query_filter=models.Filter(must_not=[models.HasIdCondition(has_id=ids)]),
            search_params=models.SearchParams(exact=True), with_payload=True, with_vectors=["als", "text"]).points

        # ALS factor of the new user from its liked games
        als_score = np.full(len(points), np.nan)
        with_als = [(g, w) for g, w, l in zip(games, weights, liked) if l and "als" in g.vector]
        if with_als:
            Y = np.array([g.vector["als"] for g, _ in with_als])
            c = self.manifest["als"]["alpha"] * np.array([w for _, w in with_als])
            A = self.side["als_yty"] + self.manifest["als"]["regularization"] * np.eye(Y.shape[1]) + (Y.T * (c - 1)) @ Y
            x = np.linalg.solve(A, (c[:, None] * Y).sum(0))
            als_score = np.array([np.dot(p.vector["als"], x) if "als" in p.vector else np.nan for p in points])

        tag_overlap = text_similarity = np.full(len(points), np.nan)
        liked_games = [g for g, l in zip(games, liked) if l]
        if liked_games:
            profile = np.mean([g.vector["text"] for g in liked_games], axis=0)
            text_similarity = np.array([p.vector["text"] for p in points]) @ (profile / np.linalg.norm(profile))
            liked_tags = set().union(*(g.payload["tags"] for g in liked_games))
            tag_overlap = np.array([len(set(p.payload["tags"]) & liked_tags) / max(len(p.payload["tags"]), 1)
                                    for p in points])

        features = pd.DataFrame({
            "tt_score": [p.score for p in points],
            "tt_rank": np.arange(len(points)),
            "als_score": als_score,
            **{name: [p.payload[name] for p in points] for name in
               ("log_reviews", "log_recent_reviews", "positive_share", "log_price", "log_age_days")},
            "tag_overlap": tag_overlap,
            "text_similarity": text_similarity,
            "log_history_len": np.log1p(len(known)),
        })[self.manifest["ranker_features"]]
        scores = self.ranker.predict(features)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [{"app_id": points[i].id, "title": points[i].payload["title"], "score": float(scores[i]),
                 "price": points[i].payload["price"], "tags": points[i].payload["tags"]} for i in order]

    def fallback(self, seen: set, top_k: int) -> list:
        ids = [a for a in self.manifest["fallback_app_ids"] if a not in seen][:top_k]
        points = {p.id: p for p in self.client.retrieve(self.collection, ids=ids, with_payload=True)}
        return [{"app_id": a, "title": points[a].payload["title"], "score": None,
                 "price": points[a].payload["price"], "tags": points[a].payload["tags"]} for a in ids if a in points]
