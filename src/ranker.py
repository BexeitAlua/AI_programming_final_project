"""Second stage: LightGBM LambdaRank re-ranks the top-N candidates of the Two-Tower model.

The ranker is trained on the period right after the fit history: candidate generators are fit on train,
their candidates for val users are labelled with the val likes. For test the generators are refit on
train + val and the same ranker scores their candidates, so labels never come from the evaluated period.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.game_features import review_stats, text_embeddings
from src.recommenders import ALSRecommender, BaseRecommender
from src.two_towers import TwoTowerRecommender
from src.utils import PROCESSED_DIR

FEATURES = ["tt_score", "tt_rank", "als_score", "log_reviews", "log_recent_reviews", "positive_share",
            "log_price", "log_age_days", "tag_overlap", "text_similarity", "log_history_len"]


class RankerRecommender(BaseRecommender):
    name = "two_tower_ranker"

    def __init__(self, generator_params: dict, als_params: dict, n_candidates: int = 200,
                 holdout_share: float = 0.2, chunk_size: int = 4096):
        self.generator_params = generator_params
        self.als_params = als_params
        self.n_candidates = n_candidates
        self.holdout_share = holdout_share
        self.chunk_size = chunk_size

    def fit_generators(self, history: pd.DataFrame) -> "RankerRecommender":
        """Fit Two-Tower and ALS on `history` and compute game and user statistics as of its last day."""
        self.generator = TwoTowerRecommender(**self.generator_params).fit(history)
        self.als = ALSRecommender(**self.als_params).fit(history)
        self.als_col = pd.Series(np.arange(len(self.als.item_ids)), index=self.als.item_ids)
        history_end = history["date"].max()

        games = pd.read_parquet(PROCESSED_DIR / "games.parquet").sort_values("app_id")
        self.app_ids = games["app_id"].to_numpy()
        stats = review_stats(self.app_ids, history_end)
        self.log_reviews = np.log1p(stats["n_reviews"].to_numpy(np.float32))
        self.log_recent_reviews = np.log1p(stats["n_recent"].to_numpy(np.float32))
        self.positive_share = stats["positive_share"].to_numpy(np.float32)
        self.log_price = np.log1p(games["price_original"].to_numpy(np.float32))
        age_days = (history_end - games["date_release"]).dt.days.clip(lower=0)
        self.log_age_days = np.log1p(age_days.to_numpy(np.float32))
        self.text = text_embeddings().reindex(self.app_ids).to_numpy(np.float32)

        meta = pd.read_parquet(PROCESSED_DIR / "games_metadata.parquet").set_index("app_id").reindex(self.app_ids)
        all_tags = sorted({tag for tags in meta["tags"].dropna() for tag in tags})
        tag_index = {tag: i for i, tag in enumerate(all_tags)}
        self.tags = np.zeros((len(self.app_ids), len(all_tags)), dtype=bool)
        for row, game_tags in enumerate(meta["tags"]):
            if isinstance(game_tags, np.ndarray):
                self.tags[row, [tag_index[tag] for tag in game_tags]] = True

        liked = history[history["is_recommended"]]
        self.liked_rows = {user: np.searchsorted(self.app_ids, apps)
                           for user, apps in liked.groupby("user_id")["app_id"]}
        self.history_len = history.groupby("user_id").size()
        return self

    def candidate_features(self, user_ids: np.ndarray, seen: dict) -> pd.DataFrame:
        """One row per (user, candidate): user_pos (index in user_ids), user_id, app_id and FEATURES."""
        parts = []
        for start in range(0, len(user_ids), self.chunk_size):
            users = user_ids[start:start + self.chunk_size]
            recs, scores = self.generator.candidates(users, seen, self.n_candidates)
            rows = np.searchsorted(self.app_ids, np.maximum(recs, 0))

            als_cols = self.als_col.reindex(recs.ravel()).to_numpy().reshape(recs.shape)
            als_users = np.array([self.als.user_index.get(user, -1) for user in users])
            user_factors = self.als.model.user_factors[np.maximum(als_users, 0)]
            item_factors = self.als.model.item_factors[np.nan_to_num(als_cols).astype(np.int64)]
            als_score = np.einsum("ud,ucd->uc", user_factors, item_factors)
            als_score[np.isnan(als_cols) | (als_users < 0)[:, None]] = np.nan

            tag_overlap = np.full(recs.shape, np.nan, dtype=np.float32)
            text_similarity = np.full(recs.shape, np.nan, dtype=np.float32)
            for i, user in enumerate(users):
                liked = self.liked_rows.get(user)
                if liked is None:
                    continue
                profile = self.text[liked].mean(0)
                text_similarity[i] = self.text[rows[i]] @ (profile / np.linalg.norm(profile))
                candidate_tags = self.tags[rows[i]]
                common = (candidate_tags & self.tags[liked].any(0)).sum(1)
                tag_overlap[i] = common / np.maximum(candidate_tags.sum(1), 1)

            part = pd.DataFrame({
                "user_pos": np.repeat(np.arange(start, start + len(users)), self.n_candidates),
                "user_id": np.repeat(users, self.n_candidates),
                "app_id": recs.ravel(),
                "tt_score": scores.ravel(),
                "tt_rank": np.tile(np.arange(self.n_candidates), len(users)),
                "als_score": als_score.ravel(),
                "log_reviews": self.log_reviews[rows].ravel(),
                "log_recent_reviews": self.log_recent_reviews[rows].ravel(),
                "positive_share": self.positive_share[rows].ravel(),
                "log_price": self.log_price[rows].ravel(),
                "log_age_days": self.log_age_days[rows].ravel(),
                "tag_overlap": tag_overlap.ravel(),
                "text_similarity": text_similarity.ravel(),
                "log_history_len": np.log1p(self.history_len.reindex(np.repeat(users, self.n_candidates))
                                            .fillna(0).to_numpy()),
            })
            parts.append(part[part["app_id"] >= 0])
        return pd.concat(parts, ignore_index=True)

    def train_ranker(self, data) -> np.ndarray:
        """Train LambdaRank on candidates of `data.user_ids` labelled with `data.relevant`.

        A random `holdout_share` of users is kept out for early stopping; returns their positions in data.user_ids.
        """
        feats = self.candidate_features(data.user_ids, data.seen)
        relevant = pd.DataFrame({
            "user_id": np.repeat(data.user_ids, [len(r) for r in data.relevant]),
            "app_id": np.concatenate(data.relevant),
            "label": 1,
        })
        feats = feats.merge(relevant, on=["user_id", "app_id"], how="left").fillna({"label": 0})
        has_positive = feats.groupby("user_pos")["label"].transform("max") > 0
        feats = feats[has_positive]

        rng = np.random.default_rng(42)
        holdout = np.flatnonzero(rng.random(len(data.user_ids)) < self.holdout_share)
        is_holdout = feats["user_pos"].isin(holdout)
        train, valid = feats[~is_holdout], feats[is_holdout]

        self.ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=1000, learning_rate=0.05,
                                     num_leaves=31, min_child_samples=100, random_state=42, verbose=-1)
        self.ranker.fit(train[FEATURES], train["label"], group=train.groupby("user_pos").size().to_numpy(),
                        eval_X=valid[FEATURES], eval_y=valid["label"],
                        eval_group=[valid.groupby("user_pos").size().to_numpy()], eval_at=[10],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        print(f"ranker: {len(train)} rows, {train['user_pos'].nunique()} users with a liked candidate, "
              f"best iteration {self.ranker.best_iteration_}")
        return holdout

    def feature_importance(self) -> pd.Series:
        gain = self.ranker.booster_.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=FEATURES).sort_values(ascending=False)

    def recommend(self, user_ids, seen, k) -> np.ndarray:
        feats = self.candidate_features(user_ids, seen)
        feats["score"] = self.ranker.predict(feats[FEATURES])
        feats = feats.sort_values(["user_pos", "score"], ascending=[True, False])
        feats["slot"] = feats.groupby("user_pos").cumcount()
        top = feats[feats["slot"] < k]
        recs = np.full((len(user_ids), k), -1, dtype=np.int64)
        recs[top["user_pos"].to_numpy(), top["slot"].to_numpy()] = top["app_id"].to_numpy()
        return recs
