from __future__ import annotations

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from src.preprocessing import build_user_item_matrix


class BaseRecommender:
    """Common interface for every model: fit on a review history, return top-k app_ids per user."""

    name = "base"

    def fit(self, history: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def recommend(self, user_ids: np.ndarray, seen: dict, k: int) -> np.ndarray:
        """Array [len(user_ids), k] of app_ids in rank order; -1 marks an empty slot.

        `seen` maps user_id -> app_ids the user already reviewed; they must not be recommended.
        """
        raise NotImplementedError


def top_k_unseen(ranking: np.ndarray, seen_items: np.ndarray, k: int) -> np.ndarray:
    """First k items of a global ranking that the user has not seen, padded with -1."""
    head = ranking[: k + len(seen_items)]
    head = head[~np.isin(head, seen_items)][:k]
    out = np.full(k, -1, dtype=np.int64)
    out[: len(head)] = head
    return out


class PopularityRecommender(BaseRecommender):
    """Games with the most positive reviews, optionally only over the last `window_days` of history."""

    def __init__(self, window_days: int | None = None):
        self.window_days = window_days
        self.name = "popularity_all" if window_days is None else f"popularity_{window_days}d"
        self.ranking = np.empty(0, dtype=np.int64)

    def fit(self, history: pd.DataFrame) -> "PopularityRecommender":
        pos = history[history["is_recommended"]]
        if self.window_days is not None:
            start = history["date"].max() - pd.Timedelta(days=self.window_days)
            pos = pos[pos["date"] > start]
        counts = pos.groupby("app_id").size()
        # ties broken by app_id, so the ranking does not depend on the pandas version
        order = np.lexsort((counts.index.to_numpy(), -counts.to_numpy()))
        self.ranking = counts.index.to_numpy()[order].astype(np.int64)
        return self

    def recommend(self, user_ids: np.ndarray, seen: dict, k: int) -> np.ndarray:
        empty = np.empty(0, dtype=np.int64)
        return np.vstack([top_k_unseen(self.ranking, seen.get(u, empty), k) for u in user_ids])


class ALSRecommender(BaseRecommender):
    """Implicit ALS on positive reviews weighted by 1 + log1p(hours).

    Games reviewed in the fit history (liked or not) are never recommended.
    Users without positive reviews get no recommendations (-1); evaluation fills them with a fallback.
    """

    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.01, alpha: float = 1.0,
                 iterations: int = 15):
        self.params = {"factors": factors, "regularization": regularization,
                       "alpha": alpha, "iterations": iterations}

    def fit(self, history: pd.DataFrame) -> "ALSRecommender":
        matrix, self.user_index, item_index = build_user_item_matrix(history)
        self.item_ids = np.array(list(item_index))

        known = history[history["user_id"].isin(list(self.user_index))
                        & history["app_id"].isin(list(item_index))]
        rows = known["user_id"].map(self.user_index)
        cols = known["app_id"].map(item_index)
        self.reviewed = csr_matrix((np.ones(len(known)), (rows, cols)), shape=matrix.shape)

        with threadpool_limits(1, "blas"):
            self.model = AlternatingLeastSquares(**self.params, random_state=42)
            self.model.fit(matrix, show_progress=False)
        return self

    def recommend(self, user_ids: np.ndarray, seen: dict, k: int) -> np.ndarray:
        recs = np.full((len(user_ids), k), -1, dtype=np.int64)
        known = np.array([user in self.user_index for user in user_ids])
        if known.any():
            rows = np.array([self.user_index[user] for user in user_ids[known]])
            ids, _ = self.model.recommend(rows, self.reviewed[rows], N=k)
            recs[known] = self.item_ids[ids]
        return recs
