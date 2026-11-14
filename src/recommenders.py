from __future__ import annotations

import numpy as np
import pandas as pd


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
