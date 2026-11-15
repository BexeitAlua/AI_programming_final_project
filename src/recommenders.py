from __future__ import annotations

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, diags, vstack
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


class ItemCFRecommender(BaseRecommender):
    """Item-based collaborative filtering with cosine similarity between games.

    Two games are similar when the same users liked them: cosine similarity between the game columns
    of the user x game matrix of positive reviews (weights 1 + log1p(hours)). For every game only its
    `n_neighbors` most similar games are kept, so the game x game matrix stays sparse.

    A user's score for a game = sum over the games they liked of (their weight x similarity).
    Games reviewed in the fit history (liked or not) are never recommended. Slots left empty because the
    liked games have too few neighbours are filled with the most popular unseen games.
    Users without positive reviews get no recommendations (-1); evaluation fills them with a fallback.
    """

    name = "item_cf"

    def __init__(self, n_neighbors: int = 50, block_size: int = 2000):
        self.params = {"n_neighbors": n_neighbors}
        self.block_size = block_size

    def fit(self, history: pd.DataFrame) -> "ItemCFRecommender":
        self.user_matrix, self.user_index, item_index = build_user_item_matrix(history)
        self.item_ids = np.array(list(item_index))
        self.item_index = item_index

        known = history[history["user_id"].isin(list(self.user_index))
                        & history["app_id"].isin(list(item_index))]
        rows = known["user_id"].map(self.user_index)
        cols = known["app_id"].map(item_index)
        self.reviewed = csr_matrix((np.ones(len(known)), (rows, cols)), shape=self.user_matrix.shape)

        norms = np.sqrt(self.user_matrix.multiply(self.user_matrix).sum(axis=0)).A1
        normalized = (self.user_matrix @ diags(1 / np.where(norms > 0, norms, 1))).tocsc()
        # game x game similarity, computed in blocks of games so memory stays bounded
        blocks = []
        for start in range(0, normalized.shape[1], self.block_size):
            block = (normalized[:, start:start + self.block_size].T @ normalized).tocsr()
            blocks.append(self._top_neighbors(block, start))
        self.similarity = vstack(blocks).tocsr().astype(np.float32)

        self.popular = PopularityRecommender().fit(history).ranking
        return self

    def _top_neighbors(self, block: csr_matrix, offset: int) -> csr_matrix:
        """Keep the n_neighbors largest similarities in every row, drop each game's similarity to itself."""
        n = self.params["n_neighbors"]
        indptr, indices, data = [0], [], []
        for row in range(block.shape[0]):
            lo, hi = block.indptr[row], block.indptr[row + 1]
            cols, vals = block.indices[lo:hi], block.data[lo:hi]
            keep = (cols != offset + row) & (vals > 0)
            cols, vals = cols[keep], vals[keep]
            if len(vals) > n:
                top = np.argpartition(-vals, n - 1)[:n]
                cols, vals = cols[top], vals[top]
            indices.append(cols)
            data.append(vals)
            indptr.append(indptr[-1] + len(cols))
        return csr_matrix((np.concatenate(data) if data else [], np.concatenate(indices) if indices else [],
                           indptr), shape=block.shape)

    def recommend(self, user_ids: np.ndarray, seen: dict, k: int, batch_size: int = 1024) -> np.ndarray:
        recs = np.full((len(user_ids), k), -1, dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        for start in range(0, len(user_ids), batch_size):
            batch = user_ids[start:start + batch_size]
            known = np.flatnonzero([user in self.user_index for user in batch])
            if not len(known):
                continue
            rows = np.array([self.user_index[batch[i]] for i in known])
            scores = (self.user_matrix[rows] @ self.similarity).toarray()
            scores[self.reviewed[rows].nonzero()] = 0          # never recommend reviewed games
            top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for i, user_row, order in zip(known, scores, top):
                chosen = self.item_ids[order[user_row[order] > 0]]
                if len(chosen) < k:                             # fill with popular unseen games
                    exclude = np.concatenate([seen.get(batch[i], empty), chosen])
                    chosen = np.concatenate([chosen, top_k_unseen(self.popular, exclude, k - len(chosen))])
                recs[start + i] = chosen[:k]
        return recs

    def similar_items(self, app_id: int, k: int = 10) -> pd.DataFrame:
        """The k games most similar to `app_id` (columns app_id, similarity); empty for unknown games."""
        if app_id not in self.item_index:
            return pd.DataFrame({"app_id": pd.Series(dtype=np.int64), "similarity": pd.Series(dtype=float)})
        row = self.similarity[self.item_index[app_id]]
        order = np.argsort(-row.data, kind="stable")[:k]
        return pd.DataFrame({"app_id": self.item_ids[row.indices[order]],
                             "similarity": row.data[order].astype(float)})
