import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from src.game_features import build_catalog, build_game_features
from src.recommenders import BaseRecommender

PAD = 0
UNK = 1

def build_item_index(history, min_count) -> tuple[dict, torch.Tensor]:
    item_ids: list = [-1, -1]
    item_index: dict = {}
    k = 2
    review_count = history["app_id"].value_counts()

    for app_id in sorted(review_count.index):
        if review_count[app_id] >= min_count:
            item_ids.append(int(app_id))
            item_index[int(app_id)] = k
            k += 1
    return item_index, torch.tensor(item_ids)

def encode_history(app_ids, hours, liked, item_index, max_len) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    app_ids = app_ids[-max_len:]
    hours = hours[-max_len:]
    liked = liked[-max_len:]

    items: list = [0] * max_len
    weights: list = [0.0] * max_len
    likes: list = [0] * max_len

    n = len(app_ids)

    for i in range(n):
        items[i] = item_index.get(int(app_ids[i]), UNK)
        weights[i] = 1+np.log1p(hours[i])
        likes[i] = int(liked[i])

    items = torch.tensor(items, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float32)
    likes = torch.tensor(likes, dtype=torch.long)

    return items, weights, likes

def build_training_examples(history, item_index, max_len) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = history.sort_values(by=["user_id", "date", "review_id"]) 

    items: list = []
    weights: list = []
    liked: list = []
    targets: list = []

    for user_id, user_reviews in history.groupby("user_id"):
        app_ids = user_reviews["app_id"].to_numpy()
        hours = user_reviews["hours"].to_numpy()
        likes = user_reviews["is_recommended"].to_numpy()

        for t in range(1, len(app_ids)):
            if likes[t] and int(app_ids[t]) in item_index:
                ctx_items, ctx_weights, ctx_liked = encode_history(app_ids[:t], hours[:t], likes[:t], item_index, max_len)
                items.append(ctx_items)
                weights.append(ctx_weights)
                liked.append(ctx_liked)
                targets.append(item_index[int(app_ids[t])])

    items = torch.stack(items)
    weights = torch.stack(weights)
    liked = torch.stack(liked)
    targets = torch.tensor(targets)
    return items, weights, liked, targets

class TwoTowerModel(nn.Module):
    """Item tower: [ID embedding] + [tags, text, numeric features] -> MLP -> L2 norm.

    features=None gives the ID-only model, use_id=False the content-only model.
    In the hybrid, rare games share the UNK id (id_of_item), and during training every id is replaced
    by UNK with probability id_dropout, so the content part learns to describe a game on its own.
    """

    def __init__(self, n_items, dim, use_id=True, id_of_item=None, features=None, id_dropout=0.0):
        super().__init__()
        self.use_id = use_id
        self.use_content = features is not None
        self.id_dropout = id_dropout
        n_parts = 0
        if self.use_id:
            self.item_embedding = nn.Embedding(n_items, dim, padding_idx=PAD)
            self.register_buffer("id_of_item", torch.arange(n_items) if id_of_item is None else id_of_item)
            n_parts += 1
        if self.use_content:
            self.register_buffer("tags", features["tags"])
            self.register_buffer("text", features["text"])
            self.register_buffer("numeric", features["numeric"])
            n_tags = int(features["tags"].max()) + 1
            self.tag_embedding = nn.EmbeddingBag(n_tags, dim, mode="mean", padding_idx=PAD)
            self.text_linear = nn.Linear(features["text"].shape[1], dim)
            self.numeric_linear = nn.Linear(features["numeric"].shape[1], dim)
            n_parts += 3
        self.item_mlp = nn.Sequential(nn.Linear(n_parts * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.liked_embedding = nn.Embedding(2, dim)
        nn.init.zeros_(self.liked_embedding.weight)
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def item_tower(self, items):
        parts = []
        if self.use_id:
            ids = self.id_of_item[items]
            if self.training and self.id_dropout > 0:
                drop = torch.rand(ids.shape, device=ids.device) < self.id_dropout
                ids = ids.masked_fill(drop & (ids != PAD), UNK)
            parts.append(self.item_embedding(ids))
        if self.use_content:
            tags = self.tags[items]
            tag_vecs = self.tag_embedding(tags.reshape(-1, tags.shape[-1]))
            parts.append(tag_vecs.reshape(*items.shape, -1))
            parts.append(self.text_linear(self.text[items]))
            parts.append(self.numeric_linear(self.numeric[items]))
        x = self.item_mlp(torch.cat(parts, dim=-1))
        return F.normalize(x, dim=-1)

    def user_tower(self, items, weights, liked):
        h = self.item_tower(items) + self.liked_embedding(liked)
        numerator = (h * weights.unsqueeze(-1)).sum(1)
        denominator = weights.sum(1, keepdim=True).clamp(min=1e-6)
        u = numerator / denominator
        u = self.user_mlp(u)
        return F.normalize(u, dim=-1)


def in_batch_softmax_loss(user_vecs, item_vecs, targets, log_q, temperature) -> torch.Tensor:
    B = len(targets)
    logits = user_vecs @ item_vecs.T / temperature
    logits = logits - log_q[targets][None, :]

    same = targets[:, None] == targets[None, :]
    mask = same & ~torch.eye(B, dtype=torch.bool, device=targets.device)
    logits = logits.masked_fill(mask, float("-inf"))

    return F.cross_entropy(logits, torch.arange(B, device=targets.device))


class TwoTowerRecommender(BaseRecommender):
    """variant "id": ID embeddings of games with >= min_count reviews in history;
    "content" / "hybrid": every game of the catalog (released by the end of history or already reviewed)
    described by its features, plus an own ID for games with >= id_min_count reviews in the hybrid."""

    def __init__(self, variant="id", dim=64, max_len=50, min_count=2, id_min_count=10, id_dropout=0.3,
                 use_review_stats=True, temperature=0.05, batch_size=1024, lr=1e-3, epochs=10):
        self.variant = variant
        self.name = "two_tower" if variant == "id" else f"two_tower_{variant}"
        if variant != "id" and not use_review_stats:
            self.name += "_no_review_stats"
        self.params = {"variant": variant, "dim": dim, "max_len": max_len,
                        "min_count": min_count, "id_min_count": id_min_count, "id_dropout": id_dropout,
                        "use_review_stats": use_review_stats, "temperature": temperature,
                        "batch_size": batch_size, "lr": lr, "epochs": epochs}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare(self, history):
        torch.manual_seed(42)
        features, id_of_item = None, None
        if self.variant == "id":
            self.item_index, self.item_ids = build_item_index(history, self.params["min_count"])
        else:
            catalog = build_catalog(history)
            self.item_index = {int(app_id): i + 2 for i, app_id in enumerate(catalog)}
            self.item_ids = torch.tensor([-1, -1] + catalog.tolist())
            features = build_game_features(catalog, history["date"].max(), self.params["use_review_stats"])
        if self.variant == "hybrid":
            id_of_item = torch.full((len(self.item_ids),), UNK)
            id_of_item[PAD] = PAD
            review_count = history["app_id"].value_counts()
            for app_id in review_count[review_count >= self.params["id_min_count"]].index:
                idx = self.item_index[int(app_id)]
                id_of_item[idx] = idx

        examples = build_training_examples(history, self.item_index, self.params["max_len"])
        self.examples = [x.to(self.device) for x in examples]

        targets = self.examples[3]
        counts = torch.bincount(targets, minlength=len(self.item_ids))
        self.log_q = torch.log((counts + 1) / len(targets))

        id_dropout = self.params["id_dropout"] if self.variant == "hybrid" else 0.0
        self.model = TwoTowerModel(len(self.item_ids), self.params["dim"], use_id=self.variant != "content",
                                   id_of_item=id_of_item, features=features, id_dropout=id_dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        self.histories = {}
        history = history.sort_values(["user_id", "date", "review_id"])
        for user_id, user_reviews in history.groupby("user_id"):
            self.histories[user_id] = (user_reviews["app_id"].to_numpy(),
                                       user_reviews["hours"].to_numpy(),
                                       user_reviews["is_recommended"].to_numpy())

    def train_epoch(self) -> float:
        self.model.train()
        batch_size = self.params["batch_size"]
        n = len(self.examples[3])
        order = torch.randperm(n, device=self.device)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            items, weights, liked, targets = [x[batch] for x in self.examples]
            user_vecs = self.model.user_tower(items, weights, liked)
            item_vecs = self.model.item_tower(targets)
            loss = in_batch_softmax_loss(user_vecs, item_vecs, targets, self.log_q, self.params["temperature"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def fit(self, history) -> "TwoTowerRecommender":
        self.prepare(history)
        for _ in range(self.params["epochs"]):
            self.train_epoch()
        return self

    def recommend(self, user_ids, seen, k) -> np.ndarray:
        return self.candidates(user_ids, seen, k)[0]

    def candidates(self, user_ids, seen, k) -> tuple[np.ndarray, np.ndarray]:
        """Top-k app_ids per user and their scores u·g; empty slots are -1 with score -inf."""
        recs = np.full((len(user_ids), k), -1, dtype=np.int64)
        rec_scores = np.full((len(user_ids), k), -np.inf, dtype=np.float32)
        known = np.array([user in self.histories for user in user_ids])
        if not known.any():
            return recs, rec_scores

        self.model.eval()
        with torch.no_grad():
            all_items = self.model.item_tower(torch.arange(len(self.item_ids), device=self.device))

            items, weights, liked = [], [], []
            for user in user_ids[known]:
                user_items, user_weights, user_liked = encode_history(
                    *self.histories[user], self.item_index, self.params["max_len"])
                items.append(user_items)
                weights.append(user_weights)
                liked.append(user_liked)
            user_vecs = self.model.user_tower(torch.stack(items).to(self.device),
                                              torch.stack(weights).to(self.device),
                                              torch.stack(liked).to(self.device))

            scores = user_vecs @ all_items.T
            scores[:, :2] = float("-inf")
            for row, user in enumerate(user_ids[known]):
                seen_items = [self.item_index[int(a)] for a in seen.get(user, []) if int(a) in self.item_index]
                scores[row, seen_items] = float("-inf")

            top = scores.topk(k, dim=1)
        top_ids = self.item_ids[top.indices.cpu()]
        top_ids[top.values.cpu() == float("-inf")] = -1
        recs[known] = top_ids.numpy()
        rec_scores[known] = top.values.cpu().numpy()
        return recs, rec_scores
