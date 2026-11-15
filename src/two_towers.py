import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
    def __init__(self, n_items, dim):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=PAD)
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.liked_embedding = nn.Embedding(2, dim)
        nn.init.zeros_(self.liked_embedding.weight)
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def item_tower(self, items):
        x = self.item_embedding(items)
        x = self.item_mlp(x)
        x = F.normalize(x, dim=-1)
        return x

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
    mask = same & ~torch.eye(B, dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))

    return F.cross_entropy(logits, torch.arange(B))


class TwoTowerRecommender(BaseRecommender):
    name = "two_tower"

    def __init__(self, dim=64, max_len=50, min_count=2, temperature=0.05,
                 batch_size=1024, lr=1e-3, epochs=10):
        self.params = {"dim": dim, "max_len": max_len,
                        "min_count": min_count, "temperature": temperature,
                        "batch_size": batch_size,
                        "lr": lr, "epochs": epochs}
        

    def prepare(self, history):
        torch.manual_seed(42)
        self.item_index, self.item_ids = build_item_index(history, self.params["min_count"])
        self.examples = build_training_examples(history, self.item_index, self.params["max_len"])

        targets = self.examples[3]
        counts = torch.bincount(targets, minlength=len(self.item_ids))
        self.log_q = torch.log((counts + 1) / len(targets))

        self.model = TwoTowerModel(len(self.item_ids), self.params["dim"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        self.histories = {}
        history = history.sort_values(["user_id", "date", "review_id"])
        for user_id, user_reviews in history.groupby("user_id"):
            self.histories[user_id] = (user_reviews["app_id"].to_numpy(),
                                       user_reviews["hours"].to_numpy(),
                                       user_reviews["is_recommended"].to_numpy())

    def train_epoch(self) -> float:
        self.model.train()
        loader = DataLoader(TensorDataset(*self.examples), batch_size=self.params["batch_size"], shuffle=True)
        total_loss = 0.0
        for items, weights, liked, targets in loader:
            user_vecs = self.model.user_tower(items, weights, liked)
            item_vecs = self.model.item_tower(targets)
            loss = in_batch_softmax_loss(user_vecs, item_vecs, targets, self.log_q, self.params["temperature"])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, history) -> "TwoTowerRecommender":
        self.prepare(history)
        for _ in range(self.params["epochs"]):
            self.train_epoch()
        return self

    def recommend(self, user_ids, seen, k) -> np.ndarray:
        recs = np.full((len(user_ids), k), -1, dtype=np.int64)
        known = np.array([user in self.histories for user in user_ids])
        if not known.any():
            return recs

        self.model.eval()
        with torch.no_grad():
            all_items = self.model.item_tower(torch.arange(len(self.item_ids)))

            items, weights, liked = [], [], []
            for user in user_ids[known]:
                user_items, user_weights, user_liked = encode_history(
                    *self.histories[user], self.item_index, self.params["max_len"])
                items.append(user_items)
                weights.append(user_weights)
                liked.append(user_liked)
            user_vecs = self.model.user_tower(torch.stack(items), torch.stack(weights), torch.stack(liked))

            scores = user_vecs @ all_items.T
            scores[:, :2] = float("-inf")
            for row, user in enumerate(user_ids[known]):
                seen_items = [self.item_index[int(a)] for a in seen.get(user, []) if int(a) in self.item_index]
                scores[row, seen_items] = float("-inf")

            top = scores.topk(k, dim=1)
        top_ids = self.item_ids[top.indices]
        top_ids[top.values == float("-inf")] = -1
        recs[known] = top_ids.numpy()
        return recs
