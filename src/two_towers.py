import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

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
