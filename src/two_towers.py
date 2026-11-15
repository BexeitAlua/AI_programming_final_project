import numpy as np
import pandas as pd
import torch

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