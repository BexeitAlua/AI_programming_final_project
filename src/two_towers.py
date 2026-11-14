import numpy as np
import pandas as pd
import torch

def build_item_index(history, min_count) -> tuple[dict, np.ndarray]:
    item_ids: list = [-1, -1]
    item_index: dict = {}
    k = 2
    review_count = history["app_id"].value_counts()
    for app_id in (review_count.index):
        if review_count[app_id] >= min_count:
            item_ids.append(int(app_id))
            item_index[int(app_id)] = k
            k += 1
    return item_index, torch.tensor(item_ids)

def encode_history(app_ids, hours, liked, item_index, max_len) -> tuple[dict, np.ndarray]:
    return None