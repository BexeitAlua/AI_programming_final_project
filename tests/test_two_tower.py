"""Two-Tower building blocks on a tiny hand-made dataset.

Run: uv run pytest tests/test_two_tower.py
"""
import math

import numpy as np
import pandas as pd
import torch

from src.two_towers import (UNK, TwoTowerModel, TwoTowerRecommender, build_item_index,
                            build_training_examples, encode_history, in_batch_softmax_loss)


def _reviews(rows):
    return pd.DataFrame(rows, columns=["review_id", "user_id", "app_id", "is_recommended", "date"]).assign(
        date=lambda d: pd.to_datetime(d["date"]), hours=10.0)


HISTORY = _reviews([
    (1, 1, 10, True, "2021-01-01"), (2, 1, 20, False, "2021-01-02"), (3, 1, 30, True, "2021-01-03"),
    (5, 2, 10, True, "2021-01-01"), (4, 2, 30, True, "2021-01-01"),   # same day: review_id decides order
    (6, 3, 20, True, "2021-01-01"), (7, 3, 40, True, "2021-01-02"),   # game 40 has a single review
])


def test_build_item_index():
    item_index, item_ids = build_item_index(HISTORY, min_count=2)
    assert item_index == {10: 2, 20: 3, 30: 4}
    assert item_ids.tolist() == [-1, -1, 10, 20, 30]


def test_encode_history_keeps_last_reviews():
    items, weights, liked = encode_history([10, 99, 20], [0.0, 5.0, 10.0], [True, False, True],
                                           {10: 2, 20: 3}, max_len=2)
    assert items.tolist() == [UNK, 3]
    assert torch.allclose(weights, torch.tensor([1 + math.log1p(5), 1 + math.log1p(10)]))
    assert liked.tolist() == [0, 1]


def test_encode_history_pads_with_zeros():
    items, weights, liked = encode_history([10], [0.0], [True], {10: 2}, max_len=3)
    assert items.tolist() == [2, 0, 0]
    assert weights.tolist() == [1.0, 0.0, 0.0]
    assert liked.tolist() == [1, 0, 0]
    assert items.dtype == torch.long and weights.dtype == torch.float32 and liked.dtype == torch.long


def test_training_examples_use_only_the_past():
    item_index, _ = build_item_index(HISTORY, min_count=2)
    items, weights, liked, targets = build_training_examples(HISTORY, item_index, max_len=4)
    # user 1: [10+, 20-] -> 30;  user 2: [30+] -> 10;  user 3: target 40 is not in the index
    assert items.tolist() == [[2, 3, 0, 0], [4, 0, 0, 0]]
    assert liked.tolist() == [[1, 0, 0, 0], [1, 0, 0, 0]]
    assert targets.tolist() == [4, 2]
    assert not (items == targets[:, None]).any()


def test_model_shapes_and_norms():
    torch.manual_seed(0)
    model = TwoTowerModel(n_items=10, dim=8)
    items = torch.randint(0, 10, (4, 5))
    weights = torch.rand(4, 5)
    liked = torch.randint(0, 2, (4, 5))

    assert model.item_tower(items[:, 0]).shape == (4, 8)
    assert model.item_tower(items).shape == (4, 5, 8)
    user_vecs = model.user_tower(items, weights, liked)
    assert user_vecs.shape == (4, 8)
    assert torch.allclose(user_vecs.norm(dim=-1), torch.ones(4))
    assert not model.user_tower(items, torch.zeros(4, 5), liked).isnan().any()


def test_content_and_hybrid_models():
    torch.manual_seed(0)
    features = {"tags": torch.randint(0, 5, (10, 3)), "text": torch.rand(10, 6), "numeric": torch.rand(10, 4)}
    items = torch.randint(0, 10, (4, 5))
    for use_id in (False, True):
        model = TwoTowerModel(n_items=10, dim=8, use_id=use_id, features=features, id_dropout=0.3)
        assert model.item_tower(items).shape == (4, 5, 8)
        user_vecs = model.user_tower(items, torch.rand(4, 5), torch.randint(0, 2, (4, 5)))
        assert torch.allclose(user_vecs.norm(dim=-1), torch.ones(4))


def test_loss_masks_duplicate_targets():
    vecs = torch.full((4, 8), 8 ** -0.5)   # identical unit vectors -> all logits are equal
    log_q = torch.zeros(10)
    loss = in_batch_softmax_loss(vecs, vecs, torch.tensor([2, 3, 4, 5]), log_q, 0.05)
    assert math.isclose(loss.item(), math.log(4), rel_tol=1e-5)
    # rows 0 and 2 share a target: each loses one negative and chooses among 3 columns
    loss = in_batch_softmax_loss(vecs, vecs, torch.tensor([2, 3, 2, 5]), log_q, 0.05)
    assert math.isclose(loss.item(), (math.log(3) + math.log(4)) / 2, rel_tol=1e-5)


def test_recommend_skips_seen_games_and_unknown_users():
    model = TwoTowerRecommender(dim=8, batch_size=4, epochs=1, min_count=1).fit(HISTORY)
    seen = HISTORY.groupby("user_id")["app_id"].unique().to_dict()
    recs = model.recommend(np.array([1, 3, 99]), seen, k=3)
    assert recs[0].tolist() == [40, -1, -1]          # only game 40 is unseen for user 1
    assert sorted(recs[1, :2].tolist()) == [10, 30] and recs[1, 2] == -1
    assert recs[2].tolist() == [-1, -1, -1]          # no history -> evaluation uses the fallback
