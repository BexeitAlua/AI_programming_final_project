"""Hand-checked metrics on a tiny synthetic dataset.

Run: python -m tests.test_evaluation   (or pytest, if installed)
"""
import numpy as np
import pandas as pd

from src.evaluation import build_eval_data, evaluate
from src.recommenders import ALSRecommender, PopularityRecommender


def _reviews(rows):
    return pd.DataFrame(rows, columns=["user_id", "app_id", "is_recommended", "date"]).assign(
        date=lambda d: pd.to_datetime(d["date"]), hours=10.0)


HISTORY = _reviews([
    (1, 10, True, "2021-01-01"), (1, 11, True, "2021-06-29"),
    (2, 10, True, "2021-01-01"), (2, 12, False, "2021-01-01"),
    (4, 12, True, "2021-06-30"), (4, 13, True, "2021-06-30"),   # user 4: no target reviews
])
TARGET = _reviews([
    (1, 12, True, "2021-08-01"), (1, 99, True, "2021-08-01"),   # 99 is a new (cold) game
    (2, 11, True, "2021-08-01"), (2, 13, False, "2021-08-01"),  # negative -> not relevant
    (3, 10, True, "2021-08-01"),                                # user 3: no history -> skipped
])


def test_build_eval_data():
    data = build_eval_data(HISTORY, TARGET)
    assert data.user_ids.tolist() == [1, 2]
    assert [sorted(r) for r in data.relevant] == [[12, 99], [11]]
    assert sorted(data.seen[1]) == [10, 11] and sorted(data.seen[2]) == [10, 12]
    assert data.catalog.tolist() == [10, 11, 12, 13]


def test_metrics():
    data = build_eval_data(HISTORY, TARGET)
    recs = np.array([[13, 12],    # user 1: hit at rank 2, cold game 99 missed
                     [11, 13]])   # user 2: hit at rank 1
    m = evaluate(recs, data, ks=(1, 2))
    ndcg_u1 = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert np.isclose(m["recall@1"], 0.5) and np.isclose(m["ndcg@1"], 0.5)
    assert np.isclose(m["recall@2"], (0.5 + 1) / 2)
    assert np.isclose(m["ndcg@2"], (ndcg_u1 + 1) / 2, atol=1e-5)
    assert m["hit_rate@2"] == 1.0
    assert m["recall_warm@2"] == 1.0 and m["recall_cold@2"] == 0.0
    assert m["coverage@2"] == 0.75                      # {11, 12, 13} of 4 catalog games
    assert m["cold_share"] == round(1 / 3, 5) and m["recall_ceiling"] == 0.75


def test_popularity():
    data = build_eval_data(HISTORY, TARGET)
    model = PopularityRecommender().fit(HISTORY)        # 10 has 2 likes; 11, 12, 13 tie -> by app_id
    assert model.ranking.tolist() == [10, 11, 12, 13]
    assert model.recommend(data.user_ids, data.seen, 2).tolist() == [[12, 13], [11, 13]]

    recent = PopularityRecommender(window_days=7).fit(HISTORY)   # only likes after 2021-06-23
    assert recent.ranking.tolist() == [11, 12, 13]
    assert recent.recommend(np.array([1]), data.seen, 3).tolist() == [[12, 13, -1]]


def test_als():
    model = ALSRecommender(factors=4, iterations=5).fit(HISTORY)
    recs = model.recommend(np.array([1, 2, 5]), {}, 2)
    assert sorted(recs[0]) == [12, 13]     # 10, 11 already liked
    assert sorted(recs[1]) == [11, 13]     # 12 was reviewed negatively -> also excluded
    assert recs[2].tolist() == [-1, -1]    # user 5 is unknown


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
