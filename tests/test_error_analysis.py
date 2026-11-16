import numpy as np
import pandas as pd

from src.error_analysis import (game_groups, group_recall, miss_reasons, per_user_metrics, pick_examples,
                                popularity_bias, segment_of, segment_table)
from src.evaluation import build_eval_data
from tests.test_evaluation import HISTORY, TARGET

# user 1: relevant {12, 99 (cold)}; user 2: relevant {11}; catalog {10, 11, 12, 13}
RECS = np.array([[13, 12, 11, 10],     # user 1: 12 at rank 2, cold 99 impossible
                 [10, 13, 12, 11]])    # user 2: 11 at rank 4


def test_per_user_metrics_and_segments():
    data = build_eval_data(HISTORY, TARGET)
    users = per_user_metrics(RECS, data, k=2)
    assert users["user_id"].tolist() == [1, 2]
    assert users["history_len"].tolist() == [2, 2]
    assert users["recall"].tolist() == [0.5, 0.0] and users["hit"].tolist() == [1.0, 0.0]
    assert np.isclose(users["ndcg"].iloc[0], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert [segment_of(n) for n in (1, 2, 3, 9, 10, 24, 25, 400)] == ["1-2", "1-2", "3-9", "3-9", "10-24", "10-24", "25+", "25+"]
    table = segment_table(users)
    assert table["segment"].tolist() == ["1-2"] and table["n_users"].tolist() == [2]
    assert np.isclose(table["recall"].iloc[0], 0.25)


def test_game_groups_and_recall():
    groups = game_groups(HISTORY)       # likes: 10 -> 2, 11/12/13 -> 1 each: 10 is head (1% rounds up to 1 game)
    assert groups[10] == "head" and set(groups.drop(10)) <= {"mid", "tail"}
    data = build_eval_data(HISTORY, TARGET)
    table = group_recall(RECS, data, groups, ks=(2,)).set_index("group")
    assert table.loc["cold", "relevant"] == 1 and table.loc["cold", "hits"] == 0
    hits_warm = table.drop("cold")["hits"].sum()
    assert hits_warm == 1                 # only 12 for user 1 inside the top 2
    bias = popularity_bias(RECS, groups, k=2)
    assert np.isclose(sum(bias.values()), 1.0) and np.isclose(bias["head"], 0.25)   # 10 in 1 of 4 slots


def test_miss_reasons():
    data = build_eval_data(HISTORY, TARGET)
    deep = np.hstack([RECS, np.full((2, 96), -1)])    # 100 columns
    reasons = miss_reasons(deep, data)
    # relevant: 12 (rank 2 -> top10 hit), 99 (cold), 11 (rank 4 -> top10 hit)
    assert np.isclose(reasons["hit_top10"], 2 / 3) and np.isclose(reasons["cold_game"], 1 / 3)
    assert reasons["rank_11_100"] == 0 and reasons["beyond_100"] == 0


def test_pick_examples():
    users = pd.DataFrame({"user_id": [5, 6, 7, 8], "history_len": [1, 2, 5, 30], "n_relevant": [3, 1, 2, 4],
                          "recall": 0.0, "ndcg": 0.0, "hit": [0.0, 0.0, 0.0, 1.0]})
    assert pick_examples(users) == [5, 7]   # user 6 has one relevant game, user 8 had a hit
