"""Recommender against the real Qdrant. Skipped when models/ or the loaded collection are not available.

Prepare: docker compose up -d qdrant && uv run --env-file .env python -m src.load_qdrant
"""
import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.serving import Recommender

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _collection_loaded() -> bool:
    try:
        return QdrantClient(url=QDRANT_URL, timeout=2).collection_exists("games")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not (MODELS_DIR / "manifest.json").exists() or not _collection_loaded(),
                                reason="needs models/ and the 'games' collection in Qdrant")


@pytest.fixture(scope="module")
def recommender():
    return Recommender(MODELS_DIR, QDRANT_URL)


def test_control_example(recommender):
    example = recommender.manifest["control_example"]
    items = recommender.recommend(example["request"]["history"], example["request"]["top_k"])
    assert [item["app_id"] for item in items] == [e["app_id"] for e in example["expected"]]
    assert all(item["score"] is not None for item in items)


def test_history_games_are_not_recommended(recommender):
    history = recommender.manifest["control_example"]["request"]["history"]
    items = recommender.recommend(history, 100)
    assert len(items) == 100
    assert not {item["app_id"] for item in items} & {game["app_id"] for game in history}


def test_unknown_history_falls_back_to_popularity(recommender):
    items = recommender.recommend([{"app_id": 999999999, "playtime": 1.0, "is_recommended": True}], 5)
    assert len(items) == 5 and all(item["score"] is None for item in items)
    assert recommender.recommend([], 3)[0]["app_id"] == recommender.manifest["fallback_app_ids"][0]
