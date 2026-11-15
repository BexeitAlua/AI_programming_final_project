"""FastAPI service end to end with embedded Qdrant (no server needed). Skipped when models/ is not available.

Run: uv run pytest tests/test_api.py
"""
import json
import os
from pathlib import Path

import pytest

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
pytestmark = pytest.mark.skipif(not (MODELS_DIR / "manifest.json").exists(), reason="needs models/")

from fastapi.testclient import TestClient  # noqa: E402

import app.main as service  # noqa: E402
from app.main import app  # noqa: E402

service.EMBEDDED = True   # embedded Qdrant for the test, without changing QDRANT_URL for other tests


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:      # runs the lifespan: models + Qdrant collection
        yield test_client


def test_control_example(client):
    example = client.get("/api/example").json()
    response = client.post("/recommendations", json=example)
    assert response.status_code == 200
    body = response.json()
    manifest = json.loads((MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    expected = [e["app_id"] for e in manifest["control_example"]["expected"]]
    assert [item["app_id"] for item in body["items"]] == expected
    assert body["total"] == len(expected) and body["cached"] is False
    assert body["inference_time_ms"] >= 0


def test_validation(client):
    bad = [{"history": [{"app_id": 0, "playtime": 1, "is_recommended": True}]},
           {"history": [{"app_id": 730, "playtime": -1, "is_recommended": True}]},
           {"history": [], "top_k": 0},
           {"history": [], "top_k": 101}]
    for payload in bad:
        assert client.post("/recommendations", json=payload).status_code == 422


def test_empty_history_falls_back_to_popularity(client):
    body = client.post("/recommendations", json={"history": [], "top_k": 3}).json()
    assert body["total"] == 3 and all(item["score"] is None for item in body["items"])


def test_search_and_pages(client):
    games = client.get("/api/games", params={"q": "a", "limit": 5}).json()
    assert 0 < len(games) <= 5 and all("a" in g["title"].lower() for g in games)
    first = games[0]
    assert client.get(f"/api/game/{first['app_id']}").json()["title"] == first["title"]
    assert client.get("/api/game/999999999").status_code == 404
    assert client.get("/health").json()["status"] == "ok"
    assert "Steam Game Recommender" in client.get("/").text
