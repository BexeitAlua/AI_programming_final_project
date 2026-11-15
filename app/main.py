"""FastAPI service of the Steam Game Recommender (SERVICE_SPEC.md) with a web page.

    POST /recommendations   recommendations for a play history (Two-Tower candidates + LightGBM re-ranking)
    GET  /api/games?q=...   game search for the web page
    GET  /api/game/{id}     one game by app_id
    GET  /api/example       the control example from models/manifest.json
    GET  /health            status
    GET  /                  web page (app/static/index.html); interactive API docs at /docs

Settings come from the environment:
    MODELS_DIR         folder with items.parquet, user_side.npz, ranker.txt, manifest.json (default: models)
    QDRANT_URL         Qdrant server (default: http://localhost:6333);
                       ":memory:" = embedded Qdrant filled from models/items.parquet at startup (no server needed,
                       used for the public deployment)
    QDRANT_API_KEY     empty locally
    QDRANT_COLLECTION  default: games

Run locally: uv run --env-file .env uvicorn app.main:app --reload
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.load_qdrant import load_collection
from src.serving import Recommender

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "games")
EMBEDDED = QDRANT_URL == ":memory:"
STATIC_DIR = Path(__file__).parent / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once at startup (not per request)."""
    url = "http://localhost:6333" if EMBEDDED else QDRANT_URL
    recommender = Recommender(MODELS_DIR, url, QDRANT_API_KEY, QDRANT_COLLECTION)
    if EMBEDDED:
        recommender.client = QdrantClient(location=":memory:")
        start = time.perf_counter()
        n = load_collection(recommender.client, MODELS_DIR, QDRANT_COLLECTION)
        print(f"embedded Qdrant: {n} games loaded in {time.perf_counter() - start:.1f} s")
    catalog = pd.read_parquet(MODELS_DIR / "items.parquet", columns=["app_id", "title", "price", "log_reviews"])
    state["recommender"] = recommender
    state["catalog"] = catalog.sort_values("log_reviews", ascending=False).reset_index(drop=True)
    yield
    state.clear()


app = FastAPI(title="Steam Game Recommender", version="1.0.0", lifespan=lifespan,
              description="Two-Tower candidates from Qdrant re-ranked by LightGBM. See SERVICE_SPEC.md.")


class HistoryItem(BaseModel):
    app_id: int = Field(ge=1, description="Steam app_id")
    playtime: float = Field(ge=0, description="hours played")
    is_recommended: bool


class RecommendationRequest(BaseModel):
    history: list[HistoryItem] = Field(default_factory=list, description="oldest first, may be empty")
    top_k: int = Field(10, ge=1, le=100)


class RecommendedGame(BaseModel):
    app_id: int
    title: str
    score: float | None = Field(description="null = popularity fallback")
    price: float
    tags: list[str]


class RecommendationResponse(BaseModel):
    items: list[RecommendedGame]
    total: int
    cached: bool
    inference_time_ms: float


class GameHit(BaseModel):
    app_id: int
    title: str
    price: float


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    history = [item.model_dump() for item in request.history]
    start = time.perf_counter()
    try:
        items = state["recommender"].recommend(history, request.top_k)
    except Exception as error:                      # Qdrant unreachable, collection missing, ...
        raise HTTPException(status_code=503, detail=f"recommender unavailable: {error.__class__.__name__}")
    elapsed = (time.perf_counter() - start) * 1000
    return RecommendationResponse(items=items, total=len(items), cached=False,
                                  inference_time_ms=round(elapsed, 1))


@app.get("/api/games", response_model=list[GameHit])
def search_games(q: str = Query("", max_length=100), limit: int = Query(10, ge=1, le=50)) -> list[GameHit]:
    """Games whose title contains `q` (case-insensitive), most reviewed first; empty q = most reviewed."""
    catalog = state["catalog"]
    if q.strip():
        catalog = catalog[catalog["title"].str.contains(q.strip(), case=False, regex=False, na=False)]
    hits = catalog.head(limit)
    return [GameHit(app_id=int(r.app_id), title=r.title, price=round(float(r.price), 2))
            for r in hits.itertuples()]


@app.get("/api/game/{app_id}", response_model=GameHit)
def game(app_id: int) -> GameHit:
    row = state["catalog"][state["catalog"]["app_id"] == app_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"no game with app_id {app_id}")
    r = row.iloc[0]
    return GameHit(app_id=int(r["app_id"]), title=r["title"], price=round(float(r["price"]), 2))


@app.get("/api/example", response_model=RecommendationRequest)
def example() -> dict:
    return state["recommender"].manifest["control_example"]["request"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "games": len(state["catalog"]),
            "qdrant": "embedded" if EMBEDDED else QDRANT_URL, "collection": QDRANT_COLLECTION}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
