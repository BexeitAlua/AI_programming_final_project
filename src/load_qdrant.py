"""Load models/items.parquet into the Qdrant collection used by the FastAPI service.

Recreates the collection: one point per game, id = app_id, named vectors two_tower (64), als (64, only games
known to ALS) and text (384), all with Dot distance, plus the payload the service returns and ranks with.
Settings come from the environment (.env): QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, MODELS_DIR.

Run: uv run --env-file .env python -m src.load_qdrant
"""
import json
import os
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models

PAYLOAD_FLOATS = ["log_reviews", "log_recent_reviews", "positive_share", "log_price", "log_age_days"]


def load_collection(client: QdrantClient, models_dir: Path, collection: str = "games") -> int:
    """(Re)create `collection` from models_dir/items.parquet and return the number of points.

    Used by this script against a Qdrant server and by the FastAPI app in embedded mode (QDRANT_URL=:memory:).
    """
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    items = pd.read_parquet(models_dir / "items.parquet")

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection, vectors_config={
        name: models.VectorParams(size=size, distance=models.Distance.DOT)
        for name, size in manifest["qdrant"]["vectors"].items()})

    points = []
    for row in items.itertuples(index=False):
        vector = {"two_tower": row.two_tower.tolist(), "text": row.text.tolist()}
        if row.als is not None:
            vector["als"] = row.als.tolist()
        payload = {"app_id": int(row.app_id), "title": row.title, "price": round(float(row.price), 2),
                   "tags": [str(t) for t in row.tags], "date_release": row.date_release}
        payload |= {name: float(getattr(row, name)) for name in PAYLOAD_FLOATS}
        points.append(models.PointStruct(id=int(row.app_id), vector=vector, payload=payload))
    client.upload_points(collection, points, batch_size=512, wait=True)
    return client.count(collection).count


def main() -> None:
    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    collection = os.environ.get("QDRANT_COLLECTION", "games")
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    print(f"{collection}: {load_collection(client, models_dir, collection)} points at {url}")


if __name__ == "__main__":
    main()
