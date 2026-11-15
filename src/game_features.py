"""Content features of games for the Two-Tower item tower.

Everything is computed as of `history_end`, the last day of the fit history:
review counts and positive shares come from the full review log up to that day, because
`user_reviews` / `positive_ratio` in games.csv are a 2023 snapshot that already contains
the val and test periods.
"""
import numpy as np
import pandas as pd
import torch

from src.preprocessing import load_clean_reviews
from src.utils import PROCESSED_DIR

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_CACHE = PROCESSED_DIR / "text_embeddings.npz"
MAX_TAGS = 20
PRIOR_STRENGTH = 10


def build_catalog(history: pd.DataFrame) -> np.ndarray:
    """Sorted app_ids that can be recommended right after the history ends:
    released by then, or already reviewed (early access games get reviews before their release date)."""
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet", columns=["app_id", "date_release"])
    released = games.loc[games["date_release"] <= history["date"].max(), "app_id"]
    return np.union1d(released.to_numpy(), history["app_id"].unique()).astype(np.int64)


def text_embeddings() -> pd.DataFrame:
    """Frozen sentence embedding of "title. description" for every game, cached on disk."""
    if not TEXT_CACHE.exists():
        from sentence_transformers import SentenceTransformer

        games = pd.read_parquet(PROCESSED_DIR / "games.parquet", columns=["app_id", "title"])
        meta = pd.read_parquet(PROCESSED_DIR / "games_metadata.parquet", columns=["app_id", "description"])
        games = games.merge(meta, on="app_id", how="left").sort_values("app_id")
        texts = (games["title"].fillna("") + ". " + games["description"].fillna("")).tolist()
        encoder = SentenceTransformer(TEXT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
        vectors = encoder.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        np.savez(TEXT_CACHE, app_ids=games["app_id"].to_numpy(), vectors=vectors.astype(np.float32))
    cache = np.load(TEXT_CACHE)
    return pd.DataFrame(cache["vectors"], index=cache["app_ids"])


def build_game_features(app_ids: np.ndarray, history_end: pd.Timestamp) -> dict:
    """Feature tables aligned with item indices: row 2 + i describes app_ids[i], rows PAD and UNK are zeros.

    tags    [n_items, MAX_TAGS] long  - tag numbers, 0 = no tag
    text    [n_items, 384]      float - sentence embedding
    numeric [n_items, 9]        float - standardized: log price, free, age, platforms, reviews so far
    """
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet").set_index("app_id").reindex(app_ids)
    meta = pd.read_parquet(PROCESSED_DIR / "games_metadata.parquet").set_index("app_id").reindex(app_ids)

    all_tags = sorted({tag for tags in meta["tags"].dropna() for tag in tags})
    tag_index = {tag: i + 1 for i, tag in enumerate(all_tags)}
    tags = np.zeros((len(app_ids) + 2, MAX_TAGS), dtype=np.int64)
    for row, game_tags in enumerate(meta["tags"]):
        if isinstance(game_tags, np.ndarray):
            for col, tag in enumerate(game_tags[:MAX_TAGS]):
                tags[row + 2, col] = tag_index[tag]

    reviews = load_clean_reviews(["app_id", "date", "is_recommended"])
    reviews = reviews[reviews["date"] <= history_end]
    stats = reviews.groupby("app_id")["is_recommended"].agg(["size", "sum"]).reindex(app_ids, fill_value=0)
    prior = reviews["is_recommended"].mean()

    numeric = pd.DataFrame({
        "log_price": np.log1p(games["price_original"]),
        "free": games["price_original"] == 0,
        "age_years": ((history_end - games["date_release"]).dt.days / 365).clip(lower=0),
        "win": games["win"],
        "mac": games["mac"],
        "linux": games["linux"],
        "steam_deck": games["steam_deck"],
        "log_reviews": np.log1p(stats["size"]),
        "positive_share": (stats["sum"] + prior * PRIOR_STRENGTH) / (stats["size"] + PRIOR_STRENGTH),
    }).astype(np.float32).fillna(0)
    numeric = (numeric - numeric.mean()) / numeric.std().replace(0, 1)

    text = text_embeddings().reindex(app_ids, fill_value=0).to_numpy(np.float32)
    numeric = numeric.to_numpy(np.float32)

    return {
        "tags": torch.from_numpy(tags),
        "text": torch.from_numpy(np.vstack([np.zeros((2, text.shape[1]), np.float32), text])),
        "numeric": torch.from_numpy(np.vstack([np.zeros((2, numeric.shape[1]), np.float32), numeric])),
    }
