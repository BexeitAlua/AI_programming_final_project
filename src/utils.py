from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

try:
    import psutil
except ImportError:  
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_FILES = {
    "recommendations": "recommendations.csv",
    "games": "games.csv",
    "users": "users.csv",
    "metadata": "games_metadata.json",
}


def ensure_dirs() -> None:
    for folder in (PROCESSED_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def df_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1024 ** 2


def process_rss_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / 1024 ** 2


@contextmanager
def timer(label: str, log: dict | None = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if log is not None:
            log[label] = round(elapsed, 3)
        print(f"[time] {label}: {elapsed:.2f} s")


def save_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path