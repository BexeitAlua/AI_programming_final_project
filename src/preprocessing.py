from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    PROCESSED_DIR,
    RAW_DIR,
    RAW_FILES,
    RESULTS_DIR,
    df_memory_mb,
    ensure_dirs,
    process_rss_mb,
    save_json,
    timer,
)

REC_COLS = ["review_id", "user_id", "app_id", "date", "is_recommended", "hours"]
KEY_COLS = ["user_id", "app_id", "date"]

EXPECTED_COLUMNS = {
    "recommendations": ["app_id", "helpful", "funny", "date", "is_recommended",
                        "hours", "user_id", "review_id"],
    "games": ["app_id", "title", "date_release", "win", "mac", "linux", "rating",
              "positive_ratio", "user_reviews", "price_final", "price_original",
              "discount", "steam_deck"],
    "users": ["user_id", "products", "reviews"],
    "metadata": ["app_id", "description", "tags"],
}

TARGET_INT_DTYPES = {"review_id": "int32", "user_id": "int32", "app_id": "int32"}

STEAM_LAUNCH = pd.Timestamp("2003-09-12")  

CLEAN_DIR_NAME = "recommendations_clean"
BUCKET_DIR_NAME = "_buckets_tmp"

CLEANING_RULES = [
    {"key": "missing_keys", "problem": "Missing user_id / app_id / date",
     "rule": "Drop the row",
     "reason": "An interaction without user, game or time cannot be used in the matrix or the temporal split"},
    {"key": "bad_date", "problem": "Unparseable date or date outside [2003-09-12, today]",
     "rule": "Drop the row",
     "reason": "Wrong dates would break the temporal train/test split"},
    {"key": "bad_flag", "problem": "is_recommended is not true/false",
     "rule": "Drop the row",
     "reason": "The target (positive feedback) is undefined"},
    {"key": "bad_hours", "problem": "hours missing or negative",
     "rule": "Drop the row",
     "reason": "Playtime is used as confidence weight 1 + a*log1p(hours); it must be >= 0"},
    {"key": "unknown_app", "problem": "app_id not present in games.csv",
     "rule": "Drop the row (configurable)",
     "reason": "Game has no title/metadata and would violate the FK reviews.app_id -> games"},
    {"key": "unknown_user", "problem": "user_id not present in users.csv",
     "rule": "Drop the row (configurable)",
     "reason": "Would violate the FK reviews.user_id -> users"},
    {"key": "dup_review_id", "problem": "Duplicate review_id",
     "rule": "Keep the first occurrence",
     "reason": "review_id is the primary key of a review"},
    {"key": "dup_pair", "problem": "Several reviews for the same (user_id, app_id)",
     "rule": "Keep the latest by date",
     "reason": "Steam keeps one review per user and game; the latest one is the current opinion"},
    {"key": "high_hours", "problem": "Very large hours values (long tail)",
     "rule": "Keep, do not clip; use log1p(hours) later",
     "reason": "Large playtime is real behaviour, not an error"},
]

def check_raw_files(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Check that the four Kaggle files are in data/raw and show their size."""
    rows = []
    for name, filename in RAW_FILES.items():
        path = raw_dir / filename
        exists = path.exists()
        size_mb = round(path.stat().st_size / 1024 ** 2, 1) if exists else None
        rows.append({"dataset": name, "file": filename, "exists": exists, "size_mb": size_mb})
    table = pd.DataFrame(rows)
    missing = table.loc[~table["exists"], "file"].tolist()
    if missing:
        print(f"WARNING: missing files in {raw_dir}: {missing}. "
              "Download them from Kaggle (see README).")
    return table


def load_recommendations_sample(n_rows: int = 100_000, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / RAW_FILES["recommendations"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the Kaggle files in data/raw/.")
    return pd.read_csv(path, nrows=n_rows, parse_dates=["date"])


def load_games(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = raw_dir / RAW_FILES["games"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path, parse_dates=["date_release"])


def load_users(raw_dir: Path = RAW_DIR, n_rows: int | None = None) -> pd.DataFrame:
    path = raw_dir / RAW_FILES["users"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path, nrows=n_rows)


def load_metadata(raw_dir: Path = RAW_DIR, n_rows: int | None = None) -> pd.DataFrame:

    path = raw_dir / RAW_FILES["metadata"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    try:
        return pd.read_json(path, lines=True, nrows=n_rows)
    except ValueError:
        df = pd.read_json(path)
        return df.head(n_rows) if n_rows else df


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        example = non_null.iloc[0] if len(non_null) else None
        if isinstance(example, (list, np.ndarray)):
            example = list(example)[:5]
        try:
            n_unique = int(series.nunique())
        except TypeError:  
            n_unique = None
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_pct": round(100 * series.isna().mean(), 3),
            "n_unique": n_unique,
            "example": example,
        })
    return pd.DataFrame(rows)


def compare_with_expected_schema(df: pd.DataFrame, dataset: str) -> dict:
    expected = set(EXPECTED_COLUMNS[dataset])
    actual = set(df.columns)
    result = {
        "dataset": dataset,
        "missing_columns": sorted(expected - actual),
        "extra_columns": sorted(actual - expected),
        "ok": expected <= actual,
    }
    status = "OK" if result["ok"] else "DIFFERENT"
    print(f"{dataset}: schema {status}; missing={result['missing_columns']} "
          f"extra={result['extra_columns']}")
    return result


def quality_audit(reviews: pd.DataFrame,
                  games: pd.DataFrame | None = None,
                  users: pd.DataFrame | None = None) -> dict:
    """Rows, missing values, duplicates, ranges and referential integrity."""
    audit = {
        "rows": int(len(reviews)),
        "missing_by_column": {c: int(v) for c, v in reviews.isna().sum().items()},
        "duplicate_full_rows": int(reviews.duplicated().sum()),
        "duplicate_pairs_user_app": int(reviews.duplicated(["user_id", "app_id"]).sum()),
        "n_users": int(reviews["user_id"].nunique()),
        "n_games": int(reviews["app_id"].nunique()),
    }
    if "review_id" in reviews.columns:
        audit["duplicate_review_id"] = int(reviews["review_id"].duplicated().sum())
    if "hours" in reviews.columns:
        hours = reviews["hours"]
        audit.update({
            "min_hours": float(hours.min()),
            "max_hours": float(hours.max()),
            "median_hours": float(hours.median()),
            "negative_hours": int((hours < 0).sum()),
            "zero_hours": int((hours == 0).sum()),
        })
    if "date" in reviews.columns:
        dates = pd.to_datetime(reviews["date"], errors="coerce")
        audit.update({
            "min_date": str(dates.min().date()) if dates.notna().any() else None,
            "max_date": str(dates.max().date()) if dates.notna().any() else None,
            "unparseable_dates": int(dates.isna().sum() - reviews["date"].isna().sum()),
            "dates_before_steam_launch": int((dates < STEAM_LAUNCH).sum()),
            "dates_in_future": int((dates > pd.Timestamp.today()).sum()),
        })
    if "is_recommended" in reviews.columns:
        audit["is_recommended_values"] = {
            str(k): int(v) for k, v in reviews["is_recommended"].value_counts(dropna=False).items()
        }
        audit["positive_share"] = round(float(_to_bool(reviews["is_recommended"]).mean()), 4)
    if games is not None:
        unknown = ~reviews["app_id"].isin(games["app_id"])
        audit["reviews_with_unknown_app_id"] = int(unknown.sum())
        audit["unknown_app_ids"] = int(reviews.loc[unknown, "app_id"].nunique())
        audit["games_duplicate_app_id"] = int(games["app_id"].duplicated().sum())
    if users is not None:
        unknown = ~reviews["user_id"].isin(users["user_id"])
        audit["reviews_with_unknown_user_id"] = int(unknown.sum())
        audit["users_duplicate_user_id"] = int(users["user_id"].duplicated().sum())
    return audit


def cleaning_rules_table(counts: dict | None = None) -> pd.DataFrame:
    table = pd.DataFrame(CLEANING_RULES)
    if counts is not None:
        table["rows_affected"] = table["key"].map(counts).astype("Int64")
    return table



def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return series.astype(str).str.strip().str.lower().map(mapping).astype("boolean")


def _isin_sorted(values: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_ids, values)
    pos = np.clip(pos, 0, len(sorted_ids) - 1)
    return sorted_ids[pos] == values


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, dtype in TARGET_INT_DTYPES.items():
        if col not in df.columns:
            continue
        info = np.iinfo(dtype)
        col_min, col_max = df[col].min(), df[col].max()
        if col_min < info.min or col_max > info.max:
            raise ValueError(f"{col} range [{col_min}, {col_max}] does not fit into {dtype}")
        df[col] = df[col].astype(dtype)
    if "hours" in df.columns:
        df["hours"] = df["hours"].astype("float32")
    if "is_recommended" in df.columns:
        df["is_recommended"] = df["is_recommended"].astype(bool)
    return df


def clean_chunk(chunk: pd.DataFrame,
                stats: dict,
                valid_app_ids: np.ndarray | None = None,
                valid_user_ids: np.ndarray | None = None) -> pd.DataFrame:

    def drop(mask: pd.Series, key: str) -> None:
        nonlocal chunk
        n = int(mask.sum())
        stats[key] = stats.get(key, 0) + n
        if n:
            chunk = chunk.loc[~mask]

    drop(chunk[KEY_COLS].isna().any(axis=1), "missing_keys")

    dates = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="coerce")
    bad_date = dates.isna() | (dates < STEAM_LAUNCH) | (dates > pd.Timestamp.today())
    chunk = chunk.assign(date=dates)
    drop(bad_date, "bad_date")

    flag = _to_bool(chunk["is_recommended"])
    chunk = chunk.assign(is_recommended=flag)
    drop(flag.isna(), "bad_flag")

    drop(chunk["hours"].isna() | (chunk["hours"] < 0), "bad_hours")

    if valid_app_ids is not None:
        drop(pd.Series(~_isin_sorted(chunk["app_id"].to_numpy(), valid_app_ids),
                       index=chunk.index), "unknown_app")
    if valid_user_ids is not None:
        drop(pd.Series(~_isin_sorted(chunk["user_id"].to_numpy(), valid_user_ids),
                       index=chunk.index), "unknown_user")
    return chunk


def _update_raw_stats(raw: dict, chunk: pd.DataFrame) -> None:
    raw["rows"] = raw.get("rows", 0) + len(chunk)
    missing = raw.setdefault("missing_by_column", {})
    for col, n in chunk.isna().sum().items():
        missing[col] = missing.get(col, 0) + int(n)
    hours = chunk["hours"]
    if hours.notna().any():
        raw["min_hours"] = min(raw.get("min_hours", np.inf), float(hours.min()))
        raw["max_hours"] = max(raw.get("max_hours", -np.inf), float(hours.max()))
    dates = chunk["date"].dropna().astype(str)
    dates = dates[dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}")]
    if len(dates): 
        raw["min_date_raw"] = min(raw.get("min_date_raw", "9999"), dates.min())
        raw["max_date_raw"] = max(raw.get("max_date_raw", "0000"), dates.max())
    flags = raw.setdefault("is_recommended_values", {})
    for k, v in chunk["is_recommended"].astype(str).value_counts(dropna=False).items():
        flags[k] = flags.get(k, 0) + int(v)


def _load_valid_ids(raw_dir: Path, drop_unknown_apps: bool, drop_unknown_users: bool):
    """Sorted id arrays from games.csv / users.csv for the integrity checks."""
    app_ids = user_ids = None
    if drop_unknown_apps:
        app_ids = np.sort(pd.read_csv(raw_dir / RAW_FILES["games"],
                                      usecols=["app_id"])["app_id"].to_numpy())
    if drop_unknown_users:
        user_ids = np.sort(pd.read_csv(raw_dir / RAW_FILES["users"],
                                       usecols=["user_id"])["user_id"].to_numpy())
    return app_ids, user_ids


def process_in_chunks(chunksize: int = 500_000,
                      n_buckets: int = 16,
                      max_chunks: int | None = None,
                      drop_unknown_apps: bool = True,
                      drop_unknown_users: bool = True,
                      raw_dir: Path = RAW_DIR,
                      out_dir: Path = PROCESSED_DIR):

    path = raw_dir / RAW_FILES["recommendations"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the Kaggle files in data/raw/.")

    bucket_root = out_dir / BUCKET_DIR_NAME
    if bucket_root.exists():
        shutil.rmtree(bucket_root)
    bucket_root.mkdir(parents=True)

    with timer("load valid ids (games/users)"):
        valid_apps, valid_users = _load_valid_ids(raw_dir, drop_unknown_apps, drop_unknown_users)

    raw_stats: dict = {}
    counts: dict = {}
    log_rows = []
    reader = pd.read_csv(path, usecols=REC_COLS, chunksize=chunksize)
    for idx, chunk in enumerate(reader):
        if max_chunks is not None and idx >= max_chunks:
            break
        start = time.perf_counter()
        mem_before = df_memory_mb(chunk)          # as read by pandas defaults
        _update_raw_stats(raw_stats, chunk)

        cleaned = clean_chunk(chunk, counts, valid_apps, valid_users)
        cleaned = optimize_dtypes(cleaned[REC_COLS])
        mem_after = df_memory_mb(cleaned)

        for b, part in cleaned.groupby(cleaned["user_id"] % n_buckets):
            bdir = bucket_root / f"bucket_{int(b):03d}"
            bdir.mkdir(exist_ok=True)
            part.to_parquet(bdir / f"part-{idx:05d}.parquet", index=False)

        seconds = time.perf_counter() - start
        log_rows.append({
            "chunk": idx, "rows_in": len(chunk), "rows_out": len(cleaned),
            "mem_before_mb": round(mem_before, 2), "mem_after_mb": round(mem_after, 2),
            "seconds": round(seconds, 3), "process_rss_mb": process_rss_mb(),
        })
        print(f"chunk {idx:4d}: {len(chunk):>8,} -> {len(cleaned):>8,} rows | "
              f"{mem_before:7.1f} MB -> {mem_after:6.1f} MB | {seconds:5.2f} s")

    return pd.DataFrame(log_rows), raw_stats, counts


def deduplicate_buckets(out_dir: Path = PROCESSED_DIR, counts: dict | None = None) -> dict:
    counts = counts if counts is not None else {}
    bucket_root = out_dir / BUCKET_DIR_NAME
    clean_dir = out_dir / CLEAN_DIR_NAME
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True)

    counts.setdefault("dup_review_id", 0)
    counts.setdefault("dup_pair", 0)
    total_rows = 0
    for bdir in sorted(bucket_root.glob("bucket_*")):
        df = pd.read_parquet(bdir)
        n_dup_id = int(df["review_id"].duplicated().sum())
        df = df.drop_duplicates("review_id", keep="first")
        df = df.sort_values(["user_id", "app_id", "date"], kind="stable")
        n_dup_pair = int(df.duplicated(["user_id", "app_id"], keep="last").sum())
        df = df.drop_duplicates(["user_id", "app_id"], keep="last")
        df.sort_values("date", kind="stable").to_parquet(
            clean_dir / f"{bdir.name.replace('bucket', 'part')}.parquet", index=False)
        counts["dup_review_id"] += n_dup_id
        counts["dup_pair"] += n_dup_pair
        total_rows += len(df)
    shutil.rmtree(bucket_root)
    counts["final_rows"] = total_rows
    return counts


def load_clean_reviews(columns: list | None = None, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    clean_dir = out_dir / CLEAN_DIR_NAME
    if not clean_dir.exists():
        raise FileNotFoundError(f"{clean_dir} not found. Run: python -m src.preprocessing")
    return pd.read_parquet(clean_dir, columns=columns)


def convert_games(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    games = load_games(raw_dir).drop_duplicates("app_id")
    games["app_id"] = games["app_id"].astype("int32")
    for col in ("price_final", "price_original", "discount"):
        if col in games.columns:
            games[col] = games[col].astype("float32")
    for col in ("positive_ratio", "user_reviews"):
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], downcast="integer")
    if "rating" in games.columns:
        games["rating"] = games["rating"].astype("category")
    games.to_parquet(out_dir / "games.parquet", index=False)
    return games


def convert_users(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    users = load_users(raw_dir).drop_duplicates("user_id")
    for col in users.columns:
        users[col] = pd.to_numeric(users[col], downcast="integer")
    users.to_parquet(out_dir / "users.parquet", index=False)
    return users


def convert_metadata(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    meta = load_metadata(raw_dir).drop_duplicates("app_id")
    meta["app_id"] = meta["app_id"].astype("int32")
    meta.to_parquet(out_dir / "games_metadata.parquet", index=False)
    return meta


def build_experiment_sample(n_users: int = 50_000,
                            min_interactions: int = 5,
                            seed: int = 42,
                            out_dir: Path = PROCESSED_DIR) -> dict:

    user_counts = load_clean_reviews(["user_id"], out_dir)["user_id"].value_counts()
    # sort: value_counts() tie order differs between pandas versions -> different sample for the same seed
    eligible = np.sort(user_counts[user_counts >= min_interactions].index.to_numpy())
    if len(eligible) == 0:
        raise ValueError(f"No users with >= {min_interactions} reviews")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False))

    parts = []
    for part_path in sorted((out_dir / CLEAN_DIR_NAME).glob("*.parquet")):
        part = pd.read_parquet(part_path)
        parts.append(part[_isin_sorted(part["user_id"].to_numpy(), chosen)])
    sample = pd.concat(parts, ignore_index=True).sort_values("date", kind="stable")
    sample.to_parquet(out_dir / "sample_reviews.parquet", index=False)

    info = {
        "seed": seed,
        "min_interactions": min_interactions,
        "eligible_users": int(len(eligible)),
        "sampled_users": int(len(chosen)),
        "sample_rows": int(len(sample)),
        "sample_games": int(sample["app_id"].nunique()),
        "positive_share": round(float(sample["is_recommended"].mean()), 4),
        "min_date": str(sample["date"].min().date()),
        "max_date": str(sample["date"].max().date()),
    }
    save_json(info, RESULTS_DIR / "sample_info.json")
    return info


def run_preprocessing(chunksize: int = 500_000,
                      n_buckets: int = 16,
                      max_chunks: int | None = None,
                      sample_users: int = 50_000,
                      min_interactions: int = 5,
                      seed: int = 42) -> dict:
    ensure_dirs()
    timings: dict = {}

    with timer("convert games/users/metadata", timings):
        games = convert_games()
        users = convert_users()
        if (RAW_DIR / RAW_FILES["metadata"]).exists():
            convert_metadata()

    with timer("pass 1: chunked cleaning", timings):
        memory_log, raw_stats, counts = process_in_chunks(
            chunksize=chunksize, n_buckets=n_buckets, max_chunks=max_chunks)
    with timer("pass 2: global deduplication", timings):
        counts = deduplicate_buckets(counts=counts)
    with timer("experiment sample", timings):
        sample_info = build_experiment_sample(sample_users, min_interactions, seed)

    memory_log.to_csv(RESULTS_DIR / "memory_log.csv", index=False)
    rules = cleaning_rules_table(counts)
    rules.drop(columns="key").to_csv(RESULTS_DIR / "cleaning_rules.csv", index=False)

    mem_before = float(memory_log["mem_before_mb"].sum())
    mem_after = float(memory_log["mem_after_mb"].sum())
    clean_files = list((PROCESSED_DIR / CLEAN_DIR_NAME).glob("*.parquet"))
    audit = {
        "raw_file": raw_stats,
        "games": {"rows": int(len(games)), "unique_app_id": int(games["app_id"].nunique())},
        "users": {"rows": int(len(users)), "unique_user_id": int(users["user_id"].nunique())},
        "removed_rows_by_rule": {k: v for k, v in counts.items() if k != "final_rows"},
        "final_rows": counts["final_rows"],
        "kept_share": round(counts["final_rows"] / max(raw_stats["rows"], 1), 6),
        "memory": {
            "chunksize": chunksize,
            "chunks": int(len(memory_log)),
            "pandas_default_mb_total": round(mem_before, 1),
            "optimized_mb_total": round(mem_after, 1),
            "reduction_pct": round(100 * (1 - mem_after / mem_before), 1) if mem_before else None,
            "parquet_on_disk_mb": round(sum(f.stat().st_size for f in clean_files) / 1024 ** 2, 1),
            "csv_on_disk_mb": round((RAW_DIR / RAW_FILES["recommendations"]).stat().st_size
                                    / 1024 ** 2, 1),
            "peak_process_rss_mb": memory_log["process_rss_mb"].max(),
        },
        "timings_sec": timings,
        "sample": sample_info,
        "parameters": {"n_buckets": n_buckets, "max_chunks": max_chunks},
    }
    save_json(audit, RESULTS_DIR / "data_audit.json")
    print(f"\nDone: {raw_stats['rows']:,} raw rows -> {counts['final_rows']:,} clean rows")
    print(f"Memory (sum over chunks): {mem_before:.1f} MB -> {mem_after:.1f} MB")
    print(f"Saved: {RESULTS_DIR / 'data_audit.json'}")
    return audit


def temporal_split(df, train_ratio=0.7, val_ratio=0.1):
    df = df.sort_values("date").copy()

    train_cutoff = df["date"].quantile(train_ratio)
    test_cutoff = df["date"].quantile(
        train_ratio + val_ratio
    )

    train = df[df["date"] < train_cutoff].copy()

    val = df[
        (df["date"] >= train_cutoff)
        & (df["date"] < test_cutoff)
    ].copy()

    test = df[
        df["date"] >= test_cutoff
    ].copy()

    return train, val, test


from scipy.sparse import csr_matrix


def build_user_item_matrix(train):
    # последний отзыв каждой пары
    pos = (
        train.sort_values("date")
        .drop_duplicates(
            ["user_id", "app_id"], keep="last"
        )
    )

    pos = pos[
        pos["is_recommended"].eq(True)
    ].copy()

    users = pos["user_id"].unique()
    items = pos["app_id"].unique()

    u_map = {u: i for i, u in enumerate(users)}
    i_map = {a: j for j, a in enumerate(items)}

    rows = pos["user_id"].map(u_map).to_numpy()
    cols = pos["app_id"].map(i_map).to_numpy()

    weights = (
        1 + np.log1p(pos["hours"].clip(lower=0))
    ).astype("float32")

    mat = csr_matrix(
        (weights, (rows, cols)),
        shape=(len(users), len(items)),
        dtype=np.float32
    )

    return mat, u_map, i_map

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean recommendations.csv into Parquet")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--buckets", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="process only the first N chunks (quick test)")
    parser.add_argument("--sample-users", type=int, default=50_000)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_preprocessing(chunksize=args.chunksize, n_buckets=args.buckets,
                      max_chunks=args.max_chunks, sample_users=args.sample_users,
                      min_interactions=args.min_interactions, seed=args.seed)