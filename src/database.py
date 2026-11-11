from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

from src.utils import PROCESSED_DIR, PROJECT_ROOT, RESULTS_DIR, save_json

DATABASE_DIR = PROJECT_ROOT / "database"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
QUERIES_PATH = DATABASE_DIR / "queries.sql"
DB_PATH = PROJECT_ROOT / "data" / "steam.db"
SQL_RESULTS_DIR = RESULTS_DIR / "sql"
INDEX_MARKER = "-- @indexes"

GAME_COLUMNS = ["app_id", "title", "date_release", "win", "mac", "linux", "rating",
                "positive_ratio", "user_reviews", "price_final", "price_original",
                "discount", "steam_deck", "description"]
REVIEW_COLUMNS = ["review_id", "user_id", "app_id", "review_date", "is_recommended", "hours"]


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"{db_path} not found. Run: python -m src.database")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _read_schema() -> tuple:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    tables_sql, _, index_sql = text.partition(INDEX_MARKER)
    return tables_sql, index_sql


def _insert_rows(conn: sqlite3.Connection, table: str, df: pd.DataFrame, batch: int = 500_000) -> int:
    cols = list(df.columns)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    for start in range(0, len(df), batch):
        part = df.iloc[start:start + batch]
        columns = []
        for c in cols:
            values = part[c]
            if values.isna().any():
                values = values.astype(object).where(values.notna(), None)
            columns.append(values.tolist())
        rows = list(zip(*columns))
        conn.executemany(sql, rows)
    return len(df)


def _prepare_games() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    meta_path = PROCESSED_DIR / "games_metadata.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path, columns=["app_id", "description"])
        games = games.merge(meta, on="app_id", how="left")
    else:
        games["description"] = None
    games["date_release"] = pd.to_datetime(games["date_release"]).dt.strftime("%Y-%m-%d")
    for col in ("win", "mac", "linux", "steam_deck"):
        games[col] = games[col].astype(int)
    games["rating"] = games["rating"].astype(str)
    for col in ("price_final", "price_original", "discount"):
        games[col] = games[col].astype(float).round(2)
    return games[GAME_COLUMNS]


def _prepare_tags(valid_app_ids: set) -> tuple:
    meta_path = PROCESSED_DIR / "games_metadata.parquet"
    if not meta_path.exists():
        return pd.DataFrame(columns=["tag_id", "name"]), pd.DataFrame(columns=["app_id", "tag_id"])
    meta = pd.read_parquet(meta_path, columns=["app_id", "tags"])
    pairs = meta.explode("tags").dropna(subset=["tags"])
    pairs = pairs[pairs["app_id"].isin(valid_app_ids)]
    names = sorted(pairs["tags"].unique())
    tags = pd.DataFrame({"tag_id": range(1, len(names) + 1), "name": names})
    tag_map = dict(zip(tags["name"], tags["tag_id"]))
    game_tags = pd.DataFrame({"app_id": pairs["app_id"].astype(int),
                              "tag_id": pairs["tags"].map(tag_map)}).drop_duplicates()
    return tags, game_tags


def _review_parts(source: str) -> list:
    if source == "sample":
        return [PROCESSED_DIR / "sample_reviews.parquet"]
    return sorted((PROCESSED_DIR / "recommendations_clean").glob("*.parquet"))


def create_database(db_path: Path = DB_PATH, source: str = "full", check_fk: bool = True) -> dict:
    """Create steam.db from the Parquet files of step 5.
    source='full' loads all clean reviews, source='sample' only sample_reviews.parquet."""
    if source not in ("full", "sample"):
        raise ValueError("source must be 'full' or 'sample'")
    parts = _review_parts(source)
    if not parts or not parts[0].exists():
        raise FileNotFoundError("Clean Parquet not found. Run: python -m src.preprocessing")

    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tables_sql, index_sql = _read_schema()
    timings = {}
    info = {"source": source, "db_path": str(db_path)}

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.executescript(tables_sql)
        conn.execute("PRAGMA foreign_keys = OFF")

        start = time.perf_counter()
        games = _prepare_games()
        info["games"] = _insert_rows(conn, "games", games)
        tags, game_tags = _prepare_tags(set(games["app_id"]))
        info["tags"] = _insert_rows(conn, "tags", tags)
        info["game_tags"] = _insert_rows(conn, "game_tags", game_tags)

        users = pd.read_parquet(PROCESSED_DIR / "users.parquet")
        if source == "sample":
            sample_users = pd.read_parquet(parts[0], columns=["user_id"])["user_id"].unique()
            users = users[users["user_id"].isin(sample_users)]
        info["users"] = _insert_rows(conn, "users", users[["user_id", "products", "reviews"]])
        conn.commit()
        timings["games_users_tags_sec"] = round(time.perf_counter() - start, 2)

        start = time.perf_counter()
        n_reviews = 0
        for i, part_path in enumerate(parts):
            df = pd.read_parquet(part_path)
            df = df.rename(columns={"date": "review_date"})
            df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")
            df["is_recommended"] = df["is_recommended"].astype(int)
            df["hours"] = df["hours"].astype(float).round(1)
            n_reviews += _insert_rows(conn, "reviews", df[REVIEW_COLUMNS])
            conn.commit()
            print(f"reviews part {i + 1}/{len(parts)}: {n_reviews:,} rows")
        info["reviews"] = n_reviews
        timings["reviews_sec"] = round(time.perf_counter() - start, 2)

        start = time.perf_counter()
        conn.executescript(index_sql)
        timings["indexes_sec"] = round(time.perf_counter() - start, 2)

        conn.execute("PRAGMA foreign_keys = ON")
        if check_fk:
            start = time.perf_counter()
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            info["foreign_key_violations"] = len(violations)
            timings["fk_check_sec"] = round(time.perf_counter() - start, 2)
        conn.commit()
    finally:
        conn.close()

    info["db_size_mb"] = round(db_path.stat().st_size / 1024 ** 2, 1)
    info["timings"] = timings
    save_json(info, RESULTS_DIR / "database_info.json")
    print(info)
    return info


def load_queries(path: Path = QUERIES_PATH) -> dict:
    """Parse queries.sql: every query starts with a line '-- name: <query_name>'."""
    queries = {}
    name = None
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("-- name:"):
            if name:
                queries[name] = "\n".join(lines).strip().rstrip(";")
            name = line.split(":", 1)[1].strip()
            lines = []
        elif name:
            lines.append(line)
    if name:
        queries[name] = "\n".join(lines).strip().rstrip(";")
    return queries


def run_query(name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    queries = load_queries()
    if name not in queries:
        raise KeyError(f"Unknown query {name}. Available: {list(queries)}")
    with get_connection(db_path) as conn:
        return pd.read_sql_query(queries[name], conn)


def run_all_queries(db_path: Path = DB_PATH, save: bool = True) -> dict:
    results = {}
    timings = {}
    SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        for name, sql in load_queries().items():
            start = time.perf_counter()
            df = pd.read_sql_query(sql, conn)
            timings[name] = round(time.perf_counter() - start, 2)
            results[name] = df
            if save:
                df.to_csv(SQL_RESULTS_DIR / f"{name}.csv", index=False)
            print(f"{name}: {len(df)} rows, {timings[name]} s")
    save_json(timings, SQL_RESULTS_DIR / "query_timings.json")
    return results


def draw_er_diagram(path: Path = DATABASE_DIR / "er_diagram.png"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    tables = {
        "users": (0.3, 7.2, ["PK user_id", "products", "reviews"]),
        "reviews": (5.0, 7.2, ["PK review_id", "FK user_id", "FK app_id", "review_date",
                               "is_recommended", "hours"]),
        "games": (9.7, 7.2, ["PK app_id", "title", "date_release", "rating", "positive_ratio",
                             "user_reviews", "price_final", "price_original", "discount",
                             "win / mac / linux", "steam_deck", "description"]),
        "game_tags": (9.7, 1.9, ["PK,FK app_id", "PK,FK tag_id"]),
        "tags": (5.0, 1.9, ["PK tag_id", "name"]),
    }
    width, row_h, head_h = 3.6, 0.38, 0.5
    fig, ax = plt.subplots(figsize=(14, 9))
    boxes = {}
    for name, (x, y_top, cols) in tables.items():
        height = head_h + row_h * len(cols)
        y = y_top - height
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02",
                                    fc="#f4f6fb", ec="#1b2a49", lw=1.5))
        ax.add_patch(FancyBboxPatch((x, y_top - head_h), width, head_h, boxstyle="round,pad=0.02",
                                    fc="#1b2a49", ec="#1b2a49", lw=1.5))
        ax.text(x + width / 2, y_top - head_h / 2, name, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        for i, col in enumerate(cols):
            bold = col.startswith("PK") or col.startswith("FK")
            ax.text(x + 0.15, y_top - head_h - row_h * (i + 0.5), col, va="center",
                    fontsize=10, fontweight="bold" if bold else "normal")
        boxes[name] = (x, y, height)

    color = "#1b8a7a"

    def hline(x1, x2, y, label):
        ax.plot([x1, x2], [y, y], color=color, lw=1.5)
        ax.text((x1 + x2) / 2, y + 0.15, label, ha="center", fontsize=10,
                color=color, fontweight="bold")

    def row_y(table, row):
        return tables[table][1] - head_h - row_h * (row + 0.5)

    hline(boxes["users"][0] + width, boxes["reviews"][0], row_y("reviews", 1), "1 : N")
    hline(boxes["reviews"][0] + width, boxes["games"][0], row_y("reviews", 2), "N : 1")
    hline(boxes["tags"][0] + width, boxes["game_tags"][0], row_y("game_tags", 1), "1 : N")
    x_mid = boxes["games"][0] + width / 2
    y_top = tables["game_tags"][1]
    y_bottom = boxes["games"][1]
    ax.plot([x_mid, x_mid], [y_top, y_bottom], color=color, lw=1.5)
    ax.text(x_mid + 0.15, (y_top + y_bottom) / 2, "1 : N", va="center", fontsize=10,
            color=color, fontweight="bold")

    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    ax.set_title("Steam Game Recommender: ER diagram", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build steam.db and run the SQL queries")
    parser.add_argument("--source", choices=["full", "sample"], default="full")
    parser.add_argument("--skip-fk-check", action="store_true")
    parser.add_argument("--queries-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    draw_er_diagram()
    if not args.queries_only:
        create_database(source=args.source, check_fk=not args.skip_fk_check)
    run_all_queries()