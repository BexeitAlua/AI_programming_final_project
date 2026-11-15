"""Console menu of the Steam Game Recommender.

Run from the project root:
    uv run python main.py                 # Popularity, Item-CF, ALS (+ Two-Tower if Qdrant is running)
    uv run python main.py --no-two-tower  # without Qdrant / models/

The menu fits Popularity, Item-CF and ALS on the whole experiment sample
(data/processed/sample_reviews.parquet) at startup. Offline metrics on the temporal split come from
`python -m src.evaluation` and are read from results/metrics/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.recommenders import ALSRecommender, ItemCFRecommender, PopularityRecommender
from src.utils import PROCESSED_DIR, PROJECT_ROOT, RESULTS_DIR

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
SAVE_PATH = RESULTS_DIR / "recommendations" / "saved_recommendations.csv"
TOP_K = 10
LINE = "=" * 50

MODEL_NAMES = {"1": "Popularity", "2": "Item-CF", "3": "ALS", "4": "Two-Tower"}
METRIC_NAMES = {"popularity_all": "Popularity", "popularity_180d": "Popularity (last 180 days)",
                "item_cf": "Item-CF", "als": "ALS", "two_tower": "Two-Tower (id)",
                "two_tower_content": "Two-Tower (content)", "two_tower_hybrid": "Two-Tower (hybrid)",
                "two_tower_content_no_review_stats": "Two-Tower (content, no review stats)",
                "two_tower_ranker": "Two-Tower + LightGBM"}


def ask(prompt: str) -> str:
    """input() that treats Ctrl+D / the end of piped input as 'exit'."""
    try:
        return input(prompt).strip()
    except EOFError:
        return "0"


def load_data() -> tuple:
    """Experiment sample and game catalog from step 5."""
    reviews_path = PROCESSED_DIR / "sample_reviews.parquet"
    games_path = PROCESSED_DIR / "games.parquet"
    for path in (reviews_path, games_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run first: uv run python -m src.preprocessing")
    reviews = pd.read_parquet(reviews_path).sort_values("date", kind="stable")
    games = pd.read_parquet(games_path, columns=["app_id", "title", "date_release", "price_final", "user_reviews"])
    return reviews, games.set_index("app_id")


def best_params(model: str, names: tuple, defaults: dict) -> dict:
    """Hyper-parameters chosen on val by src.evaluation (results/metrics/val_<model>.json), else defaults."""
    path = METRICS_DIR / f"val_{model}.json"
    if not path.exists():
        return defaults
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {name: saved.get(name, defaults[name]) for name in names}


def fit_models(reviews: pd.DataFrame) -> dict:
    """Fit the three classic models on the whole sample and print the time each one takes."""
    als_params = best_params("als", ("factors", "regularization", "alpha", "iterations"),
                             {"factors": 64, "regularization": 0.01, "alpha": 1.0, "iterations": 15})
    cf_params = best_params("item_cf", ("n_neighbors",), {"n_neighbors": 50})
    builders = {"1": lambda: PopularityRecommender(),
                "2": lambda: ItemCFRecommender(**cf_params),
                "3": lambda: ALSRecommender(**als_params)}
    models = {}
    for key, build in builders.items():
        start = time.perf_counter()
        models[key] = build().fit(reviews)
        print(f"  {MODEL_NAMES[key]:<11} fitted in {time.perf_counter() - start:5.1f} s")
    return models


def connect_two_tower():
    """Two-Tower + LightGBM from models/ through Qdrant, or None with the reason printed."""
    if not (MODELS_DIR / "manifest.json").exists():
        print("  Two-Tower   off: models/ folder not found (the files are shared by hand, see SERVICE_SPEC.md)")
        return None
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        from src.serving import Recommender
        recommender = Recommender(MODELS_DIR, url, os.environ.get("QDRANT_API_KEY"),
                                  os.environ.get("QDRANT_COLLECTION", "games"))
        if not recommender.client.collection_exists(recommender.collection):
            print("  Two-Tower   off: the 'games' collection is empty. Run: "
                  "uv run --env-file .env python -m src.load_qdrant")
            return None
    except OSError as error:            # e.g. LightGBM without libomp on macOS
        print(f"  Two-Tower   off: {error.__class__.__name__}: {str(error)[:80]}")
        return None
    except Exception:                   # Qdrant is not reachable
        print(f"  Two-Tower   off: Qdrant is not running at {url}. Start it: docker compose up -d qdrant")
        return None
    print(f"  Two-Tower   connected to Qdrant at {url}")
    return recommender


def search_games(games: pd.DataFrame, query: str, limit: int = 15) -> pd.DataFrame:
    """Games whose title contains `query` (case-insensitive), most reviewed first."""
    found = games[games["title"].str.contains(query, case=False, regex=False, na=False)]
    found = found.sort_values("user_reviews", ascending=False).head(limit)
    return found.reset_index()[["app_id", "title", "date_release", "price_final", "user_reviews"]]


def describe(app_ids: list, games: pd.DataFrame) -> pd.DataFrame:
    """Title, release year and price for a list of app_ids, in the same order."""
    table = games.reindex(app_ids)
    return pd.DataFrame({"rank": range(1, len(app_ids) + 1), "app_id": app_ids,
                         "title": table["title"].fillna("(unknown game)").to_numpy(),
                         "year": pd.to_datetime(table["date_release"]).dt.year.astype("Int64").to_numpy(),
                         "price": table["price_final"].astype(float).round(2).to_numpy()})


def recommend_for_user(user_id: int, model_key: str, ctx: dict, k: int = TOP_K) -> tuple:
    """Top-k games for a user from the sample. Returns (table, note).

    Unknown users and users without liked games get the most popular games instead.
    """
    history = ctx["reviews"][ctx["reviews"]["user_id"] == user_id]
    if history.empty:
        recs = ctx["models"]["1"].recommend(np.array([user_id]), {}, k)[0]
        return describe(recs.tolist(), ctx["games"]), (
            f"User {user_id} is not in the sample of {ctx['n_users']:,} users. "
            f"Showing the {k} most popular games instead.")

    if model_key == "4":
        items = ctx["two_tower"].recommend(
            [{"app_id": int(r.app_id), "playtime": float(r.hours), "is_recommended": bool(r.is_recommended)}
             for r in history.itertuples()], k)
        table = describe([item["app_id"] for item in items], ctx["games"])
        table["score"] = [item["score"] for item in items]
        note = "No game from the history is in the catalog: popular games shown." if items and \
            items[0]["score"] is None else ""
        return table, note

    seen = {user_id: history["app_id"].to_numpy()}
    recs = ctx["models"][model_key].recommend(np.array([user_id]), seen, k)[0]
    note = ""
    if recs[0] < 0:                     # ALS / Item-CF know nothing about users without liked games
        recs = ctx["models"]["1"].recommend(np.array([user_id]), seen, k)[0]
        note = f"User {user_id} has no liked games, so {MODEL_NAMES[model_key]} has no signal: popular games shown."
    return describe([a for a in recs.tolist() if a >= 0], ctx["games"]), note


def similar_games(app_id: int, ctx: dict, k: int = TOP_K) -> pd.DataFrame:
    """Games most similar to `app_id` by Item-CF cosine similarity."""
    similar = ctx["models"]["2"].similar_items(app_id, k)
    table = describe(similar["app_id"].tolist(), ctx["games"])
    table["similarity"] = similar["similarity"].round(3).to_numpy()
    return table


def metrics_summary() -> pd.DataFrame:
    """Test metrics of every evaluated model, best NDCG@10 first."""
    rows = []
    for path in sorted(METRICS_DIR.glob("test_*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"model": METRIC_NAMES.get(m["model"], m["model"]),
                     "recall@10": m.get("recall@10"), "ndcg@10": m.get("ndcg@10"),
                     "recall@20": m.get("recall@20"), "hit_rate@10": m.get("hit_rate@10"),
                     "coverage@10": m.get("coverage@10")})
    if not rows:
        raise FileNotFoundError(f"No metrics in {METRICS_DIR}. Run: uv run python -m src.evaluation --model als")
    return pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).reset_index(drop=True)


def save_recommendations(last: dict) -> int:
    """Append the last shown recommendations to results/recommendations/saved_recommendations.csv."""
    table = last["table"]
    rows = pd.DataFrame({"user_id": last["user_id"], "app_id": table["app_id"], "title": table["title"],
                         "rank": table["rank"], "model": last["model"],
                         "score": table["score"] if "score" in table else None,
                         "generated_at": datetime.now().isoformat(timespec="seconds")})
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SAVE_PATH, mode="a", header=not SAVE_PATH.exists(), index=False)
    return len(rows)


def print_table(table: pd.DataFrame) -> None:
    print(table.to_string(index=False, na_rep="-", float_format=lambda x: f"{x:.3f}"))


def print_menu(ctx: dict) -> None:
    print(f"\n{LINE}\n   STEAM GAME RECOMMENDER")
    print(f"   {ctx['n_users']:,} users · {ctx['n_games']:,} games · {len(ctx['available'])} models")
    print(LINE)
    print(" 1. Search games by title\n 2. Recommendations for a user\n 3. Similar games\n"
          " 4. Model metrics\n 5. Save last recommendations\n 0. Exit")
    print("-" * len(LINE))


def main() -> None:
    parser = argparse.ArgumentParser(description="Steam Game Recommender: console menu")
    parser.add_argument("--no-two-tower", action="store_true", help="do not connect to Qdrant")
    args = parser.parse_args()

    print("Loading data ...")
    reviews, games = load_data()
    print(f"  {len(reviews):,} reviews from {reviews['user_id'].nunique():,} users")
    print("Fitting models on the sample ...")
    models = fit_models(reviews)
    two_tower = None if args.no_two_tower else connect_two_tower()

    counts = reviews["user_id"].value_counts()
    ctx = {"reviews": reviews, "games": games, "models": models, "two_tower": two_tower,
           "n_users": reviews["user_id"].nunique(), "n_games": reviews["app_id"].nunique(),
           "available": [k for k in MODEL_NAMES if k != "4" or two_tower is not None],
           "examples": sorted(counts[counts.between(10, 30)].index[:3].tolist()),
           "last": None}

    while True:
        print_menu(ctx)
        choice = ask("Choose an action: ")
        try:
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == "1":
                query = ask("Part of the title: ")
                if not query:
                    print("Empty query.")
                    continue
                found = search_games(games, query)
                if found.empty:
                    print(f"No games with '{query}' in the title.")
                else:
                    print_table(found)
            elif choice == "2":
                user_id = int(ask(f"Steam user_id (e.g. {', '.join(map(str, ctx['examples']))}): "))
                options = " ".join(f"[{k}] {MODEL_NAMES[k]}" for k in ctx["available"])
                model_key = ask(f"Model {options}: ")
                if model_key not in ctx["available"]:
                    print("Unknown model." if model_key != "4" else
                          "Two-Tower is off: start Qdrant and restart the menu.")
                    continue
                history = reviews[reviews["user_id"] == user_id]
                if not history.empty:
                    liked = games.reindex(history.loc[history["is_recommended"], "app_id"].tail(5))["title"]
                    print(f"History: {len(history)} reviews, recently liked: {', '.join(liked.fillna('?'))}")
                table, note = recommend_for_user(user_id, model_key, ctx)
                if note:
                    print(f"! {note}")
                print(f"\nTop-{TOP_K} for user {user_id} ({MODEL_NAMES[model_key]})")
                print_table(table)
                ctx["last"] = {"user_id": user_id, "model": MODEL_NAMES[model_key], "table": table}
            elif choice == "3":
                app_id = int(ask("app_id of the game (find it with action 1): "))
                if app_id not in games.index:
                    print(f"No game with app_id {app_id}.")
                    continue
                table = similar_games(app_id, ctx)
                if table.empty:
                    print(f"'{games.loc[app_id, 'title']}' has no liked reviews in the sample, "
                          "so Item-CF cannot find similar games (cold start).")
                else:
                    print(f"\nGames similar to '{games.loc[app_id, 'title']}' (Item-CF, cosine similarity)")
                    print_table(table)
            elif choice == "4":
                print("\nTest metrics (temporal split, results/metrics/)")
                print_table(metrics_summary())
            elif choice == "5":
                if ctx["last"] is None:
                    print("Nothing to save yet: show recommendations first (action 2).")
                else:
                    n = save_recommendations(ctx["last"])
                    print(f"Saved {n} rows to {SAVE_PATH.relative_to(PROJECT_ROOT)}")
            else:
                print("Unknown action, choose 0-5.")
        except ValueError:
            print("Please enter a whole number.")
        except FileNotFoundError as error:
            print(f"File not found: {error}")
        except Exception as error:      # keep the menu alive (e.g. Qdrant stopped while the menu runs)
            print(f"Error: {error.__class__.__name__}: {error}")
        finally:                        # runs after `continue` too
            if choice != "0":
                ask("\nPress Enter to return to the menu")


if __name__ == "__main__":
    main()
