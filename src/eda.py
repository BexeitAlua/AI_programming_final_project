from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR, RESULTS_DIR, save_json

FIGURES_DIR = RESULTS_DIR / "figures"
COLOR = "#1b5e8a"
COLOR_2 = "#1b8a7a"


def load_eda_data(source: str = "full") -> tuple:
    if source == "sample":
        reviews = pd.read_parquet(PROCESSED_DIR / "sample_reviews.parquet")
    else:
        reviews = pd.read_parquet(PROCESSED_DIR / "recommendations_clean",
                                  columns=["user_id", "app_id", "date", "is_recommended", "hours"])
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    return reviews, games


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def _subtitle(n: int) -> str:
    return f"n = {n:,} reviews"


def plot_top_games(reviews: pd.DataFrame, games: pd.DataFrame, n: int = 20) -> tuple:
    counts = reviews["app_id"].value_counts().head(n)
    titles = games.set_index("app_id")["title"].reindex(counts.index).fillna("unknown")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(titles.to_numpy()[::-1], counts.to_numpy()[::-1], color=COLOR)
    ax.set_xlabel("Number of reviews")
    ax.set_title(f"Top-{n} games by number of reviews\n{_subtitle(len(reviews))}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "top20_games")
    share = counts.sum() / len(reviews)
    stats = {"top_games": dict(zip(titles, counts.astype(int))),
             f"top{n}_share_of_reviews": round(float(share), 4)}
    return fig, stats


def plot_long_tail(reviews: pd.DataFrame) -> tuple:
    counts = reviews["app_id"].value_counts().to_numpy()
    ranks = np.arange(1, len(counts) + 1)
    cum_share = counts.cumsum() / counts.sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ranks, counts, color=COLOR)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Game rank (log)")
    axes[0].set_ylabel("Number of reviews (log)")
    axes[0].set_title("Long tail of game popularity")
    axes[1].plot(ranks / len(ranks) * 100, cum_share * 100, color=COLOR_2)
    axes[1].set_xlabel("Top % of games")
    axes[1].set_ylabel("% of all reviews")
    axes[1].set_title("Cumulative share of reviews")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(_subtitle(len(reviews)))
    fig.tight_layout()
    _save(fig, "long_tail")
    n_top1 = max(1, int(len(counts) * 0.01))
    n_top10 = max(1, int(len(counts) * 0.10))
    stats = {
        "games_with_reviews": int(len(counts)),
        "top1pct_games_share": round(float(cum_share[n_top1 - 1]), 4),
        "top10pct_games_share": round(float(cum_share[n_top10 - 1]), 4),
        "games_with_lt_10_reviews": int((counts < 10).sum()),
        "median_reviews_per_game": float(np.median(counts)),
    }
    return fig, stats


def plot_degree_distributions(reviews: pd.DataFrame) -> tuple:
    user_deg = reviews["user_id"].value_counts().to_numpy()
    item_deg = reviews["app_id"].value_counts().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, deg, name, color in ((axes[0], user_deg, "user", COLOR), (axes[1], item_deg, "game", COLOR_2)):
        bins = np.logspace(0, np.log10(deg.max() + 1), 40)
        ax.hist(deg, bins=bins, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Reviews per {name} (log)")
        ax.set_ylabel(f"Number of {name}s (log)")
        ax.set_title(f"{name.capitalize()} degree: median {np.median(deg):.0f}, max {deg.max():,}")
        ax.grid(alpha=0.3)
    fig.suptitle(_subtitle(len(reviews)))
    fig.tight_layout()
    _save(fig, "degree_distributions")
    stats = {
        "n_users": int(len(user_deg)),
        "user_median_reviews": float(np.median(user_deg)),
        "user_mean_reviews": round(float(user_deg.mean()), 2),
        "users_with_1_review_share": round(float((user_deg == 1).mean()), 4),
        "users_with_5plus_reviews_share": round(float((user_deg >= 5).mean()), 4),
        "n_games": int(len(item_deg)),
        "game_median_reviews": float(np.median(item_deg)),
        "game_mean_reviews": round(float(item_deg.mean()), 2),
    }
    return fig, stats


def plot_hours(reviews: pd.DataFrame) -> tuple:
    hours = reviews["hours"].to_numpy(dtype="float64")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(hours, bins=100, color=COLOR)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Hours played")
    axes[0].set_ylabel("Number of reviews (log)")
    axes[0].set_title("Playtime: raw hours")
    axes[1].hist(np.log1p(hours), bins=60, color=COLOR_2)
    axes[1].set_xlabel("log1p(hours)")
    axes[1].set_ylabel("Number of reviews")
    axes[1].set_title("Playtime: log1p(hours)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(_subtitle(len(reviews)))
    fig.tight_layout()
    _save(fig, "hours_distribution")
    by_flag = reviews.groupby("is_recommended")["hours"].median()
    stats = {
        "median_hours": round(float(np.median(hours)), 1),
        "mean_hours": round(float(hours.mean()), 2),
        "p90_hours": round(float(np.percentile(hours, 90)), 1),
        "max_hours": round(float(hours.max()), 1),
        "zero_hours_share": round(float((hours == 0).mean()), 4),
        "share_at_max": round(float((hours >= hours.max()).mean()), 4),
        "median_hours_positive": round(float(by_flag.get(True, np.nan)), 1),
        "median_hours_negative": round(float(by_flag.get(False, np.nan)), 1),
    }
    return fig, stats


def plot_reviews_over_time(reviews: pd.DataFrame) -> tuple:
    monthly = reviews.groupby(reviews["date"].dt.to_period("M")).size()
    yearly = reviews.groupby(reviews["date"].dt.year)["is_recommended"].agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(monthly.index.to_timestamp(), monthly.to_numpy(), color=COLOR)
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Number of reviews")
    axes[0].set_title("Reviews per month")
    axes[1].bar(yearly.index.astype(str), yearly["mean"] * 100, color=COLOR_2)
    axes[1].axhline(reviews["is_recommended"].mean() * 100, color="black", ls="--", lw=1,
                    label="overall")
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("% positive")
    axes[1].set_title("Share of positive reviews by year")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(_subtitle(len(reviews)))
    fig.tight_layout()
    _save(fig, "reviews_over_time")
    stats = {
        "positive_share": round(float(reviews["is_recommended"].mean()), 4),
        "peak_month": str(monthly.idxmax()),
        "peak_month_reviews": int(monthly.max()),
        "reviews_by_year": {str(k): int(v) for k, v in yearly["size"].items()},
        "positive_share_by_year": {str(k): round(float(v), 4) for k, v in yearly["mean"].items()},
    }
    return fig, stats


def sparsity_summary(reviews: pd.DataFrame) -> dict:
    n_users = int(reviews["user_id"].nunique())
    n_games = int(reviews["app_id"].nunique())
    nnz = int(len(reviews))
    return {
        "n_users": n_users,
        "n_games": n_games,
        "interactions": nnz,
        "density": float(nnz / (n_users * n_games)),
        "sparsity": float(1 - nnz / (n_users * n_games)),
        "dense_matrix_gb_float32": round(n_users * n_games * 4 / 1024 ** 3, 1),
    }


def plot_interactive_games(reviews: pd.DataFrame, games: pd.DataFrame, min_reviews: int = 20):
    import plotly.express as px

    stats = reviews.groupby("app_id").agg(review_count=("is_recommended", "size"),
                                          positive_pct=("is_recommended", "mean"),
                                          median_hours=("hours", "median")).reset_index()
    stats = stats[stats["review_count"] >= min_reviews]
    stats["positive_pct"] = (stats["positive_pct"] * 100).round(1)
    stats = stats.merge(games[["app_id", "title", "price_final", "date_release"]], on="app_id", how="left")
    stats["price_category"] = pd.cut(stats["price_final"], bins=[-0.01, 0, 10, 30, np.inf],
                                     labels=["Free", "Under 10", "10-30", "30+"])
    stats["release_year"] = pd.to_datetime(stats["date_release"]).dt.year
    fig = px.scatter(
        stats, x="review_count", y="positive_pct", color="price_category", log_x=True,
        hover_name="title",
        hover_data={"review_count": ":,", "positive_pct": True, "median_hours": ":.1f",
                    "price_final": ":.2f", "release_year": True, "price_category": False},
        labels={"review_count": "Number of reviews (log)", "positive_pct": "% positive reviews",
                "price_category": "Price"},
        title=f"Games: popularity vs share of positive reviews (games with >= {min_reviews} reviews)",
        render_mode="webgl", opacity=0.6, height=600,
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIGURES_DIR / "interactive_games.html", include_plotlyjs="cdn")
    return fig


def run_eda(source: str = "full") -> dict:
    matplotlib.use("Agg")
    reviews, games = load_eda_data(source)
    summary = {"source": source, "rows": int(len(reviews))}
    for name, func in (("top_games", lambda: plot_top_games(reviews, games)),
                       ("long_tail", lambda: plot_long_tail(reviews)),
                       ("degrees", lambda: plot_degree_distributions(reviews)),
                       ("hours", lambda: plot_hours(reviews)),
                       ("time", lambda: plot_reviews_over_time(reviews))):
        fig, stats = func()
        plt.close(fig)
        summary[name] = stats
        print(f"{name}: done")
    summary["sparsity"] = sparsity_summary(reviews)
    plot_interactive_games(reviews, games)
    save_json(summary, RESULTS_DIR / "eda_summary.json")
    print(f"Figures saved to {FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA figures")
    parser.add_argument("--source", choices=["full", "sample"], default="full")
    run_eda(parser.parse_args().source)