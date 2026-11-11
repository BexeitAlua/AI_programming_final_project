# Steam Game Recommender (AI programming final project)

Recommend Steam games to users from their review history and playtime.
Compared methods: popularity baseline, item-based CF, ALS (temporal split, Recall@K, NDCG@K, cold-start).

## Data

Source: **Game Recommendations on Steam** (Kaggle, Anton Kozyriev) —
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam


| File | Columns / role |
|---|---|
| `recommendations.csv` | `app_id, helpful, funny, date, is_recommended, hours, user_id, review_id` — ~41M user reviews (interactions) |
| `games.csv` | `app_id, title, date_release, win, mac, linux, rating, positive_ratio, user_reviews, price_final, price_original, discount, steam_deck` |
| `users.csv` | `user_id, products, reviews` |
| `games_metadata.json` | JSON Lines: `app_id, description, tags` |

`data/raw/*` and `data/processed/*` are in `.gitignore`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Preprocessing 

Run from the project root:

```bash
python -m src.preprocessing                 
python -m src.preprocessing --max-chunks 4   
```

Options: `--chunksize 500000 --buckets 16 --sample-users 50000 --min-interactions 5 --seed 42`.

What it does (`src/preprocessing.py`):

1. Reads `recommendations.csv` with `chunksize` and only the needed columns
   (`review_id, user_id, app_id, date, is_recommended, hours`).
2. Cleans every chunk with fixed rules (see `results/cleaning_rules.csv`): drops rows with missing keys,
   invalid dates, invalid flag, missing/negative hours, `app_id` not in `games.csv`, `user_id` not in `users.csv`.
3. Casts ids to `int32` and hours to `float32` after a range check; measures memory before/after and time per chunk.
4. Writes Parquet parts bucketed by `user_id`, then removes **global** duplicates
   (`review_id` – keep first; `(user_id, app_id)` – keep the latest review).
5. Builds a controlled experiment sample: 50 000 random users (seed 42) with at least 5 reviews.

Outputs:

| Path | Content |
|---|---|
| `data/processed/recommendations_clean/` | cleaned reviews, Parquet parts |
| `data/processed/sample_reviews.parquet` | experiment sample |
| `data/processed/{games,users,games_metadata}.parquet` | other tables with optimised dtypes |
| `results/data_audit.json` | raw-file statistics, removed rows per rule, memory and time measurements |
| `results/data_audit_sample.json` | quality audit of the first 100k rows (notebook) |
| `results/cleaning_rules.csv` | "problem → rule → reason → rows affected" table |
| `results/memory_log.csv` | memory (MB) before/after and seconds for every chunk |
| `results/sample_info.json` | size and parameters of the experiment sample |

The notebook `notebooks/experiment.ipynb` shows the first look at the data (100k rows), the audit and the results.
