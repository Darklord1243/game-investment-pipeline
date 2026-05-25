# Game Investment Potential Predictor

**Predictive analytics for game investment using automated multi-platform ETL and ensemble machine learning.** The system mines engagement data from Steam, Twitch, YouTube, and Reddit, engineers 50+ features through SQL-native window functions, and trains a temporal-aware VotingRegressor ensemble — all orchestrated by a weekly GitHub Actions CI/CD pipeline.

## Architecture & Tech Stack

| Layer | Technology |
|---|---|
| **Database** | SQLite via SQLAlchemy 2.0 ORM (`game_metrics.db`) |
| **ETL** | Python 3.12, `requests`, `BeautifulSoup`, custom rate-limiters |
| **Feature engineering** | SQL CTEs with window functions (`PARTITION BY batch_date`) |
| **ML** | LightGBM, XGBoost, Random Forest → `VotingRegressor` (scikit-learn) |
| **Serving** | Flask (`app.py` for prediction, `webapp.py` for interactive mining) |
| **CI/CD** | GitHub Actions — weekly cron + `workflow_dispatch` |

## The ETL Layer

Each platform miner is a self-contained module that writes directly to the normalized SQLite schema. No intermediate CSVs, no flat-file staging.

- **Strictly typed DataClasses** — `GameSnapshot`, `StreamSnapshot`, `VideoSnapshot`, `PostSnapshot` are frozen, `slots=True` dataclasses that serve as the single contract between API ingestion and ORM persistence.
- **Custom RateLimiters** — `SteamRateLimiter`, `TwitchRateLimiter`, `YouTubeRateLimiter`, and `RedditRateLimiter` implement sliding-window pacing with exponential backoff. Every API call is gated.
- **Seeder-first design** — `steam_data_miner.py` seeds the canonical `Game` dimension table before any engagement miner runs, ensuring referential integrity across all four platforms.
- **Checkpointing** — miners query the database for the latest record per game and ingest only delta data, making each run incremental and idempotent.

## SQL Feature Engineering

> **Key design decision:** In-memory Pandas global sums were replaced with SQL Window Functions scoped to `batch_date` to prevent temporal data leakage.

The `DatabaseFeatureEngineer` class (`src/features/sql_feature_engineer.py`) builds a single CTE-based query that produces 50+ features directly in the database engine:

- **Phase 1** — Platform aggregates: viewer stats (Twitch), view/like/comment stats (YouTube), score/sentiment/karma stats (Reddit).
- **Phase 2** — Derived features: viral velocity, cross-platform synergy, growth momentum, sentiment authenticity, creator concentration.
- **Phase 3** — Competitive scores: `market_share`, `platform_dominance`, and `competitive_score`, each computed using `SUM(...) OVER (PARTITION BY batch_date)` so denominators are scoped to the **current batch only**, never peeking at future data.

Custom SQLite UDFs (`sqrt`, `power`) are registered at query time to support population standard deviation and root-cube expressions that SQLite lacks natively.

## Model Training & Validation

`GameInvestmentPredictor` in `src/models/model_trainer.py`:

- **Target construction** — A 16-feature min-max normalized weighted composite scaled to 1–100 (`target_score`), with normalization statistics fit **only on the training split**.
- **Temporal split** — Train/test separation uses `release_date` quantile cutoff (80/20 default). **No random shuffling.** Older releases train, newer releases test — mirroring real-world deployment where you predict the future from the past.
- **Ensemble** — `VotingRegressor` averaging LightGBM (300 estimators), XGBoost (300 estimators), and Random Forest (200 estimators), all with fixed `random_state=42`.
- **Persistence** — Best model, feature names, and `StandardScaler` are serialized to `enhanced_model_artifacts.pkl` via `joblib` and loaded by both Flask inference apps.

## Continuous Deployment (MLOps)

The `.github/workflows/weekly_pipeline.yml` workflow acts as a **living deployment**:

1. **Triggers** — Every Monday at 00:00 UTC, or on-demand via `workflow_dispatch`.
2. **Execution order** — Steam (seed) → Twitch → YouTube → Reddit → Model Trainer.
3. **Auto-commit** — Updated `data/game_metrics.db` and `enhanced_model_artifacts.pkl` are committed back to the repository with the message `chore(data): automated weekly pipeline run`.

Fresh engagement data flows into the model every week with zero manual intervention.

## Local Setup

```bash
# 1. Clone and install
git clone <repo-url> && cd <repo>
pip install -r requirements.txt

# 2. Create .env from template (fill in API credentials)
cp .env.example .env

# 3. Seed the game dimension table and mine all platforms
python src/data_collection/steam_data_miner.py
python src/data_collection/twitch_miner.py
python src/data_collection/youtube_data_miner.py
python src/data_collection/reddit_data_miner.py

# 4. Train the model
python src/models/model_trainer.py
```

Set `DATABASE_URL=sqlite:///data/game_metrics.db` in your environment (or rely on the default in `src/database/models.py`). Run the Flask apps with `python src/api/app.py` or `python src/api/webapp.py` to serve predictions.
