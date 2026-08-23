# Game Engagement Intelligence

**Cross-platform community engagement and sentiment intelligence for Steam games.** The system mines engagement data from Steam, Twitch, YouTube, and Reddit, engineers 50+ features through SQL-native window functions, and serves a transparent **Engagement Score** (0–100) via a Flask demo webapp. *(Legacy name "Game Investment Potential Predictor" is deprecated — the product is descriptive engagement intelligence, not a sales forecast.)*

## Architecture & Tech Stack

| Layer | Technology |
|---|---|
| **Database** | SQLite via SQLAlchemy 2.0 ORM (`game_metrics.db`) |
| **ETL** | Python 3.12, `requests`, `BeautifulSoup`, custom rate-limiters |
| **Feature engineering** | SQL CTEs with window functions (`PARTITION BY batch_date`) |
| **Scoring** | Descriptive engagement index (`src/features/engagement_index.py`) |
| **ML (Phase 2)** | LightGBM + XGBoost + Random Forest ensemble — gated until external labels exist |
| **Serving** | Flask + Jinja (`src/api/webapp.py`) |
| **CI/CD** | GitHub Actions — weekly cron + `workflow_dispatch` |

## The ETL Layer

Each platform miner is a self-contained module that writes directly to the normalized SQLite schema. No intermediate CSVs, no flat-file staging.

- **Strictly typed DataClasses** — `GameSnapshot`, `StreamSnapshot`, `VideoSnapshot`, `PostSnapshot` are frozen, `slots=True` dataclasses that serve as the single contract between API ingestion and ORM persistence.
- **Custom RateLimiters** — `SteamRateLimiter`, `TwitchRateLimiter`, `YouTubeRateLimiter`, and `RedditRateLimiter` gate **every** API call behind sliding-window pacing (`src/utils/http.py`). Rate-limit *responses* are then handled per platform, and they differ: **Reddit backs off exponentially** (60s, doubling, capped at 300s); **Steam, Twitch and YouTube retry at a fixed interval** (5s, 60s and 60s respectively, up to 3 attempts). Twitch additionally honours Helix `Ratelimit-*` headers.
- **Seeder-first design** — `steam_data_miner.py` seeds the canonical `Game` dimension table and appends `SteamMetric` time-series rows before engagement miners run.
- **Checkpointing** — miners query the database for the latest record per game and ingest only delta data, making each run incremental and idempotent.

## SQL Feature Engineering

> **Key design decision:** In-memory Pandas global sums were replaced with SQL Window Functions scoped to `batch_date` to prevent temporal data leakage.

The `DatabaseFeatureEngineer` class (`src/features/sql_feature_engineer.py`) builds a single CTE-based query that produces 50+ features directly in the database engine:

- **Phase 1** — Platform aggregates: viewer stats (Twitch), view/like/comment stats (YouTube), score/sentiment/karma stats (Reddit).
- **Phase 2** — Derived features: viral velocity, cross-platform synergy, growth momentum, sentiment authenticity, creator concentration.
- **Phase 3** — Competitive scores: `market_share`, `platform_dominance`, and `competitive_score`, each computed using `SUM(...) OVER (PARTITION BY batch_date)` so denominators are scoped to the **current batch only**, never peeking at future data.

## Engagement Score (UI)

The descriptive **Engagement Score** is computed by `src/features/engagement_index.py` — a weighted, batch-normalized composite of 13 cross-platform features. It is **not** ML supervision and does not predict sales or ROI. The webapp exposes:

- `/sandbox` — offline manual-input scoring (quota-proof demo)
- `/predict` — live mine with cached-sample fallback
- `/samples` — 10 cached demo games

## Model Training (Phase 2 — currently gated)

`GameInvestmentPredictor` in `src/models/model_trainer.py` skips training when no external `target_variable_score` label exists. Circular `target_score` composite training is disabled until review-velocity labels are available.

## Continuous Deployment (MLOps)

The `.github/workflows/weekly_pipeline.yml` workflow acts as a **living deployment**:

1. **Triggers** — Every Monday at 00:00 UTC, or on-demand via `workflow_dispatch`.
2. **Execution order** — Steam (seed) → Twitch → YouTube → Reddit → Model Trainer (skips if no external label).
3. **Artifact publishing** — Updated `data/game_metrics.db` is pushed to **S3**; `enhanced_model_artifacts.pkl` and `deployment_summary.json` are published to **GitHub Releases** (model registry). Artifacts are **not** auto-committed back to the repository.

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

# 4. Start the demo webapp
python src/api/webapp.py
```

Set `DATABASE_URL=sqlite:///data/game_metrics.db` in your environment (or rely on the default in `src/database/models.py`).
