# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game investment analysis system that mines engagement data from Twitch, YouTube, Reddit, and Steam APIs, engineers 50+ features, trains ensemble ML models, and serves investment-potential predictions via Flask web apps.

## Commands

```bash
# Install dependencies (Anaconda recommended)
pip install -r requirements.txt
# Additional ML libraries for enhanced pipeline:
pip install xgboost lightgbm catboost optuna scipy statsmodels

# Run tests (mock API responses — no live endpoints)
pytest tests/

# Run a single test file
pytest tests/test_twitch_games.py

# Start the simple Flask predictor (app.py)
python src/api/app.py

# Start the full mining + prediction webapp (webapp.py)
python src/api/webapp.py

# Fix test script (Anaconda path)
run_test.bat
```

## Architecture

**Data pipeline:** Mining scripts (per-platform) → per-platform CSVs → aggregation & merge on `game_title` → feature engineering → model training / prediction.

### Module layout

| Directory | Purpose |
|---|---|
| `src/data_collection/` | API miners for Twitch, YouTube, Reddit. `mining.py` provides unified `mine_*()` entry points used by the Flask app. |
| `src/features/` | `enhanced_features.py` — `AdvancedFeatureEngineer` class with 6 feature groups (temporal, cross-platform, engagement quality, network, viral, competitive). `feature_pipeline.py` — `GameInvestmentPredictor` that orchestrates mining → feature engineering → prediction. |
| `src/models/` | `enhanced_models.py` — `AdvancedModelEnsemble` and `EnhancedMLPipeline` classes for multi-model training, hyperparameter tuning, and statistical validation. |
| `src/api/` | Two Flask apps: `app.py` (simple single-game form with JSON/PDF download) and `webapp.py` (tabbed UI with "Mine & Predict" async workflow + "Direct Analysis" form). |
| `src/database/` | SQLAlchemy 2.0 ORM schema (`Game`, `TwitchMetric`, `YouTubeMetric`, `RedditMetric`) for a normalized ELT pipeline. Not yet wired into the mining scripts. |

### Key artifacts

- `enhanced_model_artifacts.pkl` — serialized best model, feature names, and scaler, loaded by both Flask apps at startup
- `deployment_summary.json` — metadata from the last training run (best model name, R², top recommendations)
- Per-platform CSVs: `twitch_game_streams.csv`, `youtube_game_videos.csv`, `reddit_game_posts.csv`, `steam_significant_games.csv`

### Jupyter notebooks

- `Enhanced_Game_Investment_Analysis.ipynb` — primary analysis notebook with the full enhanced pipeline
- `Game_Investment_Potential_Prediction.ipynb` — original/simpler analysis
- `Game_Investment_Potential_Predictor_StepByStep.ipynb` — step-by-step tutorial version
- `leakage_free_model_check.ipynb` — data leakage audit

## Environment & Secrets

Copy `.env.example` to `.env` and fill in credentials for Twitch (client ID + secret), YouTube (API key), Reddit (client ID + secret + user/pass), and Flask (secret key). The `.gitignore` excludes `.env`, `*.json` secrets, `*.pkl`, and `*.db`.

## Code conventions (from `.cursor/rules/`)

- PEP8, type hints, docstrings on all functions/classes
- Use official APIs for all data collection; scraping only as documented last resort
- Mock API responses in tests (never hit live endpoints)
- pytest for all tests; target 80%+ coverage
- No global variables; organize into logical modules
