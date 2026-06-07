---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Game engagement intelligence system that mines engagement data from Twitch, YouTube, Reddit, and Steam APIs, engineers 50+ SQL features, computes a descriptive Engagement Score, and serves it via a Flask + Jinja demo webapp. ML training is gated until an external review-velocity label exists (Phase 2).

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

# Start the demo webapp (sandbox + live predict + samples)
python src/api/webapp.py

# Regenerate cached demo samples
python scripts/build_demo_samples.py
```

## Architecture

**Data pipeline:** Mining scripts (per-platform) → SQLite ORM tables → SQL feature engineering → engagement index (UI) / ML training (Phase 2, gated).

### Module layout

| Directory | Purpose |
|---|---|
| `src/data_collection/` | API miners for Steam, Twitch, YouTube, Reddit. Each writes to normalized ORM tables. |
| `src/features/` | `sql_feature_engineer.py` — `DatabaseFeatureEngineer` with CTE-based 50+ feature matrix. `engagement_index.py` — single source of truth for the descriptive Engagement Score (UI-only). |
| `src/models/` | `model_trainer.py` — `GameInvestmentPredictor` ensemble trainer (gated when no external label). |
| `src/api/` | `webapp.py` — Flask + Jinja demo (`/`, `/sandbox`, `/predict`, `/samples`, `/health`). Templates in `src/api/templates/`. |
| `src/database/` | SQLAlchemy 2.0 ORM schema (`Game`, `TwitchMetric`, `YouTubeMetric`, `RedditMetric`, `SteamMetric`). |

### Key artifacts

- `data/demo_samples.json` — 10 cached demo games + reference distribution for offline scoring
- `enhanced_model_artifacts.pkl` — legacy ML artifact (not required for Phase 1 demo)
- `deployment_summary.json` — metadata from the last training run
- `data/game_metrics.db` — SQLite warehouse (gitignored; published to S3 by CI)

### Archived notebooks

Exploratory notebooks live under `archive/notebooks/`. `new_model_try.ipynb` is kept as honest baseline evidence (Test R²≈0.03), non-authoritative.

## Environment & Secrets

Copy `.env.example` to `.env` and fill in credentials for Twitch (client ID + secret), YouTube (API key), Reddit (client ID + secret + user/pass), Flask (secret key), and AWS (CI artifact upload). The `.gitignore` excludes `.env`, `client_secret_*.json`, `*.pkl`, and `*.db`.

## Code conventions (from `.cursor/rules/`)

- PEP8, type hints, docstrings on all functions/classes
- Use official APIs for all data collection; scraping only as documented last resort
- Mock API responses in tests (never hit live endpoints)
- pytest for all tests; target 80%+ coverage on miners/utils
- No global variables; organize into logical modules
