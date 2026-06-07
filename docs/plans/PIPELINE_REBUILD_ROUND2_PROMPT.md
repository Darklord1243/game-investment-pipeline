# Claude Code Prompt: Round 2 — Phase 0–1 Detailed Plan

Use after Round 1 decisions (locked in `docs/plans/PIPELINE_REBUILD_SRS.md` §17). Save output to **`docs/plans/PIPELINE_REBUILD_PHASE0-1.md`**. No implementation code — actionable plan for Cursor Agent.

---

## Context

Round 1 locked the rebuild scope. Read **`docs/plans/PIPELINE_REBUILD_SRS.md`** (full SRS + **§17 locked decisions**).

**North star (one sentence):** *We measure and explain cross-platform community engagement and sentiment for a Steam game — not sales or ROI.*

**Persona:** Indie developer / small studio (delivered as a course demo).

**ML v1 (hybrid):** Documented **engagement index** for UI (not ML supervision); optional ML on **review-velocity @ 30d** when `SteamMetric` time series exists; post-release games only; baselines required if ML ships.

**Demo UX:** Sandbox-first presentation → optional live mine → cached fallback. Flask + Jinja.

**Data:** API-first + existing Steam wishlist scrape; **add `SteamMetric` now**; raise `MIN_REVIEWS_FOR_SEED` to 50–100.

---

## Task

Produce a **detailed, ordered implementation plan** for **Phase 0** and **Phase 1** only (per SRS §12). Cursor Agent will execute from this document.

Do **NOT** write code. Do **NOT** replan Round 1 decisions — treat §17 as fixed requirements.

---

## Required sections in `PIPELINE_REBUILD_PHASE0-1.md`

### 1. Executive summary

What Phase 0–1 delivers and what remains for Phase 2+.

### 2. Engagement index specification (replaces `target_score` in UI)

- Exact formula: which features, weights, normalization (min-max scope: batch vs global vs training-only)
- Explicit list of **forbidden overlaps** if any ML label is added later
- Output range, name in UI (`engagement_score` or similar — not "investment potential")
- One-paragraph **disclaimer** + one-sentence **label definition** for the web app footer
- Mapping from current `TARGET_SCORE_SPEC` (`src/models/model_trainer.py:39-56`): what to keep, rename, drop

### 3. `SteamMetric` schema & migration

- Full column list aligned with `GameSnapshot` (`src/data_collection/steam_data_miner.py:131-150`)
- ORM model sketch (types, indexes, FK to `Game`, `mined_at` semantics)
- Migration steps for existing SQLite DB (Alembic optional — manual SQLAlchemy `create_all` + backfill script is fine for course scope)
- Changes to `persist_game_snapshot` / batch miners
- Recommended `MIN_REVIEWS_FOR_SEED` (50 vs 100) with rationale
- How review-velocity @ 30d label will be computed **once** two+ snapshots exist (formula + temporal firewall — implement in Phase 2 if not Phase 1)

### 4. Demo web application — wireframes & routes

Text wireframes (ASCII or mermaid) for:

- **`/`** — landing; persona one-liner; links to Analyze (live) vs Sandbox
- **`/sandbox`** — grouped feature inputs, sample prefill dropdown, predict button
- **`/predict`** (POST) — Flow A; cache banner on fallback
- **`/samples`** (GET) — list cached games

**Presenter script (sandbox-first):** step-by-step 5-minute demo aligned with SRS §2.6, updated for locked decisions.

**Explain panel (always visible):** label definition, freshness, top 3 plain-language drivers, per-platform sentiment/emotion, limitations.

**Tech:** Flask + Jinja templates; which files to create under `src/api/templates/`; what to remove from inline `HTML_TEMPLATE` in `webapp.py`.

### 5. Cached demo dataset

- Final list of **8–12 post-release games** (hits, flops, no-Twitch, etc.)
- For each: expected engagement-index tier (high/medium/low), one-line narrative for the comparison slide
- File format: e.g. `data/demo_samples.json` + precomputed feature rows — specify schema
- How `/predict` selects cache vs live

### 6. Phase 0 task list (security + doc hygiene)

Ordered tasks with **exit criteria**:

- Secret audit (`client_secret_*.json`, git history commands to run)
- Archive list: notebooks, stale MDs, root `youtube_data_miner.py`
- README / CLAUDE.md alignment (remove phantom modules, fix CI description S3 vs git commit)
- `.env.example` AWS fields

Each task: **files touched**, **verification command**, **rollback**.

### 7. Phase 1 task list (schema + index + minimal demo backend)

Ordered tasks for Cursor Agent:

1. `SteamMetric` ORM + miner persistence
2. Engagement index module (single source of truth — suggest path under `src/features/` or `src/models/`)
3. Retire or gate circular `target_score` training in `model_trainer.py` (describe behavior: skip ML until label exists, or train only on external label stub)
4. Webapp routes: `/sandbox`, `/samples`, cache layer; SHAP or interim explainability
5. Tests to add (Steam miner, engagement index, `/sandbox` API)

Each task: dependencies, exit criteria, **do not** scope-creep into Phase 2 collection volume or YouTube top-N.

### 8. Testing & verification checklist

Commands to run after Phase 0–1 (pytest targets, manual demo dry-run).

### 9. Risks & open micro-decisions

Only **implementation-level** items still open (e.g. exact `MIN_REVIEWS=50` vs `100`). No replays of Round 1 persona/positioning.

---

## Instructions

- Cite repo files and line numbers where relevant.
- Keep course-project scope: SQLite, Flask, no React, no paid APIs.
- Phase 0–1 plan should be completable before Phase 2 (repeated snapshots, YouTube top-N, full ML retrain).
- End with **"Cursor Agent kickoff prompt"** — a single paragraph the owner can paste into Cursor Agent to start implementation from this plan.

---

## Locked decisions reference (do not reopen)

| # | Decision |
|---|---|
| P-2 | Engagement-intelligence wedge; not sales/ROI |
| P-1 | Course demo → indie developer persona |
| ML-1 | Hybrid: engagement index (UI) + review-velocity ML stretch |
| ML-2 | Post-release only |
| UX-1 | Sandbox-first + live + cache |
| DATA-1/3 | API + wishlist scrape; `SteamMetric` now |
| Scope | MVP table in SRS §17.5 |
