# Claude Code Prompt: Pipeline Rebuild SRS / Blueprint

Use this file as the initial prompt when running Claude Code (`/model opus`). Save Claude Code's output to `docs/plans/PIPELINE_REBUILD_SRS.md`. Implementation is done later in Cursor Agent per `docs/plans/README.md`.

---

## Task

Produce a multi-round **SRS (Software Requirements Specification) and technical blueprint** for rebuilding the Game Investment Potential pipeline. Do **NOT** write implementation code yet. The project owner will review the SRS over multiple discussion rounds before any coding begins.

---

## Project context

**Stated goal:** A web-mining tool that helps assess whether a game is worth investing in, by collecting multi-platform engagement signals and training a predictive model with scientific rigor.

**Original product intent** (from `Project/Web Mining Idea.txt`):

- Mine Steam tabs/comments, YouTube/Twitch/Reddit discussion volume for successful games
- Define "success" scientifically via sales, cost, ROI — not just engagement alone
- Eventually support a demo/promo workflow for AI-generated marketing assets

**Course context:** This is a Web Mining course project. The repo includes lecture/lab notebooks on API usage, BeautifulSoup, Selenium, and regex scraping. Production code lives under `src/`; notebooks are exploratory/legacy.

**Honest state of the product idea:** The concept is currently messy and under-specified. The team has **not** investigated the current market, competitive landscape, or who would actually use this product. The SRS must treat **product discovery** as a first-class deliverable — not assume "investors want a score from 1–100" without evidence.

---

## Product & user discovery (NEW — required in SRS)

The SRS must include a dedicated **Product & Market** section that goes beyond technical architecture.

### What we need to figure out

1. **Who are the potential users?** At minimum, consider and compare:
   - **Game investors / publishers** — scouting titles for funding or acquisition
   - **Indie game developers / studios** — self-assessing launch readiness or pitch strength
   - **Marketing / UA teams** — deciding where to allocate spend pre- or post-launch
   - **Course/demo audience** — professors, classmates, portfolio reviewers
   - Other segments you identify from brief competitive research

2. **What job are they hiring this tool to do?** (Jobs-to-be-done framing)
   - Example: "Tell me if this Steam title is worth a meeting" vs "Help me tune my launch trailer strategy" — different users, different inputs/outputs.

3. **Competitive / adjacent landscape**
   - What exists today? (SteamDB, Gamalytic, Newzoo, Steam wishlist trackers, social listening tools, Steam review analyzers, etc.)
   - Where would our tool differentiate — or should we narrow scope to a niche wedge?
   - What do incumbents do that we should not rebuild?

4. **Demo web application**
   - We want a **web app suitable for demonstration** (course presentation, stakeholder walkthrough, portfolio).
   - **Current state:** `src/api/webapp.py` exists but is minimal — single Steam title input, mines live APIs, returns JSON `target_score`. There is no polished UX, no "enter features manually" flow, no role-based views, and no `app.py` simple predictor mentioned in stale docs.
   - **Desired direction (high level, to be refined in SRS):**
     - User selects or is assigned a **persona** (investor vs game builder — or a unified flow if personas converge)
     - User provides inputs — either by **game name** (trigger live/historical mining) or by **entering feature values directly** (what-if / sandbox mode for demo when APIs are slow or quota-limited)
     - System returns **prediction + explanation** (top contributing features, confidence/uncertainty, recommended next action)
     - Demo mode should work **offline or with cached sample games** when live mining fails (quota, missing game)
   - The SRS should specify: pages/routes, input modes, output format, persona differences (if any), and MVP vs stretch UI — without jumping to React vs Flask templates unless justified.

5. **Success criteria for the product (not just the model)**
   - What does a successful demo look like in 5 minutes?
   - What would make an investor vs a developer trust (or distrust) the output?
   - Minimum viable credibility signals (explainability, data freshness, label definition transparency)

Flag unresolved product choices as `[DECISION REQUIRED]` with options and recommendations.

---

## Current production architecture (authoritative — inspect these files)

Read and cite specific files when making claims:

| Component | Path |
|---|---|
| Steam seeder/miner | `src/data_collection/steam_data_miner.py` |
| Twitch miner | `src/data_collection/twitch_miner.py` |
| YouTube miner | `src/data_collection/youtube_data_miner.py` |
| Reddit miner | `src/data_collection/reddit_data_miner.py` |
| ORM schema | `src/database/models.py` |
| SQL feature engineering | `src/features/sql_feature_engineer.py` |
| Model training | `src/models/model_trainer.py` |
| Flask inference API (minimal demo today) | `src/api/webapp.py` |
| CI/CD pipeline | `.github/workflows/weekly_pipeline.yml` |
| Feature math audit | `FEATURE_MATHEMATICS_AUDIT.md` |
| Leakage audit notebook | `leakage_free_model_check.ipynb` |
| README (production docs) | `README.md` |
| Stale docs (may contradict code) | `CLAUDE.md`, `SYSTEM_STATUS_REPORT.md`, `Integration_Guide.md`, `Enhanced_Analysis_README.md` |

**Known production flow:** Steam seeds `games` table → platform miners write fact tables → `DatabaseFeatureEngineer` builds ~77 batch-scoped features → `GameInvestmentPredictor` trains VotingRegressor (LGBM+XGB+RF) → `enhanced_model_artifacts.pkl` → `webapp.py`.

---

## Critical problems already identified (validate and expand)

1. **No ground-truth investment label.** `target_score` is a weighted composite of engagement features the model also sees — circular/tautological supervision.
2. **Steam enrichment not persisted.** `GameSnapshot` fetches reviews/sentiment/players/wishlist but ORM only stores `appid`, `steam_name`, `release_date`. Target spec references `steam_*` columns that default to 0.
3. **Thin data sampling.** YouTube: 1 video/game; Reddit: ~15 posts; Twitch: live snapshot only; no time series.
4. **API quota bottlenecks.** YouTube 10k units/day; Reddit/Twitch/Steam rate limits. Owner is considering web scraping as alternative — evaluate tradeoffs per platform (ToS, fragility, volume, legal).
5. **Documentation drift.** `enhanced_features.py`, `enhanced_models.py`, `app.py`, `mining.py` referenced in docs/notebooks but missing from repo. Root `youtube_data_miner.py` is legacy CSV path. Notebooks import missing modules.
6. **Inflated metrics.** `SYSTEM_STATUS_REPORT.md` / `deployment_summary.json` report R² ~0.96–0.99 from notebook-era ElasticNet on engagement-derived targets.
7. **Test gaps.** Miner unit tests with mocks exist; no tests for SQL features, training, or inference.
8. **Partial scraping already.** Steam store HTML parsed for wishlists; `selenium` in requirements but unused in `src/`.
9. **No product/market definition.** User personas, competitive positioning, and demo UX are unspecified; technical pipeline exists without a validated user problem.

---

## Owner's rebuild goals

1. **Data collection:** Evaluate hybrid API + scraping strategy to increase volume and reduce quota pain — without ignoring legal/ToS constraints.
2. **Scientific rigor:** Proper label definition, feature justification, model selection with ablation — not "this seemed to work."
3. **Tightened architecture:** Single source of truth; deprecate dead paths; align docs with code.
4. **Demo web application:** Design a demonstrable web app for course/stakeholder use — manual feature input, game-name mining, explainable output; persona-aware if warranted.
5. **Product clarity:** Define target users, use cases, and competitive context before locking ML details.
6. **Iterative planning:** This SRS is round 1. Flag every major decision as `[DECISION REQUIRED]` with options and tradeoffs for owner discussion.

---

## Required deliverable structure

Write **`docs/plans/PIPELINE_REBUILD_SRS.md`** with these sections:

### 1. Executive Summary

One page: current state, why rebuild is needed, proposed north-star outcome (technical + product).

### 2. Product & Market Discovery *(new — do not skip)*

- Problem statement from the **user's** perspective (not the model's)
- Candidate user personas (investor, game builder, others) with pains, goals, and willingness to trust ML output
- Jobs-to-be-done per persona (or unified JTBD if personas merge)
- Competitive / adjacent tools landscape (brief desk research — name real products/sources)
- Positioning options: who is the **primary** user for v1 demo vs long-term product
- Demo narrative: 5-minute walkthrough script for presentation day
- `[DECISION REQUIRED]` items for persona and scope

### 3. Problem Statement & Success Criteria (ML + product)

- Define "investment potential" operationally
- Propose 2–3 candidate **ground-truth labels** (e.g., revenue proxy, review velocity at 30/90 days, player retention, wishlist-to-purchase conversion) with data availability analysis
- Business KPIs for model success (not just R²)
- Product KPIs for demo success (task completion time, comprehension, trust signals)

### 4. Stakeholder & Constraints

- Course project scope vs production ambitions
- Legal/ToS constraints per platform (API-only vs scrape-allowed vs gray area)
- Budget: free API tiers, no paid data sources unless justified
- Infra: Anaconda Python, SQLite (or justify Postgres), GitHub Actions, optional AWS S3

### 5. Demo Web Application Specification *(new)*

- Current gap analysis vs `src/api/webapp.py`
- User flows (diagrams welcome):
  - **Flow A:** Enter Steam game name → mine (or cache) → predict → explain
  - **Flow B:** Enter feature values manually → predict → explain (sandbox / what-if)
  - **Flow C (optional):** Persona-specific views or recommendations
- Pages, API endpoints, input validation, error states (quota exhausted, unknown game)
- Offline / demo fallback (cached sample games, precomputed features)
- MVP UI scope vs stretch (charts, PDF export, comparison mode)
- Tech stack recommendation (extend Flask + templates vs SPA) with rationale
- Accessibility and "explainability UI" requirements (show label definition, data freshness, limitations disclaimer)

### 6. Data Requirements

- Entity-relationship diagram (extend current schema: propose `SteamMetric`, time-series batches, etc.)
- Per-platform field catalog: what to collect, granularity, refresh cadence
- Volume targets (games, observations/game, history depth)
- Data quality rules and monitoring

### 7. Collection Strategy — API vs Scraping Decision Matrix

For Steam, Twitch, YouTube, Reddit:

| Platform | Recommended primary | Fallback | Rationale | Estimated throughput | Risk |

Include specific endpoints/scrape targets, rate-limit math, and whether Selenium is warranted.

### 8. Feature Engineering Specification

- Feature catalog grouped by: pre-release signals, post-release signals, cross-platform, temporal
- Leakage policy (reference `FEATURE_MATHEMATICS_AUDIT.md` findings)
- Which current ~77 features to keep, redesign, or drop
- Map features to **web app inputs** (which are user-enterable in sandbox mode vs mined-only)
- Time-window semantics (7d/30d/90d post-release)

### 9. Modeling Specification

- Label → feature separation (eliminate circular targets)
- Baseline models (linear, tree, naive engagement index)
- Candidate models with selection criteria
- Validation protocol: temporal split, walk-forward, grouped CV by genre/publisher
- Hyperparameter tuning scope (Optuna yes/no)
- Calibration and uncertainty quantification
- Explainability requirements for investment decisions and demo UI

### 10. System Architecture (target state)

- Component diagram: ingestion → warehouse → feature store → training → registry → serving → **demo web app**
- Deprecation list: notebooks, CSV path, stale docs, root-level scripts
- API design: extend/replace `webapp.py`; note missing `app.py`

### 11. MLOps & Operations

- Extend or replace `weekly_pipeline.yml`
- Model registry, drift detection, retraining triggers
- Secrets management (`.env.example` fields)

### 12. Migration Plan (phased)

Phase 0 (product + audit) → Phase 1 (labels + schema) → Phase 2 (collection) → Phase 3 (features) → Phase 4 (model) → Phase 5 (demo web app + serving)

Each phase: scope, exit criteria, rollback.

### 13. Testing Strategy

- Unit, integration, contract tests for miners
- Feature regression tests (SQL snapshot comparisons)
- Model smoke tests with fixed seed data
- **Web app E2E / API tests** for predict flows (live and sandbox)
- Target: coverage goals per module

### 14. Risks & Mitigations

Include: ToS violations, label noise, small-N games, Steam API changes, YouTube quota, circular features, overfitting to historical hits, **building a model nobody asked for**, demo failure during live presentation.

### 15. Open Questions — `[DECISION REQUIRED]`

List 12–18 concrete decisions for owner review. Must include at least:

- Primary user persona for v1
- Pre-release vs post-release scoring
- Label definition
- API vs scrape posture per platform
- Demo app: mine-live vs sandbox-first for presentation
- Build vs integrate (e.g., embed existing Steam analytics)

Format each as:

- **Question:**
- **Options:** A / B / C
- **Recommendation:**
- **Impact if wrong:**

### 16. Appendices

- Glossary
- Current vs target file inventory
- References (Steam Web API, IGDB, Gamalytic, SteamDB policies, etc.)
- Sample demo games list (hits, flops, edge cases) for testing narrative

---

## Instructions for Claude Code

- Read the files listed above before writing. Cite file paths and line-level findings where relevant.
- Be skeptical of high R² claims; explain why current metrics are misleading.
- **Do not assume the product is validated.** Propose how to validate users and use cases within course-project constraints (lightweight desk research, instructor/stakeholder interviews, competitor feature matrix).
- Prefer pragmatic course-project scope over enterprise over-engineering, but do not sacrifice label validity or demo credibility.
- Do **NOT** produce implementation code or diffs.
- Write for a technical owner who will discuss this document over **2–4 review rounds** before Cursor Agent implements it.
- After completing the SRS, end with a **"Round 1 Discussion Agenda"** — the **7** highest-priority decisions to resolve first (must include at least 2 product/UX decisions, not only ML/data).

---

## Round 1 discussion agenda (seed list for Claude Code to refine)

Owner expects these themes to surface in the agenda Claude Code produces:

1. **Primary user persona** — investor vs game builder vs dual-audience
2. **What problem we solve in one sentence** — validated against competitors
3. **Ground-truth label** — what are we actually predicting?
4. **Pre-release vs post-release** — when in the lifecycle does the tool apply?
5. **Demo app UX** — game-name mining vs manual feature sandbox (or both)
6. **API vs scraping** — posture per platform
7. **MVP scope cut** — what ships for course demo vs what is deferred
