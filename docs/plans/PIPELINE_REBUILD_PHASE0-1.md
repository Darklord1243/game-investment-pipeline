# Phase 0–1 Implementation Plan — Game Engagement Intelligence

> **Status:** Round 2 detailed plan. **Executable by Cursor Agent.** Round 1 decisions are LOCKED (SRS §17) — do **not** reopen positioning, persona, label strategy, demo modes, stack, or scope.
> **Source of truth:** `docs/plans/PIPELINE_REBUILD_SRS.md` (§17 = locked decisions; §5/§6/§12 = referenced specs).
> **Scope of this doc:** Phase 0 (security + doc hygiene) and Phase 1 (schema + engagement index + minimal demo backend) only. Phase 2+ (repeated snapshots, YouTube top-N, full ML retrain, review-velocity label computation) is explicitly out of scope here and called out where it abuts.
> **Constraints (CLAUDE.md):** PEP8, type hints, docstrings; official APIs only (no new scraping); mock all API responses in tests (never hit live endpoints); pytest 80%+ on miners/utils; no global variables; SQLite + Flask + Jinja; no React, no paid APIs.

---

## 1. Executive Summary

### What Phase 0–1 delivers

**Phase 0 — Security & documentation hygiene (no behavior change).**
- The leaked Google OAuth secret (`client_secret_543127885875-…apps.googleusercontent.com.json`, present at repo root) is audited against git history, rotated, and purged if ever committed.
- Stale/contradictory docs and notebooks are moved to `archive/`, leaving one authoritative `README.md` + `CLAUDE.md` that match the code.
- Phantom module references (`app.py`, `mining.py`, `enhanced_features.py`, `enhanced_models.py`) are removed from docs.
- README CI description is corrected (S3 + GitHub Releases, **not** "auto-commit to repository").
- `.env.example` gains the AWS fields the CI workflow actually needs.

**Phase 1 — Schema, index, minimal demo backend.**
- A new **`SteamMetric`** fact table persists the Steam enrichment currently fetched into `GameSnapshot` then discarded (`steam_data_miner.py:642-681`).
- `MIN_REVIEWS_FOR_SEED` is raised from `1` to **50** (rationale in §3.5).
- A single-source-of-truth **engagement index** module replaces the circular `target_score` *in the UI* — documented formula, batch-min-max normalized, named `engagement_score`, range 0–100.
- The circular `target_score` ML training path in `model_trainer.py` is **gated off** (skipped with a clear log message) until a real external label exists. No tautological model ships.
- The webapp gains `/sandbox` (offline manual-input prediction of the engagement index), `/samples` (list cached demo games), a cached-fallback layer for `/predict`, an always-visible explain panel, and Jinja templates replacing the inline `HTML_TEMPLATE`.
- 8–12 cached demo games ship as `data/demo_samples.json` so the presentation never depends on live API quota.
- Tests are added for the Steam miner, the engagement index, and the `/sandbox` route.

### What remains for Phase 2+

- Repeated time-series snapshots (currently every miner produces one observation/game/day).
- YouTube top-N videos (currently 1/game).
- Relaxing `YouTubeMetric.UNIQUE(video_id)` → `UNIQUE(video_id, mined_at)` for re-observation.
- **Computing** the review-velocity @ 30d label (needs ≥2 `SteamMetric` snapshots ≥30d apart). Phase 1 only *persists the columns that make it possible*; the formula is specified in §3.6 but **implemented in Phase 2**.
- Real ML training on the external label; baselines; SHAP (Phase 1 ships an interim plain-language driver, SHAP optional).
- Flow C comparison UI, charts, 90d windows, walk-forward CV.

---

## 2. Engagement Index Specification (replaces `target_score` in UI)

> This is the **descriptive** index (SRS §3.1(i), §17.2 ML-1). It is **UI-only**. It must **never** be used as ML supervision. It is computed deterministically from current-batch engagement features — that is exactly why it cannot be a learning target.

### 2.1 Single source of truth

Create **`src/features/engagement_index.py`** containing one public function plus its spec constant. Both the webapp and any offline scripts import from here. Do **not** duplicate the formula anywhere else (the current duplication between `TARGET_SCORE_SPEC` in `model_trainer.py:39-56` and ad-hoc UI logic is part of the drift being eliminated).

### 2.2 Exact formula

The index reuses the *intent* of `TARGET_SCORE_SPEC` but is re-grounded on features the SQL feature matrix actually emits today (`sql_feature_engineer.py` `FEATURE_COLUMNS`), so it is computable without the never-persisted Steam columns. Until `SteamMetric` is wired into the feature matrix (a Phase 2 join), the Steam-derived components are included **only if present**, with explicit presence handling — not silent zero-fill that biases the score.

**Components and weights (`ENGAGEMENT_INDEX_SPEC`):**

| Component (feature column) | Weight | Source phase |
|---|---|---|
| `youtube_total_views` | 0.15 | Phase 1 |
| `youtube_engagement_rate` | 0.12 | Phase 1 (derived) |
| `youtube_total_likes` | 0.05 | Phase 1 |
| `youtube_total_comments` | 0.05 | Phase 1 |
| `youtube_avg_sentiment` | 0.08 | Phase 1 |
| `reddit_total_score` | 0.12 | Phase 1 |
| `reddit_post_count` | 0.05 | Phase 1 |
| `reddit_total_comments` | 0.05 | Phase 1 |
| `reddit_avg_sentiment` | 0.08 | Phase 1 |
| `twitch_total_viewers` | 0.08 | Phase 1 |
| `twitch_stream_count` | 0.05 | Phase 1 |
| `cross_platform_engagement_rate` | 0.07 | Phase 2 derived |
| `platform_presence` | 0.05 | Phase 2 derived |

Weights sum to **1.00** exactly (no renormalization hack like the `1.05`-summing legacy spec at `model_trainer.py:152-164`). If any future edit changes the weights, the module must assert `abs(sum(weights) - 1.0) < 1e-9` at import time and fail loudly.

**Normalization:** per-component **min–max**, scope = **the current batch being scored** (the set of games passed to the function in one call), matching the legacy single-batch inference behavior (`model_trainer.py:149`). Formula per component *c* with value *v*:

```
norm_c(v) = (v - min_batch_c) / (max_batch_c - min_batch_c + EPSILON)   # EPSILON = 1e-8
index_raw = Σ_c  weight_c · norm_c(v)          # ∈ [0, 1]
engagement_score = round(100 · index_raw, 2)   # ∈ [0, 100]
```

- **Single-game batch (the sandbox & single-game /predict case):** with one row, every component's min == max, so `norm_c == 0` for all → index would be 0. Handle this explicitly: when the batch has `< MIN_BATCH_FOR_RELATIVE` rows (set `= 5`), normalize each component against the **cached demo-sample reference distribution** (`data/demo_samples.json`, §5) instead of the degenerate single-row batch. This is the "reference set" pattern and must be documented in the disclaimer. Store the reference min/max in the module as a loaded constant.
- **EPSILON** guards divide-by-zero, identical to `_EPSILON` at `model_trainer.py:35`.
- **Output range:** clamp to `[0.0, 100.0]`. Do **not** clamp to `[1, 100]` — 0 is a meaningful "no engagement" floor here, unlike the legacy score.

### 2.3 Forbidden overlaps (if ML is added later)

When/if the Phase 2 review-velocity ML label is introduced, the following **must not** appear as ML input features (they are components of, or proxies for, this descriptive index and would re-introduce circularity if the index were ever used to derive a label):

- All 13 `ENGAGEMENT_INDEX_SPEC` components above.
- `viral_potential_score`, `competitive_score`, `market_share`, `platform_dominance` (composite/batch-coupled — SRS §8.2).
- Any Steam **review-count** feature measured *after* the label's window-start date (temporal firewall — SRS §9.1).

Encode this as a module constant `FORBIDDEN_AS_ML_FEATURES: frozenset[str]` in `engagement_index.py` so the Phase 2 leakage test (SRS §13) can assert against it. **Phase 1 only declares the set; it is enforced in Phase 2.**

### 2.4 UI naming, label definition, disclaimer

- **UI name:** `engagement_score` (display label: **"Engagement Score"**, 0–100). Retire all "investment potential" / "target_score" user-facing strings (SRS §17.1).
- **One-sentence label definition (web app footer + explain panel):**
  > *"Engagement Score (0–100) is a transparent, weighted measure of a game's current cross-platform community activity and sentiment across YouTube, Reddit, and Twitch — relative to a reference set of games. It is descriptive, not a prediction, and not a sales or revenue estimate."*
- **One-paragraph disclaimer (web app footer):**
  > *"This tool measures community engagement and sentiment intelligence, not commercial success. The Engagement Score is a descriptive composite of publicly mined signals (YouTube, Reddit, Twitch, Steam), normalized against a small reference set of games for demonstration. It does not predict sales, revenue, wishlists, or return on investment, and must not be used as financial or investment advice. Samples per game are limited, data freshness varies, and scores are comparative within the displayed set. Built as a course project (NUS SWS3023 Web Mining)."*

### 2.5 Mapping from current `TARGET_SCORE_SPEC` (`model_trainer.py:39-56`)

| Legacy component | Action | Notes |
|---|---|---|
| `youtube_total_views` (0.15) | **Keep** | same weight |
| `youtube_like_ratio` (0.10) | **Drop** | recomputed inside legacy via likes/views; replaced by `youtube_engagement_rate` (a real feature column) |
| `youtube_total_likes` (0.05) | Keep | |
| `youtube_total_comments` (0.05) | Keep | |
| `youtube_engagement_rate` (0.10) | Keep → 0.12 | real Phase-1 derived column |
| `steam_metacritic` (0.10) | **Drop for v1** | not persisted; re-add in Phase 2 once `SteamMetric` joins the feature matrix |
| `steam_review_count` (0.05) | **Drop for v1** | same — Phase 2 |
| `steam_positive_rate` (0.05) | **Drop for v1** | same — Phase 2 |
| `steam_wishlist_count` (0.05) | **Drop for v1** | same — Phase 2 |
| `reddit_total_score` (0.10) | Keep → 0.12 | |
| `reddit_post_count` (0.05) | Keep | |
| `reddit_total_comments` (0.05) | Keep | |
| `twitch_total_viewers` (0.05) | Keep → 0.08 | |
| `twitch_stream_count` (0.03) | Keep → 0.05 | |
| `viral_potential_score` (0.02) | **Drop** | opaque composite; not user-interpretable |
| `cross_platform_engagement_rate` (0.05) | Keep → 0.07 | |
| — (new) `youtube_avg_sentiment` 0.08 | **Add** | sentiment is the product's differentiated value (SRS §2.4) |
| — (new) `reddit_avg_sentiment` 0.08 | **Add** | same |
| — (new) `platform_presence` 0.05 | **Add** | rewards multi-platform reach |

The Steam weight mass (0.25 legacy) is redistributed to sentiment + cross-platform + Twitch for v1, and reclaimed in Phase 2 when Steam features become available. Document this redistribution in the module docstring.

---

## 3. `SteamMetric` Schema & Migration

### 3.1 Column list (aligned to `GameSnapshot`, `steam_data_miner.py:131-150`)

`GameSnapshot` already fetches all of these; today only 3 reach the DB (`persist_game_snapshot:669-673`). Persist the rest in a fact table:

| Column | Type | Nullable | Source field on `GameSnapshot` |
|---|---|---|---|
| `id` | Integer PK autoincrement | no | — |
| `game_id` | Integer FK → `games.id` (ondelete CASCADE) | no | resolved from `appid` |
| `mined_at` | DateTime | no | batch timestamp (UTC, set at persist time) |
| `total_reviews` | Integer | yes | `total_reviews` |
| `positive_rate` | Float | yes | (add to snapshot — currently parsed into `ReviewSummaryData.positive_rate` but dropped before `GameSnapshot`; see §3.4) |
| `review_sentiment` | Float | yes | `sentiment_score` (VADER compound) |
| `current_players` | Integer | yes | `current_players` |
| `wishlist_count` | Integer | yes | `wishlist_count` |
| `supported_languages` | Text | yes | `supported_languages` |

> **Note:** `metacritic` and `price` are listed in SRS §6.1 as desirable but are **not** currently fetched by `GameSnapshot`. Adding them requires extending `parse_app_details_payload` — defer to Phase 2 to keep Phase 1 a pure persistence change. Document the deferral.

### 3.2 ORM model sketch (add to `src/database/models.py`)

Mirror the existing `*Metric` conventions (see `TwitchMetric`, `models.py:125-169`):

- `__tablename__ = "steam_metrics"`.
- Indexes (match the pattern at `models.py:133-137`): `ix_steam_metrics_game_id` on `game_id`; `ix_steam_metrics_game_id_mined_at` on `(game_id, mined_at)` — this composite index is what makes the Phase 2 review-velocity window query efficient.
- **No `UNIQUE` constraint on any natural key** — `SteamMetric` is intentionally a time series; repeated `(game_id, mined_at)` observations are the whole point (learn from the `YouTubeMetric.UNIQUE(video_id)` mistake at `models.py:182`).
- `mined_at` semantics: UTC timestamp at the moment the snapshot is persisted, matching `_utc_day_bounds` batching in `sql_feature_engineer.py:185-188` so a future Steam aggregate CTE can day-bucket consistently.
- Add a `steam_metrics` relationship on `Game` with `cascade="all, delete-orphan"` and `order_by="SteamMetric.mined_at"` (mirror `models.py:94-108`).
- Add `SteamMetric` to `__all__` (`models.py:333-342`).

### 3.3 Migration steps (manual `create_all` + backfill — Alembic not required for course scope)

1. Add the ORM class. `ensure_steam_schema()` in `steam_data_miner.py:627-629` already calls `Base.metadata.create_all(bind=engine)` — which creates **new** tables but **does not** alter existing ones. Since `steam_metrics` is brand-new, `create_all` is sufficient; no `ALTER TABLE` needed.
2. **Back up first:** copy `data/game_metrics.db` → `data/game_metrics.db.bak` before running anything (rollback path).
3. Run `python -c "from src.data_collection.steam_data_miner import ensure_schema; ensure_schema()"` to create the empty `steam_metrics` table in the existing DB.
4. **Optional backfill:** existing `Game` rows have no historical Steam metrics (they were discarded). A one-shot script `scripts/backfill_steam_metrics.py` may re-mine current Steam enrichment for already-seeded games and insert one `SteamMetric` row each. This is **optional for Phase 1** (the demo uses cached samples); if run, it must respect `SteamRateLimiter`. Mark it clearly as best-effort.
5. Verify: `SELECT COUNT(*) FROM steam_metrics;` returns ≥ 0 without error; `PRAGMA table_info(steam_metrics);` shows all columns.

### 3.4 Changes to `persist_game_snapshot` / miners

- **`GameSnapshot` gains `positive_rate: Optional[float]`** (currently `ReviewSummaryData.positive_rate` is computed at `steam_data_miner.py:291-294` then dropped). Thread it through `build_game_snapshot` (`:607-616`).
- **`persist_game_snapshot` (`:642-681`)**: after inserting/looking up the `Game` row, also insert one `SteamMetric` row from the snapshot's enrichment fields. Keep the existing `qualifies_for_seed` gate for the `Game` insert; the `SteamMetric` insert should occur for any qualifying game **and** for already-seeded games (so re-running the single-game miner from the webapp accumulates Steam time series — the Phase 2 enabler). Guard against inserting a duplicate `(game_id, mined_at)` only if the same batch re-runs within the same second (use the resolved `Game.id`).
- **`run_single_game` (`:822-851`)** and `persist_snapshots` (`:684-694`): no signature change; they call the updated `persist_game_snapshot`.
- Behavior must remain **idempotent per day** in spirit: re-running on the same day appends a new `SteamMetric` row (acceptable — it's a time series), but the `Game` dimension stays single-row per `appid` (unchanged).

### 3.5 `MIN_REVIEWS_FOR_SEED` — recommend **50**

Locked decision OPS-2 (SRS §17.4) is "raise to 50–100." **Recommend 50** for v1:
- At `=1` (current, `steam_data_miner.py:48`) the seed admits noise-tier games with no meaningful community signal, polluting the engagement-index reference distribution.
- `50` removes the long tail of essentially-unreviewed apps while **retaining indie titles** — the locked persona (SRS §17.1) is indie developers, and a `100`+ threshold would exclude legitimate small-studio launches that have only a few dozen reviews in their first weeks.
- `50` keeps the seed set large enough for a credible reference distribution and demo.
- **Open micro-decision** (§9): if the seeded set after re-mining is too sparse for a good reference distribution, fall back to `50` is already the conservative floor; raising toward `100` only if noise persists.

Change the constant and the `--min-reviews` CLI default (`:921`) together; update the docstring at `:148-150`.

### 3.6 Review-velocity @ 30d label — formula (IMPLEMENT IN PHASE 2)

Specified here for completeness; **not built in Phase 1** (needs ≥2 snapshots ≥30d apart, which the data does not yet have):

```
label(game) = (reviews_at[release + 30d] − reviews_at[release + 0d]) / 30
            = mean daily new reviews over the first 30 post-release days
```

- **Temporal firewall (SRS §9.1):** all ML *features* for `game` must be computed from observations with `mined_at < release_date + 0d` window-start; the label uses `SteamMetric.total_reviews` snapshots at the window endpoints. No feature may read review counts inside `[release, release+30d]`.
- **Availability gate:** computable only when `SteamMetric` has a snapshot within ±3 days of `release` **and** within ±3 days of `release+30d` for that `game_id`. Until then, the label is `NULL` and the game is excluded from ML training (post-release-only, SRS §17.2 ML-2).
- Phase 1 deliverable is **only** the persistence (§3.1–§3.4) that makes this query possible later. Do not write the label builder now.

---

## 4. Demo Web Application — Wireframes & Routes

> Stack locked to Flask + Jinja (SRS §17.3 UX-2). Move the inline `HTML_TEMPLATE` (`webapp.py:83-152`) into `src/api/templates/`. Sandbox-first presentation (UX-1).

### 4.1 Routes (extends `webapp.py`)

| Route | Method | Purpose | Network? | Error states |
|---|---|---|---|---|
| `/` | GET | Landing: persona one-liner, links to Analyze (live) and Sandbox | none | — |
| `/sandbox` | GET | Render manual-input form (grouped features, sample prefill dropdown) | none | — |
| `/sandbox` | POST | Compute `engagement_score` from posted feature values | **none** | 400 out-of-range / missing field |
| `/predict` | POST | Flow A: live mine → features → score; cached fallback on quota/404 | live (with cache fallback) | 400 invalid name; 404 unknown game → cached suggestion; 429 quota → cached-sample banner; 503 model/index missing |
| `/samples` | GET | List cached demo games (JSON or rendered table) | none | — |
| `/health` | GET | Liveness + index/artifact availability (exists, `webapp.py:500-506`) | none | 503 if dependencies absent |

> `/compare` (Flow C) is **deferred** (SRS §17.3 UX-3) — do not build.

### 4.2 Text wireframes

**`/` — Landing**
```
┌──────────────────────────────────────────────────────────────┐
│  Game Engagement Intelligence                                  │
│  Cross-platform community engagement & sentiment for a Steam   │
│  game — built for indie developers. Not a sales forecast.      │
│                                                                │
│   ┌───────────────────────┐   ┌───────────────────────────┐   │
│   │  ▶ Analyze a game     │   │  ▶ Try the Sandbox        │   │
│   │  (live mine + cache)  │   │  (offline, manual inputs) │   │
│   └───────────────────────┘   └───────────────────────────┘   │
│                                                                │
│  [footer] Engagement Score = descriptive composite, 0–100.     │
│  Not sales/ROI. Course project. ⓘ label definition · disclaimer│
└──────────────────────────────────────────────────────────────┘
```

**`/sandbox` (GET form)**
```
┌──────────────────────────────────────────────────────────────┐
│  Sandbox — score a hypothetical game (no network)              │
│  Prefill from sample: [ Counter-Strike 2 ▼ ]   [Load values]   │
│                                                                │
│  ── YouTube ──────────────────────────────────────────────    │
│   Total views        [  1200000 ]   Engagement rate [ 0.04 ]   │
│   Total likes        [    45000 ]   Total comments  [ 8000 ]   │
│   Avg sentiment      [     0.35 ]  (−1 … 1)                     │
│  ── Reddit ───────────────────────────────────────────────    │
│   Total score        [    25000 ]   Post count      [   15 ]   │
│   Total comments     [     4200 ]   Avg sentiment   [ 0.18 ]   │
│  ── Twitch ───────────────────────────────────────────────    │
│   Total viewers      [    90000 ]   Stream count    [   20 ]   │
│  ── Cross-platform ───────────────────────────────────────    │
│   Cross-platform engagement rate [ 0.05 ]                      │
│   Platforms present (0–3)        [ 3 ]                         │
│                                                                │
│                         [  Compute Engagement Score  ]         │
└──────────────────────────────────────────────────────────────┘
```

**`/sandbox` (POST result) & `/predict` result — shared Explain panel**
```
┌──────────────────────────────────────────────────────────────┐
│  Engagement Score:  72.4 / 100      [ tier: HIGH ]             │
│  ⓘ What this means: descriptive cross-platform engagement,     │
│     relative to a reference set. Not a sales prediction.       │
│                                                                │
│  Top drivers (plain language):                                 │
│   1. Strong YouTube viewership (1.2M views) ............ +     │
│   2. Healthy Reddit discussion (25k score, 15 posts) ... +     │
│   3. Positive YouTube sentiment (+0.35) ................ +     │
│                                                                │
│  Per-platform sentiment / emotion:                             │
│   YouTube  ▓▓▓▓▓▓▓░░  +0.35     Reddit  ▓▓▓▓▓░░░░  +0.18        │
│                                                                │
│  Data freshness:  mined 2026-06-07 14:02 UTC   [ LIVE ]        │
│        (or)       cached sample                [ CACHED ⚠ ]    │
│                                                                │
│  Limitations: small sample per game; comparative within set;   │
│  engagement intelligence, not investment advice.              │
└──────────────────────────────────────────────────────────────┘
```

**`/predict` cached-fallback banner (on 429/404)**
```
┌──────────────────────────────────────────────────────────────┐
│  ⚠ Live mining unavailable (quota/unknown game). Showing the   │
│    nearest cached sample: "Stardew Valley".                    │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 Presenter script (sandbox-first, ~5 min — updated from SRS §2.6 for locked decisions)

1. **(0:30) Frame honestly.** "We measure and explain cross-platform community engagement and sentiment for a Steam game — *not* sales or ROI." Point at the on-screen label definition + disclaimer.
2. **(1:30) Sandbox first (quota-proof).** Open `/sandbox`, prefill from a known hit, hit Compute → show Engagement Score + top drivers + per-platform sentiment. Tweak one field (drop YouTube sentiment) → score and driver list update. "The model is inspectable and the demo never depends on a live API."
3. **(1:00) Sandbox flop.** Load a low-engagement sample → score drops to LOW tier → the gap is explainable. (Replaces SRS §2.6's side-by-side, which is deferred Flow C.)
4. **(1:00) One live mine (if quota allows).** `/predict` with a known game → live score + `LIVE` freshness stamp. If quota fails, the cached-fallback banner appears — *demonstrate that this is by design.*
5. **(1:00) Honesty slide.** State limitations, why we don't claim R²=0.99, that review-velocity ML is a documented Phase-2 stretch once Steam time series exists, and what investment-grade labeling would require.

### 4.4 Explain panel (always visible — SRS §5.7)

Must render on every prediction (`/sandbox` and `/predict`):
1. **Label definition** (the one-sentence string from §2.4).
2. **Data freshness** — `mined_at` timestamp + a `LIVE` / `CACHED` badge.
3. **Top 3 plain-language drivers** — translate feature names to sentences (Phase 1 uses the index component contributions, i.e. `weight_c · norm_c(v)` ranked; SHAP is optional/Phase 2). A `FEATURE_LABELS: dict[str,str]` mapping in `engagement_index.py` provides the plain-language strings.
4. **Per-platform sentiment/emotion** — YouTube + Reddit `avg_sentiment` bars; Reddit emotion distribution if present (lazy-loaded, OPS-1).
5. **Limitations disclaimer** (the paragraph from §2.4).

### 4.5 Templates: create / remove

Create under `src/api/templates/`:
- `base.html` — shared layout, footer with label definition + disclaimer, freshness badge macro.
- `index.html` — landing (`/`).
- `sandbox.html` — the GET form (grouped inputs + sample prefill dropdown).
- `result.html` — shared result/explain panel (used by both `/sandbox` POST and `/predict`).
- (optional) `samples.html` — rendered `/samples` table.

Remove from `webapp.py`: the inline `HTML_TEMPLATE` string (`:83-152`) and the `render_template_string` import usage (`:23`, `:460`); switch to `render_template`. Set `app = Flask(__name__, template_folder="templates")` (the default already resolves to `src/api/templates/` since `webapp.py` lives in `src/api/`).

---

## 5. Cached Demo Dataset

### 5.1 Game list (10 post-release titles — hits, flops, edge cases)

| # | Game | Tier (expected) | Narrative for the demo |
|---|---|---|---|
| 1 | Counter-Strike 2 | HIGH | F2P juggernaut; huge Twitch + YouTube; the "ceiling" anchor |
| 2 | Baldur's Gate 3 | HIGH | Critical hit; strong Reddit discussion + positive sentiment |
| 3 | Stardew Valley | HIGH | Sleeper indie hit; durable community, modest Twitch |
| 4 | Dota 2 | HIGH | F2P with massive CCU but fewer "sales" — skews revenue proxies (why we don't forecast sales) |
| 5 | Hades | MEDIUM-HIGH | Acclaimed indie; balanced cross-platform, very positive sentiment |
| 6 | Vampire Survivors | MEDIUM | Cheap breakout; high engagement-per-dollar; strong Reddit |
| 7 | Hollow Knight | MEDIUM | Beloved indie; steady, not viral; good sentiment |
| 8 | A low-traffic indie title | LOW | Quiet launch; thin signal across platforms — the "floor" anchor |
| 9 | A known commercial flop | LOW | Negative/sparse sentiment despite some marketing buzz |
| 10 | An indie with no Twitch presence | LOW-MEDIUM | Platform-absence test: `platform_presence` < 3; explain how the score handles it |

> Pick the concrete titles for #8/#9/#10 from public post-mortems during implementation; keep the *shape* (one floor, one negative-sentiment, one platform-absent). These exercise the reference distribution and the presence handling.

### 5.2 File format — `data/demo_samples.json`

Each entry is a precomputed feature row (the 13 index components + identifiers + per-platform sentiment for the panel), so `/sandbox` prefill and the index reference distribution both read from one file:

```json
{
  "schema_version": 1,
  "generated_at": "2026-06-07T00:00:00Z",
  "reference_stats": {
    "youtube_total_views": {"min": 0, "max": 5000000},
    "reddit_total_score":  {"min": 0, "max": 80000}
    /* … per-component min/max used as the §2.2 reference distribution … */
  },
  "samples": [
    {
      "steam_name": "Stardew Valley",
      "appid": 413150,
      "tier_expected": "high",
      "narrative": "Sleeper indie hit; durable community.",
      "features": {
        "youtube_total_views": 1200000,
        "youtube_engagement_rate": 0.04,
        "youtube_total_likes": 45000,
        "youtube_total_comments": 8000,
        "youtube_avg_sentiment": 0.35,
        "reddit_total_score": 25000,
        "reddit_post_count": 15,
        "reddit_total_comments": 4200,
        "reddit_avg_sentiment": 0.18,
        "twitch_total_viewers": 90000,
        "twitch_stream_count": 20,
        "cross_platform_engagement_rate": 0.05,
        "platform_presence": 3
      },
      "precomputed_engagement_score": 71.0,
      "mined_at": "2026-06-01T00:00:00Z"
    }
    /* … 9 more … */
  ]
}
```

- `reference_stats` is the §2.2 reference distribution for single-game normalization.
- `precomputed_engagement_score` lets `/samples` and the cached-fallback render instantly without recompute, and serves as a golden value for tests.
- A small builder `scripts/build_demo_samples.py` may generate this from the DB if seeded; otherwise hand-author plausible values (clearly flagged as demo data).

### 5.3 How `/predict` selects cache vs. live

```
/predict(steam_name):
  resolve canonical name (Steam app-list fuzzy match, existing webapp.py:212-226)
  try:
     mine live → build feature row → engagement_index(row vs reference_stats)
     freshness = LIVE, mined_at = now
  except (QuotaExhaustedError 429, GameNotFoundError 404):
     pick nearest cached sample (exact name match → else first sample)
     load its precomputed score + features
     freshness = CACHED  (render banner)
  render result.html with explain panel
```

`/sandbox` never touches the network: it reads posted form values, validates ranges, calls `engagement_index(values vs reference_stats)`, renders the same `result.html`.

---

## 6. Phase 0 Task List (security + doc hygiene)

> Ordered. Each task: files touched · verification · rollback. **Do task 0.1 first** — it gates everything.

### 0.1 — Secret audit & remediation (P0, blocking)
- **Files touched:** repo root (`client_secret_543127885875-…json`), `.gitignore` (already has `client_secret_*.json` at line 3 — verify), git history.
- **Steps:**
  1. Confirm the file is gitignored going forward (it is — `.gitignore:3`).
  2. **Check whether it was ever committed:** `git log --all --full-history -- "client_secret_*.json"` and `git log --all --oneline -S "apps.googleusercontent.com"`.
  3. **If history shows it:** rotate/revoke the OAuth client in Google Cloud Console **immediately**, then purge with `git filter-repo --path-glob 'client_secret_*.json' --invert-paths` (or BFG), force-push, and notify any collaborators to re-clone.
  4. **If history is clean** (only working-tree, untracked): rotate the key anyway (it sat on disk), delete the file locally, keep the gitignore rule.
  5. Scan for other secrets: `git log --all -S "AKIA"` (AWS), `-S "client_secret"`, review `.env` is untracked.
- **Verification:** `git log --all --full-history -- "client_secret_*.json"` returns empty; `git ls-files | grep -i client_secret` returns empty; the OAuth credential is rotated in GCP.
- **Rollback:** none needed for rotation; if `filter-repo` is botched, restore from the pre-rewrite backup clone (make one before rewriting: `git clone --mirror . ../repo-backup.git`).

### 0.2 — Archive stale docs & notebooks
- **Files touched:** create `archive/`; move `SYSTEM_STATUS_REPORT.md`, `Integration_Guide.md`, `Enhanced_Analysis_README.md`, `README_FIX.md`, `NOTEBOOK_FIXES.md`, `API_USAGE.md` (review each — keep if still accurate); move notebooks `Enhanced_Game_Investment_Analysis.ipynb`, `leakage_free_model_check.ipynb`, `Game_Investment_Potential_*.ipynb`, `new_model_try.ipynb`; move root `youtube_data_miner.py` (legacy CSV miner — **not** the `src/data_collection/` one) and `clean_post_release_columns.py`.
- **Keep as evidence:** `new_model_try.ipynb` → `archive/notebooks/` with a header note "Pre-release→post-release model, Test R²≈0.03 — honest baseline, non-authoritative." Keep `FEATURE_MATHEMATICS_AUDIT.md` (referenced by the feature engineer).
- **Verification:** repo root has no notebooks except (optionally) a single current one; `archive/` contains the moved files; `pytest` still passes (nothing in `src/` imported them).
- **Rollback:** `git mv` back (low risk; pure moves).

### 0.3 — Align README.md to code
- **Files touched:** `README.md`.
- **Steps:** remove the phantom `app.py` reference (`README.md:13`, `:76` — only `webapp.py` exists); fix the CI description (`:46-54`) — it claims "Updated `…db` and `…pkl` are committed back to the repository"; the actual workflow pushes to **S3 + GitHub Releases** (SRS §11, `weekly_pipeline.yml:80-108`). Reword to match. Update the product framing to "Engagement Intelligence" per §2.4 (retire "Investment Potential" as the primary claim; a one-line note that the legacy name is deprecated is fine).
- **Verification:** grep README for `app.py`, "committed back", "investment potential" → only intentional/deprecation mentions remain.
- **Rollback:** revert the file.

### 0.4 — Align CLAUDE.md to code
- **Files touched:** `CLAUDE.md`.
- **Steps:** remove references to nonexistent `enhanced_features.py`, `enhanced_models.py`, `src/api/app.py`, `mining.py` (the "Module layout" + "Commands" sections). Update commands to drop `python src/api/app.py` and `run_test.bat` if stale. Add `SteamMetric` to the `src/database/` description (after Phase 1). Note the engagement-index module under `src/features/`.
- **Verification:** every file path in CLAUDE.md resolves to an existing file.
- **Rollback:** revert the file.

### 0.5 — `.env.example` AWS fields
- **Files touched:** `.env.example`.
- **Steps:** append the AWS CI vars the workflow needs (SRS §11): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET`, each with a placeholder + comment. Optionally add `DATABASE_URL` with the default value as a comment.
- **Verification:** `.env.example` lists all 9 existing + 4 AWS fields; no real values present.
- **Rollback:** revert the file.

**Phase 0 exit criteria:** secret rotated + history verified clean/purged; `pytest` green; every doc path resolves; README CI description matches the workflow; `.env.example` complete.

---

## 7. Phase 1 Task List (schema + index + minimal demo backend)

> Ordered with dependencies. Each task names exit criteria. **Do not** scope-creep into Phase 2 (no repeated-snapshot orchestration, no YouTube top-N, no review-velocity computation, no full ML retrain).

### 1.1 — `SteamMetric` ORM + miner persistence
- **Depends on:** Phase 0 complete (DB backed up).
- **Do:** add `SteamMetric` to `models.py` (§3.2); add `positive_rate` to `GameSnapshot` + thread through `build_game_snapshot` (§3.4); extend `persist_game_snapshot` to insert a `SteamMetric` row (§3.4); raise `MIN_REVIEWS_FOR_SEED` → 50 and CLI default (§3.5).
- **Exit criteria:** `ensure_schema()` creates `steam_metrics`; running the single-game Steam miner inserts one `Game` (if new) + one `SteamMetric` row; re-running appends a second `SteamMetric` (time series) without duplicating the `Game`; existing miner tests still pass.

### 1.2 — Engagement index module (single source of truth)
- **Depends on:** none (pure function).
- **Do:** create `src/features/engagement_index.py` with `ENGAGEMENT_INDEX_SPEC`, `FORBIDDEN_AS_ML_FEATURES`, `FEATURE_LABELS`, `LABEL_DEFINITION`, `DISCLAIMER`, and `compute_engagement_score(rows: pd.DataFrame | dict, reference_stats: Mapping | None) -> float | pd.Series` per §2.2. Import-time assert weights sum to 1.0.
- **Exit criteria:** unit tests (§7.5) pass; single-game input uses the reference distribution; known sample rows reproduce `precomputed_engagement_score` within tolerance.

### 1.3 — Gate the circular `target_score` training
- **Depends on:** 1.2 (so the UI has a replacement).
- **Do:** in `model_trainer.py`, make `train()` **skip** when no external label column is present. Concretely: add a guard at the top of `train()` — if `Game.target_variable_score` (or a configured external-label column) is entirely null across the frame, log a clear warning ("No external label available; ML training skipped — engagement index is descriptive-only. See Phase 2.") and return early **without** fitting the tautological composite. Keep `calculate_target_score` for backward-compatible artifact loading but stop calling it as supervision. Do **not** delete the file or the existing artifact (the webapp still loads it for the legacy `/predict` path until §1.4 routes the index in).
- **Exit criteria:** running `python src/models/model_trainer.py` on the current DB logs the skip message and exits 0 without producing a new circular artifact; existing artifact load path unaffected.

### 1.4 — Webapp: routes, templates, cache layer, explainability
- **Depends on:** 1.2 (index), 1.5-data (`data/demo_samples.json` can be hand-authored in parallel).
- **Do:** create the Jinja templates (§4.5); add `GET /` (landing), `GET/POST /sandbox`, `GET /samples`; rewrite `/predict` to compute the **engagement index** (not the legacy model score) with the cache-fallback logic (§5.3); add the always-visible explain panel (§4.4) driven by ranked index contributions (interim explainability; SHAP optional). Replace `render_template_string`/`HTML_TEMPLATE` with `render_template`. Reuse `validate_steam_name` (`webapp.py:178-194`); add range validation for sandbox inputs.
- **Exit criteria:** `/` renders; `/sandbox` GET shows the prefill form, POST returns a score + explain panel with **no network call**; `/predict` returns a live score on success and a cached sample with a banner on simulated 429/404; `/samples` lists the cached games; `/health` still works.

### 1.5 — Cached demo dataset
- **Depends on:** 1.2 (to compute/validate scores).
- **Do:** create `data/demo_samples.json` (§5.2) with 10 games (§5.1) and `reference_stats`; optionally `scripts/build_demo_samples.py`.
- **Exit criteria:** file validates against the schema; loading it powers `/samples`, sandbox prefill, and the index reference distribution; `precomputed_engagement_score` matches `compute_engagement_score` for each sample within tolerance.

### 1.6 — Tests
- **Depends on:** 1.1, 1.2, 1.4.
- **Do:** see §8. Add Steam-miner tests (none exist today), engagement-index tests, `/sandbox` route test.
- **Exit criteria:** new tests pass; coverage targets met (§8).

**Phase 1 exit criteria (overall):** `steam_metrics` populated by the miner; engagement index is the single UI scoring path; circular training is gated off; the 5-minute sandbox-first demo (§4.3) runs end-to-end **offline**; cached fallback works; new tests green.

---

## 8. Testing & Verification Checklist

> Mock all API responses; never hit live endpoints (CLAUDE.md). pytest only.

**Automated (run after Phase 0–1):**
- `pytest tests/` — all existing + new tests green.
- **New `tests/test_steam_data_miner.py`** (none exists today): mocked `appdetails` / `appreviews` / `GetNumberOfCurrentPlayers` / store-page HTML → assert `GameSnapshot` fields (incl. new `positive_rate`), `qualifies_for_seed` at threshold 50, `persist_game_snapshot` inserts `Game`+`SteamMetric`, `resolve_appid_by_name` fuzzy match, wishlist HTML parse (`parse_store_page_html`).
- **New `tests/test_engagement_index.py`:** weights sum to 1.0 (import-time); monotonicity (raising a positively-weighted component raises the score); single-game path uses reference stats; clamp to [0,100]; each `data/demo_samples.json` sample reproduces its `precomputed_engagement_score` within tolerance; `FORBIDDEN_AS_ML_FEATURES` contains all 13 components.
- **New `tests/test_webapp_sandbox.py`:** `POST /sandbox` with valid payload → 200 + score in [0,100] + explain fields present, **no network** (assert miners not called); out-of-range/missing field → 400; `GET /samples` → 200 lists 10 games; `/predict` with mocked 429 → 200 cached-sample + banner flag.
- **Schema check:** `PRAGMA table_info(steam_metrics)` shows all §3.1 columns; composite index present.
- **Coverage:** miners & utils ≥ 80%; engagement index ≥ 80%; API routes ≥ 70% (SRS §13).

**Manual demo dry-run (must pass before presenting):**
1. `python src/api/webapp.py` → open `/`.
2. `/sandbox` → prefill a hit sample → Compute → HIGH tier + 3 drivers + sentiment bars + disclaimer visible. **Disconnect network** and repeat — still works.
3. `/sandbox` → load the LOW sample → score drops, drivers change.
4. `/predict` with a known game → LIVE badge (or, with network off, CACHED banner appears — confirm by design).
5. `/samples` → 10 games listed with tiers.
6. Confirm footer shows label definition + disclaimer on every page.

---

## 9. Risks & Open Micro-Decisions (implementation-level only)

> No replays of Round 1. These are small, code-level choices left to the implementer.

| ID | Micro-decision | Default | Trigger to revisit |
|---|---|---|---|
| M-1 | `MIN_REVIEWS_FOR_SEED` exact value | **50** (§3.5) | If the re-seeded reference set is noisy/sparse, nudge toward 100 |
| M-2 | `MIN_BATCH_FOR_RELATIVE` (when to use reference stats vs in-batch min/max) | **5** rows | If multi-game `/predict` batches are common, lower it |
| M-3 | Interim explainability vs SHAP in Phase 1 | **Interim** (ranked index contributions) | Add SHAP only if a reviewer demands per-prediction attribution and time allows (OPS, Phase 2) |
| M-4 | Concrete titles for demo slots #8/#9/#10 | Pick from public post-mortems | Must keep: one floor, one negative-sentiment, one no-Twitch |
| M-5 | `scripts/backfill_steam_metrics.py` — run it now? | **Optional** (demo uses cache) | Run if a live `/predict` demo against seeded games is wanted |
| M-6 | `archive/` vs `notebooks/exploratory/` location | **`archive/`** | Team preference; cosmetic |
| M-7 | Emotion model in explain panel | **Lazy-load, optional** (OPS-1) | Disable if CI/demo latency hurts |

**Risks specific to Phase 0–1:**
- **History rewrite (0.1)** is destructive if the secret was committed — make a mirror backup first and coordinate re-clones. If history is clean, avoid the rewrite entirely (rotate + delete only).
- **`create_all` won't migrate existing tables** — fine for the brand-new `steam_metrics`, but if any *existing* table needs a column later (Phase 2 `YouTubeMetric` uniqueness), that needs a real migration. Out of scope here; noted so it isn't assumed solved.
- **Single-game normalization** is the subtlest correctness point — if the reference-stats path is wrong, every sandbox score is 0 or 100. The §8 monotonicity + reference-stats tests guard it.

---

## Cursor Agent kickoff prompt

> Implement Phases 0 and 1 from `docs/plans/PIPELINE_REBUILD_PHASE0-1.md` exactly, treating SRS §17 as locked (do not reopen positioning, persona, labels, demo modes, stack, or scope). Start with Phase 0 task 0.1: run `git log --all --full-history -- "client_secret_*.json"` and `git log --all -S "apps.googleusercontent.com"`; if the secret is in history, stop and tell me to rotate the GCP OAuth credential before you purge it with `git filter-repo` (make a mirror backup first) — if history is clean, just confirm `.gitignore:3` covers it and have me rotate+delete locally. Then do 0.2–0.5 (archive stale docs/notebooks to `archive/`, keep `new_model_try.ipynb` as labeled evidence, align README/CLAUDE to code, fix the README CI description from "committed back to the repository" to "S3 + GitHub Releases", add the 4 AWS fields to `.env.example`). For Phase 1: add the `SteamMetric` ORM model to `src/database/models.py` (time-series, no UNIQUE, composite `(game_id, mined_at)` index, `Game.steam_metrics` relationship, add to `__all__`); add `positive_rate` to `GameSnapshot` and persist a `SteamMetric` row in `persist_game_snapshot`; raise `MIN_REVIEWS_FOR_SEED` to 50 (constant + CLI default + docstring); create `src/features/engagement_index.py` as the single source of truth (the §2.2 formula, weights summing to exactly 1.0 with an import-time assert, batch min–max with a reference-stats fallback for single-game inputs, `FEATURE_LABELS`, `LABEL_DEFINITION`, `DISCLAIMER`, `FORBIDDEN_AS_ML_FEATURES`); gate `model_trainer.train()` to skip and log when no external label exists (do not delete the file or existing artifact); create `data/demo_samples.json` with the 10 games and `reference_stats` from §5; refactor `src/api/webapp.py` to Jinja templates under `src/api/templates/` (remove the inline `HTML_TEMPLATE`), add `GET /`, `GET/POST /sandbox` (offline, no network), `GET /samples`, and a cache-fallback `/predict` that scores via the engagement index, all rendering the shared explain panel (label definition, freshness LIVE/CACHED badge, top-3 plain-language drivers, per-platform sentiment, limitations); back up `data/game_metrics.db` before any schema change. Add the three new test files in §8 with mocked APIs (no live endpoints) and meet the coverage targets. Run `pytest tests/` and the §8 manual demo dry-run; report results honestly — if anything fails, leave it in-progress and tell me rather than marking it done.
