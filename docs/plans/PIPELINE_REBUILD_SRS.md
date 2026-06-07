# Software Requirements Specification & Technical Blueprint
## Game Investment Potential Pipeline — Rebuild (Round 1 Draft)

> **Status:** Round 1 **decisions locked** (see §17). Round 2 deepens schema, wireframes, and engagement-index spec. Implementation follows Round 2 approval.
> **Audience:** Project owner (technical) + course stakeholders.
> **Scope:** Web Mining course project (NUS SWS3023, CRISP-DM), with a pathway toward a credible product.
> **Convention:** `[DECISION REQUIRED]` marks choices the owner must resolve before coding. Findings cite real files/line numbers from the current repo.

---

## 1. Executive Summary

The repository contains a working multi-platform mining + ML pipeline (Steam, Twitch, YouTube, Reddit → SQLite warehouse → SQL features → `VotingRegressor` → Flask demo), but it has **three foundational problems that invalidate its core claim**:

1. **The supervised target is circular.** `target_score` (1–100) is a deterministic weighted sum of 16 engagement components, of which **11+ are also model input features** (`src/models/model_trainer.py:39-56`, overlapping `src/features/sql_feature_engineer.py:39-140`). The model is trained to predict a function of its own inputs. The reported R² ≈ 0.96–0.998 is a mathematical inevitability, not evidence of predictive value.

2. **There is no ground-truth "investment" signal.** The ORM's `Game.target_variable_score` column is documented as an external label (`src/database/models.py:79`) but is **never populated from any external source** — no sales, ROI, revenue, or retention data is collected or persisted. The only honest attempt at a real label (`new_model_try.ipynb`) predicts a post-release `success_score` from pre-release features and achieves **Test R² ≈ 0.0016** — effectively zero. This is the true difficulty of the stated problem.

3. **The product was never validated.** No personas, no competitive analysis, no defined job-to-be-done. A mature commercial landscape already exists (Gamalytic, VG Insights, SteamDB, SteamSpy) that estimates exactly the financial outcomes this project gestures at.

Compounding these: heavy documentation drift (files like `enhanced_features.py`, `enhanced_models.py`, `src/api/app.py`, `mining.py` are referenced across docs but **do not exist**), thin data sampling (1 YouTube video/game, ≤15 Reddit posts/game, single Twitch snapshot), and a **live Google OAuth client secret committed at the repo root** (`client_secret_*.apps.googleusercontent.com.json`) — an immediate P0 security exposure.

**Proposed north star.** Rebuild around honesty and demonstrability, not a fictional accuracy number:

- **Product:** Narrow to a single, validated primary persona and a *defensible wedge* the incumbents don't own (engagement-and-sentiment intelligence, not sales estimation — that race is lost to Gamalytic/VGI).
- **ML:** Define a *real, external* ground-truth label; separate it cleanly from features; baseline honestly; report calibrated, uncertainty-aware metrics; accept that R² may be modest and *that is the scientific result*.
- **Demo:** A credible web app with (A) game-name → mine → predict → **explain**, and (B) a **manual-input sandbox** so the presentation never depends on live API quota. Transparency (label definition, data freshness, limitations) is the trust mechanism.
- **Architecture:** One source of truth (the `src/` ORM pipeline); deprecate notebooks/CSV/stale docs; align docs to code.

This SRS treats **product discovery and label validity as gating deliverables** — the technical rebuild is wasted effort if it serves no validated user or predicts a tautology.

---

## 2. Product & Market Discovery

> The prompt is explicit: do **not** assume "investors want a 1–100 score." This section treats discovery as a first-class deliverable. Competitive claims below come from light desk research (sources in §16).

### 2.1 Problem statement from the user's perspective

The original brief (`Project/Web Mining Idea.txt`) frames the user as *"the Product Manager of a game investment company"* who wants to *"verify whether a game has the potential to succeed and [is] worthy of investing,"* defining success *"through its sales and cost."* That is a **financial forecasting** problem. The current pipeline does **not** answer it — it measures *current cross-platform engagement* and relabels that as "investment potential."

The honest user problem we can actually serve with web-mined data is narrower:

> *"For a given game, how strong, healthy, and authentic is its multi-platform community engagement and sentiment right now — and how does that compare to peers?"*

This is an **engagement-intelligence** problem, not a sales-forecasting one. Framing v1 honestly around it is the single most important product decision (see §2.7).

### 2.2 Candidate personas

| Persona | Pain | Job they'd hire the tool for | Willingness to trust ML output |
|---|---|---|---|
| **Game investor / publisher (scout)** | Too many titles, too little time to triage | "Is this title worth a meeting?" — fast triage signal | **Low** without financials. They already buy Gamalytic/VGI for revenue estimates; an engagement-only score won't earn a funding decision. |
| **Indie developer / small studio** | No budget for paid analytics; needs to know if community traction is real | "Is my game's buzz genuine and growing vs. peers? Where is sentiment weak?" | **Medium.** They value cheap, explainable signal and sentiment breakdowns more than a single number. |
| **Marketing / UA team** | Where to spend limited pre/post-launch budget | "Which platform is underperforming for my title? Is sentiment souring?" | **Medium.** Cares about per-platform breakdown + trend, less about a composite. |
| **Course / demo audience** (professor, classmates, portfolio reviewer) | Needs to see scientific rigor + working web app | "Did this team mine real data, define a defensible label, and ship a working, explainable demo?" | **High** — *if* the team is transparent about limitations. Overclaiming R²=0.99 *destroys* trust here. |

### 2.3 Jobs-to-be-done

- **JTBD-1 (triage):** *"When I'm scanning many titles, help me rank which deserve a closer look, so I don't waste time."*
- **JTBD-2 (self-assessment):** *"When I'm a builder, tell me whether my community engagement is real and where it's weak, so I can act before launch."*
- **JTBD-3 (explain & defend):** *"When I present a verdict, show me *why*, with data freshness and caveats, so I can trust and defend it."*

JTBD-3 cuts across all personas and is the demo's centerpiece.

### 2.4 Competitive / adjacent landscape (desk research)

| Tool | What it does | Implication for us |
|---|---|---|
| **Gamalytic** | 50,000+ Steam titles; estimated revenue/sales/players via top-seller-rank + Boxleiter + regression; 77% of estimates within 30% error, 98% within 50%. | Owns *sales estimation*. Do **not** rebuild this. |
| **VG Insights (VGI)** | 100,000–150,000 games, 50+ datapoints; sales estimates, wishlists, genre supply/demand, competitor analysis; dedicated Unit Sales Estimator. | Owns *market/genre analytics*. Don't compete head-on. |
| **SteamDB** | Daily CCU, followers, wishlist charts, hot releases. | Free, authoritative raw Steam signals. Potential *data reference*, not a competitor to displace. |
| **SteamSpy** | Owner/popularity estimates (degraded since Steam privacy change). | Legacy; methodology now weak. |
| **IMPRESS / GameDiscoverCo** | Wishlist→sales calculators; benchmark conversion (~5% day-1, ~20% week-1, ~60% year-1). | Useful *label-design reference* if we ever use wishlist conversion. |
| **Wayline (Signals/Forecast)** | Revenue/profit forecasting from wishlists + price. | Adjacent forecasting; not engagement/sentiment. |

**Gap / wedge.** Incumbents are strong on *financial estimation* and weak on **multi-platform community engagement + sentiment/emotion intelligence with explainability**. None combine Twitch + YouTube + Reddit sentiment/emotion into a single comparative, explainable view. That is our defensible niche — and it aligns with what the pipeline *actually* collects (VADER sentiment, HuggingFace emotion distribution in `reddit_data_miner.py`).

> **Caveat:** This is light desk research and must be validated with one instructor/stakeholder conversation and a 30-minute competitor walkthrough before scope is locked.

### 2.5 Positioning options

- **A — Engagement-intelligence wedge (recommended):** "Cross-platform community health & sentiment intelligence for a game." Honest given the data; differentiates from incumbents; the existing miners already support it.
- **B — Sales/investment forecaster:** Head-on with Gamalytic/VGI. Requires sales/financial ground truth we cannot collect for free. Not viable for a course project.
- **C — Builder's pre-launch readiness check:** Niche of A focused on indie devs and pre-release wishlist/sentiment momentum.

### 2.6 Demo narrative (5-minute walkthrough)

1. **(0:30) Frame the honest problem** — "We don't predict sales; incumbents do that. We measure cross-platform community engagement & sentiment, explainably." Show the label definition on screen.
2. **(1:30) Flow A — live:** Enter a known hit (e.g., *Counter-Strike 2*). Mine (or use cache), show prediction + **top contributing features** + per-platform sentiment/emotion + data-freshness stamp.
3. **(1:00) Flow B — sandbox:** Manually enter feature values for a hypothetical title; show the prediction updates and the explanation — proves the model is inspectable and the demo is quota-proof.
4. **(1:00) Comparison:** Show a hit vs. a flop side by side; the engagement gap is visible and explainable.
5. **(1:00) Honesty slide:** State limitations, label definition, why we *don't* claim R²=0.99, and what real investment-grade labeling would require.

### 2.7 `[DECISION REQUIRED]` — Product

- **P-1 Primary persona for v1.** Options: investor-scout / indie-developer / course-demo-first. **Recommendation:** course-demo-first *positioned as the indie-developer engagement-intelligence wedge* (honest, achievable, differentiated). **Impact if wrong:** building features for a user who won't trust or use the output.
- **P-2 Positioning.** A (engagement wedge) / B (sales forecaster) / C (builder readiness). **Recommendation:** A. **Impact if wrong:** competing with Gamalytic/VGI on a problem our free data cannot solve.

---

## 3. Problem Statement & Success Criteria (ML + Product)

### 3.1 Operational definition of "investment potential"

The phrase must be **operationalized or abandoned**. Three honest framings:

- **(i) Descriptive engagement index (no ML):** A transparent, documented composite of *current* engagement — exactly what `target_score` is today, but *re-labeled as a descriptive index, not a prediction.* Useful, honest, not ML.
- **(ii) Predicted future engagement/community health:** Predict a *future* observable (e.g., engagement 30/90 days post-release) from *earlier* signals. Requires time-series collection the pipeline lacks today.
- **(iii) Predicted commercial success:** Requires external financial ground truth (sales/revenue/wishlist conversion). Not free; incumbents already do it.

### 3.2 Candidate ground-truth labels (replace the circular target)

| Candidate label | Definition | Data availability | Verdict |
|---|---|---|---|
| **Review velocity @ 30/90d** | Δ Steam review count over fixed window post-release | Steam `appreviews` API (free, already partly fetched) — **needs time-series persistence** | **Strongest free option.** Real, external, leak-resistant if features predate the window. |
| **Player retention proxy** | CCU at day 30 ÷ CCU at launch | `GetNumberOfCurrentPlayers` (free, already fetched, **not persisted**) — needs longitudinal capture | Viable but noisy; needs ~30–90 day capture window. |
| **Wishlist→sales conversion** | Sales ÷ pre-launch wishlists | Wishlists scraped (already done, fragile); **sales not available for free** | Not viable without paid sales data. |
| **Revenue proxy (Boxleiter)** | `reviews × multiplier × price` | Reviews + price are free | Possible but replicates incumbents and is crude; use only as a *weak* comparison label. |
| **Curated hit/flop binary** | Hand-labeled hits vs. flops from public lists | Manual, small-N | Useful for a **classification** sanity baseline and demo comparison. |

**Recommendation:** Adopt **review velocity @ 30/90d** as the primary candidate label, with a **strict temporal firewall** (features must be measured *before* the label window). Keep the old composite only as a *clearly labeled descriptive index* in the UI — never as supervision. See `[DECISION REQUIRED] ML-1`.

### 3.3 Business / model KPIs (not just R²)

- **Ranking quality:** Spearman/Kendall correlation, NDCG@k for triage ranking (more decision-relevant than R²).
- **Calibration:** reliability of uncertainty intervals (coverage of prediction intervals).
- **Baseline lift:** must beat a *naive engagement index* and a *mean predictor* on a temporal/grouped split — the bar is "better than trivial," stated honestly.
- **Honesty gate:** any reported metric must name its label, its split, and whether features could leak.

### 3.4 Product KPIs (demo success)

- Task completion: a reviewer can get prediction + explanation for a game in **< 60s**.
- Comprehension: a non-ML viewer can state *what the score means* and *one reason for it* after the demo.
- Trust signals present: visible label definition, data-freshness timestamp, per-platform breakdown, explicit limitations disclaimer.
- Robustness: demo works **offline** via cached sample games (no live-quota dependency).

---

## 4. Stakeholders & Constraints

- **Course scope vs. production:** This is a graded course project (CRISP-DM). Prioritize *scientific defensibility + a working, honest demo* over enterprise features.
- **Legal / ToS per platform** (detail in §7): YouTube & Reddit & Twitch have official APIs with explicit ToS that **discourage scraping**; Steam has a public Web API plus a store front-end whose HTML scraping is a gray area (already done for wishlists at `steam_data_miner.py:342-376`). ToS posture is a `[DECISION REQUIRED]`.
- **Budget:** Free API tiers only. No paid sales data (rules out direct Gamalytic/VGI-style labels). YouTube hard cap: 10,000 quota units/day.
- **Infra:** Anaconda Python 3.12; SQLite default (`sqlite:///data/game_metrics.db`); GitHub Actions weekly cron; optional AWS S3 + GitHub Releases for artifacts (already wired in `.github/workflows/weekly_pipeline.yml`). Postgres only if multi-writer concurrency becomes real (it is not, for a course demo) — **stay on SQLite**.
- **Security (P0):** A live Google OAuth client secret is committed at repo root. Must be rotated and purged from git history before any further work (see §11, §14).

---

## 5. Demo Web Application Specification

### 5.1 Current-state gap analysis (`src/api/webapp.py`)

What exists today (verified):
- One Flask app, one form: enter a Steam name → `POST /predict` → mines **live** Steam+Twitch+YouTube+Reddit → builds SQL features → returns `target_score` (1–100) + `top_features` (`webapp.py:402-447`). A `/health` probe exists (`:500`).
- The HTML is an inline string template (`HTML_TEMPLATE`, `:83-152`).

Gaps vs. desired demo:
- **No manual-input / sandbox mode** — the demo *requires* live API success, so quota exhaustion or an unknown game breaks the presentation.
- **No persona views, no comparison, no charts, no explanation UI** beyond a raw JSON dump of feature contributions.
- **No offline/cached fallback** — a missing game or exhausted quota yields a 404/429 with nothing to show.
- **Explainability is opaque:** `top_contributing_features` (`:350-390`) reports tree-importance × scaled magnitude — not human-readable, and no label-definition or freshness disclosure.
- Docs reference a second simple `app.py` predictor that **does not exist** — remove the reference.

### 5.2 User flows

**Flow A — Game-name → mine/cache → predict → explain**
```
[Enter game name] → resolve canonical Steam title
   → cache hit?  ── yes → load cached features
                └─ no  → mine live (Steam/Twitch/YouTube/Reddit)
                            └─ quota/404? → fall back to cached sample OR clear error state
   → build features → predict → render: score + per-platform breakdown
       + top human-readable drivers + sentiment/emotion + freshness stamp + caveats
```

**Flow B — Manual feature sandbox → predict → explain**
```
[Form of model-input features, grouped & pre-filled with a sample game's values]
   → validate ranges → predict → same explanation panel as Flow A
   (No network calls — guarantees the demo always works.)
```

**Flow C (optional) — Persona/comparison view**
```
[Pick 2 games] → side-by-side engagement/sentiment + which platform drives the gap
(Only build if persona work in §2 justifies it.)
```

### 5.3 Pages, endpoints, validation, error states

| Route | Method | Purpose | Error states |
|---|---|---|---|
| `/` | GET | Landing + flow selector | — |
| `/predict` | POST | Flow A (live/cache) | 400 invalid name; 404 unknown game; 429 quota → **auto-fallback to cached sample with a banner**; 503 model missing |
| `/sandbox` | POST | Flow B (manual features) | 400 out-of-range/missing feature |
| `/compare` | POST | Flow C (optional) | 404 if either game unavailable |
| `/health` | GET | Liveness + artifact check | 503 if artifact absent (exists today) |
| `/samples` | GET | List cached demo games | — |

Input validation: reuse/extend `validate_steam_name` (`webapp.py:178-194`); sandbox inputs validated against documented feature ranges.

### 5.4 Offline / demo fallback

- Ship a **cached sample set** (precomputed feature rows + predictions) for ~8–12 games covering hits, flops, and edge cases (§16.4).
- On quota/404, Flow A degrades to the nearest cached sample with a visible "showing cached data" banner.
- Flow B never touches the network.

### 5.5 MVP vs. stretch

- **MVP:** Flows A + B; per-platform breakdown; human-readable top drivers; freshness + label-definition + limitations panel; cached fallback.
- **Stretch:** Flow C comparison; charts (sentiment/emotion bars, engagement spider); PDF export; persona-tailored copy.

### 5.6 Tech stack recommendation

**Extend Flask + server-rendered templates (Jinja) + a light JS sprinkle** (recommended). Rationale: the team's competency and existing code are Flask; a course demo does not justify an SPA's build complexity; server-side rendering keeps the explanation panel simple and reviewable. Move the inline `HTML_TEMPLATE` into proper `templates/`. Charts via a CDN lib (Chart.js) only if stretch UI is approved. **Avoid React/SPA** unless a reviewer requirement forces it. → `[DECISION REQUIRED] UX-2`.

### 5.7 Explainability UI requirements

The UI must always display: (1) **what the score means** (the label definition, in one sentence), (2) **data freshness** (mined-at timestamp), (3) **top human-readable drivers** (translate feature names to plain language), (4) **per-platform sentiment/emotion**, (5) a **limitations disclaimer** ("engagement intelligence, not a sales forecast; small sample per game; not investment advice").

---

## 6. Data Requirements

### 6.1 Entity-relationship (target state)

Current schema (`src/database/models.py`): `Game (1)─(N) {TwitchMetric, YouTubeMetric, RedditMetric}`. `Game` stores only `appid, steam_name, release_date, target_variable_score, timestamps`. **No Steam metric table exists** — Steam enrichment (reviews, sentiment, players, wishlist) is fetched into `GameSnapshot` (`steam_data_miner.py:131-150`) and then **discarded** (`persist_game_snapshot:642-681` writes only 3 fields).

Proposed additions:
- **`SteamMetric`** (new fact table): persist `mined_at, total_reviews, positive_rate, review_sentiment (VADER), current_players, wishlist_count, metacritic?, price, supported_languages`. This is the single highest-value schema fix — it unblocks real Steam-based labels.
- **Time-series semantics:** all `*Metric` tables already carry `mined_at`; enforce *repeated* observations per game over time (not one snapshot) so windows (7/30/90d) and the review-velocity label become computable. `YouTubeMetric` currently has a `UNIQUE(video_id)` constraint (`models.py:182`) that **prevents re-observing the same video over time** — must change to `UNIQUE(video_id, mined_at)` or a batch key.
- **`Label` table (or columns):** store the chosen external label with its computation window and as-of date, separate from features.

### 6.2 Per-platform field catalog (collect / granularity / cadence)

| Platform | Collect (target) | Granularity | Refresh cadence |
|---|---|---|---|
| **Steam** | reviews, positive rate, players, wishlist, price, release date | per game, repeated | weekly (CI) + on-demand (demo) |
| **YouTube** | top-N videos (not 1), views/likes/comments, comment sentiment | per video, repeated | weekly + on-demand |
| **Twitch** | live streams snapshot (≤20), viewers, streamers | per snapshot, repeated | **multiple/day** ideally (live data) |
| **Reddit** | posts across subreddits, score/comments, sentiment, emotion | per post, repeated | weekly + on-demand |

### 6.3 Volume targets

- **Games:** ≥ 300–500 with valid release dates and ≥ 2 platforms present (current `MIN_REVIEWS_FOR_SEED = 1` at `steam_data_miner.py:48` is too permissive — raise it).
- **Observations/game:** YouTube ≥ 5 videos (today: 1); Reddit keep ≤15; Twitch ≥ 1 snapshot/day over the window.
- **History depth:** ≥ 30–90 days of repeated observations to support windowed labels — this is the **single biggest data gap** today.

### 6.4 Data-quality rules & monitoring

- Reject games with no `release_date` for any temporal-split or windowed-label use.
- Null/zero policy is currently `coalesce → 0` (`sql_feature_engineer.py:217`); distinguish *"absent platform"* from *"zero engagement"* with a presence flag (`platform_presence` exists at `:565` — extend per-platform).
- Track per-batch row counts and miner success/failure (extend `mining_status.json`).
- Drift checks on feature distributions (`src/api/model_monitoring.py` has unused `BacktestValidator` infra at `:406-467` — wire it in).

---

## 7. Collection Strategy — API vs. Scraping Decision Matrix

| Platform | Recommended primary | Fallback | Rationale | Throughput (free tier) | Risk |
|---|---|---|---|---|---|
| **Steam** | Web API (`appdetails`, `appreviews`, `GetNumberOfCurrentPlayers`) | Store-page HTML for wishlist only (already done, `:342-376`) | Official API covers most needs; wishlist isn't in the API | 55 req/min (`SteamRateLimiter`, buffer 5) | Store HTML fragile (selector `#WishlistBtn` can break); gray-area ToS |
| **Twitch** | Helix API | none | Generous limits, official, already authenticated | 800 req/min (`TwitchRateLimiter`) + header-aware backoff | Live-only; no history unless we snapshot repeatedly |
| **YouTube** | Data API v3 | Return-YouTube-Dislike API for dislikes (already used, `:47`) | Official; quota is the binding constraint | **10,000 units/day** — search=100 units each; this caps games/day hard | Quota exhaustion mid-demo → sandbox fallback essential |
| **Reddit** | PRAW (official API) | none | Official, sufficient | 1.1s min delay/req (`RedditRateLimiter`) | Emotion model (HuggingFace) is slow/heavy (`torch`+`transformers`) |

**Scraping verdict:** Beyond the existing Steam wishlist scrape, **do not add Selenium-based scraping.** `selenium` is in `requirements.txt` (line 1) but **unused in `src/`** (verified). Official APIs cover the needs; scraping adds ToS risk, fragility, and maintenance for marginal volume. The quota problem is better solved by **caching + repeated lightweight snapshots + the sandbox demo mode** than by scraping. → `[DECISION REQUIRED] DATA-1` (per-platform posture).

**Rate-limit math (YouTube example):** 1 search (100 units) + 1 videos.list (1 unit) + comment threads (~1–3 units) ≈ ~105 units/game → **~95 games/day** ceiling at 1 video/game; raising to 5 videos/game multiplies search/detail cost — must budget explicitly and cache aggressively.

---

## 8. Feature Engineering Specification

### 8.1 Feature catalog (grouped)

The current pipeline emits ~77 features in 3 phases (`sql_feature_engineer.py`): Phase 1 raw aggregates (#1–49), Phase 2 derived (viral/cross-platform/temporal/quality/sentiment/network, #50–95 minus dead), Phase 3 batch-scoped competitive (#55, #57, #58).

Target grouping:
- **Pre-release signals:** wishlist trajectory, pre-launch YouTube/Reddit volume, sentiment before release. *(Currently weak — needs time-series.)*
- **Post-release signals:** review velocity, player counts, sustained engagement.
- **Cross-platform:** synergy, balance, reach (#59–62).
- **Temporal:** growth momentum, consistency, trend (#63–66) — *currently computed within a single day, so "temporal" is a misnomer until multi-day history exists.*

### 8.2 Leakage policy (per `FEATURE_MATHEMATICS_AUDIT.md`)

- **Cross-game leakage:** `market_share` (#55), `platform_dominance` (#57), `competitive_score` (#58) use sums across *all games in the batch* — game *i*'s value depends on every other game (`FEATURE_MATHEMATICS_AUDIT.md:112,207-210`). The SQL version scopes them to a batch window (`sql_feature_engineer.py:130-134`) but they still couple games. **Policy:** exclude from supervised features or compute strictly from a *frozen historical reference set*, never the prediction batch.
- **Target leakage (the big one):** any feature that also appears in the label must be removed from inputs once a real label is chosen (§9.1).
- **Temporal leakage:** features must be measured strictly *before* the label window.

### 8.3 Keep / redesign / drop

- **Drop (dead, always-zero):** #69–71, #73, #85–94 — already omitted by the SQL engineer (`:5`); formally delete from any spec.
- **Redesign:** "temporal" features (#63–66) to use real multi-day history; competitive features (#55/57/58) to a frozen reference baseline.
- **Keep:** per-platform aggregates and sentiment/emotion features — these are the product's differentiated value.

### 8.4 Map features → web-app inputs

- **User-enterable in sandbox (Flow B):** the interpretable Phase-1 aggregates (e.g., `youtube_total_views`, `reddit_total_score`, `twitch_avg_viewers`, sentiment ratios). Provide sensible ranges + a sample prefill.
- **Mined-only (not user-entered):** derived/composite features (viral, synergy, batch-scoped) — computed by the pipeline, shown read-only in the explanation.

### 8.5 Time-window semantics

Define 7d/30d/90d windows **relative to release date**, computed from repeated `mined_at` observations. This is *aspirational* until §6.3 history depth exists; until then, windows collapse to single-snapshot and must be labeled as such in the UI.

---

## 9. Modeling Specification

### 9.1 Label ↔ feature separation (eliminate the circular target)

**Mandatory:** the supervised label must be **external to the feature set**. With review-velocity@30/90d (§3.2), enforce a temporal firewall: features use data with `mined_at < release + window_start`; the label uses review counts at `release + window_end`. No feature may be a component of the label. The current `target_score` is retired as supervision and survives only as a UI descriptive index.

### 9.2 Baselines (report honestly, first)

- **Naive mean predictor** (floor).
- **Naive engagement index** (today's composite) — the bar a real model must beat.
- **Single linear model** (Ridge/ElasticNet) on leak-free features.
- **Single tree model** (LightGBM) on leak-free features.

### 9.3 Candidate models & selection

Keep the option of the `VotingRegressor` (LGBM+XGB+RF, `model_trainer.py:270-308`) **but only if it beats baselines on a leak-free, temporally/grouped split.** Selection criteria: ranking metric (Spearman/NDCG) + calibrated intervals + parsimony. Prefer a single interpretable model for the demo unless the ensemble shows clear, honest lift.

### 9.4 Validation protocol

- **Temporal split** by `release_date` (already done, `model_trainer.py:245-262`) — keep, but it is *insufficient alone* given circular labels; fixing the label is the real cure.
- **Grouped CV** by publisher/genre to prevent franchise leakage (a sequel of a hit leaking into its prequel's fold).
- **Walk-forward** validation for any windowed label (the unused `BacktestValidator` at `model_monitoring.py:406-467` is the starting point).
- Report mean ± std across folds (the notebook's 5-fold std was large: ElasticNet 0.91 ± 0.13 — instability we must surface, not hide).

### 9.5 Hyperparameter tuning

Optuna is referenced in stale docs but **not in `requirements.txt`** — add it only if baselines justify tuning. For a course demo, light `GridSearchCV`/`RandomizedSearch` on the chosen model is sufficient. → `[DECISION REQUIRED] ML-3`.

### 9.6 Calibration & uncertainty

Provide prediction intervals (quantile regression, conformal prediction, or ensemble spread). The demo must show uncertainty — a point score without it is the exact overconfidence that produced the bogus R²=0.99 narrative.

### 9.7 Explainability

SHAP or permutation importance (model-agnostic) for honest per-prediction drivers, translated to plain language in the UI. The current `top_contributing_features` (importance × scaled magnitude, `webapp.py:350-390`) is a crude proxy — replace with SHAP for the demo's credibility.

---

## 10. System Architecture (Target State)

```
                    ┌─────────────── Ingestion ───────────────┐
   Steam API/HTML ─▶│ steam_data_miner  → SteamMetric (NEW)   │
   Twitch Helix   ─▶│ twitch_miner      → TwitchMetric        │
   YouTube v3     ─▶│ youtube_data_miner→ YouTubeMetric (top-N)│──▶ SQLite warehouse
   Reddit PRAW    ─▶│ reddit_data_miner → RedditMetric        │    (game_metrics.db)
                    └──────────────────────────────────────────┘            │
                                                                             ▼
                                    Label builder (NEW: review-velocity@30/90d, temporal firewall)
                                                                             │
                                                                             ▼
                                    Feature store (sql_feature_engineer, leak-free subset)
                                                                             │
                                                                             ▼
                                    Training (baselines → model) → registry (artifact + metadata + label def)
                                                                             │
                                                                             ▼
                                    Serving / Demo web app (Flow A live/cache, Flow B sandbox, explain)
```

### 10.1 Deprecation list

- **Notebooks:** `Enhanced_Game_Investment_Analysis.ipynb` (imports missing `enhanced_*` modules), `leakage_free_model_check.ipynb`, `Game_Investment_Potential_*`, `new_model_try.ipynb` → move to `archive/` or `notebooks/exploratory/`, clearly marked non-authoritative. *(Keep `new_model_try.ipynb`'s finding — Test R²≈0.0016 — as documented evidence.)*
- **CSV path:** root `youtube_data_miner.py` (legacy CSV miner) and `clean_post_release_columns.py` → archive.
- **Stale docs:** `SYSTEM_STATUS_REPORT.md`, `Integration_Guide.md`, `Enhanced_Analysis_README.md` → archive or rewrite; they reference nonexistent files and contradictory metrics.
- **Duplicate artifacts:** two copies of `enhanced_model_artifacts.pkl` (root + `models/`) → keep one canonical path.
- **Phantom module references** in `CLAUDE.md`/`README.md` (`app.py`, `mining.py`, `enhanced_features.py`, `enhanced_models.py`) → remove.

### 10.2 API design

Extend `src/api/webapp.py`: add `/sandbox`, `/samples`, optional `/compare`; move HTML to `templates/`; replace opaque importance with SHAP; **remove the nonexistent `app.py` from all docs** (do not create a second app).

---

## 11. MLOps & Operations

- **CI (`weekly_pipeline.yml`):** Keep the Monday 00:00 UTC cron and the mypy gate (good). **Fix the README drift:** README claims CI "commits updated DB + model back to the repo" — it actually pushes to **S3 + GitHub Releases** (workflow `:80-108`), no git commit. Align docs.
- **Add to CI:** a label-build step + a model smoke test (fixed-seed) before publishing an artifact; fail the run if metrics regress or if features overlap the label (an automated leakage check).
- **Model registry:** artifact already carries `best_model, feature_names, scaler` (`ModelArtifact`, `model_trainer.py:89-95`). **Add metadata:** label definition, split type, metrics, training date, data window — so a reviewer can audit any artifact.
- **Drift detection / retraining triggers:** wire `BacktestValidator`; retrain weekly or on drift.
- **Secrets management (P0):**
  - **Immediately rotate** the exposed Google OAuth secret (`client_secret_*.json` at repo root) and **purge it from git history** (`git filter-repo`/BFG); add to `.gitignore`.
  - `.env.example` documents 9 fields (Twitch×2, YouTube, Reddit×5, Flask). **Add the undocumented AWS fields** the workflow needs: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET`.

---

## 12. Migration Plan (Phased)

| Phase | Scope | Exit criteria | Rollback |
|---|---|---|---|
| **0 — Product + Audit + Security** | Resolve persona/positioning (§2.7); **rotate & purge leaked secret**; archive stale docs/notebooks; align README/CLAUDE to code | Persona decided; secret purged & rotated; docs match repo | Revert doc moves (low risk) |
| **1 — Labels + Schema** | Add `SteamMetric`; relax YouTube `UNIQUE(video_id)`; add `Label` storage; implement review-velocity label builder with temporal firewall | Schema migrated; ≥1 real label computed for ≥N games; leakage check passes | Keep old schema branch; SQLite file backup |
| **2 — Collection** | Persist Steam enrichment; YouTube top-N; repeated snapshots to build history; caching layer | ≥30–90d history for a pilot game set; cache populated | Disable new miners; fall back to seeded DB |
| **3 — Features** | Leak-free feature subset; redesign temporal/competitive features; presence flags | Feature regression tests pass; no feature ∈ label | Pin previous feature set |
| **4 — Model** | Baselines → candidate model; grouped/temporal/walk-forward CV; calibration; SHAP | Beats naive baselines on leak-free split; honest metrics + intervals reported | Revert to baseline model artifact |
| **5 — Demo web app + serving** | Flows A+B (MVP), explanation UI, cached fallback, freshness/limitations panel | 5-min demo runs offline; sandbox works; explanation legible | Serve previous webapp version |

---

## 13. Testing Strategy

- **Unit / contract (miners):** Extend existing mocked tests (`tests/test_twitch_*`, `test_reddit_data_miner.py`, `test_youtube_data_miner.py`). **Add Steam miner tests** (none exist) covering `GameSnapshot`, wishlist HTML parsing, `resolve_appid_by_name`. Contract tests pin expected API response shapes (mocked).
- **Feature regression:** golden-file snapshot of the SQL feature matrix on a fixed seed DB; diff on change. **Add an automated leakage assertion:** no feature column may be a component of the label.
- **Model smoke tests:** fixed-seed train → assert metrics within a band, artifact keys present, predict() returns finite clipped values. *(None exist today.)*
- **Web app E2E / API:** test `/predict` (mock miners → cached features), `/sandbox` (deterministic, no network), error states (400/404/429 fallback/503).
- **Coverage targets:** miners & utils ≥ 80% (current convention); features & model ≥ 70%; API routes ≥ 70%.

---

## 14. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Leaked OAuth secret in repo/history** | **P0** | Rotate now; purge history; gitignore; audit for other secrets |
| **Circular/tautological label** | **Critical** | Replace with external label + temporal firewall + automated leakage check (§9.1) |
| **Building a tool nobody asked for** | **Critical** | Persona validation + competitor matrix + instructor check before locking scope (§2) |
| **Label noise / small-N games** | High | Grouped CV, report intervals, raise `MIN_REVIEWS_FOR_SEED`, curate sample set |
| **YouTube quota mid-demo** | High | Sandbox mode + cached samples (§5.4) — demo never depends on live quota |
| **Demo failure during live presentation** | High | Flow B offline; cached fallback banner; dry-run before presenting |
| **Overfitting to historical hits / franchise leakage** | Medium | Grouped CV by publisher; walk-forward; honest baselines |
| **Steam store HTML change breaks wishlist scrape** | Medium | Wrap in try/except → null + presence flag; treat as optional feature |
| **ToS violation via scraping** | Medium | API-first posture; no Selenium; document gray-area Steam scrape (§7) |
| **Heavy deps (torch/transformers) slow CI & demo** | Medium | Lazy-load emotion model (already done, `reddit_data_miner.py:261-277`); consider making emotion optional |

---

## 15. Open Questions — `[DECISION REQUIRED]`

**P-1 Primary persona (v1).**
- Options: A investor-scout / B indie-developer / C course-demo-first.
- Recommendation: **C, positioned as the indie-developer engagement-intelligence wedge.**
- Impact if wrong: features and copy serve a user who won't trust the output.

**P-2 Positioning / what we solve in one sentence.**
- Options: A engagement-intelligence wedge / B sales forecaster / C builder readiness.
- Recommendation: **A** — *"cross-platform community engagement & sentiment intelligence for a game, explained."*
- Impact if wrong: competing with Gamalytic/VGI on a problem free data can't solve.

**ML-1 Ground-truth label.**
- Options: A review-velocity@30/90d / B player-retention proxy / C keep composite as descriptive index only (no ML) / D revenue proxy (Boxleiter).
- Recommendation: **A** as supervision, **C** as a UI index. Never the old circular target.
- Impact if wrong: another tautological model; R² meaningless.

**ML-2 Lifecycle stage.**
- Options: A pre-release scoring / B post-release scoring / C both.
- Recommendation: **B** for v1 (data is post-release-rich; pre-release was the Test R²≈0.0016 failure).
- Impact if wrong: predicting a stage the data can't support.

**ML-3 Hyperparameter tuning scope.**
- Options: A none / B light grid/random / C Optuna.
- Recommendation: **B** unless baselines justify C.
- Impact if wrong: wasted effort or under-tuned model.

**DATA-1 API vs. scrape posture (per platform).**
- Options: A API-only / B API + keep Steam wishlist scrape / C add Selenium scraping.
- Recommendation: **B**. No new scraping.
- Impact if wrong: ToS risk, fragility, maintenance burden.

**DATA-2 History depth investment.**
- Options: A single-snapshot (fast, weak labels) / B 30d / C 90d windows.
- Recommendation: **B** as a pragmatic course-project target; C if time allows.
- Impact if wrong: windowed labels/features remain aspirational; "temporal" features stay misnamed.

**DATA-3 Steam enrichment persistence.**
- Options: A add `SteamMetric` now / B defer.
- Recommendation: **A** — it's the cheapest high-value fix and unblocks real labels.
- Impact if wrong: Steam financial-proxy signals remain unusable.

**UX-1 Demo mode: live vs. sandbox-first.**
- Options: A live mining / B sandbox-first / C both (live with cached fallback).
- Recommendation: **C**.
- Impact if wrong: live presentation breaks on quota/unknown game.

**UX-2 Web stack.**
- Options: A extend Flask+templates / B SPA (React).
- Recommendation: **A**.
- Impact if wrong: build complexity with no demo benefit.

**UX-3 Persona views / comparison (Flow C).**
- Options: A build / B defer to stretch.
- Recommendation: **B** unless persona work demands it.
- Impact if wrong: scope creep before MVP lands.

**ARCH-1 Notebook & CSV deprecation.**
- Options: A archive all / B keep some authoritative.
- Recommendation: **A** (archive; keep `new_model_try.ipynb` as evidence).
- Impact if wrong: continued doc/code drift.

**ARCH-2 Database engine.**
- Options: A SQLite / B Postgres.
- Recommendation: **A** for course scope.
- Impact if wrong: needless ops complexity.

**OPS-1 Emotion model in pipeline.**
- Options: A keep (Reddit emotion via HuggingFace) / B make optional / C drop.
- Recommendation: **B** — valuable differentiator but heavy; lazy-load + toggle.
- Impact if wrong: slow CI/demo, or losing a differentiating feature.

**OPS-2 `MIN_REVIEWS_FOR_SEED` threshold.**
- Options: A keep =1 / B raise (e.g., 50–500).
- Recommendation: **B** — `=1` admits noise-tier games (`steam_data_miner.py:48`).
- Impact if wrong: tiny-N games pollute training and labels.

---

## 16. Appendices

### 16.1 Glossary

- **Circular / tautological supervision:** training a model to predict a target computed from its own input features.
- **Boxleiter method:** sales estimation by multiplying review count by a genre/era multiplier (used by Gamalytic/VGI).
- **Temporal firewall:** strict rule that all features predate the label's observation window.
- **Grouped CV:** cross-validation that keeps related items (same publisher/franchise) in the same fold to prevent leakage.
- **Engagement index (descriptive):** a transparent composite of current engagement, used for display, *not* as ML supervision.

### 16.2 Current vs. target file inventory

| Item | Current | Target |
|---|---|---|
| Steam metrics | Fetched into `GameSnapshot`, **discarded** | Persist to new `SteamMetric` |
| YouTube sampling | 1 video/game (`SEARCH_PAGE_SIZE=5`→1) | top-N (≥5), repeated over time |
| `YouTubeMetric` uniqueness | `UNIQUE(video_id)` blocks re-observation | `UNIQUE(video_id, mined_at/batch)` |
| Label | Circular `target_score` computed from features | External review-velocity label + temporal firewall |
| Explainability | importance×magnitude (`webapp.py:350-390`) | SHAP, plain-language |
| Demo modes | live-only | live+cache (A) + sandbox (B) |
| `enhanced_features.py`, `enhanced_models.py`, `src/api/app.py`, `mining.py` | **Referenced in docs, do not exist** | Remove references |
| Stale docs | `SYSTEM_STATUS_REPORT.md`, `Integration_Guide.md`, `Enhanced_Analysis_README.md` | Archive/rewrite |
| Leaked secret | `client_secret_*.json` at root | Purge + rotate |
| Duplicate artifacts | `enhanced_model_artifacts.pkl` ×2 | One canonical path |

### 16.3 References (desk research — validate before locking scope)

- Gamalytic — Steam analytics & sales estimation: https://gamalytic.com/ , methodology: https://gamalytic.com/blog/how-to-accurately-estimate-steam-sales
- VG Insights — games industry data & Steam unit sales estimator: https://app.sensortower.com/vgi/ , https://app.sensortower.com/vgi/indie-tools/unit-sales-estimator
- SteamDB: https://steamdb.info/ · SteamSpy: https://steamspy.com/
- IMPRESS wishlist→sales calculator: https://impress.games/steam-wishlists-sales-calculator
- Wayline forecasting: https://www.wayline.io/blog/forecasting-indie-game-sales-wishlist-data-revenue-projections
- Steam Web API docs: https://partner.steamgames.com/doc/webapi · Reviews: https://partner.steamgames.com/doc/store/getreviews
- IGDB (Twitch-owned game DB, possible enrichment): https://api-docs.igdb.com/
- Internal: `FEATURE_MATHEMATICS_AUDIT.md`, `new_model_try.ipynb` (Test R²≈0.0016), `src/models/model_trainer.py:39-56` (target spec).

### 16.4 Sample demo games (hits / flops / edge cases)

- **Hits:** Counter-Strike 2, Dota 2, Baldur's Gate 3, Stardew Valley.
- **Flops / quiet:** a known commercial flop with low engagement (curate from public post-mortems).
- **Edge cases:** F2P with huge CCU but few "sales" (skews revenue proxies); a sleeper hit with low wishlists but strong sales (e.g., the *Nubby's Number Factory* pattern — ~1,500 wishlists, ~200k sales); a sequel of a hit (franchise-leakage test); a game with no Twitch presence (platform-absence test).

---

## Round 1 Discussion Agenda — Top 7 Decisions to Resolve First

> Resolve these before Phase 1 coding. At least two are product/UX (not ML/data).

1. **[Product] P-2 Positioning / one-sentence problem.** Engagement-intelligence wedge (A) vs. sales forecaster (B)? Everything downstream — label, features, UI copy — depends on this. *Recommend A.*
2. **[Product] P-1 Primary persona for v1.** Investor-scout, indie-developer, or course-demo-first? *Recommend course-demo-first as the indie-developer engagement wedge.*
3. **[ML] ML-1 Ground-truth label.** Replace the circular `target_score`. Review-velocity@30/90d as supervision; old composite demoted to a UI descriptive index. *This is the project's make-or-break decision.*
4. **[ML] ML-2 Lifecycle stage.** Pre-release vs. post-release scoring — given pre-release features yielded Test R²≈0.0016, *recommend post-release for v1.*
5. **[UX] UX-1 Demo mode.** Live-mining vs. sandbox-first vs. both. *Recommend both (live + cached fallback + offline sandbox)* so the presentation can't fail on quota.
6. **[Data] DATA-1 + DATA-3 Collection posture & Steam persistence.** API-only vs. keep-Steam-scrape (no Selenium); add `SteamMetric` now? *Recommend API + existing wishlist scrape, and add `SteamMetric` immediately.*
7. **[Scope/Security] Phase-0 gate: MVP cut + P0 secret remediation.** Define what ships for the course demo vs. deferred, and **rotate + purge the leaked Google OAuth secret** before any further work. *Non-negotiable security item bundled with scope-cut.*

---

## 17. Round 1 Decisions — LOCKED (owner confirmed)

All seven agenda items resolved. These override conflicting `[DECISION REQUIRED]` recommendations elsewhere in this doc where noted.

### 17.1 Product

| ID | Decision | Locked choice |
|---|---|---|
| **P-2** | Positioning | **A — Engagement-intelligence wedge.** One-sentence problem: *"We measure and explain cross-platform community engagement and sentiment for a Steam game — not sales or ROI."* |
| **P-1** | Primary persona (v1) | **C → positioned as B:** course-demo delivery, **framed for indie developers / small studios** who need cheap, explainable community signal. Investor-scout is out of scope for v1 copy. |

**Naming / copy:** Retire public-facing **"investment potential"** language. Working product name direction: **Game Engagement Intelligence** (or similar — finalize in Round 2 wireframes).

### 17.2 ML & labels (hybrid v1)

| ID | Decision | Locked choice |
|---|---|---|
| **ML-1** | Ground truth | **Hybrid for v1:** (1) **Engagement index** — transparent composite, UI-only, *never* used as ML supervision; (2) **ML stretch** — review-velocity @ 30/90d when `SteamMetric` time series exists; (3) **Demo sanity** — optional hit/flop comparison on curated games. **Retire circular `target_score` as supervision.** |
| **ML-2** | Lifecycle | **B — Post-release only** for v1. Default demo games are released titles with observable engagement. Pre-release is deferred. |

**Grading / credibility bar:** A **strong descriptive dashboard + documented engagement index + baselines** (beat naive mean / raw index where ML runs) is sufficient for the course demo. High R² is not a goal.

**Note:** Pre-release → post-release modeling in `new_model_try.ipynb` achieved Test R² ≈ **0.03** (near zero), not 0.0016 — cite honestly if referenced.

### 17.3 Demo UX

| ID | Decision | Locked choice |
|---|---|---|
| **UX-1** | Demo modes | **C — Both:** sandbox (Flow B) **first in live presentation**, then live/cache (Flow A) for one known game if quota allows. Cached fallback on 404/429. |
| **UX-2** | Web stack | **A — Flask + Jinja** (extend `webapp.py`; move inline HTML to `templates/`). |
| **UX-3** | Comparison view | **Deferred** (Flow C stretch). |

### 17.4 Data & collection

| ID | Decision | Locked choice |
|---|---|---|
| **DATA-1** | API vs scrape | **B — API-first + keep existing Steam wishlist HTML scrape.** No new Selenium scraping. |
| **DATA-3** | Steam persistence | **A — Add `SteamMetric` table now**; persist enrichment currently discarded in `GameSnapshot`. |
| **DATA-2** | History depth | **30d target** for review-velocity stretch; single-snapshot mode OK until history accumulates (label clearly in UI). |
| **OPS-2** | `MIN_REVIEWS_FOR_SEED` | **Raise** from `1` to **50–100** (exact value in Round 2 schema spec). |

### 17.5 MVP scope & security (Phase 0 gate)

**Ship for course demo**

- Phase 0: archive stale docs/notebooks; align README/CLAUDE to code; secret audit
- `SteamMetric` + persist Steam fields
- Demo webapp: Flow A + B + cached samples + explain panel (label def, freshness, limitations)
- Engagement index (documented formula) + SHAP or equivalent plain-language drivers
- 8–12 cached demo games (post-release hits, flops, edge cases)
- Baseline comparisons if ML training runs in v1

**Defer**

- Flow C side-by-side compare UI
- 90d windows / walk-forward CV until data exists
- Optuna / ensemble tuning beyond simple baselines
- Pre-release scoring
- Postgres; emotion model always-on (optional/lazy-load)
- Public "investment potential" / sales-forecast messaging

**Security (P0 before implementation):** Audit git history for `client_secret_*.json`; rotate if ever committed; extend `.env.example` with AWS CI vars.

### 17.6 Implied answers (owner confirmed)

- **Presentation strategy:** Sandbox-first live demo; one optional live mine.
- **Product rename:** Yes — drop "investment potential" from user-facing copy.
- **Model bar:** Descriptive + explainable dashboard acceptable; external-label ML is stretch after `SteamMetric` history.

---

## Round 2 scope (next Claude Code pass)

Produce **`docs/plans/PIPELINE_REBUILD_PHASE0-1.md`** (or update this SRS §5–§6–§12 with detail) covering:

1. **Engagement index spec** — exact formula, feature list, 1–100 scaling, disclaimer text
2. **`SteamMetric` schema** — columns, migration, miner changes, `MIN_REVIEWS_FOR_SEED` value
3. **Demo wireframes** — `/`, `/sandbox`, `/predict`, `/samples`; sandbox-first presenter script
4. **Cached demo game list** — 8–12 titles with expected narrative
5. **Phase 0 + Phase 1 task breakdown** — ordered, with exit criteria for Cursor Agent

Use prompt: **`docs/plans/PIPELINE_REBUILD_ROUND2_PROMPT.md`**.

---

*Round 1 closed. Proceed to Round 2 planning, then Cursor Agent implementation.*
