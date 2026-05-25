# Feature Mathematics Audit — `AdvancedFeatureEngineer` & Merge Pipeline

**Date:** 2026-05-25  
**Files audited:** `src/features/enhanced_features.py`, `src/features/feature_pipeline.py`  
**Scope:** Every engineered feature column, its raw formula, dependencies, and leakage risk classification.

---

## Leakage Risk Taxonomy

| Risk | Definition |
|---|---|
| **HIGH** | Uses a **global** `.sum()`, `.mean()`, `.max()`, or `.std()` across the entire DataFrame (cross-game). Every prediction row "knows" about every other row. |
| **MEDIUM** | Uses `groupby().transform()` (e.g., per-genre stats). Leaks information across games *within the same group* if the model is deployed in a temporal context. |
| **LOW** | Row-local: only depends on columns from its own `game_title` group. Safe for window-function translation. |

---

## PART 1: Per-Platform Aggregation Features

These are produced by `groupby('game_title').agg(...)` — each feature is computed over the **rows belonging to a single game** only. All features in this section are **LOW** risk.

### 1.1 Twitch Features (`aggregate_twitch_data`, line 95–107)

| # | Feature Name | Math / Logic | Raw Dependencies | Leakage |
|---|---|---|---|---|
| 1 | `twitch_total_viewers` | $\sum(\text{viewer\_count})$ per game | `viewer_count` | LOW |
| 2 | `twitch_avg_viewers` | $\bar{x} = \frac{1}{n}\sum(\text{viewer\_count})$ per game | `viewer_count` | LOW |
| 3 | `twitch_max_viewers` | $\max(\text{viewer\_count})$ per game | `viewer_count` | LOW |
| 4 | `twitch_viewer_std` | $\sigma(\text{viewer\_count})$ per game | `viewer_count` | LOW |
| 5 | `twitch_stream_count` | $\text{count}(\text{viewer\_count})$ per game | `viewer_count` | LOW |
| 6 | `twitch_unique_streamers` | $\text{nunique}(\text{user\_name})$ per game | `user_name` | LOW |
| 7 | `twitch_viewer_cv` | $\displaystyle\frac{\text{twitch\_viewer\_std}}{\text{twitch\_avg\_viewers} + 1}$ | `twitch_viewer_std`, `twitch_avg_viewers` | LOW |

### 1.2 YouTube Features (`aggregate_youtube_data`, line 109–130)

| # | Feature Name | Math / Logic | Raw Dependencies | Leakage |
|---|---|---|---|---|
| 8 | `youtube_total_views` | $\sum(\text{view\_count})$ per game | `view_count` | LOW |
| 9 | `youtube_avg_views` | $\bar{x} = \frac{1}{n}\sum(\text{view\_count})$ per game | `view_count` | LOW |
| 10 | `youtube_max_views` | $\max(\text{view\_count})$ per game | `view_count` | LOW |
| 11 | `youtube_view_std` | $\sigma(\text{view\_count})$ per game | `view_count` | LOW |
| 12 | `youtube_video_count` | $\text{count}(\text{view\_count})$ per game | `view_count` | LOW |
| 13 | `youtube_total_likes` | $\sum(\text{like\_count})$ per game | `like_count` | LOW |
| 14 | `youtube_avg_likes` | $\bar{x} = \frac{1}{n}\sum(\text{like\_count})$ per game | `like_count` | LOW |
| 15 | `youtube_max_likes` | $\max(\text{like\_count})$ per game | `like_count` | LOW |
| 16 | `youtube_total_comments` | $\sum(\text{comment\_count})$ per game | `comment_count` | LOW |
| 17 | `youtube_avg_comments` | $\bar{x} = \frac{1}{n}\sum(\text{comment\_count})$ per game | `comment_count` | LOW |
| 18 | `youtube_max_comments` | $\max(\text{comment\_count})$ per game | `comment_count` | LOW |
| 19 | `youtube_avg_sentiment` | $\bar{x} = \frac{1}{n}\sum(\text{avg\_comment\_sentiment})$ per game | `avg_comment_sentiment` | LOW |
| 20 | `youtube_sentiment_std` | $\sigma(\text{avg\_comment\_sentiment})$ per game | `avg_comment_sentiment` | LOW |
| 21 | `youtube_pos_ratio` | $\bar{x} = \frac{1}{n}\sum(\text{pos\_comment\_ratio})$ per game | `pos_comment_ratio` | LOW |
| 22 | `youtube_pos_ratio_std` | $\sigma(\text{pos\_comment\_ratio})$ per game | `pos_comment_ratio` | LOW |
| 23 | `youtube_total_subscribers` | $\sum(\text{channel\_subscriber\_count})$ per game | `channel_subscriber_count` | LOW |
| 24 | `youtube_avg_subscribers` | $\bar{x} = \frac{1}{n}\sum(\text{channel\_subscriber\_count})$ per game | `channel_subscriber_count` | LOW |
| 25 | `youtube_max_subscribers` | $\max(\text{channel\_subscriber\_count})$ per game | `channel_subscriber_count` | LOW |
| 26 | `youtube_engagement_rate` | $\displaystyle\frac{\text{youtube\_total\_likes} + \text{youtube\_total\_comments}}{\text{youtube\_total\_views} + 1}$ | `youtube_total_likes`, `youtube_total_comments`, `youtube_total_views` | LOW |

### 1.3 Reddit Features (`aggregate_reddit_data`, line 132–156)

| # | Feature Name | Math / Logic | Raw Dependencies | Leakage |
|---|---|---|---|---|
| 27 | `reddit_total_score` | $\sum(\text{score})$ per game | `score` | LOW |
| 28 | `reddit_avg_score` | $\bar{x} = \frac{1}{n}\sum(\text{score})$ per game | `score` | LOW |
| 29 | `reddit_max_score` | $\max(\text{score})$ per game | `score` | LOW |
| 30 | `reddit_score_std` | $\sigma(\text{score})$ per game | `score` | LOW |
| 31 | `reddit_post_count` | $\text{count}(\text{score})$ per game | `score` | LOW |
| 32 | `reddit_total_comments` | $\sum(\text{num\_comments})$ per game | `num_comments` | LOW |
| 33 | `reddit_avg_comments` | $\bar{x} = \frac{1}{n}\sum(\text{num\_comments})$ per game | `num_comments` | LOW |
| 34 | `reddit_max_comments` | $\max(\text{num\_comments})$ per game | `num_comments` | LOW |
| 35 | `reddit_avg_sentiment` | $\bar{x} = \frac{1}{n}\sum(\text{avg\_comment\_sentiment})$ per game | `avg_comment_sentiment` | LOW |
| 36 | `reddit_sentiment_std` | $\sigma(\text{avg\_comment\_sentiment})$ per game | `avg_comment_sentiment` | LOW |
| 37 | `reddit_pos_ratio` | $\bar{x} = \frac{1}{n}\sum(\text{pos\_comment\_ratio})$ per game | `pos_comment_ratio` | LOW |
| 38 | `reddit_pos_ratio_std` | $\sigma(\text{pos\_comment\_ratio})$ per game | `pos_comment_ratio` | LOW |
| 39 | `reddit_avg_author_karma` | $\bar{x} = \frac{1}{n}\sum(\text{author\_link\_karma})$ per game | `author_link_karma` | LOW |
| 40 | `reddit_max_author_karma` | $\max(\text{author\_link\_karma})$ per game | `author_link_karma` | LOW |
| 41 | `reddit_avg_comment_karma` | $\bar{x} = \frac{1}{n}\sum(\text{author\_comment\_karma})$ per game | `author_comment_karma` | LOW |
| 42 | `reddit_max_comment_karma` | $\max(\text{author\_comment\_karma})$ per game | `author_comment_karma` | LOW |
| 43 | `reddit_total_commenters` | $\sum(\text{unique\_commenters})$ per game | `unique_commenters` | LOW |
| 44 | `reddit_avg_commenters` | $\bar{x} = \frac{1}{n}\sum(\text{unique\_commenters})$ per game | `unique_commenters` | LOW |
| 45 | `reddit_max_commenters` | $\max(\text{unique\_commenters})$ per game | `unique_commenters` | LOW |
| 46 | `reddit_total_awards` | $\sum(\text{num\_awards})$ per game | `num_awards` | LOW |
| 47 | `reddit_avg_awards` | $\bar{x} = \frac{1}{n}\sum(\text{num\_awards})$ per game | `num_awards` | LOW |
| 48 | `reddit_max_awards` | $\max(\text{num\_awards})$ per game | `num_awards` | LOW |
| 49 | `reddit_engagement_rate` | $\displaystyle\frac{\text{reddit\_total\_comments} + \text{reddit\_total\_awards}}{\text{reddit\_post\_count} + 1}$ | `reddit_total_comments`, `reddit_total_awards`, `reddit_post_count` | LOW |

> **Note on `feature_pipeline.py` discrepancy (lines 70–71, 93):** The `GameInvestmentPredictor.engineer_features` method computes `youtube_engagement_rate` with **no +1 in denominator** and `reddit_engagement_rate` with **no +1 in denominator**, unlike the module-level functions. This means the same feature can take different values depending on which code path is called.

---

## PART 2: Advanced Engineered Features (`AdvancedFeatureEngineer`)

### 2.1 Viral Potential Features (`create_viral_potential_features`, lines 240–282)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 50 | `viral_velocity` | $\bigl(\text{youtube\_total\_views} \times \text{youtube\_engagement\_rate} \times \text{reddit\_total\_score}\bigr)^{1/3}$ | `youtube_total_views`, `youtube_engagement_rate`, `reddit_total_score` | LOW |
| 51 | `shareability_score` | $\text{youtube\_total\_likes} + \text{reddit\_total\_score} + 0.1 \times \text{twitch\_total\_viewers}$ | `youtube_total_likes`, `reddit_total_score`, `twitch_total_viewers` | LOW |
| 52 | `cross_platform_viral` | $0.001 \times \text{youtube\_total\_views} + 0.1 \times \text{twitch\_total\_viewers} + \text{reddit\_total\_score}$ | `youtube_total_views`, `twitch_total_viewers`, `reddit_total_score` | LOW |
| 53 | `engagement_amplification` | $\displaystyle\frac{\text{youtube\_engagement\_rate} + \text{reddit\_engagement\_rate} + \text{twitch\_viewer\_cv}}{3}$ | `youtube_engagement_rate`, `reddit_engagement_rate`, `twitch_viewer_cv` | LOW |
| 54 | `viral_potential_score` | $0.3 \times \text{viral\_velocity} + 0.3 \times 0.0001 \times \text{shareability\_score} + 0.2 \times 0.0001 \times \text{cross\_platform\_viral} + 0.2 \times \text{engagement\_amplification}$ | `viral_velocity`, `shareability_score`, `cross_platform_viral`, `engagement_amplification` | LOW |

### 2.2 Competitive Features (`create_competitive_features`, lines 284–321)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 55 | `market_share` | $\displaystyle\frac{\text{youtube\_total\_views}_i}{\sum_{j=1}^{N}\text{youtube\_total\_views}_j + 1}$ | `youtube_total_views` | **HIGH** |
| 56 | `competitive_advantage` | $\displaystyle\frac{\text{youtube\_engagement\_rate} + \text{reddit\_engagement\_rate} + \text{twitch\_viewer\_cv}}{3}$ | `youtube_engagement_rate`, `reddit_engagement_rate`, `twitch_viewer_cv` | LOW |
| 57 | `platform_dominance` | Let $E_i = \text{twitch\_total\_viewers}_i + 0.001 \times \text{youtube\_total\_views}_i + \text{reddit\_total\_score}_i$. Then $\displaystyle\frac{E_i}{\sum_{j=1}^{N} E_j + 1}$ | `twitch_total_viewers`, `youtube_total_views`, `reddit_total_score` | **HIGH** |
| 58 | `competitive_score` | $0.4 \times \text{market\_share} + 0.3 \times \text{competitive\_advantage} + 0.3 \times \text{platform\_dominance}$ | `market_share` (HIGH), `competitive_advantage`, `platform_dominance` (HIGH) | **HIGH** |

> **Why HIGH for #55 and #57:** Both use `.sum()` on the **entire** column, meaning the value for game $i$ depends on the values of every other game in the batch. This is impossible to compute correctly with a row-local window function and constitutes severe data leakage — the model sees the aggregate of "future" or "peer" data.

### 2.3 Cross-Platform Features (`create_cross_platform_features`, lines 323–358)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 59 | `cross_platform_engagement_rate` | $\displaystyle\frac{\text{youtube\_engagement\_rate} + \text{reddit\_engagement\_rate} + \text{twitch\_viewer\_cv}}{3}$ | `youtube_engagement_rate`, `reddit_engagement_rate`, `twitch_viewer_cv` | LOW |
| 60 | `platform_synergy` | $\bigl(0.001 \times \text{youtube\_total\_views} \times 0.1 \times \text{twitch\_total\_viewers} \times \text{reddit\_total\_score}\bigr)^{1/3}$ | `youtube_total_views`, `twitch_total_viewers`, `reddit_total_score` | LOW |
| 61 | `cross_platform_reach` | $0.001 \times \text{youtube\_total\_views} + 0.1 \times \text{twitch\_total\_viewers} + \text{reddit\_total\_score}$ | `youtube_total_views`, `twitch_total_viewers`, `reddit_total_score` | LOW |
| 62 | `platform_balance` | $1 - \displaystyle\frac{\sigma\bigl([e_{yt},\, e_{rd},\, e_{tw}]\bigr)}{\mu\bigl([e_{yt},\, e_{rd},\, e_{tw}]\bigr) + 1}$ where $e_{yt}=\text{youtube\_engagement\_rate}$, $e_{rd}=\text{reddit\_engagement\_rate}$, $e_{tw}=\text{twitch\_viewer\_cv}$, and $\sigma,\mu$ are computed across the 3 platform values **per row** (`axis=0` on a $(3,N)$ array) | `youtube_engagement_rate`, `reddit_engagement_rate`, `twitch_viewer_cv` | LOW |

> **Note on `platform_balance` (line 356):** Despite using `np.std()` and `np.mean()`, the `axis=0` parameter causes the aggregation across the 3-element platform axis **for each row independently**, not across rows. This is row-local and safe. However, the code is fragile — if the array were constructed differently, this could silently become a global aggregation.

### 2.4 Temporal Features (`create_temporal_features`, lines 360–397)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 63 | `growth_momentum` | $\displaystyle\frac{1}{3}\left(\frac{\text{youtube\_total\_views}}{\max(\text{youtube\_video\_count}, 1)} + \frac{\text{twitch\_total\_viewers}}{\max(\text{twitch\_stream\_count}, 1)} + \frac{\text{reddit\_total\_score}}{\max(\text{reddit\_post\_count}, 1)}\right)$ | `youtube_total_views`, `youtube_video_count`, `twitch_total_viewers`, `twitch_stream_count`, `reddit_total_score`, `reddit_post_count` | LOW |
| 64 | `activity_consistency` | $\displaystyle\frac{1}{\frac{\text{youtube\_view\_std}}{\text{youtube\_avg\_views}+1} + \frac{\text{twitch\_viewer\_std}}{\text{twitch\_avg\_viewers}+1} + \frac{\text{reddit\_score\_std}}{\text{reddit\_avg\_score}+1} + 1}$ | `youtube_view_std`, `youtube_avg_views`, `twitch_viewer_std`, `twitch_avg_viewers`, `reddit_score_std`, `reddit_avg_score` | LOW |
| 65 | `engagement_trend` | $\displaystyle\frac{1}{3}\bigl(\text{youtube\_video\_count} \times \text{youtube\_engagement\_rate} + \text{twitch\_stream\_count} \times \text{twitch\_viewer\_cv} + \text{reddit\_post\_count} \times \text{reddit\_engagement\_rate}\bigr)$ | `youtube_video_count`, `youtube_engagement_rate`, `twitch_stream_count`, `twitch_viewer_cv`, `reddit_post_count`, `reddit_engagement_rate` | LOW |
| 66 | `peak_activity` | $0.001 \times \text{youtube\_max\_views} + 0.1 \times \text{twitch\_max\_viewers} + \text{reddit\_max\_score}$ | `youtube_max_views`, `twitch_max_viewers`, `reddit_max_score` | LOW |

### 2.5 Engagement Quality Features (`create_engagement_quality_features`, lines 399–425)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 67 | `like_per_view` | $\displaystyle\frac{\text{youtube\_total\_likes}}{\max(\text{youtube\_total\_views}, 1)}$ | `youtube_total_likes`, `youtube_total_views` | LOW |
| 68 | `comment_per_view` | $\displaystyle\frac{\text{youtube\_total\_comments}}{\max(\text{youtube\_total\_views}, 1)}$ | `youtube_total_comments`, `youtube_total_views` | LOW |
| 69 | `share_per_view` | $\displaystyle\frac{\text{total\_shares}}{\max(\text{youtube\_total\_views}, 1)}$ | `total_shares` *, `youtube_total_views` | LOW |
| 70 | `avg_watch_time` | $\displaystyle\frac{\text{total\_watch\_time}}{\max(\text{youtube\_total\_views}, 1)}$ | `total_watch_time` *, `youtube_total_views` | LOW |
| 71 | `completion_rate` | $\text{completion\_rate}$ (identity; defaults to $0.5$ if missing) | `completion_rate` * | LOW |
| 72 | `subscriber_engagement` | $\displaystyle\frac{\text{youtube\_total\_likes}}{\max(\text{youtube\_total\_subscribers}, 1)}$ | `youtube_total_likes`, `youtube_total_subscribers` | LOW |
| 73 | `viral_coefficient` | $\displaystyle\frac{\text{total\_shares}}{\max(\text{youtube\_total\_views}, 1)} \times \frac{\text{youtube\_total\_comments}}{\max(\text{youtube\_total\_views}, 1)}$ | `total_shares` *, `youtube_total_views`, `youtube_total_comments` | LOW |

> **Columns marked `*`** (`total_shares`, `total_watch_time`, `completion_rate`) **do not exist** in the aggregation pipeline output. They are never populated by `aggregate_twitch_data`, `aggregate_youtube_data`, `aggregate_reddit_data`, or the Steam merge. Because `.get(col, default)` returns the default, these features will always be **constant zero** (or the hardcoded default) in practice. They are dead features in the current pipeline.

### 2.6 Sentiment Features (`create_sentiment_features`, lines 427–444)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 74 | `sentiment_volatility` | $\text{youtube\_sentiment\_std}$ (identity pass-through) | `youtube_sentiment_std` | LOW |
| 75 | `sentiment_trend` | $\text{youtube\_avg\_sentiment}$ (identity pass-through) | `youtube_avg_sentiment` | LOW |
| 76 | `sentiment_authenticity` | $1 - 2 \times \lvert\text{youtube\_pos\_ratio} - 0.5\rvert$ | `youtube_pos_ratio` | LOW |

### 2.7 Network Features (`create_network_features`, lines 446–466)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 77 | `avg_creator_influence` | $\text{youtube\_avg\_subscribers}$ (identity pass-through) | `youtube_avg_subscribers` | LOW |
| 78 | `creator_diversity` | $\text{twitch\_unique\_streamers}$ (identity pass-through) | `twitch_unique_streamers` | LOW |
| 79 | `community_cohesion` | $\displaystyle\frac{\text{reddit\_total\_commenters}}{\max(\text{reddit\_total\_comments}, 1)}$ | `reddit_total_commenters`, `reddit_total_comments` | LOW |
| 80 | `creator_concentration` | $\displaystyle\frac{\text{youtube\_max\_subscribers}}{\max(\text{youtube\_avg\_subscribers}, 1)}$ | `youtube_max_subscribers`, `youtube_avg_subscribers` | LOW |

### 2.8 Market Context Features (`create_market_context_features`, lines 492–511)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 81 | `market_density` | $\text{COUNT}(\text{game\_title})$ partitioned by `genre` (via `groupby('genre').transform('count')`) | `genre` †, `game_title` | **MEDIUM** |
| 82 | `relative_performance` | $\text{PERCENT\_RANK}(\text{engagement\_score})$ partitioned by `genre` (via `groupby('genre').rank(pct=True)`) | `genre` †, `engagement_score` † | **MEDIUM** |
| 83 | `market_share` (genre) | $\displaystyle\frac{\text{engagement\_score}_i}{\sum_{k \in \text{genre}(i)} \text{engagement\_score}_k}$ via `groupby('genre').transform('sum')` | `genre` †, `engagement_score` † | **MEDIUM** |
| 84 | `genre_momentum` | $\displaystyle\bar{x}_{\text{genre}} = \frac{1}{\lvert G\rvert}\sum_{k \in G} \text{engagement\_score}_k$ via `groupby('genre').transform('mean')` | `genre` †, `engagement_score` † | **MEDIUM** |

> **Columns marked `†`** (`genre`, `engagement_score`) are **not produced** by any upstream aggregation function in the current pipeline. The `create_market_context_features` method is gated on `if 'genre' in enhanced_df.columns` (line 548), and `engagement_score` must also exist. In the current merge pipeline, neither column exists, so **features #81–#84 are never created**. However, if the pipeline were extended to include Steam genre data and compute an engagement score, these would all be MEDIUM risk.

> **Why MEDIUM:** These use `groupby().transform()`, which computes statistics within each genre. A game's feature value depends on all other games in the same genre that happen to be in the training/inference batch. For a temporal prediction scenario (predicting future investment potential), this leaks cross-game information within the same genre.

### 2.9 Interaction Features (`create_interaction_features`, lines 468–490)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 85–94 | `total_views total_likes`, `total_views total_comments`, `total_views avg_sentiment`, `total_views platform_presence`, `total_likes total_comments`, `total_likes avg_sentiment`, `total_likes platform_presence`, `total_comments avg_sentiment`, `total_comments platform_presence`, `avg_sentiment platform_presence` | All pairwise products: $x_i \times x_j$ for $\{i,j\} \subset \{\text{total\_views}, \text{total\_likes}, \text{total\_comments}, \text{avg\_sentiment}, \text{platform\_presence}\}$ via `PolynomialFeatures(degree=2, interaction_only=True)` | `total_views` †, `total_likes` †, `total_comments` †, `avg_sentiment` †, `platform_presence` | LOW |

> **Columns marked `†`** (`total_views`, `total_likes`, `total_comments`, `avg_sentiment`) are **not produced** by the aggregation pipeline in their unprefixed form. The `create_interaction_features` method checks `available_features = [f for f in key_features if f in df.columns]` (line 480) and returns an empty DataFrame if fewer than 2 are present. In practice, this means **features #85–#94 are never created** — only `platform_presence` exists among the required columns, and 1 < 2. This entire method is dead code in the current pipeline.

### 2.10 Platform Presence (`feature_pipeline.py`, lines 110–114)

| # | Feature Name | Math / Logic | Dependencies | Leakage |
|---|---|---|---|---|
| 95 | `platform_presence` | $\mathbb{1}[\text{twitch\_total\_viewers} > 0] + \mathbb{1}[\text{youtube\_total\_views} > 0] + \mathbb{1}[\text{reddit\_total\_score} > 0]$ | `twitch_total_viewers`, `youtube_total_views`, `reddit_total_score` | LOW |

---

## PART 3: Steam Columns (passthrough)

The `load_and_merge_all_platforms` function (lines 162–179) prefixes all Steam columns with `steam_` and merges them via left join. These are **raw columns**, not engineered features, but they flow into the final feature matrix. All are **LOW** risk (they are per-game attributes from the Steam store API).

---

## Summary: Leakage Findings

### HIGH Risk (3 features — must be redesigned)

| Feature | Global Aggregation Used | Location |
|---|---|---|
| `market_share` | `youtube_total_views.sum()` across all games | `enhanced_features.py:294-296` |
| `platform_dominance` | `total_engagement.sum()` across all games | `enhanced_features.py:306-312` |
| `competitive_score` | Derived from `market_share` and `platform_dominance` | `enhanced_features.py:315-318` |

**Recommended fix:** Replace global `.sum()` with a **pre-computed constant** (e.g., total market size from a prior time window or external benchmark) or use a **window function scoped to a time range** in SQL. The `competitive_score` is automatically fixed when its two constituents are fixed.

### MEDIUM Risk (4 features — conditional on columns that don't exist yet)

| Feature | Group-Level Aggregation | Location |
|---|---|---|
| `market_density` | `groupby('genre').transform('count')` | `enhanced_features.py:499` |
| `relative_performance` | `groupby('genre').rank(pct=True)` | `enhanced_features.py:502` |
| `market_share` (genre) | `groupby('genre').transform('sum')` | `enhanced_features.py:505-506` |
| `genre_momentum` | `groupby('genre').transform('mean')` | `enhanced_features.py:509` |

These are currently **not generated** because `genre` and `engagement_score` columns don't exist in the upstream pipeline. If the pipeline is extended to provide these columns, use SQL `PARTITION BY genre` window functions **scoped to a training-time snapshot** to prevent temporal leakage.

### LOW Risk (88+ features)

All per-game aggregation features (#1–#49), all row-local derived features (#50–#54, #56, #59–#80, #95), and Steam passthrough columns are safe to translate to window functions without modification.

### Dead Features (10–14 features)

Features #69–#71, #73, and #85–#94 are **never populated** because their required upstream columns (`total_shares`, `total_watch_time`, `completion_rate`, `total_views`, `total_likes`, `total_comments`, `avg_sentiment`) do not exist in the merge output. They resolve to constant zeros or empty DataFrames. Remove them from the SQL migration scope.

---

## SQL Translation Priority

1. **Phase 1 (safe):** Port all LOW-risk per-platform aggregations (#1–#49) as `GROUP BY game_title` with standard aggregate functions.
2. **Phase 2 (safe):** Port all LOW-risk derived features (#50–#54, #56, #59–#80, #95) as scalar expressions over the Phase 1 columns.
3. **Phase 3 (redesign required):** Replace HIGH-risk global-sum features (#55, #57, #58) with pre-materialized market totals or temporally-scoped window functions.
4. **Phase 4 (defer):** MEDIUM-risk genre features (#81–#84) — only implement if/when `genre` and `engagement_score` enter the pipeline; use `PARTITION BY genre` with a snapshot-time constraint.
5. **Do not migrate:** Dead features (#69–#71, #73, #85–#94).
