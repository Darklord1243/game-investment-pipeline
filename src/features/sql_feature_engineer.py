"""
SQL-native feature engineering for the game-investment ELT pipeline.

Translates FEATURE_MATHEMATICS_AUDIT.md formulas into a single CTE-based query
executed in the database engine. Dead features (#69-71, #73, #81-94) are omitted.
HIGH-risk global sums (#55, #57, #58) use batch-scoped window denominators.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import TYPE_CHECKING, Any, Final, Sequence

import pandas as pd
from sqlalchemy import Float, Select, case, func, literal, select, union
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from src.database.models import Game, RedditMetric, TwitchMetric, YouTubeMetric

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema (audit feature numbers in comments)
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "appid",
    "steam_name",
    "batch_date",
)

PHASE1_COLUMNS: Final[tuple[str, ...]] = (
    # Twitch #1-7
    "twitch_total_viewers",
    "twitch_avg_viewers",
    "twitch_max_viewers",
    "twitch_viewer_std",
    "twitch_stream_count",
    "twitch_unique_streamers",
    "twitch_viewer_cv",
    # YouTube #8-26
    "youtube_total_views",
    "youtube_avg_views",
    "youtube_max_views",
    "youtube_view_std",
    "youtube_video_count",
    "youtube_total_likes",
    "youtube_avg_likes",
    "youtube_max_likes",
    "youtube_total_comments",
    "youtube_avg_comments",
    "youtube_max_comments",
    "youtube_avg_sentiment",
    "youtube_sentiment_std",
    "youtube_pos_ratio",
    "youtube_pos_ratio_std",
    "youtube_total_subscribers",
    "youtube_avg_subscribers",
    "youtube_max_subscribers",
    "youtube_engagement_rate",
    # Reddit #27-49
    "reddit_total_score",
    "reddit_avg_score",
    "reddit_max_score",
    "reddit_score_std",
    "reddit_post_count",
    "reddit_total_comments",
    "reddit_avg_comments",
    "reddit_max_comments",
    "reddit_avg_sentiment",
    "reddit_sentiment_std",
    "reddit_pos_ratio",
    "reddit_pos_ratio_std",
    "reddit_avg_author_karma",
    "reddit_max_author_karma",
    "reddit_avg_comment_karma",
    "reddit_max_comment_karma",
    "reddit_total_commenters",
    "reddit_avg_commenters",
    "reddit_max_commenters",
    "reddit_total_awards",
    "reddit_avg_awards",
    "reddit_max_awards",
    "reddit_engagement_rate",
)

PHASE2_COLUMNS: Final[tuple[str, ...]] = (
    # Viral #50-54
    "viral_velocity",
    "shareability_score",
    "cross_platform_viral",
    "engagement_amplification",
    "viral_potential_score",
    # Competitive #56 (LOW); #55, #57-58 added in final phase
    "competitive_advantage",
    # Cross-platform #59-62
    "cross_platform_engagement_rate",
    "platform_synergy",
    "cross_platform_reach",
    "platform_balance",
    # Temporal #63-66
    "growth_momentum",
    "activity_consistency",
    "engagement_trend",
    "peak_activity",
    # Engagement quality #67-68, #72 (omit #69-71, #73)
    "like_per_view",
    "comment_per_view",
    "subscriber_engagement",
    # Sentiment #74-76
    "sentiment_volatility",
    "sentiment_trend",
    "sentiment_authenticity",
    # Network #77-80
    "avg_creator_influence",
    "creator_diversity",
    "community_cohesion",
    "creator_concentration",
    # Platform presence #95
    "platform_presence",
)

PHASE3_COLUMNS: Final[tuple[str, ...]] = (
    "market_share",  # #55 — batch-scoped window
    "platform_dominance",  # #57
    "competitive_score",  # #58
)

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *PHASE1_COLUMNS,
    *PHASE2_COLUMNS,
    *PHASE3_COLUMNS,
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    *IDENTIFIER_COLUMNS,
    *FEATURE_COLUMNS,
)

_INT64_COLS: Final[frozenset[str]] = frozenset(
    {
        "game_id",
        "appid",
        "twitch_total_viewers",
        "twitch_max_viewers",
        "twitch_stream_count",
        "twitch_unique_streamers",
        "youtube_total_views",
        "youtube_max_views",
        "youtube_video_count",
        "youtube_total_likes",
        "youtube_max_likes",
        "youtube_total_comments",
        "youtube_max_comments",
        "youtube_total_subscribers",
        "youtube_max_subscribers",
        "reddit_total_score",
        "reddit_max_score",
        "reddit_post_count",
        "reddit_total_comments",
        "reddit_max_comments",
        "reddit_max_author_karma",
        "reddit_max_comment_karma",
        "reddit_total_commenters",
        "reddit_max_commenters",
        "reddit_total_awards",
        "reddit_max_awards",
        "creator_diversity",
        "platform_presence",
    }
)

_FLOAT64_COLS: Final[frozenset[str]] = frozenset(
    {col for col in OUTPUT_COLUMNS if col not in _INT64_COLS}
)


def _utc_day_bounds(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Inclusive start and exclusive end for a UTC calendar day (matches miners)."""
    start = dt.datetime.combine(day, dt.time.min)
    return start, start + dt.timedelta(days=1)


def _ensure_sqlite_math_functions(session: Session) -> None:
    """
    Register ``sqrt`` and ``power`` on SQLite connections.

    Core SQLite lacks these builtins; the feature SQL relies on them for std,
    cbrt, and platform-balance expressions.
    """
    bind = session.get_bind()
    if bind is None or bind.dialect.name != "sqlite":
        return

    def _sqrt(value: float | None) -> float | None:
        if value is None:
            return None
        return math.sqrt(float(value))

    def _power(base: float | None, exponent: float | None) -> float | None:
        if base is None or exponent is None:
            return None
        return math.pow(float(base), float(exponent))

    raw = session.connection().connection
    raw.create_function("sqrt", 1, _sqrt)
    raw.create_function("power", 2, _power)


def _coalesce0(expr: ColumnElement[Any]) -> ColumnElement[Any]:
    return func.coalesce(expr, 0.0)


def _scalar_max(
    left: ColumnElement[Any],
    right: ColumnElement[Any] | float,
) -> ColumnElement[Any]:
    """Portable two-argument max (SQLite lacks ``GREATEST``)."""
    if isinstance(right, (int, float)):
        right_val: ColumnElement[Any] = literal(float(right))
    else:
        right_val = right
    return case((left > right_val, left), else_=right_val)


def _pop_stddev(column: ColumnElement[Any]) -> ColumnElement[Any]:
    """
    Population standard deviation (numpy ddof=0).

    Returns 0 when fewer than two non-null observations, matching pandas
    ``groupby().agg('std').fillna(0)`` on sparse game-day slices.
    """
    col_f = column.cast(Float)
    cnt = func.count(col_f)
    mean_expr = func.avg(col_f)
    mean_sq = func.avg(col_f * col_f)
    variance = mean_sq - mean_expr * mean_expr
    return case(
        (cnt < 2, literal(0.0)),
        else_=func.sqrt(_scalar_max(variance, 0.0)),
    )


def _greatest1(expr: ColumnElement[Any]) -> ColumnElement[Any]:
    return _scalar_max(_coalesce0(expr), 1.0)


def _platform_balance(
    youtube_engagement_rate: ColumnElement[Any],
    reddit_engagement_rate: ColumnElement[Any],
    twitch_viewer_cv: ColumnElement[Any],
) -> ColumnElement[Any]:
    """Row-local std/mean across three platform engagement scalars (#62)."""
    e_yt = _coalesce0(youtube_engagement_rate)
    e_rd = _coalesce0(reddit_engagement_rate)
    e_tw = _coalesce0(twitch_viewer_cv)
    eng_mean = (e_yt + e_rd + e_tw) / 3.0
    eng_std = func.sqrt(
        (
            func.power(e_yt - eng_mean, 2)
            + func.power(e_rd - eng_mean, 2)
            + func.power(e_tw - eng_mean, 2)
        )
        / 3.0
    )
    return 1.0 - eng_std / (eng_mean + 1.0)


def _cbrt(expr: ColumnElement[Any]) -> ColumnElement[Any]:
    return func.power(_scalar_max(_coalesce0(expr), 0.0), 1.0 / 3.0)


class DatabaseFeatureEngineer:
    """
    Build a leakage-aware per-game feature matrix for one UTC batch day.

    All aggregations run as SQL CTEs; a single round-trip returns the matrix.
    """

    def build_feature_matrix(self, session: Session, batch_date: dt.date) -> pd.DataFrame:
        """
        Execute the CTE pipeline for ``batch_date`` and return a typed DataFrame.

        Args:
            session: SQLAlchemy 2.0 ``Session`` bound to the metrics database.
            batch_date: UTC calendar day to aggregate (matches miner batching).

        Returns:
            DataFrame with identifier columns plus audit features #1-68, #72,
            #74-80, #95, and batch-scoped #55, #57-58. Empty when no metrics
            exist for the day.
        """
        day_start, day_end = _utc_day_bounds(batch_date)
        _ensure_sqlite_math_functions(session)
        stmt = self._build_statement(batch_date, day_start, day_end)
        logger.debug(
            "Executing feature matrix query for batch_date=%s [%s, %s)",
            batch_date,
            day_start,
            day_end,
        )
        rows = session.execute(stmt).mappings().all()
        return self._to_typed_dataframe(rows, batch_date)

    def _build_statement(
        self,
        batch_date: dt.date,
        day_start: dt.datetime,
        day_end: dt.datetime,
    ) -> Select[Any]:
        """Compose the full CTE query (Phases 1-3)."""
        twitch_agg = self._twitch_agg_cte(day_start, day_end)
        youtube_agg = self._youtube_agg_cte(day_start, day_end)
        reddit_agg = self._reddit_agg_cte(day_start, day_end)

        game_ids = union(
            select(twitch_agg.c.game_id),
            select(youtube_agg.c.game_id),
            select(reddit_agg.c.game_id),
        ).subquery("game_ids_union")

        batch_literal = literal(batch_date)

        base = (
            select(
                Game.id.label("game_id"),
                Game.appid,
                Game.steam_name,
                batch_literal.label("batch_date"),
                # Twitch #1-6 raw aggregates
                _coalesce0(twitch_agg.c.twitch_total_viewers).label("twitch_total_viewers"),
                _coalesce0(twitch_agg.c.twitch_avg_viewers).label("twitch_avg_viewers"),
                _coalesce0(twitch_agg.c.twitch_max_viewers).label("twitch_max_viewers"),
                _coalesce0(twitch_agg.c.twitch_viewer_std).label("twitch_viewer_std"),
                _coalesce0(twitch_agg.c.twitch_stream_count).label("twitch_stream_count"),
                _coalesce0(twitch_agg.c.twitch_unique_streamers).label(
                    "twitch_unique_streamers"
                ),
                # YouTube #8-25
                _coalesce0(youtube_agg.c.youtube_total_views).label("youtube_total_views"),
                _coalesce0(youtube_agg.c.youtube_avg_views).label("youtube_avg_views"),
                _coalesce0(youtube_agg.c.youtube_max_views).label("youtube_max_views"),
                _coalesce0(youtube_agg.c.youtube_view_std).label("youtube_view_std"),
                _coalesce0(youtube_agg.c.youtube_video_count).label("youtube_video_count"),
                _coalesce0(youtube_agg.c.youtube_total_likes).label("youtube_total_likes"),
                _coalesce0(youtube_agg.c.youtube_avg_likes).label("youtube_avg_likes"),
                _coalesce0(youtube_agg.c.youtube_max_likes).label("youtube_max_likes"),
                _coalesce0(
                    youtube_agg.c.youtube_total_comments
                ).label("youtube_total_comments"),
                _coalesce0(youtube_agg.c.youtube_avg_comments).label("youtube_avg_comments"),
                _coalesce0(youtube_agg.c.youtube_max_comments).label("youtube_max_comments"),
                _coalesce0(youtube_agg.c.youtube_avg_sentiment).label(
                    "youtube_avg_sentiment"
                ),
                _coalesce0(youtube_agg.c.youtube_sentiment_std).label(
                    "youtube_sentiment_std"
                ),
                _coalesce0(youtube_agg.c.youtube_pos_ratio).label("youtube_pos_ratio"),
                _coalesce0(youtube_agg.c.youtube_pos_ratio_std).label(
                    "youtube_pos_ratio_std"
                ),
                _coalesce0(youtube_agg.c.youtube_total_subscribers).label(
                    "youtube_total_subscribers"
                ),
                _coalesce0(youtube_agg.c.youtube_avg_subscribers).label(
                    "youtube_avg_subscribers"
                ),
                _coalesce0(youtube_agg.c.youtube_max_subscribers).label(
                    "youtube_max_subscribers"
                ),
                # Reddit #27-48
                _coalesce0(reddit_agg.c.reddit_total_score).label("reddit_total_score"),
                _coalesce0(reddit_agg.c.reddit_avg_score).label("reddit_avg_score"),
                _coalesce0(reddit_agg.c.reddit_max_score).label("reddit_max_score"),
                _coalesce0(reddit_agg.c.reddit_score_std).label("reddit_score_std"),
                _coalesce0(reddit_agg.c.reddit_post_count).label("reddit_post_count"),
                _coalesce0(reddit_agg.c.reddit_total_comments).label(
                    "reddit_total_comments"
                ),
                _coalesce0(reddit_agg.c.reddit_avg_comments).label("reddit_avg_comments"),
                _coalesce0(reddit_agg.c.reddit_max_comments).label("reddit_max_comments"),
                _coalesce0(reddit_agg.c.reddit_avg_sentiment).label("reddit_avg_sentiment"),
                _coalesce0(reddit_agg.c.reddit_sentiment_std).label("reddit_sentiment_std"),
                _coalesce0(reddit_agg.c.reddit_pos_ratio).label("reddit_pos_ratio"),
                _coalesce0(reddit_agg.c.reddit_pos_ratio_std).label("reddit_pos_ratio_std"),
                _coalesce0(reddit_agg.c.reddit_avg_author_karma).label(
                    "reddit_avg_author_karma"
                ),
                _coalesce0(reddit_agg.c.reddit_max_author_karma).label(
                    "reddit_max_author_karma"
                ),
                _coalesce0(reddit_agg.c.reddit_avg_comment_karma).label(
                    "reddit_avg_comment_karma"
                ),
                _coalesce0(reddit_agg.c.reddit_max_comment_karma).label(
                    "reddit_max_comment_karma"
                ),
                _coalesce0(reddit_agg.c.reddit_total_commenters).label(
                    "reddit_total_commenters"
                ),
                _coalesce0(reddit_agg.c.reddit_avg_commenters).label(
                    "reddit_avg_commenters"
                ),
                _coalesce0(reddit_agg.c.reddit_max_commenters).label(
                    "reddit_max_commenters"
                ),
                _coalesce0(reddit_agg.c.reddit_total_awards).label("reddit_total_awards"),
                _coalesce0(reddit_agg.c.reddit_avg_awards).label("reddit_avg_awards"),
                _coalesce0(reddit_agg.c.reddit_max_awards).label("reddit_max_awards"),
            )
            .select_from(Game)
            .join(game_ids, Game.id == game_ids.c.game_id)
            .outerjoin(twitch_agg, Game.id == twitch_agg.c.game_id)
            .outerjoin(youtube_agg, Game.id == youtube_agg.c.game_id)
            .outerjoin(reddit_agg, Game.id == reddit_agg.c.game_id)
        ).cte("base_joined")

        b = base.c
        twitch_viewer_cv = _coalesce0(b.twitch_viewer_std) / (
            _coalesce0(b.twitch_avg_viewers) + 1.0
        )
        youtube_engagement_rate = (
            _coalesce0(b.youtube_total_likes) + _coalesce0(b.youtube_total_comments)
        ) / (_coalesce0(b.youtube_total_views) + 1.0)
        reddit_engagement_rate = (
            _coalesce0(b.reddit_total_comments) + _coalesce0(b.reddit_total_awards)
        ) / (_coalesce0(b.reddit_post_count) + 1.0)

        derived = (
            select(
                b.game_id,
                b.appid,
                b.steam_name,
                b.batch_date,
                # Phase 1 derived rates #7, #26, #49
                twitch_viewer_cv.label("twitch_viewer_cv"),
                youtube_engagement_rate.label("youtube_engagement_rate"),
                reddit_engagement_rate.label("reddit_engagement_rate"),
                # Pass-through Phase 1 scalars needed downstream
                b.twitch_total_viewers,
                b.twitch_avg_viewers,
                b.twitch_max_viewers,
                b.twitch_viewer_std,
                b.twitch_stream_count,
                b.twitch_unique_streamers,
                b.youtube_total_views,
                b.youtube_avg_views,
                b.youtube_max_views,
                b.youtube_view_std,
                b.youtube_video_count,
                b.youtube_total_likes,
                b.youtube_avg_likes,
                b.youtube_max_likes,
                b.youtube_total_comments,
                b.youtube_avg_comments,
                b.youtube_max_comments,
                b.youtube_avg_sentiment,
                b.youtube_sentiment_std,
                b.youtube_pos_ratio,
                b.youtube_pos_ratio_std,
                b.youtube_total_subscribers,
                b.youtube_avg_subscribers,
                b.youtube_max_subscribers,
                b.reddit_total_score,
                b.reddit_avg_score,
                b.reddit_max_score,
                b.reddit_score_std,
                b.reddit_post_count,
                b.reddit_total_comments,
                b.reddit_avg_comments,
                b.reddit_max_comments,
                b.reddit_avg_sentiment,
                b.reddit_sentiment_std,
                b.reddit_pos_ratio,
                b.reddit_pos_ratio_std,
                b.reddit_avg_author_karma,
                b.reddit_max_author_karma,
                b.reddit_avg_comment_karma,
                b.reddit_max_comment_karma,
                b.reddit_total_commenters,
                b.reddit_avg_commenters,
                b.reddit_max_commenters,
                b.reddit_total_awards,
                b.reddit_avg_awards,
                b.reddit_max_awards,
            )
        ).cte("derived_rates")

        d = derived.c
        yt_views = _coalesce0(d.youtube_total_views)
        yt_likes = _coalesce0(d.youtube_total_likes)
        yt_comments = _coalesce0(d.youtube_total_comments)
        yt_subs = _coalesce0(d.youtube_total_subscribers)
        tw_viewers = _coalesce0(d.twitch_total_viewers)
        rd_score = _coalesce0(d.reddit_total_score)
        yt_er = _coalesce0(d.youtube_engagement_rate)
        rd_er = _coalesce0(d.reddit_engagement_rate)
        tw_cv = _coalesce0(d.twitch_viewer_cv)

        viral_velocity = _cbrt(yt_views * yt_er * rd_score)
        shareability_score = yt_likes + rd_score + tw_viewers * 0.1
        cross_platform_viral = yt_views * 0.001 + tw_viewers * 0.1 + rd_score
        engagement_amplification = (yt_er + rd_er + tw_cv) / 3.0
        viral_potential_score = (
            viral_velocity * 0.3
            + shareability_score * 0.0001 * 0.3
            + cross_platform_viral * 0.0001 * 0.2
            + engagement_amplification * 0.2
        )
        competitive_advantage = engagement_amplification
        cross_platform_engagement_rate = engagement_amplification
        platform_synergy = _cbrt(yt_views * 0.001 * tw_viewers * 0.1 * rd_score)
        cross_platform_reach = cross_platform_viral
        platform_balance = _platform_balance(yt_er, rd_er, tw_cv)

        growth_momentum = (
            yt_views / _greatest1(d.youtube_video_count)
            + tw_viewers / _greatest1(d.twitch_stream_count)
            + rd_score / _greatest1(d.reddit_post_count)
        ) / 3.0
        activity_consistency = 1.0 / (
            _coalesce0(d.youtube_view_std) / (_coalesce0(d.youtube_avg_views) + 1.0)
            + _coalesce0(d.twitch_viewer_std) / (_coalesce0(d.twitch_avg_viewers) + 1.0)
            + _coalesce0(d.reddit_score_std) / (_coalesce0(d.reddit_avg_score) + 1.0)
            + 1.0
        )
        engagement_trend = (
            _coalesce0(d.youtube_video_count) * yt_er
            + _coalesce0(d.twitch_stream_count) * tw_cv
            + _coalesce0(d.reddit_post_count) * rd_er
        ) / 3.0
        peak_activity = (
            _coalesce0(d.youtube_max_views) * 0.001
            + _coalesce0(d.twitch_max_viewers) * 0.1
            + _coalesce0(d.reddit_max_score)
        )

        like_per_view = yt_likes / _greatest1(yt_views)
        comment_per_view = yt_comments / _greatest1(yt_views)
        subscriber_engagement = yt_likes / _greatest1(yt_subs)

        sentiment_volatility = _coalesce0(d.youtube_sentiment_std)
        sentiment_trend = _coalesce0(d.youtube_avg_sentiment)
        sentiment_authenticity = 1.0 - 2.0 * func.abs(
            _coalesce0(d.youtube_pos_ratio) - 0.5
        )

        avg_creator_influence = _coalesce0(d.youtube_avg_subscribers)
        creator_diversity = _coalesce0(d.twitch_unique_streamers)
        community_cohesion = _coalesce0(d.reddit_total_commenters) / _greatest1(
            _coalesce0(d.reddit_total_comments)
        )
        creator_concentration = _coalesce0(d.youtube_max_subscribers) / _greatest1(
            _coalesce0(d.youtube_avg_subscribers)
        )

        platform_presence = (
            case((tw_viewers > 0, 1), else_=0)
            + case((yt_views > 0, 1), else_=0)
            + case((rd_score > 0, 1), else_=0)
        )

        composite_engagement = tw_viewers + yt_views * 0.001 + rd_score

        phase2 = (
            select(
                d.game_id,
                d.appid,
                d.steam_name,
                d.batch_date,
                d.twitch_total_viewers,
                d.twitch_avg_viewers,
                d.twitch_max_viewers,
                d.twitch_viewer_std,
                d.twitch_stream_count,
                d.twitch_unique_streamers,
                d.twitch_viewer_cv,
                d.youtube_total_views,
                d.youtube_avg_views,
                d.youtube_max_views,
                d.youtube_view_std,
                d.youtube_video_count,
                d.youtube_total_likes,
                d.youtube_avg_likes,
                d.youtube_max_likes,
                d.youtube_total_comments,
                d.youtube_avg_comments,
                d.youtube_max_comments,
                d.youtube_avg_sentiment,
                d.youtube_sentiment_std,
                d.youtube_pos_ratio,
                d.youtube_pos_ratio_std,
                d.youtube_total_subscribers,
                d.youtube_avg_subscribers,
                d.youtube_max_subscribers,
                d.youtube_engagement_rate,
                d.reddit_total_score,
                d.reddit_avg_score,
                d.reddit_max_score,
                d.reddit_score_std,
                d.reddit_post_count,
                d.reddit_total_comments,
                d.reddit_avg_comments,
                d.reddit_max_comments,
                d.reddit_avg_sentiment,
                d.reddit_sentiment_std,
                d.reddit_pos_ratio,
                d.reddit_pos_ratio_std,
                d.reddit_avg_author_karma,
                d.reddit_max_author_karma,
                d.reddit_avg_comment_karma,
                d.reddit_max_comment_karma,
                d.reddit_total_commenters,
                d.reddit_avg_commenters,
                d.reddit_max_commenters,
                d.reddit_total_awards,
                d.reddit_avg_awards,
                d.reddit_max_awards,
                d.reddit_engagement_rate,
                viral_velocity.label("viral_velocity"),
                shareability_score.label("shareability_score"),
                cross_platform_viral.label("cross_platform_viral"),
                engagement_amplification.label("engagement_amplification"),
                viral_potential_score.label("viral_potential_score"),
                competitive_advantage.label("competitive_advantage"),
                cross_platform_engagement_rate.label("cross_platform_engagement_rate"),
                platform_synergy.label("platform_synergy"),
                cross_platform_reach.label("cross_platform_reach"),
                platform_balance.label("platform_balance"),
                growth_momentum.label("growth_momentum"),
                activity_consistency.label("activity_consistency"),
                engagement_trend.label("engagement_trend"),
                peak_activity.label("peak_activity"),
                like_per_view.label("like_per_view"),
                comment_per_view.label("comment_per_view"),
                subscriber_engagement.label("subscriber_engagement"),
                sentiment_volatility.label("sentiment_volatility"),
                sentiment_trend.label("sentiment_trend"),
                sentiment_authenticity.label("sentiment_authenticity"),
                avg_creator_influence.label("avg_creator_influence"),
                creator_diversity.label("creator_diversity"),
                community_cohesion.label("community_cohesion"),
                creator_concentration.label("creator_concentration"),
                platform_presence.label("platform_presence"),
                composite_engagement.label("_composite_engagement"),
            )
        ).cte("phase2_derived")

        p = phase2.c
        batch_views_total = func.sum(p.youtube_total_views).over(
            partition_by=p.batch_date
        )
        batch_engagement_total = func.sum(p._composite_engagement).over(
            partition_by=p.batch_date
        )

        market_share = p.youtube_total_views / (batch_views_total + 1.0)
        platform_dominance = p._composite_engagement / (batch_engagement_total + 1.0)
        competitive_score = (
            market_share * 0.4
            + p.competitive_advantage * 0.3
            + platform_dominance * 0.3
        )

        return select(
            p.game_id,
            p.appid,
            p.steam_name,
            p.batch_date,
            p.twitch_total_viewers,
            p.twitch_avg_viewers,
            p.twitch_max_viewers,
            p.twitch_viewer_std,
            p.twitch_stream_count,
            p.twitch_unique_streamers,
            p.twitch_viewer_cv,
            p.youtube_total_views,
            p.youtube_avg_views,
            p.youtube_max_views,
            p.youtube_view_std,
            p.youtube_video_count,
            p.youtube_total_likes,
            p.youtube_avg_likes,
            p.youtube_max_likes,
            p.youtube_total_comments,
            p.youtube_avg_comments,
            p.youtube_max_comments,
            p.youtube_avg_sentiment,
            p.youtube_sentiment_std,
            p.youtube_pos_ratio,
            p.youtube_pos_ratio_std,
            p.youtube_total_subscribers,
            p.youtube_avg_subscribers,
            p.youtube_max_subscribers,
            p.youtube_engagement_rate,
            p.reddit_total_score,
            p.reddit_avg_score,
            p.reddit_max_score,
            p.reddit_score_std,
            p.reddit_post_count,
            p.reddit_total_comments,
            p.reddit_avg_comments,
            p.reddit_max_comments,
            p.reddit_avg_sentiment,
            p.reddit_sentiment_std,
            p.reddit_pos_ratio,
            p.reddit_pos_ratio_std,
            p.reddit_avg_author_karma,
            p.reddit_max_author_karma,
            p.reddit_avg_comment_karma,
            p.reddit_max_comment_karma,
            p.reddit_total_commenters,
            p.reddit_avg_commenters,
            p.reddit_max_commenters,
            p.reddit_total_awards,
            p.reddit_avg_awards,
            p.reddit_max_awards,
            p.reddit_engagement_rate,
            p.viral_velocity,
            p.shareability_score,
            p.cross_platform_viral,
            p.engagement_amplification,
            p.viral_potential_score,
            p.competitive_advantage,
            p.cross_platform_engagement_rate,
            p.platform_synergy,
            p.cross_platform_reach,
            p.platform_balance,
            p.growth_momentum,
            p.activity_consistency,
            p.engagement_trend,
            p.peak_activity,
            p.like_per_view,
            p.comment_per_view,
            p.subscriber_engagement,
            p.sentiment_volatility,
            p.sentiment_trend,
            p.sentiment_authenticity,
            p.avg_creator_influence,
            p.creator_diversity,
            p.community_cohesion,
            p.creator_concentration,
            p.platform_presence,
            market_share.label("market_share"),
            platform_dominance.label("platform_dominance"),
            competitive_score.label("competitive_score"),
        ).order_by(p.game_id)

    @staticmethod
    def _twitch_agg_cte(day_start: dt.datetime, day_end: dt.datetime) -> Any:
        """Phase 1 Twitch aggregates (#1-6); #7 computed in base join."""
        day_filter = (
            TwitchMetric.mined_at >= day_start,
            TwitchMetric.mined_at < day_end,
        )
        return (
            select(
                TwitchMetric.game_id,
                func.sum(TwitchMetric.viewer_count).label("twitch_total_viewers"),
                func.avg(TwitchMetric.viewer_count.cast(Float)).label("twitch_avg_viewers"),
                func.max(TwitchMetric.viewer_count).label("twitch_max_viewers"),
                _pop_stddev(TwitchMetric.viewer_count).label("twitch_viewer_std"),
                func.count(TwitchMetric.viewer_count).label("twitch_stream_count"),
                func.count(func.distinct(TwitchMetric.streamer_name)).label(
                    "twitch_unique_streamers"
                ),
            )
            .where(*day_filter)
            .group_by(TwitchMetric.game_id)
        ).cte("twitch_agg")

    @staticmethod
    def _youtube_agg_cte(day_start: dt.datetime, day_end: dt.datetime) -> Any:
        """Phase 1 YouTube aggregates (#8-25); #26 computed in base join."""
        day_filter = (
            YouTubeMetric.mined_at >= day_start,
            YouTubeMetric.mined_at < day_end,
        )
        return (
            select(
                YouTubeMetric.game_id,
                func.sum(YouTubeMetric.view_count).label("youtube_total_views"),
                func.avg(YouTubeMetric.view_count.cast(Float)).label("youtube_avg_views"),
                func.max(YouTubeMetric.view_count).label("youtube_max_views"),
                _pop_stddev(YouTubeMetric.view_count).label("youtube_view_std"),
                func.count(YouTubeMetric.view_count).label("youtube_video_count"),
                func.sum(YouTubeMetric.like_count).label("youtube_total_likes"),
                func.avg(YouTubeMetric.like_count.cast(Float)).label("youtube_avg_likes"),
                func.max(YouTubeMetric.like_count).label("youtube_max_likes"),
                func.sum(YouTubeMetric.comment_count).label("youtube_total_comments"),
                func.avg(YouTubeMetric.comment_count.cast(Float)).label(
                    "youtube_avg_comments"
                ),
                func.max(YouTubeMetric.comment_count).label("youtube_max_comments"),
                func.avg(YouTubeMetric.avg_comment_sentiment).label(
                    "youtube_avg_sentiment"
                ),
                _pop_stddev(YouTubeMetric.avg_comment_sentiment).label(
                    "youtube_sentiment_std"
                ),
                func.avg(YouTubeMetric.pos_comment_ratio).label("youtube_pos_ratio"),
                _pop_stddev(YouTubeMetric.pos_comment_ratio).label(
                    "youtube_pos_ratio_std"
                ),
                func.sum(YouTubeMetric.channel_subscriber_count).label(
                    "youtube_total_subscribers"
                ),
                func.avg(YouTubeMetric.channel_subscriber_count.cast(Float)).label(
                    "youtube_avg_subscribers"
                ),
                func.max(YouTubeMetric.channel_subscriber_count).label(
                    "youtube_max_subscribers"
                ),
            )
            .where(*day_filter)
            .group_by(YouTubeMetric.game_id)
        ).cte("youtube_agg")

    @staticmethod
    def _reddit_agg_cte(day_start: dt.datetime, day_end: dt.datetime) -> Any:
        """Phase 1 Reddit aggregates (#27-48); #49 computed in base join."""
        day_filter = (
            RedditMetric.mined_at >= day_start,
            RedditMetric.mined_at < day_end,
        )
        return (
            select(
                RedditMetric.game_id,
                func.sum(RedditMetric.score).label("reddit_total_score"),
                func.avg(RedditMetric.score.cast(Float)).label("reddit_avg_score"),
                func.max(RedditMetric.score).label("reddit_max_score"),
                _pop_stddev(RedditMetric.score).label("reddit_score_std"),
                func.count(RedditMetric.score).label("reddit_post_count"),
                func.sum(RedditMetric.num_comments).label("reddit_total_comments"),
                func.avg(RedditMetric.num_comments.cast(Float)).label(
                    "reddit_avg_comments"
                ),
                func.max(RedditMetric.num_comments).label("reddit_max_comments"),
                func.avg(RedditMetric.avg_comment_sentiment).label(
                    "reddit_avg_sentiment"
                ),
                _pop_stddev(RedditMetric.avg_comment_sentiment).label(
                    "reddit_sentiment_std"
                ),
                func.avg(RedditMetric.pos_comment_ratio).label("reddit_pos_ratio"),
                _pop_stddev(RedditMetric.pos_comment_ratio).label(
                    "reddit_pos_ratio_std"
                ),
                func.avg(RedditMetric.author_link_karma.cast(Float)).label(
                    "reddit_avg_author_karma"
                ),
                func.max(RedditMetric.author_link_karma).label("reddit_max_author_karma"),
                func.avg(RedditMetric.author_comment_karma.cast(Float)).label(
                    "reddit_avg_comment_karma"
                ),
                func.max(RedditMetric.author_comment_karma).label(
                    "reddit_max_comment_karma"
                ),
                func.sum(RedditMetric.unique_commenters).label("reddit_total_commenters"),
                func.avg(RedditMetric.unique_commenters.cast(Float)).label(
                    "reddit_avg_commenters"
                ),
                func.max(RedditMetric.unique_commenters).label("reddit_max_commenters"),
                func.sum(RedditMetric.num_awards).label("reddit_total_awards"),
                func.avg(RedditMetric.num_awards.cast(Float)).label("reddit_avg_awards"),
                func.max(RedditMetric.num_awards).label("reddit_max_awards"),
            )
            .where(*day_filter)
            .group_by(RedditMetric.game_id)
        ).cte("reddit_agg")

    @staticmethod
    def _to_typed_dataframe(
        rows: Sequence[Mapping[str, Any]],
        batch_date: dt.date,
    ) -> pd.DataFrame:
        """Materialize query rows into a strictly typed, column-ordered DataFrame."""
        if not rows:
            empty = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
            for col in _INT64_COLS:
                if col in empty.columns:
                    empty[col] = empty[col].astype("int64")
            for col in _FLOAT64_COLS:
                if col in empty.columns:
                    empty[col] = empty[col].astype("float64")
            if "batch_date" in empty.columns:
                empty["batch_date"] = pd.Series(dtype="datetime64[ns]")
            return empty

        df = pd.DataFrame(rows)
        if "batch_date" in df.columns:
            df["batch_date"] = pd.to_datetime(df["batch_date"]).dt.normalize()

        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df = df.loc[:, list(OUTPUT_COLUMNS)]

        for col in _INT64_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

        for col in _FLOAT64_COLS:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
                )

        if "batch_date" in df.columns:
            df["batch_date"] = pd.to_datetime(batch_date)

        return df


__all__: Sequence[str] = [
    "DatabaseFeatureEngineer",
    "FEATURE_COLUMNS",
    "OUTPUT_COLUMNS",
]
