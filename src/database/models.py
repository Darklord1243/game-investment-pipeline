"""
SQLAlchemy 2.0 ORM schema for the game-investment ELT pipeline.

Normalized layout:
  - ``Game``: canonical Steam entity (indexed ``appid`` / ``steam_name``).
  - ``*Metric``: one row per mined observation; feature engineering aggregates
    these tables instead of scanning denormalized CSVs.

Ingestion loaders are implemented separately; this module defines schema only.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import TYPE_CHECKING, Any, Final, Optional

from sqlalchemy import (
    create_engine,
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.types import JSON

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Game (dimension / core entity)
# ---------------------------------------------------------------------------


class Game(Base):
    """
    Canonical game record keyed by Steam ``appid``.

    ``steam_name`` and ``appid`` are indexed so ingestion and feature pipelines
    can resolve games in O(log N) time instead of full-table CSV scans.
    """

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    appid: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        unique=True,
        index=True,
        doc="Steam application ID.",
    )
    steam_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        index=True,
        doc="Official Steam store title; joins to legacy ``game_title`` in raw extracts.",
    )
    release_date: Mapped[Optional[dt.date]] = mapped_column(
        Date,
        nullable=True,
        doc="Parsed Steam release date when available.",
    )
    target_variable_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Supervised label for investment-potential models (populated post-labeling).",
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        insert_default=dt.datetime.now,
        nullable=False,
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        insert_default=dt.datetime.now,
        onupdate=dt.datetime.now,
        nullable=False,
    )

    twitch_metrics: Mapped[list[TwitchMetric]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
        order_by="TwitchMetric.mined_at",
    )
    youtube_metrics: Mapped[list[YouTubeMetric]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
        order_by="YouTubeMetric.mined_at",
    )
    reddit_metrics: Mapped[list[RedditMetric]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
        order_by="RedditMetric.mined_at",
    )
    steam_metrics: Mapped[list[SteamMetric]] = relationship(
        back_populates="game",
        cascade="all, delete-orphan",
        order_by="SteamMetric.mined_at",
    )

    def __repr__(self) -> str:
        return f"Game(id={self.id!r}, appid={self.appid!r}, steam_name={self.steam_name!r})"


# ---------------------------------------------------------------------------
# Platform metric tables (facts / observations)
# ---------------------------------------------------------------------------

_RUN_ID_LEN: Final[int] = 36
_PLATFORM_ID_LEN: Final[int] = 128
_TITLE_LEN: Final[int] = 1024
_URL_LEN: Final[int] = 2048
_USERNAME_LEN: Final[int] = 256


class TwitchMetric(Base):
    """
    One Twitch live-stream snapshot for a game.

    Mirrors ``twitch_game_streams.csv``; ``twitch_game_id`` is the Helix category id.
    """

    __tablename__ = "twitch_metrics"
    __table_args__ = (
        Index("ix_twitch_metrics_game_id", "game_id"),
        Index("ix_twitch_metrics_game_id_mined_at", "game_id", "mined_at"),
        Index("ix_twitch_metrics_run_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(String(_RUN_ID_LEN), nullable=False)
    mined_at: Mapped[dt.datetime] = mapped_column(
        nullable=False,
        doc="UTC timestamp when this row was extracted (CSV ``timestamp``).",
    )
    twitch_game_id: Mapped[Optional[str]] = mapped_column(
        String(_PLATFORM_ID_LEN),
        nullable=True,
        doc="Twitch Helix category/game id.",
    )
    streamer_name: Mapped[Optional[str]] = mapped_column(String(_USERNAME_LEN), nullable=True)
    stream_started_at: Mapped[Optional[dt.datetime]] = mapped_column(
        nullable=True,
        doc="ISO-8601 stream start from Twitch (CSV ``started_at``).",
    )
    stream_title: Mapped[Optional[str]] = mapped_column(String(_TITLE_LEN), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    viewer_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    game: Mapped[Game] = relationship(back_populates="twitch_metrics")

    def __repr__(self) -> str:
        return (
            f"TwitchMetric(id={self.id!r}, game_id={self.game_id!r}, "
            f"viewer_count={self.viewer_count!r})"
        )


class YouTubeMetric(Base):
    """
    One YouTube video observation for a game.

    Holds view/engagement counters and comment-level sentiment aggregates used
    by the feature-engineering pipeline (``youtube_*`` prefixed features).
    """

    __tablename__ = "youtube_metrics"
    __table_args__ = (
        UniqueConstraint("video_id", name="uq_youtube_metrics_video_id"),
        Index("ix_youtube_metrics_game_id", "game_id"),
        Index("ix_youtube_metrics_game_id_mined_at", "game_id", "mined_at"),
        Index("ix_youtube_metrics_video_id", "video_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
    )
    video_id: Mapped[str] = mapped_column(String(32), nullable=False)
    mined_at: Mapped[dt.datetime] = mapped_column(
        insert_default=dt.datetime.now,
        nullable=False,
        doc="When this video row was mined (no batch run_id in legacy CSV).",
    )
    title: Mapped[Optional[str]] = mapped_column(String(_TITLE_LEN), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[Optional[dt.datetime]] = mapped_column(nullable=True)
    duration_iso8601: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        doc="YouTube contentDetails.duration (e.g. PT16M44S).",
    )
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    view_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    like_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dislike_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comment_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channel_title: Mapped[Optional[str]] = mapped_column(String(_TITLE_LEN), nullable=True)
    channel_subscriber_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channel_video_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channel_view_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_comment_sentiment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pos_comment_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    neg_comment_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(_URL_LEN), nullable=True)

    game: Mapped[Game] = relationship(back_populates="youtube_metrics")

    def __repr__(self) -> str:
        return (
            f"YouTubeMetric(id={self.id!r}, game_id={self.game_id!r}, "
            f"video_id={self.video_id!r})"
        )


class RedditMetric(Base):
    """
    One Reddit post observation for a game.

    Captures post score, comment sentiment ratios, and emotion distributions
    produced by the HuggingFace emotion classifier in the Reddit miner.
    """

    __tablename__ = "reddit_metrics"
    __table_args__ = (
        UniqueConstraint("post_id", name="uq_reddit_metrics_post_id"),
        Index("ix_reddit_metrics_game_id", "game_id"),
        Index("ix_reddit_metrics_game_id_mined_at", "game_id", "mined_at"),
        Index("ix_reddit_metrics_run_id", "run_id"),
        Index("ix_reddit_metrics_post_id", "post_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(String(_RUN_ID_LEN), nullable=False)
    mined_at: Mapped[dt.datetime] = mapped_column(
        nullable=False,
        doc="UTC batch timestamp (CSV ``timestamp``).",
    )
    subreddit: Mapped[str] = mapped_column(String(128), nullable=False)
    post_id: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(_TITLE_LEN), nullable=True)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_comments: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_utc: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Reddit post.created_utc (Unix epoch seconds).",
    )
    avg_comment_sentiment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pos_comment_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    neg_comment_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    analyzed_comment_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Number of comments sampled for sentiment (CSV ``comment_count``).",
    )
    post_url: Mapped[Optional[str]] = mapped_column(String(_URL_LEN), nullable=True)
    author_username: Mapped[Optional[str]] = mapped_column(String(_USERNAME_LEN), nullable=True)
    author_link_karma: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    author_comment_karma: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    author_account_age_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_awards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_crossposts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    post_flair: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    is_stickied: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_original_content: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    unique_commenters: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    emotion_distribution: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Label -> share mapping from the emotion classifier.",
    )

    game: Mapped[Game] = relationship(back_populates="reddit_metrics")

    def __repr__(self) -> str:
        return (
            f"RedditMetric(id={self.id!r}, game_id={self.game_id!r}, "
            f"post_id={self.post_id!r})"
        )


class SteamMetric(Base):
    """
    One Steam enrichment snapshot for a game (time series).

    Repeated observations per ``game_id`` are intentional — no UNIQUE on natural
    keys. Phase 2 review-velocity labels query ``(game_id, mined_at)`` windows.
    """

    __tablename__ = "steam_metrics"
    __table_args__ = (
        Index("ix_steam_metrics_game_id", "game_id"),
        Index("ix_steam_metrics_game_id_mined_at", "game_id", "mined_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(
        ForeignKey("games.id", ondelete="CASCADE"),
        nullable=False,
    )
    mined_at: Mapped[dt.datetime] = mapped_column(
        nullable=False,
        doc="UTC timestamp when this row was persisted.",
    )
    total_reviews: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    positive_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    review_sentiment: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="VADER compound score from recent Steam reviews.",
    )
    current_players: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    wishlist_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    supported_languages: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    game: Mapped[Game] = relationship(back_populates="steam_metrics")

    def __repr__(self) -> str:
        return (
            f"SteamMetric(id={self.id!r}, game_id={self.game_id!r}, "
            f"total_reviews={self.total_reviews!r})"
        )


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

DATABASE_URL: Final[str] = os.getenv("DATABASE_URL", "sqlite:///data/game_metrics.db")


def _ensure_sqlite_directory(url: str) -> None:
    if not url.startswith("sqlite:///"):
        return
    db_path = url.removeprefix("sqlite:///")
    if db_path in (":memory:", ""):
        return
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


_ensure_sqlite_directory(DATABASE_URL)
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__: Sequence[str] = [
    "Base",
    "DATABASE_URL",
    "Game",
    "RedditMetric",
    "SessionLocal",
    "SteamMetric",
    "TwitchMetric",
    "YouTubeMetric",
    "engine",
]
