"""Tests for Steam data miner — mocked APIs only."""

from __future__ import annotations

import datetime as dt
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_collection.steam_data_miner import (  # noqa: E402
    MIN_REVIEWS_FOR_SEED,
    GameSnapshot,
    SteamAPIClient,
    SteamRateLimiter,
    parse_app_details_payload,
    parse_review_summary_payload,
    parse_store_page_html,
    persist_game_snapshot,
    resolve_appid_by_name,
    SteamAppListEntry,
)
from src.database.models import Base, Game, SteamMetric  # noqa: E402


@pytest.fixture()
def db_session():
    """In-memory SQLite session for persistence tests."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    try:
        yield session
        session.commit()
    finally:
        session.close()


def _sample_snapshot(
    *,
    appid: int = 42,
    total_reviews: int = 100,
    positive_rate: float = 85.0,
) -> GameSnapshot:
    return GameSnapshot(
        appid=appid,
        steam_name="Test Game",
        release_date=dt.date(2024, 1, 1),
        total_reviews=total_reviews,
        positive_rate=positive_rate,
        sentiment_score=0.25,
        current_players=500,
        wishlist_count=1_000,
        supported_languages="English",
    )


def test_min_reviews_for_seed_default_is_50():
    assert MIN_REVIEWS_FOR_SEED == 50


def test_qualifies_for_seed_threshold():
    snap = _sample_snapshot(total_reviews=49)
    assert snap.qualifies_for_seed(50) is False
    snap_ok = _sample_snapshot(total_reviews=50)
    assert snap_ok.qualifies_for_seed(50) is True


def test_persist_game_snapshot_inserts_game_and_steam_metric(db_session):
    snapshot = _sample_snapshot()
    inserted = persist_game_snapshot(db_session, snapshot, min_reviews=50)
    assert inserted is True

    game = db_session.scalars(select(Game).where(Game.appid == 42)).one()
    assert game.steam_name == "Test Game"

    metrics = db_session.scalars(select(SteamMetric).where(SteamMetric.game_id == game.id)).all()
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.total_reviews == 100
    assert metric.positive_rate == 85.0
    assert metric.review_sentiment == 0.25
    assert metric.current_players == 500
    assert metric.wishlist_count == 1_000
    assert metric.supported_languages == "English"


def test_persist_game_snapshot_appends_steam_metric_for_existing_game(db_session):
    snapshot = _sample_snapshot()
    assert persist_game_snapshot(db_session, snapshot, min_reviews=50) is True
    assert persist_game_snapshot(db_session, snapshot, min_reviews=50) is False

    games = db_session.scalars(select(Game).where(Game.appid == 42)).all()
    assert len(games) == 1
    metrics = db_session.scalars(
        select(SteamMetric).where(SteamMetric.game_id == games[0].id)
    ).all()
    assert len(metrics) == 2


def test_persist_skips_below_threshold(db_session):
    snapshot = _sample_snapshot(total_reviews=10)
    assert persist_game_snapshot(db_session, snapshot, min_reviews=50) is False
    assert db_session.scalars(select(Game)).first() is None
    assert db_session.scalars(select(SteamMetric)).first() is None


def test_resolve_appid_by_name_fuzzy_match():
    entries = [
        SteamAppListEntry(appid=730, name="Counter-Strike 2"),
        SteamAppListEntry(appid=570, name="Dota 2"),
    ]
    match = resolve_appid_by_name(entries, "counter strike 2")
    assert match is not None
    assert match.appid == 730


def test_parse_review_summary_positive_rate():
    payload = {
        "query_summary": {
            "total_reviews": 200,
            "review_score": 8.5,
        }
    }
    result = parse_review_summary_payload(payload)
    assert result.total_reviews == 200
    assert result.positive_rate == pytest.approx(85.0)


def test_parse_store_page_html_wishlist():
    html = '<div id="WishlistBtn" data-tooltip-html="12,345 people have this on their wishlist"></div>'
    data = parse_store_page_html(html, "English")
    assert data.wishlist_count == 12_345
    assert data.supported_languages == "English"


def test_build_game_snapshot_includes_positive_rate():
    limiter = SteamRateLimiter()
    client = SteamAPIClient(limiter=limiter, session=MagicMock())

    details_payload = {
        "730": {
            "success": True,
            "data": {
                "name": "Counter-Strike 2",
                "release_date": {"date": "21 Aug, 2012"},
                "supported_languages": "English<strong></strong>",
                "dlc": [],
                "achievements": {"total": 1},
            },
        }
    }
    review_payload = {
        "query_summary": {"total_reviews": 500, "review_score": 9.0},
        "reviews": [{"review": "Great game"}],
    }
    players_payload = {"response": {"player_count": 100_000}}
    store_html = '<div id="WishlistBtn" data-tooltip-html="50,000"></div>'

    with patch.object(client, "fetch_app_details") as mock_details:
        mock_details.return_value = parse_app_details_payload(details_payload, 730)
        with patch.object(client, "fetch_review_summary") as mock_summary:
            from src.data_collection.steam_data_miner import ReviewSummaryData

            mock_summary.return_value = ReviewSummaryData(
                total_reviews=500,
                positive_rate=90.0,
            )
            with patch.object(client, "fetch_review_sentiment", return_value=0.5):
                with patch.object(client, "fetch_current_players", return_value=100_000):
                    with patch.object(
                        client,
                        "fetch_store_page",
                        return_value=parse_store_page_html(store_html, "English"),
                    ):
                        snapshot = client.build_game_snapshot(730)

    assert snapshot is not None
    assert snapshot.positive_rate == 90.0
    assert snapshot.total_reviews == 500
