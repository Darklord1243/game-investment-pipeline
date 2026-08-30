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
    STEAM_STORE_APP_LIST_URL,
    GameSnapshot,
    SteamAPIClient,
    SteamRateLimiter,
    parse_app_details_payload,
    parse_review_summary_payload,
    parse_store_app_list_payload,
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


def test_parse_args_limit_default_is_none():
    from src.data_collection.steam_data_miner import parse_args

    args = parse_args([])
    assert args.limit is None


def test_parse_args_limit():
    from src.data_collection.steam_data_miner import parse_args

    args = parse_args(["--limit", "25"])
    assert args.limit == 25


def test_run_full_seed_caps_pending_appids():
    from src.data_collection.steam_data_miner import run_full_seed

    processed: list[int] = []
    entries = [SteamAppListEntry(appid=i, name=str(i)) for i in range(10)]

    with patch("src.data_collection.steam_data_miner.SteamAPIClient") as mock_client_cls:
        mock_client_cls.return_value.fetch_app_list.return_value = entries
        with patch("src.data_collection.steam_data_miner.db_session") as mock_db:
            mock_db.return_value.__enter__.return_value = MagicMock()
            with patch(
                "src.data_collection.steam_data_miner.get_mined_appids",
                return_value=set(),
            ):
                with patch(
                    "src.data_collection.steam_data_miner.process_appid_batch"
                ) as mock_proc:

                    def capture(batch, **kwargs):
                        processed.extend(batch)
                        return 0

                    mock_proc.side_effect = capture
                    run_full_seed(
                        limiter=MagicMock(),
                        max_workers=1,
                        batch_size=100,
                        min_reviews=50,
                        limit=3,
                    )

    assert processed == [0, 1, 2]


def test_parse_store_app_list_payload_filters_and_cursor() -> None:
    payload = {
        "response": {
            "apps": [
                {"appid": 10, "name": "Counter-Strike"},
                {"appid": 20, "name": ""},
                {"appid": 30, "name": "Day of Defeat"},
            ],
            "have_more_results": True,
            "last_appid": 30,
        }
    }
    entries, last_appid, have_more = parse_store_app_list_payload(payload)
    assert [entry.appid for entry in entries] == [10, 30]
    assert last_appid == 30
    assert have_more is True


def test_parse_store_app_list_payload_complete_when_flag_absent() -> None:
    payload = {"response": {"apps": [{"appid": 1, "name": "Half-Life"}]}}
    entries, last_appid, have_more = parse_store_app_list_payload(payload)
    assert len(entries) == 1
    assert last_appid is None
    assert have_more is False


def test_fetch_app_list_requires_steam_api_key() -> None:
    client = SteamAPIClient(limiter=SteamRateLimiter(), session=MagicMock())
    with patch("src.data_collection.steam_data_miner.os.getenv", return_value=None):
        with pytest.raises(RuntimeError, match="STEAM_API_KEY"):
            client.fetch_app_list()


def test_fetch_app_list_paginates_istore_service() -> None:
    client = SteamAPIClient(limiter=SteamRateLimiter(), session=MagicMock())
    pages = [
        {
            "response": {
                "apps": [{"appid": 1, "name": "One"}],
                "have_more_results": True,
                "last_appid": 1,
            }
        },
        {
            "response": {
                "apps": [{"appid": 2, "name": "Two"}],
                "have_more_results": False,
                "last_appid": 2,
            }
        },
    ]
    with patch("src.data_collection.steam_data_miner.os.getenv", return_value="test-key"):
        with patch.object(client, "_get_json", side_effect=pages) as mock_get:
            entries = client.fetch_app_list()

    assert [entry.name for entry in entries] == ["One", "Two"]
    assert mock_get.call_count == 2
    first_call = mock_get.call_args_list[0]
    second_call = mock_get.call_args_list[1]
    assert first_call.args[0] == STEAM_STORE_APP_LIST_URL
    assert first_call.kwargs["params"]["key"] == "test-key"
    assert "last_appid" not in first_call.kwargs["params"]
    assert second_call.kwargs["params"]["last_appid"] == 1

