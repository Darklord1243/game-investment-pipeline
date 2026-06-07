"""Tests for Flask demo routes — no live network calls."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.webapp import (  # noqa: E402
    QuotaExhaustedError,
    app,
    default_sandbox_values,
)


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _valid_sandbox_payload() -> dict[str, str]:
    return {key: str(value) for key, value in default_sandbox_values().items() | {
        "youtube_total_views": "1200000",
        "youtube_engagement_rate": "0.04",
        "youtube_total_likes": "45000",
        "youtube_total_comments": "8000",
        "youtube_avg_sentiment": "0.35",
        "reddit_total_score": "25000",
        "reddit_post_count": "15",
        "reddit_total_comments": "4200",
        "reddit_avg_sentiment": "0.18",
        "twitch_total_viewers": "90000",
        "twitch_stream_count": "20",
        "cross_platform_engagement_rate": "0.05",
        "platform_presence": "3",
    }.items()}


def test_post_sandbox_valid_returns_score_and_explain_panel(client):
    response = client.post("/sandbox", data=_valid_sandbox_payload())
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Engagement Score" in body
    assert "Top drivers" in body
    assert "Per-platform sentiment" in body
    assert "Limitations" in body


def test_post_sandbox_missing_field_returns_400(client):
    payload = _valid_sandbox_payload()
    del payload["youtube_total_views"]
    response = client.post("/sandbox", data=payload)
    assert response.status_code == 400
    assert "Missing required field" in response.get_data(as_text=True)


def test_post_sandbox_out_of_range_sentiment(client):
    payload = _valid_sandbox_payload()
    payload["youtube_avg_sentiment"] = "2.0"
    response = client.post("/sandbox", data=payload)
    assert response.status_code == 400
    assert "must be between" in response.get_data(as_text=True)


def test_get_samples_lists_ten_games(client):
    response = client.get("/samples", headers={"Accept": "application/json"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == 10
    assert len(data["samples"]) == 10


@patch("src.api.webapp.run_platform_miners")
def test_predict_quota_fallback_returns_cached_banner(mock_miners, client):
    mock_miners.side_effect = QuotaExhaustedError("YouTube API quota exhausted")
    response = client.post(
        "/predict",
        json={"steam_name": "Stardew Valley"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "engagement_score" in data
    assert data["freshness"] == "CACHED"
    assert "cache_banner" in data
    mock_miners.assert_called_once()


def test_health_requires_demo_samples_and_index(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["demo_samples_available"] is True
    assert data["engagement_index_available"] is True


def test_sandbox_does_not_call_miners(client):
    with patch("src.api.webapp.run_platform_miners") as mock_miners:
        with patch("src.api.webapp.run_live_engagement") as mock_live:
            client.post("/sandbox", data=_valid_sandbox_payload())
            mock_miners.assert_not_called()
            mock_live.assert_not_called()
