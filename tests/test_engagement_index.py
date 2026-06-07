"""Tests for the descriptive engagement index module."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.engagement_index import (  # noqa: E402
    ENGAGEMENT_INDEX_SPEC,
    FORBIDDEN_AS_ML_FEATURES,
    compute_engagement_score,
    default_reference_stats,
    load_demo_samples,
    score_to_tier,
)


def test_weights_sum_to_one():
    total = sum(weight for _, weight in ENGAGEMENT_INDEX_SPEC)
    assert abs(total - 1.0) < 1e-9


def test_forbidden_features_include_all_components():
    for name, _ in ENGAGEMENT_INDEX_SPEC:
        assert name in FORBIDDEN_AS_ML_FEATURES


def test_single_game_uses_reference_stats_not_zero():
    ref = default_reference_stats()
    features = ref["youtube_total_views"]
    row = {
        "youtube_total_views": (features["min"] + features["max"]) / 2,
        "youtube_engagement_rate": 0.04,
        "youtube_total_likes": 10_000,
        "youtube_total_comments": 2_000,
        "youtube_avg_sentiment": 0.3,
        "reddit_total_score": 5_000,
        "reddit_post_count": 10,
        "reddit_total_comments": 800,
        "reddit_avg_sentiment": 0.2,
        "twitch_total_viewers": 20_000,
        "twitch_stream_count": 5,
        "cross_platform_engagement_rate": 0.03,
        "platform_presence": 3,
    }
    score = compute_engagement_score(row, ref)
    assert 0.0 <= float(score) <= 100.0
    assert float(score) > 0.0


def test_monotonicity_youtube_views():
    ref = default_reference_stats()
    base = {
        "youtube_total_views": 100_000,
        "youtube_engagement_rate": 0.04,
        "youtube_total_likes": 5_000,
        "youtube_total_comments": 1_000,
        "youtube_avg_sentiment": 0.2,
        "reddit_total_score": 5_000,
        "reddit_post_count": 10,
        "reddit_total_comments": 500,
        "reddit_avg_sentiment": 0.1,
        "twitch_total_viewers": 10_000,
        "twitch_stream_count": 5,
        "cross_platform_engagement_rate": 0.03,
        "platform_presence": 3,
    }
    low = float(compute_engagement_score(base, ref))
    higher = dict(base)
    higher["youtube_total_views"] = 2_000_000
    high = float(compute_engagement_score(higher, ref))
    assert high > low


def test_clamp_to_zero_hundred():
    ref = default_reference_stats()
    zeros = {name: 0.0 for name, _ in ENGAGEMENT_INDEX_SPEC}
    score = float(compute_engagement_score(zeros, ref))
    assert 0.0 <= score <= 100.0


def test_demo_samples_precomputed_scores_match():
    payload = load_demo_samples()
    ref = payload["reference_stats"]
    for sample in payload["samples"]:
        expected = float(sample["precomputed_engagement_score"])
        actual = float(compute_engagement_score(sample["features"], ref))
        assert actual == pytest.approx(expected, abs=0.05)


def test_score_to_tier_cutoffs():
    assert score_to_tier(70.0) == "HIGH"
    assert score_to_tier(65.0) == "HIGH"
    assert score_to_tier(50.0) == "MEDIUM"
    assert score_to_tier(35.0) == "MEDIUM"
    assert score_to_tier(20.0) == "LOW"
