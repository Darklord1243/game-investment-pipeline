"""Generate data/demo_samples.json with precomputed engagement scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.engagement_index import ENGAGEMENT_INDEX_SPEC, compute_engagement_score

SAMPLES_RAW = [
    {
        "steam_name": "Counter-Strike 2",
        "appid": 730,
        "tier_expected": "high",
        "narrative": "F2P juggernaut; huge Twitch + YouTube; ceiling anchor",
        "features": {
            "youtube_total_views": 4_500_000,
            "youtube_engagement_rate": 0.06,
            "youtube_total_likes": 180_000,
            "youtube_total_comments": 42_000,
            "youtube_avg_sentiment": 0.22,
            "reddit_total_score": 65_000,
            "reddit_post_count": 120,
            "reddit_total_comments": 18_000,
            "reddit_avg_sentiment": 0.15,
            "twitch_total_viewers": 350_000,
            "twitch_stream_count": 450,
            "cross_platform_engagement_rate": 0.08,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Baldur's Gate 3",
        "appid": 1_086_940,
        "tier_expected": "high",
        "narrative": "Critical hit; strong Reddit discussion + positive sentiment",
        "features": {
            "youtube_total_views": 3_200_000,
            "youtube_engagement_rate": 0.055,
            "youtube_total_likes": 140_000,
            "youtube_total_comments": 35_000,
            "youtube_avg_sentiment": 0.45,
            "reddit_total_score": 72_000,
            "reddit_post_count": 200,
            "reddit_total_comments": 25_000,
            "reddit_avg_sentiment": 0.38,
            "twitch_total_viewers": 120_000,
            "twitch_stream_count": 85,
            "cross_platform_engagement_rate": 0.07,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Stardew Valley",
        "appid": 413_150,
        "tier_expected": "high",
        "narrative": "Sleeper indie hit; durable community, modest Twitch",
        "features": {
            "youtube_total_views": 2_600_000,
            "youtube_engagement_rate": 0.048,
            "youtube_total_likes": 88_000,
            "youtube_total_comments": 15_000,
            "youtube_avg_sentiment": 0.40,
            "reddit_total_score": 52_000,
            "reddit_post_count": 70,
            "reddit_total_comments": 9_000,
            "reddit_avg_sentiment": 0.32,
            "twitch_total_viewers": 130_000,
            "twitch_stream_count": 42,
            "cross_platform_engagement_rate": 0.058,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Dota 2",
        "appid": 570,
        "tier_expected": "high",
        "narrative": "F2P massive CCU; skews revenue proxies",
        "features": {
            "youtube_total_views": 3_400_000,
            "youtube_engagement_rate": 0.04,
            "youtube_total_likes": 115_000,
            "youtube_total_comments": 28_000,
            "youtube_avg_sentiment": 0.18,
            "reddit_total_score": 58_000,
            "reddit_post_count": 110,
            "reddit_total_comments": 15_000,
            "reddit_avg_sentiment": 0.12,
            "twitch_total_viewers": 320_000,
            "twitch_stream_count": 380,
            "cross_platform_engagement_rate": 0.065,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Hades",
        "appid": 1_145_360,
        "tier_expected": "medium-high",
        "narrative": "Acclaimed indie; balanced cross-platform, very positive sentiment",
        "features": {
            "youtube_total_views": 850_000,
            "youtube_engagement_rate": 0.048,
            "youtube_total_likes": 38_000,
            "youtube_total_comments": 6_500,
            "youtube_avg_sentiment": 0.52,
            "reddit_total_score": 18_000,
            "reddit_post_count": 35,
            "reddit_total_comments": 3_200,
            "reddit_avg_sentiment": 0.42,
            "twitch_total_viewers": 45_000,
            "twitch_stream_count": 18,
            "cross_platform_engagement_rate": 0.045,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Vampire Survivors",
        "appid": 1_794_680,
        "tier_expected": "medium",
        "narrative": "Cheap breakout; high engagement-per-dollar; strong Reddit",
        "features": {
            "youtube_total_views": 620_000,
            "youtube_engagement_rate": 0.05,
            "youtube_total_likes": 28_000,
            "youtube_total_comments": 5_000,
            "youtube_avg_sentiment": 0.28,
            "reddit_total_score": 14_000,
            "reddit_post_count": 28,
            "reddit_total_comments": 2_800,
            "reddit_avg_sentiment": 0.25,
            "twitch_total_viewers": 35_000,
            "twitch_stream_count": 12,
            "cross_platform_engagement_rate": 0.04,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Hollow Knight",
        "appid": 367_520,
        "tier_expected": "medium",
        "narrative": "Beloved indie; steady, not viral; good sentiment",
        "features": {
            "youtube_total_views": 480_000,
            "youtube_engagement_rate": 0.038,
            "youtube_total_likes": 22_000,
            "youtube_total_comments": 3_800,
            "youtube_avg_sentiment": 0.40,
            "reddit_total_score": 11_000,
            "reddit_post_count": 22,
            "reddit_total_comments": 2_100,
            "reddit_avg_sentiment": 0.32,
            "twitch_total_viewers": 28_000,
            "twitch_stream_count": 10,
            "cross_platform_engagement_rate": 0.035,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Creaks",
        "appid": 956_030,
        "tier_expected": "low",
        "narrative": "Quiet puzzle launch; thin cross-platform signal — floor anchor",
        "features": {
            "youtube_total_views": 45_000,
            "youtube_engagement_rate": 0.025,
            "youtube_total_likes": 1_200,
            "youtube_total_comments": 180,
            "youtube_avg_sentiment": 0.15,
            "reddit_total_score": 800,
            "reddit_post_count": 4,
            "reddit_total_comments": 120,
            "reddit_avg_sentiment": 0.10,
            "twitch_total_viewers": 500,
            "twitch_stream_count": 2,
            "cross_platform_engagement_rate": 0.01,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "Concord",
        "appid": 2_672_530,
        "tier_expected": "low",
        "narrative": "Known commercial flop; negative/sparse sentiment despite marketing buzz",
        "features": {
            "youtube_total_views": 380_000,
            "youtube_engagement_rate": 0.02,
            "youtube_total_likes": 8_000,
            "youtube_total_comments": 15_000,
            "youtube_avg_sentiment": -0.35,
            "reddit_total_score": 5_200,
            "reddit_post_count": 45,
            "reddit_total_comments": 8_000,
            "reddit_avg_sentiment": -0.28,
            "twitch_total_viewers": 12_000,
            "twitch_stream_count": 8,
            "cross_platform_engagement_rate": 0.015,
            "platform_presence": 3,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
    {
        "steam_name": "A Short Hike",
        "appid": 1_055_540,
        "tier_expected": "low-medium",
        "narrative": "Indie with no Twitch presence; platform-absence test",
        "features": {
            "youtube_total_views": 180_000,
            "youtube_engagement_rate": 0.042,
            "youtube_total_likes": 9_500,
            "youtube_total_comments": 1_600,
            "youtube_avg_sentiment": 0.48,
            "reddit_total_score": 6_500,
            "reddit_post_count": 18,
            "reddit_total_comments": 1_400,
            "reddit_avg_sentiment": 0.35,
            "twitch_total_viewers": 0,
            "twitch_stream_count": 0,
            "cross_platform_engagement_rate": 0.02,
            "platform_presence": 2,
        },
        "mined_at": "2026-06-01T00:00:00Z",
    },
]


def main() -> None:
    comp_names = [name for name, _ in ENGAGEMENT_INDEX_SPEC]
    reference_stats: dict[str, dict[str, float]] = {}
    for name in comp_names:
        values = [sample["features"][name] for sample in SAMPLES_RAW]
        reference_stats[name] = {"min": float(min(values)), "max": float(max(values))}

    for sample in SAMPLES_RAW:
        sample["precomputed_engagement_score"] = compute_engagement_score(
            sample["features"],
            reference_stats,
        )

    payload = {
        "schema_version": 1,
        "generated_at": "2026-06-07T00:00:00Z",
        "reference_stats": reference_stats,
        "samples": SAMPLES_RAW,
    }
    out_path = _ROOT / "data" / "demo_samples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for sample in SAMPLES_RAW:
        print(f"{sample['steam_name']}: {sample['precomputed_engagement_score']}")


if __name__ == "__main__":
    main()
