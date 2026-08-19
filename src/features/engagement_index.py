"""
Descriptive engagement index — single source of truth for UI scoring.

The legacy ``target_score`` ML path used a 16-component composite that summed to
1.05 and included unpersisted Steam columns. This module re-grounds the formula
on ``FEATURE_COLUMNS`` the SQL feature matrix emits today. The 0.25 legacy Steam
weight mass is redistributed to sentiment, cross-platform, and Twitch until
``SteamMetric`` joins the feature matrix in Phase 2.

This index is **UI-only** and must never be used as ML supervision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence, Union, cast

import pandas as pd

logger = logging.getLogger(__name__)

_EPSILON: Final[float] = 1e-8
MIN_BATCH_FOR_RELATIVE: Final[int] = 5
TIER_HIGH_MIN: Final[float] = 65.0
TIER_MEDIUM_MIN: Final[float] = 35.0

ENGAGEMENT_INDEX_SPEC: Final[tuple[tuple[str, float], ...]] = (
    ("youtube_total_views", 0.15),
    ("youtube_engagement_rate", 0.12),
    ("youtube_total_likes", 0.05),
    ("youtube_total_comments", 0.05),
    ("youtube_avg_sentiment", 0.08),
    ("reddit_total_score", 0.12),
    ("reddit_post_count", 0.05),
    ("reddit_total_comments", 0.05),
    ("reddit_avg_sentiment", 0.08),
    ("twitch_total_viewers", 0.08),
    ("twitch_stream_count", 0.05),
    ("cross_platform_engagement_rate", 0.07),
    ("platform_presence", 0.05),
)

_weight_sum = sum(weight for _, weight in ENGAGEMENT_INDEX_SPEC)
if abs(_weight_sum - 1.0) >= 1e-9:
    raise ValueError(
        f"ENGAGEMENT_INDEX_SPEC weights must sum to 1.0 exactly; got {_weight_sum}"
    )

FORBIDDEN_AS_ML_FEATURES: Final[frozenset[str]] = frozenset(
    name for name, _ in ENGAGEMENT_INDEX_SPEC
) | frozenset(
    {
        "viral_potential_score",
        "competitive_score",
        "market_share",
        "platform_dominance",
    }
)

FEATURE_LABELS: Final[dict[str, str]] = {
    "youtube_total_views": "YouTube viewership",
    "youtube_engagement_rate": "YouTube engagement rate",
    "youtube_total_likes": "YouTube likes",
    "youtube_total_comments": "YouTube comments",
    "youtube_avg_sentiment": "YouTube sentiment",
    "reddit_total_score": "Reddit discussion score",
    "reddit_post_count": "Reddit post volume",
    "reddit_total_comments": "Reddit comment volume",
    "reddit_avg_sentiment": "Reddit sentiment",
    "twitch_total_viewers": "Twitch viewership",
    "twitch_stream_count": "Twitch stream count",
    "cross_platform_engagement_rate": "Cross-platform engagement",
    "platform_presence": "Multi-platform presence",
}

LABEL_DEFINITION: Final[str] = (
    "Engagement Score (0–100) is a transparent, weighted measure of a game's "
    "current cross-platform community activity and sentiment across YouTube, "
    "Reddit, and Twitch — relative to a reference set of games. It is "
    "descriptive, not a prediction, and not a sales or revenue estimate."
)

DISCLAIMER: Final[str] = (
    "This tool measures community engagement and sentiment intelligence, not "
    "commercial success. The Engagement Score is a descriptive composite of "
    "publicly mined signals (YouTube, Reddit, Twitch, Steam), normalized "
    "against a small reference set of games for demonstration. It does not "
    "predict sales, revenue, wishlists, or return on investment, and must not "
    "be used as financial or investment advice. Samples per game are limited, "
    "data freshness varies, and scores are comparative within the displayed "
    "set. Built as a course project (NUS SWS3023 Web Mining)."
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_SAMPLES_PATH: Final[Path] = _PROJECT_ROOT / "data" / "demo_samples.json"

RowInput = Union[pd.DataFrame, Mapping[str, Any], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True)
class ComponentContribution:
    """One weighted index component with plain-language context."""

    feature: str
    label: str
    raw_value: float
    weight: float
    normalized: float
    contribution: float


@dataclass(frozen=True)
class EngagementResult:
    """Full engagement-index output for the explain panel."""

    engagement_score: float
    tier: str
    drivers: tuple[ComponentContribution, ...]
    used_reference_stats: bool


def _component_names() -> list[str]:
    return [name for name, _ in ENGAGEMENT_INDEX_SPEC]


def load_demo_samples(path: Path | None = None) -> dict[str, Any]:
    """Load cached demo samples and reference distribution."""
    demo_path = path or DEFAULT_DEMO_SAMPLES_PATH
    if not demo_path.is_file():
        raise FileNotFoundError(f"Demo samples not found: {demo_path}")
    with demo_path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Demo samples root must be a JSON object, got {type(payload).__name__}."
        )
    return cast(dict[str, Any], payload)


def default_reference_stats() -> dict[str, dict[str, float]]:
    """Return reference min/max from ``data/demo_samples.json``."""
    payload = load_demo_samples()
    stats = payload.get("reference_stats")
    if not isinstance(stats, dict):
        raise KeyError("demo_samples.json missing 'reference_stats' object.")
    return stats


def _rows_to_dataframe(rows: RowInput) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, Mapping):
        return pd.DataFrame([dict(rows)])
    return pd.DataFrame([dict(row) for row in rows])


def _resolve_norm_bounds(
    batch: pd.DataFrame,
    reference_stats: Mapping[str, Mapping[str, float]] | None,
) -> tuple[dict[str, float], dict[str, float], bool]:
    """Choose per-component min/max for normalization."""
    use_reference = len(batch) < MIN_BATCH_FOR_RELATIVE
    ref = reference_stats if reference_stats is not None else default_reference_stats()

    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    for name, _ in ENGAGEMENT_INDEX_SPEC:
        if use_reference:
            bounds = ref.get(name, {})
            mins[name] = float(bounds.get("min", 0.0))
            maxs[name] = float(bounds.get("max", 0.0))
        else:
            values = pd.to_numeric(batch.get(name, 0.0), errors="coerce").fillna(0.0)
            mins[name] = float(values.min())
            maxs[name] = float(values.max())
    return mins, maxs, use_reference


def _normalize_component(value: float, min_val: float, max_val: float) -> float:
    denom = max_val - min_val + _EPSILON
    return (value - min_val) / denom


def compute_engagement_score(
    rows: RowInput,
    reference_stats: Mapping[str, Mapping[str, float]] | None = None,
) -> float | pd.Series:
    """
    Compute Engagement Score in [0, 100] for one or more feature rows.

    Batches with fewer than ``MIN_BATCH_FOR_RELATIVE`` rows normalize against
    ``reference_stats`` (demo sample distribution) to avoid degenerate 0 scores.
    """
    batch = _rows_to_dataframe(rows)
    if batch.empty:
        return pd.Series(dtype="float64", name="engagement_score")

    mins, maxs, _ = _resolve_norm_bounds(batch, reference_stats)
    scores: list[float] = []
    for _, row in batch.iterrows():
        index_raw = 0.0
        for name, weight in ENGAGEMENT_INDEX_SPEC:
            raw = float(pd.to_numeric(row.get(name, 0.0), errors="coerce") or 0.0)
            norm = _normalize_component(raw, mins[name], maxs[name])
            index_raw += weight * norm
        scores.append(round(min(100.0, max(0.0, 100.0 * index_raw)), 2))

    series = pd.Series(scores, index=batch.index, name="engagement_score")
    if len(series) == 1:
        return float(series.iloc[0])
    return series


def score_to_tier(score: float) -> str:
    """Map a score to HIGH / MEDIUM / LOW tier labels."""
    if score >= TIER_HIGH_MIN:
        return "HIGH"
    if score >= TIER_MEDIUM_MIN:
        return "MEDIUM"
    return "LOW"


def compute_engagement_details(
    rows: RowInput,
    reference_stats: Mapping[str, Mapping[str, float]] | None = None,
) -> list[EngagementResult]:
    """Compute scores plus ranked component contributions for explainability."""
    batch = _rows_to_dataframe(rows)
    if batch.empty:
        return []

    mins, maxs, used_reference = _resolve_norm_bounds(batch, reference_stats)
    results: list[EngagementResult] = []

    for _, row in batch.iterrows():
        contributions: list[ComponentContribution] = []
        index_raw = 0.0
        for name, weight in ENGAGEMENT_INDEX_SPEC:
            raw = float(pd.to_numeric(row.get(name, 0.0), errors="coerce") or 0.0)
            norm = _normalize_component(raw, mins[name], maxs[name])
            contrib = weight * norm
            index_raw += contrib
            contributions.append(
                ComponentContribution(
                    feature=name,
                    label=FEATURE_LABELS.get(name, name),
                    raw_value=raw,
                    weight=weight,
                    normalized=norm,
                    contribution=contrib,
                )
            )
        score = round(min(100.0, max(0.0, 100.0 * index_raw)), 2)
        ranked = tuple(
            sorted(contributions, key=lambda c: c.contribution, reverse=True)
        )
        results.append(
            EngagementResult(
                engagement_score=score,
                tier=score_to_tier(score),
                drivers=ranked,
                used_reference_stats=used_reference,
            )
        )
    return results


def driver_plain_language(contrib: ComponentContribution) -> str:
    """Translate a component contribution into a presenter-friendly sentence."""
    label = contrib.label
    value = contrib.raw_value
    if contrib.feature == "youtube_total_views" and value >= 1_000_000:
        detail = f"{value / 1_000_000:.1f}M views"
    elif contrib.feature == "youtube_total_views" and value >= 1_000:
        detail = f"{value / 1_000:.0f}k views"
    elif contrib.feature == "reddit_total_score" and value >= 1_000:
        detail = f"{value / 1_000:.0f}k score"
    elif "sentiment" in contrib.feature:
        detail = f"{value:+.2f}"
    elif contrib.feature == "platform_presence":
        detail = f"{int(value)} platform(s)"
    else:
        detail = f"{value:g}"
    strength = "Strong" if contrib.contribution >= 0.08 else "Moderate"
    return f"{strength} {label} ({detail})"


__all__ = [
    "DEFAULT_DEMO_SAMPLES_PATH",
    "DISCLAIMER",
    "ENGAGEMENT_INDEX_SPEC",
    "FEATURE_LABELS",
    "FORBIDDEN_AS_ML_FEATURES",
    "LABEL_DEFINITION",
    "MIN_BATCH_FOR_RELATIVE",
    "TIER_HIGH_MIN",
    "TIER_MEDIUM_MIN",
    "ComponentContribution",
    "EngagementResult",
    "compute_engagement_details",
    "compute_engagement_score",
    "default_reference_stats",
    "driver_plain_language",
    "load_demo_samples",
    "score_to_tier",
]
