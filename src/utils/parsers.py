"""
Robust type-coercion and text-normalisation helpers.

Used by every data-collection miner so that parsing rules stay consistent
across the pipeline and only need to be updated in one place.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_positive_int(raw: Any, field_name: str) -> int | None:
    """Parse a non-negative integer; log and return *None* on failure.

    Handles numeric strings that include thousands separators (``","``),
    which Steam and some YouTube endpoints return.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        logger.warning("Boolean provided for integer field %s: %r", field_name, raw)
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, str):
        cleaned = raw.replace(",", "").strip()
        if cleaned.isdigit():
            return int(cleaned)
    logger.warning(
        "Cannot parse %s as int: %r (%s)",
        field_name,
        raw,
        type(raw).__name__,
    )
    return None


def normalize_text(raw: Any, field_name: str) -> str:
    """Normalize optional text fields; collapse internal whitespace.

    Returns an empty string when *raw* is ``None``.
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        logger.warning(
            "Expected str for %s, got %s; coercing via str().",
            field_name,
            type(raw).__name__,
        )
        raw = str(raw)
    return re.sub(r"[\n\r\t]+", " ", raw).strip()
