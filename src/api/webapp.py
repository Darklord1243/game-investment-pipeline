"""
Game Engagement Intelligence API.

Flask + Jinja demo backend: offline sandbox scoring via the descriptive
engagement index, live mining with cached fallback, and explainability panel.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Final, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from googleapiclient.errors import HttpError
from prawcore.exceptions import TooManyRequests
from requests.exceptions import HTTPError as RequestsHTTPError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.reddit_data_miner import (  # noqa: E402
    RedditAPIClient,
    ensure_schema as ensure_reddit_schema,
    load_reddit_credentials,
    run_single_game as run_reddit_single_game,
)
from src.data_collection.steam_data_miner import (  # noqa: E402
    MIN_REVIEWS_FOR_SEED,
    SteamAPIClient,
    SteamRateLimiter,
    ensure_schema as ensure_steam_schema,
    resolve_appid_by_name,
    run_single_game as run_steam_single_game,
)
from src.data_collection.twitch_miner import (  # noqa: E402
    TwitchHelixClient,
    ensure_schema as ensure_twitch_schema,
    load_twitch_credentials,
    run_single_game as run_twitch_single_game,
)
from src.data_collection.youtube_data_miner import (  # noqa: E402
    YouTubeAPIClient,
    ensure_schema as ensure_youtube_schema,
    is_youtube_quota_error,
    load_youtube_api_key,
    run_single_game as run_youtube_single_game,
)
from src.database.models import Game, SessionLocal  # noqa: E402
from src.features.engagement_index import (  # noqa: E402
    DEFAULT_DEMO_SAMPLES_PATH,
    DISCLAIMER,
    ENGAGEMENT_INDEX_SPEC,
    LABEL_DEFINITION,
    compute_engagement_details,
    compute_engagement_score,
    default_reference_stats,
    driver_plain_language,
    load_demo_samples,
    score_to_tier,
)
from src.features.sql_feature_engineer import DatabaseFeatureEngineer  # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)

STEAM_NAME_MAX_LEN: Final[int] = 512
STEAM_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[\w\s\-:''.,&!()+®™]+$",
    re.UNICODE,
)
INDEX_COMPONENTS: Final[tuple[str, ...]] = tuple(name for name, _ in ENGAGEMENT_INDEX_SPEC)
DEMO_SAMPLES_PATH: Final[Path] = DEFAULT_DEMO_SAMPLES_PATH

SANDBOX_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "youtube_engagement_rate": (0.0, 1.0),
    "cross_platform_engagement_rate": (0.0, 1.0),
    "youtube_avg_sentiment": (-1.0, 1.0),
    "reddit_avg_sentiment": (-1.0, 1.0),
    "platform_presence": (0.0, 3.0),
}


class GameNotFoundError(Exception):
    """Raised when a title cannot be resolved on Steam or in the warehouse."""


class QuotaExhaustedError(Exception):
    """Raised when an upstream platform API quota is exhausted."""


def configure_logging() -> None:
    """Configure application logging once."""
    if logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def utc_today() -> dt.date:
    """Return the current UTC calendar date (batch key)."""
    return dt.datetime.now(dt.timezone.utc).date()


def utc_now_iso() -> str:
    """Return current UTC timestamp for freshness display."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def validate_steam_name(raw: Optional[str]) -> str:
    """
    Validate and normalize ``steam_name`` from user input.

    Raises:
        ValueError: When empty, too long, or contains disallowed characters.
    """
    if raw is None:
        raise ValueError("steam_name is required.")
    name = str(raw).strip()
    if not name:
        raise ValueError("steam_name must not be empty.")
    if len(name) > STEAM_NAME_MAX_LEN:
        raise ValueError(f"steam_name must be at most {STEAM_NAME_MAX_LEN} characters.")
    if not STEAM_NAME_PATTERN.match(name):
        raise ValueError("steam_name contains invalid characters.")
    return name


def ensure_database_schema() -> None:
    """Create ORM tables if they do not exist."""
    ensure_steam_schema()
    ensure_twitch_schema()
    ensure_youtube_schema()
    ensure_reddit_schema()


def lookup_game_by_steam_name(session: Session, steam_name: str) -> Optional[Game]:
    """Case-insensitive lookup of a ``Game`` by ``steam_name``."""
    normalized = steam_name.strip().lower()
    stmt = select(Game).where(func.lower(Game.steam_name) == normalized)
    return session.scalars(stmt).first()


def resolve_canonical_steam_name(client: SteamAPIClient, query: str) -> str:
    """
    Fuzzy-resolve a user query to the canonical Steam store title.

    Raises:
        GameNotFoundError: When no app-list match exists.
    """
    entries = client.fetch_app_list()
    match = resolve_appid_by_name(entries, query)
    if match is None:
        raise GameNotFoundError(
            f"No Steam app list match for {query!r}. "
            "Check spelling or try the exact Steam store title."
        )
    return match.name


def is_quota_exhausted(exc: BaseException) -> bool:
    """Return True when *exc* (or its cause chain) signals API quota exhaustion."""
    if isinstance(exc, TooManyRequests):
        return True
    if isinstance(exc, HttpError) and is_youtube_quota_error(exc):
        return True
    if isinstance(exc, RequestsHTTPError):
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 429:
            return True
    if isinstance(exc, RuntimeError) and exc.__cause__ is not None:
        return is_quota_exhausted(exc.__cause__)
    message = str(exc).lower()
    return "quota" in message and "exceed" in message


def run_platform_miners(steam_query: str) -> str:
    """
    Seed Steam (if needed) and mine Twitch, YouTube, and Reddit for one title.

    Returns:
        Canonical ``steam_name`` stored in the warehouse.

    Raises:
        GameNotFoundError: Steam title not found or game not seeded.
        QuotaExhaustedError: Upstream API quota exhausted.
    """
    limiter = SteamRateLimiter()
    steam_client = SteamAPIClient(limiter=limiter, session=requests.Session())
    canonical_name = resolve_canonical_steam_name(steam_client, steam_query)

    platform_runners: list[tuple[str, Any]] = [
        (
            "steam",
            lambda: run_steam_single_game(
                steam_client,
                steam_query,
                min_reviews=MIN_REVIEWS_FOR_SEED,
            ),
        ),
        ("twitch", lambda: _run_twitch(canonical_name)),
        ("youtube", lambda: _run_youtube(canonical_name)),
        ("reddit", lambda: _run_reddit(canonical_name)),
    ]

    for platform, runner in platform_runners:
        try:
            runner()
            logger.info("Miner %s completed for %r.", platform, canonical_name)
        except GameNotFoundError:
            raise
        except ValueError as exc:
            message = str(exc)
            if "No Steam app list match" in message or "No Game record found" in message:
                raise GameNotFoundError(message) from exc
            if is_quota_exhausted(exc):
                raise QuotaExhaustedError(
                    f"{platform.title()} API quota exhausted; try again later."
                ) from exc
            logger.exception(
                "Non-fatal miner failure on %s for %r: %s",
                platform,
                canonical_name,
                exc,
            )
        except Exception as exc:
            if is_quota_exhausted(exc):
                logger.error("Quota exhausted on %s for %r.", platform, canonical_name)
                raise QuotaExhaustedError(
                    f"{platform.title()} API quota exhausted; try again later."
                ) from exc
            logger.exception(
                "Non-fatal miner failure on %s for %r: %s",
                platform,
                canonical_name,
                exc,
            )

    with SessionLocal() as session:
        game = lookup_game_by_steam_name(session, canonical_name)
    if game is None:
        raise GameNotFoundError(
            f"Game {canonical_name!r} could not be seeded in the warehouse. "
            "It may be below the review threshold or already failed validation."
        )
    return canonical_name


def _run_twitch(steam_name: str) -> None:
    client_id, client_secret = load_twitch_credentials()
    client = TwitchHelixClient(client_id=client_id, client_secret=client_secret)
    client.authenticate()
    run_twitch_single_game(client, steam_name)


def _run_youtube(steam_name: str) -> None:
    api_key = load_youtube_api_key()
    client = YouTubeAPIClient(api_key=api_key)
    run_youtube_single_game(client, steam_name)


def _run_reddit(steam_name: str) -> None:
    client_id, client_secret, username, password, user_agent = load_reddit_credentials()
    client = RedditAPIClient(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password,
    )
    run_reddit_single_game(client, steam_name)


def filter_game_row(features: pd.DataFrame, steam_name: str) -> pd.DataFrame:
    """Return the feature-matrix row for *steam_name* (case-insensitive)."""
    if features.empty or "steam_name" not in features.columns:
        return features.iloc[0:0]
    mask = features["steam_name"].astype(str).str.strip().str.lower() == steam_name.strip().lower()
    return features.loc[mask]


def extract_index_features(row: pd.Series) -> dict[str, float]:
    """Pull engagement-index components from a feature-matrix row."""
    features: dict[str, float] = {}
    for name in INDEX_COMPONENTS:
        raw = row.get(name, 0.0)
        features[name] = float(pd.to_numeric(raw, errors="coerce") or 0.0)
    return features


def sentiment_bar_width(value: float) -> float:
    """Map sentiment in [-1, 1] to a 0–100 bar width."""
    return min(100.0, max(0.0, (float(value) + 1.0) * 50.0))


def build_result_context(
    *,
    steam_name: str,
    features: dict[str, float],
    freshness: str,
    mined_at: str,
    reference_stats: Optional[dict[str, dict[str, float]]] = None,
    cache_banner: Optional[str] = None,
) -> dict[str, Any]:
    """Build explain-panel context from feature values."""
    ref = reference_stats or default_reference_stats()
    details = compute_engagement_details(features, ref)[0]
    top_drivers = [
        driver_plain_language(contrib)
        for contrib in details.drivers[:3]
    ]
    return {
        "steam_name": steam_name,
        "engagement_score": details.engagement_score,
        "tier": details.tier,
        "top_drivers": top_drivers,
        "youtube_sentiment": features.get("youtube_avg_sentiment", 0.0),
        "reddit_sentiment": features.get("reddit_avg_sentiment", 0.0),
        "youtube_bar": sentiment_bar_width(features.get("youtube_avg_sentiment", 0.0)),
        "reddit_bar": sentiment_bar_width(features.get("reddit_avg_sentiment", 0.0)),
        "freshness": freshness,
        "mined_at": mined_at,
        "cache_banner": cache_banner,
    }


def default_sandbox_values() -> dict[str, float]:
    """Empty-ish defaults for the sandbox form."""
    return {name: 0.0 for name in INDEX_COMPONENTS}


def parse_sandbox_form(form: Any) -> dict[str, float]:
    """
    Parse and validate sandbox form fields.

    Raises:
        ValueError: On missing or out-of-range fields.
    """
    values: dict[str, float] = {}
    for name in INDEX_COMPONENTS:
        raw = form.get(name)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"Missing required field: {name}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {name}.") from exc
        if name in SANDBOX_BOUNDS:
            low, high = SANDBOX_BOUNDS[name]
            if value < low or value > high:
                raise ValueError(f"{name} must be between {low} and {high}.")
        if value < 0 and name not in SANDBOX_BOUNDS:
            raise ValueError(f"{name} must be non-negative.")
        values[name] = value
    return values


def find_cached_sample(query: str) -> dict[str, Any]:
    """Return the best matching cached demo sample for *query*."""
    payload = load_demo_samples()
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise FileNotFoundError("No cached samples available.")

    normalized = query.strip().lower()
    for sample in samples:
        if isinstance(sample, dict) and str(sample.get("steam_name", "")).strip().lower() == normalized:
            return sample
    for sample in samples:
        if isinstance(sample, dict) and normalized in str(sample.get("steam_name", "")).strip().lower():
            return sample
    fallback = samples[0]
    if not isinstance(fallback, dict):
        raise TypeError("Cached sample is not a JSON object.")
    return fallback


def cached_result_for_query(query: str, reason: str) -> dict[str, Any]:
    """Build a cached-fallback result payload."""
    sample = find_cached_sample(query)
    steam_name = str(sample["steam_name"])
    features = dict(sample["features"])
    banner = (
        f"Live mining unavailable ({reason}). Showing the nearest cached sample: "
        f'"{steam_name}".'
    )
    return build_result_context(
        steam_name=steam_name,
        features=features,
        freshness="CACHED",
        mined_at=str(sample.get("mined_at", "cached")),
        cache_banner=banner,
    )


def run_live_engagement(steam_name: str, batch_date: dt.date) -> dict[str, Any]:
    """Mine live data and score via the engagement index."""
    canonical_name = run_platform_miners(steam_name)

    with SessionLocal() as session:
        features_df = DatabaseFeatureEngineer().build_feature_matrix(session, batch_date)

    game_df = filter_game_row(features_df, canonical_name)
    if game_df.empty:
        raise GameNotFoundError(
            f"No feature row for {canonical_name!r} on batch_date={batch_date.isoformat()}."
        )

    row = game_df.iloc[0]
    index_features = extract_index_features(row)
    return build_result_context(
        steam_name=canonical_name,
        features=index_features,
        freshness="LIVE",
        mined_at=utc_now_iso(),
    )


def wants_json_response() -> bool:
    """True when the client expects a JSON body (API/tests)."""
    if request.args.get("format") == "json":
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept and "text/html" not in accept


def template_context(**extra: Any) -> dict[str, Any]:
    """Inject shared footer strings into every template."""
    return {
        "label_definition": LABEL_DEFINITION,
        "disclaimer": DISCLAIMER,
        **extra,
    }


configure_logging()
ensure_database_schema()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


@app.route("/", methods=["GET"])
def index() -> str:
    """Landing page with links to analyze and sandbox flows."""
    return render_template("index.html", **template_context())


@app.route("/sandbox", methods=["GET", "POST"])
def sandbox() -> Any:
    """Offline manual-input engagement scoring (no network)."""
    payload = load_demo_samples()
    sample_list = payload.get("samples", [])
    values = default_sandbox_values()
    selected_sample: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None

    if request.method == "GET":
        prefill = request.args.get("sample_prefill", "").strip()
        if prefill:
            for sample in sample_list:
                if sample.get("steam_name") == prefill:
                    values = dict(sample["features"])
                    selected_sample = prefill
                    break

    if request.method == "POST":
        selected_sample = request.form.get("sample_prefill") or None
        if selected_sample:
            for sample in sample_list:
                if sample.get("steam_name") == selected_sample:
                    values = dict(sample["features"])
                    break
        try:
            values = parse_sandbox_form(request.form)
            result = build_result_context(
                steam_name=selected_sample or "Sandbox input",
                features=values,
                freshness="CACHED",
                mined_at="offline",
            )
        except ValueError as exc:
            error = str(exc)
            if wants_json_response():
                return jsonify({"error": error}), 400
            return (
                render_template(
                    "sandbox.html",
                    **template_context(
                        samples=sample_list,
                        values=values,
                        selected_sample=selected_sample,
                        error=error,
                        result=None,
                    ),
                ),
                400,
            )

    return render_template(
        "sandbox.html",
        **template_context(
            samples=sample_list,
            values=values,
            selected_sample=selected_sample,
            error=error,
            result=result,
        ),
    )


@app.route("/samples", methods=["GET"])
def samples() -> Any:
    """List cached demo games."""
    payload = load_demo_samples()
    rows = []
    for sample in payload.get("samples", []):
        score = float(sample.get("precomputed_engagement_score", 0.0))
        rows.append(
            {
                "steam_name": sample.get("steam_name"),
                "precomputed_engagement_score": score,
                "tier": score_to_tier(score),
                "narrative": sample.get("narrative", ""),
            }
        )
    if wants_json_response():
        return jsonify({"samples": rows, "count": len(rows)}), 200
    return render_template("samples.html", **template_context(samples=rows))


@app.route("/predict", methods=["GET", "POST"])
def predict() -> Any:
    """Live mine → engagement index, with cached fallback on quota/404."""
    if request.method == "GET" and not request.args.get("steam_name"):
        return render_template("predict.html", **template_context())

    steam_name_raw = None
    if request.method == "POST":
        if request.is_json:
            body = request.get_json(silent=True) or {}
            steam_name_raw = body.get("steam_name")
        else:
            steam_name_raw = request.form.get("steam_name")
    else:
        steam_name_raw = request.args.get("steam_name")

    try:
        steam_name = validate_steam_name(steam_name_raw)
    except ValueError as exc:
        if wants_json_response():
            return jsonify({"error": str(exc)}), 400
        return render_template(
            "predict.html",
            **template_context(error=str(exc)),
        )

    batch_date = utc_today()
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    try:
        result = run_live_engagement(steam_name, batch_date)
    except GameNotFoundError as exc:
        logger.warning("Game not found for %r: %s", steam_name, exc)
        if wants_json_response():
            result = cached_result_for_query(steam_name, "unknown game")
            result["requested_name"] = steam_name
            return jsonify(result), 200
        result = cached_result_for_query(steam_name, "unknown game")
    except QuotaExhaustedError as exc:
        logger.warning("Quota exhausted for %r: %s", steam_name, exc)
        if wants_json_response():
            result = cached_result_for_query(steam_name, "quota exhausted")
            result["requested_name"] = steam_name
            return jsonify(result), 200
        result = cached_result_for_query(steam_name, "quota exhausted")
    except Exception as exc:
        logger.exception("Inference failed for %r", steam_name)
        message = f"Inference failed: {exc}"
        if wants_json_response():
            return jsonify({"error": message}), 500
        error = message

    if wants_json_response() and result is not None:
        payload = dict(result)
        payload["requested_name"] = steam_name
        payload["batch_date"] = batch_date.isoformat()
        return jsonify(payload), 200

    return render_template(
        "predict.html",
        **template_context(steam_name=steam_name, result=result, error=error),
    )


@app.route("/predict/form", methods=["GET"])
def predict_form() -> str:
    """HTML form entry point for live analysis."""
    return render_template("predict.html", **template_context())


@app.route("/health", methods=["GET"])
def health() -> tuple[Any, int]:
    """Liveness probe; requires demo samples and engagement index module."""
    demo_ok = DEMO_SAMPLES_PATH.is_file()
    index_ok = True
    try:
        compute_engagement_score(
            {name: 0.0 for name in INDEX_COMPONENTS},
            default_reference_stats(),
        )
    except Exception as exc:
        logger.error("Engagement index health check failed: %s", exc)
        index_ok = False

    status = "ok" if demo_ok and index_ok else "degraded"
    code = 200 if demo_ok and index_ok else 503
    return (
        jsonify(
            {
                "status": status,
                "demo_samples_available": demo_ok,
                "engagement_index_available": index_ok,
            }
        ),
        code,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
