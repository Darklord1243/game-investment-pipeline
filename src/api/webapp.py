"""
Database-backed game investment inference API.

Single Flask entry point: mines today's metrics for one Steam title, builds
SQL features, and returns a standardized ``target_score`` (1–100) with top
feature contributions.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from googleapiclient.errors import HttpError
from prawcore.exceptions import TooManyRequests
from requests.exceptions import HTTPError as RequestsHTTPError
from sklearn.ensemble import VotingRegressor
from sqlalchemy import func, select
from sqlalchemy.orm import Session

# Ensure project root is importable when launched as ``python src/api/webapp.py``.
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
from src.features.sql_feature_engineer import DatabaseFeatureEngineer  # noqa: E402
from src.models.model_trainer import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    GameInvestmentPredictor,
    ModelArtifact,
)

load_dotenv()

logger = logging.getLogger(__name__)

ARTIFACT_PATH: Final[Path] = _PROJECT_ROOT / DEFAULT_ARTIFACT_PATH
TOP_FEATURE_COUNT: Final[int] = 10
STEAM_NAME_MAX_LEN: Final[int] = 512
STEAM_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[\w\s\-:''.,&!()+®™]+$",
    re.UNICODE,
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Game Investment Inference</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2.5rem; background: #f4f6f8; color: #1a1a1a; }
    .card { max-width: 640px; background: #fff; padding: 1.5rem 2rem; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,.08); }
    label { display: block; font-weight: 600; margin-bottom: .35rem; }
    input[type=text] { width: 100%; padding: .55rem .65rem; border: 1px solid #ccc;
                      border-radius: 4px; box-sizing: border-box; }
    button { margin-top: 1rem; padding: .55rem 1.2rem; background: #1565c0; color: #fff;
             border: none; border-radius: 4px; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    pre { margin-top: 1.25rem; background: #f0f4f8; padding: 1rem; border-radius: 6px;
          overflow-x: auto; font-size: .85rem; }
    .error { color: #b71c1c; margin-top: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Game Investment Inference</h1>
    <p>Submit a Steam store title. The API mines today's metrics, engineers SQL features,
       and returns a <code>target_score</code> (1–100).</p>
    <form id="predict-form">
      <label for="steam_name">Steam game name</label>
      <input id="steam_name" name="steam_name" type="text" required
             placeholder="e.g. Counter-Strike 2" maxlength="512">
      <button type="submit" id="submit-btn">Predict</button>
    </form>
    <div id="error" class="error" hidden></div>
    <pre id="result" hidden></pre>
  </div>
  <script>
    document.getElementById('predict-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('submit-btn');
      const errEl = document.getElementById('error');
      const outEl = document.getElementById('result');
      errEl.hidden = true;
      outEl.hidden = true;
      btn.disabled = true;
      try {
        const steam_name = document.getElementById('steam_name').value.trim();
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ steam_name }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          errEl.textContent = data.error || resp.statusText;
          errEl.hidden = false;
        } else {
          outEl.textContent = JSON.stringify(data, null, 2);
          outEl.hidden = false;
        }
      } catch (err) {
        errEl.textContent = String(err);
        errEl.hidden = false;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


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


def top_contributing_features(
    model: VotingRegressor,
    feature_names: list[str],
    x_scaled: np.ndarray,
    *,
    top_n: int = TOP_FEATURE_COUNT,
) -> dict[str, float]:
    """
    Rank features by average tree importance weighted by scaled magnitude.

    Returns:
        Mapping of feature name → contribution score (descending, top *top_n*).
    """
    if not feature_names or x_scaled.size == 0:
        return {}

    importances = np.zeros(len(feature_names), dtype=np.float64)
    estimator_count = 0
    for _name, estimator in model.named_estimators_.items():
        raw = getattr(estimator, "feature_importances_", None)
        if raw is None:
            continue
        importances += np.asarray(raw, dtype=np.float64)
        estimator_count += 1

    if estimator_count == 0 or importances.sum() <= 0:
        return {}

    importances /= estimator_count
    row = np.asarray(x_scaled, dtype=np.float64).reshape(-1)
    if row.shape[0] != len(feature_names):
        row = row[: len(feature_names)]
    contributions = importances[: row.shape[0]] * np.abs(row)
    order = np.argsort(contributions)[::-1]
    result: dict[str, float] = {}
    for idx in order[:top_n]:
        score = float(contributions[idx])
        if score <= 0:
            break
        result[feature_names[idx]] = round(score, 6)
    return result


def build_predictor(artifact: ModelArtifact) -> GameInvestmentPredictor:
    """Hydrate a :class:`GameInvestmentPredictor` from a loaded artifact."""
    predictor = GameInvestmentPredictor(artifact_path=ARTIFACT_PATH)
    predictor.best_model = artifact.best_model
    predictor.feature_names = list(artifact.feature_names)
    predictor.scaler = artifact.scaler
    return predictor


def run_inference(steam_name: str, batch_date: dt.date) -> dict[str, Any]:
    """
    Full inference pipeline: mine → feature matrix → predict → explain.

    Returns:
        JSON-serializable response payload.

    Raises:
        GameNotFoundError: Title missing from Steam or today's feature matrix.
        QuotaExhaustedError: Platform quota exhausted.
        FileNotFoundError: Model artifact missing.
        RuntimeError: Model not loaded or prediction failed.
    """
    canonical_name = run_platform_miners(steam_name)

    with SessionLocal() as session:
        features = DatabaseFeatureEngineer().build_feature_matrix(session, batch_date)

    game_df = filter_game_row(features, canonical_name)
    if game_df.empty:
        raise GameNotFoundError(
            f"No feature row for {canonical_name!r} on batch_date={batch_date.isoformat()}. "
            "Mining may have produced no metrics for today."
        )

    artifact = GameInvestmentPredictor.load_artifact(ARTIFACT_PATH)
    predictor = build_predictor(artifact)

    x = game_df.reindex(columns=predictor.feature_names, fill_value=0.0).astype("float64")
    x_scaled = predictor.scaler.transform(x)  # type: ignore[union-attr]
    raw_score = float(predictor.predict(game_df)[0])
    target_score = round(float(np.clip(raw_score, 1.0, 100.0)), 2)

    top_features = top_contributing_features(
        artifact.best_model,
        predictor.feature_names,
        x_scaled,
    )

    return {
        "steam_name": canonical_name,
        "requested_name": steam_name,
        "batch_date": batch_date.isoformat(),
        "target_score": target_score,
        "top_features": top_features,
    }


configure_logging()
ensure_database_schema()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


@app.route("/", methods=["GET"])
def index() -> str:
    """Minimal UI for manual inference requests."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Any, int]:
    """
    Mine today's metrics for one game and return investment potential.

    Request JSON: ``{"steam_name": "<Steam store title>"}``

    Response JSON: ``target_score`` (1–100) and ``top_features``.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be JSON."}), 400

    try:
        steam_name = validate_steam_name(payload.get("steam_name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    batch_date = utc_today()

    try:
        result = run_inference(steam_name, batch_date)
        return jsonify(result), 200
    except GameNotFoundError as exc:
        logger.warning("Game not found for %r: %s", steam_name, exc)
        return jsonify({"error": str(exc)}), 404
    except QuotaExhaustedError as exc:
        logger.warning("Quota exhausted for %r: %s", steam_name, exc)
        return jsonify({"error": str(exc)}), 429
    except FileNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.exception("Inference failed for %r", steam_name)
        return jsonify({"error": f"Inference failed: {exc}"}), 500


@app.route("/health", methods=["GET"])
def health() -> tuple[Any, int]:
    """Liveness probe; reports artifact availability."""
    artifact_ok = ARTIFACT_PATH.is_file()
    status = "ok" if artifact_ok else "degraded"
    code = 200 if artifact_ok else 503
    return jsonify({"status": status, "artifact_loaded": artifact_ok}), code


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
