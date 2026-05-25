"""
Twitch Helix ETL miner — database-backed.

Extracts live stream snapshots per ``Game`` and persists ``TwitchMetric`` rows.
Checkpointing is derived from the database: games without a metric for the
current UTC batch day are eligible for mining.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Final, Optional

import requests
from dotenv import load_dotenv
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, RequestException, Timeout
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.database.models import Base, Game, SessionLocal, TwitchMetric, engine
from src.database.session import db_session
from src.utils.http import BaseRateLimiter
from src.utils.parsers import parse_positive_int

load_dotenv()

LOG_FILE: Final[str] = "miner.log"
MAX_REQUESTS_PER_MINUTE: Final[int] = 800
REQUESTS_BUFFER: Final[int] = 20
RETRY_LIMIT: Final[int] = 3
RETRY_SLEEP_SECONDS: Final[float] = 5.0
RATE_LIMIT_SLEEP_SECONDS: Final[float] = 60.0
FUZZY_MATCH_CUTOFF: Final[float] = 0.7
STREAMS_PAGE_SIZE: Final[int] = 20

TWITCH_TOKEN_URL: Final[str] = "https://id.twitch.tv/oauth2/token"
TWITCH_GAMES_URL: Final[str] = "https://api.twitch.tv/helix/games"
TWITCH_SEARCH_CATEGORIES_URL: Final[str] = "https://api.twitch.tv/helix/search/categories"
TWITCH_STREAMS_URL: Final[str] = "https://api.twitch.tv/helix/streams"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging for file and console output."""
    if logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamSnapshot:
    """Normalized Twitch stream payload ready for ORM persistence."""

    twitch_game_id: str
    streamer_name: str
    viewer_count: int
    stream_title: str
    language: str
    stream_started_at: dt.datetime


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TwitchRateLimiter(BaseRateLimiter):
    """Extends ``BaseRateLimiter`` with Helix header-aware backoff."""

    def __init__(
        self,
        max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
        buffer: int = REQUESTS_BUFFER,
    ) -> None:
        super().__init__(
            max_requests_per_minute=max_requests_per_minute,
            buffer=buffer,
        )

    def apply_response_headers(self, response: Response) -> None:
        """Honor Twitch ``Ratelimit-*`` headers when the bucket is nearly empty."""
        remaining_raw = response.headers.get("Ratelimit-Remaining")
        reset_raw = response.headers.get("Ratelimit-Reset")
        if remaining_raw is None or reset_raw is None:
            return
        try:
            remaining = int(remaining_raw)
            reset_epoch = int(reset_raw)
        except ValueError:
            logger.warning(
                "Non-integer rate-limit headers: remaining=%r reset=%r",
                remaining_raw,
                reset_raw,
            )
            return
        if remaining >= self._buffer:
            return
        sleep_for = reset_epoch - int(time.time())
        if sleep_for > 0:
            logger.info(
                "Helix rate-limit headers low (%d remaining); sleeping %d seconds.",
                remaining,
                sleep_for,
            )
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Parsing helpers (explicit — no silent coercion)
# ---------------------------------------------------------------------------


def utc_now() -> dt.datetime:
    """Return the current UTC naive datetime (stored as UTC in SQLite)."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def utc_day_bounds(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Inclusive start and exclusive end for a UTC calendar day."""
    start = dt.datetime.combine(day, dt.time.min)
    end = start + dt.timedelta(days=1)
    return start, end


def parse_twitch_timestamp(raw: str) -> Optional[dt.datetime]:
    """
    Parse Twitch ISO-8601 timestamps (e.g. ``2025-07-11T08:25:46Z``).

    Returns None when the value is empty or malformed.
    """
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        logger.warning("Malformed Twitch timestamp: %r", raw)
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def parse_helix_stream(item: dict[str, Any], twitch_game_id: str) -> Optional[StreamSnapshot]:
    """Map a Helix stream object to ``StreamSnapshot``; skip invalid rows."""
    streamer = item.get("user_name")
    if not isinstance(streamer, str) or not streamer:
        logger.warning("Stream missing user_name for game_id=%s", twitch_game_id)
        return None

    viewer_count = parse_positive_int(item.get("viewer_count"), "viewer_count")
    if viewer_count is None:
        return None

    title = item.get("title")
    stream_title = title if isinstance(title, str) else ""

    language = item.get("language")
    lang = language if isinstance(language, str) else ""

    started_raw = item.get("started_at")
    started_at = (
        parse_twitch_timestamp(started_raw)
        if isinstance(started_raw, str)
        else None
    )
    if started_at is None:
        logger.warning(
            "Stream for %s missing valid started_at (%r); skipping.",
            streamer,
            started_raw,
        )
        return None

    return StreamSnapshot(
        twitch_game_id=twitch_game_id,
        streamer_name=streamer,
        viewer_count=viewer_count,
        stream_title=stream_title,
        language=lang,
        stream_started_at=started_at,
    )


# ---------------------------------------------------------------------------
# Twitch Helix HTTP client
# ---------------------------------------------------------------------------


class TwitchHelixClient:
    """Thin, typed wrapper around Twitch Helix with retries and rate limiting."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        limiter: Optional[TwitchRateLimiter] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._limiter = limiter or TwitchRateLimiter()
        self._http = session or requests.Session()
        self._access_token: Optional[str] = None

    @property
    def headers(self) -> dict[str, str]:
        if not self._access_token:
            raise RuntimeError("Client is not authenticated; call authenticate() first.")
        return {
            "Client-ID": self._client_id,
            "Authorization": f"Bearer {self._access_token}",
        }

    def authenticate(self) -> str:
        """Obtain an app access token via client credentials."""
        params = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "client_credentials",
        }
        try:
            response = self._http.post(
                TWITCH_TOKEN_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(
                f"Twitch authentication failed: {exc.response.status_code} {exc.response.text}"
            ) from exc
        except (RequestsConnectionError, Timeout) as exc:
            raise RuntimeError(f"Twitch authentication network error: {exc}") from exc
        except RequestException as exc:
            raise RuntimeError(f"Twitch authentication request error: {exc}") from exc

        payload = response.json()
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("Twitch authentication response missing access_token.")
        self._access_token = token
        logger.info("Authenticated with Twitch Helix.")
        return token

    def _get(self, url: str, params: dict[str, str | int]) -> dict[str, Any]:
        """Perform a rate-limited GET with retries; return parsed JSON body."""
        last_error: Optional[Exception] = None
        for attempt in range(1, RETRY_LIMIT + 1):
            self._limiter.wait_if_needed()
            mined_at = utc_now()
            run_id = str(uuid.uuid4())
            try:
                response = self._http.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30,
                )
                self._limiter.record_request()
                self._limiter.apply_response_headers(response)

                if response.status_code == 429:
                    logger.warning(
                        "Rate limited (429) on %s attempt %d/%d; run_id=%s mined_at=%s",
                        url,
                        attempt,
                        RETRY_LIMIT,
                        run_id,
                        mined_at.isoformat(),
                    )
                    time.sleep(RATE_LIMIT_SLEEP_SECONDS)
                    last_error = HTTPError(
                        f"429 Too Many Requests for {url}", response=response
                    )
                    continue

                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Expected JSON object from {url}, got {type(payload).__name__}"
                    )
                logger.debug(
                    "Helix GET %s succeeded run_id=%s mined_at=%s",
                    url,
                    run_id,
                    mined_at.isoformat(),
                )
                return payload

            except HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                logger.warning(
                    "HTTP %s from %s (attempt %d/%d): %s",
                    status,
                    url,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)
            except (RequestsConnectionError, Timeout) as exc:
                logger.warning(
                    "Network error calling %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)
            except RequestException as exc:
                logger.error(
                    "Request error calling %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)

        raise RuntimeError(
            f"Helix request failed after {RETRY_LIMIT} attempts: {url}"
        ) from last_error

    def resolve_game_id(self, game_name: str) -> Optional[str]:
        """Resolve a Twitch category/game id by exact name, then fuzzy search."""
        exact_id = self._fetch_game_id_exact(game_name)
        if exact_id:
            return exact_id
        return self._search_game_id_fuzzy(game_name)

    def _fetch_game_id_exact(self, game_name: str) -> Optional[str]:
        logger.info("Querying Twitch games endpoint for '%s'", game_name)
        payload = self._get(TWITCH_GAMES_URL, {"name": game_name})
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None
        first = data[0]
        if not isinstance(first, dict):
            return None
        game_id = first.get("id")
        return game_id if isinstance(game_id, str) and game_id else None

    def _search_game_id_fuzzy(self, game_name: str) -> Optional[str]:
        logger.info("Fuzzy-searching Twitch categories for '%s'", game_name)
        payload = self._get(TWITCH_SEARCH_CATEGORIES_URL, {"query": game_name})
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None

        candidates: list[tuple[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            game_id = item.get("id")
            if isinstance(name, str) and isinstance(game_id, str):
                candidates.append((name, game_id))

        if not candidates:
            return None

        names = [name for name, _ in candidates]
        matches = difflib.get_close_matches(game_name, names, n=1, cutoff=FUZZY_MATCH_CUTOFF)
        if not matches:
            return None

        best_name = matches[0]
        matched_id = next(game_id for name, game_id in candidates if name == best_name)
        logger.info(
            "Fuzzy matched '%s' to Twitch category '%s' (id=%s)",
            game_name,
            best_name,
            matched_id,
        )
        return matched_id

    def fetch_live_streams(self, twitch_game_id: str) -> list[StreamSnapshot]:
        """Return live stream snapshots for a Twitch game/category id."""
        payload = self._get(
            TWITCH_STREAMS_URL,
            {"game_id": twitch_game_id, "first": STREAMS_PAGE_SIZE},
        )
        data = payload.get("data")
        if not isinstance(data, list):
            logger.warning(
                "Unexpected streams payload for game_id=%s: %r",
                twitch_game_id,
                data,
            )
            return []

        snapshots: list[StreamSnapshot] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            parsed = parse_helix_stream(item, twitch_game_id)
            if parsed is not None:
                snapshots.append(parsed)
        return snapshots


def ensure_schema() -> None:
    """Create ORM tables when they do not yet exist."""
    Base.metadata.create_all(bind=engine)


def get_games_pending_for_batch(
    session: Session,
    batch_day: dt.date,
) -> list[Game]:
    """
    Return games with no ``TwitchMetric`` rows mined on ``batch_day`` (UTC).

    This replaces CSV/checkpoint-file deduplication.
    """
    day_start, day_end = utc_day_bounds(batch_day)
    mined_today = (
        select(TwitchMetric.game_id)
        .where(
            TwitchMetric.mined_at >= day_start,
            TwitchMetric.mined_at < day_end,
        )
        .distinct()
    )
    stmt = select(Game).where(Game.id.not_in(mined_today)).order_by(Game.id)
    return list(session.scalars(stmt).all())


def get_game_by_steam_name(session: Session, steam_name: str) -> Optional[Game]:
    """Look up a game by exact ``steam_name``."""
    stmt = select(Game).where(Game.steam_name == steam_name)
    return session.scalars(stmt).first()


def persist_stream_metrics(
    session: Session,
    game: Game,
    snapshots: list[StreamSnapshot],
) -> int:
    """
    Insert one ``TwitchMetric`` per snapshot.

    Each row receives a unique ``run_id`` and ``mined_at`` timestamp.
    Returns the number of rows inserted.
    """
    inserted = 0
    for snapshot in snapshots:
        metric = TwitchMetric(
            game_id=game.id,
            run_id=str(uuid.uuid4()),
            mined_at=utc_now(),
            twitch_game_id=snapshot.twitch_game_id,
            streamer_name=snapshot.streamer_name,
            stream_started_at=snapshot.stream_started_at,
            stream_title=snapshot.stream_title or None,
            language=snapshot.language or None,
            viewer_count=snapshot.viewer_count,
        )
        session.add(metric)
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Mining orchestration
# ---------------------------------------------------------------------------


def load_twitch_credentials() -> tuple[str, str]:
    """Load Twitch API credentials from the environment."""
    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set in the environment."
        )
    return client_id, client_secret


def mine_game(
    session: Session,
    client: TwitchHelixClient,
    game: Game,
) -> int:
    """
    Resolve Twitch id, fetch live streams, and persist metrics for one game.

    Returns the number of ``TwitchMetric`` rows written.
    """
    twitch_game_id = client.resolve_game_id(game.steam_name)
    if not twitch_game_id:
        logger.warning(
            "No Twitch category found for game id=%s steam_name=%r; skipping.",
            game.id,
            game.steam_name,
        )
        return 0

    snapshots = client.fetch_live_streams(twitch_game_id)
    if not snapshots:
        logger.info(
            "No live streams for game id=%s steam_name=%r (twitch_game_id=%s).",
            game.id,
            game.steam_name,
            twitch_game_id,
        )
        return 0

    count = persist_stream_metrics(session, game, snapshots)
    logger.info(
        "Persisted %d TwitchMetric row(s) for game id=%s steam_name=%r.",
        count,
        game.id,
        game.steam_name,
    )
    return count


def run_batch(client: TwitchHelixClient, batch_day: dt.date) -> None:
    """Mine all games that lack Twitch metrics for ``batch_day``."""
    with db_session() as session:
        pending = get_games_pending_for_batch(session, batch_day)
        pending_ids = [game.id for game in pending]

    total = len(pending_ids)
    if total == 0:
        logger.info("No games pending for Twitch batch on %s.", batch_day.isoformat())
        return

    logger.info(
        "Starting Twitch batch for %s: %d game(s) pending.",
        batch_day.isoformat(),
        total,
    )

    for index, game_id in enumerate(pending_ids, start=1):
        progress = 100.0 * index / total
        try:
            with db_session() as session:
                game = session.get(Game, game_id)
                if game is None:
                    logger.warning("Game id=%s not found; skipping.", game_id)
                    continue
                logger.info(
                    "[%d/%d] (%.1f%%) Mining Twitch streams for steam_name=%r",
                    index,
                    total,
                    progress,
                    game.steam_name,
                )
                mine_game(session, client, game)
        except (RuntimeError, SQLAlchemyError, RequestException) as exc:
            logger.error(
                "Failed mining game id=%s: %s",
                game_id,
                exc,
                exc_info=True,
            )

    logger.info("Twitch batch for %s complete.", batch_day.isoformat())


def run_single_game(client: TwitchHelixClient, steam_name: str) -> None:
    """Mine a single game by ``steam_name`` (ignores batch-day checkpoint)."""
    with db_session() as session:
        game = get_game_by_steam_name(session, steam_name)
        if game is None:
            raise ValueError(
                f"No Game record found with steam_name={steam_name!r}. "
                "Seed the games table before mining."
            )
        mine_game(session, client, game)
    logger.info("Single-game Twitch mining finished for %r.", steam_name)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Mine Twitch live-stream metrics into the SQLite warehouse.",
    )
    parser.add_argument(
        "--game",
        type=str,
        help="Mine one game by steam_name (bypasses batch-day checkpoint).",
    )
    parser.add_argument(
        "--batch-date",
        type=str,
        default=None,
        help="UTC batch day as YYYY-MM-DD (default: today UTC).",
    )
    return parser.parse_args(argv)


def parse_batch_date(raw: Optional[str]) -> dt.date:
    """Parse ``--batch-date`` or default to today's UTC date."""
    if raw is None:
        return utc_now().date()
    try:
        return dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --batch-date {raw!r}; expected YYYY-MM-DD."
        ) from exc


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint."""
    configure_logging()
    args = parse_args(argv)

    ensure_schema()
    client_id, client_secret = load_twitch_credentials()
    client = TwitchHelixClient(client_id=client_id, client_secret=client_secret)
    client.authenticate()

    if args.game:
        run_single_game(client, args.game.strip())
        return

    batch_day = parse_batch_date(args.batch_date)
    run_batch(client, batch_day)


if __name__ == "__main__":
    main()
