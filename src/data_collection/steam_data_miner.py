"""
Steam Store ETL seeder — database-backed.

Fetches Steam app metadata and seeds the canonical ``Game`` dimension table.
Checkpointing is derived from the database: only AppIDs absent from ``games``
are eligible for ingestion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Final, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from requests import Response
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, JSONDecodeError, RequestException, Timeout
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.database.models import Base, Game, SessionLocal, SteamMetric, engine
from src.database.session import db_session
from src.utils.http import BaseRateLimiter
from src.utils.parsers import normalize_text, parse_positive_int

load_dotenv()

LOG_FILE: Final[str] = "miner.log"
MAX_REQUESTS_PER_MINUTE: Final[int] = 60
REQUESTS_BUFFER: Final[int] = 5
RETRY_LIMIT: Final[int] = 3
RETRY_SLEEP_SECONDS: Final[float] = 5.0
REQUEST_TIMEOUT_SECONDS: Final[float] = 10.0
BATCH_SIZE: Final[int] = 10_000
DEFAULT_MAX_WORKERS: Final[int] = 2
FUZZY_MATCH_CUTOFF: Final[float] = 0.7
MIN_REVIEWS_FOR_SEED: Final[int] = 50
RECENT_REVIEWS_PAGE_SIZE: Final[int] = 20

STEAM_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
STEAM_APP_LIST_URL: Final[str] = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
STEAM_APP_DETAILS_URL: Final[str] = "https://store.steampowered.com/api/appdetails"
STEAM_REVIEW_SUMMARY_URL: Final[str] = "https://store.steampowered.com/appreviews/{appid}"
STEAM_REVIEW_RECENT_URL: Final[str] = "https://store.steampowered.com/appreviews/{appid}"
STEAM_CURRENT_PLAYERS_URL: Final[str] = (
    "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
)
STEAM_STORE_PAGE_URL: Final[str] = "https://store.steampowered.com/app/{appid}/"

RELEASE_DATE_FORMATS: Final[tuple[str, ...]] = (
    "%d %b, %Y",
    "%b %d, %Y",
    "%d %B, %Y",
    "%B %d, %Y",
    "%Y-%m-%d",
    "%Y",
)

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
class SteamAppListEntry:
    """Lightweight entry from the Steam global app list."""

    appid: int
    name: str


@dataclass(frozen=True, slots=True)
class AppDetailsData:
    """Parsed fields from the Steam appdetails endpoint."""

    steam_name: str
    release_date: Optional[dt.date]
    supported_languages_raw: str
    dlc_count: int
    has_achievements: bool


@dataclass(frozen=True, slots=True)
class ReviewSummaryData:
    """Parsed fields from the Steam review summary endpoint."""

    total_reviews: Optional[int]
    positive_rate: Optional[float]


@dataclass(frozen=True, slots=True)
class StorePageData:
    """Parsed fields from the Steam store HTML page."""

    wishlist_count: Optional[int]
    supported_languages: str


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    """
    Normalized Steam game payload ready for ``Game`` ORM persistence.

    Enrichment fields are persisted on ``SteamMetric``; only dimension keys
    land on ``Game``.
    """

    appid: int
    steam_name: str
    release_date: Optional[dt.date]
    total_reviews: Optional[int]
    positive_rate: Optional[float]
    sentiment_score: Optional[float]
    current_players: Optional[int]
    wishlist_count: Optional[int]
    supported_languages: Optional[str]

    def qualifies_for_seed(self, min_reviews: int = MIN_REVIEWS_FOR_SEED) -> bool:
        """Return True when total_reviews meets the seed threshold (default 50)."""
        return self.total_reviews is not None and self.total_reviews >= min_reviews


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class SteamRateLimiter(BaseRateLimiter):
    """Thread-safe sliding-window limiter for Steam HTTP calls."""

    def __init__(
        self,
        max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE,
        buffer: int = REQUESTS_BUFFER,
    ) -> None:
        super().__init__(
            max_requests_per_minute=max_requests_per_minute,
            buffer=buffer,
        )


# ---------------------------------------------------------------------------
# Parsing helpers (explicit — no silent coercion)
# ---------------------------------------------------------------------------


def parse_non_negative_float(raw: Any, field_name: str) -> Optional[float]:
    """Parse a non-negative float; log and return None on failure."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        logger.warning("Boolean provided for float field %s: %r", field_name, raw)
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value >= 0.0 else None
    if isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Cannot parse %s as float: %r", field_name, raw)
            return None
        return value if value >= 0.0 else None
    logger.warning(
        "Cannot parse %s as float: %r (%s)",
        field_name,
        raw,
        type(raw).__name__,
    )
    return None


def parse_steam_release_date(raw: str) -> Optional[dt.date]:
    """
    Parse Steam release-date strings (e.g. ``12 Mar, 2020`` or ``2020``).

    Returns None when the value is empty or unparseable.
    """
    cleaned = normalize_text(raw, "release_date")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in ("unknown", "coming soon", "tbd", "to be announced"):
        return None
    for fmt in RELEASE_DATE_FORMATS:
        try:
            return dt.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    logger.warning("Unparseable Steam release_date: %r", raw)
    return None


def parse_app_details_payload(
    payload: dict[str, Any],
    appid: int,
) -> Optional[AppDetailsData]:
    """Map appdetails JSON to ``AppDetailsData``; return None when invalid."""
    entry = payload.get(str(appid))
    if not isinstance(entry, dict):
        logger.warning("appdetails missing entry for appid=%s", appid)
        return None
    if not entry.get("success"):
        logger.info("appdetails success=false for appid=%s", appid)
        return None

    data = entry.get("data")
    if not isinstance(data, dict):
        logger.warning("appdetails data block missing for appid=%s", appid)
        return None

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        logger.warning("appdetails missing name for appid=%s", appid)
        return None

    release_raw = ""
    release_info = data.get("release_date")
    if isinstance(release_info, dict):
        date_value = release_info.get("date")
        if isinstance(date_value, str):
            release_raw = date_value

    languages_raw = data.get("supported_languages", "")
    supported_languages_raw = (
        languages_raw if isinstance(languages_raw, str) else str(languages_raw)
    )

    dlc = data.get("dlc")
    dlc_count = len(dlc) if isinstance(dlc, list) else 0

    achievements = data.get("achievements")
    achievement_total = 0
    if isinstance(achievements, dict):
        achievement_total = parse_positive_int(achievements.get("total"), "achievements.total") or 0

    return AppDetailsData(
        steam_name=name.strip(),
        release_date=parse_steam_release_date(release_raw),
        supported_languages_raw=supported_languages_raw,
        dlc_count=dlc_count,
        has_achievements=achievement_total > 0,
    )


def parse_review_summary_payload(payload: dict[str, Any]) -> ReviewSummaryData:
    """Map appreviews summary JSON to ``ReviewSummaryData``."""
    query_summary = payload.get("query_summary")
    if not isinstance(query_summary, dict):
        logger.warning("Review summary missing query_summary: %r", query_summary)
        return ReviewSummaryData(total_reviews=None, positive_rate=None)

    total_reviews = parse_positive_int(
        query_summary.get("total_reviews"),
        "total_reviews",
    )
    review_score = parse_non_negative_float(
        query_summary.get("review_score"),
        "review_score",
    )
    positive_rate: Optional[float] = None
    if review_score is not None and total_reviews:
        positive_rate = review_score * 10.0

    return ReviewSummaryData(
        total_reviews=total_reviews,
        positive_rate=positive_rate,
    )


def parse_review_sentiment(
    payload: dict[str, Any],
    analyzer: SentimentIntensityAnalyzer,
) -> Optional[float]:
    """Compute average VADER compound score from recent review texts."""
    reviews = payload.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        return None

    texts: list[str] = []
    for review in reviews:
        if not isinstance(review, dict):
            continue
        text = review.get("review")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    if not texts:
        return None

    scores = [float(analyzer.polarity_scores(text)["compound"]) for text in texts]
    return round(sum(scores) / len(scores), 3)


def parse_current_players_payload(payload: dict[str, Any]) -> Optional[int]:
    """Map GetNumberOfCurrentPlayers JSON to an integer player count."""
    response_block = payload.get("response")
    if not isinstance(response_block, dict):
        logger.warning("Current players response missing 'response' block.")
        return None
    return parse_positive_int(
        response_block.get("player_count"),
        "player_count",
    )


def strip_html(raw: str) -> str:
    """Remove HTML tags from Steam language strings."""
    return re.sub(r"<[^>]+>", "", raw).replace("\n", " ").strip()


def parse_store_page_html(
    html: str,
    supported_languages_raw: str,
) -> StorePageData:
    """Parse wishlist count and languages from a Steam store page."""
    wishlist_count: Optional[int] = None
    supported_languages = strip_html(supported_languages_raw)

    try:
        soup = BeautifulSoup(html, "html.parser")
        wishlist_btn = soup.select_one("#WishlistBtn")
        if wishlist_btn is not None:
            tooltip = wishlist_btn.get("data-tooltip-html")
            if isinstance(tooltip, list):
                tooltip_html = " ".join(str(part) for part in tooltip)
            elif tooltip is not None:
                tooltip_html = str(tooltip)
            else:
                tooltip_html = ""
            match = re.search(r"([\d,]+)", tooltip_html)
            if match:
                wishlist_count = parse_positive_int(
                    match.group(1).replace(",", ""),
                    "wishlist_count",
                )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Failed parsing wishlist HTML: %s", exc)

    if not supported_languages:
        supported_languages = ""

    return StorePageData(
        wishlist_count=wishlist_count,
        supported_languages=supported_languages,
    )


# ---------------------------------------------------------------------------
# Steam HTTP client
# ---------------------------------------------------------------------------


class SteamAPIClient:
    """Thread-safe Steam Store client with retries, pacing, and typed parsers."""

    def __init__(
        self,
        limiter: SteamRateLimiter,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._limiter = limiter
        self._http = session or requests.Session()
        self._http.headers.update({"User-Agent": STEAM_USER_AGENT})
        self._sentiment = SentimentIntensityAnalyzer()

    def _get(
        self,
        url: str,
        params: Optional[dict[str, str | int]] = None,
        endpoint: str = "",
    ) -> Optional[Response]:
        """Perform a rate-limited GET with retries; return None after exhaustion."""
        label = endpoint or url
        last_error: Optional[Exception] = None
        for attempt in range(1, RETRY_LIMIT + 1):
            self._limiter.wait_if_needed()
            try:
                response = self._http.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                self._limiter.record_request()
                if response.status_code == 429:
                    logger.warning(
                        "Rate limited (429) on %s attempt %d/%d.",
                        label,
                        attempt,
                        RETRY_LIMIT,
                    )
                    time.sleep(RETRY_SLEEP_SECONDS)
                    last_error = HTTPError(
                        f"429 Too Many Requests for {label}",
                        response=response,
                    )
                    continue
                response.raise_for_status()
                return response
            except HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                logger.warning(
                    "HTTP %s from %s (attempt %d/%d): %s",
                    status,
                    label,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)
            except (RequestsConnectionError, Timeout) as exc:
                logger.warning(
                    "Network error calling %s (attempt %d/%d): %s",
                    label,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)
            except RequestException as exc:
                logger.error(
                    "Request error calling %s (attempt %d/%d): %s",
                    label,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)

        logger.error(
            "Steam GET failed after %d attempts for %s: %s",
            RETRY_LIMIT,
            label,
            last_error,
        )
        return None

    def _get_json(
        self,
        url: str,
        params: Optional[dict[str, str | int]] = None,
        endpoint: str = "",
    ) -> Optional[dict[str, Any]]:
        """GET and parse JSON object; log and return None on failure."""
        response = self._get(url, params=params, endpoint=endpoint)
        if response is None:
            return None
        try:
            payload = response.json()
        except JSONDecodeError as exc:
            logger.warning("Invalid JSON from %s: %s", endpoint or url, exc)
            return None
        if not isinstance(payload, dict):
            logger.warning(
                "Expected JSON object from %s, got %s",
                endpoint or url,
                type(payload).__name__,
            )
            return None
        return payload

    def fetch_app_list(self) -> list[SteamAppListEntry]:
        """Download the global Steam app list."""
        payload = self._get_json(STEAM_APP_LIST_URL, endpoint="GetAppList/v2")
        if payload is None:
            raise RuntimeError("Failed to fetch Steam app list.")

        applist = payload.get("applist")
        if not isinstance(applist, dict):
            raise RuntimeError("Steam app list missing 'applist' object.")

        apps = applist.get("apps")
        if not isinstance(apps, list):
            raise RuntimeError("Steam app list missing 'apps' array.")

        entries: list[SteamAppListEntry] = []
        for item in apps:
            if not isinstance(item, dict):
                continue
            appid = parse_positive_int(item.get("appid"), "appid")
            name = item.get("name")
            if appid is None or not isinstance(name, str) or not name.strip():
                continue
            entries.append(SteamAppListEntry(appid=appid, name=name.strip()))
        logger.info("Fetched %d Steam app list entries.", len(entries))
        return entries

    def fetch_app_details(self, appid: int) -> Optional[AppDetailsData]:
        """Fetch and parse appdetails for a single AppID."""
        payload = self._get_json(
            STEAM_APP_DETAILS_URL,
            params={"appids": appid, "l": "en", "cc": "us"},
            endpoint=f"appdetails/{appid}",
        )
        if payload is None:
            return None
        return parse_app_details_payload(payload, appid)

    def fetch_review_summary(self, appid: int) -> ReviewSummaryData:
        """Fetch and parse the all-language review summary."""
        url = STEAM_REVIEW_SUMMARY_URL.format(appid=appid)
        payload = self._get_json(
            url,
            params={
                "json": "1",
                "language": "all",
                "filter": "all",
                "num_per_page": "1",
            },
            endpoint=f"appreviews/summary/{appid}",
        )
        if payload is None:
            logger.warning("Review summary unavailable for appid=%s", appid)
            return ReviewSummaryData(total_reviews=None, positive_rate=None)
        return parse_review_summary_payload(payload)

    def fetch_review_sentiment(self, appid: int) -> Optional[float]:
        """Fetch recent reviews and compute VADER sentiment."""
        url = STEAM_REVIEW_RECENT_URL.format(appid=appid)
        payload = self._get_json(
            url,
            params={
                "json": "1",
                "language": "all",
                "filter": "recent",
                "num_per_page": str(RECENT_REVIEWS_PAGE_SIZE),
            },
            endpoint=f"appreviews/recent/{appid}",
        )
        if payload is None:
            logger.warning("Recent reviews unavailable for appid=%s", appid)
            return None
        return parse_review_sentiment(payload, self._sentiment)

    def fetch_current_players(self, appid: int) -> Optional[int]:
        """Fetch concurrent player count."""
        payload = self._get_json(
            STEAM_CURRENT_PLAYERS_URL,
            params={"appid": appid},
            endpoint=f"current_players/{appid}",
        )
        if payload is None:
            logger.warning("Current players unavailable for appid=%s", appid)
            return None
        return parse_current_players_payload(payload)

    def fetch_store_page(self, appid: int, supported_languages_raw: str) -> StorePageData:
        """Scrape wishlist and language fields from the store HTML page."""
        url = STEAM_STORE_PAGE_URL.format(appid=appid)
        response = self._get(url, endpoint=f"store_page/{appid}")
        if response is None:
            logger.warning("Store page unavailable for appid=%s", appid)
            return StorePageData(
                wishlist_count=None,
                supported_languages=strip_html(supported_languages_raw),
            )
        return parse_store_page_html(response.text, supported_languages_raw)

    def build_game_snapshot(self, appid: int) -> Optional[GameSnapshot]:
        """
        Aggregate all Steam sources into a ``GameSnapshot``.

        Returns None when appdetails fails (invalid/non-game AppID).
        """
        details = self.fetch_app_details(appid)
        if details is None:
            return None

        review_summary = self.fetch_review_summary(appid)
        sentiment_score = self.fetch_review_sentiment(appid)
        current_players = self.fetch_current_players(appid)
        store_page = self.fetch_store_page(appid, details.supported_languages_raw)

        snapshot = GameSnapshot(
            appid=appid,
            steam_name=details.steam_name,
            release_date=details.release_date,
            total_reviews=review_summary.total_reviews,
            positive_rate=review_summary.positive_rate,
            sentiment_score=sentiment_score,
            current_players=current_players,
            wishlist_count=store_page.wishlist_count,
            supported_languages=store_page.supported_languages or None,
        )
        logger.info(
            "[%d] %s seeded-candidate reviews=%s players=%s",
            appid,
            snapshot.steam_name,
            snapshot.total_reviews,
            snapshot.current_players,
        )
        return snapshot


def ensure_schema() -> None:
    """Create ORM tables when they do not yet exist."""
    Base.metadata.create_all(bind=engine)


def get_mined_appids(session: Session) -> set[int]:
    """
    Return AppIDs already present in ``games``.

    Replaces flat-file ``done_ids.txt`` checkpointing.
    """
    stmt = select(Game.appid)
    return set(session.scalars(stmt).all())


def persist_game_snapshot(
    session: Session,
    snapshot: GameSnapshot,
    min_reviews: int = MIN_REVIEWS_FOR_SEED,
) -> bool:
    """
    Persist Steam enrichment and optionally seed the ``Game`` dimension.

    When the snapshot qualifies (``total_reviews >= min_reviews``), always
    appends one ``SteamMetric`` row. Inserts ``Game`` only when ``appid`` is
    not yet present.

    Returns True when a new ``Game`` row was inserted.
    """
    if not snapshot.qualifies_for_seed(min_reviews):
        logger.debug(
            "Skipping appid=%s (%r): total_reviews=%s below threshold %d.",
            snapshot.appid,
            snapshot.steam_name,
            snapshot.total_reviews,
            min_reviews,
        )
        return False

    mined_at = dt.datetime.now(dt.timezone.utc)
    existing = session.scalars(
        select(Game).where(Game.appid == snapshot.appid)
    ).first()

    game_inserted = False
    if existing is None:
        game = Game(
            appid=snapshot.appid,
            steam_name=snapshot.steam_name,
            release_date=snapshot.release_date,
        )
        session.add(game)
        session.flush()
        game_row = game
        game_inserted = True
        logger.info(
            "Inserted Game appid=%s steam_name=%r release_date=%s.",
            snapshot.appid,
            snapshot.steam_name,
            snapshot.release_date,
        )
    else:
        game_row = existing
        logger.debug(
            "Game appid=%s already seeded; appending SteamMetric only.",
            snapshot.appid,
        )

    session.add(
        SteamMetric(
            game_id=game_row.id,
            mined_at=mined_at,
            total_reviews=snapshot.total_reviews,
            positive_rate=snapshot.positive_rate,
            review_sentiment=snapshot.sentiment_score,
            current_players=snapshot.current_players,
            wishlist_count=snapshot.wishlist_count,
            supported_languages=snapshot.supported_languages,
        )
    )
    session.flush()
    logger.info(
        "Inserted SteamMetric for appid=%s game_id=%s mined_at=%s.",
        snapshot.appid,
        game_row.id,
        mined_at.isoformat(),
    )
    return game_inserted


def persist_snapshots(
    session: Session,
    snapshots: list[GameSnapshot],
    min_reviews: int = MIN_REVIEWS_FOR_SEED,
) -> int:
    """Persist a batch of snapshots; return insert count."""
    inserted = 0
    for snapshot in snapshots:
        if persist_game_snapshot(session, snapshot, min_reviews=min_reviews):
            inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Concurrency helpers
# ---------------------------------------------------------------------------


def fetch_snapshot_for_appid(appid: int, limiter: SteamRateLimiter) -> Optional[GameSnapshot]:
    """
    Worker entrypoint: build an isolated client and return a snapshot.

    Each worker uses its own ``requests.Session``; pacing is shared via ``limiter``.
    """
    client = SteamAPIClient(limiter=limiter, session=requests.Session())
    return client.build_game_snapshot(appid)


def fetch_snapshots_concurrent(
    appids: list[int],
    limiter: SteamRateLimiter,
    max_workers: int,
) -> tuple[list[GameSnapshot], int]:
    """
    Fetch snapshots concurrently without mutating shared collections in workers.

    Returns (valid_snapshots, invalid_count).
    """
    snapshots: list[GameSnapshot] = []
    invalid_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_snapshot_for_appid, appid, limiter): appid
            for appid in appids
        }
        for future in as_completed(futures):
            appid = futures[future]
            try:
                snapshot = future.result()
            except (RuntimeError, RequestException, SQLAlchemyError) as exc:
                logger.error(
                    "Worker failed for appid=%s: %s",
                    appid,
                    exc,
                    exc_info=True,
                )
                invalid_count += 1
                continue
            if snapshot is None:
                invalid_count += 1
            else:
                snapshots.append(snapshot)

    return snapshots, invalid_count


def process_appid_batch(
    appids: list[int],
    limiter: SteamRateLimiter,
    max_workers: int,
    min_reviews: int,
) -> int:
    """
    Fetch a batch concurrently and persist results on the main thread.

    Returns the number of ``Game`` rows inserted.
    """
    if not appids:
        return 0

    logger.info(
        "Processing AppID batch %s → %s (%d ids).",
        appids[0],
        appids[-1],
        len(appids),
    )

    retry_once = False
    while True:
        snapshots, invalid_count = fetch_snapshots_concurrent(
            appids,
            limiter=limiter,
            max_workers=max_workers,
        )

        if invalid_count >= len(appids):
            if not retry_once:
                logger.warning(
                    "Batch %s→%s all invalid; retrying once.",
                    appids[0],
                    appids[-1],
                )
                retry_once = True
                continue
            logger.warning(
                "Batch %s→%s still invalid after retry; skipping.",
                appids[0],
                appids[-1],
            )
            return 0
        break

    with db_session() as session:
        return persist_snapshots(session, snapshots, min_reviews=min_reviews)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def resolve_appid_by_name(
    entries: list[SteamAppListEntry],
    query: str,
) -> Optional[SteamAppListEntry]:
    """Fuzzy-match a game name against the Steam app list."""
    names = [entry.name for entry in entries]
    matches = difflib.get_close_matches(query, names, n=1, cutoff=FUZZY_MATCH_CUTOFF)
    if not matches:
        return None
    best_name = matches[0]
    for entry in entries:
        if entry.name == best_name:
            return entry
    return None


def run_single_game(
    client: SteamAPIClient,
    game_query: str,
    min_reviews: int,
) -> None:
    """Seed one game by fuzzy name match against the Steam app list."""
    entries = client.fetch_app_list()
    match = resolve_appid_by_name(entries, game_query)
    if match is None:
        raise ValueError(f"No Steam app list match for game query={game_query!r}.")

    logger.info(
        "Single-game seed: %r → appid=%s (%r).",
        game_query,
        match.appid,
        match.name,
    )
    snapshot = client.build_game_snapshot(match.appid)
    if snapshot is None:
        raise RuntimeError(f"Failed to fetch Steam data for appid={match.appid}.")

    with db_session() as session:
        inserted = persist_game_snapshot(session, snapshot, min_reviews=min_reviews)
    if not inserted:
        logger.warning(
            "Game not inserted (already exists or below review threshold): appid=%s",
            match.appid,
        )
    else:
        logger.info("Single-game Steam seed complete for %r.", match.name)


def run_full_seed(
    limiter: SteamRateLimiter,
    max_workers: int,
    batch_size: int,
    min_reviews: int,
) -> None:
    """Seed all Steam AppIDs not yet present in the database."""
    list_client = SteamAPIClient(limiter=limiter, session=requests.Session())
    entries = list_client.fetch_app_list()
    all_appids = [entry.appid for entry in entries]
    logger.info("Steam app list contains %d AppIDs.", len(all_appids))

    with db_session() as session:
        mined_appids = get_mined_appids(session)

    pending_appids = [appid for appid in all_appids if appid not in mined_appids]
    logger.info(
        "%d AppIDs already seeded; %d remaining.",
        len(mined_appids),
        len(pending_appids),
    )

    total_inserted = 0
    for start in range(0, len(pending_appids), batch_size):
        batch = pending_appids[start : start + batch_size]
        inserted = process_appid_batch(
            batch,
            limiter=limiter,
            max_workers=max_workers,
            min_reviews=min_reviews,
        )
        total_inserted += inserted
        logger.info(
            "Batch progress: %d/%d pending processed; %d inserted this run so far.",
            min(start + len(batch), len(pending_appids)),
            len(pending_appids),
            total_inserted,
        )

    logger.info("Steam seed complete. Inserted %d new Game row(s).", total_inserted)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Seed the Game dimension table from Steam Store data.",
    )
    parser.add_argument(
        "--game",
        type=str,
        help="Seed one game by fuzzy-matching this name against the Steam app list.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent fetch workers (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"AppIDs per processing batch (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=MIN_REVIEWS_FOR_SEED,
        help=f"Minimum total_reviews required to insert (default: {MIN_REVIEWS_FOR_SEED}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint."""
    configure_logging()
    args = parse_args(argv)

    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.min_reviews < 0:
        raise ValueError("--min-reviews must be >= 0.")

    ensure_schema()
    limiter = SteamRateLimiter()
    client = SteamAPIClient(limiter=limiter, session=requests.Session())

    if args.game:
        run_single_game(client, args.game.strip(), min_reviews=args.min_reviews)
        return

    run_full_seed(
        limiter=limiter,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        min_reviews=args.min_reviews,
    )


if __name__ == "__main__":
    main()
