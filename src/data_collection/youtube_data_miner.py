"""
YouTube Data API v3 ETL miner — database-backed.

Extracts video snapshots per ``Game`` and persists ``YouTubeMetric`` rows.
Checkpointing is derived from the database: games without a metric for the
current UTC batch day are eligible for mining.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Final, Optional

import requests
from dotenv import load_dotenv
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, RequestException, Timeout
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.database.models import Base, Game, SessionLocal, YouTubeMetric, engine
from src.database.session import db_session
from src.utils.http import BaseRateLimiter
from src.utils.parsers import normalize_text, parse_positive_int

load_dotenv()

LOG_FILE: Final[str] = "miner.log"
MAX_REQUESTS_PER_MINUTE: Final[int] = 60
REQUESTS_BUFFER: Final[int] = 5
RETRY_LIMIT: Final[int] = 3
RETRY_SLEEP_SECONDS: Final[float] = 5.0
QUOTA_SLEEP_SECONDS: Final[float] = 60.0
SEARCH_PAGE_SIZE: Final[int] = 5
MAX_COMMENTS_PER_VIDEO: Final[int] = 20
RYD_API_URL: Final[str] = "https://returnyoutubedislikeapi.com/votes"
RYD_TIMEOUT_SECONDS: Final[float] = 5.0

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
class VideoSnapshot:
    """Normalized YouTube video payload ready for ORM persistence."""

    video_id: str
    title: str
    description: str
    published_at: Optional[dt.datetime]
    duration_iso8601: str
    tags: str
    view_count: int
    like_count: Optional[int]
    dislike_count: Optional[int]
    comment_count: Optional[int]
    channel_title: str
    channel_subscriber_count: Optional[int]
    channel_video_count: Optional[int]
    channel_view_count: Optional[int]
    avg_comment_sentiment: float
    pos_comment_ratio: float
    neg_comment_ratio: float
    thumbnail_url: str


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class YouTubeRateLimiter(BaseRateLimiter):
    """Sliding-window pacer for YouTube Data API v3 and RYD HTTP calls."""

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


def utc_now() -> dt.datetime:
    """Return the current UTC naive datetime (stored as UTC in SQLite)."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def utc_day_bounds(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Inclusive start and exclusive end for a UTC calendar day."""
    start = dt.datetime.combine(day, dt.time.min)
    end = start + dt.timedelta(days=1)
    return start, end


def parse_youtube_timestamp(raw: str) -> Optional[dt.datetime]:
    """
    Parse YouTube ISO-8601 timestamps (e.g. ``2025-07-11T08:25:46Z``).

    Returns None when the value is empty or malformed.
    """
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        logger.warning("Malformed YouTube timestamp: %r", raw)
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def analyze_comments_sentiment(comments: list[str]) -> tuple[float, float, float]:
    """Return average compound score and positive/negative comment ratios."""
    if not comments:
        return 0.0, 0.0, 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(comment)["compound"] for comment in comments]
    avg = sum(scores) / len(scores)
    pos = sum(1 for score in scores if score > 0.05) / len(scores)
    neg = sum(1 for score in scores if score < -0.05) / len(scores)
    return avg, pos, neg


def is_youtube_quota_error(exc: HttpError) -> bool:
    """Return True when the error payload indicates quota exhaustion."""
    try:
        payload = exc.error_details if hasattr(exc, "error_details") else []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    reason = item.get("reason", "")
                    if reason in ("quotaExceeded", "userRateLimitExceeded"):
                        return True
        content = exc.content.decode("utf-8") if exc.content else ""
        if "quotaExceeded" in content or "userRateLimitExceeded" in content:
            return True
    except (AttributeError, UnicodeDecodeError) as decode_exc:
        logger.warning("Could not inspect YouTube quota error: %s", decode_exc)
    return exc.resp is not None and exc.resp.status == 403


# ---------------------------------------------------------------------------
# Return YouTube Dislike (RYD) helper
# ---------------------------------------------------------------------------


def fetch_ryd_dislike_count(
    video_id: str,
    session: Optional[requests.Session] = None,
    limiter: Optional[YouTubeRateLimiter] = None,
) -> Optional[int]:
    """
    Fetch dislike count from the Return YouTube Dislike API.

    Exported for optional backfill utilities; miners call via ``YouTubeAPIClient``.
    """
    if not video_id:
        return None
    http = session or requests.Session()
    rate_limiter = limiter or YouTubeRateLimiter()
    rate_limiter.wait_if_needed()
    try:
        response = http.get(
            RYD_API_URL,
            params={"videoId": video_id},
            timeout=RYD_TIMEOUT_SECONDS,
        )
        rate_limiter.record_request()
        if response.status_code != 200:
            logger.warning(
                "RYD API returned %s for video_id=%s",
                response.status_code,
                video_id,
            )
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        return parse_positive_int(payload.get("dislikes"), "dislikes")
    except (RequestsConnectionError, Timeout) as exc:
        logger.warning("RYD network error for video_id=%s: %s", video_id, exc)
        return None
    except HTTPError as exc:
        logger.warning("RYD HTTP error for video_id=%s: %s", video_id, exc)
        return None
    except RequestException as exc:
        logger.warning("RYD request error for video_id=%s: %s", video_id, exc)
        return None
    except ValueError as exc:
        logger.warning("RYD invalid JSON for video_id=%s: %s", video_id, exc)
        return None


# ---------------------------------------------------------------------------
# YouTube Data API client
# ---------------------------------------------------------------------------


class YouTubeAPIClient:
    """Thin, typed wrapper around YouTube Data API v3 with retries and rate limiting."""

    def __init__(
        self,
        api_key: str,
        limiter: Optional[YouTubeRateLimiter] = None,
        http_session: Optional[requests.Session] = None,
    ) -> None:
        self._api_key = api_key
        self._limiter = limiter or YouTubeRateLimiter()
        self._http = http_session or requests.Session()
        self._youtube: Resource = build(
            "youtube",
            "v3",
            developerKey=api_key,
            cache_discovery=False,
        )

    def _execute_with_retries(
        self,
        request: Any,
        endpoint: str,
    ) -> dict[str, Any]:
        """Execute a googleapiclient request with pacing and retries."""
        last_error: Optional[Exception] = None
        for attempt in range(1, RETRY_LIMIT + 1):
            self._limiter.wait_if_needed()
            run_id = str(uuid.uuid4())
            mined_at = utc_now()
            try:
                response = request.execute()
                self._limiter.record_request()
                if not isinstance(response, dict):
                    raise ValueError(
                        f"Expected dict from {endpoint}, got {type(response).__name__}"
                    )
                logger.debug(
                    "YouTube %s succeeded run_id=%s mined_at=%s",
                    endpoint,
                    run_id,
                    mined_at.isoformat(),
                )
                return response
            except HttpError as exc:
                if is_youtube_quota_error(exc):
                    logger.warning(
                        "YouTube quota exceeded on %s attempt %d/%d; sleeping.",
                        endpoint,
                        attempt,
                        RETRY_LIMIT,
                    )
                    time.sleep(QUOTA_SLEEP_SECONDS)
                    last_error = exc
                    continue
                status = exc.resp.status if exc.resp is not None else "?"
                logger.warning(
                    "HTTP %s from %s (attempt %d/%d): %s",
                    status,
                    endpoint,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)
            except (RequestsConnectionError, Timeout) as exc:
                logger.warning(
                    "Network error calling %s (attempt %d/%d): %s",
                    endpoint,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)
            except RequestException as exc:
                logger.error(
                    "Request error calling %s (attempt %d/%d): %s",
                    endpoint,
                    attempt,
                    RETRY_LIMIT,
                    exc,
                )
                last_error = exc
                time.sleep(RETRY_SLEEP_SECONDS)

        raise RuntimeError(
            f"YouTube request failed after {RETRY_LIMIT} attempts: {endpoint}"
        ) from last_error

    def search_videos(self, query: str, max_results: int = SEARCH_PAGE_SIZE) -> list[dict[str, Any]]:
        """Search videos by game title."""
        request = self._youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
        )
        payload = self._execute_with_retries(request, "search.list")
        items = payload.get("items")
        if not isinstance(items, list):
            logger.warning("Unexpected search.list items: %r", items)
            return []
        return [item for item in items if isinstance(item, dict)]

    def get_video_details(self, video_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch snippet, statistics, and content details for video ids."""
        if not video_ids:
            return []
        request = self._youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids),
        )
        payload = self._execute_with_retries(request, "videos.list")
        items = payload.get("items")
        if not isinstance(items, list):
            logger.warning("Unexpected videos.list items: %r", items)
            return []
        return [item for item in items if isinstance(item, dict)]

    def get_channel_details(self, channel_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch channel metadata keyed by channel id."""
        if not channel_ids:
            return {}
        request = self._youtube.channels().list(
            part="snippet,statistics",
            id=",".join(channel_ids),
        )
        payload = self._execute_with_retries(request, "channels.list")
        items = payload.get("items")
        if not isinstance(items, list):
            logger.warning("Unexpected channels.list items: %r", items)
            return {}
        result: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            channel_id = item.get("id")
            if isinstance(channel_id, str) and channel_id:
                result[channel_id] = item
        return result

    def get_video_comments(self, video_id: str, max_comments: int = MAX_COMMENTS_PER_VIDEO) -> list[str]:
        """Fetch top-level comment text for a video."""
        comments: list[str] = []
        page_token: Optional[str] = None
        while len(comments) < max_comments:
            request = self._youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                textFormat="plainText",
                pageToken=page_token,
            )
            try:
                payload = self._execute_with_retries(request, "commentThreads.list")
            except HttpError as exc:
                status = exc.resp.status if exc.resp is not None else None
                if status == 403:
                    logger.info(
                        "Comments disabled or forbidden for video_id=%s; skipping.",
                        video_id,
                    )
                    break
                raise
            items = payload.get("items")
            if not isinstance(items, list) or not items:
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                snippet = item.get("snippet")
                if not isinstance(snippet, dict):
                    continue
                top_level = snippet.get("topLevelComment")
                if not isinstance(top_level, dict):
                    continue
                top_snippet = top_level.get("snippet")
                if not isinstance(top_snippet, dict):
                    continue
                text = top_snippet.get("textDisplay")
                if isinstance(text, str) and text:
                    comments.append(text)
                if len(comments) >= max_comments:
                    break
            page_token = payload.get("nextPageToken")
            if not isinstance(page_token, str) or not page_token:
                break
        return comments

    def select_most_relevant_video(
        self,
        game_name: str,
        search_results: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """
        Select the most relevant search hit: title contains game name (case-insensitive),
        highest view count among matches, else first result.
        """
        if not search_results:
            return None

        matches: list[dict[str, Any]] = []
        for item in search_results:
            snippet = item.get("snippet")
            if not isinstance(snippet, dict):
                continue
            title = snippet.get("title")
            if isinstance(title, str) and game_name.lower() in title.lower():
                matches.append(item)

        candidates = matches if matches else search_results
        video_ids: list[str] = []
        for item in candidates:
            item_id = item.get("id")
            if not isinstance(item_id, dict):
                continue
            video_id = item_id.get("videoId")
            if isinstance(video_id, str) and video_id:
                video_ids.append(video_id)

        if not video_ids:
            return candidates[0]

        details = self.get_video_details(video_ids)
        id_to_views: dict[str, int] = {}
        for detail in details:
            if not isinstance(detail, dict):
                continue
            vid = detail.get("id")
            stats = detail.get("statistics")
            if not isinstance(vid, str) or not isinstance(stats, dict):
                continue
            views = parse_positive_int(stats.get("viewCount"), "viewCount") or 0
            id_to_views[vid] = views

        def views_for_item(item: dict[str, Any]) -> int:
            item_id = item.get("id")
            if not isinstance(item_id, dict):
                return 0
            video_id = item_id.get("videoId")
            if not isinstance(video_id, str):
                return 0
            return id_to_views.get(video_id, 0)

        return max(candidates, key=views_for_item)

    def build_video_snapshot(
        self,
        game_name: str,
        search_item: dict[str, Any],
    ) -> Optional[VideoSnapshot]:
        """Map search + detail payloads to a normalized ``VideoSnapshot``."""
        item_id = search_item.get("id")
        if not isinstance(item_id, dict):
            logger.warning("Search item missing id for game=%r", game_name)
            return None
        video_id = item_id.get("videoId")
        if not isinstance(video_id, str) or not video_id:
            logger.warning("Search item missing videoId for game=%r", game_name)
            return None

        details = self.get_video_details([video_id])
        if not details:
            logger.warning("No video details for video_id=%s", video_id)
            return None

        video = details[0]
        snippet = video.get("snippet")
        stats = video.get("statistics")
        content = video.get("contentDetails")
        if not isinstance(snippet, dict):
            logger.warning("Video %s missing snippet", video_id)
            return None
        if not isinstance(stats, dict):
            stats = {}
        if not isinstance(content, dict):
            content = {}

        view_count = parse_positive_int(stats.get("viewCount"), "viewCount")
        if view_count is None:
            logger.warning("Video %s missing view_count; skipping.", video_id)
            return None

        channel_id = snippet.get("channelId")
        channel_title = ""
        channel_subscriber_count: Optional[int] = None
        channel_video_count: Optional[int] = None
        channel_view_count: Optional[int] = None
        if isinstance(channel_id, str) and channel_id:
            channels = self.get_channel_details([channel_id])
            channel_info = channels.get(channel_id, {})
            channel_snippet = channel_info.get("snippet")
            channel_stats = channel_info.get("statistics")
            if isinstance(channel_snippet, dict):
                title = channel_snippet.get("title")
                channel_title = title if isinstance(title, str) else ""
            if isinstance(channel_stats, dict):
                channel_subscriber_count = parse_positive_int(
                    channel_stats.get("subscriberCount"),
                    "subscriberCount",
                )
                channel_video_count = parse_positive_int(
                    channel_stats.get("videoCount"),
                    "videoCount",
                )
                channel_view_count = parse_positive_int(
                    channel_stats.get("viewCount"),
                    "viewCount",
                )

        published_raw = snippet.get("publishedAt")
        published_at = (
            parse_youtube_timestamp(published_raw)
            if isinstance(published_raw, str)
            else None
        )

        duration_raw = content.get("duration")
        duration_iso8601 = duration_raw if isinstance(duration_raw, str) else ""

        tags_raw = snippet.get("tags")
        if isinstance(tags_raw, list):
            tags = ",".join(str(tag) for tag in tags_raw if tag is not None)
        else:
            tags = ""

        comments = self.get_video_comments(video_id, max_comments=MAX_COMMENTS_PER_VIDEO)
        avg_sentiment, pos_ratio, neg_ratio = analyze_comments_sentiment(comments)
        dislike_count = fetch_ryd_dislike_count(
            video_id,
            session=self._http,
            limiter=self._limiter,
        )

        title = normalize_text(snippet.get("title"), "title")
        description = normalize_text(snippet.get("description"), "description")

        return VideoSnapshot(
            video_id=video_id,
            title=title,
            description=description,
            published_at=published_at,
            duration_iso8601=duration_iso8601,
            tags=tags,
            view_count=view_count,
            like_count=parse_positive_int(stats.get("likeCount"), "likeCount"),
            dislike_count=dislike_count,
            comment_count=parse_positive_int(stats.get("commentCount"), "commentCount"),
            channel_title=channel_title,
            channel_subscriber_count=channel_subscriber_count,
            channel_video_count=channel_video_count,
            channel_view_count=channel_view_count,
            avg_comment_sentiment=avg_sentiment,
            pos_comment_ratio=pos_ratio,
            neg_comment_ratio=neg_ratio,
            thumbnail_url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        )

    def fetch_video_for_game(self, game_name: str) -> Optional[VideoSnapshot]:
        """Search, select the best video, and return a single snapshot."""
        search_results = self.search_videos(game_name, max_results=SEARCH_PAGE_SIZE)
        best_item = self.select_most_relevant_video(game_name, search_results)
        if best_item is None:
            return None
        return self.build_video_snapshot(game_name, best_item)


def ensure_schema() -> None:
    """Create ORM tables when they do not yet exist."""
    Base.metadata.create_all(bind=engine)


def get_games_pending_for_batch(
    session: Session,
    batch_day: dt.date,
) -> list[Game]:
    """
    Return games with no ``YouTubeMetric`` rows mined on ``batch_day`` (UTC).

    This replaces CSV/checkpoint-file deduplication.
    """
    day_start, day_end = utc_day_bounds(batch_day)
    mined_today = (
        select(YouTubeMetric.game_id)
        .where(
            YouTubeMetric.mined_at >= day_start,
            YouTubeMetric.mined_at < day_end,
        )
        .distinct()
    )
    stmt = select(Game).where(Game.id.not_in(mined_today)).order_by(Game.id)
    return list(session.scalars(stmt).all())


def get_game_by_steam_name(session: Session, steam_name: str) -> Optional[Game]:
    """Look up a game by exact ``steam_name``."""
    stmt = select(Game).where(Game.steam_name == steam_name)
    return session.scalars(stmt).first()


def persist_video_metric(
    session: Session,
    game: Game,
    snapshot: VideoSnapshot,
) -> int:
    """
    Insert one ``YouTubeMetric`` row.

    Returns ``1`` on success. Skips insert when ``video_id`` already exists (unique).
    """
    existing = session.scalars(
        select(YouTubeMetric).where(YouTubeMetric.video_id == snapshot.video_id)
    ).first()
    if existing is not None:
        logger.warning(
            "YouTubeMetric already exists for video_id=%s; skipping insert.",
            snapshot.video_id,
        )
        return 0

    metric = YouTubeMetric(
        game_id=game.id,
        video_id=snapshot.video_id,
        mined_at=utc_now(),
        title=snapshot.title or None,
        description=snapshot.description or None,
        published_at=snapshot.published_at,
        duration_iso8601=snapshot.duration_iso8601 or None,
        tags=snapshot.tags or None,
        view_count=snapshot.view_count,
        like_count=snapshot.like_count,
        dislike_count=snapshot.dislike_count,
        comment_count=snapshot.comment_count,
        channel_title=snapshot.channel_title or None,
        channel_subscriber_count=snapshot.channel_subscriber_count,
        channel_video_count=snapshot.channel_video_count,
        channel_view_count=snapshot.channel_view_count,
        avg_comment_sentiment=snapshot.avg_comment_sentiment,
        pos_comment_ratio=snapshot.pos_comment_ratio,
        neg_comment_ratio=snapshot.neg_comment_ratio,
        thumbnail_url=snapshot.thumbnail_url or None,
    )
    session.add(metric)
    return 1


# ---------------------------------------------------------------------------
# Mining orchestration
# ---------------------------------------------------------------------------


def load_youtube_api_key() -> str:
    """Load YouTube API key from the environment."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY must be set in the environment.")
    return api_key


def mine_game(
    session: Session,
    client: YouTubeAPIClient,
    game: Game,
) -> int:
    """
    Fetch the best-matching video and persist a ``YouTubeMetric`` for one game.

    Returns the number of rows written (0 or 1).
    """
    snapshot = client.fetch_video_for_game(game.steam_name)
    if snapshot is None:
        logger.info(
            "No YouTube video found for game id=%s steam_name=%r.",
            game.id,
            game.steam_name,
        )
        return 0

    count = persist_video_metric(session, game, snapshot)
    if count:
        logger.info(
            "Persisted YouTubeMetric for game id=%s steam_name=%r video_id=%s.",
            game.id,
            game.steam_name,
            snapshot.video_id,
        )
    return count


def run_batch(client: YouTubeAPIClient, batch_day: dt.date) -> None:
    """Mine all games that lack YouTube metrics for ``batch_day``."""
    with db_session() as session:
        pending = get_games_pending_for_batch(session, batch_day)
        pending_ids = [game.id for game in pending]

    total = len(pending_ids)
    if total == 0:
        logger.info("No games pending for YouTube batch on %s.", batch_day.isoformat())
        return

    logger.info(
        "Starting YouTube batch for %s: %d game(s) pending.",
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
                    "[%d/%d] (%.1f%%) Mining YouTube for steam_name=%r",
                    index,
                    total,
                    progress,
                    game.steam_name,
                )
                mine_game(session, client, game)
        except (RuntimeError, SQLAlchemyError, HttpError, RequestException) as exc:
            logger.error(
                "Failed mining game id=%s: %s",
                game_id,
                exc,
                exc_info=True,
            )

    logger.info("YouTube batch for %s complete.", batch_day.isoformat())


def run_single_game(client: YouTubeAPIClient, steam_name: str) -> None:
    """Mine a single game by ``steam_name`` (ignores batch-day checkpoint)."""
    with db_session() as session:
        game = get_game_by_steam_name(session, steam_name)
        if game is None:
            raise ValueError(
                f"No Game record found with steam_name={steam_name!r}. "
                "Seed the games table before mining."
            )
        mine_game(session, client, game)
    logger.info("Single-game YouTube mining finished for %r.", steam_name)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Mine YouTube video metrics into the SQLite warehouse.",
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
    api_key = load_youtube_api_key()
    client = YouTubeAPIClient(api_key=api_key)

    if args.game:
        run_single_game(client, args.game.strip())
        return

    batch_day = parse_batch_date(args.batch_date)
    run_batch(client, batch_day)


if __name__ == "__main__":
    main()
