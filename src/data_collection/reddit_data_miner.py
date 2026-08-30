"""
Reddit (PRAW) ETL miner — database-backed.

Extracts post snapshots per ``Game`` and persists ``RedditMetric`` rows.
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

import praw
from dotenv import load_dotenv
from praw.exceptions import PRAWException
from praw.models import Comment, Submission
from praw.reddit import Reddit
from prawcore.exceptions import PrawcoreException, TooManyRequests
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.database.models import Base, Game, RedditMetric, SessionLocal, engine
from src.database.session import db_session
from src.utils.http import BaseRateLimiter
from src.utils.parsers import normalize_text, parse_positive_int

load_dotenv()

LOG_FILE: Final[str] = "miner.log"
REQUEST_DELAY_SECONDS: Final[float] = 1.1
INITIAL_BACKOFF_SECONDS: Final[float] = 60.0
MAX_BACKOFF_SECONDS: Final[float] = 300.0
POSTS_PER_SUBREDDIT: Final[int] = 5
MAX_COMMENTS_PER_POST: Final[int] = 20
SUBREDDITS: Final[tuple[str, ...]] = ("gaming", "games", "pcgaming")
EMOTION_MODEL: Final[str] = "j-hartmann/emotion-english-distilroberta-base"

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
class PostSnapshot:
    """Normalized Reddit post payload ready for ORM persistence."""

    subreddit: str
    post_id: str
    title: str
    score: int
    num_comments: int
    created_utc: float
    avg_comment_sentiment: float
    pos_comment_ratio: float
    neg_comment_ratio: float
    analyzed_comment_count: int
    post_url: str
    author_username: Optional[str]
    author_link_karma: Optional[int]
    author_comment_karma: Optional[int]
    author_account_age_days: Optional[int]
    num_awards: Optional[int]
    num_crossposts: Optional[int]
    post_flair: Optional[str]
    is_stickied: Optional[bool]
    is_original_content: Optional[bool]
    unique_commenters: int
    emotion_distribution: dict[str, float]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RedditRateLimiter(BaseRateLimiter):
    """Fixed-delay pacing for Reddit API calls, with exponential backoff from base."""

    def __init__(self, delay_seconds: float = REQUEST_DELAY_SECONDS) -> None:
        super().__init__()
        self._delay_seconds = delay_seconds
        self._last_request_at: Optional[float] = None

    def wait_if_needed(self) -> None:
        """Enforce minimum spacing between consecutive requests."""
        if self._last_request_at is None:
            return
        elapsed = time.time() - self._last_request_at
        remaining = self._delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def record_request(self) -> None:
        """Record that a Reddit API interaction completed."""
        self._last_request_at = time.time()

    @staticmethod
    def sleep_for_backoff(backoff_seconds: float) -> None:
        """Sleep with logging when Reddit signals rate limiting."""
        logger.warning("Reddit rate limit; sleeping %.0f seconds.", backoff_seconds)
        time.sleep(backoff_seconds)


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


def parse_optional_bool(raw: Any, field_name: str) -> Optional[bool]:
    """Parse an optional boolean without silent coercion."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    logger.warning(
        "Cannot parse %s as bool: %r (%s)",
        field_name,
        raw,
        type(raw).__name__,
    )
    return None


def parse_epoch_seconds(raw: Any, field_name: str) -> Optional[float]:
    """Parse a Unix epoch timestamp in seconds."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        logger.warning("Boolean provided for epoch field %s: %r", field_name, raw)
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    logger.warning(
        "Cannot parse %s as epoch seconds: %r (%s)",
        field_name,
        raw,
        type(raw).__name__,
    )
    return None


def analyze_comment_sentiment(
    comments: list[str],
    analyzer: SentimentIntensityAnalyzer,
) -> tuple[float, float, float, int]:
    """Return sentiment aggregates and the number of analyzed comments."""
    if not comments:
        return 0.0, 0.0, 0.0, 0
    scores = [analyzer.polarity_scores(text)["compound"] for text in comments]
    avg = sum(scores) / len(scores)
    pos = sum(1 for score in scores if score > 0.05) / len(scores)
    neg = sum(1 for score in scores if score < -0.05) / len(scores)
    return avg, pos, neg, len(comments)


def build_emotion_distribution(labels: list[str]) -> dict[str, float]:
    """Convert emotion labels into label -> share mapping."""
    if not labels:
        return {}
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    return {label: count / total for label, count in counts.items()}


def extract_comment_texts(
    comments: list[Comment],
    max_comments: int = MAX_COMMENTS_PER_POST,
) -> list[str]:
    """Collect comment bodies up to ``max_comments``."""
    texts: list[str] = []
    for comment in comments:
        if len(texts) >= max_comments:
            break
        body = getattr(comment, "body", None)
        if isinstance(body, str) and body and body not in ("[deleted]", "[removed]"):
            texts.append(body)
    return texts


def count_unique_commenters(comments: list[Comment]) -> int:
    """Count distinct non-deleted comment authors."""
    authors: set[str] = set()
    for comment in comments:
        author = getattr(comment, "author", None)
        if author is None:
            continue
        name = getattr(author, "name", None)
        if isinstance(name, str) and name:
            authors.add(name)
    return len(authors)


# ---------------------------------------------------------------------------
# Reddit API client
# ---------------------------------------------------------------------------


class RedditAPIClient:
    """Thin, typed wrapper around PRAW with pacing, retries, and enrichment."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        username: str,
        password: str,
        limiter: Optional[RedditRateLimiter] = None,
    ) -> None:
        self._limiter = limiter or RedditRateLimiter()
        self._reddit: Reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password,
        )
        self._sentiment = SentimentIntensityAnalyzer()
        self._emotion_classifier: Any = None

    @property
    def emotion_classifier(self) -> Any:
        """Lazily load the HuggingFace emotion classifier."""
        if self._emotion_classifier is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            logger.info(
                "Loading emotion classifier on %s.",
                "cuda" if device == 0 else "cpu",
            )
            self._emotion_classifier = pipeline(
                "text-classification",
                model=EMOTION_MODEL,
                device=device,
            )
        return self._emotion_classifier

    def _replace_more_and_list(
        self,
        submission: Submission,
    ) -> list[Comment]:
        """Expand and list comments with pacing."""
        self._limiter.wait_if_needed()
        submission.comments.replace_more(limit=0)
        self._limiter.record_request()
        return [item for item in submission.comments.list() if isinstance(item, Comment)]

    def _detect_emotions(self, texts: list[str]) -> list[str]:
        """Run batch emotion classification; unknown on failure."""
        if not texts:
            return []
        try:
            results = self.emotion_classifier([text[:512] for text in texts])
            labels: list[str] = []
            for result in results:
                if isinstance(result, dict):
                    label = result.get("label")
                    labels.append(label.lower() if isinstance(label, str) else "unknown")
                else:
                    labels.append("unknown")
            return labels
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Emotion classification failed: %s", exc)
            return ["unknown"] * len(texts)

    def _author_metrics(self, submission: Submission) -> tuple[
        Optional[str],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        """Extract author username, karma, and account age."""
        author = getattr(submission, "author", None)
        if author is None:
            return None, None, None, None
        username = getattr(author, "name", None)
        link_karma = parse_positive_int(
            getattr(author, "link_karma", None),
            "author_link_karma",
        )
        comment_karma = parse_positive_int(
            getattr(author, "comment_karma", None),
            "author_comment_karma",
        )
        created_utc = parse_epoch_seconds(
            getattr(author, "created_utc", None),
            "author_created_utc",
        )
        account_age_days: Optional[int] = None
        if created_utc is not None:
            created_dt = dt.datetime.fromtimestamp(created_utc, tz=dt.timezone.utc)
            account_age_days = (dt.datetime.now(dt.timezone.utc) - created_dt).days
        return (
            username if isinstance(username, str) else None,
            link_karma,
            comment_karma,
            account_age_days,
        )

    def _post_extra_metrics(self, submission: Submission) -> tuple[
        Optional[int],
        Optional[int],
        Optional[str],
        Optional[bool],
        Optional[bool],
    ]:
        """Extract awards, crossposts, flair, and flags."""
        return (
            parse_positive_int(
                getattr(submission, "total_awards_received", None),
                "num_awards",
            ),
            parse_positive_int(
                getattr(submission, "num_crossposts", None),
                "num_crossposts",
            ),
            normalize_text(getattr(submission, "link_flair_text", None), "post_flair")
            or None,
            parse_optional_bool(getattr(submission, "stickied", None), "is_stickied"),
            parse_optional_bool(
                getattr(submission, "is_original_content", None),
                "is_original_content",
            ),
        )

    def parse_submission(
        self,
        submission: Submission,
        subreddit_name: str,
    ) -> Optional[PostSnapshot]:
        """Map a PRAW submission to ``PostSnapshot``; skip invalid rows."""
        post_id = getattr(submission, "id", None)
        if not isinstance(post_id, str) or not post_id:
            logger.warning("Submission missing id in r/%s", subreddit_name)
            return None

        title = normalize_text(getattr(submission, "title", None), "title")
        score = parse_positive_int(getattr(submission, "score", None), "score")
        if score is None:
            score = 0
        num_comments = parse_positive_int(
            getattr(submission, "num_comments", None),
            "num_comments",
        )
        if num_comments is None:
            num_comments = 0

        created_utc = parse_epoch_seconds(
            getattr(submission, "created_utc", None),
            "created_utc",
        )
        if created_utc is None:
            logger.warning("Post %s missing created_utc; skipping.", post_id)
            return None

        permalink = getattr(submission, "permalink", None)
        post_url = (
            f"https://reddit.com{permalink}"
            if isinstance(permalink, str) and permalink
            else ""
        )

        comments = self._replace_more_and_list(submission)
        comment_texts = extract_comment_texts(comments, max_comments=MAX_COMMENTS_PER_POST)
        avg_sentiment, pos_ratio, neg_ratio, analyzed_count = analyze_comment_sentiment(
            comment_texts,
            self._sentiment,
        )
        emotion_labels = self._detect_emotions(comment_texts)
        emotion_distribution = build_emotion_distribution(emotion_labels)
        unique_commenters = count_unique_commenters(comments)

        (
            author_username,
            author_link_karma,
            author_comment_karma,
            author_account_age_days,
        ) = self._author_metrics(submission)
        num_awards, num_crossposts, post_flair, is_stickied, is_original_content = (
            self._post_extra_metrics(submission)
        )

        return PostSnapshot(
            subreddit=subreddit_name,
            post_id=post_id,
            title=title,
            score=score,
            num_comments=num_comments,
            created_utc=created_utc,
            avg_comment_sentiment=avg_sentiment,
            pos_comment_ratio=pos_ratio,
            neg_comment_ratio=neg_ratio,
            analyzed_comment_count=analyzed_count,
            post_url=post_url,
            author_username=author_username,
            author_link_karma=author_link_karma,
            author_comment_karma=author_comment_karma,
            author_account_age_days=author_account_age_days,
            num_awards=num_awards,
            num_crossposts=num_crossposts,
            post_flair=post_flair,
            is_stickied=is_stickied,
            is_original_content=is_original_content,
            unique_commenters=unique_commenters,
            emotion_distribution=emotion_distribution,
        )

    def search_posts_for_game(
        self,
        game_name: str,
        posts_per_subreddit: int = POSTS_PER_SUBREDDIT,
    ) -> list[PostSnapshot]:
        """Search configured subreddits and return normalized post snapshots."""
        snapshots: list[PostSnapshot] = []
        backoff = INITIAL_BACKOFF_SECONDS

        for subreddit_name in SUBREDDITS:
            try:
                self._limiter.wait_if_needed()
                subreddit = self._reddit.subreddit(subreddit_name)
                self._limiter.record_request()

                logger.info(
                    "Searching r/%s for game=%r (limit=%d).",
                    subreddit_name,
                    game_name,
                    posts_per_subreddit,
                )

                for submission in subreddit.search(
                    game_name,
                    sort="relevance",
                    limit=posts_per_subreddit,
                ):
                    try:
                        parsed = self.parse_submission(submission, subreddit_name)
                        if parsed is not None:
                            snapshots.append(parsed)
                        self._limiter.wait_if_needed()
                        self._limiter.record_request()
                        backoff = INITIAL_BACKOFF_SECONDS
                    except TooManyRequests as exc:
                        RedditRateLimiter.sleep_for_backoff(backoff)
                        logger.warning(
                            "Rate limit parsing post in r/%s: %s",
                            subreddit_name,
                            exc,
                        )
                        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    except (PrawcoreException, PRAWException) as exc:
                        logger.error(
                            "PRAW error processing post in r/%s for %r: %s",
                            subreddit_name,
                            game_name,
                            exc,
                            exc_info=True,
                        )

            except TooManyRequests as exc:
                RedditRateLimiter.sleep_for_backoff(backoff)
                logger.warning(
                    "Rate limit accessing r/%s for %r: %s",
                    subreddit_name,
                    game_name,
                    exc,
                )
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            except (PrawcoreException, PRAWException) as exc:
                logger.error(
                    "PRAW error accessing r/%s for %r: %s",
                    subreddit_name,
                    game_name,
                    exc,
                    exc_info=True,
                )

        return snapshots


def ensure_schema() -> None:
    """Create ORM tables when they do not yet exist."""
    Base.metadata.create_all(bind=engine)


def get_games_pending_for_batch(
    session: Session,
    batch_day: dt.date,
) -> list[Game]:
    """
    Return games with no ``RedditMetric`` rows mined on ``batch_day`` (UTC).

    This replaces CSV/checkpoint-file deduplication.
    """
    day_start, day_end = utc_day_bounds(batch_day)
    mined_today = (
        select(RedditMetric.game_id)
        .where(
            RedditMetric.mined_at >= day_start,
            RedditMetric.mined_at < day_end,
        )
        .distinct()
    )
    stmt = select(Game).where(Game.id.not_in(mined_today)).order_by(Game.id)
    return list(session.scalars(stmt).all())


def get_game_by_steam_name(session: Session, steam_name: str) -> Optional[Game]:
    """Look up a game by exact ``steam_name``."""
    stmt = select(Game).where(Game.steam_name == steam_name)
    return session.scalars(stmt).first()


def persist_post_metrics(
    session: Session,
    game: Game,
    snapshots: list[PostSnapshot],
) -> int:
    """
    Insert one ``RedditMetric`` per snapshot.

    Each row receives a unique ``run_id`` and ``mined_at`` timestamp.
    Returns the number of rows inserted. Skips posts whose ``post_id`` exists.
    """
    inserted = 0
    for snapshot in snapshots:
        existing = session.scalars(
            select(RedditMetric).where(RedditMetric.post_id == snapshot.post_id)
        ).first()
        if existing is not None:
            logger.warning(
                "RedditMetric already exists for post_id=%s; skipping insert.",
                snapshot.post_id,
            )
            continue

        metric = RedditMetric(
            game_id=game.id,
            run_id=str(uuid.uuid4()),
            mined_at=utc_now(),
            subreddit=snapshot.subreddit,
            post_id=snapshot.post_id,
            title=snapshot.title or None,
            score=snapshot.score,
            num_comments=snapshot.num_comments,
            created_utc=snapshot.created_utc,
            avg_comment_sentiment=snapshot.avg_comment_sentiment,
            pos_comment_ratio=snapshot.pos_comment_ratio,
            neg_comment_ratio=snapshot.neg_comment_ratio,
            analyzed_comment_count=snapshot.analyzed_comment_count,
            post_url=snapshot.post_url or None,
            author_username=snapshot.author_username,
            author_link_karma=snapshot.author_link_karma,
            author_comment_karma=snapshot.author_comment_karma,
            author_account_age_days=snapshot.author_account_age_days,
            num_awards=snapshot.num_awards,
            num_crossposts=snapshot.num_crossposts,
            post_flair=snapshot.post_flair,
            is_stickied=snapshot.is_stickied,
            is_original_content=snapshot.is_original_content,
            unique_commenters=snapshot.unique_commenters,
            emotion_distribution=snapshot.emotion_distribution or None,
        )
        session.add(metric)
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Mining orchestration
# ---------------------------------------------------------------------------


def load_reddit_credentials() -> tuple[str, str, str, str, str]:
    """Load Reddit API credentials from the environment."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    user_agent = os.getenv("REDDIT_USER_AGENT", "GameMiningScript/0.1")
    missing = [
        name
        for name, value in (
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USERNAME", username),
            ("REDDIT_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing Reddit credentials in environment: {', '.join(missing)}"
        )
    assert client_id is not None
    assert client_secret is not None
    assert username is not None
    assert password is not None
    return client_id, client_secret, username, password, user_agent


def mine_game(
    session: Session,
    client: RedditAPIClient,
    game: Game,
) -> int:
    """
    Search Reddit and persist ``RedditMetric`` rows for one game.

    Returns the number of rows written.
    """
    snapshots = client.search_posts_for_game(game.steam_name)
    if not snapshots:
        logger.info(
            "No Reddit posts found for game id=%s steam_name=%r.",
            game.id,
            game.steam_name,
        )
        return 0

    count = persist_post_metrics(session, game, snapshots)
    logger.info(
        "Persisted %d RedditMetric row(s) for game id=%s steam_name=%r.",
        count,
        game.id,
        game.steam_name,
    )
    return count


def run_batch(client: RedditAPIClient, batch_day: dt.date) -> None:
    """Mine all games that lack Reddit metrics for ``batch_day``."""
    with db_session() as session:
        pending = get_games_pending_for_batch(session, batch_day)
        pending_ids = [game.id for game in pending]

    total = len(pending_ids)
    if total == 0:
        logger.info("No games pending for Reddit batch on %s.", batch_day.isoformat())
        return

    logger.info(
        "Starting Reddit batch for %s: %d game(s) pending.",
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
                    "[%d/%d] (%.1f%%) Mining Reddit for steam_name=%r",
                    index,
                    total,
                    progress,
                    game.steam_name,
                )
                mine_game(session, client, game)
        except (RuntimeError, SQLAlchemyError, PrawcoreException, PRAWException) as exc:
            logger.error(
                "Failed mining game id=%s: %s",
                game_id,
                exc,
                exc_info=True,
            )

    logger.info("Reddit batch for %s complete.", batch_day.isoformat())


def run_single_game(client: RedditAPIClient, steam_name: str) -> None:
    """Mine a single game by ``steam_name`` (ignores batch-day checkpoint)."""
    with db_session() as session:
        game = get_game_by_steam_name(session, steam_name)
        if game is None:
            raise ValueError(
                f"No Game record found with steam_name={steam_name!r}. "
                "Seed the games table before mining."
            )
        mine_game(session, client, game)
    logger.info("Single-game Reddit mining finished for %r.", steam_name)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Mine Reddit post metrics into the SQLite warehouse.",
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
    client_id, client_secret, username, password, user_agent = load_reddit_credentials()
    client = RedditAPIClient(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password,
    )

    if args.game:
        run_single_game(client, args.game.strip())
        return

    batch_day = parse_batch_date(args.batch_date)
    run_batch(client, batch_day)


if __name__ == "__main__":
    main()
