"""
One-shot seeder: loads Twitch, YouTube, Reddit, and Steam CSVs into the
SQLite game_metrics.db so the model_trainer pipeline has data to work with.

Run once:  python src/data_collection/seed_database.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select

# Ensure the src package is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.database.models import (
    Base,
    Game,
    RedditMetric,
    SessionLocal,
    TwitchMetric,
    YouTubeMetric,
    engine,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
CSV_PATHS: dict[str, tuple[str, str]] = {
    "twitch": (str(DATA_DIR / "twitch_game_streams.csv"), "utf-8"),
    "youtube": (str(DATA_DIR / "youtube_game_videos.csv"), "latin1"),
    "reddit": (str(DATA_DIR / "reddit_game_posts.csv"), "utf-8"),
    "steam": (str(DATA_DIR / "steam_significant_games.csv"), "latin1"),
}

RUN_ID = "csv-seed-01"
BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _neg_id(title: str) -> int:
    """Deterministic negative appid for games without a Steam entry."""
    return -abs(int(hashlib.md5(title.encode()).hexdigest()[:12], 16)) % (10**8)


def _parse_date(val: Any) -> dt.date | None:
    """Best-effort date parsing."""
    if pd.isna(val) or val in ("", "NaT", "nan"):
        return None
    try:
        ts = pd.Timestamp(val)
        if ts is pd.NaT:
            return None
        d = ts.date()
        # Reject clearly bogus dates
        if d.year < 1990 or d.year > 2030:
            return None
        return d
    except Exception:
        return None


def _earliest(*dates: dt.date | None) -> dt.date | None:
    valid = [d for d in dates if d is not None]
    return min(valid) if valid else None


# ---------------------------------------------------------------------------
# Main seeder
# ---------------------------------------------------------------------------


def seed() -> None:
    logger.info("Creating tables if needed")
    Base.metadata.create_all(bind=engine)

    # 1. Load all CSVs -------------------------------------------------------
    logger.info("Loading CSVs")
    twitch_raw = pd.read_csv(CSV_PATHS["twitch"][0], encoding=CSV_PATHS["twitch"][1])
    youtube_raw = pd.read_csv(CSV_PATHS["youtube"][0], encoding=CSV_PATHS["youtube"][1])
    reddit_raw = pd.read_csv(CSV_PATHS["reddit"][0], encoding=CSV_PATHS["reddit"][1])
    steam_raw = pd.read_csv(CSV_PATHS["steam"][0], encoding=CSV_PATHS["steam"][1])

    logger.info(
        "Raw counts — Twitch:%d YouTube:%d Reddit:%d Steam:%d",
        len(twitch_raw),
        len(youtube_raw),
        len(reddit_raw),
        len(steam_raw),
    )

    # 2. Normalise game-title columns ----------------------------------------
    twitch_raw["game_title"] = twitch_raw["game_title"].astype(str).str.strip()
    youtube_raw["game_title"] = youtube_raw["game_title"].astype(str).str.strip()
    youtube_raw.drop_duplicates("video_id", inplace=True)
    reddit_raw["game_title"] = reddit_raw["game_title"].astype(str).str.strip()
    reddit_raw.drop_duplicates("post_id", inplace=True)
    steam_raw["name"] = steam_raw["name"].astype(str).str.strip()

    # 3. Build game registry -------------------------------------------------
    # Steam games get their real appid; non-Steam titles get a deterministic
    # negative appid so the UNIQUE constraint on Game.appid is satisfied.
    steam_games = steam_raw[["appid", "name"]].drop_duplicates("appid").copy()
    steam_games = steam_games.rename(columns={"name": "steam_name"})

    all_titles: set[str] = set()
    for raw, col in [
        (twitch_raw, "game_title"),
        (youtube_raw, "game_title"),
        (reddit_raw, "game_title"),
    ]:
        all_titles.update(raw[col].dropna().unique())

    # Map title -> appid (real from Steam, or synthetic)
    title_to_appid: dict[str, int] = {}
    steam_title_to_appid = dict(zip(steam_games["steam_name"], steam_games["appid"]))

    for title in all_titles:
        if title in steam_title_to_appid:
            title_to_appid[title] = int(steam_title_to_appid[title])
        else:
            title_to_appid[title] = _neg_id(title)

    # 4. Compute proxy release_date per game ---------------------------------
    # Use the earliest YouTube published_at or Reddit created_utc.
    youtube_raw["_parsed_date"] = pd.to_datetime(
        youtube_raw["published_at"], errors="coerce"
    )
    reddit_raw["_parsed_date"] = pd.to_datetime(
        reddit_raw["created_utc"], unit="s", errors="coerce"
    )

    yt_dates = youtube_raw.groupby("game_title")["_parsed_date"].min()
    rd_dates = reddit_raw.groupby("game_title")["_parsed_date"].min()

    title_to_release: dict[str, dt.date | None] = {}
    for title in all_titles:
        yt_ts = yt_dates.get(title)
        rd_ts = rd_dates.get(title)
        yt_d = yt_ts.date() if pd.notna(yt_ts) else None
        rd_d = rd_ts.date() if pd.notna(rd_ts) else None
        title_to_release[title] = _earliest(yt_d, rd_d)

    # For games with no observed date use a spread across 2019-2023 based on
    # hash (deterministic, keeps the temporal split meaningful).
    for title in all_titles:
        if title_to_release[title] is None:
            h = abs(hash(title)) % (365 * 5)  # 5-year spread
            title_to_release[title] = dt.date(2019, 1, 1) + dt.timedelta(days=h)

    # 5. Insert Game records -------------------------------------------------
    game_rows: list[dict[str, Any]] = []
    seen_appids: set[int] = set()
    for title in sorted(all_titles):
        appid = title_to_appid[title]
        if appid in seen_appids:
            continue
        seen_appids.add(appid)
        game_rows.append(
            {
                "appid": appid,
                "steam_name": title,
                "release_date": title_to_release.get(title),
            }
        )

    with SessionLocal() as session:
        existing = session.execute(select(Game.appid)).scalars().all()
        existing_set = set(existing)
        new_games = [g for g in game_rows if g["appid"] not in existing_set]
        logger.info(
            "Games: %d total, %d new, %d already in DB",
            len(game_rows),
            len(new_games),
            len(game_rows) - len(new_games),
        )
        for i in range(0, len(new_games), BATCH_SIZE):
            batch = new_games[i : i + BATCH_SIZE]
            session.execute(Game.__table__.insert(), batch)
        session.commit()

    # Reload ID mapping after insert
    with SessionLocal() as session:
        rows = session.execute(select(Game.id, Game.appid, Game.steam_name)).all()
    appid_to_game_id: dict[int, int] = {}
    title_to_game_id: dict[str, int] = {}
    for gid, appid, sname in rows:
        appid_to_game_id[appid] = gid
        title_to_game_id[sname] = gid

    def _resolve_game_id(title: str) -> int:
        return title_to_game_id.get(title, -1)

    # 6. Insert metric rows --------------------------------------------------
    with SessionLocal() as session:
        # -- Twitch --
        existing_count = session.execute(
            select(func.count(TwitchMetric.id))
        ).scalar_one()
        if existing_count == 0:
            twitch_rows: list[dict[str, Any]] = []
            for _, r in twitch_raw.iterrows():
                gid = _resolve_game_id(r["game_title"])
                if gid == -1:
                    continue
                ts = pd.Timestamp(r["timestamp"])
                twitch_rows.append(
                    {
                        "game_id": gid,
                        "run_id": str(r.get("run_id", RUN_ID)),
                        "mined_at": ts.to_pydatetime() if ts is not pd.NaT else dt.datetime.now(),
                        "twitch_game_id": str(r.get("game_id", "")),
                        "streamer_name": str(r.get("user_name", "")),
                        "stream_started_at": (
                            pd.Timestamp(r["started_at"]).to_pydatetime()
                            if pd.notna(r.get("started_at"))
                            else None
                        ),
                        "stream_title": str(r.get("title", "")),
                        "language": str(r.get("language", "")),
                        "viewer_count": int(r["viewer_count"]) if pd.notna(r["viewer_count"]) else 0,
                    }
                )
            for i in range(0, len(twitch_rows), BATCH_SIZE):
                session.execute(TwitchMetric.__table__.insert(), twitch_rows[i : i + BATCH_SIZE])
            logger.info("Inserted %d Twitch metrics", len(twitch_rows))
        else:
            logger.info("Twitch metrics already present (%d rows), skipping", existing_count)

        # -- YouTube --
        existing_count = session.execute(
            select(func.count(YouTubeMetric.id))
        ).scalar_one()
        if existing_count == 0:
            youtube_rows: list[dict[str, Any]] = []
            for _, r in youtube_raw.iterrows():
                gid = _resolve_game_id(r["game_title"])
                if gid == -1:
                    continue
                youtube_rows.append(
                    {
                        "game_id": gid,
                        "video_id": str(r["video_id"]),
                        "mined_at": dt.datetime.now(),
                        "title": str(r.get("title", "")),
                        "description": str(r.get("description", "")),
                        "published_at": (
                            pd.Timestamp(r["published_at"]).to_pydatetime()
                            if pd.notna(r.get("published_at"))
                            else None
                        ),
                        "duration_iso8601": str(r.get("duration", "")),
                        "tags": str(r.get("tags", "")),
                        "view_count": int(r["view_count"]) if pd.notna(r["view_count"]) else 0,
                        "like_count": int(r["like_count"]) if pd.notna(r["like_count"]) else 0,
                        "dislike_count": int(r["dislike_count"]) if pd.notna(r["dislike_count"]) else 0,
                        "comment_count": int(r["comment_count"]) if pd.notna(r["comment_count"]) else 0,
                        "channel_title": str(r.get("channel_title", "")),
                        "channel_subscriber_count": (
                            int(r["channel_subscriber_count"])
                            if pd.notna(r.get("channel_subscriber_count"))
                            else 0
                        ),
                        "channel_video_count": (
                            int(r["channel_video_count"])
                            if pd.notna(r.get("channel_video_count"))
                            else 0
                        ),
                        "channel_view_count": (
                            int(r["channel_view_count"])
                            if pd.notna(r.get("channel_view_count"))
                            else 0
                        ),
                        "avg_comment_sentiment": (
                            float(r["avg_comment_sentiment"])
                            if pd.notna(r.get("avg_comment_sentiment"))
                            else 0.0
                        ),
                        "pos_comment_ratio": (
                            float(r["pos_comment_ratio"])
                            if pd.notna(r.get("pos_comment_ratio"))
                            else 0.0
                        ),
                        "neg_comment_ratio": (
                            float(r["neg_comment_ratio"])
                            if pd.notna(r.get("neg_comment_ratio"))
                            else 0.0
                        ),
                        "thumbnail_url": str(r.get("thumbnail_url", "")),
                    }
                )
            for i in range(0, len(youtube_rows), BATCH_SIZE):
                session.execute(
                    YouTubeMetric.__table__.insert(), youtube_rows[i : i + BATCH_SIZE]
                )
            logger.info("Inserted %d YouTube metrics", len(youtube_rows))
        else:
            logger.info("YouTube metrics already present (%d rows), skipping", existing_count)

        # -- Reddit --
        existing_count = session.execute(
            select(func.count(RedditMetric.id))
        ).scalar_one()
        if existing_count == 0:
            reddit_rows: list[dict[str, Any]] = []
            for _, r in reddit_raw.iterrows():
                gid = _resolve_game_id(r["game_title"])
                if gid == -1:
                    continue
                ts = pd.Timestamp(r["timestamp"])
                reddit_rows.append(
                    {
                        "game_id": gid,
                        "run_id": str(r.get("run_id", RUN_ID)),
                        "mined_at": ts.to_pydatetime() if ts is not pd.NaT else dt.datetime.now(),
                        "subreddit": str(r.get("subreddit", "")),
                        "post_id": str(r["post_id"]),
                        "title": str(r.get("title", "")),
                        "score": int(r["score"]) if pd.notna(r["score"]) else 0,
                        "num_comments": int(r["num_comments"]) if pd.notna(r["num_comments"]) else 0,
                        "created_utc": float(r["created_utc"]) if pd.notna(r["created_utc"]) else 0.0,
                        "avg_comment_sentiment": (
                            float(r["avg_comment_sentiment"])
                            if pd.notna(r.get("avg_comment_sentiment"))
                            else 0.0
                        ),
                        "pos_comment_ratio": (
                            float(r["pos_comment_ratio"])
                            if pd.notna(r.get("pos_comment_ratio"))
                            else 0.0
                        ),
                        "neg_comment_ratio": (
                            float(r["neg_comment_ratio"])
                            if pd.notna(r.get("neg_comment_ratio"))
                            else 0.0
                        ),
                        "analyzed_comment_count": int(r["comment_count"]) if pd.notna(r["comment_count"]) else 0,
                        "post_url": str(r.get("post_url", "")),
                        "author_username": str(r.get("author_username", "")),
                        "author_link_karma": int(r["author_link_karma"]) if pd.notna(r["author_link_karma"]) else 0,
                        "author_comment_karma": int(r["author_comment_karma"]) if pd.notna(r["author_comment_karma"]) else 0,
                        "author_account_age_days": int(r["author_account_age_days"]) if pd.notna(r["author_account_age_days"]) else 0,
                        "num_awards": int(r["num_awards"]) if pd.notna(r["num_awards"]) else 0,
                        "num_crossposts": int(r["num_crossposts"]) if pd.notna(r["num_crossposts"]) else 0,
                        "post_flair": (
                            str(r["post_flair"]) if pd.notna(r.get("post_flair")) else None
                        ),
                        "is_stickied": bool(r["is_stickied"]) if pd.notna(r.get("is_stickied")) else False,
                        "is_original_content": bool(r["is_original_content"]) if pd.notna(r.get("is_original_content")) else False,
                        "unique_commenters": int(r["unique_commenters"]) if pd.notna(r["unique_commenters"]) else 0,
                        "emotion_distribution": None,
                    }
                )
            for i in range(0, len(reddit_rows), BATCH_SIZE):
                session.execute(
                    RedditMetric.__table__.insert(), reddit_rows[i : i + BATCH_SIZE]
                )
            logger.info("Inserted %d Reddit metrics", len(reddit_rows))
        else:
            logger.info("Reddit metrics already present (%d rows), skipping", existing_count)

        session.commit()

    logger.info("Seeding complete.")


if __name__ == "__main__":
    seed()
