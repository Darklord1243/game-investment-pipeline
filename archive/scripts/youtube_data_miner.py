"""
YouTube Game Video Miner (API-Dominant)

This script collects YouTube video, channel, and comment data for a list of games using the YouTube Data API v3.
- All data collection is performed via the official API (no scraping).
- Dislike counts are retrieved via the Return YouTube Dislike API (RYD).
- Sentiment analysis is performed on comments using VADER.

See API_USAGE.md for full API usage documentation.
"""
import os
import csv
import time
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import argparse
from tqdm import tqdm
import re
from collections import defaultdict

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('miner.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_CSV = 'youtube_game_videos.csv'
STEAM_CSV = 'steam_significant_games.csv'
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
YOUTUBE_API_URL = 'https://www.googleapis.com/youtube/v3'

if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY environment variable not set.")

# --- Helper: Checkpointing ---
def get_already_mined_games(output_csv: str) -> set[str]:
    if not os.path.exists(output_csv):
        return set()
    df = pd.read_csv(output_csv)
    return set(df['game_title'].astype(str))

def read_game_names_from_steam_csv(steam_csv: str) -> List[str]:
    df = pd.read_csv(steam_csv)
    return df['name'].dropna().astype(str).tolist()

# --- Helper: Dislike count via RYD API ---
def extract_dislike_count(video_id: str) -> Optional[int]:
    try:
        api_url = f"https://returnyoutubedislikeapi.com/votes?videoId={video_id}"
        resp = requests.get(api_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('dislikes')
    except Exception as e:
        logger.warning(f"Failed to get dislike count for {video_id}: {e}")
    return None

# --- Helper: Sentiment analysis ---
def analyze_comments_sentiment(comments: List[str]) -> tuple[float, float, float]:
    analyzer = SentimentIntensityAnalyzer()
    if not comments:
        return 0.0, 0.0, 0.0
    scores = [analyzer.polarity_scores(c)['compound'] for c in comments]
    avg = sum(scores) / len(scores)
    pos = sum(1 for s in scores if s > 0.05) / len(scores)
    neg = sum(1 for s in scores if s < -0.05) / len(scores)
    return avg, pos, neg

# --- API Usage Stats ---
api_usage = defaultdict(int)

def count_api_call(endpoint: str):
    api_usage[endpoint] += 1

# --- Quota Error Helper ---
def is_quota_error(resp: requests.Response) -> bool:
    if resp.status_code == 403:
        try:
            err = resp.json().get('error', {})
            for e in err.get('errors', []):
                reason = e.get('reason', '')
                if reason in ('quotaExceeded', 'userRateLimitExceeded'):
                    return True
        except Exception:
            pass
    return False

def handle_quota_error(resp: requests.Response):
    logger.error("YouTube API quota or rate limit exceeded. Exiting miner. You can resume later.")
    logger.error(f"Response: {resp.text}")
    print_api_usage_stats()
    sys.exit(1)

# --- YouTube Data API Functions ---
def search_videos(game_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
    count_api_call('search')
    params = {
        'part': 'snippet',
        'q': game_name,
        'type': 'video',
        'maxResults': max_results,
        'key': YOUTUBE_API_KEY
    }
    resp = requests.get(f"{YOUTUBE_API_URL}/search", params=params)
    if is_quota_error(resp):
        handle_quota_error(resp)
    resp.raise_for_status()
    items = resp.json().get('items', [])
    return items

def get_video_details(video_ids: List[str]) -> List[Dict[str, Any]]:
    count_api_call('videos')
    params = {
        'part': 'snippet,contentDetails,statistics',
        'id': ','.join(video_ids),
        'key': YOUTUBE_API_KEY
    }
    resp = requests.get(f"{YOUTUBE_API_URL}/videos", params=params)
    if is_quota_error(resp):
        handle_quota_error(resp)
    resp.raise_for_status()
    return resp.json().get('items', [])

def get_channel_details(channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    count_api_call('channels')
    params = {
        'part': 'snippet,statistics',
        'id': ','.join(channel_ids),
        'key': YOUTUBE_API_KEY
    }
    resp = requests.get(f"{YOUTUBE_API_URL}/channels", params=params)
    if is_quota_error(resp):
        handle_quota_error(resp)
    resp.raise_for_status()
    items = resp.json().get('items', [])
    return {item['id']: item for item in items}

def get_video_comments(video_id: str, max_comments: int = 20) -> List[str]:
    count_api_call('commentThreads')
    comments = []
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'maxResults': 100,
        'textFormat': 'plainText',
        'key': YOUTUBE_API_KEY
    }
    next_page_token = None
    while len(comments) < max_comments:
        if next_page_token:
            params['pageToken'] = next_page_token
        resp = requests.get(f"{YOUTUBE_API_URL}/commentThreads", params=params)
        if is_quota_error(resp):
            handle_quota_error(resp)
        if resp.status_code != 200:
            break
        data = resp.json()
        for item in data.get('items', []):
            top_comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(top_comment)
            if len(comments) >= max_comments:
                break
        next_page_token = data.get('nextPageToken')
        if not next_page_token:
            break
    return comments

def clean_text(val):
    if not isinstance(val, str):
        return '' if val is None else str(val)
    # Remove newlines, tabs, excessive whitespace
    return re.sub(r'[\n\r\t]+', ' ', val).strip()

def clean_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return ''

def clean_row(row: dict) -> dict:
    # Clean and validate all fields
    return {
        'game_title': clean_text(row.get('game_title', '')),
        'video_id': clean_text(row.get('video_id', '')),
        'title': clean_text(row.get('title', '')),
        'description': clean_text(row.get('description', '')),
        'published_at': clean_text(row.get('published_at', '')),
        'duration': clean_text(row.get('duration', '')),
        'tags': clean_text(row.get('tags', '')),
        'view_count': clean_int(row.get('view_count', '')),
        'like_count': clean_int(row.get('like_count', '')),
        'dislike_count': clean_int(row.get('dislike_count', '')),
        'comment_count': clean_int(row.get('comment_count', '')),
        'channel_title': clean_text(row.get('channel_title', '')),
        'channel_subscriber_count': clean_int(row.get('channel_subscriber_count', '')),
        'channel_video_count': clean_int(row.get('channel_video_count', '')),
        'channel_view_count': clean_int(row.get('channel_view_count', '')),
        'avg_comment_sentiment': float(row.get('avg_comment_sentiment', 0.0)) if row.get('avg_comment_sentiment') not in (None, '') else 0.0,
        'pos_comment_ratio': float(row.get('pos_comment_ratio', 0.0)) if row.get('pos_comment_ratio') not in (None, '') else 0.0,
        'neg_comment_ratio': float(row.get('neg_comment_ratio', 0.0)) if row.get('neg_comment_ratio') not in (None, '') else 0.0,
        'thumbnail_url': clean_text(row.get('thumbnail_url', '')),
    }

# --- CLI Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="YouTube Game Video Miner (API-Dominant)")
    parser.add_argument('--input', type=str, default='steam_significant_games.csv', help='Input CSV with game names')
    parser.add_argument('--output', type=str, default='youtube_game_videos.csv', help='Output CSV for mined data')
    parser.add_argument('--max-videos', type=int, default=1, help='Max videos per game (only one will be mined per game)')
    parser.add_argument('--dry-run', action='store_true', help='Process only 3 games for quick testing')
    parser.add_argument('--game', type=str, help='Game name to mine (single-game mode)')
    return parser.parse_args()

def select_most_relevant_video(game_name: str, search_results: List[dict]) -> Optional[dict]:
    """
    Select the most relevant video: strict title match (case-insensitive, partial allowed),
    highest view count among matches, fallback to first result if none match.
    """
    if not search_results:
        return None
    # Filter videos where game_name is in the title (case-insensitive)
    matches = []
    for item in search_results:
        title = item['snippet'].get('title', '')
        if game_name.lower() in title.lower():
            matches.append(item)
    if matches:
        # Fetch video details for all matches to get view counts
        video_ids = [item['id']['videoId'] for item in matches]
        details = get_video_details(video_ids)
        # Map videoId to viewCount
        id_to_views = {v['id']: int(v.get('statistics', {}).get('viewCount', 0)) for v in details}
        # Pick the match with the highest view count
        best_item = max(matches, key=lambda item: id_to_views.get(item['id']['videoId'], 0))
        return best_item
    # Fallback: return the first result
    return search_results[0]

# --- Main API Miner ---
def mine_youtube_metrics(game_name: str, max_videos: int = 5) -> List[Dict[str, Any]]:
    results = []
    try:
        search_results = search_videos(game_name, max_results=max_videos)
        # Select the most relevant video only
        best_item = select_most_relevant_video(game_name, search_results)
        if not best_item:
            return []
        video_id = best_item['id']['videoId']
        video_details = get_video_details([video_id])
        if not video_details:
            return []
        v = video_details[0]
        snippet = v['snippet']
        stats = v.get('statistics', {})
        content = v.get('contentDetails', {})
        channel_id = snippet['channelId']
        channel_details = get_channel_details([channel_id])
        channel_info = channel_details.get(channel_id, {})
        channel_snippet = channel_info.get('snippet', {})
        channel_stats = channel_info.get('statistics', {})
        comments = get_video_comments(video_id, max_comments=20)
        avg_sentiment, pos_ratio, neg_ratio = analyze_comments_sentiment(comments)
        results.append({
            'game_title': game_name,
            'video_id': video_id,
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'published_at': snippet.get('publishedAt', ''),
            'duration': content.get('duration', ''),
            'tags': ','.join(snippet.get('tags', [])) if 'tags' in snippet else '',
            'view_count': stats.get('viewCount', ''),
            'like_count': stats.get('likeCount', ''),
            'dislike_count': extract_dislike_count(video_id),
            'comment_count': stats.get('commentCount', ''),
            'channel_title': channel_snippet.get('title', ''),
            'channel_subscriber_count': channel_stats.get('subscriberCount', ''),
            'channel_video_count': channel_stats.get('videoCount', ''),
            'channel_view_count': channel_stats.get('viewCount', ''),
            'avg_comment_sentiment': avg_sentiment,
            'pos_comment_ratio': pos_ratio,
            'neg_comment_ratio': neg_ratio,
            'thumbnail_url': f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
        })
    except Exception as e:
        logger.error(f"Error mining YouTube data for {game_name}: {e}")
    return results

# --- API Usage Stats Reporting ---
def print_api_usage_stats():
    msg = "\nAPI Usage Stats:"
    for endpoint, count in api_usage.items():
        msg += f"\n  {endpoint}: {count} calls"
    print(msg)
    logger.info(msg)

# --- Main mining loop ---
def main():
    args = parse_args()
    global OUTPUT_CSV, STEAM_CSV
    OUTPUT_CSV = args.output
    STEAM_CSV = args.input
    max_videos = args.max_videos
    if args.game:
        game_names = [args.game]
    else:
        already_mined = get_already_mined_games(OUTPUT_CSV)
        game_names = read_game_names_from_steam_csv(STEAM_CSV)
        if args.dry_run:
            game_names = game_names[:3]
    try:
        with open(OUTPUT_CSV, mode="a", newline='', encoding="utf-8") as csvfile:
            writer = None
            for game_name in tqdm(game_names, desc="Mining games"):
                if game_name in already_mined:
                    logger.info(f"Skipping already-mined game: {game_name}")
                    continue
                logger.info(f"Mining YouTube for: {game_name}")
                video_metrics = mine_youtube_metrics(game_name, max_videos=max_videos)
                if not video_metrics:
                    logger.warning(f"No videos found for {game_name}")
                    continue
                for row in video_metrics:
                    clean = clean_row(row)
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=clean.keys())
                        if csvfile.tell() == 0:
                            writer.writeheader()
                    writer.writerow(clean)
                csvfile.flush()
                logger.info(f"Finished {game_name}")
                time.sleep(3)  # Polite delay
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}")
    finally:
        print_api_usage_stats()

if __name__ == "__main__":
    main() 