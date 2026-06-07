import pytest
import csv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from youtube_data_miner import read_game_names_from_steam_csv, get_already_mined_games
from unittest.mock import patch, MagicMock
# If these functions are not yet implemented, comment out the imports and their tests.
try:
    from youtube_data_miner import extract_pinned_comment, extract_dislike_count
except ImportError:
    extract_pinned_comment = None
    extract_dislike_count = None
    # TODO: Implement extract_pinned_comment and extract_dislike_count in youtube_data_miner.py

# Example YouTube video with a pinned comment (replace with a stable one if needed)
YOUTUBE_VIDEO_WITH_PINNED = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
EXPECTED_PINNED_AUTHOR = None  # Fill in with the actual author if known
EXPECTED_PINNED_TEXT = None    # Fill in with the actual text if known

YOUTUBE_VIDEO_FOR_DISLIKE = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example video

# Remove test_extract_pinned_comment and any references to pinned comments

import pytest

# Example YouTube video with visible dislike count (replace with a stable one if needed)
YOUTUBE_VIDEO_WITH_DISLIKES = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

def test_extract_pinned_comment():
    if extract_pinned_comment is None:
        pytest.skip("extract_pinned_comment not implemented")
    else:
        author, text = extract_pinned_comment(YOUTUBE_VIDEO_WITH_PINNED)
        # Optionally, update EXPECTED_PINNED_AUTHOR and EXPECTED_PINNED_TEXT if known
        assert author is not None and text is not None

def test_extract_dislike_count():
    if extract_dislike_count is None:
        pytest.skip("extract_dislike_count not implemented")
    else:
        count = extract_dislike_count(YOUTUBE_VIDEO_WITH_DISLIKES)
        assert count is None or isinstance(count, int)

def test_read_game_names_from_csv(tmp_path):
    """Test reading game names from a sample steam_all_games.csv file."""
    csv_path = tmp_path / "steam_all_games.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["appid", "name", "release_date"])
        writer.writerow(["1", "Game A", "2020-01-01"])
        writer.writerow(["2", "Game B", "2021-01-01"])
    games = read_game_names_from_steam_csv(str(csv_path))
    assert games == ["Game A", "Game B"]

def test_get_already_mined_games(tmp_path):
    """Test checkpoint logic by checking already mined games from output CSV."""
    output_csv = tmp_path / "youtube_game_videos.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["game_title", "video_id"])
        writer.writerow(["Game A", "abc123"])
        writer.writerow(["Game B", "def456"])
    mined = get_already_mined_games(str(output_csv))
    assert set(mined) == {"Game A", "Game B"}

# --- API Function Unit Tests ---

def mock_response(json_data, status=200):
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = json_data
    mock.text = str(json_data)
    return mock

@patch('youtube_data_miner.requests.get')
def test_search_videos_success(mock_get):
    mock_get.return_value = mock_response({'items': [{'id': {'videoId': 'abc123'}}]})
    results = search_videos('Test Game', max_results=1)
    assert isinstance(results, list)
    assert results[0]['id']['videoId'] == 'abc123'

@patch('youtube_data_miner.requests.get')
def test_search_videos_quota_error(mock_get):
    quota_json = {'error': {'errors': [{'reason': 'quotaExceeded'}]}}
    mock_get.return_value = mock_response(quota_json, status=403)
    with pytest.raises(SystemExit):
        search_videos('Test Game', max_results=1)

@patch('youtube_data_miner.requests.get')
def test_get_video_details_success(mock_get):
    mock_get.return_value = mock_response({'items': [{'id': 'abc123', 'snippet': {}, 'statistics': {}, 'contentDetails': {}}]})
    results = get_video_details(['abc123'])
    assert isinstance(results, list)
    assert results[0]['id'] == 'abc123'

@patch('youtube_data_miner.requests.get')
def test_get_channel_details_success(mock_get):
    mock_get.return_value = mock_response({'items': [{'id': 'chan1', 'snippet': {}, 'statistics': {}}]})
    results = get_channel_details(['chan1'])
    assert isinstance(results, dict)
    assert 'chan1' in results

@patch('youtube_data_miner.requests.get')
def test_get_video_comments_success(mock_get):
    mock_get.return_value = mock_response({'items': [{'snippet': {'topLevelComment': {'snippet': {'textDisplay': 'Nice!'}}}}]})
    comments = get_video_comments('abc123', max_comments=1)
    assert isinstance(comments, list)
    assert comments[0] == 'Nice!'

@patch('youtube_data_miner.requests.get')
def test_get_video_comments_quota_error(mock_get):
    quota_json = {'error': {'errors': [{'reason': 'quotaExceeded'}]}}
    mock_get.return_value = mock_response(quota_json, status=403)
    with pytest.raises(SystemExit):
        get_video_comments('abc123', max_comments=1) 