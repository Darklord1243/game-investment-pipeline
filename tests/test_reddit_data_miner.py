import pytest
import csv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from reddit_data_miner import read_game_names_from_csv, get_already_mined_games

def test_read_game_names_from_csv(tmp_path):
    """Test reading game names from a sample steam_all_games.csv file."""
    csv_path = tmp_path / "steam_all_games.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["appid", "name", "release_date"])
        writer.writerow(["1", "Game A", "2020-01-01"])
        writer.writerow(["2", "Game B", "2021-01-01"])
    games = read_game_names_from_csv(str(csv_path))
    assert games == ["Game A", "Game B"]

def test_get_already_mined_games(tmp_path):
    """Test checkpoint logic by checking already mined games from output CSV."""
    output_csv = tmp_path / "reddit_game_posts.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["game_title", "post_id"])
        writer.writerow(["Game A", "abc123"])
        writer.writerow(["Game B", "def456"])
    mined = get_already_mined_games(str(output_csv))
    assert set(mined) == {"Game A", "Game B"} 