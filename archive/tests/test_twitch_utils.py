import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import csv
from twitch_miner import save_streams_to_csv

def test_save_streams_to_csv(tmp_path):
    streams = [
        {'user_name': 'streamer1', 'viewer_count': 100},
        {'user_name': 'streamer2', 'viewer_count': 200}
    ]
    fieldnames = ['user_name', 'viewer_count']
    csv_file = tmp_path / 'streams.csv'
    save_streams_to_csv(streams, str(csv_file), fieldnames)
    # Optionally, add assertions to check the file contents
    with open(csv_file, newline='', encoding='utf-8') as f:
        lines = f.readlines()
    assert lines[0].strip() == 'user_name,viewer_count'
    assert 'streamer1,100' in lines[1]
    assert 'streamer2,200' in lines[2] 