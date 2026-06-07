import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from twitch_miner import get_live_streams_for_game, get_twitch_access_token

def test_get_live_streams_for_game():
    # Fortnite game_id is '33214' (as of July 2024)
    access_token = get_twitch_access_token()
    streams = get_live_streams_for_game('33214', access_token)
    assert isinstance(streams, list)
    assert len(streams) > 0
    for stream in streams:
        assert isinstance(stream, dict)
        assert 'user_name' in stream and 'viewer_count' in stream
        assert isinstance(stream['user_name'], str)
        assert isinstance(stream['viewer_count'], int)
        # New fields
        assert 'title' in stream
        assert 'language' in stream
        assert 'started_at' in stream
        assert 'game_id' in stream 