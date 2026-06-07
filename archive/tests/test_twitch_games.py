import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from twitch_miner import get_top_twitch_games

def test_get_top_twitch_games():
    games = get_top_twitch_games()
    assert isinstance(games, list)
    assert len(games) > 0
    for game in games:
        assert isinstance(game, dict)
        assert 'id' in game and 'name' in game
        assert isinstance(game['id'], str)
        assert isinstance(game['name'], str) 