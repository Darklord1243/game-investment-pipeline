import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from twitch_miner import get_twitch_access_token
from dotenv import load_dotenv

load_dotenv()

def test_get_twitch_access_token():
    token = get_twitch_access_token()
    assert isinstance(token, str)
    assert len(token) > 0 