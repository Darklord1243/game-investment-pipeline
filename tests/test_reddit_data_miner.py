"""Tests for Reddit data miner — mocked PRAW only."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from praw.models import Comment
from praw.models.reddit.more import MoreComments
from prawcore.exceptions import PrawcoreException, TooManyRequests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_collection.reddit_data_miner import (  # noqa: E402
    RedditAPIClient,
    RedditRateLimiter,
    SUBREDDITS,
)


def _too_many_requests() -> TooManyRequests:
    """Build a TooManyRequests with the response shape prawcore expects."""
    response = MagicMock()
    response.headers = {"retry-after": "1"}
    response.text = "rate limited"
    response.status_code = 429
    return TooManyRequests(response)


@pytest.fixture()
def client() -> RedditAPIClient:
    """RedditAPIClient whose PRAW Reddit instance is a MagicMock."""
    with patch("src.data_collection.reddit_data_miner.praw.Reddit"):
        return RedditAPIClient(
            client_id="id",
            client_secret="secret",
            user_agent="test-agent",
            username="user",
            password="pass",
            limiter=RedditRateLimiter(delay_seconds=0.0),
        )


def test_replace_more_and_list_returns_comments_only(client: RedditAPIClient) -> None:
    """Flatten the forest and drop MoreComments placeholders."""
    comment = MagicMock(spec=Comment)
    more = MagicMock(spec=MoreComments)
    forest = MagicMock()
    forest.list.return_value = [comment, more]
    submission = MagicMock()
    submission.comments = forest

    result = client._replace_more_and_list(submission)

    forest.replace_more.assert_called_once_with(limit=0)
    forest.list.assert_called_once_with()
    assert result == [comment]


@patch.object(RedditRateLimiter, "sleep_for_backoff")
def test_search_posts_too_many_requests_backs_off(
    mock_sleep: MagicMock,
    client: RedditAPIClient,
) -> None:
    """Rate limits trigger backoff and must not raise AttributeError."""
    subreddit = MagicMock()
    subreddit.search.side_effect = _too_many_requests()
    client._reddit.subreddit.return_value = subreddit

    snapshots = client.search_posts_for_game("Hades")

    assert snapshots == []
    assert mock_sleep.call_count == len(SUBREDDITS)
    assert subreddit.search.call_count == len(SUBREDDITS)


def test_search_posts_prawcore_error_continues_other_subreddits(
    client: RedditAPIClient,
) -> None:
    """A PrawcoreException on one subreddit does not abort the remaining searches."""

    def make_subreddit(name: str) -> MagicMock:
        sub = MagicMock()
        if name == SUBREDDITS[0]:
            sub.search.side_effect = PrawcoreException("forbidden")
        else:
            sub.search.return_value = iter([])
        return sub

    client._reddit.subreddit.side_effect = make_subreddit

    snapshots = client.search_posts_for_game("Hades")

    assert snapshots == []
    assert client._reddit.subreddit.call_count == len(SUBREDDITS)
