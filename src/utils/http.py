"""
Thread-safe rate limiter with sliding-window pacing and exponential backoff.

Platform-specific miners can either instantiate ``BaseRateLimiter`` directly
with appropriate parameters or subclass it to add bespoke behaviour (e.g.
header-aware backoff for Twitch, fixed-delay pacing for Reddit).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class BaseRateLimiter:
    """Sliding-window request pacer with exponential backoff on 429 / rate-limit responses.

    Parameters
    ----------
    max_requests_per_minute:
        Hard ceiling for the sliding window (client-side).
    buffer:
        Requests subtracted from the ceiling so the limiter fires *before*
        the server-side limit is reached.
    backoff_base:
        Initial sleep duration (seconds) used by ``handle_rate_limit``.
    backoff_max:
        Ceiling for the exponential backoff sequence.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        buffer: int = 5,
        backoff_base: float = 60.0,
        backoff_max: float = 300.0,
    ) -> None:
        self._max_requests_per_minute = max_requests_per_minute
        self._buffer = buffer
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._current_backoff = backoff_base
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Sliding-window pacing (thread-safe)
    # ------------------------------------------------------------------

    def wait_if_needed(self) -> None:
        """Block until another request is allowed under the sliding window."""
        with self._lock:
            now = time.time()
            self._timestamps = [ts for ts in self._timestamps if now - ts < 60.0]
            threshold = self._max_requests_per_minute - self._buffer
            if len(self._timestamps) < threshold:
                return
            oldest = self._timestamps[0]
            sleep_for = 60.0 - (now - oldest)
        if sleep_for > 0:
            logger.info(
                "Client-side rate buffer reached; sleeping %.0f seconds.",
                sleep_for,
            )
            time.sleep(sleep_for)

    def record_request(self) -> None:
        """Record that an HTTP request was dispatched."""
        with self._lock:
            self._timestamps.append(time.time())

    # ------------------------------------------------------------------
    # Exponential backoff for server-side rate-limit responses
    # ------------------------------------------------------------------

    def handle_rate_limit(self) -> None:
        """Sleep with exponential backoff and double the backoff window.

        Call this on a 429 / 403-quota response so that subsequent calls
        wait progressively longer until the server recovers.
        """
        wait = self._current_backoff
        self._current_backoff = min(self._current_backoff * 2, self._backoff_max)
        logger.warning("Rate limit hit; backing off %.0f seconds.", wait)
        time.sleep(wait)

    def reset_backoff(self) -> None:
        """Reset the backoff interval after a successful request window."""
        self._current_backoff = self._backoff_base
