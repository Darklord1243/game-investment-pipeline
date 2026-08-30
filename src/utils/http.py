"""
Thread-safe sliding-window rate limiter shared by the platform miners.

Platform-specific miners can either instantiate ``BaseRateLimiter`` directly
with appropriate parameters or subclass it to add bespoke behaviour (e.g.
header-aware backoff for Twitch, fixed-delay pacing for Reddit).

Deliberately pacing-only. This class used to carry a ``handle_rate_limit``
exponential-backoff helper, together with its ``backoff_base`` / ``backoff_max``
parameters and a ``reset_backoff`` companion. Nothing ever called any of it. It
was removed rather than wired in: every miner already handles its own
rate-limit responses, and retrofitting a 60s-doubling schedule onto Steam,
Twitch and YouTube would have changed their retry behaviour with no test
covering the change. Reddit's exponential backoff is hand-rolled in
``reddit_data_miner`` and is unaffected.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class BaseRateLimiter:
    """Sliding-window request pacer.

    Parameters
    ----------
    max_requests_per_minute:
        Hard ceiling for the sliding window (client-side).
    buffer:
        Requests subtracted from the ceiling so the limiter fires *before*
        the server-side limit is reached.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        buffer: int = 5,
    ) -> None:
        self._max_requests_per_minute = max_requests_per_minute
        self._buffer = buffer
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
