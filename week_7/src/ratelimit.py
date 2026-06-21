"""A small sliding-window rate limiter for the Gemini free tier.

The free tier caps requests-per-minute (RPM). This limiter records the
timestamps of recent calls and, before allowing a new one, blocks just long
enough that we never exceed ``rpm`` calls in any 60-second window. It is shared
by *all* Gemini calls (answer generation AND the LLM judge) so the combined
request rate stays within quota.
"""
from __future__ import annotations

import threading
import time
from collections import deque

_WINDOW = 60.0  # seconds


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until issuing one more request keeps us under the RPM cap."""
        with self._lock:
            now = time.monotonic()
            self._evict(now)
            if len(self._calls) >= self.rpm:
                sleep_for = _WINDOW - (now - self._calls[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self._evict(time.monotonic())
            self._calls.append(time.monotonic())

    def _evict(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= _WINDOW:
            self._calls.popleft()
