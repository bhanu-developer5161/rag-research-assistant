"""A daily cap on questions, shared by every visitor, so a public demo cannot drain the free API quota."""
import threading
from datetime import datetime, timezone


class DailyLimit:
    """Thread-safe counter that resets at midnight UTC."""

    def __init__(self, max_per_day: int, clock=None):
        self.max = max_per_day
        self._clock = clock or (lambda: datetime.now(timezone.utc).date())
        self._day = self._clock()
        self.count = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Use up one question for today. False when today's allowance is gone."""
        with self._lock:
            today = self._clock()
            if today != self._day:
                self._day, self.count = today, 0
            if self.count >= self.max:
                return False
            self.count += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            if self._clock() != self._day:
                return self.max
            return max(self.max - self.count, 0)
