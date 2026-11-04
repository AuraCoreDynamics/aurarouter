"""Circuit breaker for provider fault tolerance.

Per-provider circuit breaker with closed → open → half_open states.

State machine:
- closed (normal): requests pass through. After failure_threshold
  consecutive failures → open.
- open (tripped): requests short-circuit immediately. After
  reset_timeout seconds → half_open.
- half_open (probing): one request allowed through.
  Success → closed. Failure → back to open.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from aurarouter._logging import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    def __init__(self, provider_name: str, failure_threshold: int = 5,
                 reset_timeout: float = 60.0):
        self._provider_name = provider_name
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._lock = threading.Lock()
        self._state = "closed"  # closed, open, half_open
        self._consecutive_failures = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._last_failure_utc: datetime | None = None
        self._last_success_utc: datetime | None = None
        self._half_open_granted = False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._last_success_time = time.monotonic()
            self._last_success_utc = datetime.now(timezone.utc)
            if self._state in ("half_open", "open"):
                logger.info("Circuit breaker %s: %s → closed", self._provider_name, self._state)
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()
            self._last_failure_utc = datetime.now(timezone.utc)
            if self._state == "half_open":
                self._state = "open"
                self._half_open_granted = False
                logger.warning("Circuit breaker %s: half_open → open (probe failed)", self._provider_name)
            elif self._consecutive_failures >= self._failure_threshold:
                if self._state != "open":
                    logger.warning("Circuit breaker %s: closed → open (%d consecutive failures)",
                                   self._provider_name, self._consecutive_failures)
                self._state = "open"

    def peek_state(self) -> tuple[str, int]:
        """Lock-free, non-mutating inspection of (effective_state, consecutive_failures).

        Safe for routing-stack ranking and shadow evaluation without consuming
        the single ``half_open`` probe token.
        """
        state = self._state
        failures = self._consecutive_failures
        last_fail = self._last_failure_time
        if state == "open":
            if last_fail is not None and (time.monotonic() - last_fail) >= self._reset_timeout:
                return "half_open", failures
            return "open", failures
        if state == "half_open" and self._half_open_granted:
            return "open", failures
        return state, failures

    def is_available(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self._reset_timeout:
                        self._state = "half_open"
                        self._half_open_granted = False
                        logger.info("Circuit breaker %s: open → half_open (%.1fs elapsed)",
                                    self._provider_name, elapsed)
                    else:
                        return False
                else:
                    return False
            # half_open: allow one probe
            if self._state == "half_open" and not self._half_open_granted:
                self._half_open_granted = True
                return True
            return self._state == "half_open" and not self._half_open_granted

    def seconds_since_last_failure(self) -> float | None:
        """Return seconds elapsed since the last recorded failure, or None if never failed.
        Uses monotonic clock — safe for duration comparisons only, not wall time."""
        with self._lock:
            if self._last_failure_time is None:
                return None
            return time.monotonic() - self._last_failure_time

    def get_health_state(self):
        """Return ProviderHealthState DTO. Lazy import for standalone mode."""
        try:
            from aurarouter.auragrid.contracts import ProviderHealthState
        except ImportError:
            return None
        with self._lock:
            return ProviderHealthState(
                provider_name=self._provider_name,
                is_healthy=self._state == "closed",
                consecutive_failures=self._consecutive_failures,
                last_success=self._last_success_utc,
                last_failure=self._last_failure_utc,
                circuit_state=self._state,
            )


class CircuitBreakerRegistry:
    """Manages circuit breakers for all known providers."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def peek(self, provider_name: str) -> CircuitBreaker | None:
        """Lock-free lookup of an existing breaker (returns None if never created)."""
        return self._breakers.get(provider_name)

    def get_or_create(self, provider_name: str) -> CircuitBreaker:
        existing = self._breakers.get(provider_name)
        if existing is not None:
            return existing
        with self._lock:
            if provider_name not in self._breakers:
                new_map = dict(self._breakers)
                new_map[provider_name] = CircuitBreaker(
                    provider_name, self._failure_threshold, self._reset_timeout
                )
                self._breakers = new_map
            return self._breakers[provider_name]

    def get_health_summary(self) -> dict:
        with self._lock:
            return {name: cb.get_health_state() for name, cb in self._breakers.items()}
