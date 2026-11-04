"""Budget enforcement for AuraRouter cost capping."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from aurarouter.savings.pricing import CostEngine, PricingCatalog, is_cloud_tier

_CACHE_TTL_SECONDS = 60.0


@dataclass
class BudgetStatus:
    """Result of a budget check."""

    allowed: bool
    reason: str
    daily_spend: float
    monthly_spend: float
    daily_limit: float | None
    monthly_limit: float | None


class BudgetManager:
    """Thread-safe budget enforcement with cached spend lookups.

    Checks daily and monthly spending against configured limits before
    cloud provider calls.  Local providers are always allowed.
    """

    def __init__(
        self,
        cost_engine: CostEngine,
        config: dict,
        event_reporter: Any = None,
    ) -> None:
        self._cost_engine = cost_engine
        self._config = dict(config)
        self._event_reporter = event_reporter
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._refresh_in_progress = False
        # Cache: {"daily": (spend, monotonic_time), "monthly": (spend, monotonic_time)}
        self._spend_cache: dict[str, tuple[float, float]] = {}

    def set_event_reporter(self, event_reporter: Any) -> None:
        """Attach an EventReporter for non-blocking background spend refreshes."""
        self._event_reporter = event_reporter
        if self.is_enabled():
            self._schedule_async_refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return self._config.get("enabled", False)

    @staticmethod
    def _is_cloud_target(
        provider: str,
        hosting_tier: str | None = None,
        endpoint: str | None = None,
    ) -> bool:
        if hosting_tier is not None or endpoint is not None:
            return is_cloud_tier(hosting_tier, provider, endpoint=endpoint)
        return PricingCatalog.is_cloud_provider(provider)

    def _schedule_async_refresh(self) -> None:
        """Singleflight background refresh of daily & monthly spend off the hot path."""
        if self._refresh_in_progress:
            return
        with self._lock:
            if self._refresh_in_progress:
                return
            self._refresh_in_progress = True

        def _job() -> None:
            try:
                self._cached_spend("daily")
                self._cached_spend("monthly")
            finally:
                with self._lock:
                    self._refresh_in_progress = False

        reporter = self._event_reporter
        if reporter is not None and hasattr(reporter, "submit"):
            try:
                reporter.submit(_job)
                return
            except Exception:
                pass

        t = threading.Thread(target=_job, name="AuraRouter-BudgetRefresh", daemon=True)
        t.start()

    def peek_budget_allowed(
        self,
        provider: str,
        hosting_tier: str | None = None,
        endpoint: str | None = None,
    ) -> tuple[bool, str]:
        """Strictly lock-free, non-blocking hot-path budget check.

        Never executes synchronous SQLite scans on the routing thread:
        if the spend cache is cold or past its 60s TTL, returns the last
        known stale evaluation (or ``(True, "")`` on cold start) while
        triggering a singleflight asynchronous background refresh.
        """
        cfg = self._config
        if not cfg.get("enabled", False):
            return True, ""
        if not self._is_cloud_target(provider, hosting_tier=hosting_tier, endpoint=endpoint):
            return True, ""

        now = time.monotonic()
        cache = self._spend_cache
        daily_entry = cache.get("daily")
        monthly_entry = cache.get("monthly")

        daily_fresh = daily_entry is not None and (now - daily_entry[1]) < _CACHE_TTL_SECONDS
        monthly_fresh = monthly_entry is not None and (now - monthly_entry[1]) < _CACHE_TTL_SECONDS
        if not (daily_fresh and monthly_fresh):
            self._schedule_async_refresh()

        daily_limit = cfg.get("daily_limit")
        monthly_limit = cfg.get("monthly_limit")

        if daily_limit is not None and daily_entry is not None:
            daily_spend = daily_entry[0]
            if daily_spend >= daily_limit:
                return False, f"Daily budget exceeded (${daily_spend:.2f}/${daily_limit:.2f})"

        if monthly_limit is not None and monthly_entry is not None:
            monthly_spend = monthly_entry[0]
            if monthly_spend >= monthly_limit:
                return False, f"Monthly budget exceeded (${monthly_spend:.2f}/${monthly_limit:.2f})"

        return True, ""

    def peek_remaining_ratio(self) -> float | None:
        """Lock-free remaining budget ratio in [0.0, 1.0], or None if disabled/uncached."""
        cfg = self._config
        if not cfg.get("enabled", False):
            return None
        now = time.monotonic()
        cache = self._spend_cache
        ratios: list[float] = []
        daily_limit = cfg.get("daily_limit")
        if daily_limit and daily_limit > 0:
            de = cache.get("daily")
            if de is not None:
                if (now - de[1]) >= _CACHE_TTL_SECONDS:
                    self._schedule_async_refresh()
                ratios.append(max(0.0, min(1.0, (daily_limit - de[0]) / daily_limit)))
        monthly_limit = cfg.get("monthly_limit")
        if monthly_limit and monthly_limit > 0:
            me = cache.get("monthly")
            if me is not None:
                if (now - me[1]) >= _CACHE_TTL_SECONDS:
                    self._schedule_async_refresh()
                ratios.append(max(0.0, min(1.0, (monthly_limit - me[0]) / monthly_limit)))
        return min(ratios) if ratios else None

    def check_budget(
        self,
        provider: str,
        hosting_tier: str | None = None,
        endpoint: str | None = None,
    ) -> BudgetStatus:
        """Check whether a request to *provider* is within budget.

        Local providers are always allowed.  Cloud providers (including
        ``openapi`` endpoints pointing to known cloud hosts) are checked
        against daily and monthly limits.
        """
        if not self.is_enabled():
            return BudgetStatus(
                allowed=True,
                reason="",
                daily_spend=0.0,
                monthly_spend=0.0,
                daily_limit=None,
                monthly_limit=None,
            )

        # Local providers are never budget-gated
        if not self._is_cloud_target(provider, hosting_tier=hosting_tier, endpoint=endpoint):
            return BudgetStatus(
                allowed=True,
                reason="",
                daily_spend=self.get_daily_spend(),
                monthly_spend=self.get_monthly_spend(),
                daily_limit=self._config.get("daily_limit"),
                monthly_limit=self._config.get("monthly_limit"),
            )

        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()
        daily_limit = self._config.get("daily_limit")
        monthly_limit = self._config.get("monthly_limit")

        # Check daily limit
        if daily_limit is not None and daily_spend >= daily_limit:
            return BudgetStatus(
                allowed=False,
                reason=f"Daily budget exceeded (${daily_spend:.2f}/${daily_limit:.2f})",
                daily_spend=daily_spend,
                monthly_spend=monthly_spend,
                daily_limit=daily_limit,
                monthly_limit=monthly_limit,
            )

        # Check monthly limit
        if monthly_limit is not None and monthly_spend >= monthly_limit:
            return BudgetStatus(
                allowed=False,
                reason=f"Monthly budget exceeded (${monthly_spend:.2f}/${monthly_limit:.2f})",
                daily_spend=daily_spend,
                monthly_spend=monthly_spend,
                daily_limit=daily_limit,
                monthly_limit=monthly_limit,
            )

        return BudgetStatus(
            allowed=True,
            reason="",
            daily_spend=daily_spend,
            monthly_spend=monthly_spend,
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
        )

    def get_daily_spend(self) -> float:
        """Return today's spend (cached with TTL)."""
        return self._cached_spend("daily")

    def get_monthly_spend(self) -> float:
        """Return this month's spend (cached with TTL)."""
        return self._cached_spend("monthly")

    def get_daily_remaining(self) -> float | None:
        """Remaining daily budget, or ``None`` if no daily limit is set."""
        limit = self._config.get("daily_limit")
        if limit is None:
            return None
        return max(0.0, limit - self.get_daily_spend())

    def get_monthly_remaining(self) -> float | None:
        """Remaining monthly budget, or ``None`` if no monthly limit is set."""
        limit = self._config.get("monthly_limit")
        if limit is None:
            return None
        return max(0.0, limit - self.get_monthly_spend())

    def update_config(self, config: dict) -> None:
        """Hot-reload budget settings.  Clears the spend cache."""
        with self._lock:
            self._config = dict(config)
            self._spend_cache = {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cached_spend(self, period: str) -> float:
        """Return spend for *period* (``"daily"`` or ``"monthly"``), serialized against stampedes."""
        now = time.monotonic()
        cached = self._spend_cache.get(period)
        if cached is not None and (now - cached[1]) < _CACHE_TTL_SECONDS:
            return cached[0]

        # Singleflight refresh lock prevents thundering-herd SQLite scans at TTL expiry
        with self._refresh_lock:
            now = time.monotonic()
            cached = self._spend_cache.get(period)
            if cached is not None and (now - cached[1]) < _CACHE_TTL_SECONDS:
                return cached[0]

            utc_now = datetime.now(timezone.utc)
            if period == "daily":
                start = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                start = utc_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            spend = self._cost_engine.total_spend(start=start.isoformat())

            with self._lock:
                new_cache = dict(self._spend_cache)
                new_cache[period] = (spend, time.monotonic())
                self._spend_cache = new_cache
            return spend
