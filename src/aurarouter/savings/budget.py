"""Budget enforcement for AuraRouter cost capping."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from aurarouter.savings.pricing import CostEngine, PricingCatalog

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

    def __init__(self, cost_engine: CostEngine, config: dict) -> None:
        self._cost_engine = cost_engine
        self._config = dict(config)
        self._lock = threading.Lock()
        # Cache: {"daily": (spend, monotonic_time), "monthly": (spend, monotonic_time)}
        self._spend_cache: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return self._config.get("enabled", False)

    def check_budget(self, provider: str) -> BudgetStatus:
        """Check whether a request to *provider* is within budget.

        Local providers are always allowed.  Cloud providers are checked
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
        if not PricingCatalog.is_cloud_provider(provider):
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
            self._spend_cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cached_spend(self, period: str) -> float:
        """Return spend for *period* (``"daily"`` or ``"monthly"``), using cache."""
        now = time.monotonic()
        with self._lock:
            cached = self._spend_cache.get(period)
            if cached is not None:
                value, ts = cached
                if (now - ts) < _CACHE_TTL_SECONDS:
                    return value

        # Cache miss — query cost engine (outside lock to avoid holding it
        # during the potentially slow SQLite query)
        utc_now = datetime.now(timezone.utc)
        if period == "daily":
            start = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = utc_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        spend = self._cost_engine.total_spend(start=start.isoformat())

        with self._lock:
            self._spend_cache[period] = (spend, now)
        return spend
