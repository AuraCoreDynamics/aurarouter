"""Pricing catalog and cost engine for the Shadow Accountant feature."""

from __future__ import annotations

import calendar
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from aurarouter.savings.usage_store import UsageStore

_LOCAL_PROVIDERS = frozenset({"ollama", "llamacpp", "llamacpp-server"})
_CLOUD_PROVIDERS = frozenset({"google", "claude"})


@dataclass(frozen=True)
class ModelPrice:
    """Cost per 1 million tokens for a specific model."""

    input_per_million: float
    output_per_million: float


# ── Built-in defaults ────────────────────────────────────────────────
_BUILTIN_PRICES: dict[str, ModelPrice] = {
    # Claude
    "claude-sonnet-4-5-20250929": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5-20251001": ModelPrice(0.80, 4.00),
    # Gemini
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    "gemini-2.0-pro": ModelPrice(1.25, 10.00),
    # Provider catch-alls (local = free)
    "ollama:*": ModelPrice(0.0, 0.0),
    "llamacpp:*": ModelPrice(0.0, 0.0),
    "llamacpp-server:*": ModelPrice(0.0, 0.0),
}

_ZERO = ModelPrice(0.0, 0.0)


class PricingCatalog:
    """Thread-safe lookup of per-model token prices.

    Resolution order for ``get_price(model_name, provider)``:
    1. Exact ``model_name`` match (user overrides first, then built-ins).
    2. ``provider:*`` catch-all.
    3. ``ModelPrice(0, 0)`` fallback.
    """

    def __init__(self, overrides: dict[str, ModelPrice] | None = None) -> None:
        self._prices: dict[str, ModelPrice] = dict(_BUILTIN_PRICES)
        if overrides:
            self._prices.update(overrides)
        self._lock = threading.Lock()

    def get_price(self, model_name: str, provider: str) -> ModelPrice:
        """Look up the token price for *model_name* on *provider*."""
        with self._lock:
            # 1. Exact model name
            price = self._prices.get(model_name)
            if price is not None:
                return price
            # 2. Provider catch-all
            price = self._prices.get(f"{provider}:*")
            if price is not None:
                return price
            # 3. Default to free
            return _ZERO

    @staticmethod
    def is_cloud_provider(provider: str) -> bool:
        """Return ``True`` for cloud providers (``google``, ``claude``)."""
        return provider in _CLOUD_PROVIDERS


class CostEngine:
    """Calculates actual costs, shadow costs, projections, and ROI."""

    def __init__(self, catalog: PricingCatalog, store: UsageStore) -> None:
        self._catalog = catalog
        self._store = store

    # ── Single-request cost ──────────────────────────────────────────

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        provider: str,
    ) -> float:
        """Return the dollar cost for a single request."""
        price = self._catalog.get_price(model_name, provider)
        return (
            input_tokens * price.input_per_million
            + output_tokens * price.output_per_million
        ) / 1_000_000

    # ── Shadow cost ──────────────────────────────────────────────────

    def shadow_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        actual_model: str,
        actual_provider: str,
        shadow_model: str,
        shadow_provider: str,
    ) -> dict:
        """Compare actual vs. hypothetical routing cost.

        Returns ``{"actual_cost", "shadow_cost", "savings"}``.
        Positive *savings* means the actual route was cheaper.
        """
        actual = self.calculate_cost(
            input_tokens, output_tokens, actual_model, actual_provider
        )
        shadow = self.calculate_cost(
            input_tokens, output_tokens, shadow_model, shadow_provider
        )
        return {
            "actual_cost": actual,
            "shadow_cost": shadow,
            "savings": shadow - actual,
        }

    # ── Aggregate spend ──────────────────────────────────────────────

    def total_spend(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> float:
        """Sum dollar cost of all recorded usage in the date range."""
        records = self._store.query(start=start, end=end)
        return sum(
            self.calculate_cost(r.input_tokens, r.output_tokens, r.model_id, r.provider)
            for r in records
        )

    def spend_by_provider(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, float]:
        """Per-provider dollar spend in the date range."""
        records = self._store.query(start=start, end=end)
        breakdown: dict[str, float] = {}
        for r in records:
            cost = self.calculate_cost(
                r.input_tokens, r.output_tokens, r.model_id, r.provider
            )
            breakdown[r.provider] = breakdown.get(r.provider, 0.0) + cost
        return breakdown

    # ── Projections ──────────────────────────────────────────────────

    def monthly_projection(self) -> dict:
        """Linear projection of current-month spend.

        Returns ``{"spent_so_far", "projected_monthly", "days_elapsed",
        "days_in_month"}``.
        """
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        days_in_month = calendar.monthrange(now.year, now.month)[1]
        days_elapsed = now.day

        spent = self.total_spend(start=month_start.isoformat())

        if days_elapsed > 0:
            projected = (spent / days_elapsed) * days_in_month
        else:
            projected = 0.0

        return {
            "spent_so_far": spent,
            "projected_monthly": projected,
            "days_elapsed": days_elapsed,
            "days_in_month": days_in_month,
        }

    # ── ROI ──────────────────────────────────────────────────────────

    def roi_estimate(
        self,
        hardware_cost: float,
        monthly_cloud_spend: float | None = None,
    ) -> dict:
        """Estimate GPU payback period.

        Returns ``{"monthly_cloud_spend", "payback_months",
        "annual_savings"}``.
        """
        if monthly_cloud_spend is None:
            monthly_cloud_spend = self.monthly_projection()["projected_monthly"]

        if monthly_cloud_spend > 0:
            payback = hardware_cost / monthly_cloud_spend
        else:
            payback = float("inf")

        return {
            "monthly_cloud_spend": monthly_cloud_spend,
            "payback_months": payback,
            "annual_savings": monthly_cloud_spend * 12,
        }