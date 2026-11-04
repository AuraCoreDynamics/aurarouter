"""Pricing catalog and cost engine for the Shadow Accountant feature."""

from __future__ import annotations

import calendar
import functools
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from aurarouter.savings.usage_store import UsageStore

# ── Hosting tier resolution ─────────────────────────────────────────

_PROVIDER_DEFAULT_TIERS: dict[str, str] = {
    "ollama": "on-prem",
    "llamacpp": "on-prem",
    "llamacpp-server": "on-prem",
    "openapi": "on-prem",  # Default for local vLLM/LM Studio; overridden if endpoint is a known cloud host
    "google": "cloud",
    "gemini": "cloud",
    "anthropic": "cloud",
    "claude": "cloud",
    "openai": "cloud",
    "azure_openai": "cloud",
    "mcp": "cloud",
}

_KNOWN_CLOUD_ENDPOINT_DOMAINS: tuple[str, ...] = (
    "api.openai.com",
    ".openai.azure.com",
    ".cognitiveservices.azure.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
    "openrouter.ai",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.mistral.ai",
    "api.deepseek.com",
    "api.cohere.com",
    "api.perplexity.ai",
)

_CLOUD_TIERS = frozenset({"cloud"})


@functools.lru_cache(maxsize=512)
def infer_hosting_tier_from_endpoint(endpoint: str | None) -> str | None:
    """Return ``'cloud'`` if *endpoint* points to a known cloud inference host, else ``None``."""
    if not endpoint or not isinstance(endpoint, str):
        return None
    try:
        host = (urlparse(endpoint).hostname or "").lower()
    except Exception:
        return None
    if not host:
        return None
    for domain in _KNOWN_CLOUD_ENDPOINT_DOMAINS:
        if domain.startswith("."):
            if host.endswith(domain) or host == domain[1:]:
                return "cloud"
        elif host == domain or host.endswith("." + domain):
            return "cloud"
    return None


def resolve_hosting_tier(
    hosting_tier: str | None,
    provider: str,
    endpoint: str | None = None,
) -> str:
    """Determine the effective hosting tier for a model.

    Resolution order:
    1. Explicit ``hosting_tier`` from model config (if not None).
    2. Cloud host auto-detection from ``endpoint`` (closes accidental PII/budget bypass
       when ``provider: openapi`` points to ``api.openai.com``, ``*.openai.azure.com``, etc.).
    3. Provider-name default from ``_PROVIDER_DEFAULT_TIERS``.
    4. ``"on-prem"`` as ultimate fallback.
    """
    if hosting_tier is not None:
        return hosting_tier
    inferred = infer_hosting_tier_from_endpoint(endpoint)
    if inferred is not None:
        return inferred
    return _PROVIDER_DEFAULT_TIERS.get(provider, "on-prem")


def is_cloud_tier(
    hosting_tier: str | None,
    provider: str,
    endpoint: str | None = None,
) -> bool:
    """Return True if the resolved hosting tier is cloud."""
    if hosting_tier is not None:
        return hosting_tier in _CLOUD_TIERS
    return resolve_hosting_tier(hosting_tier, provider, endpoint=endpoint) in _CLOUD_TIERS


def resolve_canonical_vendor(
    model_id: str,
    model_cfg: dict[str, Any] | None = None,
) -> str:
    """Resolve canonical upstream vendor key for rate-limiting, cache TTLs, and pricing.

    Maps generic transport providers (``mcp``, cloud ``openapi``) and aliases
    (``claude``, ``gemini``) onto canonical vendor identifiers (``anthropic``,
    ``google``, ``openai``, ``azure_openai``) so configured vendor quotas and
    built-in pricing apply seamlessly while keeping distinct unconfigured
    endpoints isolated.
    """
    if isinstance(model_cfg, dict):
        cached = model_cfg.get("_canonical_vendor")
        if cached is not None:
            return cached

    cfg = model_cfg if isinstance(model_cfg, dict) else {}
    explicit_vendor = str(cfg.get("vendor") or "").strip().lower()
    if explicit_vendor:
        if explicit_vendor == "claude":
            res = "anthropic"
        elif explicit_vendor == "gemini":
            res = "google"
        else:
            res = explicit_vendor
        if isinstance(model_cfg, dict):
            model_cfg["_canonical_vendor"] = res
        return res

    raw_prov = str(cfg.get("provider") or "").strip().lower()
    if raw_prov == "claude":
        res = "anthropic"
        if isinstance(model_cfg, dict):
            model_cfg["_canonical_vendor"] = res
        return res
    if raw_prov == "gemini":
        res = "google"
        if isinstance(model_cfg, dict):
            model_cfg["_canonical_vendor"] = res
        return res
    if raw_prov and raw_prov not in ("mcp", "openapi"):
        if isinstance(model_cfg, dict):
            model_cfg["_canonical_vendor"] = raw_prov
        return raw_prov

    endpoint = str(cfg.get("mcp_endpoint") or cfg.get("endpoint") or "").strip().lower()
    if endpoint:
        if "api.anthropic.com" in endpoint or "aurarouter-claude" in endpoint:
            res = "anthropic"
        elif (
            "generativelanguage.googleapis.com" in endpoint
            or "aurarouter-gemini" in endpoint
            or "aiplatform.googleapis.com" in endpoint
        ):
            res = "google"
        elif ".openai.azure.com" in endpoint or ".cognitiveservices.azure.com" in endpoint:
            res = "azure_openai"
        elif "api.openai.com" in endpoint or "aurarouter-openai" in endpoint:
            res = "openai"
        else:
            res = None
        if res is not None:
            if isinstance(model_cfg, dict):
                model_cfg["_canonical_vendor"] = res
            return res

    # For openapi endpoints pointing to local/on-prem servers (e.g. localhost vLLM),
    # keep "openapi" unless explicitly marked as cloud.
    if raw_prov == "openapi" and not is_cloud_tier(cfg.get("hosting_tier"), raw_prov, endpoint=endpoint):
        if isinstance(model_cfg, dict):
            model_cfg["_canonical_vendor"] = "openapi"
        return "openapi"

    model_name = str(cfg.get("model_name") or "").strip().lower()
    tags_iter = cfg.get("tags") or ()
    tags_str = " ".join(str(t).lower() for t in tags_iter) if isinstance(tags_iter, (list, tuple, set)) else ""
    candidates = (model_name, model_id.lower(), tags_str)

    for text in candidates:
        if not text:
            continue
        if "claude" in text or "anthropic" in text:
            res = "anthropic"
            if isinstance(model_cfg, dict):
                model_cfg["_canonical_vendor"] = res
            return res
        if "gemini" in text or text == "google" or " google " in f" {text} ":
            res = "google"
            if isinstance(model_cfg, dict):
                model_cfg["_canonical_vendor"] = res
            return res
        if (
            text.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
            or any(tok.startswith(("gpt-", "o1-", "o3-", "o4-")) for tok in text.replace("/", " ").split())
            or "openai" in text
        ):
            res = "openai"
            if isinstance(model_cfg, dict):
                model_cfg["_canonical_vendor"] = res
            return res

    res = raw_prov or "unknown"
    if isinstance(model_cfg, dict):
        model_cfg["_canonical_vendor"] = res
    return res


@dataclass(frozen=True)
class ModelPrice:
    """Cost per 1 million tokens for a specific model."""

    input_per_million: float
    output_per_million: float


# ── Built-in defaults ────────────────────────────────────────────────
_BUILTIN_PRICES: dict[str, ModelPrice] = {
    # Provider catch-alls (local = free)
    "ollama:*": ModelPrice(0.0, 0.0),
    "llamacpp:*": ModelPrice(0.0, 0.0),
    "llamacpp-server:*": ModelPrice(0.0, 0.0),
    # Google cloud models
    "google:*": ModelPrice(0.10, 0.40),
    "gemini:*": ModelPrice(0.15, 0.60),
    "gemini-2.0-flash": ModelPrice(0.10, 0.40),
    "gemini-2.0-pro": ModelPrice(1.25, 5.00),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    "gemini-2.5-flash": ModelPrice(0.15, 0.60),
    # Anthropic cloud models
    "anthropic:*": ModelPrice(3.00, 15.00),
    "claude:*": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-20250514": ModelPrice(3.00, 15.00),
    "claude-opus-4-20250514": ModelPrice(15.00, 75.00),
    "claude-3-5-haiku-20241022": ModelPrice(0.80, 4.00),
    # OpenAI & Azure OpenAI cloud models
    "openai:*": ModelPrice(2.50, 10.00),
    "azure_openai:*": ModelPrice(2.50, 10.00),
    "gpt-4o": ModelPrice(2.50, 10.00),
    "gpt-4o-mini": ModelPrice(0.15, 0.60),
    "gpt-4.1": ModelPrice(2.00, 8.00),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40),
    "gpt-5": ModelPrice(1.25, 10.00),
    "gpt-5-mini": ModelPrice(0.25, 2.00),
    "o3": ModelPrice(2.00, 8.00),
    "o3-mini": ModelPrice(1.10, 4.40),
    "o4-mini": ModelPrice(1.10, 4.40),
    "o1": ModelPrice(15.00, 60.00),
}

_ZERO = ModelPrice(0.0, 0.0)


class PricingCatalog:
    """Thread-safe lookup of per-model token prices.

    Resolution order for ``get_price(model_name, provider, config_pricing=None)``:
    1. Explicit config pricing (cost_per_1m_input/output from model config).
    2. Config resolver (model-level config lookup).
    3. Exact ``model_name`` match (user overrides first, then built-ins).
    4. ``provider:*`` catch-all.
    5. ``ModelPrice(0, 0)`` fallback.
    """

    def __init__(
        self,
        overrides: dict[str, ModelPrice] | None = None,
        config_resolver: Callable[[str], tuple[float | None, float | None]] | None = None,
    ) -> None:
        self._prices: dict[str, ModelPrice] = dict(_BUILTIN_PRICES)
        if overrides:
            self._prices.update(overrides)
        self._config_resolver = config_resolver
        self._lock = threading.Lock()

    def get_price(
        self,
        model_name: str,
        provider: str,
        config_pricing: tuple[float | None, float | None] | None = None,
    ) -> ModelPrice:
        """Look up the token price for *model_name* on *provider*.

        If *config_pricing* is provided as ``(input, output)`` and both
        values are not None, that takes priority over all other sources.
        """
        with self._lock:
            # 1. Explicit per-call config pricing
            if config_pricing is not None:
                inp, out = config_pricing
                if inp is not None and out is not None:
                    return ModelPrice(inp, out)

            # 2. Config resolver (model-level config)
            if self._config_resolver is not None:
                inp, out = self._config_resolver(model_name)
                if inp is not None and out is not None:
                    return ModelPrice(inp, out)

            # 3. Exact model name (overrides + built-ins)
            price = self._prices.get(model_name)
            if price is not None:
                return price

            # 4. Provider catch-all
            price = self._prices.get(f"{provider}:*")
            if price is not None:
                return price

            # 5. Default to free
            return _ZERO

    @staticmethod
    def is_cloud_provider(provider: str) -> bool:
        """Return ``True`` for cloud providers.

        Resolves via ``_PROVIDER_DEFAULT_TIERS`` — no separate frozen set.
        """
        return _PROVIDER_DEFAULT_TIERS.get(provider) == "cloud"


class CostEngine:
    """Calculates and aggregates model costs using UsageStore and PricingCatalog.
    
    Can be configured to use a dynamic BaseCostEstimator instead of the static PricingCatalog.
    """

    def __init__(
        self,
        pricing_catalog: PricingCatalog,
        usage_store: UsageStore,
        cost_estimator = None,
    ) -> None:
        self._catalog = pricing_catalog
        self._store = usage_store
        self._estimator = cost_estimator

    # ── Single-request cost ──────────────────────────────────────────

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        provider: str,
    ) -> float:
        """Return the dollar cost for a single request."""
        # Try dynamic estimator first
        if self._estimator:
            cost = self._estimator.estimate(provider, model_name, input_tokens, output_tokens)
            if cost is not None:
                return cost
                
        # Fallback to static pricing catalog
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
