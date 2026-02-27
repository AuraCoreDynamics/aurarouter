"""Tests for the pricing catalog and cost engine."""

import calendar
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from aurarouter.savings.models import UsageRecord
from aurarouter.savings.pricing import CostEngine, ModelPrice, PricingCatalog
from aurarouter.savings.usage_store import UsageStore


def _make_record(**overrides) -> UsageRecord:
    defaults = dict(
        timestamp="2026-02-10T10:00:00Z",
        model_id="gemini-2.0-flash",
        provider="google",
        role="router",
        intent="SIMPLE_CODE",
        input_tokens=1000,
        output_tokens=500,
        elapsed_s=1.0,
        success=True,
        is_cloud=True,
    )
    defaults.update(overrides)
    return UsageRecord(**defaults)


# ── PricingCatalog ───────────────────────────────────────────────────


def test_catalog_default_cloud_prices():
    catalog = PricingCatalog()
    flash = catalog.get_price("gemini-2.0-flash", "google")
    assert flash.input_per_million == 0.10
    assert flash.output_per_million == 0.40

    sonnet = catalog.get_price("claude-sonnet-4-5-20250929", "claude")
    assert sonnet.input_per_million == 3.00
    assert sonnet.output_per_million == 15.00


def test_catalog_local_free():
    catalog = PricingCatalog()
    ollama = catalog.get_price("llama3:8b", "ollama")
    assert ollama == ModelPrice(0.0, 0.0)

    llamacpp = catalog.get_price("mistral-7b", "llamacpp")
    assert llamacpp == ModelPrice(0.0, 0.0)

    llamacpp_server = catalog.get_price("phi-3", "llamacpp-server")
    assert llamacpp_server == ModelPrice(0.0, 0.0)


def test_catalog_overrides():
    custom = {"gemini-2.0-flash": ModelPrice(0.50, 1.00)}
    catalog = PricingCatalog(overrides=custom)
    price = catalog.get_price("gemini-2.0-flash", "google")
    assert price.input_per_million == 0.50
    assert price.output_per_million == 1.00


def test_catalog_unknown_model_fallback():
    catalog = PricingCatalog()
    price = catalog.get_price("totally-unknown-model", "unknown-provider")
    assert price == ModelPrice(0.0, 0.0)


def test_is_cloud_provider():
    assert PricingCatalog.is_cloud_provider("google") is True
    assert PricingCatalog.is_cloud_provider("claude") is True
    assert PricingCatalog.is_cloud_provider("ollama") is False
    assert PricingCatalog.is_cloud_provider("llamacpp") is False
    assert PricingCatalog.is_cloud_provider("llamacpp-server") is False


# ── CostEngine ───────────────────────────────────────────────────────


def test_calculate_cost_cloud():
    catalog = PricingCatalog()
    store = UsageStore.__new__(UsageStore)  # dummy, not used here
    engine = CostEngine(catalog, store)

    # Gemini Flash: 1000 input @ $0.10/1M + 500 output @ $0.40/1M
    cost = engine.calculate_cost(1000, 500, "gemini-2.0-flash", "google")
    expected = (1000 * 0.10 + 500 * 0.40) / 1_000_000
    assert cost == pytest.approx(expected)


def test_calculate_cost_local():
    catalog = PricingCatalog()
    store = UsageStore.__new__(UsageStore)
    engine = CostEngine(catalog, store)
    cost = engine.calculate_cost(50000, 30000, "llama3:8b", "ollama")
    assert cost == 0.0


def test_shadow_cost():
    catalog = PricingCatalog()
    store = UsageStore.__new__(UsageStore)
    engine = CostEngine(catalog, store)

    result = engine.shadow_cost(
        input_tokens=10000,
        output_tokens=5000,
        actual_model="gemini-2.0-flash",
        actual_provider="google",
        shadow_model="llama3:8b",
        shadow_provider="ollama",
    )

    actual_expected = (10000 * 0.10 + 5000 * 0.40) / 1_000_000
    assert result["actual_cost"] == pytest.approx(actual_expected)
    assert result["shadow_cost"] == 0.0
    assert result["savings"] == pytest.approx(-actual_expected)


def test_total_spend(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    # Two Gemini Flash requests
    store.record(_make_record(input_tokens=1_000_000, output_tokens=0))
    store.record(_make_record(input_tokens=0, output_tokens=1_000_000))

    catalog = PricingCatalog()
    engine = CostEngine(catalog, store)
    total = engine.total_spend()
    # $0.10 for 1M input + $0.40 for 1M output = $0.50
    assert total == pytest.approx(0.50)


def test_spend_by_provider(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")
    store.record(
        _make_record(
            model_id="gemini-2.0-flash",
            provider="google",
            input_tokens=1_000_000,
            output_tokens=0,
        )
    )
    store.record(
        _make_record(
            model_id="claude-sonnet-4-5-20250929",
            provider="claude",
            input_tokens=1_000_000,
            output_tokens=0,
        )
    )
    store.record(
        _make_record(
            model_id="llama3:8b",
            provider="ollama",
            input_tokens=1_000_000,
            output_tokens=0,
            is_cloud=False,
        )
    )

    catalog = PricingCatalog()
    engine = CostEngine(catalog, store)
    breakdown = engine.spend_by_provider()

    assert breakdown["google"] == pytest.approx(0.10)
    assert breakdown["claude"] == pytest.approx(3.00)
    assert breakdown["ollama"] == pytest.approx(0.0)


def test_monthly_projection(tmp_path):
    store = UsageStore(db_path=tmp_path / "usage.db")

    now = datetime.now(timezone.utc)
    days_in_month = calendar.monthrange(now.year, now.month)[1]

    # Insert a record dated today (current month)
    ts = now.replace(hour=8, minute=0, second=0, microsecond=0).isoformat()
    store.record(
        _make_record(
            timestamp=ts,
            model_id="gemini-2.0-flash",
            provider="google",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
    )

    catalog = PricingCatalog()
    engine = CostEngine(catalog, store)
    proj = engine.monthly_projection()

    # Spent: $0.10 + $0.40 = $0.50
    assert proj["spent_so_far"] == pytest.approx(0.50)
    assert proj["days_elapsed"] == now.day
    assert proj["days_in_month"] == days_in_month

    expected_projected = (0.50 / now.day) * days_in_month
    assert proj["projected_monthly"] == pytest.approx(expected_projected)


def test_roi_estimate():
    catalog = PricingCatalog()
    store = UsageStore.__new__(UsageStore)
    engine = CostEngine(catalog, store)

    result = engine.roi_estimate(hardware_cost=500.0, monthly_cloud_spend=50.0)
    assert result["monthly_cloud_spend"] == 50.0
    assert result["payback_months"] == pytest.approx(10.0)
    assert result["annual_savings"] == pytest.approx(600.0)


# ── Resolution cascade (TG3) ────────────────────────────────────────


def test_cascade_config_pricing_takes_priority():
    """Explicit config pricing beats everything."""
    catalog = PricingCatalog()
    price = catalog.get_price(
        "gemini-2.0-flash", "google",
        config_pricing=(0.99, 1.99),
    )
    assert price.input_per_million == 0.99
    assert price.output_per_million == 1.99


def test_cascade_partial_config_falls_through():
    """Partial config pricing (one None) falls through to built-in."""
    catalog = PricingCatalog()
    price = catalog.get_price(
        "gemini-2.0-flash", "google",
        config_pricing=(0.99, None),
    )
    # Should fall through to built-in Gemini Flash price
    assert price.input_per_million == 0.10
    assert price.output_per_million == 0.40


def test_cascade_config_resolver():
    """Config resolver provides pricing when no explicit config_pricing."""
    def resolver(model_name: str):
        if model_name == "custom-model":
            return (5.00, 10.00)
        return (None, None)

    catalog = PricingCatalog(config_resolver=resolver)
    price = catalog.get_price("custom-model", "google")
    assert price.input_per_million == 5.00
    assert price.output_per_million == 10.00


def test_cascade_resolver_none_falls_through():
    """Config resolver returning (None, None) falls through to built-in."""
    def resolver(model_name: str):
        return (None, None)

    catalog = PricingCatalog(config_resolver=resolver)
    price = catalog.get_price("gemini-2.0-flash", "google")
    assert price.input_per_million == 0.10


def test_cascade_override_beats_builtin():
    """User override still beats built-in prices (existing behavior)."""
    custom = {"gemini-2.0-flash": ModelPrice(0.50, 1.00)}
    catalog = PricingCatalog(overrides=custom)
    price = catalog.get_price("gemini-2.0-flash", "google")
    assert price.input_per_million == 0.50


def test_cascade_config_pricing_beats_override():
    """Explicit config pricing beats user overrides."""
    custom = {"gemini-2.0-flash": ModelPrice(0.50, 1.00)}
    catalog = PricingCatalog(overrides=custom)
    price = catalog.get_price(
        "gemini-2.0-flash", "google",
        config_pricing=(0.01, 0.02),
    )
    assert price.input_per_million == 0.01
    assert price.output_per_million == 0.02


# ── Hosting tier resolution ─────────────────────────────────────────


from aurarouter.savings.pricing import resolve_hosting_tier, is_cloud_tier


def test_resolve_hosting_tier_explicit():
    assert resolve_hosting_tier("on-prem", "google") == "on-prem"
    assert resolve_hosting_tier("cloud", "ollama") == "cloud"
    assert resolve_hosting_tier("dedicated-tenant", "google") == "dedicated-tenant"


def test_resolve_hosting_tier_fallback_to_provider():
    assert resolve_hosting_tier(None, "google") == "cloud"
    assert resolve_hosting_tier(None, "claude") == "cloud"
    assert resolve_hosting_tier(None, "ollama") == "on-prem"
    assert resolve_hosting_tier(None, "llamacpp") == "on-prem"
    assert resolve_hosting_tier(None, "llamacpp-server") == "on-prem"


def test_resolve_hosting_tier_unknown_provider():
    assert resolve_hosting_tier(None, "unknown-provider") == "on-prem"


def test_is_cloud_tier():
    assert is_cloud_tier("cloud", "google") is True
    assert is_cloud_tier("on-prem", "google") is False
    assert is_cloud_tier(None, "google") is True  # provider fallback
    assert is_cloud_tier(None, "ollama") is False
    assert is_cloud_tier("dedicated-tenant", "claude") is False


def test_config_get_model_hosting_tier():
    from aurarouter.config import ConfigLoader
    config = ConfigLoader(allow_missing=True)
    config.config = {
        "models": {
            "my_local": {"provider": "ollama", "hosting_tier": "on-prem"},
            "my_cloud": {"provider": "google"},  # no hosting_tier
        }
    }
    assert config.get_model_hosting_tier("my_local") == "on-prem"
    assert config.get_model_hosting_tier("my_cloud") is None
    assert config.get_model_hosting_tier("nonexistent") is None


# ── Boot-time pricing resolution (Phase 3 TG 3.1) ────────────────


def test_pricing_catalog_resolves_at_boot():
    """PricingCatalog with config_resolver resolves model pricing at construction time."""
    from aurarouter.config import ConfigLoader

    config = ConfigLoader(allow_missing=True)
    config.config = {
        "models": {
            "custom-model": {
                "provider": "openapi",
                "cost_per_1m_input": 5.0,
                "cost_per_1m_output": 15.0,
            },
        },
    }

    # Construct with config_resolver (the fix)
    catalog = PricingCatalog(config_resolver=config.get_model_pricing)

    # Should resolve model-level pricing immediately — no update_config needed
    price = catalog.get_price("custom-model", "openapi")
    assert price.input_per_million == 5.0
    assert price.output_per_million == 15.0


def test_pricing_catalog_without_resolver_falls_through():
    """PricingCatalog without config_resolver falls through to provider defaults."""
    catalog = PricingCatalog()  # no config_resolver

    # "custom-model" is unknown — should fall through to openapi catch-all or zero
    price = catalog.get_price("custom-model", "openapi")
    assert price == ModelPrice(0.0, 0.0)
