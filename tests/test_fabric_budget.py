"""Tests for budget enforcement in ComputeFabric."""

from datetime import datetime, timezone
from unittest.mock import patch

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.models import GenerateResult, UsageRecord
from aurarouter.savings.pricing import CostEngine, PricingCatalog
from aurarouter.savings.usage_store import UsageStore


_OLLAMA_MODEL = {"provider": "ollama", "model_name": "test", "endpoint": "http://x"}
_GOOGLE_MODEL = {"provider": "google", "model_name": "gemini-2.0-flash", "api_key": "K"}


def _make_config(models: dict, roles: dict) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return cfg


def _make_budget_manager(tmp_path, *, daily_limit=None, monthly_limit=None, spend=0.0):
    """Create a BudgetManager with a mocked total_spend return value."""
    store = UsageStore(db_path=tmp_path / "usage.db")
    catalog = PricingCatalog()
    engine = CostEngine(catalog, store)

    config = {
        "enabled": True,
        "daily_limit": daily_limit,
        "monthly_limit": monthly_limit,
    }
    mgr = BudgetManager(engine, config)

    # Seed the cache so we don't need real records
    import time
    now = time.monotonic()
    mgr._spend_cache["daily"] = (spend, now)
    mgr._spend_cache["monthly"] = (spend, now)
    return mgr


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_cloud_skipped_on_budget(tmp_path, monkeypatch):
    """Budget exceeded → cloud model skipped, local fallback used."""
    config = _make_config(
        models={
            "cloud": _GOOGLE_MODEL,
            "local": _OLLAMA_MODEL,
        },
        roles={"coding": ["cloud", "local"]},
    )
    mgr = _make_budget_manager(tmp_path, daily_limit=5.00, spend=10.00)
    fabric = ComputeFabric(config, budget_manager=mgr)

    fake_result = GenerateResult(text="local response", input_tokens=1, output_tokens=1)
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        result = fabric.execute("coding", "test prompt")

    assert result == "local response"


def test_all_cloud_budget_exceeded(tmp_path, monkeypatch):
    """All models are cloud, budget exceeded → returns budget error message."""
    config = _make_config(
        models={
            "cloud1": _GOOGLE_MODEL,
            "cloud2": {**_GOOGLE_MODEL, "model_name": "gemini-2.0-pro"},
        },
        roles={"coding": ["cloud1", "cloud2"]},
    )
    mgr = _make_budget_manager(tmp_path, daily_limit=5.00, spend=10.00)
    fabric = ComputeFabric(config, budget_manager=mgr)

    result = fabric.execute("coding", "test prompt")

    assert result is not None
    assert "BUDGET_EXCEEDED" in result
    assert "Configure local models as fallback" in result


def test_no_budget_manager_no_effect(config, monkeypatch):
    """No manager provided → cloud models used normally."""
    fabric = ComputeFabric(config)  # no budget_manager

    fake_result = GenerateResult(text="cloud ok", input_tokens=5, output_tokens=10)
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
        return_value=fake_result,
    ):
        result = fabric.execute("coding", "prompt")

    assert result == "cloud ok"
