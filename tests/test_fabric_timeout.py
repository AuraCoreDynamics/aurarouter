"""Provider timeout resilience tests.

Task Group C, TG4 — Verifies that httpx timeout exceptions trigger
fallback to the next provider in the chain, and that on_model_tried
callbacks correctly report timeout failures.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fabric(models: dict, roles: dict, **kwargs) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return ComputeFabric(cfg, **kwargs)


# ---------------------------------------------------------------------------
# Task 4.1 — Timeout Fallback Test
# ---------------------------------------------------------------------------

class TestTimeoutFallback:
    """Timeout on first provider falls back to second."""

    def test_timeout_triggers_fallback(self):
        fabric = _make_fabric(
            models={
                "slow_model": {"provider": "ollama", "model_name": "slow", "endpoint": "http://x"},
                "fast_model": {"provider": "ollama", "model_name": "fast", "endpoint": "http://y"},
            },
            roles={"coding": ["slow_model", "fast_model"]},
        )

        callback_log = []

        def callback(role, model_id, success, elapsed):
            callback_log.append({"model_id": model_id, "success": success})

        call_count = 0

        def mock_generate(prompt, json_mode=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("timed out")
            return GenerateResult(text="fast response")

        from unittest.mock import patch
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            side_effect=mock_generate,
        ):
            result = fabric.execute("coding", "test prompt", on_model_tried=callback)

        assert result == "fast response"

        # Verify callbacks
        assert len(callback_log) == 2
        assert callback_log[0]["model_id"] == "slow_model"
        assert callback_log[0]["success"] is False
        assert callback_log[1]["model_id"] == "fast_model"
        assert callback_log[1]["success"] is True


# ---------------------------------------------------------------------------
# Task 4.2 — All Providers Timeout Test
# ---------------------------------------------------------------------------

class TestAllProvidersTimeout:
    """When all providers timeout, result is None."""

    def test_all_timeout_returns_none(self):
        fabric = _make_fabric(
            models={
                "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
                "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
            },
            roles={"coding": ["m1", "m2"]},
        )

        callback_log = []

        def callback(role, model_id, success, elapsed):
            callback_log.append({"model_id": model_id, "success": success})

        from unittest.mock import patch
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            side_effect=httpx.ReadTimeout("timed out"),
        ):
            result = fabric.execute("coding", "test prompt", on_model_tried=callback)

        assert result is None

        assert len(callback_log) == 2
        assert all(c["success"] is False for c in callback_log)


# ---------------------------------------------------------------------------
# Task 4.3 — Budget + Timeout Combined Test
# ---------------------------------------------------------------------------

class TestBudgetPlusTimeout:
    """Cloud model skipped by budget, local model fails with timeout."""

    def test_budget_skip_then_timeout(self):
        from aurarouter.savings.budget import BudgetManager, BudgetStatus
        from aurarouter.savings.pricing import CostEngine, PricingCatalog

        # Build budget manager that blocks cloud providers
        pricing_catalog = PricingCatalog()
        usage_store = MagicMock()
        cost_engine = MagicMock(spec=CostEngine)
        budget_config = {"enabled": True, "daily_limit": 1.0, "monthly_limit": 10.0}
        budget_manager = BudgetManager(cost_engine, budget_config)

        # Force budget_manager to report exceeded for all providers
        budget_manager.check_budget = MagicMock(return_value=BudgetStatus(
            allowed=False,
            reason="Daily limit exceeded",
            daily_spend=5.0,
            monthly_spend=50.0,
            daily_limit=1.0,
            monthly_limit=10.0,
        ))

        fabric = _make_fabric(
            models={
                "cloud_model": {
                    "provider": "google",
                    "model_name": "gemini",
                    "api_key": "MOCK",
                },
                "local_model": {
                    "provider": "ollama",
                    "model_name": "local",
                    "endpoint": "http://x",
                },
            },
            roles={"coding": ["cloud_model", "local_model"]},
            budget_manager=budget_manager,
        )

        callback_log = []

        def callback(role, model_id, success, elapsed):
            callback_log.append({"model_id": model_id, "success": success})

        from unittest.mock import patch
        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate_with_usage",
            side_effect=httpx.ConnectTimeout("connect timeout"),
        ):
            result = fabric.execute("coding", "test prompt", on_model_tried=callback)

        assert result is None

        # cloud_model skipped (budget), local_model attempted and failed (timeout)
        assert len(callback_log) == 2
        assert callback_log[0]["model_id"] == "cloud_model"
        assert callback_log[0]["success"] is False  # Budget skip
        assert callback_log[1]["model_id"] == "local_model"
        assert callback_log[1]["success"] is False  # Timeout
