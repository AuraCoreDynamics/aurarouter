"""Tests for BudgetManager — budget enforcement logic."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from aurarouter.savings.budget import BudgetManager, BudgetStatus, _CACHE_TTL_SECONDS
from aurarouter.savings.models import UsageRecord
from aurarouter.savings.pricing import CostEngine, PricingCatalog
from aurarouter.savings.usage_store import UsageStore


def _make_budget_manager(
    tmp_path,
    *,
    enabled=True,
    daily_limit=None,
    monthly_limit=None,
    records=None,
):
    """Build a BudgetManager backed by a real UsageStore with optional seed records."""
    store = UsageStore(db_path=tmp_path / "usage.db")
    catalog = PricingCatalog()
    engine = CostEngine(catalog, store)

    for rec in records or []:
        store.record(rec)

    config = {
        "enabled": enabled,
        "daily_limit": daily_limit,
        "monthly_limit": monthly_limit,
    }
    return BudgetManager(engine, config)


def _cloud_record(input_tokens: int, output_tokens: int) -> UsageRecord:
    """Create a usage record for a cloud provider timestamped now (UTC)."""
    return UsageRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id="gemini-2.0-flash",
        provider="google",
        role="coding",
        intent="",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_s=0.5,
        success=True,
        is_cloud=True,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_budget_disabled():
    """Budget not enabled — check_budget() always returns allowed=True."""
    engine = MagicMock(spec=CostEngine)
    mgr = BudgetManager(engine, {"enabled": False, "daily_limit": 0.01})

    status = mgr.check_budget("google")
    assert status.allowed is True
    # CostEngine should never be queried when disabled
    engine.total_spend.assert_not_called()


def test_daily_limit_not_exceeded(tmp_path):
    """Spend $5, daily limit $10 → allowed."""
    # gemini-2.0-flash: $0.10/M input, $0.40/M output
    # 50M input tokens → $5.00
    mgr = _make_budget_manager(
        tmp_path,
        daily_limit=10.00,
        records=[_cloud_record(50_000_000, 0)],
    )
    status = mgr.check_budget("google")
    assert status.allowed is True
    assert status.daily_spend < 10.00


def test_daily_limit_exceeded(tmp_path):
    """Spend > $10, daily limit $10 → not allowed."""
    # 110M input tokens at $0.10/M = $11.00
    mgr = _make_budget_manager(
        tmp_path,
        daily_limit=10.00,
        records=[_cloud_record(110_000_000, 0)],
    )
    status = mgr.check_budget("google")
    assert status.allowed is False
    assert "Daily budget exceeded" in status.reason


def test_monthly_limit_exceeded(tmp_path):
    """Spend > $100, monthly limit $100 → not allowed."""
    # 1.01B input tokens at $0.10/M = $101.00
    mgr = _make_budget_manager(
        tmp_path,
        monthly_limit=100.00,
        records=[_cloud_record(1_010_000_000, 0)],
    )
    status = mgr.check_budget("google")
    assert status.allowed is False
    assert "Monthly budget exceeded" in status.reason


def test_local_always_allowed(tmp_path):
    """Budget exceeded but provider is Ollama → still allowed."""
    mgr = _make_budget_manager(
        tmp_path,
        daily_limit=1.00,
        records=[_cloud_record(110_000_000, 0)],  # $11 spend
    )
    status = mgr.check_budget("ollama")
    assert status.allowed is True


def test_spend_cache_ttl(tmp_path):
    """Verify cache doesn't re-query within TTL."""
    store = UsageStore(db_path=tmp_path / "usage.db")
    catalog = PricingCatalog()
    engine = CostEngine(catalog, store)

    config = {"enabled": True, "daily_limit": 100.00, "monthly_limit": None}
    mgr = BudgetManager(engine, config)

    with patch.object(engine, "total_spend", return_value=1.0) as mock_spend:
        # First call populates cache
        spend1 = mgr.get_daily_spend()
        assert spend1 == 1.0
        call_count_after_first = mock_spend.call_count

        # Second call should use cache
        spend2 = mgr.get_daily_spend()
        assert spend2 == 1.0
        assert mock_spend.call_count == call_count_after_first


def test_budget_config_update():
    """update_config() changes limits without restart."""
    engine = MagicMock(spec=CostEngine)
    engine.total_spend.return_value = 5.0

    mgr = BudgetManager(engine, {"enabled": True, "daily_limit": 10.00})

    # Under limit
    status = mgr.check_budget("google")
    assert status.allowed is True

    # Lower the limit below current spend
    mgr.update_config({"enabled": True, "daily_limit": 3.00})
    status = mgr.check_budget("google")
    assert status.allowed is False
    assert "Daily budget exceeded" in status.reason
