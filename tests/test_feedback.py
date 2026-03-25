"""Tests for the FeedbackStore, adaptive triage weights, and ComputeFabric integration."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.feedback_store import FeedbackStore
from aurarouter.savings.models import GenerateResult
from aurarouter.savings.triage import TriageRouter, TriageRule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_feedback.db"


@pytest.fixture
def store(tmp_db: Path) -> FeedbackStore:
    s = FeedbackStore(db_path=tmp_db)
    yield s
    s.close()


def _make_fabric(models: dict, roles: dict, **kwargs) -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": models, "roles": roles}
    return ComputeFabric(cfg, **kwargs)


# ---------------------------------------------------------------------------
# T7.1 — FeedbackStore CRUD
# ---------------------------------------------------------------------------

class TestFeedbackStoreRecord:
    """Test basic record insertion."""

    def test_record_single(self, store: FeedbackStore):
        store.record(
            role="coding", complexity=3.0, model_id="m1",
            success=True, latency=0.5,
        )
        stats = store.model_stats(window_days=30)
        assert len(stats) == 1
        assert stats[0]["model_id"] == "m1"
        assert stats[0]["call_count"] == 1
        assert stats[0]["success_rate"] == 1.0

    def test_record_multiple(self, store: FeedbackStore):
        store.record(role="coding", complexity=3.0, model_id="m1", success=True, latency=0.5)
        store.record(role="coding", complexity=5.0, model_id="m1", success=False, latency=1.0)
        store.record(role="reasoning", complexity=7.0, model_id="m2", success=True, latency=0.3)

        stats = store.model_stats(window_days=30)
        assert len(stats) == 2

    def test_record_with_tokens(self, store: FeedbackStore):
        store.record(
            role="coding", complexity=3.0, model_id="m1",
            success=True, latency=0.5,
            input_tokens=100, output_tokens=200,
        )
        stats = store.model_stats(window_days=30)
        assert stats[0]["call_count"] == 1


class TestFeedbackStoreSuccessRate:
    """Test success_rate queries."""

    def test_success_rate_basic(self, store: FeedbackStore):
        store.record(role="coding", complexity=3.0, model_id="m1", success=True, latency=0.5)
        store.record(role="coding", complexity=3.0, model_id="m1", success=True, latency=0.5)
        store.record(role="coding", complexity=3.0, model_id="m1", success=False, latency=0.5)

        rate = store.success_rate("m1", window_days=30)
        assert abs(rate - 2.0 / 3.0) < 0.01

    def test_success_rate_no_data(self, store: FeedbackStore):
        rate = store.success_rate("nonexistent", window_days=30)
        assert rate == 0.0

    def test_success_rate_complexity_band_filtering(self, store: FeedbackStore):
        """Records outside the complexity band should be excluded."""
        store.record(role="coding", complexity=2.0, model_id="m1", success=True, latency=0.1)
        store.record(role="coding", complexity=5.0, model_id="m1", success=False, latency=0.1)
        store.record(role="coding", complexity=8.0, model_id="m1", success=True, latency=0.1)

        # Only complexity 2.0 is in [0, 3]
        rate_low = store.success_rate("m1", complexity_min=0, complexity_max=3, window_days=30)
        assert rate_low == 1.0

        # Only complexity 5.0 is in [4, 6]
        rate_mid = store.success_rate("m1", complexity_min=4, complexity_max=6, window_days=30)
        assert rate_mid == 0.0

        # Only complexity 8.0 is in [7, 10]
        rate_high = store.success_rate("m1", complexity_min=7, complexity_max=10, window_days=30)
        assert rate_high == 1.0

    def test_success_rate_time_window_filtering(self, store: FeedbackStore):
        """Records outside the time window should be excluded."""
        # Insert a record, then manually backdate it in the DB
        store.record(role="coding", complexity=3.0, model_id="m1", success=True, latency=0.1)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        with store._lock:
            conn = store._connect()
            conn.execute("UPDATE feedback SET timestamp = ?", (old_timestamp,))
            conn.commit()

        # With 7-day window, the old record should be excluded
        rate = store.success_rate("m1", window_days=7)
        assert rate == 0.0

        # With 60-day window, it should be included
        rate = store.success_rate("m1", window_days=60)
        assert rate == 1.0


class TestFeedbackStoreModelStats:
    """Test model_stats aggregate queries."""

    def test_model_stats_empty(self, store: FeedbackStore):
        stats = store.model_stats()
        assert stats == []

    def test_model_stats_aggregation(self, store: FeedbackStore):
        store.record(role="coding", complexity=3.0, model_id="m1", success=True, latency=0.1)
        store.record(role="coding", complexity=5.0, model_id="m1", success=False, latency=0.3)
        store.record(role="reasoning", complexity=7.0, model_id="m2", success=True, latency=0.2)

        stats = store.model_stats(window_days=30)
        stats_by_model = {s["model_id"]: s for s in stats}

        assert stats_by_model["m1"]["call_count"] == 2
        assert stats_by_model["m1"]["success_rate"] == 0.5
        assert stats_by_model["m1"]["avg_latency_ms"] > 0

        assert stats_by_model["m2"]["call_count"] == 1
        assert stats_by_model["m2"]["success_rate"] == 1.0


# ---------------------------------------------------------------------------
# T7.3 — Adaptive Triage Weights
# ---------------------------------------------------------------------------

class TestAdaptiveTriageWeights:
    """Test TriageRouter.update_from_feedback()."""

    def test_thresholds_shift_toward_observed_data(self, store: FeedbackStore):
        """High success rate should widen the band (increase max_complexity)."""
        router = TriageRouter(
            rules=[
                TriageRule(max_complexity=3, preferred_role="coding_lite"),
                TriageRule(max_complexity=7, preferred_role="coding"),
            ],
            default_role="reasoning",
        )
        original_threshold = router.rules[0].max_complexity

        # Record many successes in the low-complexity band
        for _ in range(20):
            store.record(role="coding_lite", complexity=2.0, model_id="m1",
                         success=True, latency=0.1)

        router.update_from_feedback(store, blend_factor=0.3)

        # With 100% success rate, the shift should be positive
        # (rate=1.0, shift=(1.0-0.65)*5=1.75, observed=3+1.75=4.75)
        # EMA: 0.7*3 + 0.3*4.75 = 2.1 + 1.425 = 3.525 → rounds to 4
        assert router.rules[0].max_complexity >= original_threshold

    def test_blend_factor_zero_produces_no_change(self, store: FeedbackStore):
        """blend_factor=0.0 should leave thresholds unchanged."""
        router = TriageRouter(
            rules=[
                TriageRule(max_complexity=3, preferred_role="coding_lite"),
                TriageRule(max_complexity=7, preferred_role="coding"),
            ],
            default_role="reasoning",
        )
        original_thresholds = [r.max_complexity for r in router.rules]

        for _ in range(20):
            store.record(role="coding_lite", complexity=2.0, model_id="m1",
                         success=True, latency=0.1)

        router.update_from_feedback(store, blend_factor=0.0)

        new_thresholds = [r.max_complexity for r in router.rules]
        assert new_thresholds == original_thresholds

    def test_low_success_rate_narrows_band(self, store: FeedbackStore):
        """Low success rate should narrow the band (decrease max_complexity)."""
        router = TriageRouter(
            rules=[
                TriageRule(max_complexity=5, preferred_role="coding_lite"),
            ],
            default_role="reasoning",
        )
        original_threshold = router.rules[0].max_complexity

        # Record many failures
        for _ in range(20):
            store.record(role="coding_lite", complexity=3.0, model_id="m1",
                         success=False, latency=0.5)

        router.update_from_feedback(store, blend_factor=0.5)

        # With 0% success rate, shift=(0-0.65)*5=-3.25, observed=5-3.25=1.75
        # EMA: 0.5*5 + 0.5*1.75 = 2.5 + 0.875 = 3.375 → rounds to 3
        assert router.rules[0].max_complexity <= original_threshold

    def test_no_feedback_data_no_change(self, store: FeedbackStore):
        """Empty feedback store should not change thresholds."""
        router = TriageRouter(
            rules=[TriageRule(max_complexity=5, preferred_role="coding")],
            default_role="reasoning",
        )
        original = router.rules[0].max_complexity
        router.update_from_feedback(store, blend_factor=0.5)
        assert router.rules[0].max_complexity == original


# ---------------------------------------------------------------------------
# T7.4 — Config Accessors
# ---------------------------------------------------------------------------

class TestFeedbackConfig:
    """Test ConfigLoader feedback config accessors."""

    def test_feedback_disabled_by_default(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        assert cfg.is_feedback_enabled() is False

    def test_feedback_enabled(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"savings": {"feedback": {"enabled": True}}}
        assert cfg.is_feedback_enabled() is True

    def test_get_feedback_config(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"savings": {"feedback": {"enabled": True, "db_path": "/tmp/fb.db"}}}
        fc = cfg.get_feedback_config()
        assert fc["enabled"] is True
        assert fc["db_path"] == "/tmp/fb.db"

    def test_get_feedback_config_empty(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        assert cfg.get_feedback_config() == {}


# ---------------------------------------------------------------------------
# T7.2 — ComputeFabric Integration
# ---------------------------------------------------------------------------

class TestFabricFeedbackIntegration:
    """Test that ComputeFabric calls feedback_store.record() on model attempts."""

    def test_feedback_recorded_on_success(self):
        """Feedback store record() should be called on successful execution."""
        mock_store = MagicMock()
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
            roles={"coding": ["m1"]},
            feedback_store=mock_store,
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            return_value="hello world",
        ):
            fabric.execute("coding", "test prompt")

        # Give the background thread time to execute
        time.sleep(0.2)
        mock_store.record.assert_called_once()
        call_kwargs = mock_store.record.call_args[1]
        assert call_kwargs["role"] == "coding"
        assert call_kwargs["model_id"] == "m1"
        assert call_kwargs["success"] is True
        assert call_kwargs["complexity"] == 5.0  # neutral default

    def test_feedback_recorded_on_failure(self):
        """Feedback store record() should be called on failed execution."""
        mock_store = MagicMock()
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
            roles={"coding": ["m1"]},
            feedback_store=mock_store,
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            side_effect=Exception("connection refused"),
        ):
            fabric.execute("coding", "test prompt")

        time.sleep(0.2)
        mock_store.record.assert_called_once()
        call_kwargs = mock_store.record.call_args[1]
        assert call_kwargs["success"] is False

    def test_no_feedback_without_store(self):
        """Without a feedback store, execute should still work fine."""
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
            roles={"coding": ["m1"]},
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            return_value="result",
        ):
            result = fabric.execute("coding", "test prompt")

        assert result is not None

    def test_feedback_store_exception_does_not_break_execute(self):
        """A failing feedback store should not break execute()."""
        mock_store = MagicMock()
        mock_store.record.side_effect = Exception("db locked")
        fabric = _make_fabric(
            models={"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
            roles={"coding": ["m1"]},
            feedback_store=mock_store,
        )

        with patch(
            "aurarouter.providers.ollama.OllamaProvider.generate",
            return_value="result",
        ):
            result = fabric.execute("coding", "test prompt")

        # execute() should succeed even if feedback recording fails
        assert result is not None


# ---------------------------------------------------------------------------
# T7.5 — Additional edge-case tests
# ---------------------------------------------------------------------------

class TestFeedbackStoreClose:
    """Test close/reopen behavior."""

    def test_close_and_reopen(self, tmp_db: Path):
        store = FeedbackStore(db_path=tmp_db)
        store.record(role="coding", complexity=3.0, model_id="m1", success=True, latency=0.1)
        store.close()

        # Re-open and verify data persisted
        store2 = FeedbackStore(db_path=tmp_db)
        stats = store2.model_stats(window_days=30)
        assert len(stats) == 1
        assert stats[0]["model_id"] == "m1"
        store2.close()
