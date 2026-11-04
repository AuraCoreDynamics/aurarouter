"""Unit and concurrency tests for RateLimitTracker, token leases, and headroom scoring.

AuraRouter v2.0 §3.1, §3.3
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import time
import pytest

from aurarouter.ratelimit.models import RateLimitQuota, RateLimitSnapshot, RateState
from aurarouter.ratelimit.tracker import RateLimitTracker
from aurarouter.errors import (
    RateLimitExceededError,
    ContextLengthExceededError,
    ProviderTransientError,
)


class TestRateLimitTrackerBasics:
    def test_configure_and_reserve_lease(self):
        tracker = RateLimitTracker()
        tracker.configure_quota(
            provider="openai",
            quota=RateLimitQuota(rpm=600, itpm=120000, warning_threshold=0.80),
        )

        lease = tracker.reserve(
            model_id="gpt-4o",
            provider="openai",
            est_input_tokens=1500,
            est_output_tokens=500,
        )
        assert lease.model_id == "gpt-4o"
        assert lease.reserved_requests == 1
        assert lease.reserved_input_tokens == 1500
        assert lease.settled is False

        # Headroom should remain near 1.0 on a single small request
        score = tracker.get_headroom_score("gpt-4o", "openai")
        assert score == pytest.approx(1.0, rel=1e-2)

        # Settle
        tracker.settle(lease, actual_input_tokens=1200, actual_output_tokens=400)
        assert lease.settled is True

    def test_headroom_score_quadratic_decay(self):
        tracker = RateLimitTracker(default_warning_threshold=0.80)
        tracker.configure_quota(
            provider="anthropic",
            quota=RateLimitQuota(rpm=100, warning_threshold=0.80),
        )

        # Before warning threshold -> 1.0
        assert tracker._compute_headroom_score(0.50, RateState.NORMAL, 0.80) == 1.0
        assert tracker._compute_headroom_score(0.79, RateState.NORMAL, 0.80) == 1.0

        # At warning threshold -> 1.0
        assert tracker._compute_headroom_score(0.80, RateState.NEARING_SATURATION, 0.80) == 1.0

        # Halfway between 0.80 and 1.00 (u = 0.90)
        score_90 = tracker._compute_headroom_score(0.90, RateState.NEARING_SATURATION, 0.80)
        # norm = (0.90 - 0.80)/0.20 = 0.5. 1.0 - 0.85*(0.5^2) = 1.0 - 0.2125 = 0.7875
        assert score_90 == pytest.approx(0.7875, rel=1e-3)

        # At 1.0 utilization -> 0.15 (drops smoothly before hard exhaustion)
        score_100 = tracker._compute_headroom_score(1.0, RateState.NEARING_SATURATION, 0.80)
        assert score_100 == pytest.approx(0.15, rel=1e-2)

        # When hard exhausted -> 0.0
        assert tracker._compute_headroom_score(1.0, RateState.EXHAUSTED, 0.80) == 0.0


class TestVegasTTFTEstimator:
    def test_vegas_ttft_detects_queueing_latency(self):
        tracker = RateLimitTracker(enable_vegas=True)
        model = "vllm-local"
        provider = "openapi"

        # Prime baseline with fast responses (0.05s)
        for _ in range(10):
            tracker._get_or_create_limiter(model, provider).record_ttft(0.05)

        u = tracker.get_utilization(model, provider)
        assert u.composite_utilization == 0.0

        # Sudden queueing spike: TTFT jumps to 0.25s (5x increase)
        for _ in range(5):
            tracker._get_or_create_limiter(model, provider).record_ttft(0.25)

        u_spiked = tracker.get_utilization(model, provider)
        # Vegas should synthesize backpressure
        assert u_spiked.composite_utilization >= 0.80
        assert u_spiked.state == RateState.NEARING_SATURATION


class TestLocalSlotTelemetry:
    def test_slot_telemetry_affects_composite_utilization(self):
        tracker = RateLimitTracker()
        tracker.record_slot_telemetry(
            model_id="llama-8b",
            provider="llamacpp-server",
            slot_utilization=0.88,
            kv_cache_utilization=0.92,
        )

        util = tracker.get_utilization("llama-8b", "llamacpp-server")
        assert util.composite_utilization == 0.92
        assert util.state == RateState.NEARING_SATURATION
        assert util.headroom_score < 1.0


class TestConcurrencyLeaseStampede:
    """Verifies that 50 concurrent async tasks reserving and settling leases do not race or drop state."""

    @pytest.mark.asyncio
    async def test_50_concurrent_leases(self):
        tracker = RateLimitTracker()
        tracker.configure_quota(
            provider="anthropic",
            quota=RateLimitQuota(rpm=500, itpm=200000, warning_threshold=0.80),
        )

        async def worker(worker_id: int):
            lease = tracker.reserve(
                model_id="claude-sonnet-4",
                provider="anthropic",
                est_input_tokens=100,
                est_output_tokens=50,
            )
            assert lease is not None
            assert lease.settled is False
            await asyncio.sleep(0.01)
            tracker.settle(
                lease,
                actual_input_tokens=90,
                actual_output_tokens=45,
            )
            assert lease.settled is True

        tasks = [asyncio.create_task(worker(i)) for i in range(50)]
        await asyncio.gather(*tasks)

        # After all settle, in-flight leases should be empty
        assert len(tracker._leases) == 0
        limiter = tracker._get_or_create_limiter("claude-sonnet-4", "anthropic")
        assert limiter.rpm_cell._inflight_cost == 0
        assert limiter.itpm_cell._inflight_cost == 0


class TestTypedExceptions:
    def test_typed_exceptions(self):
        snap = RateLimitSnapshot(rpm_remaining=0, retry_after_s=12.0)
        ex = RateLimitExceededError("Rate limit exceeded", snapshot=snap)
        assert ex.snapshot.retry_after_s == 12.0

        ctx_ex = ContextLengthExceededError("Context window overflow", max_tokens=8192)
        assert ctx_ex.max_tokens == 8192

        trans_ex = ProviderTransientError("Service Unavailable", status_code=503)
        assert trans_ex.status_code == 503

