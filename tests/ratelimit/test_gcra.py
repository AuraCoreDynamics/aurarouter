"""Unit tests for the Generic Cell Rate Algorithm (GCRA) and asymmetric replica resilience.

AuraRouter v2.0 §3.3
"""
from __future__ import annotations

import pytest
from aurarouter.ratelimit.gcra import TatCell
from aurarouter.ratelimit.models import RateState


class TestTatCellBasics:
    def test_single_reservation_advances_tat(self):
        cell = TatCell()
        t0 = 1000.0
        # Quota: 60 requests per 60s -> tau = 1.0s
        admitted = cell.reserve(cost=1, quota=60, t_now=t0)
        assert admitted is True
        assert cell._tat == pytest.approx(1001.0)
        assert cell._inflight_cost == 1

        # Utilization right after 1 request: 1s / 60s ~= 0.0167
        u = cell.get_utilization(t_now=t0)
        assert u == pytest.approx(1.0 / 60.0, rel=1e-3)

    def test_settle_reconciles_delta(self):
        cell = TatCell()
        t0 = 1000.0
        # Reserve 1000 tokens (quota: 60,000 / min -> tau = 0.001s/token)
        cell.reserve(cost=1000, quota=60000, t_now=t0)
        assert cell._inflight_cost == 1000
        assert cell._tat == pytest.approx(t0 + 1.0)

        # Actual usage returned 800 tokens (saved 200 tokens)
        cell.settle(actual_cost=800, reserved_cost=1000, quota=60000, t_now=t0 + 0.1)
        assert cell._inflight_cost == 0
        assert cell._tat == pytest.approx(t0 + 0.8)

    def test_abort_relieves_tat(self):
        cell = TatCell()
        t0 = 1000.0
        cell.reserve(cost=5, quota=60, t_now=t0)
        assert cell._tat == pytest.approx(t0 + 5.0)
        assert cell._inflight_cost == 5

        cell.abort(reserved_cost=5, quota=60, t_now=t0)
        assert cell._tat == pytest.approx(t0)
        assert cell._inflight_cost == 0


class TestAsymmetricReplicaBursts:
    """Tests the fix for the replica division trap identified during spec review."""

    def test_asymmetric_single_node_burst_with_authoritative_headers(self):
        """When 1 replica out of 4 gets hammered, authoritative headers prevent artificial 1/R throttling."""
        cell = TatCell()
        t_now = 1000.0
        quota = 100  # Global quota: 100 req/min
        replica_count = 4

        # Initial request syncs authoritative header showing 90 remaining
        cell.settle(
            actual_cost=1,
            reserved_cost=1,
            quota=quota,
            limit=100,
            remaining=90,
            reset_s=50.0,
            t_now=t_now,
        )
        assert cell.is_authoritative_regime(t_now) is True

        # Hammer this single replica with 60 requests in an asymmetric burst
        # If we naively scaled tau by R (4x), 25 requests would exhaust it!
        # With unscaled global tau in authoritative regime, it admits the burst cleanly.
        admitted_count = 0
        for _ in range(60):
            if cell.reserve(
                cost=1,
                quota=quota,
                replica_count=replica_count,
                t_now=t_now,
            ):
                admitted_count += 1

        assert admitted_count == 60
        # Authoritative settle on responses rebases tat to true upstream remaining plus local in-flight
        # Settle the in-flight requests (e.g. 59 completed, 1 just returned)
        for _ in range(59):
            cell.settle(actual_cost=1, reserved_cost=1, quota=quota, t_now=t_now + 1.0)

        # Final request syncs upstream header showing 40 remaining out of 100 (60% used)
        cell.settle(
            actual_cost=1,
            reserved_cost=1,
            quota=quota,
            limit=100,
            remaining=40,
            reset_s=30.0,
            t_now=t_now + 1.0,
        )
        assert cell._inflight_cost == 0
        # Exact match to upstream 60% utilization right now (NOT 4x over-inflated!)
        u = cell.get_utilization(t_now=t_now + 1.0)
        assert u == pytest.approx(0.60, rel=1e-2)
        assert cell.get_state(t_now=t_now + 1.0) == RateState.NORMAL

        # Once the 30s upstream reset horizon expires, utilization drops cleanly to 0.0
        assert cell.get_utilization(t_now=t_now + 31.5) == 0.0

    def test_remaining_zero_triggers_exact_cooldown_until_reset(self):
        """When a 200 OK header reports remaining=0 and reset_s=12s, cell is EXHAUSTED for 12s."""
        cell = TatCell()
        t_now = 1000.0
        cell.settle(
            actual_cost=1,
            reserved_cost=1,
            quota=60,
            limit=60,
            remaining=0,
            reset_s=12.0,
            t_now=t_now,
        )
        assert cell.get_state(t_now=t_now + 1.0) == RateState.EXHAUSTED
        assert cell.reserve(cost=1, quota=60, t_now=t_now + 1.0) is False
        # After 12s reset elapses, cell recovers to NORMAL
        assert cell.get_state(t_now=t_now + 12.1) == RateState.NORMAL
        assert cell.reserve(cost=1, quota=60, t_now=t_now + 12.1) is True

    def test_no_false_exhaustion_invariant_when_headerless(self):
        """Headerless/cold-start GCRA caps at NEARING_SATURATION and never hard-prunes without confirmation."""
        cell = TatCell()
        t_now = 1000.0
        quota = 20
        replica_count = 4

        # No authoritative headers have been seen (_last_sync_ts == 0)
        assert cell.is_authoritative_regime(t_now) is False

        # Saturate the local cell past 1.0
        for _ in range(25):
            cell.reserve(
                cost=1,
                quota=quota,
                replica_count=replica_count,
                skew_damping=0.5,
                t_now=t_now,
            )

        u = cell.get_utilization(t_now=t_now)
        assert u >= 1.0

        # State MUST be NEARING_SATURATION to prevent false negative routing pruning
        state = cell.get_state(t_now=t_now)
        assert state == RateState.NEARING_SATURATION


class TestCooldownAndDailyExhaustion:
    def test_settle_error_triggers_cooldown(self):
        cell = TatCell()
        t0 = 1000.0
        cell.reserve(cost=1, quota=60, t_now=t0)

        # 429 received with Retry-After: 15s
        cell.settle_error(reserved_cost=1, retry_after_s=15.0, is_daily=False, t_now=t0 + 0.1)

        assert cell.is_in_cooldown(t_now=t0 + 5.0) is True
        assert cell.seconds_until_reset(t_now=t0 + 5.0) == pytest.approx(10.1, rel=1e-2)
        assert cell.get_state(t_now=t0 + 5.0) == RateState.EXHAUSTED

        # Cannot reserve while in cooldown
        assert cell.reserve(cost=1, quota=60, t_now=t0 + 5.0) is False

        # Cooldown expires
        assert cell.is_in_cooldown(t_now=t0 + 16.0) is False

    def test_daily_exhausted_flag(self):
        cell = TatCell()
        t0 = 1000.0
        cell.settle_error(reserved_cost=1, retry_after_s=3600.0, is_daily=True, t_now=t0)
        assert cell.get_state(t_now=t0 + 10.0) == RateState.DAILY_EXHAUSTED
