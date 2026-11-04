"""Phase 4 tests: Two-Stage RoutingPolicyStack & Lock-Free PosteriorSnapshotCache."""

from __future__ import annotations

import statistics
import time

import pytest

from aurarouter.circuit_breaker import CircuitBreakerRegistry
from aurarouter.config import ConfigLoader
from aurarouter.ratelimit.models import RateLimitQuota, RateLimitSnapshot
from aurarouter.ratelimit.tracker import RateLimitTracker
from aurarouter.routing_stack import (
    CandidateEvalContext,
    PosteriorSnapshotCache,
    RoutingPolicyStack,
)
from aurarouter.savings.feedback_store import FeedbackStore


def _make_config() -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "tiny_1b": {
                "provider": "ollama",
                "model_name": "qwen2.5:1.5b",
                "hosting_tier": "on-prem",
                "max_complexity": 4,
                "allowed_data_categories": ["pii", "internal"],
            },
            "local_8b": {
                "provider": "llamacpp-server",
                "model_name": "llama-3.1-8b",
                "hosting_tier": "on-prem",
                "max_complexity": 10,
                "allowed_data_categories": ["pii", "internal"],
            },
            "claude_sonnet": {
                "provider": "anthropic",
                "model_name": "claude-sonnet-4-20250514",
                "hosting_tier": "cloud",
                "cost_per_1m_input": 3.0,
                "cost_per_1m_output": 15.0,
                "allowed_data_categories": [],
            },
            "claude_sonnet_mirror": {
                "provider": "anthropic",
                "model_name": "claude-sonnet-4-20250514",
                "hosting_tier": "cloud",
                "cost_per_1m_input": 3.0,
                "cost_per_1m_output": 15.0,
                "allowed_data_categories": [],
            },
            "claude_opus": {
                "provider": "anthropic",
                "model_name": "claude-opus-4-20250514",
                "hosting_tier": "cloud",
                "cost_per_1m_input": 15.0,
                "cost_per_1m_output": 75.0,
                "allowed_data_categories": [],
            },
        }
    }
    return cfg


class TestStage1NonCompensatoryFilters:
    def test_tiny_model_pruned_on_high_complexity_despite_zero_cost(self):
        cfg = _make_config()
        cache = PosteriorSnapshotCache()
        tracker = RateLimitTracker()
        # Slightly saturate claude_opus (85% RPM used) so its headroom < 1.0
        tracker.configure_quota("anthropic", RateLimitQuota(rpm=100), model_id="claude_opus")
        lease = tracker.reserve("claude_opus", "anthropic")
        tracker.settle(
            lease,
            snapshot=RateLimitSnapshot(rpm_limit=100, rpm_remaining=15, rpm_reset_s=30.0),
        )

        stack = RoutingPolicyStack(
            config=cfg,
            posterior_cache=cache,
            rate_tracker=tracker,
            stack_config={"mode": "active"},
        )
        trace = stack.evaluate_and_rank(
            ["tiny_1b", "claude_opus"],
            CandidateEvalContext(role="coding", complexity_score=9, deterministic=True),
        )

        # tiny_1b MUST be hard-pruned by CapabilityFilter, never winning via $0 cost!
        assert "tiny_1b" in trace.pruned
        assert "CapabilityFilter" in trace.pruned["tiny_1b"]
        assert trace.ranked_chain == ["claude_opus"]

    def test_sovereignty_filter_prunes_unauthorized_cloud_models(self):
        cfg = _make_config()
        stack = RoutingPolicyStack(
            config=cfg,
            posterior_cache=PosteriorSnapshotCache(),
            stack_config={"mode": "active"},
        )
        trace = stack.evaluate_and_rank(
            ["claude_opus", "local_8b"],
            CandidateEvalContext(
                role="coding",
                complexity_score=6,
                effective_data_categories=frozenset({"pii"}),
                sovereignty_verdict_open=False,
                deterministic=True,
            ),
        )
        assert "claude_opus" in trace.pruned
        assert "SovereigntyFilter" in trace.pruned["claude_opus"]
        assert trace.ranked_chain == ["local_8b"]

    def test_rt1_last_resort_probe_when_all_pruned_by_availability_only(self):
        cfg = _make_config()
        tracker = RateLimitTracker()
        # claude_opus hit daily quota (3600s), claude_sonnet hit short RPM 429 (6s)
        l1 = tracker.reserve("claude_opus", "anthropic")
        tracker.settle_rate_limited(
            l1, RateLimitSnapshot(retry_after_s=3600.0, is_daily_quota=True)
        )
        l2 = tracker.reserve("claude_sonnet", "anthropic")
        tracker.settle_rate_limited(
            l2, RateLimitSnapshot(retry_after_s=6.0, is_daily_quota=False)
        )

        stack = RoutingPolicyStack(
            config=cfg,
            posterior_cache=PosteriorSnapshotCache(),
            rate_tracker=tracker,
            stack_config={"mode": "active"},
        )
        trace = stack.evaluate_and_rank(
            ["claude_opus", "claude_sonnet"],
            CandidateEvalContext(role="coding", complexity_score=7, deterministic=True),
        )
        # RT1 preserves the single candidate with shortest recovery (claude_sonnet, NOT daily-exhausted opus)
        assert trace.last_resort_probe == "claude_sonnet"
        assert trace.ranked_chain == ["claude_sonnet"]

    def test_rt1_never_resurrects_sovereignty_or_capability_pruned_models(self):
        cfg = _make_config()
        stack = RoutingPolicyStack(
            config=cfg,
            posterior_cache=PosteriorSnapshotCache(),
            stack_config={"mode": "active"},
        )
        trace = stack.evaluate_and_rank(
            ["claude_opus"],
            CandidateEvalContext(
                role="coding",
                complexity_score=7,
                effective_data_categories=frozenset({"pii"}),
                sovereignty_verdict_open=False,
                deterministic=True,
            ),
        )
        assert trace.ranked_chain == []
        assert trace.last_resort_probe is None


class TestStage2UtilityAndCacheAffinity:
    def test_elastic_cost_prefers_local_on_simple_and_cloud_on_complex(self):
        cfg = _make_config()
        stack = RoutingPolicyStack(
            config=cfg,
            posterior_cache=PosteriorSnapshotCache(),
            stack_config={"mode": "active"},
        )

        # Complexity 1: free local_8b beats expensive claude_opus
        t_low = stack.evaluate_and_rank(
            ["claude_opus", "local_8b"],
            CandidateEvalContext(
                role="coding",
                complexity_score=1,
                estimated_input_tokens=2000,
                estimated_output_tokens=500,
                deterministic=True,
            ),
        )
        assert t_low.ranked_chain[0] == "local_8b"

        # Complexity 9: frontier claude_opus beats local_8b
        t_high = stack.evaluate_and_rank(
            ["local_8b", "claude_opus"],
            CandidateEvalContext(
                role="coding",
                complexity_score=9,
                estimated_input_tokens=2000,
                estimated_output_tokens=500,
                deterministic=True,
            ),
        )
        assert t_high.ranked_chain[0] == "claude_opus"

    def test_cache_affinity_stickiness_within_ttl_and_expiry_after_ttl(self):
        cfg = _make_config()
        stack = RoutingPolicyStack(
            config=cfg,
            posterior_cache=PosteriorSnapshotCache(),
            stack_config={"mode": "active"},
        )
        now = time.time()

        # Warm KV cache on claude_sonnet_mirror (20s ago < 300s Anthropic TTL)
        t_warm = stack.evaluate_and_rank(
            ["claude_sonnet", "claude_sonnet_mirror"],
            CandidateEvalContext(
                role="coding",
                complexity_score=6,
                last_model_id="claude_sonnet_mirror",
                last_routed_at=now - 20.0,
                deterministic=True,
            ),
        )
        assert t_warm.ranked_chain[0] == "claude_sonnet_mirror"
        assert t_warm.scores[0].cache_mult > 1.10

        # Expired KV cache (400s ago >= 300s TTL) -> deterministic YAML chain order wins
        t_expired = stack.evaluate_and_rank(
            ["claude_sonnet", "claude_sonnet_mirror"],
            CandidateEvalContext(
                role="coding",
                complexity_score=6,
                last_model_id="claude_sonnet_mirror",
                last_routed_at=now - 400.0,
                deterministic=True,
            ),
        )
        assert t_expired.ranked_chain[0] == "claude_sonnet"
        assert t_expired.scores[0].cache_mult == 1.0


class TestLockFreePosteriorSnapshotCacheLatency:
    def test_zero_sqlite_reads_and_zero_locks_on_hot_path(self, tmp_path):
        db_file = tmp_path / "feedback.db"
        store = FeedbackStore(db_path=db_file)
        for _ in range(25):
            store.record("coding", 8.0, "model_3", True, 0.4, 500, 200)
        for _ in range(20):
            store.record("coding", 8.0, "model_0", False, 1.2, 500, 100)

        cache = PosteriorSnapshotCache()
        rows_loaded = cache.hydrate_from_store(store)
        assert rows_loaded >= 2
        store.close()

        # Build a 10-model chain
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                f"model_{i}": {
                    "provider": "anthropic" if i % 2 == 0 else "ollama",
                    "model_name": f"m-{i}",
                    "hosting_tier": "cloud" if i % 2 == 0 else "on-prem",
                    "cost_per_1m_input": float(i),
                    "cost_per_1m_output": float(i * 3),
                }
                for i in range(10)
            }
        }
        chain = [f"model_{i}" for i in range(10)]

        cb_reg = CircuitBreakerRegistry()
        for p in ("anthropic", "ollama"):
            cb_reg.get_or_create(p)
        tracker = RateLimitTracker()

        stack = RoutingPolicyStack(
            config=cfg,
            posterior_cache=cache,
            rate_tracker=tracker,
            circuit_breakers=cb_reg,
            stack_config={"mode": "active"},
        )

        # Poison SQLite connection AND PosteriorSnapshotCache._write_lock to prove
        # evaluate_and_rank() touches NEITHER SQLite NOR locks on the hot path!
        class _ForbiddenLock:
            def __enter__(self):
                raise AssertionError("Lock acquired on hot routing path!")

            def __exit__(self, *args):
                pass

        store._connect = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
            AssertionError("SQLite queried on hot routing path!")
        )
        cache._write_lock = _ForbiddenLock()  # type: ignore[assignment]

        ctx = CandidateEvalContext(
            role="coding",
            complexity_score=8,
            estimated_input_tokens=1000,
            estimated_output_tokens=400,
        )

        # Warmup
        for _ in range(20):
            stack.evaluate_and_rank(chain, ctx)

        latencies_us: list[float] = []
        model_3_wins = 0
        for _ in range(300):
            t0 = time.perf_counter()
            trace = stack.evaluate_and_rank(chain, ctx)
            latencies_us.append((time.perf_counter() - t0) * 1_000_000.0)
            if trace.ranked_chain[0] == "model_3":
                model_3_wins += 1

        median_us = statistics.median(latencies_us)
        # 10-model full Thompson Sampling + Stage 1 + Stage 2 in pure Python is sub-millisecond (< 1200us ceiling)
        assert median_us < 1200.0, f"Median 10-model ranking took {median_us:.1f}us"
        # model_3 (25 successes) strongly dominates model_0 (20 failures)
        assert model_3_wins > 150

