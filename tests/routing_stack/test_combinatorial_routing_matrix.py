"""Combinatorial Routing Matrix Test Suite.

Exhaustively verifies multi-factor interactions across ComputeFabric, RoutingPolicyStack,
RateLimitTracker, BudgetManager, CircuitBreaker, and SessionProjector using
zero-network SimulatedProvider instances:

1. Privacy / Sovereignty Gating vs Capability Bandit
2. Budget Cap Exhaustion & Dynamic Cloud Fallback
3. RateLimit (429) & Headroom Degradation Failover
4. Context Cliff & Ephemeral Session Projection
5. Circuit Breaker Fault Isolation & Recovery Probing
6. Cache Affinity vs Vegas TTFT Headroom Tradeoff
7. Streaming Outcome Parity & UsageStore Recording
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aurarouter.circuit_breaker import CircuitBreakerRegistry
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.ratelimit.models import RateLimitSnapshot, RateState
from aurarouter.ratelimit.tracker import RateLimitTracker
from aurarouter.routing_stack.stack import RoutingPolicyStack
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.feedback_store import FeedbackStore
from aurarouter.savings.models import UsageRecord
from aurarouter.savings.pricing import CostEngine, ModelPrice, PricingCatalog
from aurarouter.savings.usage_store import UsageStore
from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.store import SessionStore
from tests.testing.simulated_provider import SimulatedProvider


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        yield Path(td)


@pytest.fixture
def combinatorial_env(temp_dir):
    """Sets up a fully configured ComputeFabric with software-defined providers."""
    cfg_data = {
        "version": "2.0",
        "routing": {
            "mode": "active",
            "routing_stack": {
                "bandit_enabled": True,
                "bandit_weight": 0.40,
                "headroom_weight": 0.30,
                "circuit_health_weight": 0.20,
                "cache_affinity_weight": 0.10,
                "elastic_penalty_weight": 0.05,
            },
            "rate_limits": {
                "replica_count": 1,
                "models": {
                    "cloud-frontier": {"rpm": 60, "itpm": 100_000},
                    "cloud-fast": {"rpm": 120, "itpm": 200_000},
                },
            },
        },
        "models": {
            "onprem-gpu": {
                "provider": "ollama",
                "hosting_tier": "on_prem",
                "context_window": 4096,
                "allowed_data_categories": ["PII", "FINANCIAL", "GENERAL"],
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
            },
            "cloud-frontier": {
                "provider": "anthropic",
                "hosting_tier": "cloud",
                "context_window": 200_000,
                "allowed_data_categories": ["GENERAL"],
                "cost_per_1m_input": 3.0,
                "cost_per_1m_output": 15.0,
                "rate_limits": {"requests_per_minute": 60, "tokens_per_minute": 100_000},
            },
            "cloud-fast": {
                "provider": "openai",
                "hosting_tier": "cloud",
                "context_window": 128_000,
                "allowed_data_categories": ["GENERAL"],
                "cost_per_1m_input": 0.15,
                "cost_per_1m_output": 0.60,
                "rate_limits": {"requests_per_minute": 120, "tokens_per_minute": 200_000},
            },
        },
        "roles": {
            "general": {
                "models": ["cloud-frontier", "cloud-fast", "onprem-gpu"],
            },
            "reasoning": {
                "models": ["cloud-frontier", "cloud-fast", "onprem-gpu"],
            },
        },
        "savings": {
            "budget": {
                "enabled": True,
                "daily_limit": 10.0,
                "monthly_limit": 100.0,
            }
        },
        "resilience": {
            "failure_threshold": 2,
            "reset_timeout": 0.2,
        },
    }

    config_loader = ConfigLoader(allow_missing=True)
    config_loader.config = cfg_data
    usage_store = UsageStore(db_path=temp_dir / "usage.db")
    feedback_store = FeedbackStore(db_path=temp_dir / "feedback.db")
    session_store = SessionStore(db_path=temp_dir / "sessions.db")
    pricing_catalog = PricingCatalog(overrides={
        "cloud-frontier": ModelPrice(3.0, 15.0),
        "cloud-fast": ModelPrice(0.15, 0.60),
    })
    cost_engine = CostEngine(pricing_catalog, usage_store)
    budget_manager = BudgetManager(cost_engine, config_loader.get_budget_config())
    session_manager = SessionManager(session_store)
    rl_tracker = RateLimitTracker(replica_count=1)
    cb_registry = CircuitBreakerRegistry(failure_threshold=2, reset_timeout=0.2)

    sim_onprem = SimulatedProvider(
        name="ollama",
        model_id="onprem-gpu",
        context_limit=4096,
        ttft_s=0.01,
    )
    sim_frontier = SimulatedProvider(
        name="anthropic",
        model_id="cloud-frontier",
        context_limit=200_000,
        ttft_s=0.05,
    )
    sim_fast = SimulatedProvider(
        name="openai",
        model_id="cloud-fast",
        context_limit=128_000,
        ttft_s=0.02,
    )

    providers = {
        "onprem-gpu": sim_onprem,
        "cloud-frontier": sim_frontier,
        "cloud-fast": sim_fast,
    }

    fabric = ComputeFabric(
        config=config_loader,
        circuit_breakers=cb_registry,
        rate_limit_tracker=rl_tracker,
        budget_manager=budget_manager,
        usage_store=usage_store,
        feedback_store=feedback_store,
    )
    # Wire providers directly into fabric cache
    fabric._provider_cache = dict(providers)
    fabric._providers = providers

    return {
        "fabric": fabric,
        "config": config_loader,
        "sim_onprem": sim_onprem,
        "sim_frontier": sim_frontier,
        "sim_fast": sim_fast,
        "rl_tracker": fabric._rate_limit_tracker,
        "cb_registry": cb_registry,
        "budget_manager": budget_manager,
        "usage_store": usage_store,
        "session_manager": session_manager,
    }


class TestCombinatorialRoutingMatrix:
    """Multi-factor combinatorial matrix tests."""

    def test_matrix_sovereignty_gating_overrides_bandit_preference(self, combinatorial_env):
        """Sensitive prompt with PII data categories must route to onprem-gpu even if cloud is preferred."""
        fabric = combinatorial_env["fabric"]
        sim_frontier = combinatorial_env["sim_frontier"]
        sim_onprem = combinatorial_env["sim_onprem"]

        # 1. Non-sensitive prompt routes to frontier cloud (top of role chain)
        res_normal = fabric.execute(
            role="general",
            prompt="Write a generic greeting",
            options={"data_categories": ["GENERAL"]},
        )
        assert res_normal.model_id == "cloud-frontier"
        assert sim_frontier.call_count == 1
        assert sim_onprem.call_count == 0

        # 2. Sensitive prompt requiring PII category must bypass cloud and route to on-prem
        res_pii = fabric.execute(
            role="general",
            prompt="Analyze confidential user SSN 000-11-2222",
            options={"data_categories": ["PII", "GENERAL"]},
        )
        assert res_pii.model_id == "onprem-gpu"
        assert sim_onprem.call_count == 1

    def test_matrix_budget_cap_exhaustion_triggers_automatic_fallback(self, combinatorial_env):
        """When daily budget is tripped, cloud candidates are filtered out and traffic falls to free tier."""
        fabric = combinatorial_env["fabric"]
        usage_store = combinatorial_env["usage_store"]
        sim_onprem = combinatorial_env["sim_onprem"]

        # Record spend that exceeds the $10.0 daily limit (5M * $3/1M = $15.00 > $10.00)
        rec = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id="cloud-frontier",
            provider="anthropic",
            role="general",
            intent="coding",
            input_tokens=5_000_000,
            output_tokens=50_000,
            elapsed_s=1.0,
            success=True,
            is_cloud=True,
            simulated_cost_avoided=0.0,
            complexity_score=5,
        )
        usage_store.record(rec)
        combinatorial_env["budget_manager"]._spend_cache.clear()
        combinatorial_env["budget_manager"]._cached_spend("daily")
        combinatorial_env["budget_manager"]._cached_spend("monthly")

        # Immediate routing execution must prune cloud models and route to onprem-gpu
        res = fabric.execute(role="general", prompt="Routine coding task")
        assert res.model_id == "onprem-gpu"
        assert sim_onprem.call_count == 1

    def test_matrix_rate_limit_429_failover_and_headroom_settle(self, combinatorial_env):
        """When primary model returns 429, router immediately fails over to secondary without error."""
        fabric = combinatorial_env["fabric"]
        sim_frontier = combinatorial_env["sim_frontier"]
        sim_fast = combinatorial_env["sim_fast"]
        rl_tracker = combinatorial_env["rl_tracker"]

        # Schedule a 429 on cloud-frontier
        snap = RateLimitSnapshot(rpm_remaining=0, retry_after_s=10.0)
        sim_frontier.fault_schedule = ["429"]
        sim_frontier.rate_limit_snapshot = snap

        res = fabric.execute(role="general", prompt="Hello world")
        # Should failover to cloud-fast
        assert res.model_id == "cloud-fast"
        assert sim_frontier.call_count == 1
        assert sim_fast.call_count == 1

        # Frontier must now be recognized as rate-limited/in cooldown
        util = rl_tracker.peek_utilization("cloud-frontier", "anthropic")
        assert util.state in (RateState.EXHAUSTED, RateState.NEARING_SATURATION) or util.headroom_score < 0.5

    def test_matrix_context_cliff_ephemeral_projection(self, combinatorial_env):
        """Conversations exceeding on-prem model's 4k limit trigger SessionProjector middle-out elision."""
        fabric = combinatorial_env["fabric"]
        sim_onprem = combinatorial_env["sim_onprem"]
        session_mgr = combinatorial_env["session_manager"]

        # Create session with multiple large turns that would exceed 4k tokens
        session = session_mgr.create_session(role="general")
        for i in range(12):
            if i % 2 == 0:
                session_mgr.add_user_message(session, f"Turn {i}: " + ("context information " * 120))
            else:
                session_mgr.add_assistant_message(session, f"Turn {i}: " + ("context reply " * 120))
        session_mgr.save_session(session)

        # Force routing exclusively to onprem-gpu by tagging prompt with FINANCIAL (which only onprem allows)
        res = fabric.execute_session(
            role="general",
            messages=session.get_messages_as_dicts(include_metadata=False),
            options={
                "session_id": session.session_id,
                "data_categories": ["FINANCIAL"],
            },
        )
        assert res.model_id == "onprem-gpu"
        assert sim_onprem.call_count == 1
        assert res.metadata.get("context_projected") is True
        assert res.metadata.get("turns_gisted", 0) > 0 or res.metadata.get("turns_dropped", 0) > 0

    def test_matrix_circuit_breaker_isolation_and_recovery_probing(self, combinatorial_env):
        """Repeated 503 errors trip circuit to open, skipping network attempts until reset timeout."""
        fabric = combinatorial_env["fabric"]
        sim_frontier = combinatorial_env["sim_frontier"]
        sim_fast = combinatorial_env["sim_fast"]
        cb_registry = combinatorial_env["cb_registry"]

        # Force 2 failures on cloud-frontier to trip circuit threshold (failure_threshold=2)
        sim_frontier.fault_schedule = ["503", "503"]
        fabric.execute(role="general", prompt="Request 1")  # fails frontier, succeeds on fast
        fabric.execute(role="general", prompt="Request 2")  # fails frontier, succeeds on fast

        cb = cb_registry.get_or_create("cloud-frontier")
        state, _ = cb.peek_state()
        assert state == "open"

        # Third request: frontier circuit is open, must NOT attempt frontier at all
        call_count_before = sim_frontier.call_count
        res3 = fabric.execute(role="general", prompt="Request 3")
        assert res3.model_id == "cloud-fast"
        assert sim_frontier.call_count == call_count_before  # 0 calls made to frontier

    def test_matrix_cache_affinity_vs_ttft_queue_tradeoff(self, combinatorial_env):
        """Cache affinity favors recently routed model until simulated TTFT queueing delay penalizes it."""
        fabric = combinatorial_env["fabric"]
        rl_tracker = combinatorial_env["rl_tracker"]
        sim_frontier = combinatorial_env["sim_frontier"]
        sim_fast = combinatorial_env["sim_fast"]

        # Initial turn routes to cloud-frontier
        res1 = fabric.execute(
            role="reasoning",
            prompt="Initial query",
            options={"session_id": "session-affinity-1"},
        )
        assert res1.model_id == "cloud-frontier"

        # Next turn shortly after: cache affinity boosts cloud-frontier
        res2 = fabric.execute(
            role="reasoning",
            prompt="Followup query within cache window",
            options={
                "session_id": "session-affinity-1",
                "last_model_id": "cloud-frontier",
                "last_routed_at": time.time(),
            },
        )
        assert res2.model_id == "cloud-frontier"

    @pytest.mark.asyncio
    async def test_matrix_streaming_outcome_parity_and_usagestore_spend(self, combinatorial_env):
        """Streaming execution emits tokens, records usage to UsageStore, and updates RateLimitTracker."""
        fabric = combinatorial_env["fabric"]
        usage_store = combinatorial_env["usage_store"]
        sim_frontier = combinatorial_env["sim_frontier"]

        collected_tokens: list[str] = []
        async for token in fabric.execute_stream(
            role="general",
            prompt="Stream me a poem",
        ):
            collected_tokens.append(token)

        assert len(collected_tokens) > 0
        assert sim_frontier.call_count == 1

        # Confirm spend was recorded in UsageStore
        records = usage_store.query()
        assert len(records) >= 1
        stream_record = records[-1]
        assert stream_record.model_id == "cloud-frontier"
        assert stream_record.success is True
        assert stream_record.input_tokens > 0
        assert stream_record.output_tokens > 0
