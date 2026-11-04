"""Full End-to-End (E2E) Integration Tests for BenchWarmer-Driven Routing.

Verifies complete execution dispatch through ComputeFabric, RoutingPolicyStack,
SessionManager, and ProviderCatalog using software-defined SimulatedProvider instances:

1. E2E Task Complexity Routing: Low complexity (Band 0) balances cost vs instruction quality
   selecting Llama-8B; high complexity (Band 2) prioritizes hard reasoning benchmarks (math_hard/gpqa)
   selecting Llama-70B.
2. E2E Hardware VRAM Gating: Live execution with `max_vram_gb` gates out 70B parameter models
   in Stage 1 CapabilityFilter and dispatches to 8B parameter models.
3. E2E Goodhart Contamination Defense: Contaminated fine-tune (`Goodhart-Overfit-7B`) is
   penalized by Stage 2 ContaminationAuditFactor and deprioritized in favor of clean foundation models.
4. E2E Streaming Outcome: `fabric.execute_stream()` streams chunks from the BenchWarmer-selected
   model with TTFT tracking and usage recording.
5. E2E Multi-Turn Session Flow: Multi-turn session execution preserves conversation history
   and adapts routing dynamically across complexity shifts.
6. E2E Dynamic Catalog Discovery & Metadata Enrichment: Auto-registered provider models
   are enriched with BenchWarmer metrics and immediately routable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aurarouter.catalog import ProviderCatalog
from aurarouter.circuit_breaker import CircuitBreakerRegistry
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.ratelimit.tracker import RateLimitTracker
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.feedback_store import FeedbackStore
from aurarouter.savings.pricing import CostEngine, ModelPrice, PricingCatalog
from aurarouter.savings.usage_store import UsageStore
from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.models import Message
from aurarouter.sessions.store import SessionStore
from tests.testing.simulated_provider import SimulatedProvider


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        yield Path(td)


@pytest.fixture
def e2e_env(temp_dir):
    """Sets up a complete ComputeFabric environment with SimulatedProviders matching BenchWarmer models."""
    cfg_data = {
        "version": "2.0",
        "routing_stack": {
            "mode": "active",
            "bandit_enabled": True,
            "audit_enabled": True,
            "cost_lambda_max": 0.35,
            "cost_elasticity_kappa": 0.32,
        },
        "models": {
            "local-llama-8b": {
                "provider": "ollama",
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "hosting_tier": "local",
                "context_window": 131072,
                "cost_per_1m_input": 0.08,
                "cost_per_1m_output": 0.08,
            },
            "local-llama-70b": {
                "provider": "ollama",
                "model_name": "meta-llama/Llama-3.1-70B-Instruct",
                "hosting_tier": "local",
                "context_window": 131072,
                "cost_per_1m_input": 0.80,
                "cost_per_1m_output": 0.80,
            },
            "clean-phi-4": {
                "provider": "ollama",
                "model_name": "microsoft/Phi-4",
                "hosting_tier": "local",
                "context_window": 16384,
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
            },
            "trap-overfit-7b": {
                "provider": "ollama",
                "model_name": "overfit-ai/Goodhart-Overfit-7B",
                "hosting_tier": "local",
                "context_window": 8192,
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
            },
        },
        "roles": {
            "analysis": {
                "models": ["local-llama-8b", "local-llama-70b"],
            },
            "audit_role": {
                "models": ["trap-overfit-7b", "clean-phi-4"],
            },
        },
        "provider_catalog": {
            "manual": [
                {"name": "mock_prov", "endpoint": "http://localhost:9999"}
            ]
        },
        "savings": {
            "budget": {
                "enabled": False,
            }
        },
        "resilience": {
            "failure_threshold": 3,
            "reset_timeout": 0.2,
        },
    }

    config_loader = ConfigLoader(allow_missing=True)
    config_loader.config = cfg_data
    usage_store = UsageStore(db_path=temp_dir / "usage.db")
    feedback_store = FeedbackStore(db_path=temp_dir / "feedback.db")
    session_store = SessionStore(db_path=temp_dir / "sessions.db")
    pricing_catalog = PricingCatalog(overrides={
        "local-llama-8b": ModelPrice(0.08, 0.08),
        "local-llama-70b": ModelPrice(0.80, 0.80),
        "clean-phi-4": ModelPrice(0.0, 0.0),
        "trap-overfit-7b": ModelPrice(0.0, 0.0),
    })
    cost_engine = CostEngine(pricing_catalog, usage_store)
    budget_manager = BudgetManager(cost_engine, config_loader.get_budget_config())
    session_manager = SessionManager(session_store)
    rl_tracker = RateLimitTracker(replica_count=1)
    cb_registry = CircuitBreakerRegistry()

    # Providers
    sim_8b = SimulatedProvider(name="ollama", model_id="local-llama-8b", context_limit=131072)
    sim_70b = SimulatedProvider(name="ollama", model_id="local-llama-70b", context_limit=131072)
    sim_phi = SimulatedProvider(name="ollama", model_id="clean-phi-4", context_limit=16384)
    sim_trap = SimulatedProvider(name="ollama", model_id="trap-overfit-7b", context_limit=8192)

    providers = {
        "local-llama-8b": sim_8b,
        "local-llama-70b": sim_70b,
        "clean-phi-4": sim_phi,
        "trap-overfit-7b": sim_trap,
    }

    fabric = ComputeFabric(
        config=config_loader,
        circuit_breakers=cb_registry,
        rate_limit_tracker=rl_tracker,
        budget_manager=budget_manager,
        usage_store=usage_store,
        feedback_store=feedback_store,
    )
    for mid, prov in providers.items():
        fabric._provider_cache[mid] = prov

    return {
        "fabric": fabric,
        "config": config_loader,
        "session_manager": session_manager,
        "providers": providers,
        "temp_dir": temp_dir,
    }


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestBenchWarmerE2EIntegration:
    def test_e2e_complexity_driven_benchmark_routing(self, e2e_env):
        """Verify empirical BenchWarmer benchmark scores drive live model dispatch.

        With zero historical feedback turns and cost elasticity:
        - Complexity 2 (Band 0: chat/instruction) prefers Llama-8B (fast, strong instruction prior, low cost)
        - Complexity 9 (Band 2: hard math/reasoning) prefers Llama-70B (high empirical math_hard/gpqa prior)
        """
        fabric = e2e_env["fabric"]

        # 1. Low complexity -> dispatches to local-llama-8b
        res_low = fabric.execute(
            "analysis",
            prompt="Capital of France?",
            options={"complexity_score": 2, "deterministic": True},
        )
        assert res_low.model_id == "local-llama-8b"
        assert res_low.text.startswith("[ollama]")

        # 2. High complexity -> dispatches to local-llama-70b based on hard reasoning benchmarks
        res_high = fabric.execute(
            "analysis",
            prompt="Derive the asymptotic complexity of this recursive algorithm with proof",
            options={"complexity_score": 9, "deterministic": True},
        )
        assert res_high.model_id == "local-llama-70b"
        assert res_high.text.startswith("[ollama]")

        # Verify trace metadata recorded the selection
        trace = fabric.last_routing_trace
        assert trace is not None
        assert trace.ranked_chain[0] == "local-llama-70b"

    def test_e2e_hardware_vram_filtering_under_execution(self, e2e_env):
        """Verify execution with max_vram_gb prunes 70B model and routes to 8B."""
        fabric = e2e_env["fabric"]

        # Request high complexity (which normally chooses 70B), but limit VRAM to 24 GB
        res = fabric.execute(
            "analysis",
            prompt="Complex mathematical derivation",
            options={
                "complexity_score": 9,
                "max_vram_gb": 24.0,
                "quant": "q4_k_m",
                "deterministic": True,
            },
        )

        # 70B requires ~48.7 GB under q4_k_m, so it must be pruned by CapabilityFilter
        assert res.model_id == "local-llama-8b"
        assert res.text.startswith("[ollama]")

        trace = fabric.last_routing_trace
        assert trace is not None
        assert "local-llama-70b" in trace.pruned
        assert "vram_exceeded" in trace.pruned["local-llama-70b"]
        assert "q4_k_m" in trace.pruned["local-llama-70b"]

    def test_e2e_goodhart_overfit_trap_defense(self, e2e_env):
        """Verify Goodhart contamination smell is penalized in Stage 2 during live execution.

        trap-overfit-7b is first in the YAML chain, but ContaminationAuditFactor penalizes
        it with 0.60x multiplier, allowing clean-phi-4 to win and execute.
        """
        fabric = e2e_env["fabric"]

        res = fabric.execute(
            "audit_role",
            prompt="Solve multi-step reasoning problem",
            options={"complexity_score": 8, "deterministic": True},
        )

        assert res.model_id == "clean-phi-4"
        assert res.text.startswith("[ollama]")

        trace = fabric.last_routing_trace
        assert trace is not None
        trap_score = next(s for s in trace.scores if s.model_id == "trap-overfit-7b")
        assert trap_score.audit_mult == pytest.approx(0.60, abs=1e-3)

    @pytest.mark.asyncio
    async def test_e2e_streaming_with_benchwarmer_selection(self, e2e_env):
        """Verify execute_stream streams tokens end-to-end from the BenchWarmer-selected model."""
        fabric = e2e_env["fabric"]

        chunks = []
        async for chunk in fabric.execute_stream(
            "analysis",
            prompt="Generate high-complexity theorem proof",
            options={"complexity_score": 9, "deterministic": True},
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert fabric.last_routing_trace.ranked_chain[0] == "local-llama-70b"

    def test_e2e_session_manager_multi_turn_with_benchwarmer(self, e2e_env):
        """Verify multi-turn session execution preserves conversation history across complexity shifts."""
        fabric = e2e_env["fabric"]
        session_mgr = e2e_env["session_manager"]

        session = session_mgr.create_session(role="analysis")

        # Turn 1: simple greeting (complexity 2 -> 8B)
        session.add_message(Message(role="user", content="Hello, what can you help me with?"))
        messages_turn1 = session.get_messages_as_dicts()
        res1 = fabric.execute_session(
            role="analysis",
            messages=messages_turn1,
            options={"complexity_score": 2, "deterministic": True},
        )
        assert res1.model_id == "local-llama-8b"
        session.add_message(Message(role="assistant", content=res1.text))

        # Turn 2: hard reasoning question (complexity 9 -> 70B)
        session.add_message(Message(role="user", content="Now prove the Riemann hypothesis equivalence"))
        messages_turn2 = session.get_messages_as_dicts()
        res2 = fabric.execute_session(
            role="analysis",
            messages=messages_turn2,
            options={"complexity_score": 9, "deterministic": True},
        )
        assert res2.model_id == "local-llama-70b"
        session.add_message(Message(role="assistant", content=res2.text))

        assert len(session.history) == 4

    def test_e2e_catalog_discovery_enrichment_and_dispatch(self, e2e_env):
        """Verify discovered models are enriched via BenchWarmer and immediately routable."""
        fabric = e2e_env["fabric"]
        config = e2e_env["config"]

        # Mock MCP client that lists models
        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.get_capabilities.return_value = ["provider.list_models"]
        mock_client.call_tool.return_value = [
            {"id": "Qwen2.5-72B-Instruct", "name": "Qwen2.5-72B-Instruct"},
        ]

        catalog = ProviderCatalog(config)
        catalog._clients["mock_prov"] = mock_client

        added = catalog.auto_register_models("mock_prov", config)
        assert added == 1

        reg_model_id = "mock_prov/Qwen2.5-72B-Instruct"
        mcfg = config.get_model_config(reg_model_id)
        assert mcfg is not None

        # Verify auto-enrichment occurred
        assert "params_b" in mcfg
        assert mcfg["params_b"] == pytest.approx(72.7, abs=1.0)
        assert "benchmarks" in mcfg
        assert "ifeval" in mcfg["benchmarks"]
        assert "estimated_vram_gb" in mcfg
        assert "q4_k_m" in mcfg["estimated_vram_gb"]

        # Wire a simulated provider for this new model and test live dispatch
        sim_qwen = SimulatedProvider(name="mock_prov", model_id=reg_model_id)
        fabric._provider_cache[reg_model_id] = sim_qwen
        config.set_role_chain("qwen_role", [reg_model_id])

        res = fabric.execute("qwen_role", prompt="Summarize this text", options={"complexity_score": 5})
        assert res.model_id == reg_model_id
        assert res.text.startswith("[mock_prov]")
