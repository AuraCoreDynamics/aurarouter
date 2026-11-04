"""Corpus Loader and Software-Defined Harness for Declarative Routing Evaluation.

Provides reusable utilities for loading, configuring, and executing declarative
scenarios from tests/fixtures/routing_evaluation_corpus.json across test suites.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from aurarouter.circuit_breaker import CircuitBreakerRegistry
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import IntentRegistry, build_intent_registry
from aurarouter.ratelimit.tracker import RateLimitTracker
from aurarouter.routing import TriageResult, analyze_intent
from aurarouter.routing_stack.context import RoutingDecisionTrace
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.feedback_store import FeedbackStore
from aurarouter.savings.models import GenerateResult, UsageRecord
from aurarouter.savings.pricing import CostEngine, ModelPrice, PricingCatalog
from aurarouter.savings.usage_store import UsageStore
from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.store import SessionStore
from tests.testing.simulated_provider import SimulatedProvider

CORPUS_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "routing_evaluation_corpus.json"
)


def load_routing_evaluation_corpus(path: Path | str | None = None) -> dict[str, Any]:
    """Load and parse the declarative routing evaluation corpus JSON artifact."""
    target_path = Path(path) if path else CORPUS_FIXTURE_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Routing evaluation corpus not found: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_routing_scenario_ids(path: Path | str | None = None) -> list[str]:
    """Return all scenario IDs declared in the corpus."""
    corpus = load_routing_evaluation_corpus(path)
    return [s["id"] for s in corpus.get("scenarios", [])]


def get_routing_scenario(scenario_id: str, path: Path | str | None = None) -> dict[str, Any]:
    """Look up a specific scenario by ID from the corpus."""
    corpus = load_routing_evaluation_corpus(path)
    for s in corpus.get("scenarios", []):
        if s.get("id") == scenario_id:
            return s
    raise KeyError(f"Scenario ID '{scenario_id}' not found in routing corpus.")


def setup_corpus_environment(scenario: dict[str, Any], temp_dir: Path) -> dict[str, Any]:
    """Construct an isolated software-defined environment for a declarative scenario."""
    stack_cfg = dict(scenario["stack_configuration"].get("routing_stack", {}))
    models_cfg = dict(scenario["stack_configuration"].get("models", {}))
    roles_cfg = dict(scenario["stack_configuration"].get("roles", {}))

    router_resp = scenario["intent_evaluation"]["simulated_router_response"]
    models_cfg["sim-router"] = {
        "provider": "simulated",
        "hosting_tier": "cloud",
        "context_window": 128000,
    }
    roles_cfg["router"] = ["sim-router"]

    cfg_data: dict[str, Any] = {
        "version": "2.0",
        "system": {
            "log_level": "INFO",
            "default_timeout": 30.0,
        },
        "routing_stack": stack_cfg,
        "models": models_cfg,
        "roles": roles_cfg,
    }

    custom_bindings = scenario["stack_configuration"].get("custom_role_bindings")
    if custom_bindings:
        catalog = {}
        for analyzer_id, bindings in custom_bindings.items():
            catalog[analyzer_id] = {
                "kind": "analyzer",
                "display_name": analyzer_id,
                "analyzer_kind": "intent_triage",
                "role_bindings": bindings,
            }
            cfg_data["system"]["active_analyzer"] = analyzer_id
        cfg_data["catalog"] = catalog

    budget_spec = scenario["stack_configuration"].get("budget")
    if budget_spec:
        cfg_data["savings"] = {"budget": dict(budget_spec)}

    config_loader = ConfigLoader(allow_missing=True)
    config_loader.config = cfg_data

    scenario_slug = scenario.get("id", "default")
    usage_store = UsageStore(db_path=temp_dir / f"usage_{scenario_slug}.db")
    feedback_store = FeedbackStore(db_path=temp_dir / f"feedback_{scenario_slug}.db")
    session_store = SessionStore(db_path=temp_dir / f"sessions_{scenario_slug}.db")

    pricing_catalog = PricingCatalog(overrides={
        mid: ModelPrice(
            float(mcfg.get("cost_per_1m_input", 0.0)),
            float(mcfg.get("cost_per_1m_output", 0.0)),
        )
        for mid, mcfg in models_cfg.items()
        if "cost_per_1m_input" in mcfg or "cost_per_1m_output" in mcfg
    })
    cost_engine = CostEngine(pricing_catalog, usage_store)
    budget_cfg = config_loader.get_budget_config()
    budget_manager = BudgetManager(cost_engine, budget_cfg) if budget_spec else None
    session_manager = SessionManager(session_store)

    rl_tracker = RateLimitTracker(replica_count=1)
    cb_registry = CircuitBreakerRegistry(failure_threshold=2, reset_timeout=30.0)

    # Seed circuit breakers
    for mid, cb_spec in scenario.get("runtime_state", {}).get("circuit_breakers", {}).items():
        cb = cb_registry.get_or_create(mid)
        if cb_spec.get("state") == "open":
            for _ in range(cb_spec.get("consecutive_failures", 3)):
                cb.record_failure()

    # Seed budget spend
    if budget_manager and "budget_spent" in scenario.get("runtime_state", {}):
        spent_today = float(scenario["runtime_state"]["budget_spent"].get("today", 0.0))
        if spent_today > 0.0:
            cloud_mid = next(
                (m for m, c in models_cfg.items() if c.get("hosting_tier") == "cloud"),
                "cloud-paid-frontier",
            )
            rec = UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model_id=cloud_mid,
                provider=models_cfg.get(cloud_mid, {}).get("provider", "anthropic"),
                role="general",
                intent="coding",
                input_tokens=int(spent_today * 1_000_000),
                output_tokens=50_000,
                elapsed_s=1.0,
                success=True,
                is_cloud=True,
                simulated_cost_avoided=0.0,
                complexity_score=5,
            )
            usage_store.record(rec)
            budget_manager._spend_cache.clear()

    # Instantiate SimulatedProviders
    providers: dict[str, SimulatedProvider] = {}
    for mid, mcfg in models_cfg.items():
        if mid == "sim-router":
            sim = SimulatedProvider(
                name="simulated",
                model_id=mid,
                context_limit=128000,
            )
            sim.generate_with_usage = lambda p, **kw: GenerateResult(
                text=router_resp,
                input_tokens=10,
                output_tokens=10,
                model_id="sim-router",
                provider="simulated",
                finish_reason="stop",
            )
        else:
            sim = SimulatedProvider(
                name=mcfg.get("provider", "simulated"),
                model_id=mid,
                context_limit=int(mcfg.get("context_limit") or mcfg.get("context_window") or 128000),
                model_config=mcfg,
            )
        providers[mid] = sim

    fabric = ComputeFabric(
        config=config_loader,
        circuit_breakers=cb_registry,
        rate_limit_tracker=rl_tracker,
        budget_manager=budget_manager,
        usage_store=usage_store,
        feedback_store=feedback_store,
    )
    fabric._provider_cache = dict(providers)
    fabric._providers = providers

    # Seed rate limits on the active fabric tracker instance
    active_rl_tracker = fabric._rate_limit_tracker
    for mid, rl_spec in scenario.get("runtime_state", {}).get("rate_limits", {}).items():
        prov = rl_spec.get("provider", "simulated")
        headers = rl_spec.get("headers", {})
        if headers:
            active_rl_tracker.sync_from_headers(prov, mid, headers)

    registry = build_intent_registry(config_loader)

    return {
        "fabric": fabric,
        "config": config_loader,
        "registry": registry,
        "providers": providers,
        "rl_tracker": active_rl_tracker,
        "cb_registry": cb_registry,
        "budget_manager": budget_manager,
        "usage_store": usage_store,
        "session_manager": session_manager,
    }


def execute_scenario_intent_and_routing(
    env: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    """Execute end-to-end intent triage and routing execution for a scenario."""
    fabric: ComputeFabric = env["fabric"]
    registry: IntentRegistry = env["registry"]

    raw_prompt = scenario["intent_evaluation"]["raw_prompt"]

    # 1. Intent Triage
    triage: TriageResult = analyze_intent(fabric, raw_prompt, intent_registry=registry)

    # Role resolution via IntentRegistry
    target_role = registry.resolve_role(triage.intent)
    if not target_role:
        target_role = scenario["intent_evaluation"].get("expected_role", "general")

    # 2. Options Extraction
    exec_opts = dict(scenario.get("execution_options", {}))
    session_affinity = scenario.get("runtime_state", {}).get("session_affinity")
    if session_affinity:
        exec_opts["session_id"] = session_affinity.get("session_id")
        exec_opts["last_model_id"] = session_affinity.get("last_model_id")
        exec_opts["last_routed_at"] = time.time() - float(session_affinity.get("seconds_ago", 15.0))

    exec_opts["complexity_score"] = triage.complexity

    # 3. Stack Explain Route
    trace: RoutingDecisionTrace = fabric.explain_route(
        role=target_role,
        prompt=raw_prompt,
        intent=triage.intent,
        complexity_score=triage.complexity,
        data_categories=exec_opts.get("data_categories"),
        session_id=exec_opts.get("session_id"),
        last_model_id=exec_opts.get("last_model_id"),
        last_routed_at=exec_opts.get("last_routed_at"),
        deterministic=exec_opts.get("deterministic", True),
        max_vram_gb=exec_opts.get("max_vram_gb"),
        quant=exec_opts.get("quant", "q4_k_m"),
    )

    # 4. Fabric Execution
    result: GenerateResult = fabric.execute(
        role=target_role,
        prompt=raw_prompt,
        options=exec_opts,
    )

    return {
        "triage": triage,
        "target_role": target_role,
        "trace": trace,
        "result": result,
    }

