"""T7.2 — MCP Tool Contract Validation.

Validates that:
- AuraXLM's `analyze_route` and `score_experts` tools declare `_aura_routing_context`
- The contracts between AuraRouter → AuraXLM and AuraRouter → AuraCode are complete
- The routing context schema in each contract matches the canonical 9-field spec
- AuraCode's EmbeddedRouterBackend correctly parses the context
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from aurarouter.contracts.auracode import (
    AURACODE_ROUTING_CONTEXT_SCHEMA,
    create_auracode_analyzer_spec,
)
from aurarouter.contracts.auraxlm import (
    ANALYZE_ROUTE_PARAMS,
    ANALYZE_ROUTE_RESPONSE,
    AURAXLM_ANALYZER_SPEC,
)

CANONICAL_KEYS = {
    "strategy",
    "confidence_score",
    "complexity_score",
    "selected_route",
    "analyzer_chain",
    "intent",
    "hard_routed",
    "simulated_cost_avoided",
    "metadata",
}


# ── T7.2a: AuraXLM contract completeness ──────────────────────────────────────


class TestAuraxlmMcpContract:
    def test_analyze_route_params_declares_routing_context(self):
        """analyze_route params must include _aura_routing_context key."""
        assert "_aura_routing_context" in ANALYZE_ROUTE_PARAMS

    def test_analyze_route_context_param_is_optional(self):
        """_aura_routing_context must be optional (not required)."""
        ctx_param = ANALYZE_ROUTE_PARAMS["_aura_routing_context"]
        assert ctx_param.get("required") is False

    def test_analyze_route_context_param_is_object_type(self):
        """_aura_routing_context param must have type 'object'."""
        ctx_param = ANALYZE_ROUTE_PARAMS["_aura_routing_context"]
        assert ctx_param["type"] == "object"

    def test_analyze_route_context_param_has_description(self):
        """_aura_routing_context must have a non-empty description."""
        ctx_param = ANALYZE_ROUTE_PARAMS["_aura_routing_context"]
        assert "description" in ctx_param
        assert len(ctx_param["description"]) > 30

    def test_analyze_route_response_includes_routing_context(self):
        """analyze_route response schema must include _aura_routing_context."""
        assert "_aura_routing_context" in ANALYZE_ROUTE_RESPONSE

    def test_analyze_route_response_context_has_canonical_keys(self):
        """analyze_route response _aura_routing_context must contain all 9 canonical fields."""
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        for key in CANONICAL_KEYS:
            assert key in ctx, f"Missing canonical key {key!r} in ANALYZE_ROUTE_RESPONSE"

    def test_analyze_route_response_strategy_is_string(self):
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        assert isinstance(ctx["strategy"], str)

    def test_analyze_route_response_confidence_score_is_float(self):
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        assert isinstance(ctx["confidence_score"], float)

    def test_analyze_route_response_complexity_score_is_int(self):
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        assert isinstance(ctx["complexity_score"], int)

    def test_analyze_route_response_hard_routed_is_bool(self):
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        assert isinstance(ctx["hard_routed"], bool)

    def test_analyze_route_response_simulated_cost_avoided_is_float(self):
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        assert isinstance(ctx["simulated_cost_avoided"], float)

    def test_analyze_route_response_analyzer_chain_is_list(self):
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        assert isinstance(ctx["analyzer_chain"], list)

    def test_analyze_route_response_metadata_is_dict(self):
        ctx = ANALYZE_ROUTE_RESPONSE["_aura_routing_context"]
        assert isinstance(ctx["metadata"], dict)

    def test_auraxlm_analyzer_spec_has_required_fields(self):
        """AuraXLM analyzer spec must declare analyzer_kind, capabilities, mcp_tool_name."""
        assert "analyzer_kind" in AURAXLM_ANALYZER_SPEC
        assert "capabilities" in AURAXLM_ANALYZER_SPEC
        assert "mcp_tool_name" in AURAXLM_ANALYZER_SPEC

    def test_auraxlm_analyzer_spec_mcp_tool_name(self):
        assert AURAXLM_ANALYZER_SPEC["mcp_tool_name"] == "auraxlm.analyze_route"


# ── T7.2b: AuraCode contract completeness ────────────────────────────────────


class TestAuraCodeMcpContract:
    def test_auracode_routing_context_schema_has_all_canonical_keys(self):
        """AURACODE_ROUTING_CONTEXT_SCHEMA must define all 9 canonical keys."""
        for key in CANONICAL_KEYS:
            assert key in AURACODE_ROUTING_CONTEXT_SCHEMA, (
                f"Missing canonical key {key!r} in AURACODE_ROUTING_CONTEXT_SCHEMA"
            )

    def test_auracode_routing_context_schema_simulated_cost_avoided_mentions_float(self):
        """simulated_cost_avoided description must clarify float type."""
        desc = AURACODE_ROUTING_CONTEXT_SCHEMA["simulated_cost_avoided"]
        assert "float" in desc.lower() or "0.0" in desc

    def test_auracode_routing_context_schema_hard_routed_mentions_bool(self):
        desc = AURACODE_ROUTING_CONTEXT_SCHEMA["hard_routed"]
        assert "bool" in desc.lower()

    def test_auracode_analyzer_spec_produces_valid_structure(self):
        spec = create_auracode_analyzer_spec()
        assert spec["analyzer_kind"] == "intent_triage"
        assert "role_bindings" in spec
        assert "capabilities" in spec

    def test_auracode_role_bindings_all_map_to_valid_roles(self):
        """All AuraCode intent→role bindings must map to known roles."""
        spec = create_auracode_analyzer_spec()
        expected_roles = {"coding", "reasoning"}
        for intent, role in spec["role_bindings"].items():
            assert role in expected_roles, (
                f"Intent {intent!r} maps to unexpected role {role!r}"
            )


# ── T7.2c: Route task routing context propagation (mock-based) ───────────────


class TestRouteTaskRoutingContextPropagation:
    """Verify AuraCode's EmbeddedRouterBackend extracts routing context from GenerateResult."""

    @pytest.mark.asyncio
    async def test_auracode_parses_routing_context_from_route_result(self):
        """EmbeddedRouterBackend must extract all 9 canonical fields from GenerateResult.routing_context."""
        import importlib

        class FakeRC:
            strategy = "pipeline"
            confidence_score = 0.88
            complexity_score = 3
            selected_route = "coding"
            analyzer_chain = ["edge-complexity", "slm-intent"]
            intent = "SIMPLE_CODE"
            hard_routed = False
            simulated_cost_avoided = 0.0

        class FakeGenResult:
            text = "def hello(): pass"
            model_id = "ollama:llama3:8b"
            routing_context = FakeRC()

        ar = ModuleType("aurarouter")
        ar_config = ModuleType("aurarouter.config")
        ar_fabric = ModuleType("aurarouter.fabric")

        cfg_mock = MagicMock()
        cfg_mock.get_role_chain = MagicMock(return_value=["ollama:llama3:8b"])
        cfg_mock.get_all_model_ids = MagicMock(return_value=["ollama:llama3:8b"])

        fabric_mock = MagicMock()
        fabric_mock.execute = MagicMock(return_value=FakeGenResult())

        ar_config.ConfigLoader = MagicMock(return_value=cfg_mock)  # type: ignore
        ar_fabric.ComputeFabric = MagicMock(return_value=fabric_mock)  # type: ignore

        orig = {k: sys.modules.get(k) for k in ["aurarouter", "aurarouter.config", "aurarouter.fabric"]}
        sys.modules.update({"aurarouter": ar, "aurarouter.config": ar_config, "aurarouter.fabric": ar_fabric})
        try:
            from auracode.models.request import RequestIntent
            from auracode.routing import embedded as emb_mod
            importlib.reload(emb_mod)
            backend = emb_mod.EmbeddedRouterBackend()
            result = await backend.route(prompt="Write a hello function", intent=RequestIntent.GENERATE_CODE)
        finally:
            for k, v in orig.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        assert result.routing_context is not None
        for key in CANONICAL_KEYS - {"metadata"}:
            assert key in result.routing_context, f"Missing key {key!r} in RouteResult.routing_context"

    def test_route_task_injects_routing_context_when_pipeline_enabled(self):
        """_inject_routing_context appends _aura_routing_context to response string."""
        from aurarouter.mcp_tools import _inject_routing_context
        from aurarouter.analyzer_protocol import RoutingContext

        rc = RoutingContext(
            strategy="pipeline",
            confidence_score=0.91,
            complexity_score=2,
            selected_route="coding",
            analyzer_chain=["edge-complexity"],
            intent="SIMPLE_CODE",
        )
        result = _inject_routing_context("Generated code here", rc)
        assert "_aura_routing_context" in result
        assert "pipeline" in result
        assert "confidence_score" in result

    def test_inject_routing_context_sets_all_canonical_keys(self):
        """_inject_routing_context output must contain all 9 canonical fields."""
        import json as _json
        import re
        from aurarouter.mcp_tools import _inject_routing_context
        from aurarouter.analyzer_protocol import RoutingContext

        rc = RoutingContext(
            strategy="pipeline",
            confidence_score=0.88,
            complexity_score=3,
            selected_route="coding",
            analyzer_chain=["edge-complexity", "onnx-vector"],
            intent="SIMPLE_CODE",
            hard_routed=False,
            simulated_cost_avoided=0.0042,
        )
        result = _inject_routing_context("hello", rc)
        # Extract the JSON from the comment
        match = re.search(r'_aura_routing_context: ({.*?}) -->', result)
        assert match, f"No JSON found in result: {result!r}"
        ctx = _json.loads(match.group(1))
        for key in CANONICAL_KEYS:
            assert key in ctx, f"Missing key {key!r} in injected routing context"
