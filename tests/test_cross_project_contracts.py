"""Cross-project analyzer contract tests for AuraRouter.

Validates AuraCode and AuraXLM analyzer contracts, intent passing through
the broker pipeline, intent-aware bid merging, and the list_intents MCP tool.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aurarouter.analyzer_schema import validate_analyzer_spec
from aurarouter.broker import (
    AnalyzerBid,
    BrokerResult,
    broadcast_to_analyzers,
    merge_bids,
)
from aurarouter.config import ConfigLoader
from aurarouter.contracts.auracode import (
    AURACODE_CAPABILITIES,
    AURACODE_INTENTS,
    create_auracode_analyzer_spec,
)
from aurarouter.contracts.auraxlm import (
    AURAXLM_ANALYZER_SPEC,
    AURAXLM_ADVISOR_CAPABILITIES,
)
from aurarouter.intent_registry import IntentRegistry, build_intent_registry
from aurarouter.mcp_tools import list_intents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**extra) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": {}, "roles": {}, **extra}
    return cfg


# ======================================================================
# T7.3: AuraCode Analyzer Contract
# ======================================================================

class TestAuraCodeContract:
    """Verify the AuraCode analyzer spec and intent registration."""

    def test_auracode_spec_passes_validation(self):
        """AuraCode analyzer spec should pass validate_analyzer_spec()."""
        spec = create_auracode_analyzer_spec()
        result = validate_analyzer_spec(spec)
        assert result.valid, f"Validation errors: {result.errors}"
        assert not result.errors

    def test_auracode_spec_declares_intents(self):
        """AuraCode spec should declare all AURACODE_INTENTS as intents."""
        spec = create_auracode_analyzer_spec()
        result = validate_analyzer_spec(spec)
        for intent_name in AURACODE_INTENTS:
            assert intent_name in result.declared_intents, (
                f"Intent '{intent_name}' not declared in validation result"
            )

    def test_auracode_spec_has_required_fields(self):
        """AuraCode spec should contain analyzer_kind, role_bindings, capabilities."""
        spec = create_auracode_analyzer_spec()
        assert spec["analyzer_kind"] == "intent_triage"
        assert spec["role_bindings"] == AURACODE_INTENTS
        assert spec["capabilities"] == AURACODE_CAPABILITIES

    def test_auracode_intents_register_in_registry(self):
        """All AuraCode intents should register correctly in IntentRegistry."""
        registry = IntentRegistry()
        registry.register_from_role_bindings("auracode-test", AURACODE_INTENTS)

        for intent_name, target_role in AURACODE_INTENTS.items():
            defn = registry.get_by_name(intent_name)
            assert defn is not None, f"Intent '{intent_name}' not found in registry"
            assert defn.target_role == target_role
            assert defn.source == "auracode-test"

    def test_auracode_intents_all_valid_identifiers(self):
        """All AuraCode intent names should be valid Python identifiers."""
        for intent_name in AURACODE_INTENTS:
            assert intent_name.isidentifier(), (
                f"Intent name '{intent_name}' is not a valid identifier"
            )

    def test_auracode_capabilities_are_strings(self):
        """All AuraCode capabilities should be strings."""
        for cap in AURACODE_CAPABILITIES:
            assert isinstance(cap, str)


# ======================================================================
# T7.4: AuraXLM Advisor Contract
# ======================================================================

class TestAuraXLMContract:
    """Verify the AuraXLM analyzer spec and advisor contract."""

    def test_auraxlm_spec_passes_validation(self):
        """AuraXLM analyzer spec should pass validate_analyzer_spec()."""
        result = validate_analyzer_spec(AURAXLM_ANALYZER_SPEC)
        assert result.valid, f"Validation errors: {result.errors}"
        assert not result.errors

    def test_auraxlm_spec_has_required_fields(self):
        """AuraXLM spec should contain analyzer_kind and capabilities."""
        assert AURAXLM_ANALYZER_SPEC["analyzer_kind"] == "moe_ranking"
        assert "code" in AURAXLM_ANALYZER_SPEC["capabilities"]
        assert "domain-expert" in AURAXLM_ANALYZER_SPEC["capabilities"]

    def test_auraxlm_spec_has_mcp_tool_name(self):
        """AuraXLM spec should declare the MCP tool name."""
        assert AURAXLM_ANALYZER_SPEC["mcp_tool_name"] == "auraxlm.analyze_route"

    def test_auraxlm_advisor_capabilities(self):
        """AuraXLM advisor capabilities should include routing_advisor."""
        assert "routing_advisor" in AURAXLM_ADVISOR_CAPABILITIES


# ======================================================================
# T7.1: Intent Passing Through Broker
# ======================================================================

class TestBrokerIntentPassing:
    """Verify that broadcast_to_analyzers includes intent in the payload."""

    def test_broadcast_includes_intent_in_call(self):
        """broadcast_to_analyzers should pass intent to _call_single_analyzer."""
        config = _make_config(
            catalog={
                "test-analyzer": {
                    "kind": "analyzer",
                    "display_name": "Test",
                    "mcp_endpoint": "http://localhost:9999",
                    "mcp_tool_name": "test.analyze",
                    "analyzer_kind": "intent_triage",
                },
            }
        )

        with patch("aurarouter.broker._call_single_analyzer", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "confidence": 0.8,
                "role": "coding",
                "claimed_files": [],
                "proposed_tasks": [],
            }

            loop = asyncio.new_event_loop()
            try:
                bids = loop.run_until_complete(
                    broadcast_to_analyzers(config, "write a function", intent="generate_code")
                )
            finally:
                loop.close()

            # Verify _call_single_analyzer was called with intent
            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args
            assert call_kwargs.kwargs.get("intent") == "generate_code" or \
                   (len(call_kwargs.args) > 5 and call_kwargs.args[5] == "generate_code") or \
                   call_kwargs.kwargs.get("intent") == "generate_code"

    def test_broadcast_without_intent(self):
        """broadcast_to_analyzers should work without intent (backwards compatible)."""
        config = _make_config(catalog={})

        loop = asyncio.new_event_loop()
        try:
            bids = loop.run_until_complete(
                broadcast_to_analyzers(config, "hello")
            )
        finally:
            loop.close()

        assert bids == []  # No remote analyzers configured

    def test_broadcast_intent_in_payload(self):
        """The JSON-RPC payload should include intent in arguments when provided."""
        import httpx
        from unittest.mock import patch, MagicMock

        config = _make_config(
            catalog={
                "remote-analyzer": {
                    "kind": "analyzer",
                    "display_name": "Remote",
                    "mcp_endpoint": "http://localhost:9999",
                    "mcp_tool_name": "remote.analyze",
                    "analyzer_kind": "moe_ranking",
                },
            }
        )

        captured_payloads = []

        async def mock_post(self_client, url, json=None, **kwargs):
            captured_payloads.append(json)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "result": {
                    "confidence": 0.9,
                    "role": "coding",
                    "claimed_files": [],
                    "proposed_tasks": [],
                }
            }
            return mock_resp

        with patch("httpx.AsyncClient.post", mock_post):
            loop = asyncio.new_event_loop()
            try:
                bids = loop.run_until_complete(
                    broadcast_to_analyzers(config, "generate code", intent="generate_code")
                )
            finally:
                loop.close()

        assert len(captured_payloads) == 1
        payload = captured_payloads[0]
        assert payload["params"]["arguments"]["intent"] == "generate_code"


# ======================================================================
# T7.2: Intent-Aware Bid Merging
# ======================================================================

class TestMergeBidsIntentAware:
    """Verify that merge_bids prefers intent-matching bids."""

    def test_merge_bids_intent_bonus(self):
        """Bids from intent-supporting analyzers should get a confidence bonus."""
        bid_a = AnalyzerBid(
            analyzer_id="auracode",
            confidence=0.7,
            role="coding",
        )
        bid_b = AnalyzerBid(
            analyzer_id="generic",
            confidence=0.75,
            role="coding",
        )

        role_bindings = {
            "auracode": {"generate_code": "coding", "edit_code": "coding"},
            "generic": {},
        }

        result = merge_bids(
            [bid_a, bid_b],
            intent="generate_code",
            analyzer_role_bindings=role_bindings,
        )

        # bid_a should have gotten a 0.1 bonus: 0.7 -> 0.8
        assert result.merged_plan is not None
        # The top bid should be auracode since 0.8 > 0.75
        assert result.merged_plan[0]["analyzer_id"] == "auracode"
        assert result.merged_plan[0]["confidence"] == pytest.approx(0.8)

    def test_merge_bids_intent_bonus_clamped(self):
        """Intent confidence bonus should be clamped to 1.0."""
        bid = AnalyzerBid(
            analyzer_id="auracode",
            confidence=0.95,
            role="coding",
        )

        role_bindings = {"auracode": {"generate_code": "coding"}}

        result = merge_bids(
            [bid],
            intent="generate_code",
            analyzer_role_bindings=role_bindings,
        )

        assert result.merged_plan is not None
        assert result.merged_plan[0]["confidence"] == 1.0

    def test_merge_bids_no_intent(self):
        """merge_bids without intent should work as before (backwards compatible)."""
        bid_a = AnalyzerBid(analyzer_id="a", confidence=0.9, role="coding")
        bid_b = AnalyzerBid(analyzer_id="b", confidence=0.5, role="reasoning")

        result = merge_bids([bid_a, bid_b])

        assert result.merged_plan is not None
        assert len(result.merged_plan) == 2
        assert result.merged_plan[0]["analyzer_id"] == "a"

    def test_merge_bids_intent_no_match(self):
        """When intent doesn't match any analyzer's bindings, no bonus is applied."""
        bid = AnalyzerBid(analyzer_id="generic", confidence=0.7, role="coding")

        role_bindings = {"generic": {"chat": "reasoning"}}

        result = merge_bids(
            [bid],
            intent="generate_code",
            analyzer_role_bindings=role_bindings,
        )

        assert result.merged_plan is not None
        assert result.merged_plan[0]["confidence"] == pytest.approx(0.7)

    def test_merge_bids_intent_trace(self):
        """Intent-aware scoring should be recorded in execution trace."""
        bid = AnalyzerBid(analyzer_id="auracode", confidence=0.7, role="coding")
        role_bindings = {"auracode": {"generate_code": "coding"}}

        # Clear any stale broadcast trace
        broadcast_to_analyzers._last_trace = []  # type: ignore[attr-defined]

        result = merge_bids(
            [bid],
            intent="generate_code",
            analyzer_role_bindings=role_bindings,
        )

        trace_text = " ".join(result.execution_trace)
        assert "intent bonus" in trace_text.lower() or "intent-aware" in trace_text.lower()


# ======================================================================
# T7.5: list_intents MCP Tool
# ======================================================================

class TestListIntentsTool:
    """Verify the list_intents MCP tool returns correct data."""

    def test_list_intents_builtin(self):
        """list_intents should return built-in intents when no analyzer is active."""
        config = _make_config()

        result_str = list_intents(config)
        result = json.loads(result_str)

        assert "active_analyzer" in result
        assert "intents" in result
        assert isinstance(result["intents"], list)

        # Should have at least the 3 built-in intents
        intent_names = {i["name"] for i in result["intents"]}
        assert "DIRECT" in intent_names
        assert "SIMPLE_CODE" in intent_names
        assert "COMPLEX_REASONING" in intent_names

    def test_list_intents_with_analyzer(self):
        """list_intents should include analyzer-declared intents."""
        config = _make_config(
            system={"active_analyzer": "aurarouter-default"},
            catalog={
                "aurarouter-default": {
                    "kind": "analyzer",
                    "display_name": "Default",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {
                        "simple_code": "coding",
                        "complex_reasoning": "reasoning",
                        "review": "reviewer",
                    },
                    "capabilities": ["code", "reasoning", "review", "planning"],
                },
            },
        )

        result_str = list_intents(config)
        result = json.loads(result_str)

        assert result["active_analyzer"] == "aurarouter-default"
        intent_names = {i["name"] for i in result["intents"]}

        # Built-in intents should still be present
        assert "DIRECT" in intent_names

        # Analyzer-declared intents should be present
        assert "simple_code" in intent_names
        assert "review" in intent_names

    def test_list_intents_structure(self):
        """Each intent entry should have name, target_role, source, description."""
        config = _make_config()

        result_str = list_intents(config)
        result = json.loads(result_str)

        for intent in result["intents"]:
            assert "name" in intent
            assert "target_role" in intent
            assert "source" in intent
            assert "description" in intent

    def test_list_intents_auracode_registration(self):
        """list_intents should reflect AuraCode intents after registration."""
        config = _make_config(
            system={"active_analyzer": "auracode-analyzer"},
            catalog={
                "auracode-analyzer": {
                    "kind": "analyzer",
                    "display_name": "AuraCode",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": AURACODE_INTENTS,
                    "capabilities": AURACODE_CAPABILITIES,
                },
            },
        )

        result_str = list_intents(config)
        result = json.loads(result_str)

        intent_names = {i["name"] for i in result["intents"]}
        for intent_name in AURACODE_INTENTS:
            assert intent_name in intent_names, (
                f"AuraCode intent '{intent_name}' not in list_intents output"
            )


# ======================================================================
# T7.6: End-to-End Integration
# ======================================================================

class TestEndToEndIntentRouting:
    """End-to-end: AuraCode intent -> classification -> routing to correct role."""

    def test_auracode_intent_resolves_to_correct_role(self):
        """AuraCode intents should resolve to the correct role via IntentRegistry."""
        registry = IntentRegistry()
        registry.register_from_role_bindings("auracode", AURACODE_INTENTS)

        # Coding intents
        assert registry.resolve_role("generate_code") == "coding"
        assert registry.resolve_role("edit_code") == "coding"
        assert registry.resolve_role("complete_code") == "coding"

        # Reasoning intents
        assert registry.resolve_role("explain_code") == "reasoning"
        assert registry.resolve_role("review") == "reasoning"
        assert registry.resolve_role("chat") == "reasoning"
        assert registry.resolve_role("plan") == "reasoning"

    def test_auracode_intent_builds_from_config(self):
        """build_intent_registry should pick up AuraCode intents from catalog."""
        config = _make_config(
            system={"active_analyzer": "auracode-test"},
            catalog={
                "auracode-test": {
                    "kind": "analyzer",
                    "display_name": "AuraCode Test",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": AURACODE_INTENTS,
                    "capabilities": AURACODE_CAPABILITIES,
                },
            },
        )

        registry = build_intent_registry(config)

        # All AuraCode intents should be registered
        for intent_name, target_role in AURACODE_INTENTS.items():
            resolved = registry.resolve_role(intent_name)
            assert resolved == target_role, (
                f"Intent '{intent_name}' resolved to '{resolved}', "
                f"expected '{target_role}'"
            )

        # Built-in intents should still be present
        assert registry.resolve_role("DIRECT") == "coding"

    def test_auracode_classifier_choices_include_all_intents(self):
        """IntentRegistry.build_classifier_choices() should list all AuraCode intents."""
        registry = IntentRegistry()
        registry.register_from_role_bindings("auracode", AURACODE_INTENTS)

        choices = registry.build_classifier_choices()

        for intent_name in AURACODE_INTENTS:
            assert intent_name in choices, (
                f"Intent '{intent_name}' missing from classifier choices"
            )

    def test_merge_bids_with_auracode_bindings(self):
        """merge_bids with AuraCode role_bindings should boost matching analyzer."""
        bid_auracode = AnalyzerBid(
            analyzer_id="auracode",
            confidence=0.7,
            role="coding",
        )
        bid_other = AnalyzerBid(
            analyzer_id="other",
            confidence=0.72,
            role="coding",
        )

        role_bindings = {
            "auracode": AURACODE_INTENTS,
            "other": {},
        }

        # Clear stale trace
        broadcast_to_analyzers._last_trace = []  # type: ignore[attr-defined]

        result = merge_bids(
            [bid_auracode, bid_other],
            intent="generate_code",
            analyzer_role_bindings=role_bindings,
        )

        assert result.merged_plan is not None
        # auracode gets 0.7 + 0.1 = 0.8, other stays at 0.72
        assert result.merged_plan[0]["analyzer_id"] == "auracode"
