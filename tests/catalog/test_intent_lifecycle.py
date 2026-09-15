"""End-to-end intent lifecycle integration tests (Task Group 9.1).

Validates the complete lifecycle: Registration -> Classification -> Routing,
explicit intent overrides, unknown intent fallback, and analyzer switching.
All external systems (providers, MCP endpoints) are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.catalog_model import ArtifactKind, CatalogArtifact
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import (
    IntentDefinition,
    IntentRegistry,
    build_intent_registry,
)
from aurarouter.mcp_tools import route_task
from aurarouter.routing import TriageResult, analyze_intent
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    active_analyzer: str = "test-analyzer",
    role_bindings: dict[str, str] | None = None,
    catalog_extras: dict | None = None,
    models: dict | None = None,
    roles: dict | None = None,
) -> ConfigLoader:
    """Build an in-memory ConfigLoader with an analyzer in the catalog."""
    cfg = ConfigLoader(allow_missing=True)
    analyzer_entry: dict = {
        "kind": "analyzer",
        "display_name": "Test Analyzer",
        "analyzer_kind": "intent_triage",
    }
    if role_bindings is not None:
        analyzer_entry["role_bindings"] = role_bindings

    catalog: dict = {active_analyzer: analyzer_entry}
    if catalog_extras:
        catalog.update(catalog_extras)

    cfg.config = {
        "system": {"active_analyzer": active_analyzer},
        "catalog": catalog,
        "models": models or {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": roles or {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    return cfg


def _make_fabric(cfg: ConfigLoader) -> ComputeFabric:
    return ComputeFabric(cfg)


# ======================================================================
# T9.1.1: Registration -> Classification -> Routing
# ======================================================================


class TestRegistrationClassificationRouting:
    """Register a custom analyzer with role_bindings, build the registry,
    mock analyze_intent() to return the custom intent, verify route_task()
    routes to the expected role chain."""

    def test_custom_intent_routes_to_target_role(self):
        """Register sar_processing -> reasoning, mock classification,
        verify route_task dispatches to reasoning role."""
        cfg = _make_config(
            role_bindings={"sar_processing": "reasoning"},
        )
        fabric = _make_fabric(cfg)

        # Build registry and verify custom intent is present
        registry = build_intent_registry(cfg)
        assert registry.get_by_name("sar_processing") is not None
        assert registry.resolve_role("sar_processing") == "reasoning"

        # Mock fabric.execute to capture which role is invoked.
        # The custom intent goes through the complex path (generate_plan),
        # so we mock that too.
        mock_result = GenerateResult(text="SAR result")
        with patch.object(fabric, "execute", return_value=mock_result) as mock_exec:
            with patch(
                "aurarouter.mcp_tools.analyze_intent",
                return_value=TriageResult(intent="sar_processing", complexity=7),
            ):
                with patch(
                    "aurarouter.mcp_tools.generate_plan",
                    return_value=["Process SAR imagery"],
                ):
                    output = route_task(
                        fabric, None, task="Process SAR imagery", config=cfg,
                    )

        assert "SAR result" in output
        # Verify execute was called with the "reasoning" role (from sar_processing binding)
        calls = mock_exec.call_args_list
        assert any(c.args[0] == "reasoning" for c in calls), (
            f"Expected 'reasoning' role call, got: {calls}"
        )

    def test_multiple_custom_intents_resolve_correctly(self):
        """Multiple role_bindings each resolve to their declared target."""
        cfg = _make_config(
            role_bindings={
                "sar_processing": "reasoning",
                "geoint_analysis": "coding",
                "sensor_review": "coding",
            },
        )
        registry = build_intent_registry(cfg)

        assert registry.resolve_role("sar_processing") == "reasoning"
        assert registry.resolve_role("geoint_analysis") == "coding"
        assert registry.resolve_role("sensor_review") == "coding"
        # Built-in intents should still be present
        assert registry.resolve_role("DIRECT") == "coding"
        assert registry.resolve_role("COMPLEX_REASONING") == "reasoning"


# ======================================================================
# T9.1.2: Explicit intent override
# ======================================================================


class TestExplicitIntentOverride:
    """Call route_task(intent="sar_processing"), verify classification is
    skipped and routing goes to the correct role."""

    def test_explicit_intent_skips_classification(self):
        cfg = _make_config(role_bindings={"sar_processing": "reasoning"})
        fabric = _make_fabric(cfg)

        mock_result = GenerateResult(text="Direct SAR result")
        with patch.object(fabric, "execute", return_value=mock_result) as mock_exec:
            # analyze_intent should NOT be called
            with patch(
                "aurarouter.mcp_tools.analyze_intent",
                side_effect=AssertionError("analyze_intent should not be called"),
            ):
                with patch(
                    "aurarouter.mcp_tools.generate_plan",
                    return_value=["Process SAR"],
                ):
                    output = route_task(
                        fabric,
                        None,
                        task="Process SAR imagery",
                        config=cfg,
                        intent="sar_processing",
                    )

        assert "Direct SAR result" in output
        # Should have been routed to reasoning (sar_processing -> reasoning)
        calls = mock_exec.call_args_list
        assert any(c.args[0] == "reasoning" for c in calls)

    def test_explicit_builtin_intent_works(self):
        """Explicit intent="COMPLEX_REASONING" should route to reasoning."""
        cfg = _make_config(role_bindings={})
        fabric = _make_fabric(cfg)

        mock_result = GenerateResult(text="Complex result")
        with patch.object(fabric, "execute", return_value=mock_result):
            with patch(
                "aurarouter.mcp_tools.analyze_intent",
                side_effect=AssertionError("should not be called for known intent"),
            ):
                with patch(
                    "aurarouter.mcp_tools.generate_plan",
                    return_value=["Design architecture"],
                ):
                    output = route_task(
                        fabric,
                        None,
                        task="Design a system architecture",
                        config=cfg,
                        intent="COMPLEX_REASONING",
                    )

        assert "Complex result" in output


# ======================================================================
# T9.1.3: Unknown intent fallback
# ======================================================================


class TestUnknownIntentFallback:
    """Call route_task(intent="nonexistent"), verify fallback to normal
    classification."""

    def test_unknown_intent_falls_back_to_classification(self):
        cfg = _make_config(role_bindings={"sar_processing": "reasoning"})
        fabric = _make_fabric(cfg)

        mock_result = GenerateResult(text="Classified result")
        with patch.object(fabric, "execute", return_value=mock_result):
            with patch(
                "aurarouter.mcp_tools.analyze_intent",
                return_value=TriageResult(intent="SIMPLE_CODE", complexity=3),
            ) as mock_classify:
                output = route_task(
                    fabric,
                    None,
                    task="Write hello world",
                    config=cfg,
                    intent="nonexistent",
                )

        # analyze_intent SHOULD be called since the intent was not found
        mock_classify.assert_called_once()
        assert output == "Classified result"

    def test_unknown_intent_without_config_falls_back(self):
        """When config is None, explicit unknown intent should still work
        via fallback classification."""
        cfg = _make_config(role_bindings={})
        fabric = _make_fabric(cfg)

        mock_result = GenerateResult(text="Fallback result")
        with patch.object(fabric, "execute", return_value=mock_result):
            with patch(
                "aurarouter.mcp_tools.analyze_intent",
                return_value=TriageResult(intent="DIRECT", complexity=1),
            ):
                output = route_task(
                    fabric,
                    None,
                    task="Hello",
                    config=None,
                    intent="nonexistent",
                )

        assert output == "Fallback result"


# ======================================================================
# T9.1.4: Analyzer switch — registry rebuilds with new intents
# ======================================================================


class TestAnalyzerSwitch:
    """Change active analyzer, verify registry rebuilds with new intents."""

    def test_switching_analyzer_rebuilds_registry(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "system": {"active_analyzer": "analyzer-a"},
            "catalog": {
                "analyzer-a": {
                    "kind": "analyzer",
                    "display_name": "Analyzer A",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {"intent_alpha": "coding"},
                },
                "analyzer-b": {
                    "kind": "analyzer",
                    "display_name": "Analyzer B",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {"intent_beta": "reasoning"},
                },
            },
            "models": {"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
            "roles": {"router": ["m1"], "reasoning": ["m1"], "coding": ["m1"]},
        }

        # Build with analyzer-a active
        reg_a = build_intent_registry(cfg)
        assert reg_a.get_by_name("intent_alpha") is not None
        assert reg_a.get_by_name("intent_beta") is None

        # Switch to analyzer-b
        cfg.set_active_analyzer("analyzer-b")
        reg_b = build_intent_registry(cfg)
        assert reg_b.get_by_name("intent_beta") is not None
        # intent_alpha should NOT be in the new registry (only comes from analyzer-a)
        assert reg_b.get_by_name("intent_alpha") is None

        # Built-in intents should always remain
        assert reg_b.get_by_name("DIRECT") is not None
        assert reg_b.get_by_name("SIMPLE_CODE") is not None
        assert reg_b.get_by_name("COMPLEX_REASONING") is not None

    def test_clearing_analyzer_leaves_only_builtins(self):
        cfg = _make_config(
            role_bindings={"custom_intent": "coding"},
        )
        reg = build_intent_registry(cfg)
        assert reg.get_by_name("custom_intent") is not None

        # Clear the active analyzer
        cfg.set_active_analyzer(None)
        reg_cleared = build_intent_registry(cfg)
        assert reg_cleared.get_by_name("custom_intent") is None
        assert len(reg_cleared.get_all()) == 3  # Only built-ins

    def test_analyzer_with_overlapping_builtin_name_uses_priority(self):
        """If an analyzer re-declares SIMPLE_CODE, it should win (priority 10 > 0)."""
        cfg = _make_config(
            role_bindings={"SIMPLE_CODE": "reasoning"},  # Override built-in
        )
        reg = build_intent_registry(cfg)
        defn = reg.get_by_name("SIMPLE_CODE")
        assert defn is not None
        assert defn.target_role == "reasoning"  # Analyzer override wins
        assert defn.priority == 10
