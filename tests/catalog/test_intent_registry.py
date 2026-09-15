"""Tests for the Intent Registry module (Task Group 1: Phase 5.3)."""

import json
from unittest.mock import patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import (
    IntentDefinition,
    IntentRegistry,
    build_intent_registry,
)
from aurarouter.routing import TriageResult, analyze_intent
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fabric() -> ComputeFabric:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    return ComputeFabric(cfg)


def _make_config_with_analyzer(
    role_bindings: dict[str, str] | None = None,
    analyzer_id: str = "aurarouter-default",
) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    analyzer_entry: dict = {
        "kind": "analyzer",
        "display_name": "Test Analyzer",
        "analyzer_kind": "intent_triage",
    }
    if role_bindings is not None:
        analyzer_entry["role_bindings"] = role_bindings
    cfg.config = {
        "system": {"active_analyzer": analyzer_id},
        "catalog": {analyzer_id: analyzer_entry},
        "models": {
            "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# T1.5a: Built-in intents are always present
# ---------------------------------------------------------------------------


class TestBuiltinIntents:
    def test_builtin_intents_present_on_init(self):
        registry = IntentRegistry()
        names = registry.get_intent_names()
        assert "DIRECT" in names
        assert "SIMPLE_CODE" in names
        assert "COMPLEX_REASONING" in names

    def test_builtin_count(self):
        registry = IntentRegistry()
        assert len(registry.get_all()) == 3


# ---------------------------------------------------------------------------
# T1.5b: register_from_role_bindings
# ---------------------------------------------------------------------------


class TestRegisterFromRoleBindings:
    def test_converts_role_bindings(self):
        registry = IntentRegistry()
        bindings = {
            "sar_processing": "coding",
            "geolocation": "reasoning",
        }
        registry.register_from_role_bindings("my-analyzer", bindings)
        assert registry.get_by_name("sar_processing") is not None
        assert registry.get_by_name("geolocation") is not None
        assert registry.resolve_role("sar_processing") == "coding"
        assert registry.resolve_role("geolocation") == "reasoning"

    def test_source_is_set(self):
        registry = IntentRegistry()
        registry.register_from_role_bindings("my-analyzer", {"custom": "coding"})
        defn = registry.get_by_name("custom")
        assert defn is not None
        assert defn.source == "my-analyzer"
        assert defn.priority == 10

    def test_re_registration_replaces(self):
        registry = IntentRegistry()
        registry.register_from_role_bindings("a", {"x": "coding"})
        assert registry.resolve_role("x") == "coding"
        registry.register_from_role_bindings("a", {"x": "reasoning"})
        assert registry.resolve_role("x") == "reasoning"

    def test_builtin_override_by_analyzer(self):
        """Analyzer role_bindings with same name as builtin should win (higher priority)."""
        registry = IntentRegistry()
        registry.register_from_role_bindings(
            "my-analyzer", {"SIMPLE_CODE": "reasoning"}
        )
        assert registry.resolve_role("SIMPLE_CODE") == "reasoning"


# ---------------------------------------------------------------------------
# T1.5c: unregister_by_source
# ---------------------------------------------------------------------------


class TestUnregisterBySource:
    def test_removes_only_source_intents(self):
        registry = IntentRegistry()
        registry.register_from_role_bindings("analyzer-a", {"custom_a": "coding"})
        registry.register_from_role_bindings("analyzer-b", {"custom_b": "reasoning"})
        assert registry.get_by_name("custom_a") is not None
        assert registry.get_by_name("custom_b") is not None

        registry.unregister_by_source("analyzer-a")
        assert registry.get_by_name("custom_a") is None
        assert registry.get_by_name("custom_b") is not None

    def test_never_removes_builtins(self):
        registry = IntentRegistry()
        registry.unregister_by_source("builtin")
        assert len(registry.get_all()) == 3


# ---------------------------------------------------------------------------
# T1.5d: build_classifier_choices
# ---------------------------------------------------------------------------


class TestBuildClassifierChoices:
    def test_includes_builtin_intents(self):
        registry = IntentRegistry()
        choices = registry.build_classifier_choices()
        assert "DIRECT" in choices
        assert "SIMPLE_CODE" in choices
        assert "COMPLEX_REASONING" in choices

    def test_includes_custom_intents(self):
        registry = IntentRegistry()
        registry.register_from_role_bindings("a", {"sar_processing": "coding"})
        choices = registry.build_classifier_choices()
        assert "sar_processing" in choices


# ---------------------------------------------------------------------------
# T1.5e: resolve_role
# ---------------------------------------------------------------------------


class TestResolveRole:
    def test_builtin_roles(self):
        registry = IntentRegistry()
        assert registry.resolve_role("DIRECT") == "coding"
        assert registry.resolve_role("SIMPLE_CODE") == "coding"
        assert registry.resolve_role("COMPLEX_REASONING") == "reasoning"

    def test_custom_role(self):
        registry = IntentRegistry()
        registry.register_from_role_bindings("a", {"custom": "reviewer"})
        assert registry.resolve_role("custom") == "reviewer"

    def test_unknown_returns_none(self):
        registry = IntentRegistry()
        assert registry.resolve_role("nonexistent") is None


# ---------------------------------------------------------------------------
# T1.5f: Priority conflict resolution
# ---------------------------------------------------------------------------


class TestPriorityConflict:
    def test_higher_priority_wins(self):
        registry = IntentRegistry()
        registry.register(
            IntentDefinition(
                name="shared",
                description="from A",
                target_role="coding",
                source="a",
                priority=5,
            )
        )
        registry.register(
            IntentDefinition(
                name="shared",
                description="from B",
                target_role="reasoning",
                source="b",
                priority=15,
            )
        )
        assert registry.resolve_role("shared") == "reasoning"

    def test_lower_priority_does_not_replace(self):
        registry = IntentRegistry()
        registry.register(
            IntentDefinition(
                name="shared",
                description="high",
                target_role="reasoning",
                source="a",
                priority=20,
            )
        )
        registry.register(
            IntentDefinition(
                name="shared",
                description="low",
                target_role="coding",
                source="b",
                priority=5,
            )
        )
        assert registry.resolve_role("shared") == "reasoning"

    def test_equal_priority_new_wins(self):
        registry = IntentRegistry()
        registry.register(
            IntentDefinition(
                name="shared",
                description="first",
                target_role="coding",
                source="a",
                priority=10,
            )
        )
        registry.register(
            IntentDefinition(
                name="shared",
                description="second",
                target_role="reasoning",
                source="b",
                priority=10,
            )
        )
        assert registry.resolve_role("shared") == "reasoning"


# ---------------------------------------------------------------------------
# T1.5g: analyze_intent with registry
# ---------------------------------------------------------------------------


class TestAnalyzeIntentWithRegistry:
    def test_custom_intent_recognized(self):
        fabric = _make_fabric()
        registry = IntentRegistry()
        registry.register_from_role_bindings("a", {"sar_processing": "coding"})

        with patch.object(
            fabric,
            "execute",
            return_value=GenerateResult(
                text=json.dumps({"intent": "sar_processing", "complexity": 7})
            ),
        ):
            result = analyze_intent(fabric, "process SAR image", intent_registry=registry)
            assert result.intent == "sar_processing"
            assert result.complexity == 7

    def test_unrecognized_intent_falls_back_to_direct(self):
        fabric = _make_fabric()
        registry = IntentRegistry()

        with patch.object(
            fabric,
            "execute",
            return_value=GenerateResult(
                text=json.dumps({"intent": "UNKNOWN_THING", "complexity": 3})
            ),
        ):
            result = analyze_intent(fabric, "do something", intent_registry=registry)
            assert result.intent == "DIRECT"

    def test_builtin_intent_still_works_with_registry(self):
        fabric = _make_fabric()
        registry = IntentRegistry()

        with patch.object(
            fabric,
            "execute",
            return_value=GenerateResult(
                text=json.dumps({"intent": "COMPLEX_REASONING", "complexity": 8})
            ),
        ):
            result = analyze_intent(fabric, "design system", intent_registry=registry)
            assert result.intent == "COMPLEX_REASONING"
            assert result.complexity == 8


# ---------------------------------------------------------------------------
# T1.5h: analyze_intent without registry (legacy)
# ---------------------------------------------------------------------------


class TestAnalyzeIntentLegacy:
    def test_legacy_simple_code(self):
        fabric = _make_fabric()
        with patch.object(
            fabric,
            "execute",
            return_value=GenerateResult(
                text=json.dumps({"intent": "SIMPLE_CODE", "complexity": 3})
            ),
        ):
            result = analyze_intent(fabric, "add numbers")
            assert result.intent == "SIMPLE_CODE"
            assert result.complexity == 3

    def test_legacy_fallback_on_error(self):
        fabric = _make_fabric()
        with patch.object(fabric, "execute", return_value=GenerateResult(text="bad")):
            result = analyze_intent(fabric, "anything")
            assert result.intent == "DIRECT"
            assert result.complexity == 1


# ---------------------------------------------------------------------------
# T1.5i: build_intent_registry convenience function
# ---------------------------------------------------------------------------


class TestBuildIntentRegistry:
    def test_with_active_analyzer(self):
        cfg = _make_config_with_analyzer(
            role_bindings={
                "simple_code": "coding",
                "complex_reasoning": "reasoning",
                "sar_chip": "coding",
            }
        )
        registry = build_intent_registry(cfg)
        assert registry.get_by_name("sar_chip") is not None
        assert registry.resolve_role("sar_chip") == "coding"
        # Builtins still present
        assert registry.get_by_name("DIRECT") is not None

    def test_without_active_analyzer(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        registry = build_intent_registry(cfg)
        assert len(registry.get_all()) == 3  # only builtins

    def test_analyzer_without_role_bindings(self):
        cfg = _make_config_with_analyzer(role_bindings=None)
        registry = build_intent_registry(cfg)
        assert len(registry.get_all()) == 3  # only builtins


# ---------------------------------------------------------------------------
# T1.5j: Explicit intent override in route_task
# ---------------------------------------------------------------------------


class TestRouteTaskIntentOverride:
    def test_explicit_intent_skips_classification(self):
        from aurarouter.mcp_tools import route_task

        fabric = _make_fabric()
        cfg = _make_config_with_analyzer(
            role_bindings={
                "simple_code": "coding",
                "complex_reasoning": "reasoning",
            }
        )

        with patch.object(
            fabric,
            "execute",
            return_value=GenerateResult(text="done"),
        ) as mock_exec:
            result = route_task(
                fabric,
                None,
                task="hello",
                config=cfg,
                intent="simple_code",
            )
            # Should NOT have called execute with a CLASSIFY prompt --
            # the first call should be the actual task execution.
            first_call_prompt = mock_exec.call_args_list[0][0][1]
            assert "CLASSIFY" not in first_call_prompt

    def test_explicit_unknown_intent_falls_back_to_classification(self):
        from aurarouter.mcp_tools import route_task

        fabric = _make_fabric()
        cfg = _make_config_with_analyzer(
            role_bindings={
                "simple_code": "coding",
                "complex_reasoning": "reasoning",
            }
        )

        # First call is analyze_intent (classification), second is execution.
        with patch.object(
            fabric,
            "execute",
            return_value=GenerateResult(
                text=json.dumps({"intent": "DIRECT", "complexity": 1})
            ),
        ) as mock_exec:
            result = route_task(
                fabric,
                None,
                task="hello",
                config=cfg,
                intent="totally_unknown_intent",
            )
            # Should have called classify since the intent was unknown.
            first_call_prompt = mock_exec.call_args_list[0][0][1]
            assert "CLASSIFY" in first_call_prompt

    def test_explicit_intent_without_config_falls_back(self):
        """When no config is provided, intent override is ignored (no registry)."""
        from aurarouter.mcp_tools import route_task

        fabric = _make_fabric()

        with patch.object(
            fabric,
            "execute",
            return_value=GenerateResult(
                text=json.dumps({"intent": "DIRECT", "complexity": 1})
            ),
        ) as mock_exec:
            result = route_task(
                fabric,
                None,
                task="hello",
                intent="SIMPLE_CODE",
            )
            # Without config, no registry is built, so classification runs.
            first_call_prompt = mock_exec.call_args_list[0][0][1]
            assert "CLASSIFY" in first_call_prompt
