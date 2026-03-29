"""Backwards compatibility tests for the intent system (Task Group 9.3).

Validates that the intent system degrades gracefully when:
- No registry is configured
- No custom intents are declared
- Models lack supported_intents
- No advisors are registered
- No --intent flag is provided on CLI

Also includes cross-TG conflict verification tests (T9.4) that exercise
both TGs' changes to shared files without modifying source.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.analyzer_schema import (
    AnalyzerSpecValidation,
    extract_declared_intents,
    validate_analyzer_spec,
)
from aurarouter.catalog_model import ArtifactKind, CatalogArtifact
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import (
    IntentDefinition,
    IntentRegistry,
    build_intent_registry,
)
from aurarouter.mcp_tools import list_intents, route_task
from aurarouter.routing import TriageResult, analyze_intent
from aurarouter.savings.models import GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config_minimal() -> ConfigLoader:
    """Minimal config with no catalog, no analyzer, no system section."""
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
    return cfg


def _make_fabric(cfg: ConfigLoader) -> ComputeFabric:
    return ComputeFabric(cfg)


# ======================================================================
# T9.3.1: No registry — analyze_intent returns one of 3 built-in intents
# ======================================================================


class TestNoRegistry:
    """analyze_intent() without a registry should return one of the 3
    built-in intents: DIRECT, SIMPLE_CODE, COMPLEX_REASONING."""

    def test_analyze_intent_no_registry_returns_builtin(self):
        cfg = _make_config_minimal()
        fabric = _make_fabric(cfg)

        # Mock fabric.execute to return a JSON classification
        mock_result = GenerateResult(text='{"intent": "SIMPLE_CODE", "complexity": 3}')
        with patch.object(fabric, "execute", return_value=mock_result):
            result = analyze_intent(fabric, "Write a hello world")

        assert result.intent in ("DIRECT", "SIMPLE_CODE", "COMPLEX_REASONING")
        assert result.intent == "SIMPLE_CODE"

    def test_analyze_intent_no_registry_fallback_on_error(self):
        """If classification fails, should fall back to DIRECT."""
        cfg = _make_config_minimal()
        fabric = _make_fabric(cfg)

        mock_result = GenerateResult(text="not json")
        with patch.object(fabric, "execute", return_value=mock_result):
            result = analyze_intent(fabric, "Hello")

        assert result.intent == "DIRECT"
        assert result.complexity == 1

    def test_analyze_intent_no_registry_none_result(self):
        """If fabric.execute returns None, should fall back to DIRECT."""
        cfg = _make_config_minimal()
        fabric = _make_fabric(cfg)

        with patch.object(fabric, "execute", return_value=None):
            result = analyze_intent(fabric, "Hello")

        assert result.intent == "DIRECT"


# ======================================================================
# T9.3.2: No custom intents — registry has only built-ins
# ======================================================================


class TestNoCustomIntents:
    """Analyzer with no role_bindings should produce a registry with only
    the 3 built-in intents."""

    def test_analyzer_without_role_bindings(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "system": {"active_analyzer": "bare-analyzer"},
            "catalog": {
                "bare-analyzer": {
                    "kind": "analyzer",
                    "display_name": "Bare",
                    "analyzer_kind": "intent_triage",
                    # No role_bindings at all
                },
            },
            "models": {"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
            "roles": {"router": ["m1"], "coding": ["m1"]},
        }
        registry = build_intent_registry(cfg)
        names = registry.get_intent_names()
        assert len(names) == 3
        assert set(names) == {"DIRECT", "SIMPLE_CODE", "COMPLEX_REASONING"}

    def test_analyzer_with_empty_role_bindings(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "system": {"active_analyzer": "empty-analyzer"},
            "catalog": {
                "empty-analyzer": {
                    "kind": "analyzer",
                    "display_name": "Empty",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {},
                },
            },
            "models": {"m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"}},
            "roles": {"router": ["m1"], "coding": ["m1"]},
        }
        registry = build_intent_registry(cfg)
        assert len(registry.get_all()) == 3

    def test_no_active_analyzer_at_all(self):
        """No system.active_analyzer set -> only built-ins."""
        cfg = _make_config_minimal()
        registry = build_intent_registry(cfg)
        assert len(registry.get_all()) == 3


# ======================================================================
# T9.3.3: No supported_intents on models — full chain returned
# ======================================================================


class TestNoSupportedIntentsOnModels:
    """filter_chain_by_intent() should return the full chain when no models
    declare supported_intents."""

    def test_full_chain_when_no_models_declare_intents(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
                "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
            },
            "roles": {"coding": ["m1", "m2"]},
        }
        fabric = ComputeFabric(cfg)

        chain = ["m1", "m2"]
        result = fabric.filter_chain_by_intent(chain, "sar_processing")
        # No models declare supported_intents -> return full chain
        assert result == chain

    def test_full_chain_when_catalog_models_have_no_intents(self):
        """Catalog models without supported_intents -> full chain."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "catalog": {
                "cm1": {"kind": "model", "display_name": "CM1"},
                "cm2": {"kind": "model", "display_name": "CM2"},
            },
            "models": {},
            "roles": {"coding": ["cm1", "cm2"]},
        }
        fabric = ComputeFabric(cfg)
        result = fabric.filter_chain_by_intent(["cm1", "cm2"], "anything")
        assert result == ["cm1", "cm2"]

    def test_unknown_model_ids_kept_in_chain(self):
        """Model IDs not found in config are kept (don't break chains)."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(cfg)
        result = fabric.filter_chain_by_intent(["unknown1", "unknown2"], "intent")
        assert result == ["unknown1", "unknown2"]


# ======================================================================
# T9.3.4: No advisors — original chain returned
# ======================================================================


class TestNoAdvisors:
    """consult_routing_advisors() should return the original chain when
    no advisors are registered."""

    def test_no_advisor_registry(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(cfg)

        chain = ["m1", "m2"]
        result = fabric.consult_routing_advisors("coding", chain)
        assert result == chain

    def test_no_advisor_registry_with_intent(self):
        """Even with intent, no advisors -> original chain."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(cfg)

        chain = ["m1", "m2"]
        result = fabric.consult_routing_advisors("coding", chain, intent="sar_processing")
        assert result == chain

    def test_empty_advisor_registry(self):
        """An empty advisor registry also returns original chain."""
        from aurarouter.mcp_client.registry import McpClientRegistry

        registry = McpClientRegistry()
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {"models": {}, "roles": {}}
        fabric = ComputeFabric(cfg, routing_advisors=registry)

        result = fabric.consult_routing_advisors("coding", ["m1"])
        assert result == ["m1"]


# ======================================================================
# T9.3.5: No --intent flag — CLI auto-classifies as before
# ======================================================================


class TestNoIntentFlag:
    """CLI run without --intent should auto-classify via analyze_intent."""

    def test_route_task_no_intent_auto_classifies(self):
        """route_task called without intent= should call analyze_intent."""
        cfg = _make_config_minimal()
        fabric = _make_fabric(cfg)

        mock_result = GenerateResult(text="Classified output")
        with patch.object(fabric, "execute", return_value=mock_result):
            with patch(
                "aurarouter.mcp_tools.analyze_intent",
                return_value=TriageResult(intent="SIMPLE_CODE", complexity=3),
            ) as mock_classify:
                output = route_task(
                    fabric, None, task="Write hello world",
                )

        mock_classify.assert_called_once()
        assert output == "Classified output"

    def test_route_task_no_config_no_intent(self):
        """route_task with no config and no intent still works via
        built-in classification."""
        cfg = _make_config_minimal()
        fabric = _make_fabric(cfg)

        mock_result = GenerateResult(text="Basic result")
        with patch.object(fabric, "execute", return_value=mock_result):
            with patch(
                "aurarouter.mcp_tools.analyze_intent",
                return_value=TriageResult(intent="DIRECT", complexity=1),
            ):
                output = route_task(fabric, None, task="Hi")

        assert output == "Basic result"


# ======================================================================
# T9.4: Cross-TG shared file conflict verification
# ======================================================================


class TestSharedFileConflicts:
    """Exercise both TGs' changes to shared files to verify they work
    together without modifying source."""

    # -- catalog_model.py (TG2 + TG6) --

    def test_catalog_model_supported_intents_and_spec_coexist(self):
        """CatalogArtifact supports both supported_intents (TG6) and
        spec fields like role_bindings (TG2) simultaneously."""
        data = {
            "kind": "analyzer",
            "display_name": "Multi-TG Artifact",
            "supported_intents": ["sar_processing"],
            "role_bindings": {"intent_a": "coding"},
            "analyzer_kind": "intent_triage",
        }
        artifact = CatalogArtifact.from_dict("multi-tg", data)
        assert artifact.supported_intents == ["sar_processing"]
        assert artifact.spec.get("role_bindings") == {"intent_a": "coding"}

        # Round-trip
        serialized = artifact.to_dict()
        assert serialized["supported_intents"] == ["sar_processing"]
        assert serialized["role_bindings"] == {"intent_a": "coding"}

    # -- config.py (TG2 + TG6) --

    def test_config_catalog_query_intents_and_supported_intents(self):
        """catalog_query supports both intents= (TG2, analyzer role_bindings)
        and supported_intents= (TG6, model field) filters."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "catalog": {
                "analyzer-1": {
                    "kind": "analyzer",
                    "display_name": "A1",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {"sar_processing": "reasoning"},
                },
                "model-1": {
                    "kind": "model",
                    "display_name": "M1",
                    "supported_intents": ["sar_processing"],
                },
                "model-2": {
                    "kind": "model",
                    "display_name": "M2",
                    "supported_intents": ["general"],
                },
            },
        }

        # TG2 filter: intents= filters analyzers by role_bindings keys
        analyzer_results = cfg.catalog_query(intents=["sar_processing"])
        analyzer_ids = [r["artifact_id"] for r in analyzer_results]
        assert "analyzer-1" in analyzer_ids
        assert "model-1" not in analyzer_ids  # intents= is analyzer-only

        # TG6 filter: supported_intents= filters by the model field
        model_results = cfg.catalog_query(supported_intents=["sar_processing"])
        model_ids = [r["artifact_id"] for r in model_results]
        assert "model-1" in model_ids
        assert "model-2" not in model_ids

    def test_config_auto_join_with_and_without_intent_registry(self):
        """auto_join_roles works both with tags (basic) and with
        intent_registry + supported_intents (TG6 extension)."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "roles": {"coding": ["existing"], "reasoning": ["existing"]},
            "models": {},
        }

        # Basic tag-based auto-join
        roles_joined = cfg.auto_join_roles("new-model", tags=["coding"])
        assert "coding" in roles_joined

        # Intent-based auto-join via registry
        registry = IntentRegistry()
        registry.register(IntentDefinition(
            name="sar_processing",
            description="SAR",
            target_role="reasoning",
            source="test",
            priority=10,
        ))

        cfg2 = ConfigLoader(allow_missing=True)
        cfg2.config = {
            "roles": {"coding": ["existing"], "reasoning": ["existing"]},
            "models": {},
        }
        roles_joined2 = cfg2.auto_join_roles(
            "intent-model",
            tags=[],
            intent_registry=registry,
            supported_intents=["sar_processing"],
        )
        assert "reasoning" in roles_joined2

    # -- config.py: catalog_get declared_intents enrichment --

    def test_catalog_get_declared_intents_enrichment(self):
        """catalog_get enriches analyzer entries with declared_intents
        from role_bindings (TG2 extract_declared_intents + TG6 enrichment)."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "catalog": {
                "my-analyzer": {
                    "kind": "analyzer",
                    "display_name": "My Analyzer",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {"alpha": "coding", "beta": "reasoning"},
                },
            },
        }
        data = cfg.catalog_get("my-analyzer")
        assert data is not None
        assert set(data["declared_intents"]) == {"alpha", "beta"}

    def test_catalog_get_declared_intents_method(self):
        """catalog_get_declared_intents() returns intent names from
        analyzer role_bindings."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "catalog": {
                "my-analyzer": {
                    "kind": "analyzer",
                    "display_name": "My Analyzer",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {"x": "coding", "y": "reasoning"},
                },
            },
        }
        intents = cfg.catalog_get_declared_intents("my-analyzer")
        assert set(intents) == {"x", "y"}

    # -- mcp_tools.py (TG1 + TG7): intent param and list_intents coexist --

    def test_route_task_intent_param_and_list_intents_coexist(self):
        """Both route_task(intent=...) (TG1) and list_intents() (TG7)
        can be called on the same config without conflict."""
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "system": {"active_analyzer": "test-analyzer"},
            "catalog": {
                "test-analyzer": {
                    "kind": "analyzer",
                    "display_name": "Test",
                    "analyzer_kind": "intent_triage",
                    "role_bindings": {"custom_intent": "coding"},
                },
            },
            "models": {
                "m1": {"provider": "ollama", "model_name": "t", "endpoint": "http://x"},
            },
            "roles": {"router": ["m1"], "coding": ["m1"], "reasoning": ["m1"]},
        }
        fabric = ComputeFabric(cfg)

        # list_intents should work
        intents_json = list_intents(cfg)
        data = json.loads(intents_json)
        names = [i["name"] for i in data["intents"]]
        assert "custom_intent" in names

        # route_task with intent= should work (custom intents go through
        # the complex path, so mock generate_plan too)
        mock_result = GenerateResult(text="routed")
        with patch.object(fabric, "execute", return_value=mock_result):
            with patch(
                "aurarouter.mcp_tools.generate_plan",
                return_value=["step one"],
            ):
                output = route_task(
                    fabric, None, task="test", config=cfg, intent="custom_intent",
                )
        assert "routed" in output

    # -- analyzer_schema.py: validate + extract coexist --

    def test_validate_and_extract_declared_intents_consistent(self):
        """validate_analyzer_spec().declared_intents and
        extract_declared_intents() return the same intents."""
        spec = {
            "analyzer_kind": "intent_triage",
            "role_bindings": {
                "sar_processing": "reasoning",
                "geoint": "coding",
            },
        }
        validation = validate_analyzer_spec(spec)
        extracted = extract_declared_intents(spec)

        assert set(validation.declared_intents) == set(extracted)
        assert set(validation.declared_intents) == {"sar_processing", "geoint"}

    # -- contracts coexist with intent system --

    def test_auracode_contract_intents_register_correctly(self):
        """AuraCode contract intents can be registered in the registry."""
        from aurarouter.contracts.auracode import (
            AURACODE_INTENTS,
            create_auracode_analyzer_spec,
        )

        registry = IntentRegistry()
        registry.register_from_role_bindings("auracode", AURACODE_INTENTS)

        for intent_name, target_role in AURACODE_INTENTS.items():
            defn = registry.get_by_name(intent_name)
            assert defn is not None, f"Missing intent: {intent_name}"
            assert defn.target_role == target_role

    def test_auraxlm_contract_spec_validates(self):
        """AuraXLM contract spec validates without errors."""
        from aurarouter.contracts.auraxlm import AURAXLM_ANALYZER_SPEC

        result = validate_analyzer_spec(AURAXLM_ANALYZER_SPEC)
        assert result.valid is True
        assert len(result.errors) == 0


# ======================================================================
# T9.4 (extra): Import compatibility
# ======================================================================


class TestImportCompatibility:
    """Verify all intent-related modules can be imported without circular
    dependency issues."""

    def test_all_intent_imports(self):
        import aurarouter.intent_registry
        import aurarouter.analyzer_schema
        import aurarouter.contracts.auracode
        import aurarouter.contracts.auraxlm
        import aurarouter.broker
        import aurarouter.mcp_tools
        import aurarouter.routing
        import aurarouter.catalog_model
        # If we get here, all imports succeeded

    def test_intent_registry_from_config_import(self):
        """build_intent_registry can be imported and used from config context."""
        from aurarouter.intent_registry import build_intent_registry
        from aurarouter.config import ConfigLoader

        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {}
        registry = build_intent_registry(cfg)
        assert len(registry.get_all()) == 3  # Built-ins only
