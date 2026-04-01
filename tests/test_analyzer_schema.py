"""Tests for analyzer spec schema validation and intent declaration.

Covers:
- AnalyzerSpecValidation via validate_analyzer_spec()
- catalog_set() logging of validation warnings
- catalog_get_declared_intents()
- catalog_query(intents=...) filtering
- Non-analyzer artifacts remain unaffected
"""

import logging

import pytest
import yaml
from pathlib import Path

from aurarouter.analyzer_schema import (
    AnalyzerSpecValidation,
    validate_analyzer_spec,
    extract_declared_intents,
)
from aurarouter.config import ConfigLoader


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def cfg(tmp_path: Path) -> ConfigLoader:
    """Config with roles and an empty catalog section."""
    config_content = {
        "system": {"log_level": "INFO"},
        "models": {
            "m1": {"provider": "ollama", "model_name": "qwen", "tags": ["local"]},
        },
        "roles": {
            "coding": ["m1"],
            "reasoning": ["m1"],
            "reviewer": ["m1"],
        },
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    return ConfigLoader(config_path=str(p))


VALID_ANALYZER_SPEC = {
    "kind": "analyzer",
    "display_name": "Test Analyzer",
    "analyzer_kind": "intent_triage",
    "role_bindings": {
        "simple_code": "coding",
        "complex_reasoning": "reasoning",
        "review": "reviewer",
    },
    "capabilities": ["code", "reasoning"],
}


# ------------------------------------------------------------------
# T2.1: validate_analyzer_spec
# ------------------------------------------------------------------


class TestValidateAnalyzerSpec:
    def test_valid_spec_passes(self):
        result = validate_analyzer_spec(VALID_ANALYZER_SPEC)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert sorted(result.declared_intents) == [
            "complex_reasoning",
            "review",
            "simple_code",
        ]

    def test_missing_analyzer_kind_produces_error(self):
        spec = {"role_bindings": {"intent_a": "coding"}}
        result = validate_analyzer_spec(spec)
        assert result.valid is False
        assert any("analyzer_kind" in e for e in result.errors)

    def test_role_bindings_unknown_role_produces_warning(self):
        spec = {
            "analyzer_kind": "custom",
            "role_bindings": {"my_intent": "nonexistent_role"},
        }
        result = validate_analyzer_spec(spec, available_roles=["coding", "reasoning"])
        assert result.valid is True
        assert any("nonexistent_role" in w for w in result.warnings)

    def test_role_bindings_valid_roles_no_warning(self):
        spec = {
            "analyzer_kind": "custom",
            "role_bindings": {"my_intent": "coding"},
        }
        result = validate_analyzer_spec(spec, available_roles=["coding", "reasoning"])
        assert result.valid is True
        assert result.warnings == []

    def test_role_bindings_invalid_key_produces_warning(self):
        spec = {
            "analyzer_kind": "custom",
            "role_bindings": {"not-an-identifier": "coding"},
        }
        result = validate_analyzer_spec(spec)
        assert any("not a valid identifier" in w for w in result.warnings)
        # Invalid key should NOT appear in declared_intents
        assert "not-an-identifier" not in result.declared_intents

    def test_role_bindings_not_a_dict_produces_error(self):
        spec = {"analyzer_kind": "custom", "role_bindings": "wrong"}
        result = validate_analyzer_spec(spec)
        assert any("role_bindings must be a dict" in e for e in result.errors)

    def test_mcp_endpoint_valid_url(self):
        spec = {
            "analyzer_kind": "remote",
            "mcp_endpoint": "http://localhost:9000/mcp",
        }
        result = validate_analyzer_spec(spec)
        assert result.valid is True
        assert result.warnings == []

    def test_mcp_endpoint_invalid_url_produces_warning(self):
        spec = {
            "analyzer_kind": "remote",
            "mcp_endpoint": "not-a-url",
        }
        result = validate_analyzer_spec(spec)
        assert result.valid is False
        assert any("not a valid URL" in e for e in result.errors)

    def test_mcp_endpoint_not_string_produces_error(self):
        spec = {"analyzer_kind": "remote", "mcp_endpoint": 12345}
        result = validate_analyzer_spec(spec)
        assert any("mcp_endpoint must be a string" in e for e in result.errors)

    def test_capabilities_not_list_produces_warning(self):
        spec = {"analyzer_kind": "custom", "capabilities": "code"}
        result = validate_analyzer_spec(spec)
        assert any("capabilities must be a list" in w for w in result.warnings)

    def test_capabilities_non_string_entries_produces_warning(self):
        spec = {"analyzer_kind": "custom", "capabilities": ["code", 42]}
        result = validate_analyzer_spec(spec)
        assert any("must be strings" in w for w in result.warnings)

    def test_no_role_bindings_still_valid(self):
        spec = {"analyzer_kind": "custom"}
        result = validate_analyzer_spec(spec)
        assert result.valid is True
        assert result.declared_intents == []

    def test_available_roles_none_skips_role_check(self):
        spec = {
            "analyzer_kind": "custom",
            "role_bindings": {"intent_a": "any_role"},
        }
        result = validate_analyzer_spec(spec, available_roles=None)
        assert result.valid is True
        assert result.warnings == []


# ------------------------------------------------------------------
# extract_declared_intents
# ------------------------------------------------------------------


class TestExtractDeclaredIntents:
    def test_extracts_keys(self):
        spec = {"role_bindings": {"a": "coding", "b": "reasoning"}}
        assert extract_declared_intents(spec) == ["a", "b"]

    def test_missing_role_bindings_returns_empty(self):
        assert extract_declared_intents({}) == []

    def test_non_dict_role_bindings_returns_empty(self):
        assert extract_declared_intents({"role_bindings": "bad"}) == []


# ------------------------------------------------------------------
# T2.2: catalog_set() validates analyzer specs
# ------------------------------------------------------------------


class TestCatalogSetValidation:
    def test_catalog_set_logs_warnings_for_invalid_spec(self, cfg, caplog):
        bad_data = {
            "kind": "analyzer",
            "display_name": "Bad Analyzer",
            # Missing analyzer_kind — should produce error log
            "role_bindings": {"intent_a": "nonexistent_role"},
        }
        with caplog.at_level(logging.WARNING, logger="AuraRouter.Config"):
            cfg.catalog_set("bad-analyzer", bad_data)

        # Should have logged the missing field error and unknown role warning
        messages = caplog.text
        assert "analyzer_kind" in messages
        assert "nonexistent_role" in messages

    def test_catalog_set_declared_intents_available_via_api(self, cfg):
        cfg.catalog_set("test-analyzer", dict(VALID_ANALYZER_SPEC))
        # declared_intents are computed dynamically, not stored in raw config
        entry = cfg.config["catalog"]["test-analyzer"]
        assert "_declared_intents" not in entry
        # But available via the proper API
        intents = cfg.catalog_get_declared_intents("test-analyzer")
        assert sorted(intents) == ["complex_reasoning", "review", "simple_code"]

    def test_catalog_set_non_analyzer_not_validated(self, cfg, caplog):
        """Non-analyzer artifacts should not trigger validation."""
        model_data = {"kind": "model", "display_name": "M"}
        with caplog.at_level(logging.WARNING, logger="AuraRouter.Config"):
            cfg.catalog_set("model-1", model_data)
        # No validation messages for non-analyzer
        assert "spec error" not in caplog.text
        assert "spec warning" not in caplog.text


# ------------------------------------------------------------------
# T2.2: catalog_get_declared_intents()
# ------------------------------------------------------------------


class TestCatalogGetDeclaredIntents:
    def test_returns_intent_names(self, cfg):
        cfg.catalog_set("test-analyzer", dict(VALID_ANALYZER_SPEC))
        intents = cfg.catalog_get_declared_intents("test-analyzer")
        assert sorted(intents) == [
            "complex_reasoning",
            "review",
            "simple_code",
        ]

    def test_returns_empty_for_non_analyzer(self, cfg):
        cfg.catalog_set("model-1", {"kind": "model", "display_name": "M"})
        assert cfg.catalog_get_declared_intents("model-1") == []

    def test_returns_empty_for_missing_artifact(self, cfg):
        assert cfg.catalog_get_declared_intents("nonexistent") == []

    def test_returns_empty_for_analyzer_without_bindings(self, cfg):
        cfg.catalog_set("bare", {
            "kind": "analyzer",
            "display_name": "Bare",
            "analyzer_kind": "custom",
        })
        assert cfg.catalog_get_declared_intents("bare") == []


# ------------------------------------------------------------------
# T2.3: catalog_query(intents=...) filtering
# ------------------------------------------------------------------


class TestCatalogQueryIntents:
    def test_query_by_intent_returns_matching_analyzer(self, cfg):
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        results = cfg.catalog_query(intents=["simple_code"])
        ids = [r["artifact_id"] for r in results]
        assert "a1" in ids

    def test_query_by_intent_excludes_non_matching(self, cfg):
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        results = cfg.catalog_query(intents=["sar_processing"])
        assert results == []

    def test_query_by_intent_excludes_non_analyzer(self, cfg):
        """Models/services are excluded when intents filter is active."""
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        results = cfg.catalog_query(intents=["simple_code"])
        kinds = {r.get("kind") for r in results}
        assert kinds == {"analyzer"}

    def test_query_by_intent_with_multiple_intents(self, cfg):
        """At least one intent must match (OR semantics)."""
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        results = cfg.catalog_query(intents=["sar_processing", "review"])
        ids = [r["artifact_id"] for r in results]
        assert "a1" in ids

    def test_query_without_intents_still_works(self, cfg):
        """Existing behaviour unchanged when intents is None."""
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        results = cfg.catalog_query()
        ids = [r["artifact_id"] for r in results]
        assert "a1" in ids
        assert "m1" in ids

    def test_query_intents_combined_with_kind(self, cfg):
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        results = cfg.catalog_query(kind="analyzer", intents=["simple_code"])
        assert len(results) == 1
        assert results[0]["artifact_id"] == "a1"


# ------------------------------------------------------------------
# T2.4: declared_intents enrichment in catalog_get / catalog_query
# ------------------------------------------------------------------


class TestDeclaredIntentsEnrichment:
    def test_catalog_get_enriches_analyzer(self, cfg):
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        result = cfg.catalog_get("a1")
        assert "declared_intents" in result
        assert sorted(result["declared_intents"]) == [
            "complex_reasoning",
            "review",
            "simple_code",
        ]

    def test_catalog_get_does_not_enrich_model(self, cfg):
        result = cfg.catalog_get("m1")
        assert result is not None
        assert "declared_intents" not in result

    def test_catalog_query_enriches_analyzer(self, cfg):
        cfg.catalog_set("a1", dict(VALID_ANALYZER_SPEC))
        results = cfg.catalog_query(kind="analyzer")
        assert len(results) == 1
        assert "declared_intents" in results[0]

    def test_catalog_query_does_not_enrich_model(self, cfg):
        results = cfg.catalog_query(kind="model")
        for r in results:
            assert "declared_intents" not in r


# ------------------------------------------------------------------
# Non-analyzer artifacts unaffected
# ------------------------------------------------------------------


class TestNonAnalyzerUnaffected:
    def test_service_artifact_round_trip(self, cfg):
        svc = {
            "kind": "service",
            "display_name": "Grid Service",
            "endpoint": "http://localhost:5000",
        }
        cfg.catalog_set("svc-1", svc)
        result = cfg.catalog_get("svc-1")
        assert result["kind"] == "service"
        assert "declared_intents" not in result

    def test_legacy_model_unaffected(self, cfg):
        result = cfg.catalog_get("m1")
        assert result is not None
        assert result.get("kind") == "model"
        assert "declared_intents" not in result
