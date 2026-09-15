"""Tests for catalog MCP tool functions."""

import json

from aurarouter.config import ConfigLoader
from aurarouter.mcp_tools import (
    catalog_get_artifact,
    catalog_list_artifacts,
    catalog_register_artifact,
    catalog_remove_artifact,
    get_active_analyzer,
    set_active_analyzer,
)


def _make_config(**extra) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "system": {"log_level": "INFO"},
        "models": {
            "m1": {"provider": "ollama", "model_name": "test", "tags": ["local"]},
        },
        "roles": {"coding": ["m1"]},
        **extra,
    }
    return cfg


# ------------------------------------------------------------------
# catalog_list_artifacts
# ------------------------------------------------------------------


class TestCatalogListArtifacts:
    def test_returns_all_kinds(self):
        cfg = _make_config()
        cfg.catalog_set("svc1", {"kind": "service", "display_name": "S1"})
        cfg.catalog_set("a1", {"kind": "analyzer", "display_name": "A1"})

        result = json.loads(catalog_list_artifacts(cfg))
        ids = [r["artifact_id"] for r in result]
        assert "svc1" in ids
        assert "a1" in ids
        assert "m1" in ids  # legacy model

    def test_filter_by_kind(self):
        cfg = _make_config()
        cfg.catalog_set("a1", {"kind": "analyzer", "display_name": "A1"})

        result = json.loads(catalog_list_artifacts(cfg, kind="analyzer"))
        ids = [r["artifact_id"] for r in result]
        assert "a1" in ids
        assert "m1" not in ids



# ------------------------------------------------------------------
# catalog_register_artifact
# ------------------------------------------------------------------


class TestCatalogRegisterArtifact:
    def test_register_model(self):
        cfg = _make_config()
        result = json.loads(catalog_register_artifact(
            cfg, artifact_id="new-model", kind="model", display_name="New Model",
        ))
        assert result["success"] is True
        assert result["artifact_id"] == "new-model"

        # Verify it's in the catalog
        get_result = json.loads(catalog_get_artifact(cfg, "new-model"))
        assert get_result["display_name"] == "New Model"

    def test_register_service(self):
        cfg = _make_config()
        result = json.loads(catalog_register_artifact(
            cfg, artifact_id="svc", kind="service", display_name="Service",
            description="A service",
        ))
        assert result["success"] is True
        assert result["kind"] == "service"

    def test_register_analyzer(self):
        cfg = _make_config()
        result = json.loads(catalog_register_artifact(
            cfg, artifact_id="a1", kind="analyzer", display_name="Analyzer 1",
            capabilities=["code", "reasoning"],
        ))
        assert result["success"] is True

    def test_register_invalid_kind(self):
        cfg = _make_config()
        result = json.loads(catalog_register_artifact(
            cfg, artifact_id="x", kind="invalid", display_name="X",
        ))
        assert "error" in result


# ------------------------------------------------------------------
# catalog_remove_artifact
# ------------------------------------------------------------------


class TestCatalogRemoveArtifact:
    def test_remove_existing(self):
        cfg = _make_config()
        cfg.catalog_set("x", {"kind": "model", "display_name": "X"})
        result = json.loads(catalog_remove_artifact(cfg, "x"))
        assert result["success"] is True

    def test_remove_nonexistent(self):
        cfg = _make_config()
        result = json.loads(catalog_remove_artifact(cfg, "nope"))
        assert "error" in result


# ------------------------------------------------------------------
# catalog_get_artifact
# ------------------------------------------------------------------


class TestCatalogGetArtifact:
    def test_get_existing(self):
        cfg = _make_config()
        cfg.catalog_set("x", {"kind": "model", "display_name": "X"})
        result = json.loads(catalog_get_artifact(cfg, "x"))
        assert result["artifact_id"] == "x"
        assert result["display_name"] == "X"

    def test_get_legacy_model(self):
        cfg = _make_config()
        result = json.loads(catalog_get_artifact(cfg, "m1"))
        assert result["artifact_id"] == "m1"
        assert result["provider"] == "ollama"

    def test_get_missing(self):
        cfg = _make_config()
        result = json.loads(catalog_get_artifact(cfg, "missing"))
        assert "error" in result


# ------------------------------------------------------------------
# set_active_analyzer / get_active_analyzer round-trip
# ------------------------------------------------------------------


class TestActiveAnalyzerTools:
    def test_set_and_get(self):
        cfg = _make_config()
        set_result = json.loads(set_active_analyzer(cfg, "my-analyzer"))
        assert set_result["success"] is True
        assert set_result["active_analyzer"] == "my-analyzer"

        get_result = json.loads(get_active_analyzer(cfg))
        assert get_result["active_analyzer"] == "my-analyzer"

    def test_clear(self):
        cfg = _make_config()
        set_active_analyzer(cfg, "my-analyzer")
        set_active_analyzer(cfg, None)
        get_result = json.loads(get_active_analyzer(cfg))
        assert get_result["active_analyzer"] is None

    def test_default_is_none(self):
        cfg = _make_config()
        get_result = json.loads(get_active_analyzer(cfg))
        assert get_result["active_analyzer"] is None
