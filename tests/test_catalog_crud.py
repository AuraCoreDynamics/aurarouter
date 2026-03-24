"""Tests for ConfigLoader catalog CRUD methods."""

import yaml
import pytest
from pathlib import Path

from aurarouter.config import ConfigLoader


@pytest.fixture
def cfg(tmp_path: Path) -> ConfigLoader:
    """Config with models and an empty catalog section."""
    config_content = {
        "system": {"log_level": "INFO"},
        "models": {
            "m1": {"provider": "ollama", "model_name": "qwen", "tags": ["local"]},
            "m2": {"provider": "openapi", "model_name": "gpt", "tags": ["cloud"]},
        },
        "roles": {
            "coding": ["m1"],
            "reasoning": ["m2"],
        },
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    return ConfigLoader(config_path=str(p))


# ------------------------------------------------------------------
# catalog_set / catalog_get
# ------------------------------------------------------------------


class TestCatalogSetGet:
    def test_set_and_get_model(self, cfg):
        cfg.catalog_set("new-model", {"kind": "model", "display_name": "New Model"})
        result = cfg.catalog_get("new-model")
        assert result is not None
        assert result["kind"] == "model"
        assert result["display_name"] == "New Model"

    def test_get_returns_none_for_missing(self, cfg):
        assert cfg.catalog_get("nonexistent") is None

    def test_get_falls_back_to_legacy_models(self, cfg):
        """catalog_get should find entries in config['models'] as kind=model."""
        result = cfg.catalog_get("m1")
        assert result is not None
        assert result["kind"] == "model"
        assert result["provider"] == "ollama"

    def test_catalog_entry_takes_precedence_over_legacy(self, cfg):
        """If same ID in both catalog and models, catalog wins."""
        cfg.catalog_set("m1", {"kind": "service", "display_name": "Override"})
        result = cfg.catalog_get("m1")
        assert result["kind"] == "service"
        assert result["display_name"] == "Override"

    def test_get_returns_copy(self, cfg):
        cfg.catalog_set("x", {"kind": "model", "display_name": "X"})
        result = cfg.catalog_get("x")
        result["INJECTED"] = True
        assert "INJECTED" not in cfg.catalog_get("x")


# ------------------------------------------------------------------
# Parameterized set/get round-trip for all three kinds
# ------------------------------------------------------------------


class TestCatalogSetGetAllKinds:
    @pytest.mark.parametrize("kind", ["model", "service", "analyzer"])
    def test_catalog_set_get_round_trip(self, kind, cfg):
        """Set and get works for all three artifact kinds."""
        artifact_id = f"test-{kind}"
        data = {
            "kind": kind,
            "display_name": f"Test {kind.title()}",
            "provider": "test-provider",
            "tags": ["test-tag"],
        }
        cfg.catalog_set(artifact_id, data)

        result = cfg.catalog_get(artifact_id)
        assert result is not None
        assert result["kind"] == kind
        assert result["display_name"] == f"Test {kind.title()}"
        assert result["provider"] == "test-provider"
        assert result["tags"] == ["test-tag"]


# ------------------------------------------------------------------
# catalog_list
# ------------------------------------------------------------------


class TestCatalogList:
    def test_list_all_includes_catalog_and_legacy(self, cfg):
        cfg.catalog_set("svc1", {"kind": "service", "display_name": "S1"})
        ids = cfg.catalog_list()
        assert "svc1" in ids
        assert "m1" in ids
        assert "m2" in ids

    def test_list_filter_model(self, cfg):
        cfg.catalog_set("svc1", {"kind": "service", "display_name": "S1"})
        cfg.catalog_set("a1", {"kind": "analyzer", "display_name": "A1"})
        ids = cfg.catalog_list(kind="model")
        assert "m1" in ids
        assert "m2" in ids
        assert "svc1" not in ids
        assert "a1" not in ids

    def test_list_filter_service(self, cfg):
        cfg.catalog_set("svc1", {"kind": "service", "display_name": "S1"})
        ids = cfg.catalog_list(kind="service")
        assert "svc1" in ids
        assert "m1" not in ids

    def test_list_filter_analyzer(self, cfg):
        cfg.catalog_set("a1", {"kind": "analyzer", "display_name": "A1"})
        ids = cfg.catalog_list(kind="analyzer")
        assert "a1" in ids
        assert "m1" not in ids

    def test_no_duplicates(self, cfg):
        """If an ID is in both catalog and models, it should appear once."""
        cfg.catalog_set("m1", {"kind": "model", "display_name": "Catalog M1"})
        ids = cfg.catalog_list()
        assert ids.count("m1") == 1


# ------------------------------------------------------------------
# catalog_remove
# ------------------------------------------------------------------


class TestCatalogRemove:
    def test_remove_existing(self, cfg):
        cfg.catalog_set("x", {"kind": "model", "display_name": "X"})
        assert cfg.catalog_remove("x") is True
        assert cfg.catalog_get("x") is None

    def test_remove_nonexistent(self, cfg):
        assert cfg.catalog_remove("nonexistent") is False

    def test_remove_does_not_affect_legacy_models(self, cfg):
        """catalog_remove only removes from config['catalog'], not models."""
        assert cfg.catalog_remove("m1") is False
        assert cfg.catalog_get("m1") is not None  # still in models


# ------------------------------------------------------------------
# catalog_query
# ------------------------------------------------------------------


class TestCatalogQuery:
    def test_query_all(self, cfg):
        cfg.catalog_set("a1", {
            "kind": "analyzer",
            "display_name": "A1",
            "tags": ["ai"],
            "capabilities": ["code"],
            "provider": "aurarouter",
        })
        results = cfg.catalog_query()
        ids = [r["artifact_id"] for r in results]
        assert "a1" in ids
        assert "m1" in ids

    def test_query_by_kind(self, cfg):
        cfg.catalog_set("a1", {"kind": "analyzer", "display_name": "A1"})
        results = cfg.catalog_query(kind="analyzer")
        assert len(results) == 1
        assert results[0]["artifact_id"] == "a1"

    def test_query_by_tags(self, cfg):
        results = cfg.catalog_query(tags=["local"])
        ids = [r["artifact_id"] for r in results]
        assert "m1" in ids
        assert "m2" not in ids

    def test_query_by_capabilities(self, cfg):
        cfg.catalog_set("a1", {
            "kind": "analyzer",
            "display_name": "A1",
            "capabilities": ["code", "reasoning"],
        })
        results = cfg.catalog_query(capabilities=["code"])
        ids = [r["artifact_id"] for r in results]
        assert "a1" in ids

    def test_query_by_provider(self, cfg):
        results = cfg.catalog_query(provider="ollama")
        ids = [r["artifact_id"] for r in results]
        assert "m1" in ids
        assert "m2" not in ids

    def test_query_combined_filters(self, cfg):
        cfg.catalog_set("a1", {
            "kind": "analyzer",
            "display_name": "A1",
            "tags": ["ai"],
            "provider": "aurarouter",
        })
        results = cfg.catalog_query(kind="analyzer", provider="aurarouter")
        assert len(results) == 1
        assert results[0]["artifact_id"] == "a1"

    def test_query_enriched_with_artifact_id(self, cfg):
        results = cfg.catalog_query()
        for r in results:
            assert "artifact_id" in r

    def test_query_empty_tags_matches_everything(self, cfg):
        """Empty tags list matches all entries (all() on empty is True)."""
        cfg.catalog_set("a1", {"kind": "analyzer", "display_name": "A1"})
        results = cfg.catalog_query(tags=[])
        ids = [r["artifact_id"] for r in results]
        # All entries should match
        assert "m1" in ids
        assert "m2" in ids
        assert "a1" in ids

    def test_query_multiple_filters(self, cfg):
        """Combine tags AND capabilities AND provider."""
        cfg.catalog_set("special", {
            "kind": "model",
            "display_name": "Special",
            "tags": ["local", "fast"],
            "capabilities": ["code", "chat"],
            "provider": "custom",
        })
        cfg.catalog_set("other", {
            "kind": "model",
            "display_name": "Other",
            "tags": ["local"],
            "capabilities": ["code"],
            "provider": "other",
        })
        results = cfg.catalog_query(
            tags=["local", "fast"],
            capabilities=["code", "chat"],
            provider="custom",
        )
        assert len(results) == 1
        assert results[0]["artifact_id"] == "special"

    def test_query_returns_empty_when_no_matches(self, cfg):
        results = cfg.catalog_query(kind="analyzer")
        assert results == []


# ------------------------------------------------------------------
# get_active_analyzer / set_active_analyzer
# ------------------------------------------------------------------


class TestActiveAnalyzer:
    def test_default_is_none(self, cfg):
        assert cfg.get_active_analyzer() is None

    def test_set_and_get(self, cfg):
        cfg.set_active_analyzer("my-analyzer")
        assert cfg.get_active_analyzer() == "my-analyzer"

    def test_clear(self, cfg):
        cfg.set_active_analyzer("my-analyzer")
        cfg.set_active_analyzer(None)
        assert cfg.get_active_analyzer() is None

    def test_survives_save_reload(self, cfg, tmp_path):
        cfg.set_active_analyzer("test-analyzer")
        target = tmp_path / "saved.yaml"
        cfg.save(path=target)
        reloaded = ConfigLoader(config_path=str(target))
        assert reloaded.get_active_analyzer() == "test-analyzer"
