"""Backward compatibility tests — old configs work without migration.

Verifies that configs containing only legacy sections (models, roles,
grid_services) still function correctly through the ConfigLoader API.
"""

import yaml
import pytest
from pathlib import Path

from aurarouter.config import ConfigLoader


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

def _make_loader(config_data: dict) -> ConfigLoader:
    """Create a ConfigLoader from a plain dict (no file needed)."""
    loader = ConfigLoader.__new__(ConfigLoader)
    loader.config = config_data
    loader._config_path = None
    return loader


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_old_config_models_only():
    """Config with only 'models' section works."""
    config_data = {
        "models": {
            "local-qwen": {
                "provider": "ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "qwen2.5-coder:7b",
            }
        },
        "roles": {"coding": ["local-qwen"]},
    }
    loader = _make_loader(config_data)

    # Verify model access works
    assert loader.get_model_config("local-qwen") is not None
    assert loader.get_model_config("local-qwen")["provider"] == "ollama"

    # Verify catalog_list includes legacy models
    ids = loader.catalog_list(kind="model")
    assert "local-qwen" in ids

    # Verify catalog_get falls back to models
    data = loader.catalog_get("local-qwen")
    assert data is not None
    assert data["kind"] == "model"
    assert data["provider"] == "ollama"


def test_old_config_no_catalog_section():
    """Config without 'catalog' section doesn't crash any accessor."""
    config_data = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "test"},
        },
        "roles": {"coding": ["m1"]},
    }
    loader = _make_loader(config_data)

    # catalog_list should still return models
    assert loader.catalog_list() == ["m1"]
    assert loader.catalog_list(kind="model") == ["m1"]
    assert loader.catalog_list(kind="service") == []
    assert loader.catalog_list(kind="analyzer") == []

    # catalog_get should fall back
    assert loader.catalog_get("m1") is not None
    assert loader.catalog_get("nonexistent") is None

    # catalog_query should work
    results = loader.catalog_query(kind="model")
    assert len(results) == 1
    assert results[0]["artifact_id"] == "m1"


def test_old_config_with_grid_services():
    """Old grid_services format is still parseable and doesn't interfere."""
    config_data = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "test"},
        },
        "roles": {"coding": ["m1"]},
        "grid_services": {
            "endpoints": [
                {"name": "auragrid-node1", "url": "http://192.168.1.50:9100"},
            ],
            "auto_sync_models": True,
        },
    }
    loader = _make_loader(config_data)

    # grid_services accessor works
    gs = loader.get_grid_services_config()
    assert gs["endpoints"][0]["name"] == "auragrid-node1"

    # models are still accessible
    assert loader.get_model_config("m1") is not None
    assert "m1" in loader.catalog_list()


def test_legacy_list_models_unchanged():
    """get_all_model_ids() returns only models section entries."""
    config_data = {
        "models": {
            "m1": {"provider": "ollama"},
            "m2": {"provider": "openapi"},
        },
        "catalog": {
            "svc1": {"kind": "service", "display_name": "Svc1"},
            "analyzer1": {"kind": "analyzer", "display_name": "A1"},
        },
    }
    loader = _make_loader(config_data)

    # get_all_model_ids should ONLY return models section keys
    model_ids = loader.get_all_model_ids()
    assert sorted(model_ids) == ["m1", "m2"]
    assert "svc1" not in model_ids
    assert "analyzer1" not in model_ids


def test_legacy_set_model_unchanged():
    """set_model() writes to models section, not catalog."""
    config_data = {
        "models": {},
        "catalog": {},
    }
    loader = _make_loader(config_data)

    loader.set_model("new-model", {"provider": "ollama", "model_name": "test"})

    # Model should be in 'models' section
    assert "new-model" in loader.config["models"]
    # Model should NOT be in 'catalog' section
    assert "new-model" not in loader.config["catalog"]


def test_legacy_remove_model_unchanged():
    """remove_model() only touches models section."""
    config_data = {
        "models": {"m1": {"provider": "ollama"}},
        "catalog": {"m1-cat": {"kind": "model", "display_name": "M1 Cat"}},
    }
    loader = _make_loader(config_data)

    removed = loader.remove_model("m1")
    assert removed is True
    assert "m1" not in loader.config["models"]
    # catalog entry is untouched
    assert "m1-cat" in loader.config["catalog"]


def test_old_config_catalog_set_creates_section():
    """catalog_set creates the catalog section if missing."""
    config_data = {
        "models": {"m1": {"provider": "ollama"}},
    }
    loader = _make_loader(config_data)

    loader.catalog_set("my-svc", {"kind": "service", "display_name": "My Svc"})
    assert "catalog" in loader.config
    assert "my-svc" in loader.config["catalog"]
