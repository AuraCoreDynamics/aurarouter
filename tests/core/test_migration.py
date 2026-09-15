"""Tests for config migration (old format -> new format with catalog)."""

import yaml
import pytest
from pathlib import Path

from aurarouter.migration import migrate_config, migrate_config_file


# ------------------------------------------------------------------
# Unit tests for migrate_config()
# ------------------------------------------------------------------

def test_migrate_adds_catalog_section():
    """Config without catalog gets one added."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
        "roles": {"coding": ["m1"]},
    }
    migrated, report = migrate_config(data)

    assert "catalog" in migrated
    assert isinstance(migrated["catalog"], dict)
    assert any("catalog" in line.lower() for line in report)


def test_migrate_grid_services_to_catalog():
    """grid_services.endpoints become catalog service entries."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
        "grid_services": {
            "endpoints": [
                {"name": "node1", "url": "http://192.168.1.50:9100"},
                {"name": "node2", "url": "http://192.168.1.51:9100"},
            ],
            "auto_sync_models": True,
        },
    }
    migrated, report = migrate_config(data)

    assert "node1" in migrated["catalog"]
    assert migrated["catalog"]["node1"]["kind"] == "service"
    assert migrated["catalog"]["node1"]["endpoint"] == "http://192.168.1.50:9100"
    assert migrated["catalog"]["node1"]["auto_sync_models"] is True

    assert "node2" in migrated["catalog"]
    assert migrated["catalog"]["node2"]["kind"] == "service"

    assert any("node1" in line for line in report)
    assert any("node2" in line for line in report)


def test_migrate_adds_active_analyzer():
    """Missing system.active_analyzer gets set."""
    data = {
        "system": {"log_level": "INFO"},
        "models": {"m1": {"provider": "ollama"}},
    }
    migrated, report = migrate_config(data)

    assert migrated["system"]["active_analyzer"] == "aurarouter-default"
    assert any("active_analyzer" in line for line in report)


def test_migrate_adds_active_analyzer_no_system():
    """Missing system section entirely gets created with active_analyzer."""
    data = {"models": {"m1": {"provider": "ollama"}}}
    migrated, report = migrate_config(data)

    assert migrated["system"]["active_analyzer"] == "aurarouter-default"


def test_migrate_preserves_models():
    """Models section is not modified."""
    original_models = {
        "m1": {"provider": "ollama", "model_name": "qwen"},
        "m2": {"provider": "openapi", "endpoint": "http://x"},
    }
    data = {"models": dict(original_models)}
    migrated, _report = migrate_config(data)

    assert migrated["models"] == original_models


def test_migrate_preserves_grid_services():
    """grid_services section is not removed."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
        "grid_services": {
            "endpoints": [{"name": "node1", "url": "http://x"}],
        },
    }
    migrated, _report = migrate_config(data)

    assert "grid_services" in migrated
    assert migrated["grid_services"]["endpoints"][0]["name"] == "node1"


def test_migrate_idempotent():
    """Running migration twice produces same result."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
        "grid_services": {
            "endpoints": [{"name": "node1", "url": "http://x"}],
        },
    }
    first, report1 = migrate_config(data)
    second, report2 = migrate_config(first)

    assert first == second
    assert report2 == ["No migration needed — config is already current"]


def test_migrate_already_current():
    """Config that's already current reports no changes."""
    data = {
        "system": {"active_analyzer": "aurarouter-default"},
        "models": {"m1": {"provider": "ollama"}},
        "catalog": {"svc1": {"kind": "service"}},
    }
    _migrated, report = migrate_config(data)

    assert report == ["No migration needed — config is already current"]


def test_migrate_does_not_mutate_input():
    """The original config_data dict is not modified."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
    }
    import copy
    original = copy.deepcopy(data)
    migrate_config(data)
    assert data == original


def test_migrate_skips_endpoints_without_name():
    """Endpoints without a 'name' field are skipped."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
        "grid_services": {
            "endpoints": [
                {"url": "http://x"},  # no name
                {"name": "valid", "url": "http://y"},
            ],
        },
    }
    migrated, report = migrate_config(data)

    assert "valid" in migrated["catalog"]
    # Only valid endpoint migrated
    assert len(migrated["catalog"]) == 1


def test_migrate_does_not_overwrite_existing_catalog_entry():
    """If catalog already has an entry with same name as endpoint, skip it."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
        "catalog": {
            "node1": {"kind": "service", "display_name": "Already There"},
        },
        "grid_services": {
            "endpoints": [{"name": "node1", "url": "http://new-url"}],
        },
    }
    migrated, report = migrate_config(data)

    # Should keep original, not overwrite
    assert migrated["catalog"]["node1"]["display_name"] == "Already There"


# ------------------------------------------------------------------
# File-level tests
# ------------------------------------------------------------------

def test_migrate_file_dry_run(tmp_path):
    """Dry run doesn't write changes."""
    data = {"models": {"m1": {"provider": "ollama"}}}
    config_file = tmp_path / "auraconfig.yaml"
    config_file.write_text(yaml.dump(data))

    original_content = config_file.read_text()
    report = migrate_config_file(str(config_file), dry_run=True)

    assert config_file.read_text() == original_content
    assert any("dry-run" in line for line in report)


def test_migrate_file_writes(tmp_path):
    """Non-dry-run writes the migrated config."""
    data = {"models": {"m1": {"provider": "ollama"}}}
    config_file = tmp_path / "auraconfig.yaml"
    config_file.write_text(yaml.dump(data))

    report = migrate_config_file(str(config_file), dry_run=False)

    assert any("Wrote" in line for line in report)

    # Reload and verify
    with open(config_file) as f:
        reloaded = yaml.safe_load(f)
    assert "catalog" in reloaded
    assert reloaded["system"]["active_analyzer"] == "aurarouter-default"


def test_migrate_file_already_current(tmp_path):
    """Already-current file is not rewritten."""
    data = {
        "system": {"active_analyzer": "aurarouter-default"},
        "models": {"m1": {"provider": "ollama"}},
        "catalog": {},
    }
    config_file = tmp_path / "auraconfig.yaml"
    config_file.write_text(yaml.dump(data))

    original_content = config_file.read_text()
    report = migrate_config_file(str(config_file), dry_run=False)

    assert config_file.read_text() == original_content
    assert "No migration needed" in report[0]


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_migrate_config_file_nonexistent_path():
    """migrate_config_file with a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        migrate_config_file("/nonexistent/path/auraconfig.yaml")


def test_migrate_idempotent_dict_equality():
    """Running migration on an already-migrated config produces dict equality."""
    data = {
        "models": {"m1": {"provider": "ollama"}},
        "grid_services": {
            "endpoints": [
                {"name": "node1", "url": "http://192.168.1.50:9100"},
            ],
            "auto_sync_models": True,
        },
    }
    first, _report1 = migrate_config(data)
    second, _report2 = migrate_config(first)

    # Deep equality — every key, every value identical
    assert first == second
    assert first["catalog"] == second["catalog"]
    assert first["system"] == second["system"]
    assert first["models"] == second["models"]
    assert first["grid_services"] == second["grid_services"]
