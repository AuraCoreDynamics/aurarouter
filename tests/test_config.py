import pytest

from aurarouter.config import ConfigLoader


def test_load_from_explicit_path():
    cfg = ConfigLoader(config_path="auraconfig.yaml")
    assert cfg.config is not None
    assert "models" in cfg.config
    assert "roles" in cfg.config


def test_allow_missing():
    cfg = ConfigLoader(allow_missing=True)
    assert cfg.config == {}


def test_missing_config_raises(monkeypatch, tmp_path):
    # Point the home-dir fallback at a directory that doesn't contain the config
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("AURACORE_ROUTER_CONFIG", raising=False)

    with pytest.raises(FileNotFoundError, match="Could not find"):
        ConfigLoader(config_path="nonexistent_path_12345.yaml")


def test_get_role_chain(config):
    chain = config.get_role_chain("coding")
    assert isinstance(chain, list)
    assert len(chain) > 0


def test_get_model_config(config):
    chain = config.get_role_chain("coding")
    cfg = config.get_model_config(chain[0])
    assert "provider" in cfg


def test_get_role_chain_empty(config):
    assert config.get_role_chain("nonexistent_role") == []


def test_get_model_config_empty(config):
    assert config.get_model_config("nonexistent_model") == {}


# ------------------------------------------------------------------
# Role format compatibility
# ------------------------------------------------------------------

def test_get_role_chain_nested_dict_with_models_key(tmp_path):
    """Verify that dict-style roles with 'models' key work (sample_config format)."""
    import yaml
    config_content = {
        "models": {"m1": {"provider": "ollama", "model_name": "x"}},
        "roles": {
            "router": {
                "description": "classify intent",
                "models": ["m1"],
                "fallback_on_error": True,
            }
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))
    assert cfg.get_role_chain("router") == ["m1"]


def test_get_role_chain_nested_dict_with_chain_key(tmp_path):
    """Verify that dict-style roles with 'chain' key work."""
    import yaml
    config_content = {
        "models": {"m1": {"provider": "ollama", "model_name": "x"}},
        "roles": {"router": {"chain": ["m1"]}},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))
    assert cfg.get_role_chain("router") == ["m1"]


# ------------------------------------------------------------------
# Mutation methods
# ------------------------------------------------------------------

def test_set_model(config):
    config.set_model("new_model", {"provider": "ollama", "model_name": "test"})
    assert config.get_model_config("new_model") == {"provider": "ollama", "model_name": "test"}


def test_set_model_overwrite(config):
    config.set_model("mock_ollama", {"provider": "google", "model_name": "replaced"})
    assert config.get_model_config("mock_ollama")["provider"] == "google"


def test_remove_model(config):
    assert config.remove_model("mock_ollama") is True
    assert config.get_model_config("mock_ollama") == {}


def test_remove_model_nonexistent(config):
    assert config.remove_model("does_not_exist") is False


def test_set_role_chain(config):
    config.set_role_chain("new_role", ["mock_ollama", "mock_google"])
    assert config.get_role_chain("new_role") == ["mock_ollama", "mock_google"]


def test_remove_role(config):
    assert config.remove_role("coding") is True
    assert config.get_role_chain("coding") == []


def test_remove_role_nonexistent(config):
    assert config.remove_role("does_not_exist") is False


def test_get_all_model_ids(config):
    ids = config.get_all_model_ids()
    assert "mock_ollama" in ids
    assert "mock_google" in ids


def test_get_all_roles(config):
    roles = config.get_all_roles()
    assert "router" in roles
    assert "coding" in roles


# ------------------------------------------------------------------
# Persistence (save / round-trip)
# ------------------------------------------------------------------

def test_save_creates_file(config, tmp_path):
    target = tmp_path / "saved_config.yaml"
    result = config.save(path=target)
    assert result == target
    assert target.is_file()


def test_save_round_trip(config, tmp_path):
    """Mutate, save, reload - changes must survive."""
    config.set_model("added_model", {"provider": "claude", "model_name": "opus"})
    config.set_role_chain("coding", ["added_model", "mock_ollama"])

    target = tmp_path / "roundtrip.yaml"
    config.save(path=target)

    reloaded = ConfigLoader(config_path=str(target))
    assert reloaded.get_model_config("added_model")["provider"] == "claude"
    assert reloaded.get_role_chain("coding") == ["added_model", "mock_ollama"]


def test_save_defaults_to_original_path(config):
    """Save without explicit path writes back to the original config file."""
    config.set_model("test_save", {"provider": "ollama", "model_name": "x"})
    config.save()

    reloaded = ConfigLoader(config_path=str(config.config_path))
    assert "test_save" in reloaded.get_all_model_ids()


# ------------------------------------------------------------------
# Grid services config
# ------------------------------------------------------------------


def test_get_grid_services_config_empty():
    """Returns {} when grid_services is absent."""
    config = ConfigLoader(allow_missing=True)
    config.config = {"models": {}, "roles": {}}
    assert config.get_grid_services_config() == {}


def test_get_grid_services_config_present(tmp_path):
    """Returns grid_services section when present."""
    import yaml

    cfg_content = {
        "models": {},
        "roles": {},
        "grid_services": {
            "endpoints": [
                {"url": "http://localhost:8080", "name": "xlm"},
            ],
            "auto_sync_models": False,
        },
    }
    path = tmp_path / "auraconfig.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg_content, f)

    config = ConfigLoader(config_path=str(path))
    grid_cfg = config.get_grid_services_config()
    assert len(grid_cfg["endpoints"]) == 1
    assert grid_cfg["endpoints"][0]["url"] == "http://localhost:8080"
    assert grid_cfg["auto_sync_models"] is False


def test_to_yaml(config):
    yaml_str = config.to_yaml()
    assert "mock_ollama" in yaml_str
    assert "roles" in yaml_str


def test_config_path_property(config):
    assert config.config_path is not None
    assert config.config_path.is_file()


def test_config_path_none_when_allow_missing():
    cfg = ConfigLoader(allow_missing=True)
    assert cfg.config_path is None


# ------------------------------------------------------------------
# MCP tools config accessors
# ------------------------------------------------------------------

def test_mcp_tool_enabled_default(config):
    """Defaults to True when no mcp section exists."""
    assert config.is_mcp_tool_enabled("route_task", default=True) is True


def test_mcp_tool_disabled_default(config):
    """Defaults to False when passed default=False and no mcp section."""
    assert config.is_mcp_tool_enabled("compare_models", default=False) is False


def test_mcp_tool_enabled_explicit(tmp_path):
    import yaml
    cfg_data = {
        "models": {}, "roles": {},
        "mcp": {"tools": {"route_task": {"enabled": False}}},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = ConfigLoader(config_path=str(p))
    assert cfg.is_mcp_tool_enabled("route_task") is False


def test_set_mcp_tool_enabled(config):
    config.set_mcp_tool_enabled("compare_models", True)
    assert config.is_mcp_tool_enabled("compare_models") is True


def test_set_mcp_tool_enabled_creates_structure(config):
    """set_mcp_tool_enabled should create the mcp.tools structure if absent."""
    config.set_mcp_tool_enabled("new_tool", False)
    assert config.config["mcp"]["tools"]["new_tool"]["enabled"] is False


def test_get_mcp_tools_config_empty(config):
    """Returns empty dict when no mcp section exists."""
    assert config.get_mcp_tools_config() == {}


def test_mcp_tools_round_trip(config, tmp_path):
    """MCP tools config should survive save/reload."""
    config.set_mcp_tool_enabled("route_task", False)
    config.set_mcp_tool_enabled("compare_models", True)

    target = tmp_path / "mcp_roundtrip.yaml"
    config.save(path=target)

    reloaded = ConfigLoader(config_path=str(target))
    assert reloaded.is_mcp_tool_enabled("route_task") is False
    assert reloaded.is_mcp_tool_enabled("compare_models") is True
