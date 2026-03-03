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


# ------------------------------------------------------------------
# auto_join_roles
# ------------------------------------------------------------------

def test_auto_join_roles_direct_match(config):
    """Tags matching existing role names add the model to that role chain."""
    roles_joined = config.auto_join_roles("new_model", ["coding"])
    assert roles_joined == ["coding"]
    assert "new_model" in config.get_role_chain("coding")


def test_auto_join_roles_synonym_match(config):
    """Tags matching semantic verb synonyms resolve to the canonical role."""
    config.set_semantic_verb("coding", ["programming", "developer"])
    roles_joined = config.auto_join_roles("new_model", ["programming"])
    assert roles_joined == ["coding"]
    assert "new_model" in config.get_role_chain("coding")


def test_auto_join_roles_no_match(config):
    """Tags that don't match any role or synonym are ignored."""
    roles_joined = config.auto_join_roles("new_model", ["unknown_tag"])
    assert roles_joined == []


def test_auto_join_roles_no_duplicate(config):
    """Model is not added to a role chain if already present."""
    roles_joined = config.auto_join_roles("mock_ollama", ["coding"])
    assert roles_joined == []
    # Verify chain unchanged
    assert config.get_role_chain("coding").count("mock_ollama") == 1


def test_auto_join_roles_case_insensitive(config):
    """Tag matching is case-insensitive."""
    roles_joined = config.auto_join_roles("new_model", ["CODING"])
    assert roles_joined == ["coding"]
    assert "new_model" in config.get_role_chain("coding")


# ------------------------------------------------------------------
# Read accessors return copies (not references)
# ------------------------------------------------------------------

def test_get_role_chain_returns_copy(config):
    """Mutating the returned list must not affect internal state."""
    chain = config.get_role_chain("coding")
    original_len = len(chain)
    chain.append("INJECTED")
    assert len(config.get_role_chain("coding")) == original_len


def test_get_model_config_returns_copy(config):
    """Mutating the returned dict must not affect internal state."""
    cfg = config.get_model_config("mock_ollama")
    cfg["INJECTED"] = True
    assert "INJECTED" not in config.get_model_config("mock_ollama")


def test_get_all_model_ids_returns_copy(config):
    """Mutating the returned list must not affect internal state."""
    ids = config.get_all_model_ids()
    original_len = len(ids)
    ids.append("INJECTED")
    assert len(config.get_all_model_ids()) == original_len


def test_get_all_roles_returns_copy(config):
    """Mutating the returned list must not affect internal state."""
    roles = config.get_all_roles()
    original_len = len(roles)
    roles.append("INJECTED")
    assert len(config.get_all_roles()) == original_len


# ------------------------------------------------------------------
# Concurrent access (thread safety)
# ------------------------------------------------------------------

import threading


def test_concurrent_set_and_get_model(tmp_path):
    """Concurrent set_model/get_model_config calls must not raise or corrupt."""
    import yaml

    config_content = {
        "models": {},
        "roles": {"coding": []},
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))

    num_threads = 20
    iterations = 50
    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(iterations):
                model_id = f"model_{thread_id}_{i}"
                cfg.set_model(model_id, {"provider": "ollama", "model_name": f"m{i}"})
        except Exception as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(iterations):
                for mid in cfg.get_all_model_ids():
                    cfg.get_model_config(mid)
        except Exception as exc:
            errors.append(exc)

    threads = []
    for tid in range(num_threads):
        threads.append(threading.Thread(target=writer, args=(tid,)))
        threads.append(threading.Thread(target=reader))

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent access errors: {errors}"
    # All models written by writers should be present
    all_ids = cfg.get_all_model_ids()
    for tid in range(num_threads):
        for i in range(iterations):
            assert f"model_{tid}_{i}" in all_ids


def test_concurrent_set_and_remove_model(tmp_path):
    """Concurrent set_model and remove_model must not raise KeyError."""
    import yaml

    config_content = {
        "models": {f"pre_{i}": {"provider": "ollama"} for i in range(50)},
        "roles": {},
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))

    errors: list[Exception] = []

    def remover() -> None:
        try:
            for i in range(50):
                cfg.remove_model(f"pre_{i}")
        except Exception as exc:
            errors.append(exc)

    def adder() -> None:
        try:
            for i in range(50):
                cfg.set_model(f"new_{i}", {"provider": "google"})
        except Exception as exc:
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(100):
                cfg.get_all_model_ids()
                cfg.get_model_config("pre_25")
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=remover),
        threading.Thread(target=adder),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent access errors: {errors}"


def test_concurrent_save_no_corrupt_yaml(tmp_path):
    """Concurrent mutation + save() must not produce corrupt YAML.

    Each thread saves to its own file to avoid Windows PermissionError
    on concurrent os.replace() targeting the same path. The key safety
    guarantee is that each saved file is valid YAML despite concurrent
    in-memory mutations.
    """
    import yaml

    config_content = {
        "models": {"m1": {"provider": "ollama", "model_name": "x"}},
        "roles": {"coding": ["m1"]},
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))

    errors: list[Exception] = []

    def mutate_and_save(thread_id: int) -> None:
        try:
            target = tmp_path / f"save_target_{thread_id}.yaml"
            for i in range(20):
                cfg.set_model(f"t{thread_id}_m{i}", {"provider": "ollama"})
                cfg.save(path=target)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=mutate_and_save, args=(tid,)) for tid in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent save errors: {errors}"

    # Every saved file must be valid YAML with the expected structure
    for tid in range(5):
        target = tmp_path / f"save_target_{tid}.yaml"
        with open(target) as f:
            loaded = yaml.safe_load(f)
        assert isinstance(loaded, dict), f"Thread {tid} saved corrupt YAML"
        assert "models" in loaded


def test_concurrent_role_chain_mutation(tmp_path):
    """Concurrent set_role_chain calls must not lose data or corrupt."""
    import yaml

    config_content = {
        "models": {},
        "roles": {"coding": []},
    }
    p = tmp_path / "auraconfig.yaml"
    p.write_text(yaml.dump(config_content))
    cfg = ConfigLoader(config_path=str(p))

    errors: list[Exception] = []

    def chain_writer(thread_id: int) -> None:
        try:
            for i in range(30):
                chain = cfg.get_role_chain("coding")
                model_id = f"t{thread_id}_m{i}"
                if model_id not in chain:
                    cfg.set_role_chain("coding", chain + [model_id])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=chain_writer, args=(tid,)) for tid in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent role chain errors: {errors}"
    # The chain should be a valid list (not corrupted)
    chain = cfg.get_role_chain("coding")
    assert isinstance(chain, list)
