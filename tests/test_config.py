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
