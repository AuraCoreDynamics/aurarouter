import yaml

from aurarouter.config import ConfigLoader


def test_get_savings_config_default(config):
    """Returns {} when no savings section exists."""
    assert config.get_savings_config() == {}


def test_get_savings_config_present(tmp_path):
    """Returns the savings dict when present in config."""
    cfg_data = {
        "models": {},
        "roles": {},
        "savings": {
            "enabled": True,
            "db_path": "/tmp/test.db",
            "privacy": {"enabled": True, "custom_patterns": []},
            "pricing_overrides": {
                "gemini-2.0-flash": {
                    "input_per_million": 0.10,
                    "output_per_million": 0.40,
                }
            },
            "budget": {"enabled": False, "daily_limit": 10.0},
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = ConfigLoader(config_path=str(p))

    savings = cfg.get_savings_config()
    assert savings["enabled"] is True
    assert savings["db_path"] == "/tmp/test.db"
    assert savings["privacy"]["enabled"] is True
    assert cfg.get_budget_config() == {"enabled": False, "daily_limit": 10.0}
    assert cfg.get_privacy_config() == {"enabled": True, "custom_patterns": []}


def test_is_savings_enabled_default(config):
    """Returns True when no savings section exists (opt-out design)."""
    assert config.is_savings_enabled() is True


def test_get_model_pricing_explicit():
    """Model with explicit cost fields returns them."""
    config = ConfigLoader(allow_missing=True)
    config.config = {
        "models": {
            "my_model": {
                "provider": "google",
                "cost_per_1m_input": 0.50,
                "cost_per_1m_output": 2.00,
            }
        }
    }
    inp, out = config.get_model_pricing("my_model")
    assert inp == 0.50
    assert out == 2.00


def test_get_model_pricing_absent():
    """Model without cost fields returns (None, None)."""
    config = ConfigLoader(allow_missing=True)
    config.config = {
        "models": {
            "my_model": {"provider": "ollama"}
        }
    }
    inp, out = config.get_model_pricing("my_model")
    assert inp is None
    assert out is None


def test_get_model_pricing_partial():
    """Model with only one cost field returns None for the missing one."""
    config = ConfigLoader(allow_missing=True)
    config.config = {
        "models": {
            "my_model": {
                "provider": "google",
                "cost_per_1m_input": 0.50,
            }
        }
    }
    inp, out = config.get_model_pricing("my_model")
    assert inp == 0.50
    assert out is None


def test_get_model_pricing_unknown_model():
    """Unknown model returns (None, None)."""
    config = ConfigLoader(allow_missing=True)
    config.config = {"models": {}}
    inp, out = config.get_model_pricing("nonexistent")
    assert inp is None
    assert out is None


def test_pricing_overrides(tmp_path):
    """Verify get_pricing_overrides() returns the overrides dict."""
    cfg_data = {
        "models": {},
        "roles": {},
        "savings": {
            "pricing_overrides": {
                "gemini-2.0-flash": {
                    "input_per_million": 0.10,
                    "output_per_million": 0.40,
                }
            }
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_data))
    cfg = ConfigLoader(config_path=str(p))

    overrides = cfg.get_pricing_overrides()
    assert "gemini-2.0-flash" in overrides
    assert overrides["gemini-2.0-flash"]["input_per_million"] == 0.10
    assert overrides["gemini-2.0-flash"]["output_per_million"] == 0.40
