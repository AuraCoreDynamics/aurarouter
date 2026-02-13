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
