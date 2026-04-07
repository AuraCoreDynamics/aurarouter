"""Tests for AURACORE_TELEMETRY_ENABLED env-var override in AuraRouter telemetry config."""
import json

import pytest

import aurarouter.telemetry_config as tc

_ENV_VAR = "AURACORE_TELEMETRY_ENABLED"


@pytest.fixture(autouse=True)
def _isolate_config_path(tmp_path, monkeypatch):
    """Point the config path at a temp directory so the real user config is never read."""
    monkeypatch.setattr(tc, "_TELEMETRY_CONFIG_PATH", tmp_path / "telemetry.json")


class TestEnvVarOverride:
    def test_env_true_no_file_returns_true(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "true")
        assert tc.is_external_telemetry_enabled() is True

    def test_env_false_overrides_file_enabled_returns_false(self, tmp_path, monkeypatch):
        telemetry_file = tmp_path / "telemetry.json"
        telemetry_file.write_text(json.dumps({"enabled": True}), encoding="utf-8")
        monkeypatch.setenv(_ENV_VAR, "false")
        assert tc.is_external_telemetry_enabled() is False

    def test_env_absent_no_file_defaults_false(self, monkeypatch):
        monkeypatch.delenv(_ENV_VAR, raising=False)
        assert tc.is_external_telemetry_enabled() is False

    def test_env_garbage_falls_through_to_file_lookup(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "garbage")
        # No file present → file-based default is False
        assert tc.is_external_telemetry_enabled() is False

    def test_env_one_truthy_alias(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "1")
        assert tc.is_external_telemetry_enabled() is True

    def test_env_yes_truthy_alias(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "yes")
        assert tc.is_external_telemetry_enabled() is True

    def test_env_zero_falsy_alias(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "0")
        assert tc.is_external_telemetry_enabled() is False

    def test_env_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(_ENV_VAR, "TRUE")
        assert tc.is_external_telemetry_enabled() is True
