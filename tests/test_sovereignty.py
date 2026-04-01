"""Tests for the sovereignty enforcement gate."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.savings.privacy import PrivacyAuditor
from aurarouter.sovereignty import (
    SovereigntyGate,
    SovereigntyResult,
    SovereigntyVerdict,
    SovereigntyViolationError,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(tmp_path: Path, overrides: dict | None = None) -> ConfigLoader:
    """Create a ConfigLoader with sovereignty-relevant config."""
    base = {
        "system": {
            "log_level": "INFO",
            "sovereignty_enforcement": True,
        },
        "models": {
            "local_llama": {
                "provider": "ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "llama3",
            },
            "cloud_gemini": {
                "provider": "google",
                "endpoint": "https://api.google.com",
                "model_name": "gemini-2.5-pro",
                "hosting_tier": "cloud",
            },
        },
        "roles": {
            "coding": ["local_llama", "cloud_gemini"],
        },
    }
    if overrides:
        base.update(overrides)
    config_path = tmp_path / "auraconfig.yaml"
    config_path.write_text(yaml.dump(base))
    return ConfigLoader(config_path=str(config_path))


# ── Evaluate tests ───────────────────────────────────────────────────


def test_evaluate_clean_prompt_returns_open(tmp_path):
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    result = gate.evaluate("How do I sort a list in Python?")
    assert result.verdict == SovereigntyVerdict.OPEN
    assert result.matched_patterns == []


def test_evaluate_pii_returns_sovereign(tmp_path):
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    result = gate.evaluate("Send this to user@example.com please")
    assert result.verdict == SovereigntyVerdict.SOVEREIGN
    assert "Email Address" in result.matched_patterns


def test_evaluate_ssn_returns_sovereign(tmp_path):
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    result = gate.evaluate("SSN: 123-45-6789")
    assert result.verdict == SovereigntyVerdict.SOVEREIGN
    assert "SSN" in result.matched_patterns


def test_evaluate_confidential_marker(tmp_path):
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    result = gate.evaluate("This document is CONFIDENTIAL and must not leak")
    assert result.verdict == SovereigntyVerdict.SOVEREIGN
    assert "Confidential Marker" in result.matched_patterns


def test_evaluate_disabled_returns_open(tmp_path):
    config = _make_config(tmp_path, {
        "system": {"sovereignty_enforcement": False}
    })
    gate = SovereigntyGate(config)
    result = gate.evaluate("SSN: 123-45-6789")
    assert result.verdict == SovereigntyVerdict.OPEN


# ── Custom sovereignty patterns ──────────────────────────────────────


def test_custom_sovereignty_pattern(tmp_path):
    config = _make_config(tmp_path, {
        "system": {
            "sovereignty_enforcement": True,
            "sovereignty_patterns": [
                {
                    "name": "FOUO Marker",
                    "pattern": r"(?i)\bfor\s+official\s+use\s+only\b",
                    "severity": "high",
                    "description": "FOUO marking",
                },
            ],
        },
    })
    gate = SovereigntyGate(config)
    result = gate.evaluate("This is FOR OFFICIAL USE ONLY material")
    assert result.verdict == SovereigntyVerdict.SOVEREIGN
    assert "FOUO Marker" in result.matched_patterns


def test_invalid_custom_pattern_skipped(tmp_path):
    config = _make_config(tmp_path, {
        "system": {
            "sovereignty_enforcement": True,
            "sovereignty_patterns": [
                {"name": "Bad Pattern", "pattern": "[invalid("},
            ],
        },
    })
    # Should not raise — invalid patterns are skipped
    gate = SovereigntyGate(config)
    result = gate.evaluate("Clean prompt")
    assert result.verdict == SovereigntyVerdict.OPEN


# ── Enforce tests ────────────────────────────────────────────────────


def test_enforce_open_returns_full_chain(tmp_path):
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    chain = ["local_llama", "cloud_gemini"]
    result = SovereigntyResult(verdict=SovereigntyVerdict.OPEN)
    filtered = gate.enforce(chain, config, result)
    assert filtered == chain


def test_enforce_sovereign_filters_cloud(tmp_path):
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    chain = ["local_llama", "cloud_gemini"]
    result = SovereigntyResult(
        verdict=SovereigntyVerdict.SOVEREIGN,
        reason="PII detected",
        matched_patterns=["Email Address"],
    )
    filtered = gate.enforce(chain, config, result)
    assert filtered == ["local_llama"]
    assert "cloud_gemini" not in filtered


def test_enforce_blocked_raises(tmp_path):
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    chain = ["local_llama", "cloud_gemini"]
    result = SovereigntyResult(
        verdict=SovereigntyVerdict.BLOCKED,
        reason="Classified content",
    )
    with pytest.raises(SovereigntyViolationError, match="Classified content"):
        gate.enforce(chain, config, result)


def test_enforce_sovereign_all_cloud_returns_empty(tmp_path):
    """When all models are cloud, enforce returns empty list."""
    config = _make_config(tmp_path)
    gate = SovereigntyGate(config)
    chain = ["cloud_gemini"]
    result = SovereigntyResult(verdict=SovereigntyVerdict.SOVEREIGN)
    filtered = gate.enforce(chain, config, result)
    assert filtered == []


# ── Config accessors ─────────────────────────────────────────────────


def test_config_is_sovereignty_enabled(tmp_path):
    config = _make_config(tmp_path)
    assert config.is_sovereignty_enforcement_enabled() is True


def test_config_sovereignty_disabled_by_default(tmp_path):
    config = _make_config(tmp_path, {"system": {"log_level": "INFO"}})
    assert config.is_sovereignty_enforcement_enabled() is False


def test_config_get_sovereignty_patterns(tmp_path):
    patterns = [{"name": "Test", "pattern": "foo", "severity": "low"}]
    config = _make_config(tmp_path, {
        "system": {"sovereignty_patterns": patterns}
    })
    assert config.get_sovereignty_patterns() == patterns
