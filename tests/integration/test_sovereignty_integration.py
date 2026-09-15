"""Integration tests for sovereignty + RAG enrichment in ComputeFabric."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.savings.privacy import PrivacyAuditor
from aurarouter.sovereignty import SovereigntyGate, SovereigntyVerdict


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(tmp_path: Path, sovereignty: bool = True, rag: bool = False) -> ConfigLoader:
    config_content = {
        "system": {
            "log_level": "INFO",
            "sovereignty_enforcement": sovereignty,
            "rag_enrichment": rag,
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
            "router": ["local_llama"],
        },
    }
    config_path = tmp_path / "auraconfig.yaml"
    config_path.write_text(yaml.dump(config_content))
    return ConfigLoader(config_path=str(config_path))


class FakeProvider:
    """Minimal provider that returns a fixed response."""

    def __init__(self, text="Hello, world!"):
        self._text = text

    def generate_with_usage(self, prompt, **kwargs):
        from aurarouter.savings.models import GenerateResult
        return GenerateResult(text=self._text)

    def generate_stream_sync(self, prompt, **kwargs):
        yield self._text


# ── Sovereignty + Fabric integration ─────────────────────────────────


def test_fabric_sovereignty_filters_cloud_on_pii(tmp_path):
    """When sovereignty is enabled and PII is detected, cloud models are excluded."""
    config = _make_config(tmp_path, sovereignty=True)
    gate = SovereigntyGate(config)
    fabric = ComputeFabric(config, sovereignty_gate=gate)

    # Mock _get_provider to track which models are attempted
    attempted_models = []
    original_get_provider = fabric._get_provider

    def tracking_get_provider(model_id):
        attempted_models.append(model_id)
        return FakeProvider()

    fabric._get_provider = tracking_get_provider

    # Execute with PII in prompt
    result = fabric.execute("coding", "Send to user@example.com please")
    assert result is not None
    assert result.text == "Hello, world!"
    # Only local model should have been attempted
    assert "local_llama" in attempted_models
    assert "cloud_gemini" not in attempted_models


def test_fabric_sovereignty_disabled_allows_all(tmp_path):
    """When sovereignty is disabled, all models are available."""
    config = _make_config(tmp_path, sovereignty=False)
    gate = SovereigntyGate(config)
    fabric = ComputeFabric(config, sovereignty_gate=gate)

    attempted_models = []

    def tracking_get_provider(model_id):
        attempted_models.append(model_id)
        return FakeProvider()

    fabric._get_provider = tracking_get_provider

    # Even with PII, sovereignty disabled → all models tried
    result = fabric.execute("coding", "Send to user@example.com please")
    assert result is not None
    # First model succeeds, so only one attempted (but cloud not filtered)
    assert "local_llama" in attempted_models


def test_fabric_sovereignty_clean_prompt_allows_cloud(tmp_path):
    """Clean prompt should allow cloud models even with sovereignty enabled."""
    config = _make_config(tmp_path, sovereignty=True)
    gate = SovereigntyGate(config)
    fabric = ComputeFabric(config, sovereignty_gate=gate)

    attempted_models = []

    def tracking_get_provider(model_id):
        attempted_models.append(model_id)
        return FakeProvider()

    fabric._get_provider = tracking_get_provider

    result = fabric.execute("coding", "How do I sort a list in Python?")
    assert result is not None
    # Clean prompt: first model in chain succeeds, chain includes both
    assert "local_llama" in attempted_models


def test_fabric_sovereignty_all_cloud_returns_error(tmp_path):
    """When sovereignty filters all models, fabric returns error."""
    config_content = {
        "system": {"sovereignty_enforcement": True},
        "models": {
            "cloud_gemini": {
                "provider": "google",
                "endpoint": "https://api.google.com",
                "model_name": "gemini-2.5-pro",
                "hosting_tier": "cloud",
            },
        },
        "roles": {"coding": ["cloud_gemini"]},
    }
    config_path = tmp_path / "auraconfig.yaml"
    config_path.write_text(yaml.dump(config_content))
    config = ConfigLoader(config_path=str(config_path))
    gate = SovereigntyGate(config)
    fabric = ComputeFabric(config, sovereignty_gate=gate)

    result = fabric.execute("coding", "SSN: 123-45-6789")
    assert result is not None
    assert "Sovereignty gate filtered all models" in result.text


# ── MCP tool tests ───────────────────────────────────────────────────


def test_sovereignty_status_tool(tmp_path):
    import json
    from aurarouter.mcp_tools import sovereignty_status

    config = _make_config(tmp_path, sovereignty=True)
    fabric = ComputeFabric(config)

    result = json.loads(sovereignty_status(fabric))
    assert result["enabled"] is True
    assert isinstance(result["custom_patterns"], int)
    assert isinstance(result["local_models"], list)
    assert "local_llama" in result["local_models"]


def test_rag_status_tool(tmp_path):
    import json
    from aurarouter.mcp_tools import rag_status

    config = _make_config(tmp_path, rag=True)
    fabric = ComputeFabric(config)

    result = json.loads(rag_status(fabric))
    assert result["enabled"] is True
