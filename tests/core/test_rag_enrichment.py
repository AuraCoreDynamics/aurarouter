"""Tests for the RAG enrichment pipeline."""

import asyncio
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from aurarouter.config import ConfigLoader
from aurarouter.rag_enrichment import RagEnrichmentPipeline, EnrichedContext


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(tmp_path: Path, rag_enabled: bool = True, xlm_endpoint: str = "http://localhost:9002") -> ConfigLoader:
    config_content = {
        "system": {
            "log_level": "INFO",
            "rag_enrichment": rag_enabled,
        },
        "xlm": {
            "endpoint": xlm_endpoint,
        },
        "models": {
            "mock_ollama": {
                "provider": "ollama",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "mock_model",
            },
        },
        "roles": {"coding": ["mock_ollama"]},
    }
    config_path = tmp_path / "auraconfig.yaml"
    config_path.write_text(yaml.dump(config_content))
    return ConfigLoader(config_path=str(config_path))


def _make_mock_registry(search_results=None):
    """Create a mock MCP registry with search capability."""
    mock_client = MagicMock()
    mock_client.call_tool.return_value = {
        "results": search_results or [
            {"content": "Python lists can be sorted with list.sort() or sorted().", "score": 0.95},
            {"content": "The key parameter accepts a callable for custom sorting.", "score": 0.88},
        ]
    }
    registry = MagicMock()
    registry.get_clients_with_capability.return_value = [mock_client]
    return registry, mock_client


# ── Basic tests ──────────────────────────────────────────────────────


def test_enriched_context_defaults():
    ctx = EnrichedContext(original_task="test")
    assert ctx.rag_snippets == []
    assert ctx.total_tokens_used == 0
    assert ctx.source == "none"


def test_is_enabled_true(tmp_path):
    config = _make_config(tmp_path, rag_enabled=True)
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    assert pipeline.is_enabled() is True


def test_is_enabled_false(tmp_path):
    config = _make_config(tmp_path, rag_enabled=False)
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    assert pipeline.is_enabled() is False


def test_enrich_disabled_returns_empty(tmp_path):
    config = _make_config(tmp_path, rag_enabled=False)
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    result = asyncio.run(pipeline.enrich("How do I sort?"))
    assert result.rag_snippets == []
    assert result.source == "none"


def test_enrich_no_endpoint_returns_empty(tmp_path):
    config = _make_config(tmp_path, rag_enabled=True, xlm_endpoint="")
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    result = asyncio.run(pipeline.enrich("How do I sort?"))
    assert result.rag_snippets == []


# ── Build enriched prompt ────────────────────────────────────────────


def test_build_enriched_prompt_with_snippets(tmp_path):
    config = _make_config(tmp_path)
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    enriched = EnrichedContext(
        original_task="Sort a list",
        rag_snippets=[{"content": "Use sorted()"}],
        source="auraxlm",
    )
    result = pipeline.build_enriched_prompt("Sort a list", enriched)
    assert "Use sorted()" in result
    assert "Relevant Context" in result


def test_build_enriched_prompt_no_snippets(tmp_path):
    config = _make_config(tmp_path)
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    enriched = EnrichedContext(original_task="Sort a list")
    result = pipeline.build_enriched_prompt("Sort a list", enriched)
    assert result == "Sort a list"


# ── Token budget ─────────────────────────────────────────────────────


def test_estimate_tokens_trims_to_budget(tmp_path):
    config = _make_config(tmp_path)
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    snippets = [
        {"content": "A" * 400},  # ~100 tokens
        {"content": "B" * 400},  # ~100 tokens
        {"content": "C" * 400},  # ~100 tokens
    ]
    trimmed = pipeline._estimate_tokens(snippets, max_tokens=150)
    assert len(trimmed) == 1  # only first fits


def test_estimate_tokens_all_fit(tmp_path):
    config = _make_config(tmp_path)
    registry = MagicMock()
    pipeline = RagEnrichmentPipeline(registry, config)
    snippets = [
        {"content": "Short text"},
        {"content": "Another short text"},
    ]
    trimmed = pipeline._estimate_tokens(snippets, max_tokens=2048)
    assert len(trimmed) == 2


# ── Config accessors ─────────────────────────────────────────────────


def test_config_is_rag_enrichment_enabled(tmp_path):
    config = _make_config(tmp_path, rag_enabled=True)
    assert config.is_rag_enrichment_enabled() is True


def test_config_rag_disabled_by_default(tmp_path):
    config_content = {
        "system": {"log_level": "INFO"},
        "models": {},
        "roles": {},
    }
    config_path = tmp_path / "auraconfig.yaml"
    config_path.write_text(yaml.dump(config_content))
    config = ConfigLoader(config_path=str(config_path))
    assert config.is_rag_enrichment_enabled() is False
