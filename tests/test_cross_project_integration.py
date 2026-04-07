"""Cross-project integration tests for AuraRouter.

Validates that external provider packages (aurarouter-claude, aurarouter-gemini),
XLM augmentation hooks, and feedback store integration work correctly together.

The Claude and Gemini tests require their respective satellite packages
(aurarouter-claude, aurarouter-gemini) to be installed.  They are skipped
when those packages are absent.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.catalog import CatalogEntry, ProviderCatalog
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.providers.protocol import ProviderMetadata


def _has_entrypoint(group: str, name: str) -> bool:
    """Return True if the named entry point is registered."""
    return any(ep.name == name for ep in entry_points(group=group))


_has_claude = _has_entrypoint("aurarouter.providers", "claude")
_has_gemini = _has_entrypoint("aurarouter.providers", "gemini")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**extra) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": {}, "roles": {}, **extra}
    return cfg


def _make_catalog(config: ConfigLoader | None = None) -> ProviderCatalog:
    return ProviderCatalog(config or _make_config())


# ======================================================================
# T8.1a: aurarouter-claude entry point discovery
# ======================================================================

@pytest.mark.skipif(not _has_claude, reason="aurarouter-claude package not installed")
class TestClaudeProviderDiscovery:
    """Verify that aurarouter-claude is discoverable via ProviderCatalog."""

    def test_claude_entrypoint_registered(self):
        """aurarouter-claude should register under aurarouter.providers group."""
        eps = entry_points(group="aurarouter.providers")
        names = [ep.name for ep in eps]
        assert "claude" in names, (
            "aurarouter-claude entry point not found. "
            f"Available: {names}"
        )

    def test_claude_entrypoint_loads_metadata(self):
        """Loading the claude entry point should return provider metadata."""
        eps = entry_points(group="aurarouter.providers")
        claude_ep = None
        for ep in eps:
            if ep.name == "claude":
                claude_ep = ep
                break
        assert claude_ep is not None

        metadata_fn = claude_ep.load()
        meta = metadata_fn()
        assert meta.name == "claude"
        assert meta.provider_type == "mcp"
        assert meta.version

    def test_claude_discovered_by_catalog(self):
        """ProviderCatalog.discover() should include claude as an entrypoint provider."""
        catalog = _make_catalog()
        entries = catalog.discover()
        entry_names = [e.name for e in entries]
        assert "claude" in entry_names

        claude_entry = next(e for e in entries if e.name == "claude")
        assert claude_entry.source == "entrypoint"
        assert claude_entry.installed is True
        assert claude_entry.provider_type == "mcp"


# ======================================================================
# T8.1b: aurarouter-gemini entry point discovery
# ======================================================================

@pytest.mark.skipif(not _has_gemini, reason="aurarouter-gemini package not installed")
class TestGeminiProviderDiscovery:
    """Verify that aurarouter-gemini is discoverable via ProviderCatalog."""

    def test_gemini_entrypoint_registered(self):
        """aurarouter-gemini should register under aurarouter.providers group."""
        eps = entry_points(group="aurarouter.providers")
        names = [ep.name for ep in eps]
        assert "gemini" in names, (
            "aurarouter-gemini entry point not found. "
            f"Available: {names}"
        )

    def test_gemini_entrypoint_loads_metadata(self):
        """Loading the gemini entry point should return provider metadata."""
        eps = entry_points(group="aurarouter.providers")
        gemini_ep = None
        for ep in eps:
            if ep.name == "gemini":
                gemini_ep = ep
                break
        assert gemini_ep is not None

        metadata_fn = gemini_ep.load()
        meta = metadata_fn()
        assert meta.name == "gemini"
        assert meta.provider_type == "mcp"

    def test_gemini_discovered_by_catalog(self):
        """ProviderCatalog.discover() should include gemini as an entrypoint provider."""
        catalog = _make_catalog()
        entries = catalog.discover()
        entry_names = [e.name for e in entries]
        assert "gemini" in entry_names

        gemini_entry = next(e for e in entries if e.name == "gemini")
        assert gemini_entry.source == "entrypoint"
        assert gemini_entry.installed is True


# ======================================================================
# T8.1c: XLM augmentation hook
# ======================================================================

class TestXlmAugmentationIntegration:
    """Verify that XLM augmentation hook invokes auraxlm.query tool via mock MCP."""

    def test_augmentation_invokes_auraxlm_query(self):
        """When XLM is configured, _augment_prompt calls auraxlm.query on the client."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "augmented_prompt": "RAG context. user question"
        }

        cfg = _make_config(xlm={
            "endpoint": "http://xlm:8080",
            "features": {"prompt_augmentation": True},
        })
        fabric = ComputeFabric(cfg, xlm_client=mock_client)

        result = fabric._augment_prompt("user question", "coding")
        assert result == "RAG context. user question"
        mock_client.call_tool.assert_called_once_with(
            "auraxlm.query",
            headers={"X-AuraCore-Replica-Count": "1"},
            prompt="user question",
            role="coding",
        )

    def test_augmentation_graceful_on_mcp_failure(self):
        """Augmentation returns original prompt when MCP call fails."""
        mock_client = MagicMock()
        mock_client.call_tool.side_effect = ConnectionError("xlm unreachable")

        cfg = _make_config(xlm={
            "endpoint": "http://xlm:8080",
            "features": {"prompt_augmentation": True},
        })
        fabric = ComputeFabric(cfg, xlm_client=mock_client)

        result = fabric._augment_prompt("original prompt", "reasoning")
        assert result == "original prompt"


# ======================================================================
# T8.1d: Feedback store records from ComputeFabric
# ======================================================================

class TestFeedbackStoreIntegration:
    """Verify that ComputeFabric records feedback data to the store."""

    def test_record_feedback_called_on_success(self):
        """_record_feedback sends data to the FeedbackStore."""
        mock_store = MagicMock()
        cfg = _make_config()
        fabric = ComputeFabric(cfg, feedback_store=mock_store)

        fabric._record_feedback(
            role="coding", model_id="test-model", success=True,
            elapsed=0.5, complexity_score=3.0,
            input_tokens=100, output_tokens=200,
        )

        # The store.record is called in a background thread, but we can
        # verify the call was dispatched. Give the thread a moment.
        import time
        time.sleep(0.2)

        mock_store.record.assert_called_once_with(
            role="coding", complexity=3.0, model_id="test-model",
            success=True, latency=0.5,
            input_tokens=100, output_tokens=200,
        )

    def test_record_feedback_noop_without_store(self):
        """Without a feedback_store, _record_feedback does nothing."""
        cfg = _make_config()
        fabric = ComputeFabric(cfg)

        # Should not raise
        fabric._record_feedback(
            role="coding", model_id="m1", success=True, elapsed=0.1,
        )

    def test_record_feedback_default_complexity(self):
        """When complexity_score is None, default of 5.0 is used."""
        mock_store = MagicMock()
        cfg = _make_config()
        fabric = ComputeFabric(cfg, feedback_store=mock_store)

        fabric._record_feedback(
            role="reasoning", model_id="m2", success=False,
            elapsed=1.0, complexity_score=None,
        )

        import time
        time.sleep(0.2)

        call_kwargs = mock_store.record.call_args
        assert call_kwargs[1]["complexity"] == 5.0 or call_kwargs.kwargs.get("complexity") == 5.0
