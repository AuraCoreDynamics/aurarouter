"""Tests for aurarouter.catalog -- ProviderCatalog discovery and lifecycle."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from aurarouter.catalog import CatalogEntry, ProviderCatalog
from aurarouter.config import ConfigLoader
from aurarouter.providers.protocol import (
    TOOL_GENERATE,
    TOOL_LIST_MODELS,
    ProviderMetadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog_config(tmp_path: Path) -> ConfigLoader:
    """ConfigLoader with provider_catalog section."""
    config_content = {
        "models": {},
        "roles": {},
        "provider_catalog": {
            "manual": [
                {
                    "name": "gemini",
                    "endpoint": "http://localhost:9001",
                    "auto_start": True,
                },
                {
                    "name": "custom-llm",
                    "endpoint": "http://localhost:9002",
                    "auto_start": False,
                },
            ],
            "auto_start_entrypoints": True,
        },
    }
    config_path = tmp_path / "auraconfig.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    return ConfigLoader(config_path=str(config_path))


@pytest.fixture
def empty_config(tmp_path: Path) -> ConfigLoader:
    """ConfigLoader with no provider_catalog section."""
    config_path = tmp_path / "auraconfig.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"models": {}, "roles": {}}, f)
    return ConfigLoader(config_path=str(config_path))


# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------

class TestGetBuiltinProviders:
    def test_returns_four_entries(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        builtins = catalog.get_builtin_providers()
        assert len(builtins) == 4

    def test_builtin_names(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        names = {e.name for e in catalog.get_builtin_providers()}
        assert names == {"ollama", "llamacpp-server", "llamacpp", "openapi"}

    def test_all_are_builtin_source(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        for entry in catalog.get_builtin_providers():
            assert entry.source == "builtin"

    def test_builtin_entries_have_descriptions(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        for entry in catalog.get_builtin_providers():
            assert entry.description != ""


# ---------------------------------------------------------------------------
# Manual providers
# ---------------------------------------------------------------------------

class TestManualProviders:
    def test_get_manual_providers(self, catalog_config):
        catalog = ProviderCatalog(catalog_config)
        manual = catalog.get_manual_providers()
        assert len(manual) == 2
        names = {e.name for e in manual}
        assert "gemini" in names
        assert "custom-llm" in names

    def test_manual_source(self, catalog_config):
        catalog = ProviderCatalog(catalog_config)
        for entry in catalog.get_manual_providers():
            assert entry.source == "manual"
            assert entry.provider_type == "mcp"

    def test_no_manual_entries(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        assert catalog.get_manual_providers() == []


# ---------------------------------------------------------------------------
# Register / unregister manual
# ---------------------------------------------------------------------------

class TestRegisterManual:
    def test_register_manual(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        catalog.discover()
        entry = catalog.register_manual("new-provider", "http://localhost:9999")
        assert entry.name == "new-provider"
        assert entry.source == "manual"
        assert "new-provider" in catalog._entries

        # Verify it's in config
        manual = empty_config.get_catalog_manual_entries()
        assert any(e["name"] == "new-provider" for e in manual)

    def test_unregister_manual(self, catalog_config):
        catalog = ProviderCatalog(catalog_config)
        catalog.discover()
        assert catalog.unregister_manual("gemini") is True
        assert "gemini" not in catalog._entries

        # Verify removed from config
        manual = catalog_config.get_catalog_manual_entries()
        assert not any(e["name"] == "gemini" for e in manual)

    def test_unregister_nonexistent(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        assert catalog.unregister_manual("does-not-exist") is False


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_discover_aggregates_all_sources(self, catalog_config):
        catalog = ProviderCatalog(catalog_config)
        entries = catalog.discover()
        names = {e.name for e in entries}
        # Should have builtins + manual
        assert "ollama" in names
        assert "openapi" in names
        assert "gemini" in names
        assert "custom-llm" in names

    def test_discover_clears_previous(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        catalog._entries["stale"] = CatalogEntry(
            name="stale", provider_type="mcp", source="manual"
        )
        catalog.discover()
        assert "stale" not in catalog._entries


# ---------------------------------------------------------------------------
# Entry point providers
# ---------------------------------------------------------------------------

class TestEntryPointProviders:
    def test_loads_entrypoints(self, empty_config):
        mock_ep = MagicMock()
        mock_ep.name = "test_provider"
        mock_ep.load.return_value = lambda: ProviderMetadata(
            name="test-ep",
            provider_type="mcp",
            version="1.0.0",
            description="Test entry point provider",
            command=["python", "-m", "test_provider.server"],
        )

        with patch(
            "aurarouter.catalog.entry_points",
            return_value=[mock_ep],
        ):
            catalog = ProviderCatalog(empty_config)
            eps = catalog.get_entrypoint_providers()
            assert len(eps) == 1
            assert eps[0].name == "test-ep"
            assert eps[0].source == "entrypoint"
            assert eps[0].version == "1.0.0"

    def test_handles_broken_entrypoint(self, empty_config):
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("package not found")

        with patch(
            "aurarouter.catalog.entry_points",
            return_value=[mock_ep],
        ):
            catalog = ProviderCatalog(empty_config)
            eps = catalog.get_entrypoint_providers()
            assert eps == []


# ---------------------------------------------------------------------------
# Auto-register models
# ---------------------------------------------------------------------------

class TestAutoRegisterModels:
    def test_auto_register_models(self, catalog_config):
        catalog = ProviderCatalog(catalog_config)
        catalog.discover()

        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.connect.return_value = True
        mock_client.get_capabilities.return_value = {
            TOOL_GENERATE, TOOL_LIST_MODELS
        }
        mock_client.call_tool.return_value = [
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
            {"id": "gemini-2.0-pro", "name": "Gemini 2.0 Pro"},
        ]

        catalog._clients["gemini"] = mock_client

        added = catalog.auto_register_models("gemini", catalog_config)
        assert added == 2

        # Verify models are in config
        cfg = catalog_config.get_model_config("gemini/gemini-2.0-flash")
        assert cfg["provider"] == "mcp"
        assert cfg["mcp_endpoint"] == "http://localhost:9001"

    def test_auto_register_skips_existing(self, catalog_config):
        catalog = ProviderCatalog(catalog_config)
        catalog.discover()

        # Pre-register one model
        catalog_config.set_model("gemini/existing", {"provider": "mcp"})

        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.get_capabilities.return_value = {
            TOOL_GENERATE, TOOL_LIST_MODELS
        }
        mock_client.call_tool.return_value = [{"id": "existing"}]

        catalog._clients["gemini"] = mock_client
        added = catalog.auto_register_models("gemini", catalog_config)
        assert added == 0

    def test_auto_register_no_endpoint(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        catalog.discover()
        added = catalog.auto_register_models("nonexistent", empty_config)
        assert added == 0


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_unknown_provider(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        healthy, msg = catalog.check_provider_health("unknown")
        assert healthy is False
        assert "not found" in msg

    def test_no_endpoint(self, empty_config):
        catalog = ProviderCatalog(empty_config)
        catalog._entries["test"] = CatalogEntry(
            name="test", provider_type="mcp", source="manual"
        )
        healthy, msg = catalog.check_provider_health("test")
        assert healthy is False
        assert "No endpoint" in msg
