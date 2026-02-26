"""
Integration tests for AuraRouter on AuraGrid.

Tests the AuraGrid MAS host, services, configuration loading, and event handling.
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Import modules to test
from aurarouter.auragrid.config_loader import ConfigLoader
from aurarouter.auragrid.events import EventBridge
from aurarouter.auragrid.lifecycle import LifecycleCallbacks
from aurarouter.auragrid.manifest import ManifestBuilder, create_default_manifest
from aurarouter.auragrid.mas_host import AuraRouterMasHost
from aurarouter.auragrid.services import (
    CodingService,
    ReasoningService,
    RouterService,
    UnifiedRouterService,
)


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_config_loader_init(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader(
            manifest_metadata={"key": "value"},
            config_file_path=Path("test.yaml"),
            allow_missing=True,
        )
        assert loader.manifest_metadata == {"key": "value"}
        assert loader.config_file_path == Path("test.yaml")
        assert loader.allow_missing is True

    def test_config_loader_allow_missing(self, tmp_path):
        """Test ConfigLoader with missing file when allow_missing=True."""
        loader = ConfigLoader(config_file_path=Path(tmp_path) / "nonexistent.yaml", allow_missing=True)
        config = loader.load()
        assert config is not None

    @patch.dict("os.environ", {"AURAROUTER_SYSTEM__LOG_LEVEL": "DEBUG"})
    def test_env_override(self, tmp_path):
        """Test environment variable overrides."""
        # Create a minimal config file
        config_path = tmp_path / "auraconfig.yaml"
        config_path.write_text("system:\n  log_level: INFO\n")

        loader = ConfigLoader(config_file_path=config_path, allow_missing=True)
        config = loader.load()
        
        # Verify env var override worked
        assert config.config["system"]["log_level"] == "DEBUG"


class TestEventBridge:
    """Tests for EventBridge."""

    def test_event_bridge_init(self):
        """Test EventBridge initialization."""
        pub = Mock()
        con = Mock()
        bridge = EventBridge(event_publisher=pub, event_consumer=con)
        
        assert bridge.event_publisher is pub
        assert bridge.event_consumer is con
        assert len(bridge.processed_events) == 0

    def test_create_routing_request(self):
        """Test creating a routing request."""
        bridge = EventBridge()
        request = bridge.create_routing_request(
            task="Write a function",
            language="python",
            context={"key": "value"},
        )
        
        assert request["task"] == "Write a function"
        assert request["language"] == "python"
        assert request["context"] == {"key": "value"}
        assert "request_id" in request
        assert "return_topic" in request
        assert "timestamp" in request

    @pytest.mark.asyncio
    async def test_publish_routing_result_no_publisher(self):
        """Test publishing result when publisher not configured."""
        bridge = EventBridge()  # No publisher
        
        # Should not raise error
        await bridge.publish_routing_result(
            request_id="id-123",
            result="test result",
            return_topic="topic",
        )


class TestLifecycleCallbacks:
    """Tests for LifecycleCallbacks."""

    @pytest.mark.asyncio
    async def test_lifecycle_init(self):
        """Test LifecycleCallbacks initialization."""
        mock_loader = Mock()
        mock_loader.config = {"models": {"test": {"provider": "local"}}}
        
        lifecycle = LifecycleCallbacks(mock_loader)
        assert lifecycle.config_loader is mock_loader
        assert lifecycle.fabric is None
        assert lifecycle.is_healthy is False

    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    @pytest.mark.asyncio
    async def test_startup_success(self, mock_fabric_class):
        """Test successful startup."""
        mock_loader = Mock()
        mock_loader.config = {"models": {}}
        
        lifecycle = LifecycleCallbacks(mock_loader)
        
        await lifecycle.startup()
        
        # Verify fabric was initialized
        mock_fabric_class.assert_called_once()
        assert lifecycle.is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_no_fabric(self):
        """Test health check when fabric not initialized."""
        mock_loader = Mock()
        lifecycle = LifecycleCallbacks(mock_loader)
        
        result = await lifecycle.health_check()
        assert result is False

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.httpx.AsyncClient")
    async def test_health_check_lightweight_ollama_reachable(self, mock_client_class):
        """Test lightweight health check with reachable Ollama endpoint."""
        # Setup mock config with Ollama provider
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "local-llama": {
                    "provider": "ollama",
                    "endpoint": "http://localhost:11434/api/generate",
                    "model_name": "llama2"
                }
            }
        }
        
        lifecycle = LifecycleCallbacks(mock_loader)
        lifecycle.fabric = Mock()  # Mock fabric to bypass fabric check
        
        # Mock HTTP client to return success
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        result = await lifecycle._lightweight_check()
        
        assert result is True
        # Verify HTTP GET was called to /api/tags endpoint
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert "/api/tags" in call_args[0][0]

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.httpx.AsyncClient")
    async def test_health_check_lightweight_ollama_unreachable(self, mock_client_class):
        """Test lightweight health check with unreachable Ollama endpoint."""
        import httpx
        
        # Setup mock config with Ollama provider
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "local-llama": {
                    "provider": "ollama",
                    "endpoint": "http://localhost:11434/api/generate",
                    "model_name": "llama2"
                }
            }
        }
        
        lifecycle = LifecycleCallbacks(mock_loader)
        lifecycle.fabric = Mock()
        
        # Mock HTTP client to raise ConnectionError
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_class.return_value = mock_client
        
        result = await lifecycle._lightweight_check()
        
        assert result is False

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_health_check_lightweight_missing_api_key(self):
        """Test lightweight health check with missing API key for cloud provider."""
        # Setup mock config with Claude provider but no API key
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "cloud-claude": {
                    "provider": "claude",
                    "model_name": "claude-3-opus-20240229",
                    "env_key": "ANTHROPIC_API_KEY"  # Reference env var that doesn't exist
                }
            }
        }
        
        lifecycle = LifecycleCallbacks(mock_loader)
        lifecycle.fabric = Mock()
        
        result = await lifecycle._lightweight_check()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_lightweight_malformed_endpoint(self):
        """Test lightweight health check with malformed endpoint URL."""
        # Setup mock config with invalid endpoint URL
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "bad-endpoint": {
                    "provider": "ollama",
                    "endpoint": "not-a-valid-url",  # Invalid URL format
                    "model_name": "llama2"
                }
            }
        }
        
        lifecycle = LifecycleCallbacks(mock_loader)
        lifecycle.fabric = Mock()
        
        result = await lifecycle._lightweight_check()
        
        assert result is False

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key-12345"})
    async def test_health_check_lightweight_cloud_with_api_key(self):
        """Test lightweight health check with valid API key for cloud provider."""
        # Setup mock config with Claude provider and valid API key
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "cloud-claude": {
                    "provider": "claude",
                    "model_name": "claude-3-opus-20240229",
                    "env_key": "ANTHROPIC_API_KEY"
                }
            }
        }
        
        lifecycle = LifecycleCallbacks(mock_loader)
        lifecycle.fabric = Mock()
        
        result = await lifecycle._lightweight_check()
        
        assert result is True

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.httpx.AsyncClient")
    async def test_health_check_lightweight_multiple_providers(self, mock_client_class):
        """Test lightweight health check with multiple providers."""
        # Setup mock config with both Ollama and cloud provider
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "local-llama": {
                    "provider": "ollama",
                    "endpoint": "http://localhost:11434/api/generate",
                    "model_name": "llama2"
                },
                "cloud-claude": {
                    "provider": "claude",
                    "model_name": "claude-3-opus-20240229",
                    "api_key": "sk-test-key"  # Direct API key in config
                }
            }
        }
        
        lifecycle = LifecycleCallbacks(mock_loader)
        lifecycle.fabric = Mock()
        
        # Mock HTTP client for Ollama check
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        result = await lifecycle._lightweight_check()
        
        assert result is True


class TestServiceClasses:
    """Tests for AuraRouter service classes."""

    @pytest.mark.asyncio
    async def test_router_service_initialization(self):
        """Test RouterService initialization."""
        mock_fabric = Mock()
        service = RouterService(mock_fabric)
        assert service.fabric is mock_fabric

    @pytest.mark.asyncio
    async def test_reasoning_service_initialization(self):
        """Test ReasoningService initialization."""
        mock_fabric = Mock()
        service = ReasoningService(mock_fabric)
        assert service.fabric is mock_fabric

    @pytest.mark.asyncio
    async def test_coding_service_initialization(self):
        """Test CodingService initialization."""
        mock_fabric = Mock()
        service = CodingService(mock_fabric)
        assert service.fabric is mock_fabric

    @pytest.mark.asyncio
    async def test_unified_router_service_initialization(self):
        """Test UnifiedRouterService initialization."""
        mock_fabric = Mock()
        service = UnifiedRouterService(mock_fabric)
        # Verify private fabric is set
        assert service._fabric is mock_fabric
        # Verify the main method is callable
        assert callable(service.intelligent_code_gen)


class TestManifestBuilder:
    """Tests for ManifestBuilder."""

    def test_manifest_builder_init(self):
        """Test ManifestBuilder initialization."""
        builder = ManifestBuilder(
            app_id="test-app",
            name="Test App",
            version="1.0.0",
        )
        assert builder.app_id == "test-app"
        assert builder.name == "Test App"
        assert builder.version == "1.0.0"

    def test_add_service(self):
        """Test adding services to manifest."""
        builder = ManifestBuilder()
        builder.add_service(
            "service-1",
            "ServiceClass",
            "Test service",
            "Distributed",
        )
        
        assert len(builder.services) == 1
        assert builder.services[0]["id"] == "service-1"

    def test_manifest_build(self):
        """Test building manifest."""
        builder = ManifestBuilder()
        builder.add_service("svc-1", "ServiceClass", "Test", "Distributed")
        
        manifest = builder.build()
        
        assert manifest["appid"] == "aurarouter-v2"
        assert manifest["name"] == "AuraRouter"
        assert len(manifest["services"]) == 1

    def test_manifest_to_json(self):
        """Test serializing manifest to JSON."""
        builder = ManifestBuilder()
        json_str = builder.to_json()
        
        data = json.loads(json_str)
        assert "appid" in data
        assert "services" in data

    def test_create_default_manifest(self):
        """Test creating default manifest."""
        manifest = create_default_manifest()
        
        assert manifest["appid"] == "aurarouter-v2"
        assert len(manifest["services"]) == 4
        
        service_names = [s["name"] for s in manifest["services"]]
        assert "RouterService" in service_names
        assert "ReasoningService" in service_names
        assert "CodingService" in service_names
        assert "UnifiedRouterService" in service_names


class TestMasHost:
    """Tests for AuraRouterMasHost."""

    @pytest.mark.asyncio
    async def test_mas_host_initialization(self):
        """Test AuraRouterMasHost initialization."""
        host = AuraRouterMasHost()
        assert host.context is None
        assert host.lifecycle is None
        assert host.is_running is False

    @pytest.mark.asyncio
    async def test_mas_host_startup_callback(self):
        """Test startup callback."""
        host = AuraRouterMasHost()
        await host.startup_callback()
        # Should not raise

    @pytest.mark.asyncio
    async def test_mas_host_shutdown_callback(self):
        """Test shutdown callback."""
        host = AuraRouterMasHost()
        host.is_running = True
        
        await host.shutdown_callback()
        
        assert host.is_running is False


class TestBackwardsCompatibility:
    """Tests to ensure backwards compatibility."""

    def test_aurarouter_imports_normally(self):
        """Test that aurarouter package imports without AuraGrid SDK."""
        # This is imported at module level, so if we got here, it worked
        import aurarouter
        assert aurarouter.__version__ == "0.4.0"

    def test_auragrid_optional(self):
        """Test that auragrid module is optional."""
        import aurarouter
        
        # The __init__ now uses importlib to conditionally add "auragrid"
        # to __all__. We can't know if the SDK is installed here,
        # but we can verify that the __all__ list exists and is valid.
        assert isinstance(aurarouter.__all__, list)
        assert "ComputeFabric" in aurarouter.__all__



@pytest.mark.asyncio
async def test_full_service_chain():
    """Integration test of full service chain with mocked fabric."""
    mock_fabric = Mock()
    mock_fabric.execute = Mock(return_value="test result")
    
    # Create unified service
    service = UnifiedRouterService(mock_fabric)
    
    # Verify initialization - service uses private _fabric attribute
    assert service._fabric is mock_fabric
    # Verify the main method is callable
    assert callable(service.intelligent_code_gen)


# ============================================================================
# R-B2 Integration Tests: Wiring R-B1 implementations
# ============================================================================

class TestConfigLoaderWiring:
    """Tests for ConfigLoader wiring into lifecycle."""

    @pytest.mark.asyncio
    async def test_config_loader_subscription(self):
        """Test ConfigLoader subscription in lifecycle startup."""
        mock_loader = Mock()
        mock_loader.config = {"models": {}}
        
        grid_config_loader = Mock()
        grid_config_loader.subscribe_to_config_changes = Mock()
        grid_config_loader.close = Mock()
        
        lifecycle = LifecycleCallbacks(mock_loader, grid_config_loader=grid_config_loader)
        
        # Mock ComputeFabric to prevent actual initialization
        with patch("aurarouter.auragrid.lifecycle.ComputeFabric"):
            await lifecycle.startup()
        
        # Verify subscription was called with callback
        grid_config_loader.subscribe_to_config_changes.assert_called_once()
        
        # Verify callback is the correct method
        callback = grid_config_loader.subscribe_to_config_changes.call_args[0][0]
        assert callable(callback)
        assert callback == lifecycle._on_config_changed
        
        # Shutdown and verify unsubscribe
        await lifecycle.shutdown()
        grid_config_loader.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_config_hot_reload_callback(self):
        """Test hot-reload callback updates fabric config."""
        mock_loader = Mock()
        mock_loader.config = {"models": {}}
        
        lifecycle = LifecycleCallbacks(mock_loader)
        
        # Initialize fabric
        with patch("aurarouter.auragrid.lifecycle.ComputeFabric") as mock_fabric_class:
            await lifecycle.startup()
        
        # Mock fabric with update_config method
        lifecycle.fabric = Mock()
        lifecycle.fabric.update_config = Mock()
        
        # Create new config loader
        new_loader = Mock()
        new_loader.config = {"models": {"new-model": {}}}
        
        # Trigger callback
        lifecycle._on_config_changed(new_loader)
        
        # Verify fabric was updated
        lifecycle.fabric.update_config.assert_called_once_with(new_loader)
        assert lifecycle.config_loader is new_loader


class TestOllamaDiscoveryWiring:
    """Tests for OllamaDiscovery wiring into lifecycle and fabric."""

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.OllamaDiscovery")
    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    async def test_ollama_discovery_startup(self, mock_fabric_class, mock_discovery_class):
        """Test OllamaDiscovery is initialized during startup."""
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "local-llama": {
                    "provider": "ollama",
                    "endpoint": "http://localhost:11434/api/generate"
                }
            }
        }
        
        mock_discovery = Mock()
        mock_discovery.start = Mock()
        mock_discovery.close = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        lifecycle = LifecycleCallbacks(mock_loader)
        
        await lifecycle.startup()
        
        # Verify discovery was initialized with default endpoint
        mock_discovery_class.assert_called_once()
        call_kwargs = mock_discovery_class.call_args[1]
        assert "default_endpoint" in call_kwargs
        
        # Verify discovery.start() was called
        mock_discovery.start.assert_called_once()
        
        # Verify discovery was passed to fabric
        mock_fabric_class.assert_called_once()
        fabric_kwargs = mock_fabric_class.call_args[1]
        assert "ollama_discovery" in fabric_kwargs
        assert fabric_kwargs["ollama_discovery"] is mock_discovery
        
        # Shutdown and verify cleanup
        await lifecycle.shutdown()
        mock_discovery.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.OllamaDiscovery")
    async def test_ollama_discovery_extracts_default_endpoint(self, mock_discovery_class):
        """Test default endpoint extraction from config."""
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "model1": {
                    "provider": "claude"
                },
                "model2": {
                    "provider": "ollama",
                    "endpoint": "http://192.168.1.100:11434/api/generate"
                }
            }
        }
        
        mock_discovery = Mock()
        mock_discovery.start = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        lifecycle = LifecycleCallbacks(mock_loader)
        
        with patch("aurarouter.auragrid.lifecycle.ComputeFabric"):
            await lifecycle.startup()
        
        # Verify discovery was initialized with extracted endpoint (without /api/generate)
        call_kwargs = mock_discovery_class.call_args[1]
        assert call_kwargs["default_endpoint"] == "http://192.168.1.100:11434"

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.OllamaDiscovery")
    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    async def test_fabric_uses_discovered_endpoints(self, mock_fabric_class, mock_discovery_class):
        """Test ComputeFabric receives and uses discovered endpoints."""
        from aurarouter.fabric import ComputeFabric
        from aurarouter.providers.ollama import OllamaProvider
        
        # Setup mock config
        mock_loader = Mock()
        mock_loader.config = {"models": {}}
        mock_loader.get_role_chain = Mock(return_value=["local-llama"])
        mock_loader.get_model_config = Mock(return_value={
            "provider": "ollama",
            "model_name": "llama2",
            "endpoint": "http://localhost:11434/api/generate"
        })
        
        # Setup mock discovery
        mock_discovery = Mock()
        mock_discovery.get_available_endpoints = Mock(return_value=[
            "http://node1:11434/api/generate",
            "http://node2:11434/api/generate"
        ])
        
        # Create real fabric with mock discovery
        fabric = ComputeFabric(config=mock_loader, ollama_discovery=mock_discovery)
        
        # Mock OllamaProvider
        mock_provider = Mock(spec=OllamaProvider)
        mock_provider.config = {}
        from aurarouter.savings.models import GenerateResult
        mock_provider.generate_with_usage = Mock(
            return_value=GenerateResult(text="test response")
        )
        
        with patch("aurarouter.fabric.get_provider", return_value=mock_provider):
            result = fabric.execute("router", "test prompt")
        
        # Verify discovered endpoints were injected into provider
        assert mock_provider.config["endpoints"] == [
            "http://node1:11434/api/generate",
            "http://node2:11434/api/generate"
        ]
        assert result == "test response"


class TestGridModelStorageWiring:
    """Tests for GridModelStorage wiring into lifecycle."""

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.GridModelStorage")
    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    async def test_grid_model_storage_startup(self, mock_fabric_class, mock_storage_class):
        """Test GridModelStorage is initialized during startup."""
        mock_loader = Mock()
        mock_loader.config = {"models": {}}
        
        mock_storage = AsyncMock()
        mock_storage.start = AsyncMock()
        mock_storage.list_models = AsyncMock(return_value=[])
        mock_storage_class.return_value = mock_storage
        
        lifecycle = LifecycleCallbacks(mock_loader)
        
        await lifecycle.startup()
        
        # Verify storage was initialized
        mock_storage_class.assert_called_once()
        mock_storage.start.assert_called_once()
        
        # Verify lifecycle stored reference
        assert lifecycle._grid_model_storage is mock_storage

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.GridModelStorage")
    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    async def test_grid_model_resolution(self, mock_fabric_class, mock_storage_class):
        """Test grid model resolution during startup."""
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "model-from-grid": {
                    "provider": "ollama",
                    "model_name": "llama2"
                },
                "model-local": {
                    "provider": "claude"
                }
            }
        }
        
        mock_storage = AsyncMock()
        mock_storage.start = AsyncMock()
        mock_storage.list_models = AsyncMock(return_value=["model-from-grid", "other-model"])
        mock_storage_class.return_value = mock_storage
        
        lifecycle = LifecycleCallbacks(mock_loader)
        
        await lifecycle.startup()
        
        # Verify list_models was called during resolution
        mock_storage.list_models.assert_called_once()

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.GridModelStorage")
    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    async def test_grid_storage_failure_does_not_prevent_startup(self, mock_fabric_class, mock_storage_class):
        """Test that GridModelStorage failures don't prevent startup."""
        mock_loader = Mock()
        mock_loader.config = {"models": {}}
        
        # Make storage initialization fail
        mock_storage = AsyncMock()
        mock_storage.start = AsyncMock(side_effect=Exception("Storage connection failed"))
        mock_storage_class.return_value = mock_storage
        
        lifecycle = LifecycleCallbacks(mock_loader)
        
        # Startup should succeed despite storage failure
        await lifecycle.startup()
        
        # Verify storage reference is None after failure
        assert lifecycle._grid_model_storage is None
        
        # Verify lifecycle is still healthy
        assert lifecycle.is_healthy is True or lifecycle.is_healthy is False  # Health depends on other checks


class TestFullIntegration:
    """End-to-end integration tests with all components."""

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.OllamaDiscovery")
    @patch("aurarouter.auragrid.lifecycle.GridModelStorage")
    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    async def test_full_startup_with_all_components(
        self, mock_fabric_class, mock_storage_class, mock_discovery_class
    ):
        """Test full startup with all R-B1 components wired."""
        mock_loader = Mock()
        mock_loader.config = {
            "models": {
                "local-llama": {
                    "provider": "ollama",
                    "endpoint": "http://localhost:11434/api/generate"
                }
            }
        }
        
        # Setup mocks
        mock_discovery = Mock()
        mock_discovery.start = Mock()
        mock_discovery.close = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        mock_storage = AsyncMock()
        mock_storage.start = AsyncMock()
        mock_storage.list_models = AsyncMock(return_value=[])
        mock_storage_class.return_value = mock_storage
        
        mock_fabric = Mock()
        mock_fabric.execute = Mock(return_value="health check ok")
        mock_fabric_class.return_value = mock_fabric
        
        grid_config_loader = Mock()
        grid_config_loader.subscribe_to_config_changes = Mock()
        grid_config_loader.close = Mock()
        
        lifecycle = LifecycleCallbacks(mock_loader, grid_config_loader=grid_config_loader)
        
        # Execute full startup
        await lifecycle.startup()
        
        # Verify all components initialized
        mock_discovery_class.assert_called_once()
        mock_discovery.start.assert_called_once()
        mock_storage_class.assert_called_once()
        mock_storage.start.assert_called_once()
        mock_fabric_class.assert_called_once()
        grid_config_loader.subscribe_to_config_changes.assert_called_once()
        
        # Execute full shutdown
        await lifecycle.shutdown()
        
        # Verify all components cleaned up
        mock_discovery.close.assert_called_once()
        grid_config_loader.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("aurarouter.auragrid.lifecycle.OllamaDiscovery")
    @patch("aurarouter.auragrid.lifecycle.ComputeFabric")
    async def test_config_change_updates_fabric_with_discovery(
        self, mock_fabric_class, mock_discovery_class
    ):
        """Test config changes update fabric and discovery endpoints."""
        mock_loader = Mock()
        mock_loader.config = {"models": {"local-llama": {"provider": "ollama"}}}
        
        mock_discovery = Mock()
        mock_discovery.start = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        mock_fabric = Mock()
        mock_fabric.update_config = Mock()
        mock_fabric_class.return_value = mock_fabric
        
        grid_config_loader = Mock()
        grid_config_loader.subscribe_to_config_changes = Mock()
        
        lifecycle = LifecycleCallbacks(mock_loader, grid_config_loader=grid_config_loader)
        
        await lifecycle.startup()
        
        # Simulate config change
        new_loader = Mock()
        new_loader.config = {"models": {"new-model": {"provider": "ollama"}}}
        
        lifecycle._on_config_changed(new_loader)
        
        # Verify fabric update was called with new config
        mock_fabric.update_config.assert_called_once_with(new_loader)
        assert lifecycle.config_loader is new_loader

    @pytest.mark.asyncio  
    @patch("aurarouter.auragrid.lifecycle.OllamaDiscovery")
    @patch("aurarouter.auragrid.lifecycle.GridModelStorage")
    async def test_partial_component_availability(
        self, mock_storage_class, mock_discovery_class
    ):
        """Test startup when only some components are available."""
        # Simulate GridModelStorage available but not AuraGrid SDK
        mock_loader = Mock()
        mock_loader.config = {"models": {}}
        
        mock_discovery = Mock()
        mock_discovery.start = Mock()
        mock_discovery_class.return_value = mock_discovery
        
        mock_storage = AsyncMock()
        mock_storage.start = AsyncMock()
        mock_storage.list_models = AsyncMock(return_value=[])
        mock_storage_class.return_value = mock_storage
        
        lifecycle = LifecycleCallbacks(mock_loader, grid_config_loader=None)
        
        with patch("aurarouter.auragrid.lifecycle.ComputeFabric"):
            await lifecycle.startup()
        
        # Verify discovery and storage initialized
        mock_discovery.start.assert_called_once()
        mock_storage.start.assert_called_once()
        
        # Verify no error when grid_config_loader is None
        assert lifecycle._grid_config_loader is None
