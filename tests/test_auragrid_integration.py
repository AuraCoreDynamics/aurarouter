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
        assert service.fabric is mock_fabric
        assert isinstance(service.router, RouterService)
        assert isinstance(service.reasoner, ReasoningService)
        assert isinstance(service.coder, CodingService)


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
        assert aurarouter.__version__ == "0.3.0"

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
    
    # This would work when fabric async execution is properly implemented
    # For now, it tests the initialization chain
    assert service.fabric is mock_fabric
    assert service.router is not None
    assert service.reasoner is not None
    assert service.coder is not None
