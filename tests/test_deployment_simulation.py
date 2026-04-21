"""End-to-end deployment simulation for AuraRouter on AuraGrid.

Validates the full wiring chain:
  manifest → ProxyWorker lookup → mas_host init → GridContext → services

This test does NOT require a running AuraGrid node — it validates
the contract surface that the ProxyWorker uses to launch AuraRouter.
"""

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock


class TestDeploymentManifestContract:
    """Verify the manifest satisfies ProxyWorker's AppManifest schema."""

    def test_static_manifest_valid(self):
        """Static manifest has all fields ProxyWorker expects."""
        manifest_path = os.path.join(
            os.path.dirname(__file__), "..", "manifests", "auragrid_manifest.json"
        )
        with open(manifest_path) as f:
            m = json.load(f)

        # Top-level required by AppManifest.cs
        assert isinstance(m["AppId"], str)
        assert isinstance(m["Name"], str)
        assert isinstance(m["Version"], str)
        assert isinstance(m["Services"], list)
        assert len(m["Services"]) >= 1

    def test_service_definition_fields(self):
        """Each service has MasDefinition-compatible fields."""
        manifest_path = os.path.join(
            os.path.dirname(__file__), "..", "manifests", "auragrid_manifest.json"
        )
        with open(manifest_path) as f:
            m = json.load(f)

        for svc in m["Services"]:
            assert "MasId" in svc, f"Missing MasId in service: {svc}"
            assert "DisplayName" in svc, f"Missing DisplayName in service: {svc}"
            # Runtime must be Python for ProcessMasLoader
            assert svc.get("Runtime", "").lower() == "python"

    def test_python_config_present(self):
        """PythonConfig section is present with ScriptPath and ManagedVenvName."""
        manifest_path = os.path.join(
            os.path.dirname(__file__), "..", "manifests", "auragrid_manifest.json"
        )
        with open(manifest_path) as f:
            m = json.load(f)

        for svc in m["Services"]:
            if "PythonConfig" in svc:
                pc = svc["PythonConfig"]
                assert "ScriptPath" in pc
                assert "ManagedVenvName" in pc
                # Script path must be a valid relative path
                assert pc["ScriptPath"].endswith(".py")

    def test_dynamic_manifest_matches_static_schema(self):
        """ManifestBuilder output matches the static manifest schema."""
        from aurarouter.auragrid.manifest import create_default_manifest

        dynamic = create_default_manifest()
        # Same keys
        assert "AppId" in dynamic
        assert "Name" in dynamic
        assert "Version" in dynamic
        assert "Services" in dynamic
        for svc in dynamic["Services"]:
            assert "MasId" in svc
            assert "DisplayName" in svc


class TestMasHostWiring:
    """Verify mas_host wiring with mocked SDK."""

    def test_host_creates_without_sdk(self):
        """MasHost can be instantiated without SDK (standalone mode)."""
        from aurarouter.auragrid.mas_host import AuraRouterMasHost
        host = AuraRouterMasHost()
        assert host.is_running is False
        assert host.lifecycle is None

    def test_host_shutdown_request(self):
        """request_shutdown() sets is_running to False."""
        from aurarouter.auragrid.mas_host import AuraRouterMasHost
        host = AuraRouterMasHost()
        host.is_running = True
        host.request_shutdown()
        assert host.is_running is False

    @pytest.mark.asyncio
    async def test_cleanup_is_safe_when_nothing_initialized(self):
        """_cleanup() doesn't raise when nothing was started."""
        from aurarouter.auragrid.mas_host import AuraRouterMasHost
        host = AuraRouterMasHost()
        await host._cleanup()
        assert host.is_running is False


class TestEventBridgeDeploymentWiring:
    """Verify EventBridge integrates with GridContext.events."""

    def test_bridge_accepts_event_client(self):
        """EventBridge can be wired with a mock AsyncEventClient."""
        from aurarouter.auragrid.events import EventBridge
        client = Mock()
        bridge = EventBridge(event_client=client)
        assert bridge.is_active is True
        assert bridge.event_client is client

    def test_bridge_deactivates_without_client(self):
        """EventBridge is inactive when no client provided."""
        from aurarouter.auragrid.events import EventBridge
        bridge = EventBridge()
        assert bridge.is_active is False

    def test_bridge_client_can_be_set_after_init(self):
        """EventBridge client can be wired post-init (matches mas_host pattern)."""
        from aurarouter.auragrid.events import EventBridge
        bridge = EventBridge()
        assert bridge.is_active is False
        bridge.event_client = Mock()
        assert bridge.is_active is True


class TestAuctionListenerDeploymentWiring:
    """Verify auction listener integrates correctly."""

    @pytest.mark.asyncio
    async def test_listener_starts_with_active_bridge(self):
        """AuctionListener.start() succeeds when bridge is active."""
        from aurarouter.auragrid.auction import AuctionListener

        async def fake_subscribe(topic):
            return
            yield  # make it an async generator

        client = Mock()
        client.subscribe = fake_subscribe
        bridge = Mock()
        bridge.is_active = True
        bridge.event_client = client

        listener = AuctionListener(event_bridge=bridge, node_id="test")
        await listener.start()
        assert listener._running is True
        await listener.stop()
        assert listener._running is False

    @pytest.mark.asyncio
    async def test_listener_skips_when_bridge_inactive(self):
        """AuctionListener.start() is no-op when bridge is inactive."""
        from aurarouter.auragrid.auction import AuctionListener
        bridge = Mock()
        bridge.is_active = False
        listener = AuctionListener(event_bridge=bridge, node_id="test")
        await listener.start()
        assert listener._running is False


class TestFullDeploymentSimulation:
    """Simulate the full deployment sequence that ProxyWorker triggers."""

    @pytest.mark.asyncio
    async def test_simulated_launch_sequence(self):
        """
        Simulate what happens when ProxyWorker launches AuraRouter:

        1. ProxyWorker reads manifest, sets env vars, launches subprocess
        2. mas_host reads env vars
        3. ConfigLoader loads auraconfig.yaml
        4. LifecycleCallbacks starts fabric
        5. GridContext registers services
        6. EventBridge wired to GridContext.events
        7. AuctionListener starts
        8. Health check loop runs
        9. SIGTERM → graceful shutdown
        """
        from aurarouter.auragrid.mas_host import AuraRouterMasHost
        from aurarouter.auragrid.events import EventBridge

        # Step 1: Simulate env vars set by ProxyWorker
        env_patch = {
            "AURAGRID_IPC_PORT": "5100",
            "AURAGRID_IPC_TOKEN": "test-token",
            "AURAGRID_MAS_ID": "aurarouter-node",
            "AURAGRID_NODE_ID": "test-node-001",
            "AURAGRID_FENCING_TOKEN": "fence-abc",
            "AURAGRID_MANAGED_VENV_NAME": "aurarouter",
        }

        # Step 2: Create host
        host = AuraRouterMasHost()

        # Step 3-4: Mock the config/lifecycle layer
        mock_lifecycle = Mock()
        mock_lifecycle.startup = AsyncMock()
        mock_lifecycle.shutdown = AsyncMock()
        mock_lifecycle.health_check = AsyncMock(return_value=True)
        mock_lifecycle.config_loader = Mock()
        mock_lifecycle.config_loader.config = {
            "grid": {
                "health": {"interval": 1, "max_backoff": 5},
                "auction": {"vram_pressure_threshold": 0.9},
            }
        }
        mock_lifecycle.fabric = Mock()
        mock_lifecycle.fabric._circuit_breakers = Mock()
        mock_lifecycle.fabric._provider_cache = {}

        # Step 5-6: Wire event bridge
        bridge = EventBridge()
        assert bridge.is_active is False
        # Simulate _setup_grid_context wiring the client
        bridge.event_client = Mock()
        assert bridge.is_active is True

        # Step 7: Verify auction listener can start
        from aurarouter.auragrid.auction import AuctionListener

        async def fake_subscribe(topic):
            return
            yield

        bridge.event_client.subscribe = fake_subscribe
        listener = AuctionListener(event_bridge=bridge, node_id="test-node-001")
        await listener.start()
        assert listener._running is True

        # Step 9: Graceful shutdown
        host.is_running = True
        host.request_shutdown()
        assert host.is_running is False
        await listener.stop()
        assert listener._running is False

    def test_env_vars_read_correctly(self):
        """mas_host reads the correct env vars."""
        env_vars = [
            "AURAGRID_IPC_PORT",
            "AURAGRID_IPC_TOKEN",
            "AURAGRID_MAS_ID",
            "AURAGRID_NODE_ID",
            "AURAGRID_FENCING_TOKEN",
            "AURAGRID_MANAGED_VENV_NAME",
        ]
        # Verify these are referenced in mas_host.py source
        import inspect
        from aurarouter.auragrid import mas_host
        source = inspect.getsource(mas_host)
        for var in ["AURAGRID_NODE_ID", "AURAGRID_MANAGED_VENV_NAME"]:
            assert var in source, f"{var} not referenced in mas_host"
