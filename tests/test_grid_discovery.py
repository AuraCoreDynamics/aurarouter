import pytest
import asyncio
import httpx
import os
from unittest.mock import Mock, AsyncMock, patch
from aurarouter.grid_discovery import start_grid_discovery
from aurarouter.fabric import ComputeFabric

class TestGridDiscovery:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {
        "AURAGRID_IPC_PORT": "12345",
        "AURAGRID_IPC_TOKEN": "my-ipc-token",
        "AURAGRID_FENCING_TOKEN": "fence-token"
    })
    async def test_ipc_mode_discovery_success(self):
        # Mock fabric and config
        mock_fabric = Mock(spec=ComputeFabric)
        mock_fabric.config = Mock()
        mock_fabric.config.get_xlm_endpoint = Mock(return_value="http://old-endpoint/mcp/message")
        mock_fabric.config.config = {"xlm": {"endpoint": "http://old-endpoint/mcp/message"}}
        mock_fabric._provider_cache = {"xlm-augmentation": Mock()}
        mock_fabric._provider_cache_lock = Mock()
        mock_fabric._provider_cache_lock.__enter__ = Mock()
        mock_fabric._provider_cache_lock.__exit__ = Mock()

        # Mock httpx response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            "endpoints": [
                {
                    "serviceName": "auraxlm",
                    "openApiEndpoint": "http://localhost:5200/mcp/message"
                }
            ]
        })

        async def mock_get(url, headers, params):
            assert url == "http://localhost:12345/cell/registry/discover"
            assert headers["X-AuraGrid-IPC-Token"] == "my-ipc-token"
            assert headers["X-AuraGrid-Fencing-Token"] == "fence-token"
            assert params["serviceName"] == "auraxlm"
            return mock_response

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            async def mock_sleep(seconds):
                raise asyncio.CancelledError()

            with patch("asyncio.sleep", side_effect=mock_sleep):
                try:
                    await start_grid_discovery(mock_fabric, interval_seconds=1.0)
                except asyncio.CancelledError:
                    pass

        assert mock_fabric.config.config["xlm"]["endpoint"] == "http://localhost:5200/mcp/message"
        assert len(mock_fabric._provider_cache) == 0  # cleared

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_management_api_mode_discovery_success(self):
        mock_fabric = Mock(spec=ComputeFabric)
        mock_fabric.config = Mock()
        mock_fabric.config.get_xlm_endpoint = Mock(return_value="http://old-endpoint/mcp/message")
        mock_fabric.config.config = {"xlm": {"endpoint": "http://old-endpoint/mcp/message"}}
        mock_fabric._provider_cache = {"xlm-augmentation": Mock()}
        mock_fabric._provider_cache_lock = Mock()
        mock_fabric._provider_cache_lock.__enter__ = Mock()
        mock_fabric._provider_cache_lock.__exit__ = Mock()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=[
            {
                "serviceName": "auraxlm",
                "openApiEndpoint": "http://localhost:5200/mcp/message"
            }
        ])

        async def mock_get(url):
            assert url == "https://localhost:7087/api/discovery/endpoints"
            return mock_response

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(side_effect=mock_get)

        with patch("httpx.AsyncClient", return_value=mock_client):
            async def mock_sleep(seconds):
                raise asyncio.CancelledError()

            with patch("asyncio.sleep", side_effect=mock_sleep):
                try:
                    await start_grid_discovery(mock_fabric, interval_seconds=1.0)
                except asyncio.CancelledError:
                    pass

        assert mock_fabric.config.config["xlm"]["endpoint"] == "http://localhost:5200/mcp/message"
        assert len(mock_fabric._provider_cache) == 0

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_graceful_degradation_on_connection_error(self):
        mock_fabric = Mock(spec=ComputeFabric)
        mock_fabric.config = Mock()
        mock_fabric.config.get_xlm_endpoint = Mock(return_value="http://old-endpoint")
        mock_fabric.config.config = {"xlm": {"endpoint": "http://old-endpoint"}}
        mock_fabric._provider_cache = {"xlm-augmentation": Mock()}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("Connection refused"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            async def mock_sleep(seconds):
                assert seconds == 15.0
                raise asyncio.CancelledError()

            with patch("asyncio.sleep", side_effect=mock_sleep):
                try:
                    await start_grid_discovery(mock_fabric, interval_seconds=15.0)
                except asyncio.CancelledError:
                    pass

        assert mock_fabric.config.config["xlm"]["endpoint"] == "http://old-endpoint"
        assert len(mock_fabric._provider_cache) == 1
