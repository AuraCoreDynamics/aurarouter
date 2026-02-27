"""Tests for McpClientRegistry."""

from unittest.mock import MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.mcp_client.client import GridMcpClient
from aurarouter.mcp_client.registry import McpClientRegistry


class TestRegistration:
    def test_register_and_get(self):
        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        reg.register("svc1", client)
        assert "svc1" in reg.get_clients()
        assert reg.get_clients()["svc1"] is client

    def test_unregister_existing(self):
        reg = McpClientRegistry()
        reg.register("svc1", MagicMock(spec=GridMcpClient))
        assert reg.unregister("svc1") is True
        assert "svc1" not in reg.get_clients()

    def test_unregister_nonexistent(self):
        reg = McpClientRegistry()
        assert reg.unregister("nope") is False

    def test_get_clients_returns_copy(self):
        reg = McpClientRegistry()
        reg.register("a", MagicMock(spec=GridMcpClient))
        clients = reg.get_clients()
        clients["b"] = MagicMock()  # mutate the copy
        assert "b" not in reg.get_clients()  # original unchanged


class TestCapabilityLookup:
    def test_get_clients_with_capability(self):
        reg = McpClientRegistry()
        c1 = MagicMock(spec=GridMcpClient)
        c1.connected = True
        c1.get_capabilities.return_value = {"chain_reorder", "rag_query"}

        c2 = MagicMock(spec=GridMcpClient)
        c2.connected = True
        c2.get_capabilities.return_value = {"rag_query"}

        reg.register("c1", c1)
        reg.register("c2", c2)

        advisors = reg.get_clients_with_capability("chain_reorder")
        assert len(advisors) == 1
        assert c1 in advisors

        rag_clients = reg.get_clients_with_capability("rag_query")
        assert len(rag_clients) == 2

    def test_disconnected_clients_excluded(self):
        reg = McpClientRegistry()
        c = MagicMock(spec=GridMcpClient)
        c.connected = False
        c.get_capabilities.return_value = {"chain_reorder"}
        reg.register("offline", c)

        assert reg.get_clients_with_capability("chain_reorder") == []

    def test_no_matching_capability(self):
        reg = McpClientRegistry()
        c = MagicMock(spec=GridMcpClient)
        c.connected = True
        c.get_capabilities.return_value = {"models"}
        reg.register("c", c)

        assert reg.get_clients_with_capability("chain_reorder") == []


class TestRemoteTools:
    def test_get_all_remote_tools_aggregates(self):
        reg = McpClientRegistry()
        c1 = MagicMock(spec=GridMcpClient)
        c1.connected = True
        c1.get_tools.return_value = [{"name": "tool1"}, {"name": "tool2"}]

        c2 = MagicMock(spec=GridMcpClient)
        c2.connected = True
        c2.get_tools.return_value = [{"name": "tool3"}]

        reg.register("svc1", c1)
        reg.register("svc2", c2)

        tools = reg.get_all_remote_tools()
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"tool1", "tool2", "tool3"}

    def test_remote_tools_tagged_with_source(self):
        reg = McpClientRegistry()
        c = MagicMock(spec=GridMcpClient)
        c.connected = True
        c.get_tools.return_value = [{"name": "search"}]
        reg.register("xlm", c)

        tools = reg.get_all_remote_tools()
        assert tools[0]["_source_client"] == "xlm"

    def test_disconnected_clients_excluded(self):
        reg = McpClientRegistry()
        c = MagicMock(spec=GridMcpClient)
        c.connected = False
        c.get_tools.return_value = [{"name": "hidden"}]
        reg.register("offline", c)

        assert reg.get_all_remote_tools() == []


class TestSyncModels:
    def test_sync_adds_models(self):
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        client.connected = True
        client.base_url = "http://host:8080"
        client.get_models.return_value = [
            {"id": "model-a", "provider": "ollama"},
            {"id": "model-b"},
        ]
        reg.register("grid1", client)

        added = reg.sync_models(config)
        assert added == 2
        assert "grid1/model-a" in config.get_all_model_ids()
        assert "grid1/model-b" in config.get_all_model_ids()

        # Verify model config structure
        cfg = config.get_model_config("grid1/model-a")
        assert cfg["provider"] == "ollama"
        assert cfg["endpoint"] == "http://host:8080"
        assert cfg["model_name"] == "model-a"
        assert "remote" in cfg["tags"]
        assert "grid:grid1" in cfg["tags"]

        # Default provider is openapi
        cfg_b = config.get_model_config("grid1/model-b")
        assert cfg_b["provider"] == "openapi"

    def test_sync_idempotent(self):
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        client.connected = True
        client.base_url = "http://host:8080"
        client.get_models.return_value = [{"id": "model-a"}]
        reg.register("svc", client)

        assert reg.sync_models(config) == 1
        assert reg.sync_models(config) == 0  # already present

    def test_sync_skips_disconnected(self):
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        client.connected = False
        client.get_models.return_value = [{"id": "model-a"}]
        reg.register("offline", client)

        assert reg.sync_models(config) == 0

    def test_sync_uses_name_field_as_fallback(self):
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        client.connected = True
        client.base_url = "http://host:8080"
        client.get_models.return_value = [{"name": "my-model"}]  # no "id" key
        reg.register("svc", client)

        assert reg.sync_models(config) == 1
        assert "svc/my-model" in config.get_all_model_ids()

    def test_sync_skips_empty_model_id(self):
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        client.connected = True
        client.base_url = "http://host:8080"
        client.get_models.return_value = [{}]  # no id or name
        reg.register("svc", client)

        assert reg.sync_models(config) == 0

    def test_sync_with_discovery_tool(self):
        """sync_models calls discover_models when tool name is provided."""
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        client.connected = True
        client.base_url = "http://host:8080"
        client.get_models.return_value = [{"id": "discovered-model"}]
        reg.register("svc", client)

        added = reg.sync_models(config, model_discovery_tool="custom.list_models")
        assert added == 1
        client.discover_models.assert_called_once_with("custom.list_models")
        assert "svc/discovered-model" in config.get_all_model_ids()

    def test_sync_without_discovery_tool_skips_probing(self):
        """sync_models without tool name does not call discover_models."""
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

        reg = McpClientRegistry()
        client = MagicMock(spec=GridMcpClient)
        client.connected = True
        client.base_url = "http://host:8080"
        client.get_models.return_value = []  # no pre-populated models
        reg.register("svc", client)

        added = reg.sync_models(config, model_discovery_tool=None)
        assert added == 0
        client.discover_models.assert_not_called()
