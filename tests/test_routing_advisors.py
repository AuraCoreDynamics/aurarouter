"""Tests for the public routing advisor API on ComputeFabric (Task Group 3)."""

from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_client.registry import McpClientRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fabric(catalog: dict | None = None, **kwargs) -> ComputeFabric:
    """Create a ComputeFabric with an in-memory ConfigLoader."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        "roles": {"coding": ["m1", "m2"]},
    }
    if catalog is not None:
        cfg.config["catalog"] = catalog
    return ComputeFabric(cfg, **kwargs)


def _mock_advisor(name: str = "advisor1", chain_result: list[str] | None = None,
                  connected: bool = True, capabilities: set[str] | None = None,
                  side_effect: Exception | None = None) -> MagicMock:
    """Create a mock advisor client."""
    client = MagicMock()
    client.name = name
    client.connected = connected
    client.get_capabilities.return_value = capabilities or {"chain_reorder"}
    if side_effect:
        client.call_tool.side_effect = side_effect
    elif chain_result is not None:
        client.call_tool.return_value = {"chain": chain_result}
    else:
        client.call_tool.return_value = {"chain": []}
    return client


# ---------------------------------------------------------------------------
# T3.1: consult_routing_advisors() — public API
# ---------------------------------------------------------------------------

class TestConsultRoutingAdvisors:
    """Tests for the promoted public consult_routing_advisors() method."""

    def test_no_advisors_returns_original_chain(self):
        """When no registry is set, the original chain is returned."""
        fabric = _make_fabric()
        result = fabric.consult_routing_advisors("coding", ["m1", "m2"])
        assert result == ["m1", "m2"]

    def test_advisor_returns_reordered_chain(self):
        """A connected advisor with chain_reorder can reorder the chain."""
        advisor = _mock_advisor(chain_result=["m2", "m1"])
        registry = McpClientRegistry()
        registry.register("advisor1", advisor)

        fabric = _make_fabric(routing_advisors=registry)
        result = fabric.consult_routing_advisors("coding", ["m1", "m2"])
        assert result == ["m2", "m1"]

    def test_intent_passed_to_advisor(self):
        """When intent is provided, it is forwarded to the advisor's call_tool."""
        advisor = _mock_advisor(chain_result=["m2", "m1"])
        registry = McpClientRegistry()
        registry.register("advisor1", advisor)

        fabric = _make_fabric(routing_advisors=registry)
        fabric.consult_routing_advisors("coding", ["m1", "m2"], intent="SIMPLE_CODE")

        # Verify call_tool was called with intent in kwargs
        advisor.call_tool.assert_called_once_with(
            "chain_reorder", role="coding", chain=["m1", "m2"], intent="SIMPLE_CODE",
        )

    def test_intent_none_not_passed_to_advisor(self):
        """When intent is None, the intent kwarg is omitted from call_tool."""
        advisor = _mock_advisor(chain_result=["m2", "m1"])
        registry = McpClientRegistry()
        registry.register("advisor1", advisor)

        fabric = _make_fabric(routing_advisors=registry)
        fabric.consult_routing_advisors("coding", ["m1", "m2"], intent=None)

        # Verify call_tool was called WITHOUT intent
        advisor.call_tool.assert_called_once_with(
            "chain_reorder", role="coding", chain=["m1", "m2"],
        )

    def test_advisor_timeout_returns_original_chain(self):
        """If the advisor raises an exception (e.g. timeout), original chain is returned."""
        advisor = _mock_advisor(side_effect=TimeoutError("deadline exceeded"))
        registry = McpClientRegistry()
        registry.register("slow_advisor", advisor)

        fabric = _make_fabric(routing_advisors=registry)
        result = fabric.consult_routing_advisors("coding", ["m1", "m2"])
        assert result == ["m1", "m2"]

    def test_advisor_empty_chain_returns_original(self):
        """If advisor returns empty chain list, original chain is preserved."""
        advisor = _mock_advisor(chain_result=[])
        registry = McpClientRegistry()
        registry.register("empty_advisor", advisor)

        fabric = _make_fabric(routing_advisors=registry)
        result = fabric.consult_routing_advisors("coding", ["m1"])
        assert result == ["m1"]

    def test_disconnected_advisor_skipped(self):
        """Disconnected advisors are skipped."""
        advisor = _mock_advisor(connected=False, chain_result=["m2", "m1"])
        registry = McpClientRegistry()
        registry.register("offline", advisor)

        fabric = _make_fabric(routing_advisors=registry)
        result = fabric.consult_routing_advisors("coding", ["m1", "m2"])
        assert result == ["m1", "m2"]
        advisor.call_tool.assert_not_called()

    def test_advisor_without_capability_skipped(self):
        """Advisors without chain_reorder capability are skipped."""
        advisor = _mock_advisor(capabilities={"rag_query"}, chain_result=["m2", "m1"])
        registry = McpClientRegistry()
        registry.register("rag_only", advisor)

        fabric = _make_fabric(routing_advisors=registry)
        result = fabric.consult_routing_advisors("coding", ["m1", "m2"])
        assert result == ["m1", "m2"]
        advisor.call_tool.assert_not_called()


# ---------------------------------------------------------------------------
# T3.2: register/unregister/list routing advisors
# ---------------------------------------------------------------------------

class TestAdvisorRegistration:
    """Tests for advisor registration, unregistration, and listing."""

    def test_register_routing_advisor(self):
        """Registering an advisor makes it visible in list_routing_advisors."""
        fabric = _make_fabric()
        advisor = _mock_advisor(name="xlm-advisor")
        fabric.register_routing_advisor(advisor)
        assert "xlm-advisor" in fabric.list_routing_advisors()

    def test_register_is_idempotent(self):
        """Re-registering the same advisor (by name) is a no-op."""
        fabric = _make_fabric()
        advisor = _mock_advisor(name="xlm-advisor")
        fabric.register_routing_advisor(advisor)
        fabric.register_routing_advisor(advisor)
        assert fabric.list_routing_advisors().count("xlm-advisor") == 1

    def test_register_creates_registry_if_none(self):
        """Registering when no registry exists creates one automatically."""
        fabric = _make_fabric()
        assert fabric._routing_advisors is None
        advisor = _mock_advisor(name="first")
        fabric.register_routing_advisor(advisor)
        assert fabric._routing_advisors is not None
        assert "first" in fabric.list_routing_advisors()

    def test_unregister_routing_advisor(self):
        """Unregistering removes the advisor from the list."""
        fabric = _make_fabric()
        advisor = _mock_advisor(name="to-remove")
        fabric.register_routing_advisor(advisor)
        assert "to-remove" in fabric.list_routing_advisors()
        fabric.unregister_routing_advisor("to-remove")
        assert "to-remove" not in fabric.list_routing_advisors()

    def test_unregister_nonexistent_is_noop(self):
        """Unregistering a non-existent advisor does not raise."""
        fabric = _make_fabric()
        fabric.unregister_routing_advisor("ghost")  # No registry yet — should not raise

        advisor = _mock_advisor(name="real")
        fabric.register_routing_advisor(advisor)
        fabric.unregister_routing_advisor("ghost")  # Not in registry — no-op
        assert fabric.list_routing_advisors() == ["real"]

    def test_list_routing_advisors_empty(self):
        """When no advisors are registered, returns empty list."""
        fabric = _make_fabric()
        assert fabric.list_routing_advisors() == []

    def test_list_routing_advisors_multiple(self):
        """Multiple advisors are all listed."""
        fabric = _make_fabric()
        for name in ["a", "b", "c"]:
            fabric.register_routing_advisor(_mock_advisor(name=name))
        result = fabric.list_routing_advisors()
        assert sorted(result) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# T3.3: Auto-registration from catalog
# ---------------------------------------------------------------------------

class TestAutoRegisterCatalogAdvisors:
    """Tests for auto-registration of advisors from the catalog."""

    def test_auto_registers_from_catalog(self):
        """Services with capabilities: [routing_advisor] are auto-registered."""
        catalog = {
            "xlm-advisor": {
                "kind": "service",
                "display_name": "AuraXLM Advisor",
                "capabilities": ["routing_advisor"],
                "endpoint": "http://xlm:8080",
            },
            "plain-service": {
                "kind": "service",
                "display_name": "Some Service",
                "capabilities": ["rag_query"],
                "endpoint": "http://rag:9090",
            },
        }
        fabric = _make_fabric(catalog=catalog)
        count = fabric._auto_register_catalog_advisors()
        assert count == 1
        assert "xlm-advisor" in fabric.list_routing_advisors()
        assert "plain-service" not in fabric.list_routing_advisors()

    def test_auto_register_skips_entries_without_endpoint(self):
        """Catalog entries without an endpoint are skipped."""
        catalog = {
            "no-endpoint": {
                "kind": "service",
                "display_name": "No Endpoint",
                "capabilities": ["routing_advisor"],
                # No endpoint key
            },
        }
        fabric = _make_fabric(catalog=catalog)
        count = fabric._auto_register_catalog_advisors()
        assert count == 0
        assert fabric.list_routing_advisors() == []

    def test_auto_register_uses_mcp_endpoint_field(self):
        """The mcp_endpoint field is preferred over endpoint."""
        catalog = {
            "xlm-mcp": {
                "kind": "service",
                "display_name": "XLM via MCP",
                "capabilities": ["routing_advisor"],
                "mcp_endpoint": "http://xlm-mcp:8080",
            },
        }
        fabric = _make_fabric(catalog=catalog)
        count = fabric._auto_register_catalog_advisors()
        assert count == 1
        assert "xlm-mcp" in fabric.list_routing_advisors()

    def test_auto_register_empty_catalog(self):
        """No catalog entries means zero registrations."""
        fabric = _make_fabric(catalog={})
        count = fabric._auto_register_catalog_advisors()
        assert count == 0

    def test_auto_register_idempotent(self):
        """Calling auto-register twice does not duplicate advisors."""
        catalog = {
            "xlm-advisor": {
                "kind": "service",
                "display_name": "AuraXLM",
                "capabilities": ["routing_advisor"],
                "endpoint": "http://xlm:8080",
            },
        }
        fabric = _make_fabric(catalog=catalog)
        fabric._auto_register_catalog_advisors()
        fabric._auto_register_catalog_advisors()
        assert fabric.list_routing_advisors().count("xlm-advisor") == 1


# ---------------------------------------------------------------------------
# Integration: consult_routing_advisors with registered advisors
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests combining registration + consultation."""

    def test_registered_advisor_consulted(self):
        """An advisor registered via register_routing_advisor is consulted."""
        fabric = _make_fabric()
        advisor = _mock_advisor(name="dynamic", chain_result=["m2", "m1"])
        fabric.register_routing_advisor(advisor)

        result = fabric.consult_routing_advisors("coding", ["m1", "m2"], intent="sar_processing")
        assert result == ["m2", "m1"]
        advisor.call_tool.assert_called_once_with(
            "chain_reorder", role="coding", chain=["m1", "m2"], intent="sar_processing",
        )

    def test_unregistered_advisor_not_consulted(self):
        """After unregistering, the advisor is no longer consulted."""
        fabric = _make_fabric()
        advisor = _mock_advisor(name="temp", chain_result=["m2", "m1"])
        fabric.register_routing_advisor(advisor)
        fabric.unregister_routing_advisor("temp")

        result = fabric.consult_routing_advisors("coding", ["m1", "m2"])
        assert result == ["m1", "m2"]
        advisor.call_tool.assert_not_called()
