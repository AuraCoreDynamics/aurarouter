"""Tests for routing advisor hooks in ComputeFabric."""

from unittest.mock import MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_client.registry import McpClientRegistry


def test_no_advisors_returns_chain_unchanged():
    """When no registry is set, chain is unmodified."""
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {"m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"}},
        "roles": {"coding": ["m1"]},
    }
    fabric = ComputeFabric(cfg)
    result = fabric._consult_routing_advisors("coding", ["m1"])
    assert result == ["m1"]


def test_advisor_reorders_chain():
    """A connected advisor with chain_reorder capability can reorder."""
    mock_client = MagicMock()
    mock_client.connected = True
    mock_client.get_capabilities.return_value = {"chain_reorder"}
    mock_client.name = "advisor1"
    mock_client.call_tool.return_value = {"chain": ["m2", "m1"]}

    registry = McpClientRegistry()
    registry.register("advisor1", mock_client)

    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "a", "endpoint": "http://x"},
            "m2": {"provider": "ollama", "model_name": "b", "endpoint": "http://y"},
        },
        "roles": {"coding": ["m1", "m2"]},
    }
    fabric = ComputeFabric(cfg, routing_advisors=registry)
    result = fabric._consult_routing_advisors("coding", ["m1", "m2"])
    assert result == ["m2", "m1"]


def test_advisor_failure_returns_chain_unchanged():
    """If advisor call fails, the original chain is returned."""
    mock_client = MagicMock()
    mock_client.connected = True
    mock_client.get_capabilities.return_value = {"chain_reorder"}
    mock_client.name = "bad_advisor"
    mock_client.call_tool.side_effect = Exception("timeout")

    registry = McpClientRegistry()
    registry.register("bad_advisor", mock_client)

    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": {}, "roles": {}}
    fabric = ComputeFabric(cfg, routing_advisors=registry)
    result = fabric._consult_routing_advisors("coding", ["m1"])
    assert result == ["m1"]


def test_advisor_empty_response_returns_chain_unchanged():
    """If advisor returns empty list, original chain is returned."""
    mock_client = MagicMock()
    mock_client.connected = True
    mock_client.get_capabilities.return_value = {"chain_reorder"}
    mock_client.name = "empty_advisor"
    mock_client.call_tool.return_value = {"chain": []}

    registry = McpClientRegistry()
    registry.register("empty_advisor", mock_client)

    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": {}, "roles": {}}
    fabric = ComputeFabric(cfg, routing_advisors=registry)
    result = fabric._consult_routing_advisors("coding", ["m1"])
    assert result == ["m1"]


def test_no_clients_with_capability():
    """When no clients have chain_reorder capability, chain unchanged."""
    mock_client = MagicMock()
    mock_client.connected = True
    mock_client.get_capabilities.return_value = {"rag_query"}  # different capability

    registry = McpClientRegistry()
    registry.register("rag_only", mock_client)

    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {"models": {}, "roles": {}}
    fabric = ComputeFabric(cfg, routing_advisors=registry)
    result = fabric._consult_routing_advisors("coding", ["m1", "m2"])
    assert result == ["m1", "m2"]
