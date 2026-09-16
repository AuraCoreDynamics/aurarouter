import pytest
from aurarouter.config import ConfigLoader
from aurarouter.mcp_tools import get_allowed_tools_for_context

def test_get_allowed_tools_for_context():
    config = ConfigLoader(allow_missing=True)
    
    # Test 1: ["*"] allows all tools (wildcard)
    config.set_role_chain("test_role", ["test_model"])
    config.config["roles"]["test_role"] = {"chain": ["test_model"], "allowed_mcp_tools": ["*"]}
    config.set_model("test_model", {"provider": "test", "allowed_mcp_tools": ["read_file", "write_file"]})
    
    tools, _ = get_allowed_tools_for_context("test_role", "test_model", config)
    assert tools == ["read_file", "write_file"]
    
    # Test 2: [] allows none
    config.config["roles"]["test_role"] = {"chain": ["test_model"], "allowed_mcp_tools": []}
    tools, _ = get_allowed_tools_for_context("test_role", "test_model", config)
    assert tools == []
    
    # Test 3: filter properly
    config.config["roles"]["test_role"] = {"chain": ["test_model"], "allowed_mcp_tools": ["read_file", "execute_code"]}
    config.set_model("test_model", {"provider": "test", "allowed_mcp_tools": ["read_file", "delete_file"]})
    tools, _ = get_allowed_tools_for_context("test_role", "test_model", config)
    assert tools == ["read_file"]
    
    # Test 4: no restrictions
    config.config["roles"]["test_role"] = ["test_model"]
    config.set_model("test_model", {"provider": "test"})
    tools, _ = get_allowed_tools_for_context("test_role", "test_model", config)
    assert tools == []
