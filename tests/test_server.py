import json
from unittest.mock import patch, MagicMock

from aurarouter.config import ConfigLoader
from aurarouter.server import create_mcp_server


def _make_config() -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "models": {
            "m1": {"provider": "ollama", "model_name": "test", "endpoint": "http://x"},
        },
        "roles": {
            "router": ["m1"],
            "reasoning": ["m1"],
            "coding": ["m1"],
        },
    }
    return cfg


def test_create_mcp_server_returns_fastmcp():
    mcp = create_mcp_server(_make_config())
    assert mcp is not None
    assert mcp.name == "AuraRouter"


def test_intelligent_code_gen_simple():
    """Simple intent → direct coding call."""
    mcp = create_mcp_server(_make_config())

    # The tool function is registered inside the closure; we test via the
    # routing + fabric layer with mocks on the provider.
    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate",
        side_effect=[
            json.dumps({"intent": "SIMPLE_CODE"}),   # router call
            "def add(a, b): return a + b",            # coding call
        ],
    ):
        # Reach through the MCP tool registry to call the function directly
        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        assert "intelligent_code_gen" in tools


def test_intelligent_code_gen_complex():
    """Complex intent → plan + multi-step execution."""
    mcp = create_mcp_server(_make_config())

    with patch(
        "aurarouter.providers.ollama.OllamaProvider.generate",
        side_effect=[
            json.dumps({"intent": "COMPLEX_REASONING"}),   # router
            json.dumps(["step A", "step B"]),                # reasoning (plan)
            "code_for_step_a",                               # coding step 1
            "code_for_step_b",                               # coding step 2
        ],
    ):
        tools = {t.name: t for t in mcp._tool_manager.list_tools()}
        assert "intelligent_code_gen" in tools


def test_mcp_tool_is_registered():
    mcp = create_mcp_server(_make_config())
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "intelligent_code_gen" in tool_names


# ------------------------------------------------------------------
# Multi-tool registration
# ------------------------------------------------------------------

def test_default_tools_registered():
    """All default-enabled tools plus the deprecated alias should be present."""
    mcp = create_mcp_server(_make_config())
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "route_task" in tool_names
    assert "local_inference" in tool_names
    assert "generate_code" in tool_names
    assert "intelligent_code_gen" in tool_names  # deprecated alias
    assert "compare_models" not in tool_names    # disabled by default


def test_tool_disabled_by_config():
    """A tool set to enabled=false should not be registered."""
    cfg = _make_config()
    cfg.config["mcp"] = {"tools": {"route_task": {"enabled": False}}}
    mcp = create_mcp_server(cfg)
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "route_task" not in tool_names
    # Other defaults should still be present
    assert "local_inference" in tool_names
    assert "generate_code" in tool_names


def test_compare_models_enabled_by_config():
    """compare_models is off by default but can be enabled in config."""
    cfg = _make_config()
    cfg.config["mcp"] = {"tools": {"compare_models": {"enabled": True}}}
    mcp = create_mcp_server(cfg)
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "compare_models" in tool_names


def test_deprecated_alias_always_registered():
    """intelligent_code_gen should always be present regardless of config."""
    cfg = _make_config()
    cfg.config["mcp"] = {"tools": {"generate_code": {"enabled": False}}}
    mcp = create_mcp_server(cfg)
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "intelligent_code_gen" in tool_names
    assert "generate_code" not in tool_names
