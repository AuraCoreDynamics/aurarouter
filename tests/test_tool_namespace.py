"""Tests for MCP tool name namespacing (TG-A2)."""

from aurarouter.server import _MCP_TOOL_DEFAULTS


class TestAssetToolNamespace:
    def test_asset_tools_use_full_namespace(self):
        """Asset tools are keyed with aurarouter.assets.* namespace."""
        assert "aurarouter.assets.list" in _MCP_TOOL_DEFAULTS
        assert "aurarouter.assets.register" in _MCP_TOOL_DEFAULTS
        assert "aurarouter.assets.unregister" in _MCP_TOOL_DEFAULTS

    def test_old_bare_names_not_present(self):
        """Old short-form asset keys are gone."""
        assert "assets.list" not in _MCP_TOOL_DEFAULTS
        assert "assets.register" not in _MCP_TOOL_DEFAULTS
        assert "assets.unregister" not in _MCP_TOOL_DEFAULTS

    def test_routing_tools_keep_bare_names(self):
        """Routing tools retain their original bare names."""
        assert "route_task" in _MCP_TOOL_DEFAULTS
        assert "local_inference" in _MCP_TOOL_DEFAULTS
        assert "generate_code" in _MCP_TOOL_DEFAULTS
        assert "compare_models" in _MCP_TOOL_DEFAULTS
        assert "list_models" in _MCP_TOOL_DEFAULTS

    def test_asset_tools_enabled_by_default(self):
        """All three asset tools default to enabled."""
        assert _MCP_TOOL_DEFAULTS["aurarouter.assets.list"] is True
        assert _MCP_TOOL_DEFAULTS["aurarouter.assets.register"] is True
        assert _MCP_TOOL_DEFAULTS["aurarouter.assets.unregister"] is True
