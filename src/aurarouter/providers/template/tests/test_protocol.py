"""Protocol compliance tests for the example provider.

These tests validate that the provider's MCP server exposes the
correct tools as required by the AuraRouter MCP Provider Protocol.
Adapt these tests for your own provider package.
"""

from __future__ import annotations

import pytest

from aurarouter.providers.protocol import (
    REQUIRED_TOOLS,
    TOOL_CAPABILITIES,
    TOOL_GENERATE,
    TOOL_HEALTH_CHECK,
    TOOL_LIST_MODELS,
    validate_provider_tools,
)


def _tool(name: str) -> dict:
    """Helper to create a minimal tool dict."""
    return {"name": name, "description": f"Tool: {name}"}


class TestProtocolCompliance:
    """Verify the example provider's tool set passes validation."""

    def test_required_tools_present(self):
        """All required tools must be present."""
        tools = [_tool(TOOL_GENERATE), _tool(TOOL_LIST_MODELS)]
        valid, errors = validate_provider_tools(tools)
        assert valid is True
        assert errors == []

    def test_full_tool_set(self):
        """Full tool set with optional tools passes validation."""
        tools = [
            _tool(TOOL_GENERATE),
            _tool(TOOL_LIST_MODELS),
            _tool(TOOL_HEALTH_CHECK),
            _tool(TOOL_CAPABILITIES),
        ]
        valid, errors = validate_provider_tools(tools)
        assert valid is True
        assert errors == []

    def test_missing_generate_fails(self):
        """Missing provider.generate must fail validation."""
        tools = [_tool(TOOL_LIST_MODELS)]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert any("provider.generate" in e for e in errors)

    def test_missing_list_models_fails(self):
        """Missing provider.list_models must fail validation."""
        tools = [_tool(TOOL_GENERATE)]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert any("provider.list_models" in e for e in errors)

    def test_unknown_provider_tool_fails(self):
        """Unrecognised provider.* tools must fail validation."""
        tools = [
            _tool(TOOL_GENERATE),
            _tool(TOOL_LIST_MODELS),
            _tool("provider.unknown_thing"),
        ]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert any("provider.unknown_thing" in e for e in errors)

    def test_non_provider_tools_are_ignored(self):
        """Tools outside the provider.* namespace are fine."""
        tools = [
            _tool(TOOL_GENERATE),
            _tool(TOOL_LIST_MODELS),
            _tool("custom.my_tool"),
            _tool("other_namespace.something"),
        ]
        valid, errors = validate_provider_tools(tools)
        assert valid is True
        assert errors == []
