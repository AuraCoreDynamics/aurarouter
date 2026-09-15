"""Tests for aurarouter.providers.protocol -- MCP Provider Protocol validation."""

from __future__ import annotations

import pytest

from aurarouter.providers.protocol import (
    ALL_TOOLS,
    OPTIONAL_TOOLS,
    REQUIRED_TOOLS,
    TOOL_CAPABILITIES,
    TOOL_GENERATE,
    TOOL_GENERATE_WITH_HISTORY,
    TOOL_HEALTH_CHECK,
    TOOL_LIST_MODELS,
    ProviderMetadata,
    validate_provider_tools,
)


def _tool(name: str) -> dict:
    return {"name": name, "description": f"Tool: {name}"}


class TestConstants:
    def test_required_tools(self):
        assert REQUIRED_TOOLS == {TOOL_GENERATE, TOOL_LIST_MODELS}

    def test_optional_tools(self):
        assert OPTIONAL_TOOLS == {
            TOOL_GENERATE_WITH_HISTORY,
            TOOL_HEALTH_CHECK,
            TOOL_CAPABILITIES,
        }

    def test_all_tools_is_union(self):
        assert ALL_TOOLS == REQUIRED_TOOLS | OPTIONAL_TOOLS

    def test_tool_name_format(self):
        for tool in ALL_TOOLS:
            assert tool.startswith("provider.")


class TestProviderMetadata:
    def test_basic_fields(self):
        meta = ProviderMetadata(
            name="test",
            provider_type="mcp",
            version="1.0.0",
            description="A test provider",
        )
        assert meta.name == "test"
        assert meta.provider_type == "mcp"
        assert meta.version == "1.0.0"
        assert meta.command == []
        assert meta.requires_config == []
        assert meta.homepage == ""

    def test_full_fields(self):
        meta = ProviderMetadata(
            name="gemini",
            provider_type="mcp",
            version="0.2.0",
            description="Google Gemini provider",
            command=["python", "-m", "aurarouter_gemini.server"],
            requires_config=["api_key"],
            homepage="https://example.com",
        )
        assert meta.command == ["python", "-m", "aurarouter_gemini.server"]
        assert meta.requires_config == ["api_key"]
        assert meta.homepage == "https://example.com"


class TestValidateProviderTools:
    def test_valid_minimal(self):
        """Only required tools => valid."""
        tools = [_tool(TOOL_GENERATE), _tool(TOOL_LIST_MODELS)]
        valid, errors = validate_provider_tools(tools)
        assert valid is True
        assert errors == []

    def test_valid_full_set(self):
        """All protocol tools => valid."""
        tools = [_tool(t) for t in ALL_TOOLS]
        valid, errors = validate_provider_tools(tools)
        assert valid is True
        assert errors == []

    def test_missing_generate(self):
        tools = [_tool(TOOL_LIST_MODELS)]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert any(TOOL_GENERATE in e for e in errors)

    def test_missing_list_models(self):
        tools = [_tool(TOOL_GENERATE)]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert any(TOOL_LIST_MODELS in e for e in errors)

    def test_missing_both_required(self):
        tools = [_tool(TOOL_HEALTH_CHECK)]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert len(errors) == 1  # single "Missing required tools" error

    def test_empty_tool_list(self):
        valid, errors = validate_provider_tools([])
        assert valid is False
        assert len(errors) >= 1

    def test_unrecognised_provider_tool(self):
        tools = [
            _tool(TOOL_GENERATE),
            _tool(TOOL_LIST_MODELS),
            _tool("provider.bogus"),
        ]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert any("provider.bogus" in e for e in errors)

    def test_non_provider_namespace_ignored(self):
        """Tools outside provider.* don't cause validation errors."""
        tools = [
            _tool(TOOL_GENERATE),
            _tool(TOOL_LIST_MODELS),
            _tool("custom.my_tool"),
            _tool("something_else"),
        ]
        valid, errors = validate_provider_tools(tools)
        assert valid is True

    def test_tools_with_missing_name_key(self):
        """Tool dicts without a 'name' key are silently skipped."""
        tools = [
            _tool(TOOL_GENERATE),
            _tool(TOOL_LIST_MODELS),
            {"description": "no name field"},
        ]
        valid, errors = validate_provider_tools(tools)
        assert valid is True

    def test_multiple_errors_reported(self):
        """Missing required + unrecognised => multiple errors."""
        tools = [_tool(TOOL_LIST_MODELS), _tool("provider.fake")]
        valid, errors = validate_provider_tools(tools)
        assert valid is False
        assert len(errors) == 2
