"""MCP Provider Protocol -- constants and validation for provider MCP servers.

Defines the tool names, metadata structure, and validation logic that
external provider packages must implement to integrate with AuraRouter
as MCP servers.

Protocol tools:
    Required:
        - ``provider.generate`` -- Single-shot text generation
        - ``provider.list_models`` -- Enumerate available models

    Optional:
        - ``provider.generate_with_history`` -- Multi-turn generation
        - ``provider.health_check`` -- Liveness/readiness probe
        - ``provider.capabilities`` -- Advertise provider features
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Tool name constants
# ---------------------------------------------------------------------------

TOOL_GENERATE: str = "provider.generate"
TOOL_LIST_MODELS: str = "provider.list_models"
TOOL_GENERATE_WITH_HISTORY: str = "provider.generate_with_history"
TOOL_HEALTH_CHECK: str = "provider.health_check"
TOOL_CAPABILITIES: str = "provider.capabilities"

REQUIRED_TOOLS: set[str] = {TOOL_GENERATE, TOOL_LIST_MODELS}
OPTIONAL_TOOLS: set[str] = {
    TOOL_GENERATE_WITH_HISTORY,
    TOOL_HEALTH_CHECK,
    TOOL_CAPABILITIES,
}
ALL_TOOLS: set[str] = REQUIRED_TOOLS | OPTIONAL_TOOLS


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

@dataclass
class ProviderMetadata:
    """Metadata returned by an external provider's entry point.

    Attributes:
        name: Human-readable provider name (e.g. ``"gemini"``).
        provider_type: Short identifier used in config (e.g. ``"mcp"``).
        version: Semantic version of the provider package.
        description: One-line description of what the provider offers.
        command: Command-line tokens to launch the provider MCP server
                 (e.g. ``["python", "-m", "aurarouter_gemini.server"]``).
        requires_config: Config keys the provider expects (e.g. ``["api_key"]``).
        homepage: URL to provider documentation or repository.
    """

    name: str
    provider_type: str
    version: str
    description: str
    command: list[str] = field(default_factory=list)
    requires_config: list[str] = field(default_factory=list)
    homepage: str = ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_provider_tools(tools: list[dict]) -> tuple[bool, list[str]]:
    """Check whether a tool list satisfies the MCP Provider Protocol.

    Args:
        tools: List of tool dicts as returned by MCP ``tools/list``.
               Each dict must have at least a ``"name"`` key.

    Returns:
        A ``(valid, errors)`` tuple.  *valid* is ``True`` when all
        required tools are present and no unrecognised tool names
        appear under the ``provider.*`` namespace.  *errors* is a list
        of human-readable problem descriptions (empty when valid).
    """
    errors: list[str] = []
    tool_names: set[str] = set()

    for tool in tools:
        name = tool.get("name", "")
        if name:
            tool_names.add(name)

    # Check required tools are present
    missing = REQUIRED_TOOLS - tool_names
    if missing:
        errors.append(
            f"Missing required tools: {', '.join(sorted(missing))}"
        )

    # Check for unrecognised provider.* tools
    for name in tool_names:
        if name.startswith("provider.") and name not in ALL_TOOLS:
            errors.append(f"Unrecognised provider tool: {name}")

    return (len(errors) == 0, errors)
