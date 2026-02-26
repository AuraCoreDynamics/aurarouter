"""External MCP Client Registry for AuraRouter.

Provides GridMcpClient for connecting to external MCP-compatible servers
and McpClientRegistry for managing multiple client connections.
"""

from aurarouter.mcp_client.client import GridMcpClient
from aurarouter.mcp_client.registry import McpClientRegistry

__all__ = ["GridMcpClient", "McpClientRegistry"]
