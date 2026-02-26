"""Grid MCP Client -- connects to external MCP-compatible servers.

Uses MCP JSON-RPC 2.0 over ``POST /mcp/message`` for all communication.
Tool discovery via ``tools/list``, tool invocation via ``tools/call``.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.McpClient")


class GridMcpClient:
    """Client for an external MCP-compatible server.

    Connects to a remote MCP server using JSON-RPC 2.0 over a single
    ``POST /mcp/message`` endpoint. Discovers tools via ``tools/list``
    and invokes them via ``tools/call``.

    Args:
        base_url: Root URL of the MCP server (e.g. ``"http://localhost:8080"``).
        name: Optional human-readable name for this client.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        name: str = "",
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._name = name or self._base_url
        self._timeout = timeout
        self._tools: list[dict] = []
        self._models: list[dict] = []
        self._capabilities: set[str] = set()
        self._connected = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def connected(self) -> bool:
        return self._connected

    def _rpc_url(self) -> str:
        return f"{self._base_url}/mcp/message"

    def _jsonrpc_request(self, method: str, params: dict | None = None) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params or {},
        }

    def connect(self) -> bool:
        """Discover tools and models from the remote MCP server.

        Uses MCP JSON-RPC 2.0 ``tools/list`` via ``POST /mcp/message``.
        Returns ``True`` if connection succeeded, ``False`` otherwise.
        Connection failures are logged but never raised.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                # Tool discovery via JSON-RPC 2.0
                rpc_request = self._jsonrpc_request("tools/list")
                resp = client.post(self._rpc_url(), json=rpc_request)
                resp.raise_for_status()
                rpc_response = resp.json()

                if "error" in rpc_response:
                    logger.error(
                        "[%s] MCP tools/list error: %s",
                        self._name,
                        rpc_response["error"],
                    )
                    return False

                result = rpc_response.get("result", {})
                self._tools = result.get("tools", [])
                self._connected = True

                # Derive capabilities from tool names
                self._capabilities = {
                    tool.get("name", "") for tool in self._tools
                }

                # Model discovery via auraxlm.models tool call (if available)
                if any(t.get("name") == "auraxlm.models" for t in self._tools):
                    try:
                        models_result = self.call_tool("auraxlm.models")
                        self._models = (
                            models_result if isinstance(models_result, list) else []
                        )
                    except Exception:
                        self._models = []
                else:
                    self._models = []

                logger.info(
                    "[%s] Connected: %d tools, %d models",
                    self._name,
                    len(self._tools),
                    len(self._models),
                )
                return True

        except Exception as exc:
            self._connected = False
            logger.warning("[%s] Connection failed: %s", self._name, exc)
            return False

    def get_tools(self) -> list[dict]:
        """Return discovered tools."""
        return self._tools

    def get_models(self) -> list[dict]:
        """Return discovered models."""
        return self._models

    def get_capabilities(self) -> set[str]:
        """Return advertised capabilities (derived from tool names)."""
        return self._capabilities

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a remote MCP tool via JSON-RPC 2.0.

        Sends ``tools/call`` to ``POST /mcp/message`` with the tool name
        and arguments.  Returns the tool result payload.

        Raises:
            ConnectionError: If not connected.
            httpx.HTTPStatusError: On HTTP-level failures.
            RuntimeError: On JSON-RPC error responses.
        """
        if not self._connected:
            raise ConnectionError(
                f"[{self._name}] Not connected. Call connect() first."
            )

        rpc_request = self._jsonrpc_request(
            "tools/call",
            {"name": tool_name, "arguments": kwargs},
        )
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(self._rpc_url(), json=rpc_request)
            resp.raise_for_status()
            rpc_response = resp.json()

        if "error" in rpc_response:
            err = rpc_response["error"]
            raise RuntimeError(
                f"MCP tool call '{tool_name}' failed: "
                f"[{err.get('code')}] {err.get('message')}"
            )

        return rpc_response.get("result")
