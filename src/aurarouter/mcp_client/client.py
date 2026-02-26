"""Grid MCP Client -- connects to external MCP-compatible servers."""

from __future__ import annotations

from typing import Any

import httpx

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.McpClient")


class GridMcpClient:
    """Client for an external MCP-compatible server.

    Connects to a remote server, discovers its tools and models,
    and exposes them for use by AuraRouter.

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

    def connect(self) -> bool:
        """Discover tools and models from the remote server.

        Returns ``True`` if connection succeeded, ``False`` otherwise.
        Connection failures are logged but never raised.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                # Discover tools
                try:
                    resp = client.get(f"{self._base_url}/api/v1/tools")
                    resp.raise_for_status()
                    data = resp.json()
                    self._tools = data if isinstance(data, list) else data.get("tools", [])
                except Exception as exc:
                    logger.debug(f"[{self._name}] No tools endpoint: {exc}")
                    self._tools = []

                # Discover models
                try:
                    resp = client.get(f"{self._base_url}/api/v1/models")
                    resp.raise_for_status()
                    data = resp.json()
                    self._models = data if isinstance(data, list) else data.get("models", [])
                except Exception as exc:
                    logger.debug(f"[{self._name}] No models endpoint: {exc}")
                    self._models = []

                # Discover capabilities
                try:
                    resp = client.get(f"{self._base_url}/api/v1/capabilities")
                    resp.raise_for_status()
                    data = resp.json()
                    raw = data if isinstance(data, list) else data.get("capabilities", [])
                    self._capabilities = set(raw)
                except Exception:
                    # Infer capabilities from what we discovered
                    self._capabilities = set()
                    if self._tools:
                        self._capabilities.add("tools")
                    if self._models:
                        self._capabilities.add("models")

            self._connected = True
            logger.info(
                f"[{self._name}] Connected: "
                f"{len(self._tools)} tools, "
                f"{len(self._models)} models, "
                f"capabilities={self._capabilities}"
            )
            return True

        except Exception as exc:
            self._connected = False
            logger.warning(f"[{self._name}] Connection failed: {exc}")
            return False

    def get_tools(self) -> list[dict]:
        """Return discovered tools."""
        return self._tools

    def get_models(self) -> list[dict]:
        """Return discovered models."""
        return self._models

    def get_capabilities(self) -> set[str]:
        """Return advertised capabilities (e.g. ``'chain_reorder'``, ``'rag_query'``)."""
        return self._capabilities

    def call_tool(self, tool_name: str, **kwargs: Any) -> dict:
        """Call a remote tool by name.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Arguments to pass to the tool.

        Returns:
            The tool response as a dict.

        Raises:
            ConnectionError: If not connected.
            httpx.HTTPStatusError: On HTTP errors.
        """
        if not self._connected:
            raise ConnectionError(
                f"[{self._name}] Not connected. Call connect() first."
            )

        url = f"{self._base_url}/api/v1/tools/{tool_name}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=kwargs)
            resp.raise_for_status()
            return resp.json()
