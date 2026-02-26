"""MCP Client Registry -- manages multiple GridMcpClient connections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader
    from aurarouter.mcp_client.client import GridMcpClient

logger = get_logger("AuraRouter.McpRegistry")


class McpClientRegistry:
    """Registry of connected external MCP clients.

    Provides centralised management of :class:`GridMcpClient` instances
    and aggregated access to their tools and models.
    """

    def __init__(self) -> None:
        self._clients: dict[str, GridMcpClient] = {}

    def register(self, name: str, client: GridMcpClient) -> None:
        """Register a client under the given name."""
        self._clients[name] = client
        logger.info(f"Registered MCP client: {name}")

    def unregister(self, name: str) -> bool:
        """Remove a client. Returns ``True`` if it existed."""
        if name in self._clients:
            del self._clients[name]
            logger.info(f"Unregistered MCP client: {name}")
            return True
        return False

    def get_clients(self) -> dict[str, GridMcpClient]:
        """Return all registered clients."""
        return dict(self._clients)

    def get_clients_with_capability(self, cap: str) -> list[GridMcpClient]:
        """Return clients that advertise the given capability."""
        return [
            c for c in self._clients.values()
            if c.connected and cap in c.get_capabilities()
        ]

    def get_all_remote_tools(self) -> list[dict]:
        """Aggregate tools from all connected clients.

        Each tool dict is enriched with a ``_source_client`` key
        identifying which client it came from.
        """
        tools: list[dict] = []
        for name, client in self._clients.items():
            if not client.connected:
                continue
            for tool in client.get_tools():
                enriched = dict(tool)
                enriched.setdefault("_source_client", name)
                tools.append(enriched)
        return tools

    def sync_models(self, config: ConfigLoader) -> int:
        """Register discovered remote models into the config.

        Creates model entries with ``provider='openapi'`` pointing to
        the remote server's endpoint.  Only adds models not already
        present in config.

        Returns the number of models added.
        """
        added = 0
        for name, client in self._clients.items():
            if not client.connected:
                continue
            for model_info in client.get_models():
                model_id = model_info.get("id") or model_info.get("name", "")
                if not model_id:
                    continue

                # Prefix with client name to avoid collisions
                remote_id = f"{name}/{model_id}"

                # Skip if already configured
                if config.get_model_config(remote_id):
                    continue

                model_cfg = {
                    "provider": model_info.get("provider", "openapi"),
                    "endpoint": client.base_url,
                    "model_name": model_id,
                    "tags": ["remote", f"grid:{name}"],
                }
                config.set_model(remote_id, model_cfg)
                logger.info(f"Auto-registered remote model: {remote_id}")
                added += 1

        return added
