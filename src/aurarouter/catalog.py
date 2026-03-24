"""Provider Catalog -- discovers and manages provider lifecycle.

Aggregates providers from three sources:

1. **Built-in** -- providers shipped with AuraRouter (ollama, llamacpp-server,
   llamacpp, openapi).
2. **Entry points** -- external Python packages that register under the
   ``aurarouter.providers`` entry-point group.
3. **Manual** -- user-configured endpoints in ``provider_catalog.manual``.

The catalog can start/stop provider MCP servers, check health, and
auto-register discovered models into the routing config.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from aurarouter._logging import get_logger
from aurarouter.mcp_client.client import GridMcpClient
from aurarouter.providers.protocol import (
    TOOL_HEALTH_CHECK,
    TOOL_LIST_MODELS,
    ProviderMetadata,
    validate_provider_tools,
)

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader

logger = get_logger("AuraRouter.Catalog")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    """A single provider known to the catalog.

    Attributes:
        name: Unique identifier for this provider.
        provider_type: Provider type key (e.g. ``"ollama"``, ``"mcp"``).
        source: How the provider was discovered: ``"builtin"``,
                ``"entrypoint"``, or ``"manual"``.
        metadata: Optional :class:`ProviderMetadata` from entry-point packages.
        installed: Whether the provider package is installed / available.
        running: Whether the provider MCP server is currently reachable.
        version: Provider version string (if known).
        description: Human-readable one-liner.
    """

    name: str
    provider_type: str
    source: str  # "builtin" | "entrypoint" | "manual"
    metadata: ProviderMetadata | None = None
    installed: bool = True
    running: bool = False
    version: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class ProviderCatalog:
    """Central registry for discovering and managing providers."""

    def __init__(self, config: ConfigLoader) -> None:
        self._config = config
        self._entries: dict[str, CatalogEntry] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._clients: dict[str, GridMcpClient] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[CatalogEntry]:
        """Aggregate providers from all sources and return the full list."""
        self._entries.clear()
        for entry in self.get_builtin_providers():
            self._entries[entry.name] = entry
        for entry in self.get_entrypoint_providers():
            self._entries[entry.name] = entry
        for entry in self.get_manual_providers():
            self._entries[entry.name] = entry
        return list(self._entries.values())

    def get_builtin_providers(self) -> list[CatalogEntry]:
        """Return catalog entries for built-in providers."""
        builtins = [
            CatalogEntry(
                name="ollama",
                provider_type="ollama",
                source="builtin",
                installed=True,
                running=False,
                version="",
                description="Ollama local model server",
            ),
            CatalogEntry(
                name="llamacpp-server",
                provider_type="llamacpp-server",
                source="builtin",
                installed=True,
                running=False,
                version="",
                description="llama.cpp HTTP server (external process)",
            ),
            CatalogEntry(
                name="llamacpp",
                provider_type="llamacpp",
                source="builtin",
                installed=self._is_llamacpp_available(),
                running=False,
                version="",
                description="llama.cpp in-process via llama-cpp-python",
            ),
            CatalogEntry(
                name="openapi",
                provider_type="openapi",
                source="builtin",
                installed=True,
                running=False,
                version="",
                description="OpenAI-API-compatible endpoints (vLLM, LM Studio, etc.)",
            ),
        ]
        return builtins

    def get_entrypoint_providers(self) -> list[CatalogEntry]:
        """Discover external providers via Python entry points.

        Looks for packages that register under the
        ``aurarouter.providers`` entry-point group.  Each entry point
        should be a callable returning a :class:`ProviderMetadata`.
        """
        entries: list[CatalogEntry] = []
        try:
            eps = entry_points(group="aurarouter.providers")
        except TypeError:
            # Python < 3.12 fallback
            eps = entry_points().get("aurarouter.providers", [])  # type: ignore[union-attr]

        for ep in eps:
            try:
                metadata_fn = ep.load()
                meta: ProviderMetadata = metadata_fn()
                entries.append(
                    CatalogEntry(
                        name=meta.name,
                        provider_type=meta.provider_type,
                        source="entrypoint",
                        metadata=meta,
                        installed=True,
                        running=False,
                        version=meta.version,
                        description=meta.description,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load entry point '%s': %s", ep.name, exc
                )
        return entries

    def get_manual_providers(self) -> list[CatalogEntry]:
        """Read manually configured providers from the config file."""
        manual_entries = self._config.get_catalog_manual_entries()
        entries: list[CatalogEntry] = []
        for item in manual_entries:
            name = item.get("name", "")
            if not name:
                continue
            entries.append(
                CatalogEntry(
                    name=name,
                    provider_type="mcp",
                    source="manual",
                    installed=True,
                    running=False,
                    description=item.get("description", f"Manual MCP provider: {name}"),
                )
            )
        return entries

    # ------------------------------------------------------------------
    # Manual registration
    # ------------------------------------------------------------------

    def register_manual(self, name: str, endpoint: str) -> CatalogEntry:
        """Register a manual provider endpoint.

        Persists the entry in config and returns the new catalog entry.
        """
        self._config.add_catalog_manual_entry(name, endpoint)
        entry = CatalogEntry(
            name=name,
            provider_type="mcp",
            source="manual",
            installed=True,
            running=False,
            description=f"Manual MCP provider: {name}",
        )
        self._entries[name] = entry
        return entry

    def unregister_manual(self, name: str) -> bool:
        """Remove a manual provider entry from config and catalog."""
        removed = self._config.remove_catalog_manual_entry(name)
        self._entries.pop(name, None)
        return removed

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def start_provider(self, name: str) -> bool:
        """Launch a provider MCP server subprocess.

        Uses ``ProviderMetadata.command`` from the catalog entry.
        Returns ``True`` if the process was started successfully.
        """
        entry = self._entries.get(name)
        if not entry or not entry.metadata or not entry.metadata.command:
            logger.warning(
                "Cannot start provider '%s': no command configured", name
            )
            return False

        if name in self._processes and self._processes[name].poll() is None:
            logger.info("Provider '%s' is already running", name)
            return True

        try:
            proc = subprocess.Popen(
                entry.metadata.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._processes[name] = proc
            entry.running = True
            logger.info(
                "Started provider '%s' (PID %d)", name, proc.pid
            )
            return True
        except Exception as exc:
            logger.error("Failed to start provider '%s': %s", name, exc)
            return False

    def stop_provider(self, name: str) -> bool:
        """Stop a running provider subprocess.

        Returns ``True`` if the process was terminated.
        """
        proc = self._processes.pop(name, None)
        if proc is None or proc.poll() is not None:
            return False

        try:
            proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        except Exception as exc:
            logger.error("Error stopping provider '%s': %s", name, exc)
            return False

        entry = self._entries.get(name)
        if entry:
            entry.running = False
        logger.info("Stopped provider '%s'", name)
        return True

    # ------------------------------------------------------------------
    # Health & model discovery
    # ------------------------------------------------------------------

    def check_provider_health(self, name: str) -> tuple[bool, str]:
        """Probe a provider's health via its MCP endpoint.

        Returns ``(healthy, message)`` tuple.
        """
        entry = self._entries.get(name)
        if not entry:
            return (False, f"Provider '{name}' not found in catalog")

        # Determine endpoint
        endpoint = self._get_endpoint(name)
        if not endpoint:
            return (False, f"No endpoint configured for provider '{name}'")

        try:
            client = self._get_or_create_client(name, endpoint)
            if not client.connected and not client.connect():
                return (False, f"Cannot connect to {endpoint}")

            if TOOL_HEALTH_CHECK in client.get_capabilities():
                result = client.call_tool(TOOL_HEALTH_CHECK)
                if isinstance(result, dict):
                    healthy = result.get("healthy", False)
                    msg = result.get("message", "")
                    return (bool(healthy), msg)
                return (True, "healthy")
            else:
                # If no health_check tool, connection success is enough
                return (True, "connected (no health_check tool)")
        except Exception as exc:
            return (False, str(exc))

    def auto_register_models(self, name: str, config: ConfigLoader) -> int:
        """Discover models from a provider and register them in config.

        Calls ``provider.list_models`` on the provider's MCP endpoint,
        then creates model entries with ``provider='mcp'`` in the config.

        Returns the number of models added.
        """
        endpoint = self._get_endpoint(name)
        if not endpoint:
            logger.warning(
                "Cannot auto-register models for '%s': no endpoint", name
            )
            return 0

        try:
            client = self._get_or_create_client(name, endpoint)
            if not client.connected and not client.connect():
                logger.warning(
                    "Cannot connect to '%s' at %s for model discovery",
                    name, endpoint,
                )
                return 0

            if TOOL_LIST_MODELS not in client.get_capabilities():
                logger.info(
                    "Provider '%s' does not expose %s",
                    name, TOOL_LIST_MODELS,
                )
                return 0

            models = client.call_tool(TOOL_LIST_MODELS)
            if not isinstance(models, list):
                return 0

            added = 0
            for model_info in models:
                model_id = model_info.get("id") or model_info.get("name", "")
                if not model_id:
                    continue

                remote_id = f"{name}/{model_id}"
                if config.get_model_config(remote_id):
                    continue

                model_cfg = {
                    "provider": "mcp",
                    "mcp_endpoint": endpoint,
                    "model_name": model_id,
                    "tags": ["remote", f"provider:{name}"],
                }
                config.set_model(remote_id, model_cfg)
                logger.info("Auto-registered model: %s", remote_id)
                added += 1

            return added

        except Exception as exc:
            logger.warning(
                "Model discovery failed for '%s': %s", name, exc
            )
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_llamacpp_available(self) -> bool:
        """Check if llama-cpp-python is importable."""
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_endpoint(self, name: str) -> str:
        """Resolve the MCP endpoint for a named provider."""
        # Check manual config entries
        for item in self._config.get_catalog_manual_entries():
            if item.get("name") == name:
                return item.get("endpoint", "")

        # Check entry metadata
        entry = self._entries.get(name)
        if entry and entry.metadata and entry.metadata.command:
            # For entry-point providers, the endpoint is typically
            # determined at startup time.  Return empty to signal
            # that the provider needs to be started first.
            pass

        return ""

    def _get_or_create_client(
        self, name: str, endpoint: str
    ) -> GridMcpClient:
        """Return a cached MCP client or create a new one."""
        if name not in self._clients:
            self._clients[name] = GridMcpClient(
                base_url=endpoint,
                name=f"catalog:{name}",
                timeout=30.0,
            )
        return self._clients[name]
