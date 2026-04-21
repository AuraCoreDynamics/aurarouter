"""
AuraRouter Managed Application Service (MAS) host for AuraGrid.

Runs AuraRouter as a long-lived service on AuraGrid using the Python SDK.
Other grid applications can discover and call aurarouter services via
the embedded ServiceServer (remoting).

Entry point: ``python -m aurarouter.auragrid.mas_host``
ProcessMasLoader starts this as a subprocess with env vars:
    AURAGRID_IPC_PORT, AURAGRID_IPC_TOKEN, AURAGRID_MAS_ID,
    AURAGRID_NODE_ID, AURAGRID_FENCING_TOKEN
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Guard SDK imports — standalone mode must not crash
_HAS_SDK = False
try:
    from auragrid import AsyncGridContext, IpcOptions
    _HAS_SDK = True
except ImportError:
    logger.info("AuraGrid SDK not available — grid features disabled")
    AsyncGridContext = None  # type: ignore[assignment,misc]
    IpcOptions = None  # type: ignore[assignment,misc]

from .config_loader import ConfigLoader
from .events import EventBridge
from .lifecycle import LifecycleCallbacks


class AuraRouterMasHost:
    """
    AuraRouter MAS host for AuraGrid.

    Creates an AsyncGridContext, registers service instances, starts
    the auction listener, and runs a health-check loop until shutdown.

    Grid Configuration Keys
    -----------------------
    Read from the ``grid`` section of ``auraconfig.yaml``:

    ``grid.health.interval``
        Health-check interval in seconds (default: 30).
    ``grid.health.max_backoff``
        Maximum backoff between health checks on consecutive failure
        (default: 120 seconds).
    ``grid.auction.vram_pressure_threshold``
        VRAM usage fraction (0-1) above which all bids are suppressed
        (default: 0.9).
    ``grid.auction.total_vram_mb``
        Total GPU VRAM in MB. When set, enables VRAM-aware bid gating.

    Environment Variables
    ---------------------
    Set by the AuraGrid ProcessMasLoader before subprocess launch:

    ``AURAGRID_IPC_PORT``
        IPC port for AsyncGridContext (default 5100).
    ``AURAGRID_IPC_TOKEN``
        Authentication token for IPC.
    ``AURAGRID_MAS_ID``
        Managed Application Service identifier.
    ``AURAGRID_NODE_ID``
        Node identifier for auction bids.
    ``AURAGRID_FENCING_TOKEN``
        Fencing token for split-brain protection.
    ``AURAGRID_MANAGED_VENV_NAME``
        Managed venv name (logged at startup for diagnostics).
    """

    def __init__(self, max_health_backoff: int = 120):
        self.lifecycle: Optional[LifecycleCallbacks] = None
        self.is_running = False
        self._max_health_backoff = max_health_backoff
        self._current_health_check_interval = 30
        self._consecutive_failures = 0
        self._auction_listener = None
        self._model_registry = None
        self._grid_context: Optional[Any] = None

    async def run(self) -> None:
        """
        Main execution loop for AuraRouter on AuraGrid.

        Creates GridContext from env vars, registers services,
        starts the auction listener, and runs until SIGTERM/SIGINT.
        """
        logger.info("AuraRouter MAS starting")

        try:
            # Load aurarouter configuration
            config_loader = ConfigLoader(allow_missing=False)
            self.lifecycle = LifecycleCallbacks(config_loader.load())
            await self.lifecycle.startup()
            self.is_running = True

            # Read grid config section
            grid_config = {}
            if hasattr(self.lifecycle, "config_loader") and hasattr(self.lifecycle.config_loader, "config"):
                grid_config = self.lifecycle.config_loader.config or {}
            auction_cfg = grid_config.get("grid", {}).get("auction", {})
            health_cfg = grid_config.get("grid", {}).get("health", {})

            # Apply configurable health check params (D16)
            self._current_health_check_interval = int(health_cfg.get("interval", 30))
            self._max_health_backoff = int(health_cfg.get("max_backoff", 120))

            # Build grid context with service registration
            event_bridge = EventBridge()  # No event client yet
            if _HAS_SDK:
                await self._setup_grid_context(event_bridge, auction_cfg)
            else:
                logger.warning("Running without AuraGrid SDK — grid features disabled")

            # Wire model registry and circuit breakers from fabric
            model_registry = None
            circuit_breaker_registry = None
            if self.lifecycle and self.lifecycle.fabric:
                circuit_breaker_registry = getattr(
                    self.lifecycle.fabric, "_circuit_breakers", None
                )
                try:
                    from aurarouter.registry import RuntimeModelRegistry
                    providers = getattr(self.lifecycle.fabric, "_provider_cache", {})
                    if providers:
                        model_registry = RuntimeModelRegistry(providers)
                        model_registry.start_polling()
                        self._model_registry = model_registry
                except Exception as e:
                    logger.warning("Could not create model registry: %s", e)

            # Start auction listener
            try:
                from .auction import AuctionListener

                node_id = os.environ.get("AURAGRID_NODE_ID", "unknown")

                self._auction_listener = AuctionListener(
                    event_bridge=event_bridge,
                    model_registry=model_registry,
                    circuit_breaker_registry=circuit_breaker_registry,
                    node_id=node_id,
                    vram_pressure_threshold=float(
                        auction_cfg.get("vram_pressure_threshold", 0.9)
                    ),
                    total_vram_mb=auction_cfg.get("total_vram_mb"),
                )
                await self._auction_listener.start()
            except Exception as e:
                logger.warning("Auction listener failed to start: %s", e)
                self._auction_listener = None

            # Log managed venv info (D15)
            mvenv_name = os.environ.get("AURAGRID_MANAGED_VENV_NAME")
            if mvenv_name:
                logger.info("Running in managed venv: %s (python: %s)", mvenv_name, sys.executable)

            logger.info("AuraRouter MAS fully initialized — entering health check loop")

            # Health check loop until shutdown
            while self.is_running:
                health = await self.lifecycle.health_check()
                if health:
                    self._consecutive_failures = 0
                    self._current_health_check_interval = int(health_cfg.get("interval", 30))
                else:
                    self._consecutive_failures += 1
                    self._current_health_check_interval = min(
                        int(health_cfg.get("interval", 30)) * (2 ** self._consecutive_failures),
                        self._max_health_backoff,
                    )
                    logger.warning(
                        "Health check failed (%d consecutive). Next check in %ds.",
                        self._consecutive_failures,
                        self._current_health_check_interval,
                    )
                await asyncio.sleep(self._current_health_check_interval)

        except Exception as e:
            logger.error("AuraRouter MAS execution failed: %s", e, exc_info=True)
            raise

        finally:
            await self._cleanup()

    async def _setup_grid_context(
        self, event_bridge: EventBridge, auction_cfg: dict
    ) -> None:
        """Create AsyncGridContext and wire the event client into EventBridge."""
        try:
            from .services import RouterService, ReasoningService, CodingService

            # Build service instances from fabric
            service_instances = []
            if self.lifecycle and self.lifecycle.fabric:
                service_instances = [
                    RouterService(self.lifecycle.fabric),
                    ReasoningService(self.lifecycle.fabric),
                    CodingService(self.lifecycle.fabric),
                ]

            options = IpcOptions()
            self._grid_context = AsyncGridContext(
                options=options,
                service_instances=service_instances,
            )
            await self._grid_context.__aenter__()

            # Wire SDK event client into EventBridge
            event_bridge.event_client = self._grid_context.events
            logger.info(
                "GridContext initialized — node=%s, mas=%s, services=%d",
                self._grid_context.node_id,
                self._grid_context.mas_id,
                len(service_instances),
            )
        except Exception as e:
            logger.warning("Failed to initialize GridContext: %s", e)
            self._grid_context = None

    async def _cleanup(self) -> None:
        """Shutdown all components in reverse order."""
        if self._auction_listener:
            try:
                await self._auction_listener.stop()
            except Exception as e:
                logger.warning("Error stopping auction listener: %s", e)

        if self._model_registry:
            try:
                self._model_registry.stop_polling()
            except Exception as e:
                logger.warning("Error stopping model registry: %s", e)

        if self._grid_context:
            try:
                await self._grid_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error closing GridContext: %s", e)

        if self.lifecycle:
            try:
                await self.lifecycle.shutdown()
            except Exception as e:
                logger.warning("Error in lifecycle shutdown: %s", e)

        self.is_running = False
        logger.info("AuraRouter MAS execution terminated")

    def request_shutdown(self) -> None:
        """Signal the health check loop to exit."""
        self.is_running = False


async def _run_mas() -> int:
    """Run the MAS host until shutdown signal."""
    host = AuraRouterMasHost()

    # Handle SIGTERM/SIGINT for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, host.request_shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for all signals
            pass

    try:
        await host.run()
        return 0
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1


def main() -> None:
    """
    Entry point for ``python -m aurarouter.auragrid.mas_host``.

    ProcessMasLoader starts this as a long-running subprocess.
    The process runs until SIGTERM or SIGINT.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    exit_code = asyncio.run(_run_mas())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
