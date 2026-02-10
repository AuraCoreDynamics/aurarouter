"""
AuraRouter Managed Application Service (MAS) host for AuraGrid.

This module implements IManagedApplicationService to run AuraRouter
as a long-lived service on AuraGrid, enabling other grid applications
to call aurarouter services via gRPC proxy or events.
"""

import asyncio
import logging
import sys
from typing import Any, Optional

from .config_loader import ConfigLoader
from .lifecycle import LifecycleCallbacks

logger = logging.getLogger(__name__)

# Try to import AuraGrid SDK; provide stub if not available
try:
    from auragrid.abstractions import IManagedApplicationService, MasExecutionContext
except ImportError:
    # Stubs for standalone testing
    class MasExecutionContext:
        """Stub execution context."""
        pass

    class IManagedApplicationService:
        """Stub service interface."""
        pass


class AuraRouterMasHost(IManagedApplicationService):
    """
    AuraRouter implementation of IManagedApplicationService.

    Runs AuraRouter as a Managed Application Service on AuraGrid,
    exposing its routing, reasoning, and coding capabilities to
    other grid applications.
    """

    def __init__(self, context: Optional[MasExecutionContext] = None, max_health_backoff: int = 120):
        """
        Initialize the MAS host.

        Args:
            context: AuraGrid execution context (if running on grid)
            max_health_backoff: Maximum backoff time for health checks in seconds.
        """
        self.context = context
        self.lifecycle: Optional[LifecycleCallbacks] = None
        self.is_running = False
        self._max_health_backoff = max_health_backoff
        self._current_health_check_interval = 30
        self._consecutive_failures = 0

    async def execute_async(
        self, context: MasExecutionContext, cancellation_token: Optional[Any] = None
    ) -> None:
        """
        Main execution method for AuraGrid MAS.

        This is the entry point when AuraRouter runs on AuraGrid.
        It initializes, validates, and runs the service indefinitely
        until shutdown is requested.

        Args:
            context: AuraGrid execution context
            cancellation_token: Cancellation token for shutdown

        Raises:
            Exception: If initialization fails
        """
        self.context = context
        logger.info("AuraRouter MAS execution started")

        try:
            # Load configuration
            config_loader = ConfigLoader(allow_missing=False)
            # Initialize lifecycle with loaded config
            self.lifecycle = LifecycleCallbacks(config_loader.load())
            await self.lifecycle.startup()

            self.is_running = True
            
            # Run until cancellation requested
            if cancellation_token:
                # Wait for cancellation signal
                await cancellation_token
            else:
                # Fallback: run indefinitely with periodic health checks
                while self.is_running:
                    health = await self.lifecycle.health_check()
                    if health:
                        # On success, reset backoff
                        self._consecutive_failures = 0
                        self._current_health_check_interval = 30
                        logger.debug("Health check successful.")
                    else:
                        # On failure, increase backoff
                        self._consecutive_failures += 1
                        self._current_health_check_interval = min(
                            30 * (2 ** self._consecutive_failures),
                            self._max_health_backoff,
                        )
                        logger.warning(
                            f"Health check failed ({self._consecutive_failures} consecutive). "
                            f"Next check in {self._current_health_check_interval}s."
                        )
                    
                    await asyncio.sleep(self._current_health_check_interval)

        except Exception as e:
            logger.error(f"AuraRouter MAS execution failed: {e}", exc_info=True)
            raise

        finally:
            if self.lifecycle:
                await self.lifecycle.shutdown()
            self.is_running = False
            logger.info("AuraRouter MAS execution terminated")

    async def startup_callback(self) -> None:
        """
        AuraGrid startup lifecycle callback.

        Called before ExecuteAsync.
        """
        logger.info("AuraRouter startup callback")

    async def shutdown_callback(self) -> None:
        """
        AuraGrid shutdown lifecycle callback.

        Called after ExecuteAsync terminates.
        """
        logger.info("AuraRouter shutdown callback")
        self.is_running = False

        if self.lifecycle:
            await self.lifecycle.shutdown()


async def main(args: Optional[list] = None) -> int:
    """
    Entry point for aurarouter-mas command.

    Can be invoked directly or via AuraGrid manifest.

    Args:
        args: Command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = args or sys.argv[1:]

    if args and args[0] == "startup":
        logger.info("Startup phase")
        return 0
    elif args and args[0] == "shutdown":
        logger.info("Shutdown phase")
        return 0

    # Default: run service
    try:
        host = AuraRouterMasHost()
        context = MasExecutionContext()

        await host.execute_async(context)
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
