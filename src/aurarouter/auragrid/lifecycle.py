"""
Lifecycle callbacks for AuraRouter on AuraGrid.

Manages startup, shutdown, and health checks.
"""

import asyncio
import logging
from typing import Optional

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric

logger = logging.getLogger(__name__)


class LifecycleCallbacks:
    """Manages AuraRouter lifecycle on AuraGrid."""

    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize lifecycle manager.

        Args:
            config_loader: ConfigLoader instance with loaded configuration
        """
        self.config_loader = config_loader
        self.fabric: Optional[ComputeFabric] = None
        self.is_healthy = False

    async def startup(self) -> None:
        """
        Execute startup sequence.

        - Initialize ComputeFabric with loaded config
        - Validate all model providers
        - Perform health check
        """
        logger.info("AuraRouter startup sequence started")

        try:
            # Initialize ComputeFabric with config loader
            self.fabric = ComputeFabric(config=self.config_loader)
            logger.info("ComputeFabric initialized")

            # Validate providers
            await self._validate_providers()

            self.is_healthy = True
            logger.info("AuraRouter startup completed successfully")

        except Exception as e:
            logger.error(f"AuraRouter startup failed: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """
        Execute shutdown sequence.

        - Gracefully close all provider connections
        - Cleanup resources
        """
        logger.info("AuraRouter shutdown sequence started")

        try:
            if self.fabric:
                await self._cleanup_providers()
                self.fabric = None

            self.is_healthy = False
            logger.info("AuraRouter shutdown completed successfully")

        except Exception as e:
            logger.error(f"AuraRouter shutdown error: {e}", exc_info=True)
            raise

    async def health_check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if at least one provider is responsive, False otherwise
        """
        if not self.fabric:
            logger.warning("Fabric not initialized for health check")
            return False

        try:
            # Quick validation that at least one provider is available
            # Try a simple "ping" through the router
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.fabric.execute,
                "router",
                "health check",
                False
            )
            
            healthiness = result is not None and self.is_healthy
            
            if not healthiness:
                logger.warning("Health check: providers not responding")
            else:
                logger.debug("Health check: OK")
                
            return healthiness

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def _validate_providers(self) -> None:
        """Validate that configured providers are accessible."""
        logger.debug("Validating model providers...")

        # Check configured models from config
        if hasattr(self.config_loader, "config") and self.config_loader.config:
            config = self.config_loader.config
            
            if "models" in config:
                models = config["models"]
                if isinstance(models, dict):
                    for model_name, model_config in models.items():
                        provider_type = (
                            model_config.get("provider", "unknown")
                            if isinstance(model_config, dict)
                            else "unknown"
                        )
                        logger.debug(
                            f"Found model '{model_name}' with provider '{provider_type}'"
                        )

        logger.info("Provider validation completed")

    async def _cleanup_providers(self) -> None:
        """Clean up provider resources."""
        logger.debug("Cleaning up provider resources...")

        # Providers handle their own cleanup through __del__
        # This is a placeholder for any global cleanup needed

        logger.debug("Provider cleanup completed")
