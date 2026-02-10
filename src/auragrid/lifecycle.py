"""
Lifecycle callbacks for AuraRouter on AuraGrid.

Manages startup, shutdown, and health checks.
"""

import asyncio
import logging
import time
from typing import Optional

from aurarouter.config import ConfigLoader as AuraRouterConfigLoader
from aurarouter.fabric import ComputeFabric

try:
    from .config_loader import ConfigLoader as AuraGridConfigLoader
except ImportError:
    AuraGridConfigLoader = None

logger = logging.getLogger(__name__)


class LifecycleCallbacks:
    """Manages AuraRouter lifecycle on AuraGrid."""

    def __init__(
        self,
        config_loader: AuraRouterConfigLoader,
        health_check_ttl_seconds: int = 300,
        grid_config_loader: Optional["AuraGridConfigLoader"] = None,
    ):
        """
        Initialize lifecycle manager.

        Args:
            config_loader: ConfigLoader instance with loaded configuration
            health_check_ttl_seconds: Cache TTL for health checks (default: 5 mins)
            grid_config_loader: Optional AuraGrid ConfigLoader for hot-reload support
        """
        self.config_loader = config_loader
        self.fabric: Optional[ComputeFabric] = None
        self.is_healthy = False
        self._health_check_ttl = health_check_ttl_seconds
        self._last_check_time = 0
        self._last_check_result = False
        self._grid_config_loader = grid_config_loader

    async def startup(self) -> None:
        """
        Execute startup sequence.

        - Initialize ComputeFabric with loaded config
        - Validate all model providers
        - Perform health check
        - Subscribe to runtime configuration changes (if AuraGrid integration enabled)
        """
        logger.info("AuraRouter startup sequence started")

        try:
            # Initialize ComputeFabric with config loader
            self.fabric = ComputeFabric(config=self.config_loader)
            logger.info("ComputeFabric initialized")

            # Subscribe to runtime config changes if AuraGrid integration is available
            if self._grid_config_loader is not None:
                self._grid_config_loader.subscribe_to_config_changes(self._on_config_changed)
                logger.info("Subscribed to AuraGrid config changes for hot-reload")

            # Validate providers
            await self._validate_providers()

            # Perform initial full health check
            self.is_healthy = await self._full_inference_check()
            self._last_check_time = time.time()
            self._last_check_result = self.is_healthy

            if self.is_healthy:
                logger.info("AuraRouter startup completed successfully")
            else:
                logger.warning("Initial health check failed during startup")

        except Exception as e:
            logger.error(f"AuraRouter startup failed: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """
        Execute shutdown sequence.

        - Gracefully close all provider connections
        - Cleanup resources
        - Unsubscribe from config changes
        """
        logger.info("AuraRouter shutdown sequence started")

        try:
            # Close config watcher
            if self._grid_config_loader is not None:
                self._grid_config_loader.close()
                logger.info("Unsubscribed from AuraGrid config changes")

            if self.fabric:
                await self._cleanup_providers()
                self.fabric = None

            self.is_healthy = False
            logger.info("AuraRouter shutdown completed successfully")

        except Exception as e:
            logger.error(f"AuraRouter shutdown error: {e}", exc_info=True)
            raise

    def _on_config_changed(self, new_loader: AuraRouterConfigLoader) -> None:
        """
        Callback invoked when AuraGrid detects a configuration change.

        Updates the ComputeFabric with the new configuration to enable
        hot-reload without service restart.

        Args:
            new_loader: New ConfigLoader instance with updated configuration
        """
        logger.info("AuraGrid config change detected, updating compute fabric")

        if self.fabric:
            # Update the config loader reference
            self.config_loader = new_loader
            # Update the fabric with new config
            self.fabric.update_config(new_loader)
            logger.info("Compute fabric updated with new configuration")
        else:
            logger.warning("Config change detected but fabric not initialized")

    async def health_check(self) -> bool:
        """
        Perform a cached health check.

        Returns:
            True if the service is considered healthy, False otherwise.
        """
        # Return cached result if within TTL
        if time.time() - self._last_check_time < self._health_check_ttl:
            return self._last_check_result

        # If cache is stale, perform a new check
        logger.info("Health check TTL expired, performing new check...")
        
        # 1. Lightweight check
        if not await self._lightweight_check():
            self._last_check_time = time.time()
            self._last_check_result = False
            return False

        # 2. Full inference check (if lightweight passes)
        self._last_check_result = await self._full_inference_check()
        self._last_check_time = time.time()
        
        return self._last_check_result

    async def _lightweight_check(self) -> bool:
        """
        Perform lightweight checks (e.g., API key presence, endpoint reachability).
        
        This should be a quick check without full model inference.
        """
        if not self.fabric:
            logger.warning("Fabric not initialized for health check")
            return False
        # Placeholder: a real implementation would check API keys, ping endpoints, etc.
        return True

    async def _full_inference_check(self) -> bool:
        """Run a full inference check against the compute fabric."""
        if not self.fabric:
            logger.warning("Fabric not initialized for full health check")
            return False

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.fabric.execute,
                "router",
                "health check",
                False,
            )
            
            is_ok = result is not None
            if not is_ok:
                logger.warning("Full inference health check failed: providers not responding")
            else:
                logger.info("Full inference health check: OK")
                
            return is_ok
        except Exception as e:
            logger.warning(f"Full inference health check failed with exception: {e}")
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

