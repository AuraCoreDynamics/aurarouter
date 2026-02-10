"""
Configuration loader for AuraRouter on AuraGrid.

Merges configuration from multiple sources:
1. Environment variables (highest priority)
2. Manifest metadata (from AuraGrid)
3. Local auraconfig.yaml file (lowest priority)
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader as AuraRouterConfigLoader

logger = get_logger("AuraRouter.ConfigLoader")


class ConfigLoader:
    """Loads aurarouter configuration merged from multiple sources."""

    def __init__(
        self,
        manifest_metadata: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[Path] = None,
        allow_missing: bool = False,
    ):
        """
        Initialize config loader.

        Args:
            manifest_metadata: Configuration from AuraGrid manifest
            config_file_path: Path to auraconfig.yaml file
            allow_missing: If True, don't fail if config not found
        """
        self.manifest_metadata = manifest_metadata or {}
        self.config_file_path = config_file_path
        self.allow_missing = allow_missing
        self._watch_task: Optional[asyncio.Task] = None
        self._current_loader: Optional[AuraRouterConfigLoader] = None
        self._grid_config_available: bool = False

    def load(self) -> AuraRouterConfigLoader:
        """
        Load configuration with proper precedence.

        Precedence (highest to lowest):
        1. Environment variables (AURAROUTER_*)
        2. Manifest metadata from AuraGrid
        3. Local auraconfig.yaml file
        4. Built-in defaults

        Returns:
            AuraRouterConfigLoader instance ready for ComputeFabric initialization

        Raises:
            ValueError: If required configuration is missing
            FileNotFoundError: If config file specified but not found
        """
        # First, try to load the aurarouter ConfigLoader normally
        # This respects its default search paths
        try:
            loader = AuraRouterConfigLoader(
                config_path=str(self.config_file_path) if self.config_file_path else None,
                allow_missing=self.allow_missing,
            )
            
            # Apply environment variable overrides
            self._apply_env_overrides(loader)
            
            # Apply manifest metadata overrides
            self._apply_manifest_overrides(loader)
            
            # Store current loader for runtime updates
            self._current_loader = loader
            return loader
            
        except FileNotFoundError:
            if self.allow_missing:
                # Create an empty config loader for testing
                loader = AuraRouterConfigLoader(allow_missing=True)
                self._apply_env_overrides(loader)
                self._apply_manifest_overrides(loader)
                self._current_loader = loader
                return loader
            raise

    def _apply_env_overrides(self, loader: AuraRouterConfigLoader) -> None:
        """Apply environment variable overrides to config."""
        prefix = "AURAROUTER_"
        
        if not hasattr(loader, "config"):
            loader.config = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys with __ separator
                # e.g., AURAROUTER_MODELS__LOCAL_QWEN__ENDPOINT
                if "__" in config_key:
                    self._set_nested_value(loader.config, config_key.split("__"), value)
                else:
                    loader.config[config_key] = value

    def _apply_manifest_overrides(self, loader: AuraRouterConfigLoader) -> None:
        """Apply AuraGrid manifest metadata overrides."""
        if not hasattr(loader, "config"):
            loader.config = {}
            
        for key, value in self.manifest_metadata.items():
            if value is not None:
                if isinstance(value, dict) and key in loader.config:
                    loader.config[key].update(value)
                else:
                    loader.config[key] = value

    @staticmethod
    def _set_nested_value(d: Dict[str, Any], keys: list, value: Any) -> None:
        """Set a nested value in a dictionary using a list of keys."""
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def subscribe_to_config_changes(
        self, callback: Callable[[AuraRouterConfigLoader], None]
    ) -> None:
        """
        Subscribe to runtime configuration changes from AuraGrid.

        When configuration changes are detected, the callback will be invoked
        with a newly created AuraRouterConfigLoader instance containing the
        updated configuration.

        Args:
            callback: Function to call when config changes, receives new ConfigLoader

        Note:
            Requires AuraGrid SDK to be installed. If SDK is not available,
            subscription is silently skipped.
        """
        # Check if SDK is available
        try:
            from auragrid.sdk.cell import get_cell_config
            self._grid_config_available = True
        except ImportError:
            logger.debug("AuraGrid SDK not available, config subscription skipped")
            self._grid_config_available = False
            return

        if self._watch_task is None:
            self._watch_task = asyncio.create_task(self._watch_config_changes(callback))
            logger.info("Subscribed to AuraGrid config changes")

    async def _watch_config_changes(
        self, callback: Callable[[AuraRouterConfigLoader], None]
    ) -> None:
        """
        Watch for configuration changes from AuraGrid cell config.

        This method continuously watches for changes to the 'aurarouter'
        configuration in the AuraGrid cell and invokes the callback whenever
        changes are detected.

        Args:
            callback: Function to call when config changes
        """
        try:
            from auragrid.sdk.cell import get_cell_config

            cell_config = await get_cell_config()
            logger.info("Watching AuraGrid cell config for 'aurarouter' changes")

            async for change in cell_config.watch_async("aurarouter"):
                logger.debug(f"Config change detected: {list(change.keys())}")
                
                # Merge the change into manifest_metadata
                for key, value in change.items():
                    if isinstance(value, dict) and key in self.manifest_metadata:
                        self.manifest_metadata[key].update(value)
                    else:
                        self.manifest_metadata[key] = value
                    
                    # Apply individual config changes
                    self._apply_config_change(key, value)

                # Reload configuration with updated metadata
                new_loader = self.load()
                callback(new_loader)
                logger.info("Config reloaded and callback invoked")

        except ImportError:
            # AuraGrid SDK not available, should not happen if subscribe_to_config_changes checks first
            logger.debug("AuraGrid SDK not available in watch loop")
        except asyncio.CancelledError:
            logger.debug("Config watch task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error watching config changes: {e}", exc_info=True)

    def _apply_config_change(self, key: str, value: Any) -> None:
        """
        Apply individual config changes for known keys.

        Args:
            key: Configuration key that changed
            value: New value for the key
        """
        if not self._current_loader or not hasattr(self._current_loader, "config"):
            return

        config = self._current_loader.config

        # Handle known config keys
        if key == "endpoint":
            # Update default endpoint for providers
            logger.info(f"Updating default endpoint to: {value}")
            if "models" in config:
                for model_config in config.get("models", {}).values():
                    if isinstance(model_config, dict) and "endpoint" not in model_config:
                        model_config["endpoint"] = value

        elif key == "enable_role" or key == "disable_role":
            # Enable/disable specific routing roles
            role = value
            enabled = key == "enable_role"
            logger.info(f"{'Enabling' if enabled else 'Disabling'} role: {role}")
            if "roles" in config:
                if role in config["roles"]:
                    config["roles"][role]["enabled"] = enabled

        elif key == "fallback_chain":
            # Update fallback chain ordering
            logger.info(f"Updating fallback chain: {value}")
            config["fallback_chain"] = value

        elif key == "models":
            # Update entire models configuration
            logger.info("Updating models configuration")
            config["models"] = value

        else:
            # Generic key update
            logger.debug(f"Applying generic config change: {key}")
            config[key] = value

    def unsubscribe(self) -> None:
        """
        Unsubscribe from configuration changes and cancel watch task.

        This method is safe to call multiple times.
        """
        if self._watch_task is not None:
            logger.info("Unsubscribing from config changes")
            self._watch_task.cancel()
            
            # Wait for cancellation to complete
            try:
                # Try to await the task if we're in an async context
                # If not, we just cancel it and move on
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the await for later
                    asyncio.create_task(self._await_cancellation())
                else:
                    # We can await directly
                    loop.run_until_complete(self._await_cancellation())
            except RuntimeError:
                # No event loop, just cancel
                pass
            
            self._watch_task = None
            logger.debug("Config watch task cancelled")

    async def _await_cancellation(self) -> None:
        """Helper to await task cancellation and suppress CancelledError."""
        if self._watch_task is not None:
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass  # Expected on cancellation

    def close(self) -> None:
        """
        Close the config loader and cancel any active watch tasks.

        Should be called during shutdown to cleanup resources.
        """
        self.unsubscribe()
