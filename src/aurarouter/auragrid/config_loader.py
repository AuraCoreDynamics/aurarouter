"""
Configuration loader for AuraRouter on AuraGrid.

Merges configuration from multiple sources:
1. Environment variables (highest priority)
2. Manifest metadata (from AuraGrid)
3. Local auraconfig.yaml file (lowest priority)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from aurarouter.config import ConfigLoader as AuraRouterConfigLoader


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
            
            return loader
            
        except FileNotFoundError:
            if self.allow_missing:
                # Create an empty config loader for testing
                loader = AuraRouterConfigLoader(allow_missing=True)
                self._apply_env_overrides(loader)
                self._apply_manifest_overrides(loader)
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
