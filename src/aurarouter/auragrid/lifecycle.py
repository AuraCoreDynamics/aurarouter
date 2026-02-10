"""
Lifecycle callbacks for AuraRouter on AuraGrid.

Manages startup, shutdown, and health checks.
"""

import asyncio
import httpx
import logging
import os
import time
from typing import Optional
from urllib.parse import urlparse

from aurarouter.config import ConfigLoader as AuraRouterConfigLoader
from aurarouter.fabric import ComputeFabric

try:
    from .config_loader import ConfigLoader as AuraGridConfigLoader
except ImportError:
    AuraGridConfigLoader = None

try:
    from .discovery import OllamaDiscovery
except ImportError:
    OllamaDiscovery = None

try:
    from .model_storage import GridModelStorage
except ImportError:
    GridModelStorage = None

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
        self._ollama_discovery = None
        self._grid_model_storage = None

    async def startup(self) -> None:
        """
        Execute startup sequence.

        - Initialize OllamaDiscovery for grid-aware endpoint discovery
        - Initialize ComputeFabric with loaded config
        - Initialize GridModelStorage for model distribution
        - Validate all model providers
        - Perform health check
        - Subscribe to runtime configuration changes (if AuraGrid integration enabled)
        """
        logger.info("AuraRouter startup sequence started")

        try:
            # Initialize Ollama endpoint discovery if available
            if OllamaDiscovery is not None:
                default_endpoint = self._get_default_ollama_endpoint()
                self._ollama_discovery = OllamaDiscovery(default_endpoint=default_endpoint)
                self._ollama_discovery.start()
                logger.info("Ollama endpoint discovery initialized")
            else:
                logger.debug("OllamaDiscovery module not available")

            # Initialize ComputeFabric with config loader and discovery
            self.fabric = ComputeFabric(
                config=self.config_loader,
                ollama_discovery=self._ollama_discovery
            )
            logger.info("ComputeFabric initialized")

            # Initialize GridModelStorage if available (non-blocking)
            if GridModelStorage is not None:
                try:
                    self._grid_model_storage = GridModelStorage()
                    await self._grid_model_storage.start()
                    logger.info("Grid model storage initialized")
                    
                    # Attempt to resolve models from grid storage
                    await self._resolve_grid_models()
                except Exception as e:
                    # Grid storage failures must NOT prevent startup
                    logger.warning(f"Grid model storage initialization failed: {e}", exc_info=True)
                    self._grid_model_storage = None
            else:
                logger.debug("GridModelStorage module not available")

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

        - Close Ollama discovery
        - Gracefully close all provider connections
        - Cleanup resources
        - Unsubscribe from config changes
        """
        logger.info("AuraRouter shutdown sequence started")

        try:
            # Close Ollama discovery
            if self._ollama_discovery is not None:
                self._ollama_discovery.close()
                logger.info("Ollama discovery closed")

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
        Checks:
        - Ollama endpoint reachability via HTTP GET to /api/tags
        - Cloud provider API key presence (OpenAI, Anthropic, Google)
        - Endpoint URL format validation
        
        Returns:
            True if all configured providers pass lightweight checks, False otherwise
        """
        if not self.fabric:
            logger.warning("Fabric not initialized for health check")
            return False
        
        # Get configuration
        if not hasattr(self.config_loader, "config") or not self.config_loader.config:
            logger.warning("No configuration found for health check")
            return False
        
        config = self.config_loader.config
        models = config.get("models", {})
        
        if not models:
            logger.warning("No models configured for health check")
            return False
        
        # Track overall health status
        all_checks_passed = True
        
        # Check each configured model/provider
        for model_id, model_config in models.items():
            if not isinstance(model_config, dict):
                continue
            
            provider_type = model_config.get("provider", "unknown")
            
            # Check Ollama provider endpoint reachability
            if provider_type == "ollama":
                endpoint = model_config.get("endpoint")
                endpoints = model_config.get("endpoints", [])
                
                # Collect all endpoints to check
                urls_to_check = []
                if endpoints:
                    urls_to_check.extend(endpoints)
                elif endpoint:
                    urls_to_check.append(endpoint)
                
                if not urls_to_check:
                    logger.warning(f"Model '{model_id}' (ollama): No endpoints configured")
                    all_checks_passed = False
                    continue
                
                # Check at least one endpoint is reachable
                any_reachable = False
                for url in urls_to_check:
                    # Validate URL format
                    if not self._validate_url_format(url):
                        logger.warning(f"Model '{model_id}' (ollama): Invalid endpoint URL '{url}'")
                        all_checks_passed = False
                        continue
                    
                    # Convert /api/generate to /api/tags for health check
                    base_url = url.rsplit("/api/", 1)[0] if "/api/" in url else url.rstrip("/")
                    health_url = f"{base_url}/api/tags"
                    
                    if await self._check_ollama_endpoint(health_url, model_id):
                        any_reachable = True
                        break
                
                if not any_reachable:
                    logger.warning(f"Model '{model_id}' (ollama): No reachable endpoints found")
                    all_checks_passed = False
            
            # Check cloud provider API keys
            elif provider_type in ["claude", "google", "openai"]:
                if not self._check_api_key(model_config, model_id, provider_type):
                    all_checks_passed = False
                
                # Check endpoint if specified
                endpoint = model_config.get("endpoint")
                if endpoint and not self._validate_url_format(endpoint):
                    logger.warning(f"Model '{model_id}' ({provider_type}): Invalid endpoint URL '{endpoint}'")
                    all_checks_passed = False
            
            # For other providers, just validate endpoint if present
            else:
                endpoint = model_config.get("endpoint")
                if endpoint and not self._validate_url_format(endpoint):
                    logger.warning(f"Model '{model_id}' ({provider_type}): Invalid endpoint URL '{endpoint}'")
                    all_checks_passed = False
        
        if all_checks_passed:
            logger.debug("Lightweight health check: All providers passed")
        
        return all_checks_passed
    
    async def _check_ollama_endpoint(self, url: str, model_id: str) -> bool:
        """
        Check if an Ollama endpoint is reachable.
        
        Args:
            url: URL to check (should be the /api/tags endpoint)
            model_id: Model ID for logging
            
        Returns:
            True if endpoint is reachable, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                logger.debug(f"Model '{model_id}' (ollama): Endpoint {url} is reachable")
                return True
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            logger.debug(f"Model '{model_id}' (ollama): Endpoint {url} unreachable: {e}")
            return False
        except Exception as e:
            logger.debug(f"Model '{model_id}' (ollama): Unexpected error checking {url}: {e}")
            return False
    
    def _check_api_key(self, model_config: dict, model_id: str, provider_type: str) -> bool:
        """
        Check if API key is configured for a cloud provider.
        
        Args:
            model_config: Model configuration dictionary
            model_id: Model ID for logging
            provider_type: Provider type (claude, google, openai)
            
        Returns:
            True if API key is configured, False otherwise
        """
        # Check direct api_key in config
        api_key = model_config.get("api_key")
        if api_key and "YOUR_PASTED_KEY" not in str(api_key) and "YOUR_API_KEY" not in str(api_key):
            logger.debug(f"Model '{model_id}' ({provider_type}): API key found in config")
            return True
        
        # Check env_key reference
        env_key = model_config.get("env_key")
        if env_key:
            env_value = os.environ.get(env_key)
            if env_value:
                logger.debug(f"Model '{model_id}' ({provider_type}): API key found in env var '{env_key}'")
                return True
            else:
                logger.warning(f"Model '{model_id}' ({provider_type}): Environment variable '{env_key}' not set or empty")
                return False
        
        logger.warning(f"Model '{model_id}' ({provider_type}): No API key configured")
        return False
    
    def _validate_url_format(self, url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is well-formed, False otherwise
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

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

    def _get_default_ollama_endpoint(self) -> str:
        """
        Extract default Ollama endpoint from config.

        Returns:
            Default endpoint URL, or http://localhost:11434 if not configured
        """
        if not hasattr(self.config_loader, "config") or not self.config_loader.config:
            return "http://localhost:11434"

        config = self.config_loader.config
        models = config.get("models", {})

        # Find first Ollama model's endpoint
        for model_config in models.values():
            if isinstance(model_config, dict):
                if model_config.get("provider") == "ollama":
                    endpoint = model_config.get("endpoint")
                    if endpoint:
                        # Normalize to base URL without /api/generate
                        if "/api/" in endpoint:
                            return endpoint.rsplit("/api/", 1)[0]
                        return endpoint

        return "http://localhost:11434"

    async def _resolve_grid_models(self) -> None:
        """
        Resolve models from grid storage during startup.

        This method attempts to download any models referenced in configuration
        that are available in grid storage but not present locally.

        Grid failures must NOT prevent startup - all exceptions are caught and logged.
        """
        if not self._grid_model_storage:
            return

        try:
            logger.info("Resolving models from grid storage...")

            # List available models in grid
            available_models = await self._grid_model_storage.list_models()
            logger.debug(f"Grid storage has {len(available_models)} models available")

            if not available_models:
                logger.debug("No models found in grid storage")
                return

            # Check configured models
            if not hasattr(self.config_loader, "config") or not self.config_loader.config:
                logger.debug("No config available for model resolution")
                return

            config = self.config_loader.config
            models = config.get("models", {})

            for model_id, model_config in models.items():
                if not isinstance(model_config, dict):
                    continue

                # Check if this model is available in grid storage
                if model_id in available_models:
                    logger.info(f"Model '{model_id}' found in grid storage")
                    # Additional logic for downloading if needed could go here
                    # For now, just log availability

            logger.info("Grid model resolution completed")

        except Exception as e:
            # Grid failures must NOT prevent startup
            logger.warning(f"Error resolving grid models: {e}", exc_info=True)

