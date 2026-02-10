"""
Grid-aware Ollama endpoint discovery via ICellMembership.

Discovers healthy Ollama endpoints from AuraGrid cell membership and
provides health checking, caching, and retry logic.
"""

import asyncio
import time
from typing import Callable, List, Optional

import httpx

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Discovery")


class OllamaDiscovery:
    """Discovers and monitors Ollama endpoints from AuraGrid cell membership."""

    def __init__(self, default_endpoint: str = "http://localhost:11434"):
        """
        Initialize endpoint discovery.

        Args:
            default_endpoint: Fallback endpoint when grid discovery fails
        """
        self._endpoints: List[str] = []
        self._default_endpoint = default_endpoint
        self._discover_task: Optional[asyncio.Task] = None
        self._watch_tasks: List[asyncio.Task] = []
        self._retry_delay = 5.0  # Start with 5s
        self._max_retry_delay = 30.0  # Cap at 30s
        self._grid_available = False
        
        # Endpoint caching with 60s TTL
        self._endpoint_cache: List[str] = []
        self._cache_timestamp = 0.0
        self._cache_ttl = 60.0

    def start(self) -> None:
        """Start background discovery task."""
        if not self._discover_task:
            self._discover_task = asyncio.create_task(self._discover_endpoints())
            logger.info("Started Ollama endpoint discovery")

    async def _discover_endpoints(self) -> None:
        """
        Discover Ollama endpoints from AuraGrid cell membership with retry logic.

        Implements exponential backoff: 5s → 10s → 30s (capped).
        """
        while True:
            try:
                from auragrid.sdk.cell import get_cell_membership
                
                self._grid_available = True
                cell_membership = await get_cell_membership()
                logger.info("Connected to AuraGrid cell membership")
                
                # Reset retry delay on successful connection
                self._retry_delay = 5.0

                async for members in cell_membership.watch_async():
                    new_endpoints = []
                    for member in members:
                        # Extract Ollama service endpoints
                        if hasattr(member, "services") and isinstance(member.services, dict):
                            if "ollama" in member.services:
                                service = member.services["ollama"]
                                endpoint = service.get("endpoint") if isinstance(service, dict) else None
                                if endpoint:
                                    new_endpoints.append(endpoint)
                                    logger.debug(f"Discovered Ollama endpoint: {endpoint}")
                    
                    # Update endpoints atomically
                    self._endpoints = new_endpoints
                    self._invalidate_cache()
                    logger.info(f"Updated endpoints: {len(new_endpoints)} available")

            except ImportError:
                # AuraGrid SDK not available
                self._grid_available = False
                logger.debug("AuraGrid SDK not available, using default endpoint")
                self._endpoints = [self._default_endpoint] if self._default_endpoint else []
                break  # No need to retry if SDK is not installed

            except asyncio.CancelledError:
                logger.debug("Discovery task cancelled")
                raise

            except Exception as e:
                # Log and retry with backoff
                logger.warning(
                    f"Error discovering endpoints, retrying in {self._retry_delay}s: {e}"
                )
                await asyncio.sleep(self._retry_delay)
                
                # Exponential backoff with cap
                self._retry_delay = min(self._retry_delay * 2, self._max_retry_delay)

    def get_available_endpoints(self) -> List[str]:
        """
        Get list of currently available endpoints.

        Returns cached endpoints if cache is valid (60s TTL).

        Returns:
            List of endpoint URLs
        """
        # Check cache validity
        now = time.time()
        if now - self._cache_timestamp < self._cache_ttl and self._endpoint_cache:
            return self._endpoint_cache

        # Update cache
        if self._endpoints:
            self._endpoint_cache = self._endpoints.copy()
        elif self._default_endpoint:
            self._endpoint_cache = [self._default_endpoint]
        else:
            self._endpoint_cache = []
        
        self._cache_timestamp = now
        return self._endpoint_cache

    async def get_healthy_endpoint(self, timeout: float = 5.0) -> Optional[str]:
        """
        Get a healthy endpoint via HTTP health probe.

        Probes each endpoint's /api/tags endpoint to verify it's responsive.

        Args:
            timeout: HTTP timeout in seconds

        Returns:
            First healthy endpoint URL, or None if all unhealthy
        """
        endpoints = self.get_available_endpoints()
        
        if not endpoints:
            logger.warning("No endpoints available for health check")
            return None

        async with httpx.AsyncClient(timeout=timeout) as client:
            for endpoint in endpoints:
                # Convert generate endpoint to tags endpoint for health check
                health_url = endpoint.replace("/api/generate", "/api/tags")
                
                try:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        logger.debug(f"Healthy endpoint found: {endpoint}")
                        return endpoint
                except httpx.RequestError as e:
                    logger.debug(f"Endpoint {endpoint} health check failed: {e}")
                    continue

        logger.warning("No healthy endpoints found")
        return None

    def watch_membership_changes(
        self, callback: Callable[[List[str]], None]
    ) -> None:
        """
        Watch for membership changes and invoke callback with new endpoint list.

        Args:
            callback: Function to call with updated endpoint list
        """
        task = asyncio.create_task(self._watch_membership(callback))
        self._watch_tasks.append(task)
        logger.info("Started watching membership changes")

    async def _watch_membership(
        self, callback: Callable[[List[str]], None]
    ) -> None:
        """
        Internal watcher that monitors endpoint changes and invokes callback.

        Args:
            callback: Function to call with new endpoint list
        """
        last_endpoints = []
        
        try:
            while True:
                current_endpoints = self.get_available_endpoints()
                
                # Invoke callback only if endpoints changed
                if current_endpoints != last_endpoints:
                    logger.debug(f"Membership changed: {len(current_endpoints)} endpoints")
                    callback(current_endpoints)
                    last_endpoints = current_endpoints.copy()
                
                # Poll every 5 seconds
                await asyncio.sleep(5.0)

        except asyncio.CancelledError:
            logger.debug("Membership watch cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in membership watch: {e}", exc_info=True)

    def _invalidate_cache(self) -> None:
        """Invalidate the endpoint cache."""
        self._cache_timestamp = 0.0
        self._endpoint_cache = []

    def close(self) -> None:
        """Close discovery and cancel all tasks."""
        logger.info("Closing Ollama discovery")
        
        # Cancel main discovery task
        if self._discover_task:
            self._discover_task.cancel()
            self._discover_task = None
        
        # Cancel all watch tasks
        for task in self._watch_tasks:
            task.cancel()
        self._watch_tasks.clear()
        
        logger.debug("All discovery tasks cancelled")
