import asyncio
import os
import httpx
from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.GridDiscovery")

async def start_grid_discovery(fabric, interval_seconds: float = 15.0) -> None:
    """
    Asynchronous background task that polls AuraGrid for discovered service endpoints.
    Updates ComputeFabric's dynamic XLM/expert endpoints in memory.
    """
    logger.info("Initializing AuraGrid dynamic discovery loop (interval=%.1fs)", interval_seconds)
    
    # Read environment variables for local IPC bridge
    ipc_port_str = os.environ.get("AURAGRID_IPC_PORT")
    ipc_token = os.environ.get("AURAGRID_IPC_TOKEN")
    
    while True:
        try:
            endpoints = []
            
            if ipc_port_str and ipc_port_str.strip().isdigit() and ipc_token:
                # IPC Mode: query the local IPC bridge
                ipc_port = int(ipc_port_str)
                url = f"http://localhost:{ipc_port}/cell/registry/discover"
                headers = {
                    "X-AuraGrid-IPC-Token": ipc_token,
                    "X-AuraGrid-Fencing-Token": os.environ.get("AURAGRID_FENCING_TOKEN", "")
                }
                params = {"serviceName": "auraxlm"}
                
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(url, headers=headers, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        endpoints = data.get("endpoints", [])
                    else:
                        logger.debug("Local IPC bridge returned status code %d for discovery.", response.status_code)
            else:
                # Management API Mode: poll the external discovery API
                management_url = os.environ.get("AURAGRID_MANAGEMENT_URL") or "https://localhost:7087"
                url = f"{management_url.rstrip('/')}/api/discovery/endpoints"
                
                # Management API uses self-signed TLS certs, bypass verification
                async with httpx.AsyncClient(verify=False, timeout=5.0) as client:
                    response = await client.get(url)
                    if response.status_code == 200:
                        endpoints = response.json()
                    else:
                        logger.debug("Management API returned status code %d for discovery.", response.status_code)
            
            # Parse endpoints and find auraxlm instances
            discovered_xlm_endpoint = None
            
            for ep in endpoints:
                service_name = ep.get("serviceName") or ep.get("service_name")
                if service_name and service_name.lower() == "auraxlm":
                    # Check for openApiEndpoint, grpcEndpoint or host/port
                    openapi_ep = ep.get("openApiEndpoint")
                    grpc_ep = ep.get("grpcEndpoint")
                    
                    if openapi_ep:
                        discovered_xlm_endpoint = openapi_ep
                        break
                    elif grpc_ep:
                        # Replace grpc:// with http:// scheme if present
                        if grpc_ep.startswith("grpc://"):
                            discovered_xlm_endpoint = grpc_ep.replace("grpc://", "http://", 1)
                        else:
                            discovered_xlm_endpoint = grpc_ep
                        break
                    else:
                        host = ep.get("host")
                        port = ep.get("port")
                        protocol = ep.get("protocol") or "http"
                        if host and port:
                            discovered_xlm_endpoint = f"{protocol}://{host}:{port}/mcp/message"
                            break
            
            if discovered_xlm_endpoint:
                current_endpoint = fabric.config.get_xlm_endpoint()
                if discovered_xlm_endpoint != current_endpoint:
                    logger.info("Discovered new AuraXLM endpoint: %s (was %s)", discovered_xlm_endpoint, current_endpoint)
                    
                    # Update in-memory configuration
                    fabric.config.config.setdefault("xlm", {})["endpoint"] = discovered_xlm_endpoint
                    
                    # Evict any cached providers
                    with fabric._provider_cache_lock:
                        fabric._provider_cache.clear()
            
        except httpx.RequestError as exc:
            # Swallow connection or request exceptions (graceful degradation)
            logger.warning("Resilient Grid Discovery: connection or request error polling grid registry: %s", exc)
        except Exception as exc:
            logger.warning("Resilient Grid Discovery: unexpected error parsing or updating endpoints: %s", exc)
            
        await asyncio.sleep(interval_seconds)
