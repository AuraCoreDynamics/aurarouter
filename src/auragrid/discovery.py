
import asyncio
from typing import List

class OllamaDiscovery:
    def __init__(self):
        self._endpoints = []
        self._discover_task = None

    def start(self):
        if not self._discover_task:
            self._discover_task = asyncio.create_task(self._discover_endpoints())

    async def _discover_endpoints(self):
        try:
            from auragrid.sdk.cell import get_cell_membership
            cell_membership = await get_cell_membership()

            async for members in cell_membership.watch_async():
                new_endpoints = []
                for member in members:
                    # Assuming member has a 'services' attribute which is a dictionary
                    if "ollama" in member.services:
                        endpoint = member.services["ollama"].get("endpoint")
                        if endpoint:
                            new_endpoints.append(endpoint)
                self._endpoints = new_endpoints
        except ImportError:
            # AuraGrid SDK not available, do nothing
            pass
        except Exception as e:
            # Log the error
            print(f"Error discovering Ollama endpoints: {e}")

    def get_available_endpoints(self) -> List[str]:
        return self._endpoints

    def close(self):
        if self._discover_task:
            self._discover_task.cancel()
