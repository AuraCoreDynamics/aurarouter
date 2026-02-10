import httpx

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider

logger = get_logger("AuraRouter.Ollama")


class OllamaProvider(BaseProvider):
    """Local Ollama HTTP API provider."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        endpoints = self._get_endpoints()
        
        payload: dict = {
            "model": self.config["model_name"],
            "prompt": prompt,
            "stream": False,
            "options": self.config.get("parameters", {}),
        }
        if json_mode:
            payload["format"] = "json"

        timeout = self.config.get("timeout", 120.0)
        last_error = None

        with httpx.Client(timeout=timeout) as client:
            for url in endpoints:
                try:
                    resp = client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json().get("response", "")
                except httpx.RequestError as e:
                    last_error = e
                    logger.warning(f"Ollama endpoint {url} failed: {e}")
                    continue
        
        if last_error:
            raise last_error
        return ""

    def _get_endpoints(self) -> list[str]:
        endpoints = self.config.get("endpoints", [])
        if not endpoints:
            # Fallback to single endpoint for backward compatibility
            endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")
            return [endpoint]
        return endpoints
