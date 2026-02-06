import httpx

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider

logger = get_logger("AuraRouter.Ollama")


class OllamaProvider(BaseProvider):
    """Local Ollama HTTP API provider."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        url = self.config.get("endpoint", "http://localhost:11434/api/generate")
        payload: dict = {
            "model": self.config["model_name"],
            "prompt": prompt,
            "stream": False,
            "options": self.config.get("parameters", {}),
        }
        if json_mode:
            payload["format"] = "json"

        timeout = self.config.get("timeout", 120.0)

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
