import httpx

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider

logger = get_logger("AuraRouter.LlamaCppServer")


class LlamaCppServerProvider(BaseProvider):
    """HTTP provider for llama-server (llama.cpp's built-in HTTP server).

    Talks to a running ``llama-server`` instance over HTTP via the
    ``/completion`` endpoint.  This requires zero native Python dependencies
    — only ``httpx`` (already a core dependency).

    Expected config in auraconfig.yaml::

        local_llama_server:
          provider: llamacpp-server
          endpoint: http://localhost:8080
          parameters:
            temperature: 0.1
            n_predict: 2048
    """

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        endpoint = self.config.get("endpoint", "http://localhost:8080")
        url = endpoint.rstrip("/") + "/completion"
        params = self.config.get("parameters", {})
        timeout = self.config.get("timeout", 120.0)

        payload: dict = {
            "prompt": prompt,
            "temperature": params.get("temperature", 0.8),
            "top_p": params.get("top_p", 0.95),
            "top_k": params.get("top_k", 40),
            "repeat_penalty": params.get("repeat_penalty", 1.1),
            "n_predict": params.get("n_predict", 2048),
            "stream": False,
        }

        if json_mode:
            payload["json_schema"] = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("content", "")
