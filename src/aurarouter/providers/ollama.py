import httpx

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.Ollama")


class OllamaProvider(BaseProvider):
    """Local Ollama HTTP API provider."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
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
                    data = resp.json()
                    return GenerateResult(
                        text=data.get("response", ""),
                        input_tokens=data.get("prompt_eval_count", 0) or 0,
                        output_tokens=data.get("eval_count", 0) or 0,
                    )
                except httpx.RequestError as e:
                    last_error = e
                    logger.warning(f"Ollama endpoint {url} failed: {e}")
                    continue

        if last_error:
            raise last_error
        return GenerateResult(text="")

    def _get_endpoints(self) -> list[str]:
        endpoints = self.config.get("endpoints", [])
        if not endpoints:
            # Fallback to single endpoint for backward compatibility
            endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")
            return [endpoint]
        return endpoints
