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

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation via Ollama /api/chat endpoint."""
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)

        payload = {
            "model": self.config["model_name"],
            "messages": chat_messages,
            "stream": False,
        }

        params = self.config.get("parameters", {})
        if params:
            payload["options"] = params
        if json_mode:
            payload["format"] = "json"

        endpoint = self._resolve_chat_endpoint()
        timeout = self.config.get("timeout", 120)

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("message", {}).get("content", "")
        if not text:
            raise RuntimeError("Empty response from Ollama /api/chat")

        return GenerateResult(
            text=text,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            model_id=self.config.get("model_name", ""),
            provider="ollama",
            context_limit=self.get_context_limit(),
        )

    def _resolve_chat_endpoint(self) -> str:
        """Resolve the /api/chat endpoint from config."""
        endpoints = self._get_endpoints()
        if endpoints:
            base = endpoints[0].rsplit("/api/", 1)[0]
            return f"{base}/api/chat"
        return "http://localhost:11434/api/chat"

    def get_context_limit(self) -> int:
        """Return context limit from config, or query Ollama /api/show."""
        limit = self.config.get("context_limit", 0)
        if limit > 0:
            return limit
        try:
            endpoints = self._get_endpoints()
            base = endpoints[0].rsplit("/api/", 1)[0] if endpoints else "http://localhost:11434"
            resp = httpx.post(
                f"{base}/api/show",
                json={"name": self.config["model_name"]},
                timeout=10,
            )
            if resp.status_code == 200:
                info = resp.json()
                params = info.get("model_info", {})
                for key, value in params.items():
                    if "context_length" in key:
                        return int(value)
        except Exception:
            pass
        return 0

    def _get_endpoints(self) -> list[str]:
        endpoints = self.config.get("endpoints", [])
        if not endpoints:
            # Fallback to single endpoint for backward compatibility
            endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")
            return [endpoint]
        return endpoints
