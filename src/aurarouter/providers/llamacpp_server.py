import httpx

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult

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
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
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
            return GenerateResult(
                text=data.get("content", ""),
                input_tokens=data.get("tokens_evaluated", 0) or 0,
                output_tokens=data.get("tokens_predicted", 0) or 0,
            )

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation via llama.cpp server /v1/chat/completions."""
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        endpoint = self.config.get("endpoint", "http://localhost:8080")
        url = f"{endpoint.rstrip('/')}/v1/chat/completions"

        params = self.config.get("parameters", {})
        payload = {
            "messages": all_messages,
            "temperature": params.get("temperature", 0.7),
            "n_predict": params.get("n_predict", 4096),
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        timeout = self.config.get("timeout", 120.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            return GenerateResult(
                text=text,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                model_id=self.config.get("model_name", ""),
                provider="llamacpp-server",
                context_limit=self.get_context_limit(),
            )
        except httpx.ConnectError:
            return super().generate_with_history(messages, system_prompt, json_mode)
