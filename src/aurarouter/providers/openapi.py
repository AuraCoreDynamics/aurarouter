"""OpenAPI-compatible chat/completions provider.

Works with vLLM, text-generation-inference, LocalAI, LM Studio,
Ollama's OpenAI-compatible endpoint, and any server implementing
the ``/v1/chat/completions`` endpoint.

Config::

    my_vllm:
      provider: openapi
      endpoint: http://localhost:8000/v1
      model_name: meta-llama/Llama-3-8B
      api_key: optional-key      # or env_key: VLLM_API_KEY
      parameters:
        temperature: 0.7
        max_tokens: 2048
"""

from __future__ import annotations

import logging

import httpx

from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult

logger = logging.getLogger("AuraRouter.OpenAPI")


class OpenAPIProvider(BaseProvider):
    """Provider for OpenAI-API-compatible endpoints."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        model_name = self.config.get("model_name", "")
        api_key = self.resolve_api_key() or ""
        params = self.config.get("parameters", {})
        timeout = float(self.config.get("timeout", 120.0))

        url = endpoint.rstrip("/") + "/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 2048),
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        logger.debug("POST %s model=%s", url, model_name)

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Empty choices in OpenAPI response")

        # Support both chat and completion response formats.
        choice = choices[0]
        if "message" in choice:
            text = choice["message"].get("content", "")
        else:
            text = choice.get("text", "")

        usage = data.get("usage", {})

        return GenerateResult(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation via /v1/chat/completions with full history."""
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        url = endpoint.rstrip("/") + "/chat/completions"

        params = self.config.get("parameters", {})
        payload = {
            "model": self.config.get("model_name", ""),
            "messages": all_messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 2048),
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        api_key = self.resolve_api_key() or ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = float(self.config.get("timeout", 120.0))
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return GenerateResult(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model_id=self.config.get("model_name", ""),
            provider="openapi",
            context_limit=self.get_context_limit(),
        )
