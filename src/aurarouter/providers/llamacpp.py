"""Managed llama-server provider.

Starts a bundled llama-server subprocess and routes inference via HTTP.
Each unique model_path gets its own ServerProcess instance, cached for
reuse across requests.
"""

import atexit
import threading
from pathlib import Path

import httpx

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider
from aurarouter.runtime import BinaryManager, ServerProcess
from aurarouter.savings.models import GenerateResult
from aurarouter.tuning import extract_gguf_metadata

logger = get_logger("AuraRouter.LlamaCpp")


class LlamaCppServerCache:
    """Thread-safe cache for managed llama-server subprocesses.

    Each unique model_path gets its own ServerProcess. The server is started
    on first access and kept running for subsequent requests.
    """

    def __init__(self) -> None:
        self._servers: dict[str, ServerProcess] = {}
        self._has_chat_template: dict[str, bool] = {}
        self._lock = threading.Lock()
        atexit.register(self.shutdown)

    def get_or_start(self, cfg: dict) -> tuple[str, bool]:
        """Start (or reuse) a server for this model.

        Returns
        -------
        tuple[str, bool]
            ``(endpoint_url, has_chat_template)``
        """
        model_path = cfg.get("model_path", "")
        resolved = str(Path(model_path).resolve())

        with self._lock:
            if resolved in self._servers and self._servers[resolved].is_running:
                return (
                    self._servers[resolved].endpoint,
                    self._has_chat_template.get(resolved, True),
                )

            if not Path(resolved).is_file():
                raise FileNotFoundError(
                    f"GGUF model not found: {resolved}\n"
                    "Download one with:  aurarouter download-model --repo <repo> --file <name>"
                )

            params = cfg.get("parameters", {})
            binary = BinaryManager.resolve_server_binary(cfg)
            timeout = cfg.get("server_timeout", 120.0)

            server = ServerProcess(
                model_path=resolved,
                binary_path=binary,
                n_ctx=params.get("n_ctx", 4096),
                n_gpu_layers=params.get("n_gpu_layers", 0),
                n_threads=params.get("n_threads"),
                verbose=params.get("verbose", False),
            )
            server.start(timeout=timeout)
            self._servers[resolved] = server

            # Detect chat template from stashed metadata or GGUF file
            gguf_meta = cfg.get("_gguf_metadata")
            if gguf_meta is not None:
                has_chat = gguf_meta.get("has_chat_template", False)
            else:
                try:
                    meta = extract_gguf_metadata(resolved)
                    has_chat = meta.get("has_chat_template", False)
                except Exception:
                    has_chat = True  # Default to chat mode for instruction-tuned models
            self._has_chat_template[resolved] = has_chat

            logger.info(
                "Server started for %s at %s (chat_template=%s)",
                Path(resolved).name,
                server.endpoint,
                "yes" if has_chat else "no",
            )
            return server.endpoint, has_chat

    def shutdown(self) -> None:
        """Stop all managed servers."""
        with self._lock:
            for path, server in list(self._servers.items()):
                try:
                    logger.info("Stopping server for: %s", path)
                    server.stop()
                except Exception:
                    pass
            self._servers.clear()
            self._has_chat_template.clear()


# Module-level singleton -- shared across all LlamaCppProvider instances.
_cache = LlamaCppServerCache()


class LlamaCppProvider(BaseProvider):
    """Managed llama-server provider -- starts a bundled llama-server subprocess
    and routes inference via HTTP."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
        endpoint, use_chat = _cache.get_or_start(self.config)
        params = self.config.get("parameters", {})
        timeout = self.config.get("timeout", 120.0)

        if use_chat:
            # /v1/chat/completions (instruction-tuned models)
            url = f"{endpoint}/v1/chat/completions"
            payload: dict = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": params.get("temperature", 0.8),
                "top_p": params.get("top_p", 0.95),
                "max_tokens": params.get("max_tokens", 2048),
                "stream": False,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

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
            )
        else:
            # /completion (base models)
            url = f"{endpoint}/completion"
            payload = {
                "prompt": prompt,
                "temperature": params.get("temperature", 0.8),
                "top_p": params.get("top_p", 0.95),
                "top_k": params.get("top_k", 40),
                "repeat_penalty": params.get("repeat_penalty", 1.1),
                "n_predict": params.get("max_tokens", 2048),
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
                input_tokens=data.get("tokens_evaluated", 0),
                output_tokens=data.get("tokens_predicted", 0),
            )

    def generate_with_history(
        self,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Multi-turn generation via managed llama-server /v1/chat/completions."""
        endpoint, _ = _cache.get_or_start(self.config)
        params = self.config.get("parameters", {})
        timeout = self.config.get("timeout", 120.0)

        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        url = f"{endpoint}/v1/chat/completions"
        payload: dict = {
            "messages": all_messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 4096),
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}

        return GenerateResult(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
            model_id=self.config.get("model_name", self.config.get("model_path", "")),
            provider="llamacpp",
            context_limit=self.get_context_limit(),
        )

    def get_context_limit(self) -> int:
        """Return context limit from config or model parameters."""
        limit = self.config.get("context_limit", 0)
        if limit > 0:
            return limit
        params = self.config.get("parameters", {})
        n_ctx = params.get("n_ctx", 0)
        if n_ctx > 0:
            return n_ctx
        return 0
