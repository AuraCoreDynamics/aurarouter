import atexit
import threading
from pathlib import Path

from llama_cpp import Llama

from aurarouter._logging import get_logger
from aurarouter.providers.base import BaseProvider
from aurarouter.savings.models import GenerateResult

logger = get_logger("AuraRouter.LlamaCpp")


class LlamaCppModelCache:
    """Thread-safe cache for loaded Llama model instances.

    GGUF models are expensive to load (seconds + VRAM), so we keep them in
    memory and reuse across requests.  Keyed by resolved model_path.
    """

    def __init__(self) -> None:
        self._models: dict[str, Llama] = {}
        self._has_chat_template: dict[str, bool] = {}
        self._lock = threading.Lock()
        atexit.register(self.shutdown)

    def get_or_load(self, cfg: dict) -> Llama:
        model_path = cfg.get("model_path", "")
        resolved = str(Path(model_path).resolve())

        with self._lock:
            if resolved not in self._models:
                if not Path(resolved).is_file():
                    raise FileNotFoundError(
                        f"GGUF model not found: {resolved}\n"
                        "Download one with:  aurarouter download-model --repo <repo> --file <name>"
                    )
                params = cfg.get("parameters", {})
                logger.info(f"Loading GGUF model: {resolved}")
                llm = Llama(
                    model_path=resolved,
                    n_ctx=params.get("n_ctx", 4096),
                    n_gpu_layers=params.get("n_gpu_layers", 0),
                    n_batch=params.get("n_batch", 512),
                    n_threads=params.get("n_threads"),
                    verbose=params.get("verbose", False),
                )
                self._models[resolved] = llm

                # Detect chat template from GGUF metadata or auto-tune
                gguf_meta = cfg.get("_gguf_metadata")
                if gguf_meta is not None:
                    has_chat = gguf_meta.get("has_chat_template", False)
                else:
                    metadata = llm.metadata or {}
                    has_chat = "tokenizer.chat_template" in metadata
                self._has_chat_template[resolved] = has_chat
                logger.info(
                    f"Model loaded: {resolved} "
                    f"(chat_template={'yes' if has_chat else 'no'})"
                )
            return self._models[resolved]

    def has_chat_template(self, cfg: dict) -> bool:
        """Check if the cached model has a chat template."""
        model_path = cfg.get("model_path", "")
        resolved = str(Path(model_path).resolve())
        return self._has_chat_template.get(resolved, True)

    def shutdown(self) -> None:
        with self._lock:
            for path in list(self._models):
                try:
                    logger.info(f"Unloading model: {path}")
                except Exception:
                    pass  # stderr may already be closed at interpreter shutdown
                del self._models[path]
            self._models.clear()
            self._has_chat_template.clear()


# Module-level singleton — shared across all LlamaCppProvider instances.
_cache = LlamaCppModelCache()


class LlamaCppProvider(BaseProvider):
    """Embedded llama.cpp inference via llama-cpp-python."""

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return self.generate_with_usage(prompt, json_mode=json_mode).text

    def generate_with_usage(
        self, prompt: str, json_mode: bool = False
    ) -> GenerateResult:
        llm = _cache.get_or_load(self.config)
        params = self.config.get("parameters", {})

        gen_kwargs: dict = {
            "max_tokens": params.get("max_tokens", 2048),
            "temperature": params.get("temperature", 0.8),
            "top_p": params.get("top_p", 0.95),
            "top_k": params.get("top_k", 40),
            "repeat_penalty": params.get("repeat_penalty", 1.1),
        }

        if json_mode:
            gen_kwargs["response_format"] = {"type": "json_object"}

        use_chat = _cache.has_chat_template(self.config)

        if use_chat:
            # Chat completion API — instruction-tuned models with a chat
            # template get proper system/user/assistant formatting.
            messages = [{"role": "user", "content": prompt}]
            response = llm.create_chat_completion(messages=messages, **gen_kwargs)
            text = response["choices"][0]["message"]["content"]
        else:
            # Raw completion API — base/completion-only models without a
            # chat template use plain text continuation.
            logger.debug("No chat template detected; using create_completion.")
            response = llm.create_completion(prompt=prompt, **gen_kwargs)
            text = response["choices"][0]["text"]

        input_tokens = 0
        output_tokens = 0
        try:
            usage = response.get("usage", {}) or {}
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
        except Exception:
            pass

        return GenerateResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
