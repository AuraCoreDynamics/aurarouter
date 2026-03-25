import threading
import time
from collections.abc import AsyncIterator
from typing import Callable, Dict, Optional

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.providers import get_provider, BaseProvider
from aurarouter.providers.ollama import OllamaProvider

logger = get_logger("AuraRouter.Fabric")

# Callback type: (role, model_id, success, elapsed_seconds)
ModelTriedCallback = Callable[[str, str, bool, float], None]


class ComputeFabric:
    """N-model routing with graceful degradation.

    Iterates through a role's model chain until one provider returns a
    valid response, then stops.
    """

    def __init__(self, config: ConfigLoader, ollama_discovery=None, xlm_client=None,
                 feedback_store=None, **kwargs):
        self._config = config
        self._provider_cache: Dict[str, BaseProvider] = {}
        self._ollama_discovery = ollama_discovery
        self._xlm_client = xlm_client
        self._feedback_store = feedback_store

    def update_config(self, new_config):
        self._config = new_config
        self._provider_cache.clear()

    # ------------------------------------------------------------------
    # XLM integration hooks
    # ------------------------------------------------------------------

    def _augment_prompt(self, prompt: str, role: str) -> str:
        """Call AuraXLM's auraxlm.query tool to prepend RAG context. Fail-safe."""
        if not self._config.is_xlm_augmentation_enabled():
            return prompt
        endpoint = self._config.get_xlm_endpoint()
        if not endpoint:
            return prompt
        try:
            if self._xlm_client is not None:
                client = self._xlm_client
            else:
                from aurarouter.mcp_client.client import GridMcpClient
                client = GridMcpClient(base_url=endpoint, name="xlm-augmentation", timeout=10.0)
                if not client.connect():
                    return prompt
            result = client.call_tool("auraxlm.query", prompt=prompt, role=role)
            if isinstance(result, dict) and result.get("augmented_prompt"):
                return result["augmented_prompt"]
            if isinstance(result, str) and result.strip():
                return result
        except Exception:
            logger.debug("XLM prompt augmentation failed, using original prompt", exc_info=True)
        return prompt

    def _record_feedback(self, role: str, model_id: str, success: bool,
                         elapsed: float, complexity_score: float | None = None,
                         input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record routing outcome to FeedbackStore asynchronously. Never blocks or raises."""
        if self._feedback_store is None:
            return
        score = complexity_score if complexity_score is not None else 5.0
        store = self._feedback_store

        def _write():
            try:
                store.record(
                    role=role, complexity=score, model_id=model_id,
                    success=success, latency=elapsed,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )
            except Exception:
                pass  # Fire and forget
        threading.Thread(target=_write, daemon=True).start()

    def _report_usage(self, role: str, model_id: str, success: bool,
                      elapsed: float, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Fire-and-forget usage event to AuraXLM. Never blocks or raises."""
        if not self._config.is_xlm_usage_reporting_enabled():
            return
        endpoint = self._config.get_xlm_endpoint()
        if not endpoint:
            return

        xlm_client = self._xlm_client

        def _send():
            try:
                if xlm_client is not None:
                    client = xlm_client
                else:
                    from aurarouter.mcp_client.client import GridMcpClient
                    client = GridMcpClient(base_url=endpoint, name="xlm-usage", timeout=5.0)
                    if not client.connect():
                        return
                client.call_tool("auraxlm.usage",
                    model_id=model_id, role=role, success=success,
                    elapsed_seconds=elapsed,
                    input_tokens=input_tokens, output_tokens=output_tokens)
            except Exception:
                pass  # Fire and forget
        threading.Thread(target=_send, daemon=True).start()

    def execute(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
    ) -> Optional[str]:
        chain = self._config.get_role_chain(role)
        if not chain:
            return f"ERROR: No models defined for role '{role}' in YAML."

        prompt = self._augment_prompt(prompt, role)

        errors: list[str] = []
        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider")
            logger.info(f"[{role.upper()}] Routing to: {model_id} ({provider_name})")

            start = time.monotonic()
            try:
                # Get provider from cache or create new
                if model_id in self._provider_cache:
                    provider = self._provider_cache[model_id]
                else:
                    provider = get_provider(provider_name, model_cfg)
                    self._provider_cache[model_id] = provider

                # For Ollama providers with discovery, inject discovered endpoints
                if isinstance(provider, OllamaProvider) and self._ollama_discovery:
                    endpoints = self._ollama_discovery.get_available_endpoints()
                    if endpoints:
                        provider.config["endpoints"] = endpoints
                        logger.debug(f"Injected {len(endpoints)} discovered endpoints for {model_id}")

                result = provider.generate(prompt, json_mode=json_mode)
                elapsed = time.monotonic() - start

                if result and result.strip():
                    logger.info(f"[{role.upper()}] Success from {model_id}.")
                    if on_model_tried:
                        on_model_tried(role, model_id, True, elapsed)
                    self._report_usage(role, model_id, True, elapsed)
                    self._record_feedback(role, model_id, True, elapsed)
                    return result
                else:
                    raise ValueError("Response was empty or invalid.")

            except Exception as e:
                elapsed = time.monotonic() - start
                err = f"{model_id} failed: {e}"
                logger.warning(err)
                errors.append(err)
                if on_model_tried:
                    on_model_tried(role, model_id, False, elapsed)
                self._report_usage(role, model_id, False, elapsed)
                self._record_feedback(role, model_id, False, elapsed)
                continue

        logger.critical(
            f"All nodes failed for role '{role}'. Errors: {errors}"
        )
        return None

    async def execute_stream(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
    ) -> AsyncIterator[str]:
        """Streaming variant of :meth:`execute` with fallback chain.

        Iterates through the role's model chain. For each model, attempts
        to stream tokens. If a failure occurs before any tokens are yielded,
        falls back to the next model. If tokens have already been yielded,
        the error is raised (no partial-then-retry).
        """
        chain = self._config.get_role_chain(role)
        if not chain:
            yield f"ERROR: No models defined for role '{role}' in YAML."
            return

        prompt = self._augment_prompt(prompt, role)

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider")
            logger.info(f"[{role.upper()}] Streaming from: {model_id} ({provider_name})")

            # Get provider from cache or create new
            if model_id in self._provider_cache:
                provider = self._provider_cache[model_id]
            else:
                provider = get_provider(provider_name, model_cfg)
                self._provider_cache[model_id] = provider

            # For Ollama providers with discovery, inject discovered endpoints
            if isinstance(provider, OllamaProvider) and self._ollama_discovery:
                endpoints = self._ollama_discovery.get_available_endpoints()
                if endpoints:
                    provider.config["endpoints"] = endpoints

            tokens_yielded = False
            try:
                async for token in provider.generate_stream(prompt, json_mode=json_mode):
                    tokens_yielded = True
                    yield token
                if on_model_tried:
                    on_model_tried(role, model_id, True, 0.0)
                return
            except Exception as e:
                if tokens_yielded:
                    raise
                logger.warning(f"{model_id} streaming failed: {e}")
                if on_model_tried:
                    on_model_tried(role, model_id, False, 0.0)
                continue

        yield ""
