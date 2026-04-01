import inspect
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.providers import get_provider, BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.savings.models import GenerateResult, UsageRecord

logger = get_logger("AuraRouter.Fabric")

# Callback type: (role, model_id, success, elapsed_seconds[, input_tokens, output_tokens])
ModelTriedCallback = Callable[..., None]

# ---------------------------------------------------------------------------
# Internal model attempt result (used by _try_model)
# ---------------------------------------------------------------------------

@dataclass
class _ModelAttempt:
    success: bool
    result: str | None = None
    error: str | None = None
    skipped: bool = False


# ---------------------------------------------------------------------------
# Modifications schema for structured code-editing responses (TG6)
# ---------------------------------------------------------------------------

MODIFICATIONS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "modifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "modification_type": {
                        "type": "string",
                        "enum": ["full_rewrite", "unified_diff"],
                    },
                    "content": {"type": "string", "description": "The complete code or the unified diff patch"},
                    "language": {"type": "string"},
                },
                "required": ["file_path", "modification_type", "content", "language"],
            },
        },
    },
    "required": ["modifications"],
}


def compile_modifications_schema(
    file_constraints: list[dict[str, str]] | None = None,
) -> dict:
    """Compile the modifications schema, optionally restricting per-file types.

    When file_constraints from TG2 mandate "unified_diff" for a specific file_path,
    the returned schema narrows the modification_type enum for that file to ONLY
    ["unified_diff"], using JSON Schema's if/then conditional composition.
    """
    import copy
    if not file_constraints:
        return MODIFICATIONS_SCHEMA
    schema = copy.deepcopy(MODIFICATIONS_SCHEMA)
    diff_only_files = [
        fc["path"] for fc in file_constraints
        if fc.get("preferred_modification") == "unified_diff"
    ]
    if not diff_only_files:
        return schema
    items = schema["properties"]["modifications"]["items"]
    conditionals = []
    for path in diff_only_files:
        conditionals.append({
            "if": {"properties": {"file_path": {"const": path}}},
            "then": {"properties": {"modification_type": {"type": "string", "enum": ["unified_diff"]}}},
        })
    items["allOf"] = conditionals
    return schema


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
        self._usage_store = kwargs.get("usage_store")
        self._budget_manager = kwargs.get("budget_manager")
        self._privacy_auditor = kwargs.get("privacy_auditor")
        self._privacy_store = kwargs.get("privacy_store")
        self._routing_advisors = kwargs.get("routing_advisors")
        self._sovereignty_gate = kwargs.get("sovereignty_gate")
        self._rag_pipeline = kwargs.get("rag_pipeline")

    @property
    def config(self) -> ConfigLoader:
        """Read-only access to the configuration."""
        return self._config

    def get_max_review_iterations(self) -> int:
        """Return max review-correct iterations from config."""
        return self._config.config.get("execution", {}).get("max_review_iterations", 3)

    def get_local_chain(self, role: str) -> list[str]:
        """Return only non-cloud models from the role's chain."""
        from aurarouter.savings.pricing import is_cloud_tier
        chain = self._config.get_role_chain(role)
        local: list[str] = []
        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue
            hosting_tier = model_cfg.get("hosting_tier")
            provider = model_cfg.get("provider", "")
            if not is_cloud_tier(hosting_tier, provider):
                local.append(model_id)
        return local

    def get_context_limit(self, model_id: str) -> int:
        """Return the context limit for a specific model."""
        model_cfg = self._config.get_model_config(model_id)
        if not model_cfg:
            return 0
        return model_cfg.get("context_limit", 0)

    def set_routing_advisors(self, registry) -> None:
        """Set (or clear) the routing advisors registry."""
        self._routing_advisors = registry

    # ------------------------------------------------------------------
    # Routing advisor registration API
    # ------------------------------------------------------------------

    def register_routing_advisor(self, client) -> None:
        """Register an MCP client as a routing advisor. Idempotent.

        If no routing-advisors registry exists yet, one is created automatically.
        Re-registering the same client (by ``name``) is a no-op.
        """
        if self._routing_advisors is None:
            from aurarouter.mcp_client.registry import McpClientRegistry
            self._routing_advisors = McpClientRegistry()
        name = getattr(client, "name", None) or str(id(client))
        existing = self._routing_advisors.get_clients()
        if name in existing:
            logger.debug("Routing advisor '%s' already registered, skipping", name)
            return
        self._routing_advisors.register(name, client)
        logger.info("Registered routing advisor: %s", name)

    def unregister_routing_advisor(self, client_id: str) -> None:
        """Remove a routing advisor by client identifier.

        No-op if the advisor is not registered or no registry exists.
        """
        if self._routing_advisors is None:
            return
        removed = self._routing_advisors.unregister(client_id)
        if removed:
            logger.info("Unregistered routing advisor: %s", client_id)

    def list_routing_advisors(self) -> list[str]:
        """Return identifiers of all registered routing advisors."""
        if self._routing_advisors is None:
            return []
        return list(self._routing_advisors.get_clients().keys())

    def _auto_register_catalog_advisors(self) -> int:
        """Scan the catalog for services with ``routing_advisor`` capability and register them.

        Returns the number of advisors auto-registered.
        """
        results = self._config.catalog_query(
            kind="service",
            capabilities=["routing_advisor"],
        )
        count = 0
        for entry in results:
            endpoint = entry.get("mcp_endpoint") or entry.get("endpoint", "")
            artifact_id = entry.get("artifact_id", "")
            if not endpoint:
                logger.debug(
                    "Catalog advisor '%s' has no endpoint, skipping", artifact_id,
                )
                continue
            try:
                from aurarouter.mcp_client.client import GridMcpClient
                client = GridMcpClient(
                    base_url=endpoint,
                    name=artifact_id,
                    timeout=self._config.config.get("system", {}).get("default_timeout", 30.0),
                )
                self.register_routing_advisor(client)
                count += 1
                logger.info("Auto-registered catalog advisor: %s -> %s", artifact_id, endpoint)
            except Exception:
                logger.debug(
                    "Failed to create client for catalog advisor '%s'",
                    artifact_id,
                    exc_info=True,
                )
        return count

    def filter_chain_by_intent(self, chain: list[str], intent: str) -> list[str]:
        """Filter model chain to only models declaring support for the given intent.

        If no models in the chain declare supported_intents, returns the full chain
        (backwards compatible -- intent filtering is additive, not restrictive).

        Parameters
        ----------
        chain:
            Ordered list of model IDs (a role chain).
        intent:
            The intent name to filter by.

        Returns
        -------
        Filtered chain containing only models whose ``supported_intents``
        includes the given intent, or the original chain if no models in
        the chain declare any ``supported_intents`` at all.
        """
        from aurarouter.catalog_model import CatalogArtifact

        any_declares = False
        filtered: list[str] = []

        for model_id in chain:
            # Check catalog first, then legacy models
            artifact_data = self._config.catalog_get(model_id)
            if artifact_data is None:
                # Model not found at all -- keep it (don't break chains)
                filtered.append(model_id)
                continue

            artifact = CatalogArtifact.from_dict(model_id, artifact_data)
            model_intents = artifact.supported_intents

            if model_intents:
                any_declares = True
                if intent in model_intents:
                    filtered.append(model_id)
            else:
                # Model doesn't declare supported_intents -- keep for now
                filtered.append(model_id)

        if not any_declares:
            # No models declare supported_intents -- return full chain
            return list(chain)

        return filtered

    def update_config(self, new_config):
        self._config = new_config
        self._provider_cache.clear()

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _get_provider(self, model_id: str) -> BaseProvider | None:
        """Get or create a provider for the given model_id."""
        model_cfg = self._config.get_model_config(model_id)
        if not model_cfg:
            return None
        if model_id in self._provider_cache:
            provider = self._provider_cache[model_id]
        else:
            provider_name = model_cfg.get("provider")
            provider = get_provider(provider_name, model_cfg)
            self._provider_cache[model_id] = provider

        # For Ollama providers with discovery, inject discovered endpoints
        if isinstance(provider, OllamaProvider) and self._ollama_discovery:
            endpoints = self._ollama_discovery.get_available_endpoints()
            if endpoints:
                provider.config["endpoints"] = endpoints
                logger.debug(f"Injected {len(endpoints)} discovered endpoints for {model_id}")
        return provider

    # ------------------------------------------------------------------
    # Routing advisor hooks
    # ------------------------------------------------------------------

    def consult_routing_advisors(self, role: str, chain: list[str], intent: str | None = None) -> list[str]:
        """Query registered routing advisors for chain reordering.

        Args:
            role: The role being executed (e.g., "coding", "reasoning")
            chain: Current model chain (ordered list of model IDs)
            intent: The classified intent (e.g., "SIMPLE_CODE", "sar_processing"). Advisors can
                    use this to make intent-aware reordering decisions.

        Returns:
            Reordered chain, or original chain if no advisors respond.
        """
        if self._routing_advisors is None:
            return chain
        clients = self._routing_advisors.get_clients()
        for name, client in clients.items():
            if not getattr(client, "connected", False):
                continue
            caps = client.get_capabilities()
            if "chain_reorder" not in caps:
                continue
            try:
                call_kwargs: dict = {"role": role, "chain": chain}
                if intent is not None:
                    call_kwargs["intent"] = intent
                result = client.call_tool("chain_reorder", **call_kwargs)
                if isinstance(result, dict):
                    new_chain = result.get("chain", [])
                    if new_chain:
                        return new_chain
            except Exception:
                logger.debug("Advisor '%s' failed, using original chain", name, exc_info=True)
        return chain

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
                    logger.warning(
                        "AuraXLM prompt augmentation is enabled but failed to "
                        "connect to %s — falling back to non-augmented prompt",
                        endpoint,
                    )
                    return prompt
            result = client.call_tool("auraxlm.query", prompt=prompt, role=role)
            if isinstance(result, dict) and result.get("augmented_prompt"):
                return result["augmented_prompt"]
            if isinstance(result, str) and result.strip():
                return result
        except Exception as exc:
            logger.warning(
                "AuraXLM prompt augmentation is enabled but the client failed to "
                "connect to %s: %s — falling back to non-augmented prompt",
                endpoint,
                exc,
            )
            logger.debug("XLM augmentation error details", exc_info=True)
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

    # ------------------------------------------------------------------
    # Usage store recording
    # ------------------------------------------------------------------

    def _record_usage(self, role: str, model_id: str, provider: str,
                      success: bool, elapsed: float,
                      input_tokens: int = 0, output_tokens: int = 0,
                      intent: str = "") -> None:
        """Record a usage event to the UsageStore."""
        if self._usage_store is None:
            return
        from aurarouter.savings.pricing import is_cloud_tier
        model_cfg = self._config.get_model_config(model_id)
        hosting_tier = model_cfg.get("hosting_tier") if model_cfg else None
        is_cloud = is_cloud_tier(hosting_tier, provider)
        record = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model_id,
            provider=provider,
            role=role,
            intent=intent,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_s=elapsed,
            success=success,
            is_cloud=is_cloud,
        )
        self._usage_store.record(record)

    # ------------------------------------------------------------------
    # Callback helper
    # ------------------------------------------------------------------

    @staticmethod
    def _fire_callback(callback, role, model_id, success, elapsed,
                       input_tokens=0, output_tokens=0):
        """Fire on_model_tried callback, supporting both 4-arg and 6-arg signatures."""
        if callback is None:
            return
        try:
            sig = inspect.signature(callback)
            nparams = len(sig.parameters)
        except (ValueError, TypeError):
            nparams = 4
        if nparams >= 6:
            callback(role, model_id, success, elapsed, input_tokens, output_tokens)
        else:
            callback(role, model_id, success, elapsed)

    # ------------------------------------------------------------------
    # _try_model (low-level per-model attempt)
    # ------------------------------------------------------------------

    def _try_model(
        self,
        role: str,
        model_id: str,
        generate_fn: Callable,
        *,
        on_model_tried: Optional[ModelTriedCallback] = None,
        audit_text: str = "",
        intent: str = "",
    ) -> _ModelAttempt:
        """Try executing on a single model with budget/privacy checks."""
        model_cfg = self._config.get_model_config(model_id)
        if not model_cfg:
            return _ModelAttempt(success=False, skipped=True, error="Model config not found")

        provider_name = model_cfg.get("provider", "")
        hosting_tier = model_cfg.get("hosting_tier")

        # Budget check
        if self._budget_manager is not None:
            from aurarouter.savings.pricing import is_cloud_tier
            if is_cloud_tier(hosting_tier, provider_name):
                status = self._budget_manager.check_budget(provider_name)
                if not status.allowed:
                    self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                    return _ModelAttempt(
                        success=False, skipped=True,
                        error=f"Budget blocked: {status.reason}",
                    )

        # Privacy check
        if self._privacy_auditor is not None and audit_text:
            event = self._privacy_auditor.audit(
                audit_text, model_id, provider_name, hosting_tier=hosting_tier,
            )
            if event and event.matches:
                if self._privacy_store is not None:
                    self._privacy_store.record(event)
                self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                return _ModelAttempt(
                    success=False, skipped=True,
                    error="PII detected in prompt for cloud model",
                )

        # Get provider
        provider = self._get_provider(model_id)
        if provider is None:
            return _ModelAttempt(success=False, skipped=True, error="Provider not found")

        start = time.monotonic()
        try:
            result = generate_fn(provider)
            elapsed = time.monotonic() - start

            # Extract token counts from GenerateResult if available
            input_tokens = getattr(result, "input_tokens", 0) or 0
            output_tokens = getattr(result, "output_tokens", 0) or 0

            self._fire_callback(on_model_tried, role, model_id, True, elapsed,
                                input_tokens, output_tokens)
            self._record_usage(role, model_id, provider_name, True, elapsed,
                               input_tokens, output_tokens, intent=intent)
            return _ModelAttempt(success=True, result=getattr(result, "text", str(result)))
        except Exception as e:
            elapsed = time.monotonic() - start
            self._fire_callback(on_model_tried, role, model_id, False, elapsed)
            self._record_usage(role, model_id, provider_name, False, elapsed,
                               0, 0, intent=intent)
            return _ModelAttempt(success=False, error=str(e))

    # ------------------------------------------------------------------
    # execute (main entry point)
    # ------------------------------------------------------------------

    def execute(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
        on_token: Optional[Callable[[str], None]] = None,
        options: dict | None = None,
        chain_override: list[str] | None = None,
    ) -> Optional[GenerateResult]:
        """Execute a prompt through the role's model chain.

        Returns a GenerateResult on success, or None if all models fail.
        If no models are defined for the role, returns a GenerateResult
        with an ERROR message.
        """
        # Intent-aware schema enforcement (TG6)
        intent = (options or {}).get("intent", "chat")
        actionable = intent in ("edit_code", "generate_code")
        if actionable:
            json_mode = True
            file_constraints = (options or {}).get("file_constraints")
            response_schema = compile_modifications_schema(file_constraints)
        else:
            response_schema = None

        chain = chain_override if chain_override is not None else self._config.get_role_chain(role)
        if not chain:
            return GenerateResult(text=f"ERROR: No models defined for role '{role}' in YAML.")

        # Consult routing advisors for potential chain reordering
        chain = self.consult_routing_advisors(role, chain, intent=intent)

        # Sovereignty gate: evaluate prompt and filter chain if needed
        sovereignty_result = None
        if self._sovereignty_gate is not None:
            sovereignty_result = self._sovereignty_gate.evaluate(prompt)
            chain = self._sovereignty_gate.enforce(chain, self._config, sovereignty_result)
            if not chain:
                return GenerateResult(
                    text="ERROR: Sovereignty gate filtered all models. "
                    "No local models available for sensitive content."
                )

        prompt = self._augment_prompt(prompt, role)

        # RAG enrichment: inject retrieved context into prompt
        if self._rag_pipeline is not None and self._rag_pipeline.is_enabled():
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        enriched = pool.submit(
                            asyncio.run, self._rag_pipeline.enrich(prompt)
                        ).result(timeout=6.0)
                else:
                    enriched = loop.run_until_complete(self._rag_pipeline.enrich(prompt))
                prompt = self._rag_pipeline.build_enriched_prompt(prompt, enriched)
            except Exception as exc:
                logger.warning("RAG enrichment failed in execute: %s", exc)

        errors: list[str] = []
        budget_skipped: list[str] = []

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider", "")
            hosting_tier = model_cfg.get("hosting_tier")
            logger.info(f"[{role.upper()}] Routing to: {model_id} ({provider_name})")

            # Budget check for cloud models
            if self._budget_manager is not None:
                from aurarouter.savings.pricing import is_cloud_tier
                if is_cloud_tier(hosting_tier, provider_name):
                    status = self._budget_manager.check_budget(provider_name)
                    if not status.allowed:
                        budget_skipped.append(model_id)
                        self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                        continue

            # Privacy audit for cloud-bound prompts
            if self._privacy_auditor is not None:
                event = self._privacy_auditor.audit(
                    prompt, model_id, provider_name, hosting_tier=hosting_tier,
                )
                if event and event.matches:
                    if self._privacy_store is not None:
                        self._privacy_store.record(event)
                    self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                    errors.append(f"{model_id}: PII detected, skipping cloud model")
                    continue

            start = time.monotonic()
            try:
                provider = self._get_provider(model_id)
                if provider is None:
                    continue

                gen_kwargs: dict = {"json_mode": json_mode}
                if response_schema is not None:
                    gen_kwargs["response_schema"] = response_schema
                
                if on_token:
                    # Use streaming path
                    tokens = []
                    for token in provider.generate_stream_sync(prompt, **gen_kwargs):
                        on_token(token)
                        tokens.append(token)
                    text = "".join(tokens)
                    result = GenerateResult(text=text)
                else:
                    result = provider.generate_with_usage(prompt, **gen_kwargs)
                
                elapsed = time.monotonic() - start

                if result and result.text and result.text.strip():
                    result.model_id = result.model_id or model_id
                    result.provider = result.provider or provider_name
                    logger.info(f"[{role.upper()}] Success from {model_id}.")
                    self._fire_callback(on_model_tried, role, model_id, True, elapsed,
                                        result.input_tokens, result.output_tokens)
                    self._report_usage(role, model_id, True, elapsed,
                                       result.input_tokens, result.output_tokens)
                    self._record_feedback(role, model_id, True, elapsed,
                                          input_tokens=result.input_tokens,
                                          output_tokens=result.output_tokens)
                    self._record_usage(role, model_id, provider_name, True, elapsed,
                                       result.input_tokens, result.output_tokens)
                    return result
                else:
                    raise ValueError("Response was empty or invalid.")

            except Exception as e:
                elapsed = time.monotonic() - start
                err = f"{model_id} failed: {e}"
                logger.warning(err)
                errors.append(err)
                self._fire_callback(on_model_tried, role, model_id, False, elapsed)
                self._report_usage(role, model_id, False, elapsed)
                self._record_feedback(role, model_id, False, elapsed)
                self._record_usage(role, model_id, provider_name, False, elapsed)
                continue

        # All models failed
        if budget_skipped and not errors:
            # All models were budget-blocked
            return GenerateResult(
                text="BUDGET_EXCEEDED: All cloud models blocked by budget limits. "
                     "Configure local models as fallback."
            )

        logger.critical(
            f"All nodes failed for role '{role}'. Errors: {errors}"
        )
        return None

    # ------------------------------------------------------------------
    # execute_all (compare models)
    # ------------------------------------------------------------------

    def execute_all(
        self,
        role: str,
        prompt: str,
        model_ids: list[str] | None = None,
        json_mode: bool = False,
    ) -> list[dict]:
        """Execute a prompt across all models in a role chain and collect results."""
        chain = model_ids if model_ids is not None else self._config.get_role_chain(role)
        results: list[dict] = []

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider", "")
            start = time.monotonic()
            try:
                provider = self._get_provider(model_id)
                if provider is None:
                    continue
                gen_result = provider.generate_with_usage(prompt, json_mode=json_mode)
                elapsed = time.monotonic() - start
                results.append({
                    "model_id": model_id,
                    "provider": provider_name,
                    "success": True,
                    "text": gen_result.text,
                    "input_tokens": gen_result.input_tokens,
                    "output_tokens": gen_result.output_tokens,
                    "elapsed_s": round(elapsed, 3),
                })
            except Exception as e:
                elapsed = time.monotonic() - start
                results.append({
                    "model_id": model_id,
                    "provider": provider_name,
                    "success": False,
                    "text": f"ERROR: {e}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_s": round(elapsed, 3),
                })

        return results

    # ------------------------------------------------------------------
    # execute_session (session-aware execution)
    # ------------------------------------------------------------------

    def execute_session(
        self,
        role: str,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> GenerateResult:
        """Execute a session-aware request through the role's model chain.

        Uses provider.generate_with_history() for multi-turn support.
        Extracts gist markers from the response.

        Returns a GenerateResult (never None -- errors return error text).
        """
        from aurarouter.sessions.gisting import extract_gist

        chain = self._config.get_role_chain(role)
        if not chain:
            return GenerateResult(
                text=f"ERROR: No models defined for role '{role}' in YAML."
            )

        errors: list[str] = []
        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider", "")
            try:
                provider = self._get_provider(model_id)
                if provider is None:
                    continue
                result = provider.generate_with_history(
                    messages, system_prompt=system_prompt, json_mode=json_mode,
                )
                if result and result.text and result.text.strip():
                    result.model_id = result.model_id or model_id
                    result.provider = result.provider or provider_name
                    # Extract gist from response
                    clean_text, gist_text = extract_gist(result.text)
                    result.text = clean_text
                    if gist_text:
                        result.gist = gist_text
                    return result
                else:
                    raise ValueError("Empty response")
            except Exception as e:
                errors.append(f"{model_id}: {e}")
                continue

        return GenerateResult(text=f"ERROR: All models failed for role '{role}'.")

    # ------------------------------------------------------------------
    # execute_stream
    # ------------------------------------------------------------------

    async def execute_stream(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        """Streaming variant of :meth:`execute` with fallback chain.

        Iterates through the role's model chain. For each model, attempts
        to stream tokens. If a failure occurs before any tokens are yielded,
        falls back to the next model. If tokens have already been yielded,
        the error is raised (no partial-then-retry).
        """
        # Intent-aware schema enforcement (TG6)
        intent = (options or {}).get("intent", "chat")
        actionable = intent in ("edit_code", "generate_code")
        if actionable:
            json_mode = True
            file_constraints = (options or {}).get("file_constraints")
            response_schema = compile_modifications_schema(file_constraints)
        else:
            response_schema = None

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

            provider = self._get_provider(model_id)
            if provider is None:
                continue

            tokens_yielded = False
            try:
                stream_kwargs: dict = {"json_mode": json_mode}
                if response_schema is not None:
                    stream_kwargs["response_schema"] = response_schema
                async for token in provider.generate_stream(prompt, **stream_kwargs):
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

    # -------------------------------------------------------------------
    # TG7: Speculative decoding execution
    # -------------------------------------------------------------------

    async def execute_speculative(
        self,
        task: str,
        context: str | None = None,
        notional_callback=None,
        correction_callback=None,
    ):
        """Execute a task using speculative decoding via the SpeculativeOrchestrator.

        Delegates to the orchestrator for drafter→verifier coordination.
        Returns a dict with result, or None on failure.
        """
        from aurarouter.speculative import SpeculativeOrchestrator

        orchestrator = SpeculativeOrchestrator(
            fabric=self,
            mcp_registry=getattr(self, '_mcp_registry', None),
            sovereignty_gate=self._sovereignty_gate,
            triage_router=getattr(self, '_triage_router', None),
        )
        return await orchestrator.execute_speculative(
            task=task,
            context=context,
            notional_callback=notional_callback,
            correction_callback=correction_callback,
        )
