from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, TYPE_CHECKING

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.mcp_client.registry import McpClientRegistry
from aurarouter.config import ConfigLoader
from aurarouter.providers import get_provider, BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.models import GenerateResult, UsageRecord
from aurarouter.savings.pricing import PricingCatalog, is_cloud_tier
from aurarouter.savings.privacy import PrivacyAuditor, PrivacyStore
from aurarouter.savings.usage_store import UsageStore

logger = get_logger("AuraRouter.Fabric")

# Callback type: (role, model_id, success, elapsed_seconds)
ModelTriedCallback = Callable[[str, str, bool, float], None]

# Extended callback: (role, model_id, success, elapsed_seconds, input_tokens, output_tokens)
ModelTriedExtCallback = Callable[[str, str, bool, float, int, int], None]


@dataclass
class _ModelAttempt:
    """Result of a single model attempt."""
    success: bool
    result: GenerateResult | None  # None if skipped/failed
    error: str  # Empty on success
    skipped: bool  # True if budget/privacy skip (not a failure)
    elapsed: float = 0.0  # Wall-clock seconds for the provider call


class ComputeFabric:
    """N-model routing with graceful degradation.

    Iterates through a role's model chain until one provider returns a
    valid response, then stops.
    """

    def __init__(
        self,
        config: ConfigLoader,
        ollama_discovery=None,
        *,
        routing_advisors: Optional[McpClientRegistry] = None,
        usage_store: Optional[UsageStore] = None,
        privacy_auditor: Optional[PrivacyAuditor] = None,
        privacy_store: Optional[PrivacyStore] = None,
        pricing_catalog: Optional[PricingCatalog] = None,
        budget_manager: Optional[BudgetManager] = None,
    ):
        self._config = config
        self._provider_cache: Dict[str, BaseProvider] = {}
        self._ollama_discovery = ollama_discovery
        self._routing_advisors = routing_advisors
        self._usage_store = usage_store
        self._privacy_auditor = privacy_auditor
        self._privacy_store = privacy_store
        self._pricing_catalog = pricing_catalog
        self._budget_manager = budget_manager

    # ------------------------------------------------------------------
    # Public accessors (avoid reaching into _config from tool code)
    # ------------------------------------------------------------------

    @property
    def config(self) -> ConfigLoader:
        """Read-only access to the current configuration."""
        return self._config

    def get_max_review_iterations(self) -> int:
        """Return the maximum number of review-correct iterations."""
        return self._config.get_max_review_iterations()

    def get_local_chain(self, role: str) -> list[str]:
        """Return only non-cloud models from a role's chain."""
        chain = self._config.get_role_chain(role)
        return [
            model_id for model_id in chain
            if not is_cloud_tier(
                self._config.get_model_hosting_tier(model_id),
                self._config.get_model_config(model_id).get("provider", ""),
            )
        ]

    def set_routing_advisors(self, registry) -> None:
        """Set the routing advisor registry (replaces direct attribute access)."""
        self._routing_advisors = registry

    def get_context_limit(self, model_id: str) -> int:
        """Return the context limit for a model, or 0 if unavailable."""
        model_cfg = self._config.get_model_config(model_id)
        if not model_cfg:
            return 0
        provider = self._get_provider(model_id, model_cfg)
        return provider.get_context_limit()

    def update_config(self, new_config):
        self._config = new_config
        self._provider_cache.clear()

        if self._pricing_catalog is not None:
            overrides_raw = new_config.get_pricing_overrides()
            new_overrides = {}
            if overrides_raw:
                from aurarouter.savings.pricing import ModelPrice
                new_overrides = {
                    k: ModelPrice(v["input_per_million"], v["output_per_million"])
                    for k, v in overrides_raw.items()
                }
            self._pricing_catalog = PricingCatalog(
                overrides=new_overrides or None,
                config_resolver=new_config.get_model_pricing,
            )

        if self._budget_manager is not None:
            self._budget_manager.update_config(new_config.get_budget_config())

        if self._privacy_auditor is not None:
            privacy_cfg = new_config.get_privacy_config()
            custom_raw = privacy_cfg.get("custom_patterns", [])
            if custom_raw:
                from aurarouter.savings.privacy import PrivacyPattern
                custom = [
                    PrivacyPattern(
                        name=p["name"], pattern=p["pattern"],
                        severity=p.get("severity", "medium"),
                        description=p.get("description", ""),
                    )
                    for p in custom_raw
                ]
                self._privacy_auditor = PrivacyAuditor(custom_patterns=custom)
            else:
                self._privacy_auditor = PrivacyAuditor()

    def _fire_callback(
        self, callback: Optional[ModelTriedCallback],
        role: str, model_id: str, success: bool,
        elapsed: float, input_tokens: int, output_tokens: int,
    ) -> None:
        """Invoke the model-tried callback, adapting to its arity."""
        if callback is None:
            return
        try:
            sig = inspect.signature(callback)
            if len(sig.parameters) >= 6:
                callback(role, model_id, success, elapsed, input_tokens, output_tokens)
            else:
                callback(role, model_id, success, elapsed)
        except Exception:
            logger.debug("on_model_tried callback raised; ignoring", exc_info=True)

    def _consult_routing_advisors(self, role: str, chain: list[str]) -> list[str]:
        """Ask registered routing advisors if the chain should be reordered."""
        if self._routing_advisors is None:
            return chain
        advisors = self._routing_advisors.get_clients_with_capability("chain_reorder")
        if not advisors:
            return chain
        for advisor in advisors:
            try:
                result = advisor.call_tool("chain_reorder", role=role, chain=chain)
                reordered = result.get("chain")
                if isinstance(reordered, list) and reordered:
                    logger.info(
                        f"[{role.upper()}] Advisor {advisor.name} reordered "
                        f"chain: {chain} -> {reordered}"
                    )
                    return reordered
            except Exception as exc:
                logger.debug(f"Routing advisor {advisor.name} failed: {exc}", exc_info=True)
        return chain

    def _get_provider(self, model_id: str, model_cfg: dict) -> BaseProvider:
        """Get a provider from cache or create a new one."""
        if model_id in self._provider_cache:
            return self._provider_cache[model_id]
        provider_name = model_cfg.get("provider", "")
        provider = get_provider(provider_name, model_cfg)
        self._provider_cache[model_id] = provider
        if isinstance(provider, OllamaProvider) and self._ollama_discovery:
            endpoints = self._ollama_discovery.get_available_endpoints()
            if endpoints:
                provider.config["endpoints"] = endpoints
        return provider

    def _record_usage(
        self, model_id: str, provider_name: str, role: str, intent: str,
        input_tokens: int, output_tokens: int, elapsed: float,
        success: bool, is_cloud: bool,
    ) -> None:
        """Write a usage record if a store is configured."""
        if self._usage_store is not None:
            self._usage_store.record(UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                model_id=model_id, provider=provider_name, role=role,
                intent=intent, input_tokens=input_tokens,
                output_tokens=output_tokens, elapsed_s=elapsed,
                success=success, is_cloud=is_cloud,
            ))

    # ------------------------------------------------------------------
    # Core per-model attempt
    # ------------------------------------------------------------------

    def _try_model(
        self,
        role: str,
        model_id: str,
        invoke_fn: Callable[[BaseProvider], GenerateResult],
        *,
        audit_text: str = "",
        intent: str = "",
        on_model_tried: Optional[ModelTriedCallback] = None,
    ) -> _ModelAttempt:
        """Run pre-flight checks, invoke a provider, and record the outcome."""
        model_cfg = self._config.get_model_config(model_id)
        if not model_cfg:
            return _ModelAttempt(success=False, result=None, error="", skipped=True)

        provider_name = model_cfg.get("provider", "")
        hosting_tier = self._config.get_model_hosting_tier(model_id)
        model_is_cloud = is_cloud_tier(hosting_tier, provider_name)
        logger.info(f"[{role.upper()}] Routing to: {model_id} ({provider_name})")

        # Budget enforcement
        if self._budget_manager is not None and model_is_cloud:
            budget_status = self._budget_manager.check_budget(provider_name)
            if not budget_status.allowed:
                logger.warning(f"[{role.upper()}] Budget exceeded for {model_id}: {budget_status.reason}")
                self._fire_callback(on_model_tried, role, model_id, False, 0.0, 0, 0)
                return _ModelAttempt(success=False, result=None, error=budget_status.reason, skipped=True)

        # Auto-tune llamacpp models
        if provider_name == "llamacpp":
            from aurarouter.tuning import auto_tune_model
            model_cfg = auto_tune_model(provider_name, model_cfg)

        start = time.monotonic()
        try:
            provider = self._get_provider(model_id, model_cfg)

            # Privacy audit
            if self._privacy_auditor is not None and audit_text:
                try:
                    event = self._privacy_auditor.audit(
                        audit_text, model_id, provider_name, hosting_tier=hosting_tier,
                    )
                    if event is not None:
                        logger.warning(
                            f"Privacy audit: {len(event.matches)} match(es) "
                            f"detected for {model_id} ({provider_name})"
                        )
                        if self._privacy_store is not None:
                            self._privacy_store.record(event)
                        model_tags = set(self._config.get_model_tags(model_id))
                        if model_is_cloud and "private" not in model_tags:
                            logger.warning(
                                f"[{role.upper()}] Skipping {model_id}: "
                                f"PII detected, model is cloud and not tagged 'private'"
                            )
                            self._fire_callback(on_model_tried, role, model_id, False, 0.0, 0, 0)
                            return _ModelAttempt(
                                success=False, result=None,
                                error=f"{model_id}: skipped (PII detected, cloud provider)",
                                skipped=True,
                            )
                except Exception:
                    logger.debug("Privacy audit raised; continuing", exc_info=True)

            gen_result = invoke_fn(provider)
            elapsed = time.monotonic() - start

            self._record_usage(
                model_id, provider_name, role, intent,
                gen_result.input_tokens, gen_result.output_tokens,
                elapsed, True, model_is_cloud,
            )
            self._fire_callback(
                on_model_tried, role, model_id, True, elapsed,
                gen_result.input_tokens, gen_result.output_tokens,
            )
            return _ModelAttempt(success=True, result=gen_result, error="", skipped=False, elapsed=elapsed)

        except Exception as e:
            elapsed = time.monotonic() - start
            err = f"{model_id} failed: {e}"
            logger.warning(err)
            self._record_usage(
                model_id, provider_name, role, intent,
                0, 0, elapsed, False, model_is_cloud,
            )
            self._fire_callback(on_model_tried, role, model_id, False, elapsed, 0, 0)
            return _ModelAttempt(success=False, result=None, error=err, skipped=False, elapsed=elapsed)

    # ------------------------------------------------------------------
    # Public execution methods
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_non_empty(gen_result: GenerateResult) -> GenerateResult:
        """Raise if the generate result text is empty or whitespace-only."""
        if not gen_result.text or not gen_result.text.strip():
            raise ValueError("Response was empty or invalid.")
        return gen_result

    def execute(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
        chain_override: Optional[list[str]] = None,
    ) -> Optional[GenerateResult]:
        chain = chain_override or self._config.get_role_chain(role)
        chain = self._consult_routing_advisors(role, chain)
        if not chain:
            return GenerateResult(text=f"ERROR: No models defined for role '{role}' in YAML.")

        errors: list[str] = []
        budget_reason: str = ""

        for model_id in chain:
            attempt = self._try_model(
                role, model_id,
                lambda p: self._validate_non_empty(
                    p.generate_with_usage(prompt, json_mode=json_mode),
                ),
                audit_text=prompt, intent="", on_model_tried=on_model_tried,
            )
            if attempt.skipped:
                if attempt.error:
                    if "skipped (PII" in attempt.error:
                        errors.append(attempt.error)
                    else:
                        budget_reason = attempt.error
                continue
            if attempt.success:
                logger.info(f"[{role.upper()}] Success from {model_id}.")
                return attempt.result
            errors.append(attempt.error)

        if budget_reason and not errors:
            return GenerateResult(text=f"BUDGET_EXCEEDED: {budget_reason}. Configure local models as fallback.")
        logger.critical(f"All nodes failed for role '{role}'. Errors: {errors}")
        return None

    def execute_session(
        self,
        role: str,
        messages: list[dict],
        system_prompt: str = "",
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
        chain_override: Optional[list[str]] = None,
    ) -> GenerateResult:
        """History-aware execution with pre-built message list.

        Unlike execute(), uses generate_with_history() with a full
        conversation history.  Does NOT touch Session state — the caller
        is responsible for all session bookkeeping.

        Args:
            role: The role to route to (e.g., "coding").
            messages: Pre-built list of {"role": ..., "content": ...} dicts.
            system_prompt: Optional system instruction.
            json_mode: Request JSON output.
            on_model_tried: Callback for each model attempt.
            chain_override: Override the role chain.

        Returns:
            GenerateResult with text, tokens, context_limit, and optional gist.
        """
        from aurarouter.sessions.gisting import extract_gist

        chain = chain_override or self._config.get_role_chain(role)
        chain = self._consult_routing_advisors(role, chain)
        errors: list[str] = []

        # Build audit text from last user message
        audit_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                audit_text = msg.get("content", "")
                break

        for model_id in chain:
            attempt = self._try_model(
                role, model_id,
                lambda p: p.generate_with_history(
                    messages=messages, system_prompt=system_prompt, json_mode=json_mode,
                ),
                audit_text=audit_text, intent="session", on_model_tried=on_model_tried,
            )
            if attempt.skipped:
                if attempt.error:
                    errors.append(f"{model_id}: {attempt.error}")
                continue
            if not attempt.success:
                errors.append(attempt.error)
                continue

            # Post-processing: extract gist, resolve context limit
            gen_result = attempt.result
            model_cfg = self._config.get_model_config(model_id)
            provider_name = model_cfg.get("provider", "") if model_cfg else ""
            provider = self._provider_cache.get(model_id)

            clean_text, gist_text = extract_gist(gen_result.text)
            context_limit = gen_result.context_limit
            if not context_limit and provider is not None:
                context_limit = provider.get_context_limit()

            return GenerateResult(
                text=clean_text, input_tokens=gen_result.input_tokens,
                output_tokens=gen_result.output_tokens, model_id=model_id,
                provider=provider_name, context_limit=context_limit, gist=gist_text,
            )

        return GenerateResult(text=f"All models failed for role '{role}': " + "; ".join(errors))

    def execute_all(
        self,
        role: str,
        prompt: str,
        *,
        model_ids: Optional[list[str]] = None,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
    ) -> list[dict]:
        """Execute prompt against ALL models in a chain, collecting every result.

        Unlike ``execute()``, does not short-circuit on first success.

        Parameters
        ----------
        model_ids:
            If provided, use this explicit list instead of the role's chain.

        Returns
        -------
        List of result dicts with keys: ``model_id``, ``provider``,
        ``success``, ``text``, ``elapsed_s``, ``input_tokens``,
        ``output_tokens``.
        """
        chain = model_ids or self._config.get_role_chain(role)
        results: list[dict] = []

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue
            provider_name = model_cfg.get("provider", "")

            attempt = self._try_model(
                role, model_id,
                lambda p: p.generate_with_usage(prompt, json_mode=json_mode),
                intent="compare", on_model_tried=on_model_tried,
            )
            if attempt.skipped:
                continue
            if attempt.success:
                gr = attempt.result
                text = gr.text
                results.append({
                    "model_id": model_id, "provider": provider_name,
                    "success": bool(text and text.strip()),
                    "text": text or "",
                    "elapsed_s": round(attempt.elapsed, 3),
                    "input_tokens": gr.input_tokens, "output_tokens": gr.output_tokens,
                })
            else:
                err_msg = attempt.error
                prefix = f"{model_id} failed: "
                if err_msg.startswith(prefix):
                    err_msg = err_msg[len(prefix):]
                results.append({
                    "model_id": model_id, "provider": provider_name,
                    "success": False, "text": f"ERROR: {err_msg}",
                    "elapsed_s": round(attempt.elapsed, 3),
                    "input_tokens": 0, "output_tokens": 0,
                })

        return results
