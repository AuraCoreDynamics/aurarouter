from __future__ import annotations

import inspect
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, TYPE_CHECKING

from aurarouter._logging import get_logger

if TYPE_CHECKING:
    from aurarouter.mcp_client.registry import McpClientRegistry
from aurarouter.config import ConfigLoader
from aurarouter.providers import get_provider, BaseProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.models import UsageRecord
from aurarouter.savings.pricing import PricingCatalog
from aurarouter.savings.privacy import PrivacyAuditor, PrivacyStore
from aurarouter.savings.usage_store import UsageStore

logger = get_logger("AuraRouter.Fabric")

# Callback type: (role, model_id, success, elapsed_seconds)
ModelTriedCallback = Callable[[str, str, bool, float], None]

# Extended callback: (role, model_id, success, elapsed_seconds, input_tokens, output_tokens)
ModelTriedExtCallback = Callable[[str, str, bool, float, int, int], None]


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

    def update_config(self, new_config):
        self._config = new_config
        self._provider_cache.clear()

        # Refresh pricing overrides if a catalog is present
        if self._pricing_catalog is not None:
            overrides_raw = new_config.get_pricing_overrides()
            if overrides_raw:
                from aurarouter.savings.pricing import ModelPrice
                new_overrides = {
                    k: ModelPrice(v["input_per_million"], v["output_per_million"])
                    for k, v in overrides_raw.items()
                }
                self._pricing_catalog = PricingCatalog(overrides=new_overrides)

        # Refresh budget config if a manager is present
        if self._budget_manager is not None:
            self._budget_manager.update_config(new_config.get_budget_config())

        # Refresh privacy patterns if an auditor is present
        if self._privacy_auditor is not None:
            privacy_cfg = new_config.get_privacy_config()
            custom_raw = privacy_cfg.get("custom_patterns", [])
            if custom_raw:
                from aurarouter.savings.privacy import PrivacyPattern
                custom = [
                    PrivacyPattern(
                        name=p["name"],
                        pattern=p["pattern"],
                        severity=p.get("severity", "medium"),
                        description=p.get("description", ""),
                    )
                    for p in custom_raw
                ]
                self._privacy_auditor = PrivacyAuditor(custom_patterns=custom)
            else:
                self._privacy_auditor = PrivacyAuditor()

    def _fire_callback(
        self,
        callback: Optional[ModelTriedCallback],
        role: str,
        model_id: str,
        success: bool,
        elapsed: float,
        input_tokens: int,
        output_tokens: int,
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
            # Never fail a request because of a callback error
            logger.debug("on_model_tried callback raised; ignoring", exc_info=True)

    def _consult_routing_advisors(
        self, role: str, chain: list[str]
    ) -> list[str]:
        """Ask registered routing advisors if the chain should be reordered.

        If no advisors are registered or none have the ``chain_reorder``
        capability, returns the chain unchanged.
        """
        if self._routing_advisors is None:
            return chain

        advisors = self._routing_advisors.get_clients_with_capability("chain_reorder")
        if not advisors:
            return chain

        for advisor in advisors:
            try:
                result = advisor.call_tool(
                    "chain_reorder",
                    role=role,
                    chain=chain,
                )
                reordered = result.get("chain")
                if isinstance(reordered, list) and reordered:
                    logger.info(
                        f"[{role.upper()}] Advisor {advisor.name} reordered chain: "
                        f"{chain} -> {reordered}"
                    )
                    return reordered
            except Exception as exc:
                logger.debug(
                    f"Routing advisor {advisor.name} failed: {exc}",
                    exc_info=True,
                )

        return chain

    def execute(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
        chain_override: Optional[list[str]] = None,
    ) -> Optional[str]:
        chain = chain_override or self._config.get_role_chain(role)
        chain = self._consult_routing_advisors(role, chain)
        if not chain:
            return f"ERROR: No models defined for role '{role}' in YAML."

        errors: list[str] = []
        budget_reason: str = ""
        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider")
            logger.info(f"[{role.upper()}] Routing to: {model_id} ({provider_name})")

            # Budget enforcement — block cloud providers when over budget
            if self._budget_manager is not None and PricingCatalog.is_cloud_provider(provider_name):
                budget_status = self._budget_manager.check_budget(provider_name)
                if not budget_status.allowed:
                    logger.warning(f"[{role.upper()}] Budget exceeded for {model_id}: {budget_status.reason}")
                    self._fire_callback(on_model_tried, role, model_id, False, 0.0, 0, 0)
                    budget_reason = budget_status.reason
                    continue

            # Auto-tune llamacpp models when parameters are absent
            if provider_name == "llamacpp":
                try:
                    from aurarouter.tuning import auto_tune_model
                    model_cfg = auto_tune_model(provider_name, model_cfg)
                except ImportError:
                    pass  # llama-cpp-python not installed

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

                # Privacy audit — auto re-route away from cloud when
                # sensitive data is detected and the model lacks a "private" tag.
                if self._privacy_auditor is not None:
                    try:
                        event = self._privacy_auditor.audit(prompt, model_id, provider_name)
                        if event is not None:
                            logger.warning(
                                f"Privacy audit: {len(event.matches)} match(es) "
                                f"detected for {model_id} ({provider_name})"
                            )
                            if self._privacy_store is not None:
                                self._privacy_store.record(event)

                            # Skip cloud models without 'private' tag when PII detected.
                            is_cloud = PricingCatalog.is_cloud_provider(provider_name)
                            model_tags = set(self._config.get_model_tags(model_id))
                            if is_cloud and "private" not in model_tags:
                                logger.warning(
                                    f"[{role.upper()}] Skipping {model_id}: "
                                    f"PII detected, model is cloud and not tagged 'private'"
                                )
                                self._fire_callback(
                                    on_model_tried, role, model_id, False, 0.0, 0, 0,
                                )
                                errors.append(
                                    f"{model_id}: skipped (PII detected, cloud provider)"
                                )
                                continue
                    except Exception:
                        logger.debug("Privacy audit raised; continuing", exc_info=True)

                gen_result = provider.generate_with_usage(prompt, json_mode=json_mode)
                elapsed = time.monotonic() - start

                text = gen_result.text
                input_tokens = gen_result.input_tokens
                output_tokens = gen_result.output_tokens

                if text and text.strip():
                    logger.info(f"[{role.upper()}] Success from {model_id}.")

                    # Record usage
                    if self._usage_store is not None:
                        is_cloud = PricingCatalog.is_cloud_provider(provider_name)
                        record = UsageRecord(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            model_id=model_id,
                            provider=provider_name,
                            role=role,
                            intent="",
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            elapsed_s=elapsed,
                            success=True,
                            is_cloud=is_cloud,
                        )
                        self._usage_store.record(record)

                    self._fire_callback(
                        on_model_tried, role, model_id, True, elapsed,
                        input_tokens, output_tokens,
                    )
                    return text
                else:
                    raise ValueError("Response was empty or invalid.")

            except Exception as e:
                elapsed = time.monotonic() - start
                err = f"{model_id} failed: {e}"
                logger.warning(err)
                errors.append(err)

                # Record failed attempt
                if self._usage_store is not None:
                    is_cloud = PricingCatalog.is_cloud_provider(provider_name)
                    record = UsageRecord(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        model_id=model_id,
                        provider=provider_name,
                        role=role,
                        intent="",
                        input_tokens=0,
                        output_tokens=0,
                        elapsed_s=elapsed,
                        success=False,
                        is_cloud=is_cloud,
                    )
                    self._usage_store.record(record)

                self._fire_callback(
                    on_model_tried, role, model_id, False, elapsed, 0, 0,
                )
                continue

        if budget_reason and not errors:
            return f"BUDGET_EXCEEDED: {budget_reason}. Configure local models as fallback."

        logger.critical(
            f"All nodes failed for role '{role}'. Errors: {errors}"
        )
        return None

    def _get_provider(self, model_id: str, model_cfg: dict) -> BaseProvider:
        """Get a provider from cache or create a new one."""
        if model_id in self._provider_cache:
            return self._provider_cache[model_id]
        provider_name = model_cfg.get("provider", "")
        provider = get_provider(provider_name, model_cfg)
        self._provider_cache[model_id] = provider

        # Inject Ollama discovery endpoints if applicable
        if isinstance(provider, OllamaProvider) and self._ollama_discovery:
            endpoints = self._ollama_discovery.get_available_endpoints()
            if endpoints:
                provider.config["endpoints"] = endpoints

        return provider

    def execute_session(
        self,
        role: str,
        session: "Session",
        message: str,
        json_mode: bool = False,
        inject_gist: bool = False,
        system_prompt: str = "",
        on_model_tried: Optional[ModelTriedCallback] = None,
        chain_override: Optional[list[str]] = None,
    ) -> "GenerateResult":
        """Session-aware execution with full message history.

        Like execute(), but uses session history and generate_with_history().

        Args:
            role: The role to route to (e.g., "coding").
            session: The Session containing message history.
            message: The new user message to add.
            json_mode: Request JSON output.
            inject_gist: Whether to inject gist instruction.
            system_prompt: Optional system instruction.
            on_model_tried: Callback for each model attempt.
            chain_override: Override the role chain.

        Returns:
            GenerateResult with text, tokens, context_limit, and optional gist.
        """
        from aurarouter.sessions.models import Message, Gist
        from aurarouter.sessions.gisting import inject_gist_instruction, extract_gist
        from aurarouter.savings.models import GenerateResult

        # Add user message to session
        session.add_message(Message(role="user", content=message))

        # Build messages list from session history
        messages = session.get_messages_as_dicts()

        # Inject gist instruction into last user message if requested
        if inject_gist and messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i] = {
                        "role": "user",
                        "content": inject_gist_instruction(messages[i]["content"]),
                    }
                    break

        # Prepend shared context
        context_prefix = session.get_context_prefix()
        if context_prefix:
            if system_prompt:
                system_prompt = f"{context_prefix}\n{system_prompt}"
            else:
                system_prompt = context_prefix

        chain = chain_override or self._config.get_role_chain(role)
        chain = self._consult_routing_advisors(role, chain)
        errors: list[str] = []

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider", "")

            # Budget enforcement
            if self._budget_manager is not None and PricingCatalog.is_cloud_provider(provider_name):
                budget_status = self._budget_manager.check_budget(provider_name)
                if not budget_status.allowed:
                    self._fire_callback(on_model_tried, role, model_id, False, 0.0, 0, 0)
                    errors.append(f"{model_id}: {budget_status.reason}")
                    continue

            # Privacy audit
            if self._privacy_auditor is not None:
                try:
                    event = self._privacy_auditor.audit(message, model_id, provider_name)
                    if event is not None:
                        if self._privacy_store is not None:
                            self._privacy_store.record(event)
                        is_cloud = PricingCatalog.is_cloud_provider(provider_name)
                        model_tags = set(self._config.get_model_tags(model_id))
                        if is_cloud and "private" not in model_tags:
                            self._fire_callback(on_model_tried, role, model_id, False, 0.0, 0, 0)
                            errors.append(f"{model_id}: PII detected, skipping cloud model")
                            continue
                except Exception:
                    logger.debug("Privacy audit raised; continuing", exc_info=True)

            start = time.monotonic()
            try:
                provider = self._get_provider(model_id, model_cfg)

                result = provider.generate_with_history(
                    messages=messages,
                    system_prompt=system_prompt,
                    json_mode=json_mode,
                )

                elapsed = time.monotonic() - start

                # Extract gist from response if present
                clean_text, gist_text = extract_gist(result.text)
                result = GenerateResult(
                    text=clean_text,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    model_id=model_id,
                    provider=provider_name,
                    context_limit=result.context_limit or provider.get_context_limit(),
                    gist=gist_text,
                )

                # Add assistant message to session
                session.add_message(Message(
                    role="assistant",
                    content=clean_text,
                    model_id=model_id,
                    tokens=result.output_tokens,
                ))

                # Update session token stats
                session.token_stats.output_tokens += result.output_tokens
                if result.context_limit > 0:
                    session.token_stats.context_limit = result.context_limit

                # Store gist in session shared context
                if gist_text:
                    session.add_gist(Gist(
                        source_role=role,
                        source_model_id=model_id,
                        summary=gist_text,
                    ))

                # Record usage
                if self._usage_store is not None:
                    is_cloud = PricingCatalog.is_cloud_provider(provider_name)
                    record = UsageRecord(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        model_id=model_id,
                        provider=provider_name,
                        role=role,
                        intent="session",
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        elapsed_s=elapsed,
                        success=True,
                        is_cloud=is_cloud,
                    )
                    self._usage_store.record(record)

                self._fire_callback(
                    on_model_tried, role, model_id, True, elapsed,
                    result.input_tokens, result.output_tokens,
                )
                return result

            except Exception as exc:
                elapsed = time.monotonic() - start
                errors.append(f"{model_id}: {exc}")
                self._fire_callback(on_model_tried, role, model_id, False, elapsed, 0, 0)
                continue

        error_text = f"All models failed for role '{role}': " + "; ".join(errors)
        return GenerateResult(text=error_text)

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

            provider_name = model_cfg.get("provider")
            logger.info(f"[{role.upper()}] Compare: routing to {model_id} ({provider_name})")
            start = time.monotonic()

            try:
                if model_id in self._provider_cache:
                    provider = self._provider_cache[model_id]
                else:
                    provider = get_provider(provider_name, model_cfg)
                    self._provider_cache[model_id] = provider

                # Inject Ollama discovery endpoints
                if isinstance(provider, OllamaProvider) and self._ollama_discovery:
                    endpoints = self._ollama_discovery.get_available_endpoints()
                    if endpoints:
                        provider.config["endpoints"] = endpoints

                gen_result = provider.generate_with_usage(prompt, json_mode=json_mode)
                elapsed = time.monotonic() - start
                text = gen_result.text
                success = bool(text and text.strip())

                # Record usage
                if self._usage_store is not None:
                    is_cloud = PricingCatalog.is_cloud_provider(provider_name)
                    record = UsageRecord(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        model_id=model_id,
                        provider=provider_name,
                        role=role,
                        intent="compare",
                        input_tokens=gen_result.input_tokens,
                        output_tokens=gen_result.output_tokens,
                        elapsed_s=elapsed,
                        success=success,
                        is_cloud=is_cloud,
                    )
                    self._usage_store.record(record)

                results.append({
                    "model_id": model_id,
                    "provider": provider_name,
                    "success": success,
                    "text": text or "",
                    "elapsed_s": round(elapsed, 3),
                    "input_tokens": gen_result.input_tokens,
                    "output_tokens": gen_result.output_tokens,
                })
                self._fire_callback(
                    on_model_tried, role, model_id, success, elapsed,
                    gen_result.input_tokens, gen_result.output_tokens,
                )

            except Exception as e:
                elapsed = time.monotonic() - start
                logger.warning(f"{model_id} failed during compare: {e}")

                if self._usage_store is not None:
                    is_cloud = PricingCatalog.is_cloud_provider(provider_name)
                    record = UsageRecord(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        model_id=model_id,
                        provider=provider_name,
                        role=role,
                        intent="compare",
                        input_tokens=0,
                        output_tokens=0,
                        elapsed_s=elapsed,
                        success=False,
                        is_cloud=is_cloud,
                    )
                    self._usage_store.record(record)

                results.append({
                    "model_id": model_id,
                    "provider": provider_name,
                    "success": False,
                    "text": f"ERROR: {e}",
                    "elapsed_s": round(elapsed, 3),
                    "input_tokens": 0,
                    "output_tokens": 0,
                })
                self._fire_callback(
                    on_model_tried, role, model_id, False, elapsed, 0, 0,
                )

        return results
