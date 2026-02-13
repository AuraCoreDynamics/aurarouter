import inspect
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from aurarouter._logging import get_logger
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
        usage_store: Optional[UsageStore] = None,
        privacy_auditor: Optional[PrivacyAuditor] = None,
        privacy_store: Optional[PrivacyStore] = None,
        pricing_catalog: Optional[PricingCatalog] = None,
        budget_manager: Optional[BudgetManager] = None,
    ):
        self._config = config
        self._provider_cache: Dict[str, BaseProvider] = {}
        self._ollama_discovery = ollama_discovery
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

    def execute(
        self,
        role: str,
        prompt: str,
        json_mode: bool = False,
        on_model_tried: Optional[ModelTriedCallback] = None,
        chain_override: Optional[list[str]] = None,
    ) -> Optional[str]:
        chain = chain_override or self._config.get_role_chain(role)
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

                # Privacy audit for cloud-bound prompts
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
