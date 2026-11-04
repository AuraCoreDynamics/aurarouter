import asyncio
import contextvars
import inspect
import os
import queue
import random
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from aurarouter._logging import get_logger
from aurarouter.circuit_breaker import CircuitBreakerRegistry
from aurarouter.config import ConfigLoader
from aurarouter.errors import (
    ContextLengthExceededError,
    RateLimitExceededError,
)
from aurarouter.event_reporter import EventReporter
from aurarouter.grid_discovery import start_grid_discovery
from aurarouter.models.routing_context import RoutingContext
from aurarouter.providers import BaseProvider, get_provider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.ratelimit import RateLimitQuota, RateLimitTracker, RateState
from aurarouter.routing_schema import (
    normalize_failover_projection_config,
    normalize_rate_limits_config,
    normalize_routing_stack_config,
    normalize_streaming_config,
)
from aurarouter.routing_stack import (
    CandidateEvalContext,
    PosteriorSnapshotCache,
    RoutingDecisionTrace,
    RoutingPolicyStack,
)
from aurarouter.savings.models import GenerateResult, UsageRecord
from aurarouter.savings.pricing import resolve_canonical_vendor
from aurarouter.sessions.projector import (
    SessionProjector,
    TargetContextBudget,
)

logger = get_logger("AuraRouter.Fabric")

# Callback type: (role, model_id, success, elapsed_seconds[, input_tokens, output_tokens])
ModelTriedCallback = Callable[..., None]


def _consume_provider_stream_meta(provider: Any) -> tuple[GenerateResult | None, bool]:
    """Safely consume ``(GenerateResult, simulated_stream)`` from a BaseProvider (ignoring MagicMocks)."""
    fn = getattr(provider, "consume_last_stream_result", None)
    if callable(fn):
        try:
            res = fn()
            if (
                isinstance(res, tuple)
                and len(res) == 2
                and (res[0] is None or isinstance(res[0], GenerateResult))
                and isinstance(res[1], bool)
            ):
                return res
        except Exception:
            pass
    return None, False


class _SyncStreamFailure(Exception):
    """Wraps a provider exception raised inside ``_stream_sync_with_ttft_deadline``."""

    def __init__(
        self,
        cause: BaseException,
        *,
        tokens_emitted: bool,
        meta: GenerateResult | None = None,
        sim: bool = False,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.tokens_emitted = tokens_emitted
        self.meta = meta
        self.sim = sim


class _SyncStreamCallbackError(Exception):
    """Wraps an exception raised by the caller's ``on_token`` callback."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _stream_sync_with_ttft_deadline(
    provider: Any,
    prompt: str,
    gen_kwargs: dict[str, Any],
    on_token: Callable[[str], None],
    ttft_deadline: float,
) -> tuple[str, float | None, GenerateResult | None, bool]:
    """Execute ``provider.generate_stream_sync`` enforcing a pre-first-token TTFT deadline
    and deterministic ``gen.close()`` cleanup on error or ``on_token`` failure (QA 3.4).
    """
    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    cancel = threading.Event()
    closed_evt = threading.Event()

    def _produce() -> None:
        gen = None
        try:
            gen = provider.generate_stream_sync(prompt, **gen_kwargs)
            for tok in gen:
                if cancel.is_set():
                    break
                q.put(("tok", tok))
                if cancel.is_set():
                    break
            meta, sim = _consume_provider_stream_meta(provider)
            q.put(("done", (meta, sim)))
        except BaseException as exc:
            meta, sim = _consume_provider_stream_meta(provider)
            q.put(("err", (exc, meta, sim)))
        finally:
            if gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception:
                    pass
            closed_evt.set()

    t0 = time.monotonic()
    tokens_emitted = False
    first_ttft_s: float | None = None
    tokens: list[str] = []

    worker = threading.Thread(
        target=contextvars.copy_context().run,
        args=(_produce,),
        daemon=True,
    )
    worker.start()

    try:
        try:
            kind, payload = q.get(timeout=ttft_deadline if ttft_deadline > 0 else None)
        except queue.Empty as exc:
            cancel.set()
            raise TimeoutError(
                f"Synchronous stream exceeded TTFT deadline ({ttft_deadline:.2f}s)"
            ) from exc

        while True:
            if kind == "tok":
                if first_ttft_s is None:
                    first_ttft_s = time.monotonic() - t0
                tokens_emitted = True
                on_token(payload)
                tokens.append(payload)
                kind, payload = q.get()
            elif kind == "done":
                meta, sim = payload
                eff_ttft = (meta.ttft_s if meta is not None else None) if sim else first_ttft_s
                return "".join(tokens), eff_ttft, meta, sim
            elif kind == "err":
                exc_val, meta, sim = payload
                raise _SyncStreamFailure(
                    exc_val, tokens_emitted=tokens_emitted, meta=meta, sim=sim
                )
    except _SyncStreamFailure:
        raise
    except BaseException as caller_exc:
        cancel.set()
        closed_evt.wait(timeout=0.5)
        if tokens_emitted and not isinstance(caller_exc, TimeoutError):
            raise _SyncStreamCallbackError(caller_exc) from caller_exc
        raise
    finally:
        cancel.set()
        closed_evt.wait(timeout=0.2)


def _emit_otel_routing_attributes(
    *,
    model_id: str,
    provider: str,
    routing_mode: str,
    headroom_score: float = 1.0,
    projection_applied: bool = False,
) -> None:
    """Attach OpenTelemetry GenAI semantic attributes to the active span if present."""
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("gen_ai.request.model", model_id)
            span.set_attribute("gen_ai.system", provider)
            span.set_attribute("aurarouter.routing.mode", routing_mode)
            span.set_attribute("aurarouter.ratelimit.headroom", float(headroom_score))
            span.set_attribute("aurarouter.projection.applied", bool(projection_applied))
    except Exception:
        pass


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
# Token-bucket rate limiter (TG6) — used to throttle outbound XLM calls
# ---------------------------------------------------------------------------

@dataclass
class _TokenBucketRateLimiter:
    """Thread-safe token bucket. max_per_minute=0 means unlimited."""
    max_per_minute: int
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _tokens: float = field(init=False)
    _last_refill: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self):
        self._tokens = float(self.max_per_minute)

    def acquire(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        if self.max_per_minute <= 0:
            return True
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._last_refill = now
            # Refill at rate of max_per_minute tokens per 60 seconds
            self._tokens = min(
                float(self.max_per_minute),
                self._tokens + elapsed * self.max_per_minute / 60.0,
            )
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


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
                 triage_router=None, feedback_store=None, **kwargs):
        self._config = config
        self._provider_cache: Dict[str, BaseProvider] = {}
        self._provider_cache_lock = threading.Lock()
        self._ollama_discovery = ollama_discovery
        self._xlm_client = xlm_client
        self._triage_router = triage_router
        self._feedback_store = feedback_store
        self._usage_store = kwargs.get("usage_store")
        self._budget_manager = kwargs.get("budget_manager")
        self._privacy_auditor = kwargs.get("privacy_auditor")
        self._privacy_store = kwargs.get("privacy_store")
        self._anonymization_pipeline = kwargs.get("anonymization_pipeline")
        self._routing_advisors = kwargs.get("routing_advisors")
        self._sovereignty_gate = kwargs.get("sovereignty_gate")
        self._rag_pipeline = kwargs.get("rag_pipeline")
        # TG6: Rate limiters for outbound XLM calls (default 50/min aug, 200/min usage)
        xlm_rl = config.config.get("xlm_rate_limit_per_minute", 50)
        usage_rl = config.config.get("usage_rate_limit_per_minute", 200)
        self._xlm_aug_limiter = _TokenBucketRateLimiter(max_per_minute=int(xlm_rl))
        self._xlm_usage_limiter = _TokenBucketRateLimiter(max_per_minute=int(usage_rl))
        # TG6: Retry budget (seconds) for the provider fallback chain. 0 = unlimited.
        self._retry_budget_seconds: float = float(
            config.config.get("max_retry_budget_seconds", 30)
        )
        # TG2: Bounded thread pool for fire-and-forget background events.
        self._event_reporter = EventReporter(
            max_workers=int(config.config.get("event_reporter_max_workers", 8)),
            max_queue_depth=int(config.config.get("event_reporter_max_queue_depth", 256)),
        )
        from aurarouter.trace_logger import TraceLogger
        self._trace_logger = TraceLogger(event_reporter=self._event_reporter)
        self._trace_lock = threading.Lock()
        if self._budget_manager is not None and hasattr(self._budget_manager, "set_event_reporter"):
            self._budget_manager.set_event_reporter(self._event_reporter)
        # T5.3: Replica count for distributed rate-limit budget splitting.
        # Reads AURACORE_REPLICA_COUNT env-var first, then config, then defaults to 1.
        _rc_env = os.environ.get("AURACORE_REPLICA_COUNT")
        _rc_cfg = config.config.get("replica_count", 1)
        _rc_raw = int(_rc_env) if _rc_env is not None and _rc_env.strip().isdigit() else int(_rc_cfg)
        self._replica_count: int = max(1, min(100, _rc_raw))
        # T7.1: Circuit breaker registry for provider fault tolerance
        _cb_threshold = int(config.config.get("resilience", {}).get("failure_threshold", 5))
        _cb_timeout = float(config.config.get("resilience", {}).get("reset_timeout", 60.0))
        self._circuit_breakers = kwargs.get("circuit_breakers") or CircuitBreakerRegistry(
            failure_threshold=_cb_threshold, reset_timeout=_cb_timeout,
        )

        # v2.1: Lock-free posterior cache, multi-dimensional GCRA rate tracker,
        # two-stage composable RoutingPolicyStack, and O(N) SessionProjector.
        self._posterior_cache = kwargs.get("posterior_cache") or PosteriorSnapshotCache()
        if self._feedback_store is not None and not kwargs.get("posterior_cache"):
            self._posterior_cache.hydrate_async(self._feedback_store, self._event_reporter)
        self._rate_limit_tracker: RateLimitTracker = kwargs.get("rate_limit_tracker") or RateLimitTracker(replica_count=self._replica_count)
        self._routing_stack: RoutingPolicyStack
        self._session_projector: SessionProjector
        self._last_routing_trace: RoutingDecisionTrace | None = None
        self._refresh_v2_routing_components(reset_tracker=True)

        # Phase 2: Dynamic discovery loop client
        enable_discovery = config.get_xlm_config().get("enable_grid_discovery", False) or \
                           os.environ.get("AURACORE_ENABLE_GRID_DISCOVERY") == "true"
        if enable_discovery:
            try:
                loop = asyncio.get_running_loop()
                self._discovery_task = loop.create_task(start_grid_discovery(self))
                logger.info("Successfully scheduled grid discovery task in running loop.")
            except RuntimeError:
                logger.debug("Grid discovery task not scheduled: no running event loop in thread.")

    # ------------------------------------------------------------------
    # v2.1 Safe Config Readers & Component Wiring
    # ------------------------------------------------------------------

    def _get_rl_cfg(self) -> dict[str, Any]:
        fn = getattr(self._config, "get_rate_limits_config", None)
        if callable(fn):
            res = fn()
            if isinstance(res, dict):
                return res
        return normalize_rate_limits_config(getattr(self._config, "config", None))

    def _get_stack_cfg(self) -> dict[str, Any]:
        fn = getattr(self._config, "get_routing_stack_config", None)
        if callable(fn):
            res = fn()
            if isinstance(res, dict):
                return res
        return normalize_routing_stack_config(getattr(self._config, "config", None))

    def _get_proj_cfg(self) -> dict[str, Any]:
        fn = getattr(self._config, "get_failover_projection_config", None)
        if callable(fn):
            res = fn()
            if isinstance(res, dict):
                return res
        return normalize_failover_projection_config(getattr(self._config, "config", None))

    def _get_stream_cfg(self) -> dict[str, Any]:
        fn = getattr(self._config, "get_streaming_config", None)
        if callable(fn):
            res = fn()
            if isinstance(res, dict):
                return res
        return normalize_streaming_config(getattr(self._config, "config", None))

    @staticmethod
    def _resolve_vendor(model_id: str, model_cfg: dict | None) -> str:
        """Resolve canonical upstream vendor for rate-limit & cost buckets (QA 2.4)."""
        return resolve_canonical_vendor(model_id, model_cfg)

    def _is_dyn_routing_enabled(self) -> bool:
        fn = getattr(self._config, "is_dynamic_routing_enabled", None)
        if callable(fn):
            val = fn()
            if isinstance(val, bool):
                return val
        return False

    def _get_allowed_data_cats(self, model_id: str) -> list[str]:
        fn = getattr(self._config, "get_model_allowed_data_categories", None)
        if callable(fn):
            val = fn(model_id)
            if isinstance(val, (list, tuple, set, frozenset)):
                return list(val)
        return []

    def _refresh_v2_routing_components(self, reset_tracker: bool = False) -> None:
        rl_cfg = self._get_rl_cfg()
        warn_thresh = float(rl_cfg.get("default_warning_threshold", 0.80))
        if reset_tracker:
            self._rate_limit_tracker = RateLimitTracker(
                replica_count=self._replica_count,
                default_warning_threshold=warn_thresh,
            )

        # Seed static provider-level quotas
        providers_q = rl_cfg.get("providers")
        if isinstance(providers_q, dict):
            for prov_name, q_raw in providers_q.items():
                if isinstance(q_raw, dict):
                    self._rate_limit_tracker.configure_quota(
                        provider=str(prov_name),
                        quota=RateLimitQuota(
                            rpm=int(q_raw.get("rpm", 0) or 0),
                            itpm=int(q_raw.get("itpm", q_raw.get("tpm", 0)) or 0),
                            otpm=int(q_raw.get("otpm", 0) or 0),
                            warning_threshold=float(q_raw.get("warning_threshold", warn_thresh)),
                            header_sync_ttl_seconds=float(q_raw.get("header_sync_ttl_seconds", 60.0)),
                            replica_skew_damping=float(q_raw.get("replica_skew_damping", 0.25)),
                        ),
                    )

        # Seed model-level quotas
        raw_root = getattr(self._config, "config", None)
        models_cfg = raw_root.get("models", {}) if isinstance(raw_root, dict) else {}
        models_q = rl_cfg.get("models")
        if isinstance(models_q, dict):
            for mid, q_raw in models_q.items():
                if isinstance(q_raw, dict):
                    m_entry = models_cfg.get(mid, {}) if isinstance(models_cfg, dict) else {}
                    prov = str(q_raw.get("provider") or self._resolve_vendor(str(mid), m_entry))
                    self._rate_limit_tracker.configure_quota(
                        provider=prov,
                        model_id=str(mid),
                        quota=RateLimitQuota(
                            rpm=int(q_raw.get("rpm", 0) or 0),
                            itpm=int(q_raw.get("itpm", q_raw.get("tpm", 0)) or 0),
                            otpm=int(q_raw.get("otpm", 0) or 0),
                            warning_threshold=float(q_raw.get("warning_threshold", warn_thresh)),
                            header_sync_ttl_seconds=float(q_raw.get("header_sync_ttl_seconds", 60.0)),
                            replica_skew_damping=float(q_raw.get("replica_skew_damping", 0.25)),
                        ),
                    )

        stack_cfg = self._get_stack_cfg()
        self._routing_stack = RoutingPolicyStack(
            config=self._config,
            posterior_cache=self._posterior_cache,
            rate_tracker=self._rate_limit_tracker,
            circuit_breakers=self._circuit_breakers,
            budget_manager=self._budget_manager,
            stack_config=stack_cfg,
        )

        proj_cfg = self._get_proj_cfg()
        self._session_projector = SessionProjector(
            safety_margin_ratio=float(proj_cfg.get("safety_margin_ratio", 0.08)),
            active_turn_head_keep_ratio=float(proj_cfg.get("active_turn_head_keep_ratio", 0.35)),
            active_turn_tail_keep_ratio=float(proj_cfg.get("active_turn_tail_keep_ratio", 0.45)),
            inject_failover_header=bool(proj_cfg.get("inject_failover_header", True)),
        )

    @property
    def config(self) -> ConfigLoader:
        """Read-only access to the configuration."""
        return self._config

    @property
    def rate_limit_tracker(self) -> RateLimitTracker:
        """Read-only access to the central RateLimitTracker."""
        return self._rate_limit_tracker

    @property
    def routing_stack(self) -> RoutingPolicyStack:
        """Read-only access to the Two-Stage RoutingPolicyStack."""
        return self._routing_stack

    @property
    def session_projector(self) -> SessionProjector:
        """Read-only access to the ephemeral O(N) SessionProjector."""
        return self._session_projector

    @property
    def last_routing_trace(self) -> RoutingDecisionTrace | None:
        """Most recent RoutingDecisionTrace emitted in shadow or active mode."""
        with self._trace_lock:
            return self._last_routing_trace

    def set_feedback_store(self, feedback_store: Any) -> None:
        """Attach or replace the FeedbackStore and trigger async posterior hydration."""
        self._feedback_store = feedback_store
        if feedback_store is not None:
            self._posterior_cache.hydrate_async(feedback_store, self._event_reporter)

    def explain_route(
        self,
        role: str,
        prompt: str = "",
        complexity_score: int = 5,
        intent: str = "DIRECT",
        session_id: str | None = None,
        last_model_id: str | None = None,
        last_routed_at: float | None = None,
        estimated_input_tokens: int | None = None,
        failover_projection_enabled: bool = False,
        deterministic: bool = True,
        max_vram_gb: float | None = None,
        quant: str = "q4_k_m",
    ) -> RoutingDecisionTrace:
        """Dry-run Stage 1 eligibility gates + Stage 2 utility scoring for explainability."""
        chain = self._config.get_role_chain(role)
        chain = self.consult_routing_advisors(role, chain, intent=intent)
        eff_cats = self.get_effective_categories(prompt) if prompt else []
        sov_open = True
        if prompt and self._sovereignty_gate is not None:
            sov_res = self._sovereignty_gate.evaluate(prompt, eff_cats)
            sov_open = "OPEN" in str(getattr(sov_res, "verdict", "OPEN")).upper()

        est_in = (
            estimated_input_tokens
            if estimated_input_tokens is not None
            else (max(1, len(prompt) // 4) if prompt else 0)
        )
        ctx = CandidateEvalContext(
            role=role,
            intent=intent,
            complexity_score=complexity_score,
            estimated_input_tokens=est_in,
            effective_data_categories=frozenset(eff_cats),
            sovereignty_verdict_open=sov_open,
            session_id=session_id,
            last_model_id=last_model_id,
            last_routed_at=last_routed_at,
            failover_projection_enabled=failover_projection_enabled,
            deterministic=deterministic,
            max_vram_gb=max_vram_gb,
            quant=quant,
        )
        return self._routing_stack.evaluate_and_rank(chain, ctx)

    def _apply_routing_stack(
        self,
        chain: list[str],
        eval_ctx: CandidateEvalContext,
    ) -> tuple[list[str], RoutingDecisionTrace | None]:
        """Run RoutingPolicyStack when mode is 'shadow' or 'active'."""
        mode = self._routing_stack.mode
        if mode not in ("shadow", "active"):
            return chain, None

        trace = self._routing_stack.evaluate_and_rank(chain, eval_ctx)
        with self._trace_lock:
            self._last_routing_trace = trace

        if eval_ctx.session_id:
            stage_name = "routing_stack_active" if mode == "active" else "routing_stack_shadow"
            self._trace_logger.log_stage(eval_ctx.session_id, stage_name, trace.to_dict())

        if mode == "active":
            return trace.ranked_chain, trace
        # Shadow mode: log counterfactual trace, preserve legacy chain execution order
        return chain, trace

    def get_effective_categories(self, prompt: str) -> list[str]:
        """Determine the effective sensitive data categories remaining in a prompt
        after potential anonymization."""
        if not self._privacy_auditor:
            return []

        head_event = self._privacy_auditor.audit(prompt, "heads", "router")
        if not head_event or not head_event.matches:
            return []

        if not self._anonymization_pipeline:
            return list(set(m.category for m in head_event.matches))

        effective_prompt, _ = self._anonymization_pipeline.anonymize(prompt, head_event.matches)
        tail_event = self._privacy_auditor.audit(effective_prompt, "tails", "router")
        if tail_event and tail_event.matches:
            return list(set(m.category for m in tail_event.matches))

        return []

    @property
    def circuit_breakers(self) -> CircuitBreakerRegistry:
        """Read-only access to the per-provider circuit breaker registry."""
        return self._circuit_breakers

    @property
    def provider_cache(self) -> "dict[str, BaseProvider]":
        """Snapshot of the currently cached provider instances (shallow copy)."""
        return dict(self._provider_cache)

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
            if not is_cloud_tier(hosting_tier, provider, endpoint=model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")):
                local.append(model_id)
        return local

    def get_context_limit(self, model_id: str) -> int:
        """Return the context limit for a specific model."""
        model_cfg = self._config.get_model_config(model_id)
        if not model_cfg:
            return 0
        return int(model_cfg.get("context_limit") or model_cfg.get("context_window", 0))

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
            except Exception as ex:
                logger.error(
                    "catalog_advisor_registration_failed",
                    artifact_id=artifact_id,
                    endpoint=endpoint,
                    error=str(ex),
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
        with self._provider_cache_lock:
            self._provider_cache.clear()
        self._refresh_v2_routing_components(reset_tracker=False)

    # ------------------------------------------------------------------
    # Provider resolution
    # ------------------------------------------------------------------

    def _get_provider(self, model_id: str) -> BaseProvider | None:
        """Get or create a provider for the given model_id."""
        model_cfg = self._config.get_model_config(model_id)
        if not model_cfg:
            return None
        with self._provider_cache_lock:
            if model_id in self._provider_cache:
                provider = self._provider_cache[model_id]
            else:
                provider = None
        if provider is None:
            provider_name = model_cfg.get("provider")
            provider = get_provider(provider_name, model_cfg)
            with self._provider_cache_lock:
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
            except Exception as ex:
                logger.error(
                    "routing_advisor_chain_reorder_failed advisor=%s role=%s error=%s",
                    name,
                    role,
                    str(ex),
                    exc_info=True,
                )
        return chain

    # ------------------------------------------------------------------
    # XLM integration hooks
    # ------------------------------------------------------------------

    def _augment_prompt(self, prompt: str, role: str) -> tuple[str, RoutingContext]:
        """Call AuraXLM's auraxlm.query tool to prepend RAG context. Fail-safe."""
        rc = RoutingContext()
        if not self._config.is_xlm_augmentation_enabled():
            return prompt, rc
        endpoint = self._config.get_xlm_endpoint()
        if not endpoint:
            return prompt, rc
        # TG6: Rate limit outbound XLM augmentation calls
        if not self._xlm_aug_limiter.acquire():
            logger.warning("XLM augmentation rate limit reached — falling back to non-augmented prompt")
            return prompt, rc
        # T5.3: Inject replica count header so AuraXLM can split its rate-limit budget.
        xlm_headers = {"X-AuraCore-Replica-Count": str(self._replica_count)}
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
                    return prompt, rc
            result = client.call_tool("auraxlm.query", headers=xlm_headers, prompt=prompt, role=role)
            # T5.4: Handle 2429 — back-off with jitter then retry once.
            if isinstance(result, dict) and result.get("error_code") == 2429:
                retry_after = int(
                    getattr(client, "last_response_headers", {}).get("Retry-After", 0)
                    or result.get("retry_after_seconds", 0)
                )
                if retry_after > 0:
                    jitter = random.uniform(0.1, 1.5)
                    logger.info(
                        "AuraXLM rate-limited (2429); sleeping %.1fs before retry.",
                        retry_after + jitter,
                    )
                    time.sleep(retry_after + jitter)
                    result = client.call_tool("auraxlm.query", headers=xlm_headers, prompt=prompt, role=role)
                else:
                    # No Retry-After — fall back immediately.
                    return prompt, rc
            if isinstance(result, dict):
                # TG11: Adaptive throttle — if XLM reports < 10% capacity remaining,
                # drain the local aug limiter to prevent hitting server limit
                remaining = result.get("x_ratelimit_remaining")
                limit = result.get("x_ratelimit_limit")
                if (remaining is not None and limit is not None
                        and limit > 0 and remaining / limit < 0.10):
                    logger.warning(
                        "AuraXLM rate limit low (remaining=%s/%s) — "
                        "applying adaptive back-pressure on augmentation",
                        remaining, limit,
                    )
                    self._xlm_aug_limiter._tokens = 0.0

                rc.retrieval_used = True
                if result.get("sources"):
                    rc.sources = result["sources"]
                if result.get("metadata"):
                    m = result["metadata"]
                    rc.author_id = m.get("author_id")
                    rc.project_id = m.get("project_id")

                if result.get("augmented_prompt"):
                    return result["augmented_prompt"], rc
            if isinstance(result, str) and result.strip():
                return result, rc
        except Exception as exc:
            logger.warning(
                "AuraXLM prompt augmentation is enabled but the client failed to "
                "connect to %s: %s — falling back to non-augmented prompt",
                endpoint,
                exc,
            )
            logger.debug("XLM augmentation error details", exc_info=True)
        return prompt, rc

    def _record_feedback(self, role: str, model_id: str, success: bool,
                         elapsed: float, complexity_score: float | None = None,
                         input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record routing outcome to PosteriorSnapshotCache (in-memory COW) & FeedbackStore."""
        score = complexity_score if complexity_score is not None else 5.0
        try:
            self._posterior_cache.record_outcome(
                model_id=model_id,
                role_or_intent=role,
                complexity=score,
                success=success,
            )
        except Exception:
            pass

        if self._feedback_store is None:
            return
        store = self._feedback_store

        def _write():
            try:
                store.record(
                    role=role, complexity=score, model_id=model_id,
                    success=success, latency=elapsed,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )
            except Exception as ex:
                logger.debug("feedback_store_write_failed model_id=%s role=%s error=%s", model_id, role, str(ex))
        self._event_reporter.submit(_write)

    def _report_usage(self, role: str, model_id: str, success: bool,
                      elapsed: float, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Fire-and-forget usage event to AuraXLM. Never blocks or raises."""
        if not self._config.is_xlm_usage_reporting_enabled():
            return
        # TG15: Respect global telemetry opt-out
        from aurarouter.telemetry_config import is_external_telemetry_enabled
        if not is_external_telemetry_enabled():
            return
        endpoint = self._config.get_xlm_endpoint()
        if not endpoint:
            return
        # TG6: Rate limit outbound XLM usage reporting calls
        if not self._xlm_usage_limiter.acquire():
            logger.debug("XLM usage reporting rate limit reached — skipping this report")
            return

        xlm_client = self._xlm_client
        replica_count = self._replica_count

        def _send():
            try:
                xlm_headers = {"X-AuraCore-Replica-Count": str(replica_count)}
                if xlm_client is not None:
                    client = xlm_client
                else:
                    from aurarouter.mcp_client.client import GridMcpClient
                    client = GridMcpClient(base_url=endpoint, name="xlm-usage", timeout=5.0)
                    if not client.connect():
                        return
                client.call_tool("auraxlm.record_usage",
                    headers=xlm_headers,
                    model_id=model_id, role=role, success=success,
                    elapsed_seconds=elapsed,
                    input_tokens=input_tokens, output_tokens=output_tokens)
            except Exception as ex:
                logger.debug("xlm_usage_report_failed model_id=%s role=%s error=%s", model_id, role, str(ex))
        self._event_reporter.submit(_send)

    # ------------------------------------------------------------------
    # Usage store recording
    # ------------------------------------------------------------------

    def _record_usage(self, role: str, model_id: str, provider: str,
                      success: bool, elapsed: float,
                      input_tokens: int = 0, output_tokens: int = 0,
                      intent: str = "",
                      routing_context=None) -> None:  # RoutingContext | None
        """Record a usage event to the UsageStore."""
        if self._usage_store is None:
            return
        from aurarouter.savings.pricing import is_cloud_tier
        model_cfg = self._config.get_model_config(model_id)
        hosting_tier = model_cfg.get("hosting_tier") if model_cfg else None
        is_cloud = is_cloud_tier(hosting_tier, provider, endpoint=(model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")) if isinstance(model_cfg, dict) else None)
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
        self._usage_store.record(record, routing_context=routing_context)

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

    def _check_model_budget(self, provider_name: str, hosting_tier: str | None, model_cfg: dict):
        endpoint = model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint") if isinstance(model_cfg, dict) else None
        try:
            return self._budget_manager.check_budget(provider_name, hosting_tier=hosting_tier, endpoint=endpoint)
        except TypeError:
            return self._budget_manager.check_budget(provider_name)

    def _handle_attempt_outcome(
        self,
        *,
        role: str,
        model_id: str,
        provider_name: str,
        cb: Any,
        lease: Any,
        elapsed: float,
        est_input_tokens: int,
        complexity_val: int = 5,
        intent: str = "",
        session_id: str | None = None,
        routing_context: Any = None,
        on_model_tried: Optional[ModelTriedCallback] = None,
        success: bool = False,
        actual_input_tokens: int = 0,
        actual_output_tokens: int = 0,
        snapshot: Any = None,
        ttft_s: float | None = None,
        exc: BaseException | None = None,
    ) -> tuple[str, str]:
        """Unified post-attempt settlement across execute, execute_session, execute_stream, and RT1 probe (QA 2.2).

        Updates RateLimitTracker, CircuitBreaker (preserving 429 & context-overflow fault isolation),
        TraceLogger, on_model_tried callback, AuraXLM usage reporter, FeedbackStore/PosteriorCache,
        and UsageStore symmetrically across every execution path.
        """
        valid_sid = session_id if isinstance(session_id, str) and session_id else None

        if success:
            cb_prev, _ = cb.peek_state()
            self._rate_limit_tracker.settle(
                lease,
                actual_input_tokens=actual_input_tokens or est_input_tokens,
                actual_output_tokens=actual_output_tokens or 0,
                snapshot=snapshot,
                ttft_s=ttft_s,
            )
            cb.record_success()
            if cb_prev != "closed" and valid_sid:
                self._trace_logger.log_stage(
                    valid_sid,
                    "circuit_breaker",
                    {
                        "model_id": model_id,
                        "provider": provider_name,
                        "from_state": cb_prev,
                        "to_state": "closed",
                        "event": "recovered",
                    },
                )
            self._fire_callback(
                on_model_tried,
                role,
                model_id,
                True,
                elapsed,
                actual_input_tokens,
                actual_output_tokens,
            )
            self._report_usage(
                role,
                model_id,
                True,
                elapsed,
                actual_input_tokens,
                actual_output_tokens,
            )
            self._record_feedback(
                role,
                model_id,
                True,
                elapsed,
                complexity_score=complexity_val,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
            )
            self._record_usage(
                role,
                model_id,
                provider_name,
                True,
                elapsed,
                actual_input_tokens,
                actual_output_tokens,
                intent=str(intent),
                routing_context=routing_context,
            )
            return "ok", ""

        if isinstance(exc, RateLimitExceededError):
            self._rate_limit_tracker.settle_rate_limited(lease, snapshot=exc.snapshot)
            err_msg = f"{model_id} rate-limited (429): {exc}"
            logger.warning(err_msg)
            if valid_sid:
                self._trace_logger.log_stage(
                    valid_sid,
                    "rate_limit_exceeded",
                    {
                        "model_id": model_id,
                        "provider": provider_name,
                        "error": str(exc),
                        "retry_after_s": getattr(exc.snapshot, "retry_after_s", None)
                        if exc.snapshot is not None
                        else None,
                    },
                )
            self._fire_callback(on_model_tried, role, model_id, False, elapsed)
            self._report_usage(role, model_id, False, elapsed)
            self._record_usage(
                role,
                model_id,
                provider_name,
                False,
                elapsed,
                intent=str(intent),
                routing_context=routing_context,
            )
            return "429_rate_limit", err_msg

        if isinstance(exc, ContextLengthExceededError):
            self._rate_limit_tracker.abort(lease)
            err_msg = f"{model_id} context length exceeded: {exc}"
            logger.warning(err_msg)
            if valid_sid:
                self._trace_logger.log_stage(
                    valid_sid,
                    "context_length_exceeded",
                    {
                        "model_id": model_id,
                        "provider": provider_name,
                        "error": str(exc),
                        "max_tokens": exc.max_tokens,
                    },
                )
            self._fire_callback(on_model_tried, role, model_id, False, elapsed)
            self._report_usage(role, model_id, False, elapsed)
            self._record_usage(
                role,
                model_id,
                provider_name,
                False,
                elapsed,
                intent=str(intent),
                routing_context=routing_context,
            )
            return "context_length_exceeded", err_msg

        cb_prev, _ = cb.peek_state()
        self._rate_limit_tracker.abort(lease)
        cb.record_failure()
        cb_after, _ = cb.peek_state()
        err_msg = f"{model_id} failed: {exc}"
        logger.warning(err_msg)
        if valid_sid and (cb_after != cb_prev or cb_after == "open"):
            self._trace_logger.log_stage(
                valid_sid,
                "circuit_breaker",
                {
                    "model_id": model_id,
                    "provider": provider_name,
                    "from_state": cb_prev,
                    "to_state": cb_after,
                    "error": str(exc),
                },
            )
        self._fire_callback(on_model_tried, role, model_id, False, elapsed)
        self._report_usage(role, model_id, False, elapsed)
        self._record_feedback(role, model_id, False, elapsed, complexity_score=complexity_val)
        self._record_usage(
            role,
            model_id,
            provider_name,
            False,
            elapsed,
            intent=str(intent),
            routing_context=routing_context,
        )
        return "provider_error", err_msg

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
            if is_cloud_tier(hosting_tier, provider_name, endpoint=model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")):
                status = self._check_model_budget(provider_name, hosting_tier, model_cfg)
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
            routing_context = getattr(result, "routing_context", None)
            self._record_usage(role, model_id, provider_name, True, elapsed,
                               input_tokens, output_tokens, intent=intent,
                               routing_context=routing_context)
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
        routing_context=None,  # RoutingContext | None — TG4, avoided circular import
        return_tokens: bool = False,
    ) -> Optional[GenerateResult]:
        """Execute a prompt through the role's model chain.

        Returns a GenerateResult on success, or None if all models fail.
        If no models are defined for the role, returns a GenerateResult
        with an ERROR message.
        """
        opts = options or {}
        # Intent-aware schema enforcement (TG6)
        intent = opts.get("intent", "chat")
        actionable = intent in ("edit_code", "generate_code")
        if actionable:
            json_mode = True
            file_constraints = opts.get("file_constraints")
            response_schema = compile_modifications_schema(file_constraints)
        else:
            response_schema = None

        chain = chain_override if chain_override is not None else self._config.get_role_chain(role)
        if not chain:
            return GenerateResult(text=f"ERROR: No models defined for role '{role}' in YAML.")

        # TG1: Legacy Dynamic Routing Optimizer (superseded when RoutingPolicyStack is active)
        dyn_enabled = self._is_dyn_routing_enabled() and self._routing_stack.mode != "active"
        strategy = "static"
        if dyn_enabled:
            from aurarouter.metrics import DefaultCostOptimizer, DynamicPricingProvider, LowestLatencyOptimizer
            strategy = self._config.get_optimizer_strategy()
            metrics_provider = DynamicPricingProvider(self._config)

            if strategy == "cheapest_valid":
                optimizer = DefaultCostOptimizer()
            elif strategy == "lowest_latency":
                optimizer = LowestLatencyOptimizer()
            else:
                optimizer = DefaultCostOptimizer()

            chain = optimizer.rank_models(chain, metrics_provider)
        elif self._routing_stack.mode == "active":
            strategy = "routing_stack_v2"

        # Trace intent_classification
        session_id = opts.get("session_id", getattr(routing_context, "session_id", None))
        if session_id:
            self._trace_logger.log_stage(session_id, "intent_classification", {
                "role": role,
                "intent": intent,
                "actionable": actionable,
                "chain": chain,
                "strategy": strategy,
            })

        # Consult routing advisors for potential chain reordering
        chain = self.consult_routing_advisors(role, chain, intent=intent)

        bypass_anonymization = opts.get("bypass_anonymization", False)

        # T4: Pluggable Sovereign Data Anonymization Pipeline (Heads)
        effective_categories: list[str] = []
        effective_prompt = prompt
        mapping: dict[str, str] = {}
        stream_supported = True
        head_event = None

        if not bypass_anonymization and self._privacy_auditor is not None:
            head_event = self._privacy_auditor.audit(prompt, "heads", "router")
            if head_event and head_event.matches:
                if self._anonymization_pipeline is not None:
                    effective_prompt, mapping = self._anonymization_pipeline.anonymize(prompt, head_event.matches)
                    tail_event = self._privacy_auditor.audit(effective_prompt, "tails", "router")
                    if tail_event and tail_event.matches:
                        effective_categories = list(set(m.category for m in tail_event.matches))

                    if not self._privacy_auditor.supports_streaming or not self._anonymization_pipeline.supports_streaming:
                        stream_supported = False
                else:
                    effective_categories = list(set(m.category for m in head_event.matches))

        if not effective_categories and opts.get("data_categories"):
            effective_categories = list(opts.get("data_categories"))

        # Update RoutingContext with ephemeral MCP Gatekeeping
        effective_prompt, rc = self._augment_prompt(effective_prompt, role)
        if routing_context is None:
            routing_context = rc
        elif hasattr(routing_context, 'retrieval_used'):
            routing_context.retrieval_used = routing_context.retrieval_used or rc.retrieval_used
            if rc.sources:
                routing_context.sources.extend(rc.sources)
            routing_context.author_id = routing_context.author_id or rc.author_id
            routing_context.project_id = routing_context.project_id or rc.project_id

        # Calculate allowed MCPs based on effective_categories
        if hasattr(routing_context, 'allowed_mcps'):
            if not bypass_anonymization and effective_categories:
                allowed_mcps = []
                if self._routing_advisors:
                    for mcp_name, client in self._routing_advisors.get_all_clients().items():
                        mcp_cfg = self._config.get_mcp_config(mcp_name)
                        if mcp_cfg:
                            allowed_cats = set(mcp_cfg.get("allowed_data_categories", []))
                            if set(effective_categories).issubset(allowed_cats):
                                allowed_mcps.append(mcp_name)
                    routing_context.allowed_mcps = allowed_mcps
                else:
                    routing_context.allowed_mcps = []
            else:
                routing_context.allowed_mcps = None

        # Sovereignty gate: evaluate effective_prompt and filter chain
        sovereignty_result = None
        sovereignty_verdict_open = True
        if not bypass_anonymization and self._sovereignty_gate is not None:
            sovereignty_result = self._sovereignty_gate.evaluate(effective_prompt, effective_categories)
            sovereignty_verdict_open = "OPEN" in str(getattr(sovereignty_result, "verdict", "OPEN")).upper()
            try:
                chain = self._sovereignty_gate.enforce(chain, self._config, sovereignty_result)
            except Exception as ex:
                return GenerateResult(
                    text=f"ERROR: Sovereignty gate filtered all models. {ex}"
                )
            if not chain:
                return GenerateResult(
                    text="ERROR: Sovereignty gate filtered all models. "
                    "No models available for sensitive content."
                )

        # RAG enrichment: inject retrieved context into prompt
        if self._rag_pipeline is not None and self._rag_pipeline.is_enabled():
            try:
                loop = asyncio.get_event_loop()
                # Use ephemeral context to filter tools during enrichment
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        enriched = pool.submit(
                            asyncio.run, self._rag_pipeline.enrich(effective_prompt, routing_context=routing_context)
                        ).result(timeout=6.0)
                else:
                    enriched = loop.run_until_complete(self._rag_pipeline.enrich(effective_prompt, routing_context=routing_context))
                effective_prompt = self._rag_pipeline.build_enriched_prompt(effective_prompt, enriched)
            except Exception as exc:
                logger.warning("RAG enrichment failed in execute: %s", exc)

        # Streaming fallback
        stream_fallback_reason = None
        if on_token and not stream_supported:
            logger.info("Falling back to blocking execution because anonymization pipeline does not support streaming.")
            stream_fallback_reason = "Anonymization pipeline does not support streaming"
            on_token = None

        # v2.1: Two-Stage RoutingPolicyStack evaluation (shadow or active)
        raw_cplx = opts.get("complexity_score", getattr(routing_context, "complexity_score", 5))
        complexity_val = int(raw_cplx) if isinstance(raw_cplx, (int, float)) else 5
        est_input_tokens = max(1, len(effective_prompt) // 4)
        est_output_tokens = int(opts.get("max_tokens", 512))
        req_caps = frozenset(opts.get("required_capabilities", ()))

        eval_ctx = CandidateEvalContext(
            role=role,
            intent=str(intent),
            complexity_score=complexity_val,
            estimated_input_tokens=est_input_tokens,
            estimated_output_tokens=est_output_tokens,
            required_capabilities=req_caps,
            effective_data_categories=frozenset(effective_categories),
            sovereignty_verdict_open=sovereignty_verdict_open,
            session_id=session_id if isinstance(session_id, str) else None,
            last_model_id=opts.get("last_model_id"),
            last_routed_at=opts.get("last_routed_at"),
            bypass_sovereignty=bool(bypass_anonymization),
            failover_projection_enabled=False,
            deterministic=bool(opts.get("deterministic", False)),
            deterministic_seed=opts.get("deterministic_seed"),
            max_vram_gb=float(opts["max_vram_gb"]) if opts.get("max_vram_gb") is not None else None,
            quant=str(opts.get("quant", "q4_k_m")),
        )
        chain, stack_trace = self._apply_routing_stack(chain, eval_ctx)
        if stack_trace is not None and not chain and stack_trace.pruned:
            if all("BudgetCapFilter" in r for r in stack_trace.pruned.values()):
                for pruned_mid in stack_trace.pruned:
                    self._fire_callback(on_model_tried, role, pruned_mid, False, 0.0)
                return GenerateResult(
                    text="BUDGET_EXCEEDED: All cloud models blocked by budget limits. "
                         "Configure local models as fallback."
                )

        errors: list[str] = []
        budget_skipped: list[str] = []
        # TG6: Retry budget tracking
        _retry_start = time.monotonic()
        _models_tried = 0
        _total_models = len(chain)
        # RT1: Track models skipped solely due to open circuit breakers
        _circuit_only_skips: list[str] = []
        _any_generate_attempted = False

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            # TG6: Enforce retry budget — abort chain if time budget exceeded
            if self._retry_budget_seconds > 0 and _models_tried > 0:
                _elapsed_so_far = time.monotonic() - _retry_start
                if _elapsed_so_far >= self._retry_budget_seconds:
                    logger.warning(
                        "Retry budget exhausted after %.1fs, %d/%d providers attempted",
                        _elapsed_so_far, _models_tried, _total_models,
                    )
                    break
            _models_tried += 1

            provider_name = model_cfg.get("provider", "")
            hosting_tier = model_cfg.get("hosting_tier")

            # T7.1: Circuit breaker — skip providers with open circuits
            cb = self._circuit_breakers.get_or_create(model_id)
            if not cb.is_available():
                logger.warning("[%s] Skipping %s — circuit breaker open", role.upper(), model_id)
                errors.append(f"{model_id}: circuit breaker open")
                _circuit_only_skips.append(model_id)
                self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                continue

            logger.info(f"[{role.upper()}] Routing to: {model_id} ({provider_name})")

            # Budget check for cloud models
            if self._budget_manager is not None:
                from aurarouter.savings.pricing import is_cloud_tier
                if is_cloud_tier(hosting_tier, provider_name, endpoint=model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")):
                    status = self._check_model_budget(provider_name, hosting_tier, model_cfg)
                    if not status.allowed:
                        budget_skipped.append(model_id)
                        self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                        continue

            # Privacy audit for cloud-bound prompts (legacy recording)
            if head_event and head_event.matches:
                if self._privacy_store is not None:
                    head_event.model_id = model_id
                    head_event.provider = provider_name
                    head_event.effective_categories = effective_categories
                    self._privacy_store.record(head_event)

            # Model Gatekeeping based on Data Categories
            model_categories = self._get_allowed_data_cats(model_id)
            if effective_categories and not all(cat in model_categories for cat in effective_categories):
                logger.warning("[%s] Skipping %s — unauthorized for categories %s", role.upper(), model_id, effective_categories)
                errors.append(f"{model_id}: unauthorized for {effective_categories}")
                self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                continue

            provider = self._get_provider(model_id)
            if provider is None:
                continue

            _any_generate_attempted = True
            prov_key = self._resolve_vendor(model_id, model_cfg)
            cb_prev_state, _ = cb.peek_state()
            lease = self._rate_limit_tracker.reserve(
                model_id,
                prov_key,
                est_input_tokens=est_input_tokens,
                est_output_tokens=est_output_tokens,
            )
            util = self._rate_limit_tracker.peek_utilization(
                model_id, prov_key, fallback_provider=provider_name
            )
            _emit_otel_routing_attributes(
                model_id=model_id,
                provider=prov_key,
                routing_mode=self._routing_stack.mode,
                headroom_score=util.headroom_score,
                projection_applied=False,
            )

            start = time.monotonic()
            try:
                gen_kwargs: dict = {"json_mode": json_mode}
                if response_schema is not None:
                    gen_kwargs["response_schema"] = response_schema
                if return_tokens:
                    gen_kwargs["return_tokens"] = True

                if on_token:
                    stream_cfg = self._get_stream_cfg()
                    if util.state == RateState.NEARING_SATURATION or cb_prev_state == "half_open":
                        ttft_deadline = float(stream_cfg.get("saturated_ttft_timeout_seconds", 5.0))
                    else:
                        ttft_deadline = float(stream_cfg.get("ttft_timeout_seconds", 15.0))
                    text, sync_ttft_s, stream_meta, _ = _stream_sync_with_ttft_deadline(
                        provider,
                        effective_prompt,
                        gen_kwargs,
                        on_token,
                        ttft_deadline,
                    )
                    result = GenerateResult(
                        text=text,
                        input_tokens=(
                            stream_meta.input_tokens
                            if stream_meta is not None and stream_meta.input_tokens > 0
                            else est_input_tokens
                        ),
                        output_tokens=(
                            stream_meta.output_tokens
                            if stream_meta is not None and stream_meta.output_tokens > 0
                            else (max(1, len(text) // 4) if text else 0)
                        ),
                        model_id=stream_meta.model_id if stream_meta is not None else "",
                        context_limit=stream_meta.context_limit if stream_meta is not None else 0,
                        rate_limit=stream_meta.rate_limit if stream_meta is not None else None,
                        ttft_s=sync_ttft_s,
                    )
                else:
                    result = provider.generate_with_usage(effective_prompt, **gen_kwargs)

                # Deanonymize
                if mapping and self._anonymization_pipeline:
                    result.text = self._anonymization_pipeline.deanonymize(result.text, mapping)
                    if getattr(result, "metadata", None) is None:
                        result.metadata = {}
                    result.metadata["anonymized"] = True

                if stream_fallback_reason:
                    if getattr(result, "metadata", None) is None:
                        result.metadata = {}
                    result.metadata["stream_fallback_reason"] = stream_fallback_reason

                elapsed = time.monotonic() - start

                if result and result.text and result.text.strip():
                    result.model_id = result.model_id or model_id
                    result.provider = result.provider or provider_name
                    result.routing_context = routing_context  # TG4
                    logger.info(f"[{role.upper()}] Success from {model_id}.")
                    self._handle_attempt_outcome(
                        role=role,
                        model_id=model_id,
                        provider_name=provider_name,
                        cb=cb,
                        lease=lease,
                        elapsed=elapsed,
                        est_input_tokens=est_input_tokens,
                        complexity_val=complexity_val,
                        intent=str(intent),
                        session_id=session_id if isinstance(session_id, str) else None,
                        routing_context=routing_context,
                        on_model_tried=on_model_tried,
                        success=True,
                        actual_input_tokens=result.input_tokens,
                        actual_output_tokens=result.output_tokens,
                        snapshot=getattr(result, "rate_limit", None),
                        ttft_s=getattr(result, "ttft_s", None),
                    )
                    return result
                else:
                    raise ValueError("Response was empty or invalid.")

            except _SyncStreamCallbackError as cb_err:
                elapsed = time.monotonic() - start
                self._rate_limit_tracker.abort(lease)
                raise cb_err.cause from None
            except _SyncStreamFailure as sf_err:
                elapsed = time.monotonic() - start
                _, err_msg = self._handle_attempt_outcome(
                    role=role,
                    model_id=model_id,
                    provider_name=provider_name,
                    cb=cb,
                    lease=lease,
                    elapsed=elapsed,
                    est_input_tokens=est_input_tokens,
                    complexity_val=complexity_val,
                    intent=str(intent),
                    session_id=session_id if isinstance(session_id, str) else None,
                    routing_context=routing_context,
                    on_model_tried=on_model_tried,
                    success=False,
                    exc=sf_err.cause,
                )
                errors.append(err_msg)
                if sf_err.tokens_emitted:
                    # Never concatenate a fallback model's response onto already-emitted partial stream tokens!
                    raise sf_err.cause from None
                continue
            except Exception as e:
                elapsed = time.monotonic() - start
                _, err_msg = self._handle_attempt_outcome(
                    role=role,
                    model_id=model_id,
                    provider_name=provider_name,
                    cb=cb,
                    lease=lease,
                    elapsed=elapsed,
                    est_input_tokens=est_input_tokens,
                    complexity_val=complexity_val,
                    intent=str(intent),
                    session_id=session_id if isinstance(session_id, str) else None,
                    routing_context=routing_context,
                    on_model_tried=on_model_tried,
                    success=False,
                    exc=e,
                )
                errors.append(err_msg)
                continue
            finally:
                if not lease.settled:
                    self._rate_limit_tracker.abort(lease)

        # RT1: Last-resort probe when all skips were circuit-breaker-only
        if _circuit_only_skips and not _any_generate_attempted:
            # Find the candidate with the most seconds elapsed since last failure —
            # it is nearest to its reset_timeout and most likely to half_open.
            best_model_id = max(
                _circuit_only_skips,
                key=lambda mid: self._circuit_breakers.get_or_create(mid).seconds_since_last_failure() or 0.0,
            )
            probe_cb = self._circuit_breakers.get_or_create(best_model_id)
            if probe_cb.is_available():  # triggers open → half_open if reset_timeout elapsed
                logger.warning(
                    "[%s] All circuits open — probing least-recently-failed: %s",
                    role.upper(), best_model_id,
                )
                probe_model_cfg = self._config.get_model_config(best_model_id)
                if probe_model_cfg:
                    probe_prov_name = probe_model_cfg.get("provider", "")
                    probe_tier = probe_model_cfg.get("hosting_tier")
                    probe_allowed = True
                    if self._budget_manager is not None:
                        from aurarouter.savings.pricing import is_cloud_tier
                        if is_cloud_tier(
                            probe_tier,
                            probe_prov_name,
                            endpoint=probe_model_cfg.get("endpoint") or probe_model_cfg.get("mcp_endpoint"),
                        ):
                            b_status = self._check_model_budget(probe_prov_name, probe_tier, probe_model_cfg)
                            if not b_status.allowed:
                                probe_allowed = False
                                budget_skipped.append(best_model_id)
                    probe_cats = self._get_allowed_data_cats(best_model_id)
                    if effective_categories and not all(cat in probe_cats for cat in effective_categories):
                        probe_allowed = False

                    probe_provider = self._get_provider(best_model_id) if probe_allowed else None
                    if probe_provider is not None:
                        gen_kwargs = {"json_mode": json_mode}
                        if response_schema is not None:
                            gen_kwargs["response_schema"] = response_schema
                        probe_vendor = self._resolve_vendor(best_model_id, probe_model_cfg)
                        probe_lease = self._rate_limit_tracker.reserve(
                            best_model_id,
                            probe_vendor,
                            est_input_tokens=est_input_tokens,
                            est_output_tokens=est_output_tokens,
                        )
                        probe_start = time.monotonic()
                        try:
                            probe_result = probe_provider.generate_with_usage(effective_prompt, **gen_kwargs)
                            probe_elapsed = time.monotonic() - probe_start

                            # Deanonymize
                            if mapping and self._anonymization_pipeline and probe_result and probe_result.text:
                                probe_result.text = self._anonymization_pipeline.deanonymize(probe_result.text, mapping)
                                if getattr(probe_result, "metadata", None) is None:
                                    probe_result.metadata = {}
                                probe_result.metadata["anonymized"] = True

                            if probe_result and probe_result.text and probe_result.text.strip():
                                probe_result.model_id = probe_result.model_id or best_model_id
                                probe_result.provider = probe_result.provider or probe_prov_name
                                probe_result.routing_context = routing_context
                                logger.info("[%s] Last-resort probe succeeded: %s", role.upper(), best_model_id)
                                self._handle_attempt_outcome(
                                    role=role,
                                    model_id=best_model_id,
                                    provider_name=probe_prov_name,
                                    cb=probe_cb,
                                    lease=probe_lease,
                                    elapsed=probe_elapsed,
                                    est_input_tokens=est_input_tokens,
                                    complexity_val=complexity_val,
                                    intent=str(intent),
                                    session_id=session_id if isinstance(session_id, str) else None,
                                    routing_context=routing_context,
                                    on_model_tried=on_model_tried,
                                    success=True,
                                    actual_input_tokens=probe_result.input_tokens,
                                    actual_output_tokens=probe_result.output_tokens,
                                    snapshot=getattr(probe_result, "rate_limit", None),
                                    ttft_s=getattr(probe_result, "ttft_s", None),
                                )
                                return probe_result
                            else:
                                raise ValueError("Probe response was empty.")
                        except Exception as probe_exc:
                            probe_elapsed = time.monotonic() - probe_start
                            _, p_err = self._handle_attempt_outcome(
                                role=role,
                                model_id=best_model_id,
                                provider_name=probe_prov_name,
                                cb=probe_cb,
                                lease=probe_lease,
                                elapsed=probe_elapsed,
                                est_input_tokens=est_input_tokens,
                                complexity_val=complexity_val,
                                intent=str(intent),
                                session_id=session_id if isinstance(session_id, str) else None,
                                routing_context=routing_context,
                                on_model_tried=on_model_tried,
                                success=False,
                                exc=probe_exc,
                            )
                            errors.append(p_err)
                        finally:
                            if not probe_lease.settled:
                                self._rate_limit_tracker.abort(probe_lease)

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
        return_tokens: bool = False,
        options: dict | None = None,
    ) -> GenerateResult:
        """Execute a session-aware request through the role's model chain.

        Uses provider.generate_with_history() for multi-turn support,
        enforces SovereigntyGate / CircuitBreaker / BudgetManager / RateLimitTracker,
        and ephemerally projects oversized histories via SessionProjector on failover
        without mutating caller inputs.

        Returns a GenerateResult (never None -- errors return error text).
        """
        from aurarouter.sessions.gisting import extract_gist

        opts = options or {}
        chain = self._config.get_role_chain(role)
        if not chain:
            return GenerateResult(
                text=f"ERROR: No models defined for role '{role}' in YAML."
            )

        intent = opts.get("intent", "chat")
        chain = self.consult_routing_advisors(role, chain, intent=intent)
        bypass_anonymization = opts.get("bypass_anonymization", False)

        # T4: Pluggable Sovereign Data Anonymization Pipeline (Heads)
        effective_categories: list[str] = []
        effective_messages = [dict(m) for m in messages]
        effective_system_prompt = system_prompt
        mapping: dict[str, str] = {}
        all_head_matches = []

        if not bypass_anonymization and self._privacy_auditor is not None:
            if system_prompt:
                ev = self._privacy_auditor.audit(system_prompt, "heads", "router")
                if ev and ev.matches:
                    all_head_matches.extend(ev.matches)
            for msg in messages:
                ev = self._privacy_auditor.audit(msg.get("content", ""), "heads", "router")
                if ev and ev.matches:
                    all_head_matches.extend(ev.matches)

            if all_head_matches:
                if self._anonymization_pipeline:
                    if system_prompt:
                        effective_system_prompt, map_sys = self._anonymization_pipeline.anonymize(system_prompt, all_head_matches)
                        mapping.update(map_sys)
                    effective_messages = []
                    for msg in messages:
                        new_msg = dict(msg)
                        if new_msg.get("content"):
                            new_content, map_msg = self._anonymization_pipeline.anonymize(new_msg["content"], all_head_matches)
                            mapping.update(map_msg)
                            new_msg["content"] = new_content
                        effective_messages.append(new_msg)

                    tail_matches = []
                    if effective_system_prompt:
                        tev = self._privacy_auditor.audit(effective_system_prompt, "tails", "router")
                        if tev and tev.matches:
                            tail_matches.extend(tev.matches)
                    for msg in effective_messages:
                        tev = self._privacy_auditor.audit(msg.get("content", ""), "tails", "router")
                        if tev and tev.matches:
                            tail_matches.extend(tev.matches)

                    if tail_matches:
                        effective_categories = list(set(m.category for m in tail_matches))
                else:
                    effective_categories = list(set(m.category for m in all_head_matches))

        if not effective_categories and opts.get("data_categories"):
            effective_categories = list(opts.get("data_categories"))

        # v2.1: SovereigntyGate enforcement parity on multi-turn sessions
        sovereignty_verdict_open = True
        if not bypass_anonymization and self._sovereignty_gate is not None:
            combined_text = "\n".join(
                ([effective_system_prompt] if effective_system_prompt else [])
                + [str(m.get("content", "")) for m in effective_messages]
            )
            sovereignty_result = self._sovereignty_gate.evaluate(combined_text, effective_categories)
            sovereignty_verdict_open = "OPEN" in str(getattr(sovereignty_result, "verdict", "OPEN")).upper()
            try:
                chain = self._sovereignty_gate.enforce(chain, self._config, sovereignty_result)
            except Exception as ex:
                return GenerateResult(
                    text=f"ERROR: Sovereignty gate filtered all models. {ex}"
                )
            if not chain:
                return GenerateResult(
                    text="ERROR: Sovereignty gate filtered all models. "
                         "No models available for sensitive content."
                )

        proj_cfg = self._get_proj_cfg()
        proj_enabled = bool(proj_cfg.get("enabled", True))
        reserved_out_tokens = int(proj_cfg.get("reserved_output_tokens", 2048))
        safety_margin = float(proj_cfg.get("safety_margin_ratio", 0.08))

        raw_est_input = max(
            1,
            (
                len(effective_system_prompt)
                + sum(len(str(m.get("content", ""))) for m in effective_messages)
            )
            // 4,
        )
        session_id = opts.get("session_id")
        raw_cplx = opts.get("complexity_score", 5)
        complexity_val = int(raw_cplx) if isinstance(raw_cplx, (int, float)) else 5

        eval_ctx = CandidateEvalContext(
            role=role,
            intent=str(intent),
            complexity_score=complexity_val,
            estimated_input_tokens=raw_est_input,
            estimated_output_tokens=reserved_out_tokens,
            effective_data_categories=frozenset(effective_categories),
            sovereignty_verdict_open=sovereignty_verdict_open,
            session_id=session_id if isinstance(session_id, str) else None,
            last_model_id=opts.get("last_model_id"),
            last_routed_at=opts.get("last_routed_at"),
            bypass_sovereignty=bool(bypass_anonymization),
            failover_projection_enabled=proj_enabled,
            deterministic=bool(opts.get("deterministic", False)),
            deterministic_seed=opts.get("deterministic_seed"),
            max_vram_gb=float(opts["max_vram_gb"]) if opts.get("max_vram_gb") is not None else None,
            quant=str(opts.get("quant", "q4_k_m")),
        )
        original_first_model = chain[0] if chain else None
        chain, stack_trace = self._apply_routing_stack(chain, eval_ctx)
        if stack_trace is not None and not chain and stack_trace.pruned:
            if all("BudgetCapFilter" in r for r in stack_trace.pruned.values()):
                return GenerateResult(
                    text="BUDGET_EXCEEDED: All cloud models blocked by budget limits. "
                         "Configure local models as fallback."
                )

        first_failed_model: str | None = None
        first_fail_reason: str = ""
        if (
            stack_trace is not None
            and original_first_model
            and original_first_model in stack_trace.pruned
        ):
            first_failed_model = original_first_model
            first_fail_reason = stack_trace.pruned[original_first_model]

        errors: list[str] = []
        budget_skipped: list[str] = []
        session_obj = opts.get("session")

        # Pre-populate cached token counts once before the candidate loop when session is absent
        # so SessionProjector never re-runs count_tokens() on every fallback candidate (QA 1.6).
        if proj_enabled and session_obj is None and effective_messages:
            from aurarouter.sessions.gisting import GIST_INSTRUCTION as _GIST_INST
            from aurarouter.sessions.projector import _fast_estimate_tokens as _fast_est_tok

            enriched_messages: list[dict] = []
            for idx_m, m_item in enumerate(effective_messages):
                m_copy = dict(m_item)
                c_str = str(m_copy.get("content", ""))
                if not m_copy.get("tokens") and not (
                    idx_m == len(effective_messages) - 1
                    and _GIST_INST
                    and c_str.endswith(_GIST_INST)
                ):
                    m_copy["tokens"] = _fast_est_tok(c_str)
                enriched_messages.append(m_copy)
            effective_messages = enriched_messages

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg:
                continue

            provider_name = model_cfg.get("provider", "")
            hosting_tier = model_cfg.get("hosting_tier")

            # Circuit breaker check
            cb = self._circuit_breakers.get_or_create(model_id)
            if not cb.is_available():
                logger.warning("[%s] Skipping %s in session — circuit breaker open", role.upper(), model_id)
                errors.append(f"{model_id}: circuit breaker open")
                if first_failed_model is None:
                    first_failed_model = model_id
                    first_fail_reason = "circuit_open"
                continue

            # Budget check for cloud models
            if self._budget_manager is not None:
                from aurarouter.savings.pricing import is_cloud_tier
                if is_cloud_tier(hosting_tier, provider_name, endpoint=model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")):
                    status = self._check_model_budget(provider_name, hosting_tier, model_cfg)
                    if not status.allowed:
                        budget_skipped.append(model_id)
                        if first_failed_model is None:
                            first_failed_model = model_id
                            first_fail_reason = "budget_exceeded"
                        continue

            # Gatekeeping
            model_categories = self._get_allowed_data_cats(model_id)
            if effective_categories and not all(cat in model_categories for cat in effective_categories):
                logger.warning("[%s] Skipping %s — unauthorized for categories %s", role.upper(), model_id, effective_categories)
                errors.append(f"{model_id}: unauthorized for {effective_categories}")
                if first_failed_model is None:
                    first_failed_model = model_id
                    first_fail_reason = "sovereignty_gatekeeping"
                continue

            provider = self._get_provider(model_id)
            if provider is None:
                continue

            # v2.1: Ephemeral O(N) context projection when candidate context_limit < raw session size
            target_ctx_limit = int(
                model_cfg.get("context_limit")
                or model_cfg.get("context_window")
                or (model_cfg.get("parameters") or {}).get("n_ctx")
                or 0
            )
            projected = None
            was_projected = False
            if proj_enabled and target_ctx_limit > 0:
                projected = self._session_projector.project(
                    messages=effective_messages,
                    system_prompt=effective_system_prompt,
                    budget=TargetContextBudget(
                        model_id=model_id,
                        total_context_limit=target_ctx_limit,
                        reserved_output_tokens=reserved_out_tokens,
                        safety_margin_ratio=safety_margin,
                    ),
                    session=session_obj,
                    failover_from=first_failed_model,
                    failover_reason=first_fail_reason,
                )
                candidate_messages = projected.messages
                candidate_sys_prompt = projected.system_prompt
                est_input = max(1, projected.estimated_input_tokens)
                was_projected = projected.was_projected
                if was_projected and isinstance(session_id, str) and session_id:
                    self._trace_logger.log_stage(session_id, "failover_context_projection", {
                        "model_id": model_id,
                        "failover_from": first_failed_model,
                        "target_context_limit": target_ctx_limit,
                        "estimated_input_tokens": projected.estimated_input_tokens,
                        "turns_kept_full": projected.turns_kept_full,
                        "turns_gisted": projected.turns_gisted,
                        "turns_dropped": projected.turns_dropped,
                        "active_turn_truncated": projected.active_turn_truncated,
                    })
            else:
                candidate_messages = [
                    {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
                    for m in effective_messages
                ]
                candidate_sys_prompt = effective_system_prompt
                est_input = raw_est_input

            prov_key = self._resolve_vendor(model_id, model_cfg)
            lease = self._rate_limit_tracker.reserve(
                model_id,
                prov_key,
                est_input_tokens=est_input,
                est_output_tokens=min(512, reserved_out_tokens),
            )
            util = self._rate_limit_tracker.peek_utilization(
                model_id, prov_key, fallback_provider=provider_name
            )
            _emit_otel_routing_attributes(
                model_id=model_id,
                provider=prov_key,
                routing_mode=self._routing_stack.mode,
                headroom_score=util.headroom_score,
                projection_applied=was_projected,
            )

            start = time.monotonic()
            try:
                gen_kwargs: dict = {"json_mode": json_mode}
                if return_tokens:
                    gen_kwargs["return_tokens"] = True
                result = provider.generate_with_history(
                    candidate_messages, system_prompt=candidate_sys_prompt, **gen_kwargs
                )

                # Auto-continuation loop (Task 1.1)
                max_continuations = 3
                continuations = 0
                while result and result.finish_reason == "length" and continuations < max_continuations:
                    logger.info(f"[{role.upper()}] Response truncated (length), auto-continuing... ({continuations+1}/{max_continuations})")

                    # Add partial response to history and ask for continuation
                    messages_with_partial = list(candidate_messages)
                    messages_with_partial.append({"role": "assistant", "content": result.text})
                    messages_with_partial.append({"role": "user", "content": "continue"})

                    next_result = provider.generate_with_history(
                        messages_with_partial, system_prompt=candidate_sys_prompt, **gen_kwargs
                    )

                    if not next_result or not next_result.text:
                        break

                    # Concatenate text and sum tokens
                    result.text += next_result.text
                    result.input_tokens += next_result.input_tokens
                    result.output_tokens += next_result.output_tokens
                    result.finish_reason = next_result.finish_reason
                    continuations += 1

                # Deanonymize the full response
                if mapping and self._anonymization_pipeline and result and result.text:
                    result.text = self._anonymization_pipeline.deanonymize(result.text, mapping)
                    if getattr(result, "metadata", None) is None:
                        result.metadata = {}
                    result.metadata["anonymized"] = True

                elapsed = time.monotonic() - start

                if result and result.text and result.text.strip():
                    result.model_id = result.model_id or model_id
                    result.provider = result.provider or provider_name
                    if was_projected and projected is not None:
                        if getattr(result, "metadata", None) is None:
                            result.metadata = {}
                        result.metadata["context_projected"] = True
                        result.metadata["turns_kept_full"] = projected.turns_kept_full
                        result.metadata["turns_gisted"] = projected.turns_gisted
                        result.metadata["turns_dropped"] = projected.turns_dropped
                        result.metadata["active_turn_truncated"] = projected.active_turn_truncated

                    # Extract gist from response
                    clean_text, gist_text = extract_gist(result.text)
                    result.text = clean_text
                    if gist_text:
                        result.gist = gist_text

                    self._handle_attempt_outcome(
                        role=role,
                        model_id=model_id,
                        provider_name=provider_name,
                        cb=cb,
                        lease=lease,
                        elapsed=elapsed,
                        est_input_tokens=est_input,
                        complexity_val=complexity_val,
                        intent=str(intent),
                        session_id=session_id if isinstance(session_id, str) else None,
                        success=True,
                        actual_input_tokens=result.input_tokens,
                        actual_output_tokens=result.output_tokens,
                        snapshot=getattr(result, "rate_limit", None),
                        ttft_s=getattr(result, "ttft_s", None),
                    )
                    return result
                else:
                    raise ValueError("Empty response")
            except Exception as e:
                elapsed = time.monotonic() - start
                reason_code, err_msg = self._handle_attempt_outcome(
                    role=role,
                    model_id=model_id,
                    provider_name=provider_name,
                    cb=cb,
                    lease=lease,
                    elapsed=elapsed,
                    est_input_tokens=est_input,
                    complexity_val=complexity_val,
                    intent=str(intent),
                    session_id=session_id if isinstance(session_id, str) else None,
                    success=False,
                    exc=e,
                )
                errors.append(err_msg)
                if first_failed_model is None:
                    first_failed_model = model_id
                    first_fail_reason = reason_code
                continue
            finally:
                if not lease.settled:
                    self._rate_limit_tracker.abort(lease)

        if budget_skipped and not errors:
            return GenerateResult(
                text="BUDGET_EXCEEDED: All cloud models blocked by budget limits. "
                     "Configure local models as fallback."
            )

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
        routing_context=None,
    ) -> AsyncIterator[str]:
        """Streaming variant of :meth:`execute` with fallback chain and pre-TTFT deadline.

        Iterates through the role's model chain. For each model, attempts
        to stream tokens within ``ttft_timeout_seconds`` (or
        ``saturated_ttft_timeout_seconds`` when saturated / half_open).
        If a failure or TTFT stall occurs before any tokens are yielded,
        falls back cleanly to the next model. If tokens have already been
        yielded, the error is raised (no partial-then-retry).
        """
        opts = options or {}
        # Task 3.1: Support monologue/speculative modes in streaming
        execution_mode = opts.get("_execution_policy", {}).get("mode", "standard")
        permissions = opts.get("permissions")

        if execution_mode == "monologue":
            # Stream monologue events (Task 3.2)
            import queue
            q = queue.Queue()

            def _callback(event: str):
                q.put(event)

            async def _run():
                await self.execute_monologue(
                    task=prompt, permissions=permissions, event_callback=_callback
                )
                q.put(None)  # Sentinel

            threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()

            while True:
                try:
                    event = await asyncio.to_thread(q.get, timeout=300)
                    if event is None:
                        break
                    yield event + "\n"
                except queue.Empty:
                    break
            return

        # Intent-aware schema enforcement (TG6)
        intent = opts.get("intent", "chat")
        actionable = intent in ("edit_code", "generate_code")
        if actionable:
            json_mode = True
            file_constraints = opts.get("file_constraints")
            response_schema = compile_modifications_schema(file_constraints)
        else:
            response_schema = None

        chain = self._config.get_role_chain(role)
        if not chain:
            yield f"ERROR: No models defined for role '{role}' in YAML."
            return

        chain = self.consult_routing_advisors(role, chain, intent=intent)
        bypass_anonymization = opts.get("bypass_anonymization", False)
        effective_categories = self.get_effective_categories(prompt) if not bypass_anonymization else []
        if not effective_categories and opts.get("data_categories"):
            effective_categories = list(opts.get("data_categories"))

        prompt, rc = self._augment_prompt(prompt, role)
        if routing_context is None:
            routing_context = rc
        elif hasattr(routing_context, "retrieval_used"):
            routing_context.retrieval_used = routing_context.retrieval_used or rc.retrieval_used
            if rc.sources:
                routing_context.sources.extend(rc.sources)
            routing_context.author_id = routing_context.author_id or rc.author_id
            routing_context.project_id = routing_context.project_id or rc.project_id

        sovereignty_verdict_open = True
        if not bypass_anonymization and self._sovereignty_gate is not None:
            sovereignty_result = self._sovereignty_gate.evaluate(prompt, effective_categories)
            sovereignty_verdict_open = "OPEN" in str(getattr(sovereignty_result, "verdict", "OPEN")).upper()
            try:
                chain = self._sovereignty_gate.enforce(chain, self._config, sovereignty_result)
            except Exception as ex:
                yield f"ERROR: Sovereignty gate filtered all models. {ex}"
                return
            if not chain:
                yield (
                    "ERROR: Sovereignty gate filtered all models. "
                    "No models available for sensitive content."
                )
                return

        session_id = opts.get("session_id", getattr(routing_context, "session_id", None))
        raw_cplx = opts.get("complexity_score", getattr(routing_context, "complexity_score", 5))
        complexity_val = int(raw_cplx) if isinstance(raw_cplx, (int, float)) else 5
        est_input = max(1, len(prompt) // 4)
        est_output = int(opts.get("max_tokens", 512))

        eval_ctx = CandidateEvalContext(
            role=role,
            intent=str(intent),
            complexity_score=complexity_val,
            estimated_input_tokens=est_input,
            estimated_output_tokens=est_output,
            effective_data_categories=frozenset(effective_categories),
            sovereignty_verdict_open=sovereignty_verdict_open,
            session_id=session_id if isinstance(session_id, str) else None,
            last_model_id=opts.get("last_model_id"),
            last_routed_at=opts.get("last_routed_at"),
            bypass_sovereignty=bool(bypass_anonymization),
            failover_projection_enabled=False,
            deterministic=bool(opts.get("deterministic", False)),
            deterministic_seed=opts.get("deterministic_seed"),
            max_vram_gb=float(opts["max_vram_gb"]) if opts.get("max_vram_gb") is not None else None,
            quant=str(opts.get("quant", "q4_k_m")),
        )
        chain, _ = self._apply_routing_stack(chain, eval_ctx)
        stream_cfg = self._get_stream_cfg()

        for model_id in chain:
            model_cfg = self._config.get_model_config(model_id)
            if not model_cfg or not isinstance(model_cfg, dict):
                continue

            provider_name = model_cfg.get("provider", "")
            hosting_tier = model_cfg.get("hosting_tier")

            cb = self._circuit_breakers.get_or_create(model_id)
            cb_prev_state, _ = cb.peek_state()
            if not cb.is_available():
                logger.warning("[%s] Skipping %s in stream — circuit breaker open", role.upper(), model_id)
                self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                continue

            if self._budget_manager is not None:
                from aurarouter.savings.pricing import is_cloud_tier
                if is_cloud_tier(hosting_tier, provider_name, endpoint=model_cfg.get("endpoint") or model_cfg.get("mcp_endpoint")):
                    status = self._check_model_budget(provider_name, hosting_tier, model_cfg)
                    if not status.allowed:
                        self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                        continue

            model_categories = self._get_allowed_data_cats(model_id)
            if effective_categories and not all(cat in model_categories for cat in effective_categories):
                self._fire_callback(on_model_tried, role, model_id, False, 0.0)
                continue

            logger.info(f"[{role.upper()}] Streaming from: {model_id} ({provider_name})")

            provider = self._get_provider(model_id)
            if provider is None:
                continue

            prov_key = self._resolve_vendor(model_id, model_cfg)
            lease = self._rate_limit_tracker.reserve(
                model_id,
                prov_key,
                est_input_tokens=est_input,
                est_output_tokens=est_output,
            )
            util = self._rate_limit_tracker.peek_utilization(
                model_id, prov_key, fallback_provider=provider_name
            )
            _emit_otel_routing_attributes(
                model_id=model_id,
                provider=prov_key,
                routing_mode=self._routing_stack.mode,
                headroom_score=util.headroom_score,
                projection_applied=False,
            )

            if util.state == RateState.NEARING_SATURATION or cb_prev_state == "half_open":
                ttft_deadline = float(stream_cfg.get("saturated_ttft_timeout_seconds", 5.0))
            else:
                ttft_deadline = float(stream_cfg.get("ttft_timeout_seconds", 15.0))

            tokens_yielded = False
            start_stream = time.monotonic()
            out_chars = 0
            agen = None
            try:
                stream_kwargs: dict = {"json_mode": json_mode}
                if response_schema is not None:
                    stream_kwargs["response_schema"] = response_schema

                agen = provider.generate_stream(prompt, **stream_kwargs).__aiter__()
                try:
                    if ttft_deadline > 0:
                        first_token = await asyncio.wait_for(agen.__anext__(), timeout=ttft_deadline)
                    else:
                        first_token = await agen.__anext__()
                except StopAsyncIteration:
                    raise ValueError("Empty stream response")

                ttft_s = time.monotonic() - start_stream
                tokens_yielded = True
                out_chars += len(first_token)
                yield first_token

                async for token in agen:
                    out_chars += len(token)
                    yield token

                elapsed_stream = time.monotonic() - start_stream
                stream_meta, simulated = _consume_provider_stream_meta(provider)
                actual_in = (
                    stream_meta.input_tokens
                    if stream_meta is not None and stream_meta.input_tokens > 0
                    else est_input
                )
                actual_out = (
                    stream_meta.output_tokens
                    if stream_meta is not None and stream_meta.output_tokens > 0
                    else max(1, out_chars // 4)
                )
                eff_ttft = (stream_meta.ttft_s if stream_meta is not None else None) if simulated else ttft_s
                self._handle_attempt_outcome(
                    role=role,
                    model_id=model_id,
                    provider_name=provider_name,
                    cb=cb,
                    lease=lease,
                    elapsed=elapsed_stream,
                    est_input_tokens=est_input,
                    complexity_val=complexity_val,
                    intent=str(intent),
                    session_id=session_id if isinstance(session_id, str) else None,
                    routing_context=routing_context,
                    on_model_tried=on_model_tried,
                    success=True,
                    actual_input_tokens=actual_in,
                    actual_output_tokens=actual_out,
                    snapshot=stream_meta.rate_limit if stream_meta is not None else None,
                    ttft_s=eff_ttft,
                )
                return
            except Exception as e:
                elapsed_stream = time.monotonic() - start_stream
                _consume_provider_stream_meta(provider)
                self._handle_attempt_outcome(
                    role=role,
                    model_id=model_id,
                    provider_name=provider_name,
                    cb=cb,
                    lease=lease,
                    elapsed=elapsed_stream,
                    est_input_tokens=est_input,
                    complexity_val=complexity_val,
                    intent=str(intent),
                    session_id=session_id if isinstance(session_id, str) else None,
                    routing_context=routing_context,
                    on_model_tried=on_model_tried if not tokens_yielded else None,
                    success=False,
                    exc=e,
                )
                if tokens_yielded:
                    raise
                continue
            finally:
                if not lease.settled:
                    self._rate_limit_tracker.abort(lease)
                if agen is not None and hasattr(agen, "aclose"):
                    try:
                        await agen.aclose()
                    except Exception:
                        pass

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
        task = self._augment_prompt(task, "speculative")

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

    # -------------------------------------------------------------------
    # TG7: Monologue reasoning execution
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Shut down background resources. Safe to call multiple times."""
        self._event_reporter.shutdown(wait=False)

    async def execute_monologue(
        self,
        task: str,
        context: str | None = None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.85,
        mas_relevancy_threshold: float = 0.4,
    ):
        """Execute a task using recursive multi-expert reasoning (AuraMonologue)."""
        task = self._augment_prompt(task, "monologue")

        from aurarouter.monologue import MonologueOrchestrator

        orchestrator = MonologueOrchestrator(
            fabric=self,
            mcp_registry=getattr(self, '_mcp_registry', None),
            sovereignty_gate=self._sovereignty_gate,
            rag_pipeline=getattr(self, '_rag_pipeline', None),
        )
        return await orchestrator.reason(
            task=task,
            context=context,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            mas_relevancy_threshold=mas_relevancy_threshold,
        )
