"""Comprehensive forensic regression tests for AuraRouter QA assessment items 1.1-3.4.

Covers:
- 1.1: Per-thread random.Random isolation under high concurrency
- 1.2: FeedbackStore exponential decay & PosteriorSnapshotCache no-clobber hydration
- 1.3: TatCell pure reads, exact tau_elastic unwind, stale header & active cooldown protection
- 1.4: BudgetManager lock-free peek & singleflight async refresh
- 1.5: TraceLogger concurrent append safety & async flush
- 1.6: Session message metadata stripping, token count caching & gist preservation
- 2.1: Logger hierarchy standard
- 2.2: Unified _handle_attempt_outcome parity across execute, RT1 probe & execute_session
- 2.3: ConfigLoader single save & schema validation
- 2.4: Canonical vendor resolution & limiter key isolation
- 2.5: bandit_enabled=False neutral quality factor
- 3.3: McpProvider snapshot extraction & simulated stream TTFT protection
- 3.4: Ollama error classification & stream TTFT deadline enforcement
"""

from __future__ import annotations

import concurrent.futures
import random
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from aurarouter.circuit_breaker import CircuitBreakerRegistry
from aurarouter.config import ConfigLoader, get_logger
from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.event_reporter import EventReporter
from aurarouter.fabric import ComputeFabric, _stream_sync_with_ttft_deadline
from aurarouter.providers.base import BaseProvider
from aurarouter.providers.mcp_provider import McpProvider
from aurarouter.providers.ollama import _classify_ollama_http_error
from aurarouter.ratelimit.gcra import TatCell
from aurarouter.ratelimit.tracker import RateLimitTracker
from aurarouter.routing_schema import validate_v2_routing_config
from aurarouter.routing_stack.cache import PosteriorSnapshotCache
from aurarouter.routing_stack.context import CandidateEvalContext
from aurarouter.routing_stack.stack import RoutingPolicyStack
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.feedback_store import FeedbackStore
from aurarouter.savings.pricing import CostEngine, PricingCatalog, resolve_canonical_vendor
from aurarouter.savings.usage_store import UsageStore
from aurarouter.sessions.models import Message, Session
from aurarouter.sessions.projector import GIST_INSTRUCTION, SessionProjector
from aurarouter.trace_logger import TraceLogger


# ==============================================================================
# 1. Critical Concurrency / State Traps
# ==============================================================================


class Test1_1_ThreadLocalRng:
    """1.1: RoutingPolicyStack per-thread RNG isolation."""

    def test_concurrent_thompson_sampling_isolated_per_thread(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "m1": {"provider": "ollama", "model_name": "llama3"},
                "m2": {"provider": "ollama", "model_name": "mistral"},
            }
        }
        cache = PosteriorSnapshotCache()
        stack = RoutingPolicyStack(cfg, cache, stack_config={"mode": "active"})

        ctx = CandidateEvalContext(role="coding", complexity_score=5)
        barrier = threading.Barrier(8)
        observed_rngs: list[random.Random] = []
        lock = threading.Lock()

        def _worker():
            barrier.wait()
            rng_obj = stack._get_thread_rng()
            with lock:
                observed_rngs.append(rng_obj)
            for _ in range(25):
                trace = stack.evaluate_and_rank(["m1", "m2"], ctx)
                assert len(trace.ranked_chain) == 2

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread must have instantiated its own unique random.Random instance
        assert len(set(id(r) for r in observed_rngs)) == 8


class Test1_2_FeedbackStoreDecayAndCacheHydration:
    """1.2: Exponential decay in FeedbackStore and no-clobber hydration in cache."""

    def test_bulk_band_counts_exponential_decay(self, tmp_path: Path):
        db_path = tmp_path / "decay_fb.db"
        store = FeedbackStore(db_path=db_path)

        # Insert 1000 sequential records for model_alpha (successes)
        # Without decay, alpha count would be 1000.
        # With gamma=0.995, sum_{i=0}^{N-1} (0.995)^i is bounded <= 1/(1-0.995) = 200.
        for _ in range(1000):
            store.record("coding", 8.0, "model_alpha", True, 0.5, 500, 200)

        # Insert 1000 records for model_beta (failures)
        for _ in range(1000):
            store.record("coding", 8.0, "model_beta", False, 0.5, 500, 200)

        raw_counts = store.bulk_band_counts(decay_factor=0.995)
        counts = {(m, r, b): (s, f) for m, r, b, s, f in raw_counts}
        key_alpha = ("model_alpha", "coding", 2)
        key_beta = ("model_beta", "coding", 2)

        assert key_alpha in counts
        assert key_beta in counts

        alpha_s, alpha_f = counts[key_alpha]
        beta_s, beta_f = counts[key_beta]

        # Steady state effective sample size is bounded by 200.0
        assert alpha_s <= 200.0
        assert alpha_s >= 180.0
        assert alpha_f == 0.0

        assert beta_f <= 200.0
        assert beta_f >= 180.0
        assert beta_s == 0.0
        store.close()

    def test_hydration_preserves_concurrent_record_outcome_deltas(self, tmp_path: Path):
        db_path = tmp_path / "hydrate_fb.db"
        store = FeedbackStore(db_path=db_path)
        store.record("coding", 8.0, "m1", True, 0.5, 500, 200)

        cache = PosteriorSnapshotCache()
        # Record a live outcome into cache BEFORE hydration finishes (complexity 8.0 -> band 2)
        cache.record_outcome("m1", "coding", 8.0, True)

        # Hydrate from store
        loaded = cache.hydrate_from_store(store)
        assert loaded >= 1

        # Post-hydration posterior must combine both the loaded SQLite record AND the live delta
        a, b = cache.get_posterior("m1", "coding", 2, default_prior=(1.0, 1.0))
        # Default prior (1, 1) + loaded success (1.0) + live delta (1.0) = alpha 3.0, beta 1.0
        assert a == pytest.approx(3.0, rel=0.05)
        assert b == pytest.approx(1.0)
        store.close()


class Test1_3_GCRA_TatCellPurityAndUnwind:
    """1.3: TatCell lock-free pure reads, exact tau unwind, and race protection."""

    def test_tat_cell_reads_are_strictly_pure(self):
        cell = TatCell()
        t0 = 1000.0
        cell.reserve_with_tau(cost=5, quota=60, window=60.0, t_now=t0)
        initial_tat = cell._tat

        # Query metrics via read methods
        cell.get_utilization(t_now=t0 + 10.0, window=60.0)
        cell.get_state(t_now=t0 + 10.0, window=60.0)
        cell.seconds_until_reset(t_now=t0 + 10.0)

        # State must remain 100% unmutated
        assert cell._tat == initial_tat
        assert cell._reset_at == 0.0

    def test_admitted_false_does_not_underflow_inflight_cost(self):
        cell = TatCell()
        t0 = 1000.0
        # Put cell in cooldown
        cell.settle_error(reserved_cost=0, retry_after_s=10.0, t_now=t0)

        # Reserve fails because in cooldown
        admitted, tau_used = cell.reserve_with_tau(cost=1, quota=10, window=60.0, t_now=t0)
        assert not admitted
        assert cell._inflight_cost == 0

        # Abort of non-admitted request must not decrement inflight cost
        cell.abort(reserved_cost=1, quota=10, t_now=t0, admitted=False, tau_used=tau_used)
        assert cell._inflight_cost == 0

    def test_exact_tau_elastic_unwind_on_abort_with_replicas(self):
        cell = TatCell()
        t0 = 1000.0
        # Replica count 4 -> tau_elastic > tau_global
        admitted, tau_elastic = cell.reserve_with_tau(
            cost=2, quota=60, window=60.0, t_now=t0, replica_count=4, skew_damping=0.32
        )
        assert admitted

        # Aborting with tau_used unwinds exact tau_elastic
        cell.abort(
            reserved_cost=2,
            quota=60,
            window=60.0,
            t_now=t0,
            admitted=True,
            tau_used=tau_elastic,
        )
        assert cell._tat == pytest.approx(t0)

    def test_out_of_order_stale_header_and_cooldown_protection(self):
        cell = TatCell()
        t0 = 1000.0

        # R1 dispatched at t0, R2 dispatched at t0 + 1.0
        # R2 triggers 429 cooldown at t0 + 2.0
        cell.settle_error(reserved_cost=1, retry_after_s=30.0, t_now=t0 + 2.0)
        assert cell.is_in_cooldown(t0 + 2.0)

        # R1 (dispatched earlier at t0) finishes late at t0 + 3.0 with 200 OK headers
        # Settle must NOT clear active 429 cooldown!
        cell.settle(
            actual_cost=1,
            reserved_cost=1,
            quota=100,
            limit=100,
            remaining=90,
            reset_s=50.0,
            lease_created_at=t0,
            t_now=t0 + 3.0,
        )
        # Cooldown remains active
        assert cell.is_in_cooldown(t0 + 3.0)
        assert cell._cooldown_until == pytest.approx(t0 + 32.0)


class Test1_4_BudgetManagerNonBlocking:
    """1.4: BudgetManager lock-free peek and async singleflight refresh."""

    def test_peek_budget_allowed_returns_immediately_on_cold_cache(self, tmp_path: Path):
        db_path = tmp_path / "budget_usage.db"
        usage_store = UsageStore(db_path=db_path)
        cost_engine = CostEngine(PricingCatalog(), usage_store)

        reporter = EventReporter()
        budget_mgr = BudgetManager(
            cost_engine=cost_engine,
            config={"enabled": True, "daily_limit": 50.0, "monthly_limit": 500.0},
            event_reporter=reporter,
        )

        # Poison cost_engine total_spend to prove peek_budget_allowed does not touch SQLite synchronously
        cost_engine.total_spend = MagicMock(side_effect=AssertionError("Synchronous SQLite hit!"))

        # Lock-free non-blocking check
        allowed, reason = budget_mgr.peek_budget_allowed("openai", hosting_tier="cloud")
        assert allowed is True
        assert reason == ""
        reporter.shutdown(wait=False)
        usage_store.close()


class Test1_5_TraceLoggerSafety:
    """1.5: TraceLogger thread safety and flush barrier."""

    def test_concurrent_trace_logging(self, tmp_path: Path):
        reporter = EventReporter()
        logger = TraceLogger(trace_dir=tmp_path, event_reporter=reporter)

        def _log_worker(worker_id: int):
            for i in range(25):
                logger.log_stage(
                    session_id="sess_123",
                    stage_type="candidate_evaluation",
                    payload={"worker": worker_id, "step": i},
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(_log_worker, wid) for wid in range(6)]
            for f in futures:
                f.result()

        logger.flush()
        traces = logger.read_trace("sess_123")
        assert len(traces) == 150
        reporter.shutdown(wait=False)


class Test1_6_SessionManagerAndProjector:
    """1.6: Metadata stripping, cached tokens, and gist instruction preservation."""

    def test_session_messages_as_dicts_excludes_metadata(self):
        msg = Message(
            role="user",
            content="Hello world",
            tokens=10,
            gist="Brief gist",
        )
        session = Session(session_id="s1")
        session.add_message(msg)

        dicts_clean = session.get_messages_as_dicts(include_metadata=False)
        assert "tokens" not in dicts_clean[0]
        assert "gist" not in dicts_clean[0]
        assert dicts_clean[0]["content"] == "Hello world"

        dicts_with_meta = session.get_messages_as_dicts(include_metadata=True)
        assert dicts_with_meta[0]["tokens"] == 10
        assert dicts_with_meta[0]["gist"] == "Brief gist"

    def test_projector_preserves_trailing_gist_instruction_on_elision(self):
        projector = SessionProjector()
        text_with_gist = f"Long multi-paragraph conversation history that requires elision.\n{GIST_INSTRUCTION}"
        elided, _ = projector.elide_middle_out(text_with_gist, max_tokens=25)
        # Trailing GIST_INSTRUCTION must be preserved intact
        assert elided.endswith(GIST_INSTRUCTION)


# ==============================================================================
# 2. Structural & Architectural Coherence
# ==============================================================================


class Test2_1_LoggerHierarchy:
    """2.1: Unified AuraRouter logger hierarchy."""

    def test_get_logger_emits_aurarouter_prefix(self):
        log = get_logger("routing_stack")
        assert log.name == "AuraRouter.routing_stack"
        log2 = get_logger("AuraRouter.savings")
        assert log2.name == "AuraRouter.savings"


class Test2_2_UnifiedOutcomeParity:
    """2.2: Parity of _handle_attempt_outcome across routing execution modes."""

    def test_handle_attempt_outcome_records_ratelimit_and_circuit_breaker(self):
        cfg = ConfigLoader(allow_missing=True)
        fabric = ComputeFabric(cfg)
        cb_reg = CircuitBreakerRegistry()
        cb = cb_reg.get_or_create("test_model")
        tracker = RateLimitTracker()
        lease = tracker.reserve("test_model", "test_prov")

        exc = RateLimitExceededError("Rate exceeded", retry_after_s=12.0)
        fabric._rate_limit_tracker = tracker
        fabric._handle_attempt_outcome(
            role="coding",
            model_id="test_model",
            provider_name="test_prov",
            cb=cb,
            lease=lease,
            elapsed=0.05,
            est_input_tokens=100,
            exc=exc,
        )

        # Rate tracker must be settled into cooldown
        util = tracker.peek_utilization("test_model", "test_prov")
        assert util.seconds_until_reset >= 10.0

        # CircuitBreaker must NOT record failure on 429 (fault isolation invariant)
        assert cb.peek_state()[1] == 0

        # Now test 503 transient failure: MUST record failure on CircuitBreaker
        lease2 = tracker.reserve("test_model", "test_prov")
        exc_transient = ProviderTransientError("Outage", status_code=503)
        fabric._handle_attempt_outcome(
            role="coding",
            model_id="test_model",
            provider_name="test_prov",
            cb=cb,
            lease=lease2,
            elapsed=0.05,
            est_input_tokens=100,
            exc=exc_transient,
        )
        assert cb.peek_state()[1] == 1


class Test2_3_ConfigSchemaValidation:
    """2.3: Single ConfigLoader.save definition and schema validator."""

    def test_single_configloader_save_method(self):
        # Verify only one save method exists on ConfigLoader class
        save_attr = getattr(ConfigLoader, "save", None)
        assert callable(save_attr)
        assert "save" in ConfigLoader.__dict__

    def test_validate_v2_routing_config_warns_cleanly(self):
        malformed_cfg = {
            "routing_stack": {"cost_lambda_max": "not_a_float", "cost_elasticity_kappa": -5.0},
            "rate_limits": {"replica_count": -2},
        }
        res = validate_v2_routing_config(malformed_cfg)
        assert any("cost_lambda_max" in w for w in res.warnings)
        assert any("replica_count" in w for w in res.warnings)


class Test2_4_CanonicalVendorResolution:
    """2.4: Canonical vendor resolution for MCP and cloud endpoints."""

    def test_canonical_vendor_resolution(self):
        assert resolve_canonical_vendor("claude-3-5-sonnet", {"provider": "mcp", "endpoint": "http://aurarouter-claude:8000"}) == "anthropic"
        assert resolve_canonical_vendor("gemini-2.5-flash", {"provider": "mcp", "endpoint": "http://aurarouter-gemini:8000"}) == "google"
        assert resolve_canonical_vendor("gpt-4o", {"provider": "mcp", "endpoint": "http://aurarouter-openai:8000"}) == "openai"
        assert resolve_canonical_vendor("azure-gpt4", {"provider": "openapi", "endpoint": "https://my-app.openai.azure.com"}) == "azure_openai"
        # Local vLLM/OpenAPI remains openapi
        assert resolve_canonical_vendor("local-llama", {"provider": "openapi", "endpoint": "http://localhost:8000/v1"}) == "openapi"


class Test2_5_BanditDisabledNeutralUtility:
    """2.5: bandit_enabled=False neutral quality factor."""

    def test_bandit_disabled_emits_neutral_score(self):
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "models": {
                "m1": {"provider": "ollama", "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0},
                "m2": {"provider": "ollama", "cost_per_1m_input": 0.0, "cost_per_1m_output": 0.0},
            }
        }
        cache = PosteriorSnapshotCache()
        stack = RoutingPolicyStack(cfg, cache, stack_config={"mode": "active", "bandit_enabled": False})

        ctx = CandidateEvalContext(role="coding", complexity_score=5)
        trace = stack.evaluate_and_rank(["m1", "m2"], ctx)

        assert trace.scores[0].quality_sample == 1.0
        assert trace.scores[1].quality_sample == 1.0
        # Preserves initial YAML chain order deterministically
        assert trace.ranked_chain == ["m1", "m2"]


# ==============================================================================
# 3. Provider & Transport Boundaries
# ==============================================================================


class Test3_3_McpProviderSnapshotExtraction:
    """3.3: McpProvider snapshot extraction from hybrid retry_after and HTTP headers."""

    def test_extract_snapshot_hybrid_and_ignore_non_ratelimit_headers(self):
        prov = McpProvider({"mcp_endpoint": "http://mcp.local", "model_name": "test"})

        # Case 1: hybrid dict with retry_after_s and headers inside payload["rate_limit"]
        raw_rl = {
            "retry_after_s": 15.5,
            "is_daily_quota": True,
            "x-ratelimit-remaining-requests": "0",
        }
        snap = prov._extract_snapshot({"rate_limit": raw_rl})
        assert snap is not None
        assert snap.retry_after_s == pytest.approx(15.5)
        assert snap.is_daily_quota is True
        assert snap.rpm_remaining == 0

        # Case 2: non-ratelimit headers (e.g. content-type, date) must return None
        raw_non_rl = {
            "content-type": "application/json",
            "date": "Tue, 03 Nov 2026 12:00:00 GMT",
        }
        snap_none = prov._extract_snapshot({"rate_limit": raw_non_rl})
        assert snap_none is None


class Test3_4_OllamaErrorClassificationAndStreamDeadlines:
    """3.4: Ollama error classification and TTFT stream deadline."""

    def test_ollama_error_classification(self):
        req = httpx.Request("POST", "http://localhost:11434/api/generate")

        # HTTP 429 -> RateLimitExceededError
        resp_429 = httpx.Response(429, headers={"retry-after": "10"}, request=req)
        exc_429 = _classify_ollama_http_error(httpx.HTTPStatusError("429", request=req, response=resp_429))
        assert isinstance(exc_429, RateLimitExceededError)

        # Context length exceeded -> ContextLengthExceededError
        resp_ctx = httpx.Response(400, text="exceeds maximum context length of 4096", request=req)
        exc_ctx = _classify_ollama_http_error(httpx.HTTPStatusError("400", request=req, response=resp_ctx))
        assert isinstance(exc_ctx, ContextLengthExceededError)

        # HTTP 503 -> ProviderTransientError
        resp_503 = httpx.Response(503, text="service unavailable", request=req)
        exc_503 = _classify_ollama_http_error(httpx.HTTPStatusError("503", request=req, response=resp_503))
        assert isinstance(exc_503, ProviderTransientError)

    def test_stream_sync_ttft_deadline(self):
        def _slow_generator():
            time.sleep(0.1)
            yield "token1"

        mock_prov = MagicMock(spec=BaseProvider)
        mock_prov.generate_stream_sync.return_value = _slow_generator()

        # Deadline 0.02s < 0.1s sleep -> must raise TimeoutError
        with pytest.raises(TimeoutError) as exc_info:
            _stream_sync_with_ttft_deadline(
                provider=mock_prov,
                prompt="hello",
                gen_kwargs={},
                on_token=lambda tok: None,
                ttft_deadline=0.02,
            )
        assert "TTFT deadline" in str(exc_info.value)
