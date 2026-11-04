"""Phase 5 Integration Tests: Unified Execution Paths, Fault Isolation,
Ephemeral Session Projection, Pre-TTFT Stream Deadlines, MCP/HTTP/API & OTel.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
import types
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.config import ConfigLoader
from aurarouter.errors import (
    ContextLengthExceededError,
    ProviderTransientError,
    RateLimitExceededError,
)
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import (
    ratelimit_status,
    route_task,
    routing_explain,
    routing_stack_status,
)
from aurarouter.ratelimit.models import RateLimitSnapshot, RateState
from aurarouter.savings.models import GenerateResult
from aurarouter.server import create_mcp_server


def _build_config(routing_mode: str = "active") -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "resilience": {"failure_threshold": 2, "reset_timeout": 60.0},
        "rate_limits": {
            "enabled": True,
            "default_warning_threshold": 0.80,
            "emit_ietf_headers": True,
            "providers": {
                "anthropic": {"rpm": 100, "itpm": 100000, "otpm": 20000},
            },
        },
        "routing_stack": {
            "mode": routing_mode,
            "bandit_enabled": True,
            "cache_affinity_enabled": True,
            "cost_lambda_max": 0.35,
            "cost_elasticity_kappa": 0.32,
        },
        "sessions": {
            "failover_projection": {
                "enabled": True,
                "safety_margin_ratio": 0.08,
                "reserved_output_tokens": 512,
                "inject_failover_header": True,
            }
        },
        "streaming": {
            "ttft_timeout_seconds": 0.12,
            "saturated_ttft_timeout_seconds": 0.03,
        },
        "models": {
            "cloud_claude": {
                "provider": "anthropic",
                "hosting_tier": "cloud",
                "endpoint": "http://cloud.local/v1",
                "model_name": "claude-sonnet",
                "context_limit": 200000,
                "cost_per_1m_input": 3.0,
                "cost_per_1m_output": 15.0,
            },
            "local_qwen": {
                "provider": "ollama",
                "hosting_tier": "on-prem",
                "endpoint": "http://localhost:11434/api/generate",
                "model_name": "qwen2.5-coder:7b",
                "context_limit": 2048,
                "cost_per_1m_input": 0.0,
                "cost_per_1m_output": 0.0,
            },
        },
        "roles": {
            "coding": ["cloud_claude", "local_qwen"],
            "router": ["local_qwen"],
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# 1. Fault Isolation: 429 vs 400 Context Overflow vs 5xx Transient Failure
# ---------------------------------------------------------------------------


class TestFaultIsolationAndPreemption:
    def test_429_triggers_failover_and_cooldown_without_tripping_circuit_breaker(self):
        cfg = _build_config(routing_mode="active")
        fabric = ComputeFabric(cfg)

        cloud_prov = MagicMock()
        cloud_prov.generate_with_usage.side_effect = RateLimitExceededError(
            "Anthropic 429 rate limit",
            provider="anthropic",
            model_id="cloud_claude",
            retry_after_s=45.0,
            snapshot=RateLimitSnapshot(retry_after_s=45.0, rpm_limit=100, rpm_remaining=0),
        )

        local_prov = MagicMock()
        local_prov.generate_with_usage.return_value = GenerateResult(
            text="Local fallback succeeded!",
            input_tokens=40,
            output_tokens=20,
            ttft_s=0.05,
        )

        def _fake_get_prov(name, mcfg):
            return cloud_prov if name == "anthropic" else local_prov

        with patch("aurarouter.fabric.get_provider", side_effect=_fake_get_prov):
            # Request 1: deterministically ranks cloud_claude first -> 429 -> fails over to local_qwen
            res1 = fabric.execute(
                "coding",
                "Design distributed consensus algorithm",
                options={"complexity_score": 10, "deterministic": True},
            )
            assert res1 is not None
            assert res1.text == "Local fallback succeeded!"
            assert res1.model_id == "local_qwen"
            assert cloud_prov.generate_with_usage.call_count == 1

            # Verify CircuitBreaker was NOT tripped (0 failures recorded!)
            cb_cloud = fabric.circuit_breakers.get_or_create("cloud_claude")
            assert cb_cloud.peek_state() == ("closed", 0)

            # Verify RateLimitTracker marked cloud_claude EXHAUSTED
            util = fabric.rate_limit_tracker.get_utilization("cloud_claude", "anthropic")
            assert util.state == RateState.EXHAUSTED
            assert util.headroom_score == 0.0

            # Request 2: Stage 1 AvailabilityFilter proactively prunes cloud_claude!
            res2 = fabric.execute(
                "coding",
                "Design another consensus protocol",
                options={"complexity_score": 10, "deterministic": True},
            )
            assert res2 is not None
            assert res2.model_id == "local_qwen"
            # cloud_prov was NOT called a second time!
            assert cloud_prov.generate_with_usage.call_count == 1
            assert local_prov.generate_with_usage.call_count == 2

    def test_context_length_exceeded_does_not_trip_circuit_breaker_unlike_5xx(self):
        cfg = _build_config(routing_mode="off")
        fabric = ComputeFabric(cfg)

        cloud_prov = MagicMock()
        local_prov = MagicMock()
        local_prov.generate_with_usage.return_value = GenerateResult(
            text="ok", input_tokens=10, output_tokens=5
        )

        with patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ):
            # Three 400 ContextLengthExceededErrors must NOT open circuit (threshold=2)
            cloud_prov.generate_with_usage.side_effect = ContextLengthExceededError(
                "prompt too long", provider="anthropic", model_id="cloud_claude"
            )
            for _ in range(3):
                r = fabric.execute("coding", "huge prompt")
                assert r is not None and r.model_id == "local_qwen"

            cb_cloud = fabric.circuit_breakers.get_or_create("cloud_claude")
            assert cb_cloud.peek_state() == ("closed", 0)

            # Now two 503 ProviderTransientErrors MUST trip circuit breaker to 'open'!
            cloud_prov.generate_with_usage.side_effect = ProviderTransientError(
                "503 upstream unavailable", provider="anthropic", model_id="cloud_claude", status_code=503
            )
            fabric.execute("coding", "normal prompt")
            fabric.execute("coding", "normal prompt")
            assert cb_cloud.peek_state() == ("open", 2)


# ---------------------------------------------------------------------------
# 2. Ephemeral Failover Context Projection in execute_session()
# ---------------------------------------------------------------------------


class TestSessionFailoverProjection:
    def test_200k_cloud_429_projects_conversation_into_small_local_window_without_mutation(self):
        cfg = _build_config(routing_mode="off")
        fabric = ComputeFabric(cfg)

        # Build a multi-turn conversation (~3,500+ tokens) that exceeds local_qwen's 2048 window
        prior_context = (
            "[Prior Context]\n"
            "- Turn 1 implemented authentication middleware.\n"
            "- Turn 2 added JWT rotation and Redis revocation.\n"
            "[End Prior Context]"
        )
        messages = [{"role": "system", "content": prior_context}]
        for i in range(12):
            messages.append({
                "role": "user",
                "content": f"Turn {i} user request: " + ("refactor module alpha beta gamma " * 45),
            })
            messages.append({
                "role": "assistant",
                "content": f"Turn {i} assistant implementation:\n" + ("def handler_x(): return True\n" * 45),
                "gist": f"Completed refactor step {i}.",
            })
        messages.append({
            "role": "user",
            "content": "Final active question: fix the race condition in token refresh.",
        })
        original_snapshot = copy.deepcopy(messages)

        cloud_prov = MagicMock()
        cloud_prov.generate_with_history.side_effect = RateLimitExceededError(
            "429 rate limited",
            provider="anthropic",
            model_id="cloud_claude",
            retry_after_s=30.0,
            snapshot=RateLimitSnapshot(retry_after_s=30.0),
        )

        captured_local_call: dict = {}

        def _local_gen_history(msgs, system_prompt="", **kwargs):
            captured_local_call["messages"] = msgs
            captured_local_call["system_prompt"] = system_prompt
            return GenerateResult(
                text="Fixed race condition using asyncio.Lock.\n---GIST---\nAdded lock around token refresh.",
                input_tokens=600,
                output_tokens=45,
            )

        local_prov = MagicMock()
        local_prov.generate_with_history.side_effect = _local_gen_history

        with patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ):
            res = fabric.execute_session(
                role="coding",
                messages=messages,
                system_prompt="You are a senior systems engineer.",
                options={"session_id": "sess-proj-1"},
            )

        assert res.model_id == "local_qwen"
        assert "Fixed race condition" in res.text
        assert res.gist == "Added lock around token refresh."
        assert res.metadata.get("context_projected") is True

        # Caller's messages list was NEVER mutated!
        assert messages == original_snapshot

        # Projected payload sent to local_qwen is drastically smaller and preserves active turn + failover header
        sent_msgs = captured_local_call["messages"]
        assert len(sent_msgs) < len(messages)
        assert "Final active question: fix the race condition" in sent_msgs[-1]["content"]
        assert "[FAILOVER CONTEXT: Routed from cloud_claude" in captured_local_call["system_prompt"]

        # Cloud circuit breaker remained closed on 429
        assert fabric.circuit_breakers.get_or_create("cloud_claude").peek_state() == ("closed", 0)

    def test_execute_session_enforces_sovereignty_gate(self):
        cfg = _build_config(routing_mode="off")
        mock_gate = MagicMock()
        mock_gate.evaluate.return_value = MagicMock(verdict="LOCAL_ONLY")
        mock_gate.enforce.return_value = ["local_qwen"]

        fabric = ComputeFabric(cfg, sovereignty_gate=mock_gate)
        cloud_prov = MagicMock()
        local_prov = MagicMock()
        local_prov.generate_with_history.return_value = GenerateResult(
            text="Processed locally.", input_tokens=15, output_tokens=5
        )

        with patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ):
            res = fabric.execute_session(
                role="coding",
                messages=[{"role": "user", "content": "SSN 123-45-6789"}],
            )

        assert res.model_id == "local_qwen"
        assert cloud_prov.generate_with_history.call_count == 0


# ---------------------------------------------------------------------------
# 3. Streaming Pre-First-Token TTFT Timeout & Failover
# ---------------------------------------------------------------------------


class TestStreamingPreTTFTFailover:
    @pytest.mark.asyncio
    async def test_pre_first_token_stall_aborts_and_fails_over_cleanly(self):
        cfg = _build_config(routing_mode="off")
        fabric = ComputeFabric(cfg)
        closed_flag = {"stalled_closed": False}

        async def stalled_stream(prompt, **kwargs):
            try:
                await asyncio.sleep(5.0)
                yield "never_reached"
            finally:
                closed_flag["stalled_closed"] = True

        async def healthy_backup_stream(prompt, **kwargs):
            yield "backup_"
            yield "token"

        cloud_prov = MagicMock()
        cloud_prov.generate_stream = stalled_stream
        local_prov = MagicMock()
        local_prov.generate_stream = healthy_backup_stream

        tried: list[tuple[str, bool]] = []

        with patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ):
            chunks = []
            async for tok in fabric.execute_stream(
                "coding",
                "stream hello",
                on_model_tried=lambda r, mid, ok, el: tried.append((mid, ok)),
            ):
                chunks.append(tok)

        assert chunks == ["backup_", "token"]
        assert ("cloud_claude", False) in tried
        assert ("local_qwen", True) in tried
        assert closed_flag["stalled_closed"] is True
        # Lease pool must have zero leaked in-flight leases!
        assert len(fabric.rate_limit_tracker._leases) == 0

    @pytest.mark.asyncio
    async def test_half_open_circuit_uses_tighter_saturated_ttft_timeout(self):
        cfg = _build_config(routing_mode="off")
        cfg.config["resilience"] = {"failure_threshold": 1, "reset_timeout": 0.01}
        cfg.config["streaming"] = {
            "ttft_timeout_seconds": 2.0,
            "saturated_ttft_timeout_seconds": 0.03,
        }
        fabric = ComputeFabric(cfg)

        # Trip cloud_claude circuit breaker and let reset_timeout elapse -> half_open
        cb = fabric.circuit_breakers.get_or_create("cloud_claude")
        cb.record_failure()
        await asyncio.sleep(0.02)
        assert cb.peek_state()[0] == "half_open"

        async def slow_first_token_stream(prompt, **kwargs):
            # 0.08s > saturated_ttft_timeout_seconds (0.03s), but << ttft_timeout_seconds (2.0s)
            await asyncio.sleep(0.08)
            yield "too_slow_for_half_open"

        async def fast_backup_stream(prompt, **kwargs):
            yield "fast_fallback"

        cloud_prov = MagicMock()
        cloud_prov.generate_stream = slow_first_token_stream
        local_prov = MagicMock()
        local_prov.generate_stream = fast_backup_stream

        t0 = time.monotonic()
        with patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ):
            chunks = [t async for t in fabric.execute_stream("coding", "probe half open")]
        elapsed = time.monotonic() - t0

        assert chunks == ["fast_fallback"]
        assert elapsed < 0.5  # Tighter 0.03s timeout aborted well before 2.0s normal timeout!

    @pytest.mark.asyncio
    async def test_stream_429_fails_over_without_tripping_circuit_breaker(self):
        cfg = _build_config(routing_mode="off")
        fabric = ComputeFabric(cfg)

        async def rate_limited_stream(prompt, **kwargs):
            raise RateLimitExceededError(
                "429 stream",
                provider="anthropic",
                model_id="cloud_claude",
                retry_after_s=20.0,
                snapshot=RateLimitSnapshot(retry_after_s=20.0),
            )
            yield  # pragma: no cover

        async def backup_stream(prompt, **kwargs):
            yield "ok"

        cloud_prov = MagicMock()
        cloud_prov.generate_stream = rate_limited_stream
        local_prov = MagicMock()
        local_prov.generate_stream = backup_stream

        with patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ):
            chunks = [t async for t in fabric.execute_stream("coding", "hi")]

        assert chunks == ["ok"]
        assert fabric.circuit_breakers.get_or_create("cloud_claude").peek_state() == ("closed", 0)
        assert (
            fabric.rate_limit_tracker.get_state("cloud_claude", "anthropic")
            == RateState.EXHAUSTED
        )


# ---------------------------------------------------------------------------
# 4. Shadow vs Active Mode, Triage Bypass & OpenTelemetry Hook
# ---------------------------------------------------------------------------


class TestShadowActiveAndOTel:
    def test_shadow_mode_logs_counterfactual_while_preserving_yaml_order(self):
        cfg = _build_config(routing_mode="shadow")
        cfg.config["routing_stack"]["bandit_enabled"] = False
        fabric = ComputeFabric(cfg)

        # Complexity=1 heavily penalizes expensive cloud_claude in RoutingPolicyStack,
        # so counterfactual ranked_chain puts local_qwen first, while shadow execution
        # still calls cloud_claude first!
        cloud_prov = MagicMock()
        cloud_prov.generate_with_usage.return_value = GenerateResult(
            text="cloud answer", input_tokens=10, output_tokens=5
        )
        local_prov = MagicMock()
        local_prov.generate_with_usage.return_value = GenerateResult(
            text="local answer", input_tokens=10, output_tokens=5
        )

        with patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ), patch.object(fabric._trace_logger, "log_stage") as mock_trace:
            res = fabric.execute(
                "coding",
                "print 1+1",
                options={"complexity_score": 1, "session_id": "shadow-sess-1"},
            )

        assert res is not None
        assert res.model_id == "cloud_claude"  # YAML order preserved in shadow mode!
        assert fabric.last_routing_trace is not None
        assert fabric.last_routing_trace.mode == "shadow"
        assert fabric.last_routing_trace.ranked_chain[0] == "local_qwen"
        stage_names = [c.args[1] for c in mock_trace.call_args_list]
        assert "routing_stack_shadow" in stage_names

    def test_active_mode_routes_trivial_task_to_free_local_and_skips_legacy_triage(self):
        cfg = _build_config(routing_mode="active")
        cfg.config["routing_stack"]["bandit_enabled"] = False
        fabric = ComputeFabric(cfg)

        mock_triage = MagicMock()
        mock_triage.select_role.return_value = "nonexistent_role"

        local_prov = MagicMock()
        local_prov.generate_with_usage.return_value = GenerateResult(
            text="42", input_tokens=8, output_tokens=2
        )
        cloud_prov = MagicMock()
        cloud_prov.generate_with_usage.return_value = GenerateResult(
            text="cloud 42", input_tokens=8, output_tokens=2
        )

        with patch(
            "aurarouter.mcp_tools.analyze_intent",
            return_value=MagicMock(intent="DIRECT", complexity=1),
        ), patch(
            "aurarouter.fabric.get_provider",
            side_effect=lambda name, mcfg: cloud_prov if name == "anthropic" else local_prov,
        ):
            out = route_task(fabric, mock_triage, task="What is 6*7?", config=cfg)

        assert out == "42"
        # Legacy triage_router.select_role was bypassed because routing_stack is active!
        mock_triage.select_role.assert_not_called()
        assert cloud_prov.generate_with_usage.call_count == 0

    def test_opentelemetry_span_attributes_emitted_when_recording(self):
        cfg = _build_config(routing_mode="active")
        fabric = ComputeFabric(cfg)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        fake_trace_mod = types.ModuleType("opentelemetry.trace")
        fake_trace_mod.get_current_span = lambda: mock_span
        fake_otel_pkg = types.ModuleType("opentelemetry")
        fake_otel_pkg.trace = fake_trace_mod

        local_prov = MagicMock()
        local_prov.generate_with_usage.return_value = GenerateResult(
            text="otel ok", input_tokens=5, output_tokens=3
        )

        with patch.dict(
            sys.modules,
            {"opentelemetry": fake_otel_pkg, "opentelemetry.trace": fake_trace_mod},
        ), patch("aurarouter.fabric.get_provider", return_value=local_prov):
            res = fabric.execute(
                "coding",
                "hello",
                options={"complexity_score": 1, "deterministic": True},
            )

        assert res is not None
        recorded_attrs = {c.args[0]: c.args[1] for c in mock_span.set_attribute.call_args_list}
        assert recorded_attrs["gen_ai.request.model"] == "local_qwen"
        assert recorded_attrs["gen_ai.system"] == "ollama"
        assert recorded_attrs["aurarouter.routing.mode"] == "active"
        assert "aurarouter.ratelimit.headroom" in recorded_attrs
        assert recorded_attrs["aurarouter.projection.applied"] is False


# ---------------------------------------------------------------------------
# 5. MCP Tools, Server IETF HTTP Route & AuraRouterAPI Facade
# ---------------------------------------------------------------------------


class TestMcpServerAndPythonAPI:
    def test_mcp_tools_and_ietf_http_endpoint(self):
        cfg = _build_config(routing_mode="active")
        mcp = create_mcp_server(cfg)
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "aurarouter.ratelimit.status" in tool_names
        assert "aurarouter.routing.stack_status" in tool_names
        assert "aurarouter.routing.explain" in tool_names

        fabric = ComputeFabric(cfg)
        rl_json = json.loads(ratelimit_status(fabric, provider="anthropic", model_id="cloud_claude"))
        assert rl_json["provider"] == "anthropic"
        assert rl_json["state"] == "normal"
        assert "ietf_headers" in rl_json

        st_json = json.loads(routing_stack_status(fabric))
        assert st_json["mode"] == "active"
        assert "posterior_cache" in st_json

        exp_json = json.loads(routing_explain(fabric, role="coding", complexity_score=2))
        assert exp_json["role"] == "coding"
        assert exp_json["ranked_chain"][0] == "local_qwen"

        # Verify custom Starlette HTTP endpoint /api/ratelimit/status emits RFC 8941 headers
        app = mcp.sse_app()
        client = TestClient(app)
        resp = client.get("/api/ratelimit/status?provider=anthropic&model_id=cloud_claude&ietf=1")
        assert resp.status_code == 200
        assert "ratelimit" in {k.lower() for k in resp.headers.keys()}
        assert "ratelimit-policy" in {k.lower() for k in resp.headers.keys()}

    def test_aurarouter_api_v2_methods(self, tmp_path):
        import yaml

        cfg_data = _build_config(routing_mode="active").config
        cfg_file = tmp_path / "auraconfig.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))

        with AuraRouterAPI(
            APIConfig(config_path=str(cfg_file), enable_savings=False, enable_privacy=False)
        ) as api:
            rate_st = api.get_rate_status(provider="anthropic", model_id="cloud_claude")
            assert rate_st["state"] == "normal"
            assert rate_st["headroom_score"] == 1.0

            stack_st = api.get_routing_stack_status()
            assert stack_st["mode"] == "active"

            explanation = api.explain_route(role="coding", prompt="fix typo", complexity=1)
            assert explanation["ranked_chain"][0] == "local_qwen"

            big_msgs = [
                {"role": "user", "content": "step " + ("x " * 900)},
                {"role": "assistant", "content": "done " + ("y " * 900), "gist": "Did step 1."},
                {"role": "user", "content": "Now finish step 2."},
            ]
            proj = api.project_session_context(
                messages=big_msgs,
                model_id="local_qwen",
                context_limit=1024,
                failover_from="cloud_claude",
                failover_reason="429_rate_limit",
            )
            assert proj["was_projected"] is True
            assert "[FAILOVER CONTEXT" in proj["system_prompt"]
