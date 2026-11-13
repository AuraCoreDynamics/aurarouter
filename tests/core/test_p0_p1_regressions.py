"""Regression tests for P0/P1 runtime, API, sovereignty, and routing fixes."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.circuit_breaker import CircuitBreaker
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import route_task
from aurarouter.routing import TriageResult, analyze_intent
from aurarouter.savings.models import GenerateResult, UsageRecord
from aurarouter.sovereignty.gate import SovereigntyGate, SovereigntyVerdict


def _make_cfg(tmp_path, extra: dict | None = None) -> ConfigLoader:
    cfg_file = tmp_path / "auraconfig.yaml"
    data = {
        "models": {
            "local_m": {
                "provider": "ollama",
                "model_name": "qwen2.5:7b",
                "endpoint": "http://localhost:11434/api/generate",
                "hosting_tier": "local",
            },
            "cloud_m": {
                "provider": "google",
                "model_name": "gemini-2.5-pro",
                "hosting_tier": "cloud",
                "cost_per_1m_input": 1.25,
                "cost_per_1m_output": 10.0,
            },
        },
        "roles": {
            "router": ["local_m"],
            "coding": ["cloud_m", "local_m"],
            "reasoning": ["local_m"],
            "reviewer": ["local_m"],
        },
    }
    if extra:
        data.update(extra)
    cfg = ConfigLoader(config_path=str(cfg_file), allow_missing=True)
    cfg.config = data
    cfg.save()
    return cfg


def test_config_add_triage_rule_persists_without_raw_attr(tmp_path):
    cfg = _make_cfg(tmp_path)
    rule = cfg.add_triage_rule(3, "coding", "Simple code rule")
    assert rule == {
        "max_complexity": 3,
        "preferred_role": "coding",
        "description": "Simple code rule",
    }
    reloaded = ConfigLoader(config_path=str(cfg.config_path))
    assert reloaded.get_savings_config()["triage"]["rules"] == [rule]


def test_server_budget_status_and_route_task_execution_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("aurarouter.savings.usage_store._DEFAULT_DB_PATH", tmp_path / "usage.db")
    monkeypatch.setattr("aurarouter.sovereignty.privacy._DEFAULT_DB_PATH", tmp_path / "usage.db")
    cfg = _make_cfg(
        tmp_path,
        extra={"savings": {"enabled": True, "budget": {"enabled": True, "daily_limit": 10.0}}},
    )
    from aurarouter.server import create_mcp_server

    mcp = create_mcp_server(cfg)
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}

    budget_json = tools["aurarouter.budget.status"].fn()
    parsed = json.loads(budget_json)
    assert parsed["enabled"] is True
    assert parsed["daily_limit"] == 10.0
    assert "daily_spend" in parsed

    with patch("aurarouter.server._route_task", return_value="ok") as mock_route:
        res = tools["route_task"].fn(task="hello", execution_mode="monologue")
        assert res == "ok"
        assert mock_route.call_args.kwargs["execution_mode"] == "monologue"


def test_fabric_speculative_and_monologue_unpack_augmented_prompt(tmp_path):
    cfg = _make_cfg(tmp_path)
    fabric = ComputeFabric(cfg)

    with patch("aurarouter.speculative.SpeculativeOrchestrator") as mock_spec_cls:
        mock_orch = MagicMock()
        mock_orch.execute_speculative = AsyncMock(return_value={"text": "spec-ok"})
        mock_spec_cls.return_value = mock_orch

        out = asyncio.run(fabric.execute_speculative("build X"))
        assert out == {"text": "spec-ok"}
        assert mock_orch.execute_speculative.call_args.kwargs["task"] == "build X"

    with patch("aurarouter.monologue.MonologueOrchestrator") as mock_mono_cls:
        mock_orch = MagicMock()
        mock_orch.reason = AsyncMock(return_value=SimpleNamespace(final_output="mono-ok"))
        mock_mono_cls.return_value = mock_orch

        out = asyncio.run(
            fabric.execute_monologue("think X", permissions={"allow": True})
        )
        assert out.final_output == "mono-ok"
        assert mock_orch.reason.call_args.kwargs["task"] == "think X"
        assert mock_orch.reason.call_args.kwargs["permissions"] == {"allow": True}


def test_fabric_execute_stream_monologue_error_does_not_hang(tmp_path):
    cfg = _make_cfg(tmp_path)
    fabric = ComputeFabric(cfg)

    async def _collect():
        with patch.object(
            fabric, "execute_monologue", side_effect=RuntimeError("boom")
        ):
            chunks = []
            async for chunk in fabric.execute_stream(
                "reasoning",
                "hi",
                options={"_execution_policy": {"mode": "monologue"}},
            ):
                chunks.append(chunk)
            return chunks

    chunks = asyncio.run(_collect())
    assert any("[MONOLOGUE ERROR] boom" in c for c in chunks)


def test_speculative_with_permissions_does_not_raise_nameerror(tmp_path):
    cfg = _make_cfg(tmp_path)
    fabric = ComputeFabric(cfg)
    from aurarouter.speculative import SpeculativeOrchestrator

    orch = SpeculativeOrchestrator(fabric, None, None, None)
    with patch.object(
        fabric, "execute", return_value=GenerateResult(text="draft", model_id="local_m")
    ):
        res = asyncio.run(
            orch.execute_speculative("write fn", permissions={"sandbox": True})
        )
    assert res is not None
    assert res["content"] == "draft"


def test_api_orchestrators_configs_rag_savings_and_local_only(tmp_path, monkeypatch):
    monkeypatch.setattr("aurarouter.savings.usage_store._DEFAULT_DB_PATH", tmp_path / "usage.db")
    monkeypatch.setattr("aurarouter.sovereignty.privacy._DEFAULT_DB_PATH", tmp_path / "usage.db")
    cfg = _make_cfg(
        tmp_path,
        extra={
            "system": {
                "monologue": {"enabled": True, "max_iterations": 7},
                "sovereignty_enforcement": True,
                "sovereignty_patterns": [{"name": "FOUO", "pattern": "FOUO"}],
                "rag_enrichment": True,
            },
            "xlm": {"endpoint": "http://xlm.internal:9002"},
        },
    )
    with AuraRouterAPI(
        APIConfig(config_path=str(cfg.config_path), enable_savings=True, enable_privacy=True)
    ) as api:
        assert api._get_speculative_orchestrator() is not None
        assert api._get_monologue_orchestrator() is not None

        mono_cfg = api.get_monologue_config()
        assert mono_cfg["enabled"] is True
        assert mono_cfg["max_iterations_default"] == 7

        sov_cfg = api.get_sovereignty_config()
        assert sov_cfg["enabled"] is True
        assert sov_cfg["custom_patterns_count"] == 1

        rag = api.get_rag_status()
        assert rag["enabled"] is True
        assert rag["endpoint_configured"] is True

        # local_only=True filters chain_override to local models on both sync and stream
        with patch.object(
            api._fabric, "execute", return_value=GenerateResult(text="local", model_id="local_m")
        ) as mock_exec:
            api.execute_direct("coding", "hi", local_only=True)
            assert mock_exec.call_args.kwargs["chain_override"] == ["local_m"]

        async def _fake_stream(*args, **kwargs):
            yield f"chain={kwargs.get('chain_override')}"

        with patch.object(api._fabric, "execute_stream", side_effect=_fake_stream):
            async def _drain():
                return [c async for c in api.execute_direct_stream("coding", "hi", local_only=True)]
            assert asyncio.run(_drain()) == ["chain=['local_m']"]

        # Real cloud_spend calculation in get_savings_summary
        api._usage_store.record(
            UsageRecord(
                timestamp="2026-11-12T10:00:00+00:00",
                model_id="cloud_m",
                provider="google",
                role="coding",
                intent="DIRECT",
                input_tokens=1_000_000,
                output_tokens=0,
                elapsed_s=0.5,
                success=True,
                is_cloud=True,
            )
        )
        summary = api.get_savings_summary()
        assert summary["cloud_spend"] == pytest.approx(1.25)


def test_sovereignty_gate_runs_builtin_privacy_auditor_when_categories_omitted(tmp_path):
    cfg = _make_cfg(tmp_path, extra={"system": {"sovereignty_enforcement": True}})
    gate = SovereigntyGate(cfg)
    res = gate.evaluate("My SSN is 123-45-6789")
    assert res.verdict == SovereigntyVerdict.SOVEREIGN
    assert "PII" in res.matched_patterns


def test_route_task_defers_standard_when_monologue_succeeds(tmp_path):
    cfg = _make_cfg(tmp_path)
    fabric = ComputeFabric(cfg)

    with patch(
        "aurarouter.mcp_tools.analyze_intent",
        return_value=TriageResult(intent="DIRECT", complexity=9),
    ), patch.object(
        fabric, "execute_monologue", AsyncMock(return_value=SimpleNamespace(final_output="mono"))
    ), patch.object(fabric, "execute") as mock_exec, patch.object(
        fabric, "get_max_review_iterations", return_value=0
    ):
        out = route_task(fabric, None, task="reason deeply", config=cfg, execution_mode="monologue")
        assert out == "mono"
        mock_exec.assert_not_called()


def test_benchmark_prior_band0_weighted_elo_and_null_scores(monkeypatch):
    from aurarouter import benchmarks

    fake_rec = SimpleNamespace(
        scores={"ifeval": 80.0, "arena_elo": 1200.0, "vision_arena_elo": None, "bbh": None}
    )
    monkeypatch.setattr(benchmarks, "resolve_model_record", lambda _m, model_name=None: fake_rec)

    # arena_elo=1200 normalizes to (1200-1000)/4 = 50.0; 80*0.6 + 50*0.4 = 68.0
    # mu = 0.68 -> alpha = 0.68 * 5.0 = 3.40
    prior_b0 = benchmarks.get_benchmark_prior("any-model", band=0)
    assert prior_b0 is not None
    assert prior_b0[0] == pytest.approx(3.40, abs=0.01)

    # Band 1 with bbh=None must not raise TypeError
    prior_b1 = benchmarks.get_benchmark_prior("any-model", band=1)
    assert prior_b1 is not None


def test_routing_analyze_intent_vision_direct_boost():
    fabric = MagicMock()
    fabric.execute.return_value = GenerateResult(
        text=json.dumps({"intent": "DIRECT", "complexity": 2})
    )
    res = analyze_intent(fabric, "describe this diagram", has_images=True)
    assert res.intent.startswith("VISION_")
    assert res.complexity >= 3


def test_circuit_breaker_peek_state_thread_safe():
    cb = CircuitBreaker("prov", failure_threshold=2, reset_timeout=60.0)
    cb.record_failure()
    cb.record_failure()
    state, fails = cb.peek_state()
    assert state == "open"
    assert fails == 2


def test_grid_mcp_client_pools_real_httpx_client():
    from aurarouter.mcp_client.client import GridMcpClient

    client = GridMcpClient("http://example.invalid:9000", timeout=5.0)
    with client._get_http_client() as c1:
        with client._get_http_client() as c2:
            assert c1 is c2
    client.close()
    assert client._http_client is None


def test_mcp_tools_pipeline_isolated_per_config_and_fabric(tmp_path):
    from aurarouter import mcp_tools as mt

    cfg_off = _make_cfg(tmp_path)
    cfg_on_1 = _make_cfg(tmp_path)
    cfg_on_1.config.setdefault("system", {})["analyzer_pipeline"] = {"enabled": True}
    cfg_on_2 = _make_cfg(tmp_path)
    cfg_on_2.config.setdefault("system", {})["analyzer_pipeline"] = {"enabled": True}

    f1 = ComputeFabric(cfg_on_1)
    f2 = ComputeFabric(cfg_on_2)

    assert mt._get_or_create_pipeline(cfg_off, f1, None) is None
    p1 = mt._get_or_create_pipeline(cfg_on_1, f1, None)
    p1_again = mt._get_or_create_pipeline(cfg_on_1, f1, None)
    p2 = mt._get_or_create_pipeline(cfg_on_2, f2, None)
    assert p1 is not None
    assert p1 is p1_again
    assert p2 is not None
    assert p1 is not p2
