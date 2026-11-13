"""Tests for unified modality assertions, ONNX vision support, custom intents,
and context/sovereignty progression across route_task paths.
"""

from __future__ import annotations

import json
import struct
import zlib
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.analyzer_protocol import AnalysisResult
from aurarouter.config import ConfigLoader
from aurarouter.media import get_model_modalities, model_supports_modality
from aurarouter.mcp_tools import route_task
from aurarouter.providers.base import BaseProvider
from aurarouter.providers.onnx import ONNXProvider
from aurarouter.routing import analyze_intent
from aurarouter.routing_stack.context import CandidateEvalContext
from aurarouter.routing_stack.filters import CapabilityFilter
from aurarouter.sovereignty.gate import SovereigntyGate


def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + (b"\x10\x80\xf0" * width)
    idat = zlib.compress(scanline * height)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _make_config(extra: dict | None = None) -> ConfigLoader:
    cfg = ConfigLoader(allow_missing=True)
    cfg.config = {
        "system": {
            "sovereignty_enforcement": False,
            "active_analyzer": "custom-mm",
        },
        "roles": {
            "router": ["local_router"],
            "coding": ["local_coder"],
            "vision": ["local_vlm", "cloud_moe"],
            "vision_classifier": ["local_onnx_vision", "local_vlm"],
        },
        "models": {
            "local_router": {
                "provider": "ollama",
                "endpoint": "http://127.0.0.1:11434/api/generate",
                "model_name": "qwen2.5:3b",
                "hosting_tier": "local",
                "modalities": ["text"],
                "tags": ["local"],
            },
            "local_coder": {
                "provider": "ollama",
                "endpoint": "http://127.0.0.1:11434/api/generate",
                "model_name": "qwen2.5-coder:7b",
                "hosting_tier": "local",
                "modalities": ["text"],
                "tags": ["local", "coding"],
                "supported_intents": ["SIMPLE_CODE", "DIRECT"],
            },
            "local_onnx_vision": {
                "provider": "onnx",
                "model_path": "/tmp/vision_cls.onnx",
                "hosting_tier": "local",
                "task": "vision_classification",
                "modalities": ["vision"],
                "tags": ["local", "vision"],
                "supported_intents": ["object_classification"],
                "labels": ["vehicle", "vessel", "aircraft"],
                "parameters": {"image_input_name": "pixel_values", "image_size": 16},
            },
            "local_vlm": {
                "provider": "ollama",
                "endpoint": "http://127.0.0.1:11434/api/generate",
                "model_name": "qwen2.5vl:7b",
                "hosting_tier": "local",
                "modalities": ["text", "vision"],
                "tags": ["local", "vision", "vlm"],
                "supported_intents": ["VISION_OCR", "VISION_EXTRACTION", "object_classification", "vision_summarization"],
            },
            "cloud_moe": {
                "provider": "openapi",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "model_name": "gpt-4o",
                "hosting_tier": "cloud",
                "modalities": ["text", "vision"],
                "tags": ["cloud", "vision", "multimodal"],
                "supported_intents": ["vision_summarization", "VISION_EXTRACTION"],
            },
        },
        "catalog": {
            "custom-mm": {
                "kind": "analyzer",
                "analyzer_kind": "intent_triage",
                "role_bindings": {
                    "DIRECT": "coding",
                    "SIMPLE_CODE": "coding",
                    "VISION_OCR": "vision",
                    "VISION_EXTRACTION": "vision",
                    "object_classification": "vision_classifier",
                    "vision_summarization": "vision",
                },
            }
        },
    }
    if extra:
        for k, v in extra.items():
            if k == "system" and isinstance(v, dict):
                cfg.config["system"].update(v)
            else:
                cfg.config[k] = v
    return cfg


def test_unified_modality_resolution_and_capability_filter():
    cfg = _make_config()
    onnx_cfg = cfg.get_model_config("local_onnx_vision")
    coder_cfg = cfg.get_model_config("local_coder")
    vlm_cfg = cfg.get_model_config("local_vlm")

    assert get_model_modalities(onnx_cfg) == frozenset({"vision"})
    assert model_supports_modality(onnx_cfg, "vision")
    assert not model_supports_modality(onnx_cfg, "text")

    assert get_model_modalities(coder_cfg) == frozenset({"text"})
    assert not model_supports_modality(coder_cfg, "vision")

    assert get_model_modalities(vlm_cfg) == frozenset({"text", "vision"})

    filt = CapabilityFilter()
    ctx = CandidateEvalContext(
        role="vision",
        intent="object_classification",
        complexity_score=4,
        required_capabilities=frozenset({"vision"}),
    )
    assert filt.evaluate("local_onnx_vision", onnx_cfg, ctx).passed is True
    assert filt.evaluate("local_coder", coder_cfg, ctx).passed is False


def test_base_provider_rejects_images_when_subclass_omits_images_param():
    class TextOnlyLegacyProvider(BaseProvider):
        def generate(self, prompt: str, json_mode: bool = False) -> str:
            return "text-only"

    prov = TextOnlyLegacyProvider({"provider": "custom", "modalities": ["text"]})
    assert prov.supports_modality("text")
    assert not prov.supports_modality("vision")

    with pytest.raises(ValueError, match="does not accept image inputs"):
        prov.generate_with_usage("describe", images=[{"data": "AA==", "mime_type": "image/png"}])


def test_onnx_provider_rejects_images_on_non_vision_model():
    text_onnx = ONNXProvider({
        "provider": "onnx",
        "model_path": "/tmp/text.onnx",
        "task": "classification",
        "modalities": ["text"],
    })
    with pytest.raises(ValueError, match="does not declare 'vision' modality"):
        text_onnx.generate_with_usage("hi", images=[{"data": "AA==", "mime_type": "image/png"}])


def test_onnx_provider_accepts_images_for_vision_classification():
    import sys
    import types
    import numpy as np

    mock_session = MagicMock()
    mock_session.run.return_value = [np.array([[0.1, 2.5, 0.2]], dtype=np.float32)]

    vision_onnx = ONNXProvider({
        "provider": "onnx",
        "model_path": "/tmp/vision_cls.onnx",
        "task": "vision_classification",
        "modalities": ["vision"],
        "labels": ["vehicle", "vessel", "aircraft"],
        "parameters": {"image_input_name": "pixel_values", "image_size": 16},
    })
    assert vision_onnx.supports_modality("vision")

    png_bytes = _make_png_bytes(8, 8)

    class _FakeImg:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def convert(self, mode):
            return self

        def resize(self, size):
            w, h = size
            return [[[0, 0, 0] for _ in range(w)] for _ in range(h)]

    fake_image_mod = types.ModuleType("PIL.Image")
    fake_image_mod.open = lambda *_a, **_kw: _FakeImg()
    fake_pil_mod = types.ModuleType("PIL")
    fake_pil_mod.Image = fake_image_mod
    fake_pil = {"PIL": fake_pil_mod, "PIL.Image": fake_image_mod}

    with patch("aurarouter.providers.onnx._session_cache.get_session", return_value=mock_session), \
         patch.dict(sys.modules, fake_pil):
        res = vision_onnx.generate_with_usage(
            "classify object",
            images=[{"data": png_bytes, "mime_type": "image/png"}],
        )

    assert res.text == "vessel"
    assert res.metadata["confidence"] > 0.7
    call_inputs = mock_session.run.call_args[0][1]
    assert "pixel_values" in call_inputs
    assert call_inputs["pixel_values"].shape == (1, 3, 16, 16)


def test_analyze_intent_receives_context():
    fabric = MagicMock()
    fabric.execute.return_value = MagicMock(
        text=json.dumps({"intent": "object_classification", "complexity": 3})
    )
    res = analyze_intent(
        fabric,
        "Process this frame",
        has_images=True,
        context="KNOWN_CLASSIFICATION: maritime_vessel",
    )
    assert res.intent == "object_classification"
    prompt_sent = fabric.execute.call_args[0][1]
    assert "KNOWN_CLASSIFICATION: maritime_vessel" in prompt_sent


def test_route_task_pipeline_forwards_images_schema_and_context():
    cfg = _make_config({
        "system": {"analyzer_pipeline": {"enabled": True}, "sovereignty_enforcement": False}
    })
    fabric = MagicMock()
    fabric.config = cfg
    fabric._config = cfg
    fabric.get_max_review_iterations.return_value = 0
    fabric.routing_stack = MagicMock(mode="off")
    fabric._sovereignty_gate = None
    fabric.execute.return_value = MagicMock(text='{"summary": "vessel"}')

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = AnalysisResult(
        intent="vision_summarization",
        confidence=0.92,
        complexity_score=6,
        analyzer_id="mock-pipe",
    )
    fake_pipeline.build_routing_context.return_value = MagicMock(
        strategy="vector",
        confidence_score=0.92,
        complexity_score=6,
        selected_route="vision",
        analyzer_chain=["mock-pipe"],
        intent="vision_summarization",
        hard_routed=False,
        simulated_cost_avoided=0.0,
    )

    png_bytes = _make_png_bytes()
    with patch("aurarouter.mcp_tools._get_or_create_pipeline", return_value=fake_pipeline):
        out = route_task(
            fabric,
            None,
            task="Summarize the scene",
            context="KNOWN_CLASSIFICATION: maritime",
            config=cfg,
            images=[png_bytes],
            response_schema="vision_extraction",
        )

    assert "vessel" in out
    fake_pipeline.run.assert_called_once_with("Summarize the scene", context="KNOWN_CLASSIFICATION: maritime")
    exec_call = fabric.execute.call_args
    assert exec_call[0][0] == "vision"
    assert "KNOWN_CLASSIFICATION: maritime" in exec_call[0][1]
    assert exec_call.kwargs["options"]["images"] == [png_bytes]
    assert exec_call.kwargs["options"]["response_schema"] == "vision_extraction"


def test_sovereignty_prevents_leaking_privacy_context_to_remote_cloud_analyzer():
    cfg = _make_config({
        "system": {
            "active_analyzer": "remote-cloud-analyzer",
            "sovereignty_enforcement": True,
            "sovereignty_patterns": [
                {"name": "privacy_tag", "pattern": r"(?i)PRIVACY:\s*PII", "category": "privacy"}
            ],
        },
        "catalog": {
            "remote-cloud-analyzer": {
                "kind": "analyzer",
                "mcp_endpoint": "https://remote-analyzer.example.com/mcp",
                "mcp_tool_name": "analyze",
            },
        },
    })
    fabric = MagicMock()
    fabric.config = cfg
    fabric._config = cfg
    fabric.get_max_review_iterations.return_value = 0
    fabric.routing_stack = MagicMock(mode="off")
    fabric._sovereignty_gate = SovereigntyGate(cfg)
    fabric.execute.return_value = MagicMock(text="local-only-result")

    with patch("aurarouter.mcp_tools._call_remote_analyzer") as mock_remote, \
         patch("aurarouter.mcp_tools.analyze_intent") as mock_intent:
        mock_intent.return_value = MagicMock(intent="VISION_OCR", complexity=4)
        out = route_task(
            fabric,
            None,
            task="Extract text from this badge",
            context="PRIVACY: PII; CLASSIFICATION: RESTRICTED",
            config=cfg,
            images=[_make_png_bytes()],
        )

    assert out == "local-only-result"
    mock_remote.assert_not_called()


def test_speculative_and_monologue_forward_images_and_fallback_when_no_vision_models():
    import asyncio
    from aurarouter.monologue import MonologueOrchestrator
    from aurarouter.speculative import SpeculativeOrchestrator

    # 1. Text-only config -> speculative returns None (falls back to standard)
    text_only_cfg = _make_config({
        "roles": {
            "coding": ["local_coder"],
            "reasoning": ["local_router"],
            "reviewer": ["local_coder"],
            "vision": [],
        }
    })
    fabric = MagicMock()
    fabric.config = text_only_cfg
    fabric._config = text_only_cfg

    spec = SpeculativeOrchestrator(fabric, None, None, None)
    assert asyncio.run(spec.execute_speculative("describe", images=[_make_png_bytes()])) is None

    mono = MonologueOrchestrator(fabric, None, None, None)
    mono_res = asyncio.run(mono.reason("describe", images=[_make_png_bytes()]))
    assert mono_res.convergence_reason == "no_models_available"

    # 2. Vision-capable config -> monologue forwards images & response_schema
    vision_cfg = _make_config({
        "roles": {
            "coding": ["local_vlm"],
            "reasoning": ["local_vlm"],
            "reviewer": ["cloud_moe"],
            "vision": ["local_vlm"],
        }
    })
    fabric2 = MagicMock()
    fabric2.config = vision_cfg
    fabric2._config = vision_cfg
    fabric2.execute.return_value = MagicMock(text='{"score": 0.95, "feedback": "good"}')

    mono2 = MonologueOrchestrator(fabric2, None, None, None)
    mono2_res = asyncio.run(
        mono2.reason("describe", images=[_make_png_bytes()], response_schema="vision_extraction", max_iterations=1)
    )
    assert mono2_res.convergence_reason == "confidence_threshold"
    first_call_opts = fabric2.execute.call_args_list[0].kwargs["options"]
    assert "images" in first_call_opts
    assert first_call_opts["response_schema"] == "vision_extraction"


def test_review_output_skips_text_only_reviewer_and_forwards_images_to_vision_reviewer():
    from aurarouter.routing import review_output

    # Text-only reviewer -> skipped with PASS
    text_rev_cfg = _make_config({"roles": {"reviewer": ["local_coder"]}})
    fabric_text = MagicMock()
    fabric_text.config = text_rev_cfg
    fabric_text._config = text_rev_cfg
    res_skip = review_output(fabric_text, "ocr", "extracted", images=[_make_png_bytes()])
    assert res_skip.verdict == "PASS"
    fabric_text.execute.assert_not_called()

    # Vision reviewer -> receives images
    vis_rev_cfg = _make_config({"roles": {"reviewer": ["local_vlm"]}})
    fabric_vis = MagicMock()
    fabric_vis.config = vis_rev_cfg
    fabric_vis._config = vis_rev_cfg
    fabric_vis.execute.return_value = MagicMock(
        text=json.dumps({"verdict": "PASS", "feedback": "verified", "correction_hints": []})
    )
    res_ok = review_output(fabric_vis, "ocr", "extracted", images=[_make_png_bytes()])
    assert res_ok.verdict == "PASS"
    assert "images" in fabric_vis.execute.call_args.kwargs["options"]


def test_openapi_provider_anthropic_and_gemini_multimodal_dialects():
    from aurarouter.providers.openapi import OpenAPIProvider

    png = _make_png_bytes()

    # Anthropic /v1/messages
    anth = OpenAPIProvider({
        "provider": "openapi",
        "endpoint": "https://api.anthropic.com/v1",
        "model_name": "claude-3-5-sonnet-latest",
        "api_key": "sk-ant-test",
    })
    mock_resp_ant = MagicMock()
    mock_resp_ant.status_code = 200
    mock_resp_ant.headers = {}
    mock_resp_ant.json.return_value = {
        "content": [{"type": "text", "text": "anthropic-vision-ok"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_client_ant = MagicMock()
    mock_client_ant.__enter__ = MagicMock(return_value=mock_client_ant)
    mock_client_ant.__exit__ = MagicMock(return_value=False)
    mock_client_ant.post.return_value = mock_resp_ant

    with patch("httpx.Client", return_value=mock_client_ant):
        out_ant = anth.generate_with_usage("ocr this", images=[{"data": png, "mime_type": "image/png"}])
    assert out_ant.text == "anthropic-vision-ok"
    sent_url, sent_kw = mock_client_ant.post.call_args[0][0], mock_client_ant.post.call_args[1]
    assert sent_url.endswith("/v1/messages")
    assert sent_kw["headers"]["x-api-key"] == "sk-ant-test"
    assert sent_kw["json"]["messages"][0]["content"][0]["type"] == "image"

    # Gemini :generateContent
    gem = OpenAPIProvider({
        "provider": "openapi",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "model_name": "gemini-2.0-flash",
        "api_key": "AIza-test",
    })
    mock_resp_gem = MagicMock()
    mock_resp_gem.status_code = 200
    mock_resp_gem.headers = {}
    mock_resp_gem.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "gemini-vision-ok"}]}}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4},
    }
    mock_client_gem = MagicMock()
    mock_client_gem.__enter__ = MagicMock(return_value=mock_client_gem)
    mock_client_gem.__exit__ = MagicMock(return_value=False)
    mock_client_gem.post.return_value = mock_resp_gem

    with patch("httpx.Client", return_value=mock_client_gem):
        out_gem = gem.generate_with_usage("ocr this", images=[{"data": png, "mime_type": "image/png"}])
    assert out_gem.text == "gemini-vision-ok"
    gem_url, gem_kw = mock_client_gem.post.call_args[0][0], mock_client_gem.post.call_args[1]
    assert ":generateContent" in gem_url
    assert gem_kw["headers"]["x-goog-api-key"] == "AIza-test"
    assert "inline_data" in gem_kw["json"]["contents"][0]["parts"][0]


def test_vision_sovereignty_preflight_and_config_diagnostics():
    from aurarouter.fabric import ComputeFabric
    from aurarouter.media import validate_vision_model_config
    from aurarouter.sovereignty.gate import SovereigntyVerdict

    # 1. Config diagnostics
    w1 = validate_vision_model_config({"provider": "llamacpp", "modalities": ["vision"]})
    assert any("mmproj_path" in w for w in w1)
    w2 = validate_vision_model_config({"provider": "onnx", "modalities": ["vision"]})
    assert any("CPUExecutionProvider" in w for w in w2)

    # 2. Preflight OCR detects SECRET banner inside untagged image and forces SOVEREIGN
    cfg = _make_config({
        "system": {
            "sovereignty_enforcement": True,
            "vision_sovereignty_preflight": True,
            "sovereignty_patterns": [
                {"name": "secret_banner", "pattern": r"(?i)TOP\s+SECRET", "category": "classification"}
            ],
        },
    })
    sov_gate = SovereigntyGate(cfg)
    fabric = ComputeFabric(cfg, sovereignty_gate=sov_gate)
    with patch.object(fabric, "execute", return_value=MagicMock(text="TOP SECRET // NOFORN")):
        res = fabric._run_vision_sovereignty_preflight(
            [],
            sov_gate.evaluate("summarize this image"),
            cfg.config["system"],
        )
    assert res.verdict == SovereigntyVerdict.SOVEREIGN
    assert "secret_banner" in res.matched_patterns

    # 3. Fail-closed when no local vision model exists
    cfg_no_local = _make_config({
        "roles": {"vision": ["cloud_moe"], "vision_classifier": [], "reasoning": []},
        "system": {
            "sovereignty_enforcement": True,
            "vision_sovereignty_preflight": True,
            "vision_sovereignty_preflight_fail_closed": True,
        },
    })
    fabric_nl = ComputeFabric(cfg_no_local, sovereignty_gate=SovereigntyGate(cfg_no_local))
    res_nl = fabric_nl._run_vision_sovereignty_preflight(
        [],
        SovereigntyGate(cfg_no_local).evaluate("summarize this image"),
        cfg_no_local.config["system"],
    )
    assert res_nl.verdict == SovereigntyVerdict.SOVEREIGN


def test_openapi_history_and_streaming_anthropic_gemini_dialects():
    import base64
    from aurarouter.providers.openapi import OpenAPIProvider

    png = base64.b64encode(_make_png_bytes()).decode("ascii")

    # Anthropic generate_with_history + generate_stream_sync
    anth = OpenAPIProvider({
        "provider": "openapi",
        "api_dialect": "anthropic",
        "endpoint": "https://api.anthropic.com/v1",
        "model_name": "claude-sonnet-4-20250514",
        "api_key": "sk-ant-test",
    })
    hist_resp = MagicMock()
    hist_resp.raise_for_status = MagicMock()
    hist_resp.headers = {}
    hist_resp.json.return_value = {
        "content": [{"type": "text", "text": "anth-history-ok"}],
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "stop_reason": "end_turn",
    }
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = hist_resp

    with patch("httpx.Client", return_value=mock_client):
        res = anth.generate_with_history(
            [{"role": "user", "content": "describe", "images": [{"data": png, "mime_type": "image/png"}]}],
            system_prompt="Be concise.",
        )
    assert res.text == "anth-history-ok"
    post_url, post_kw = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
    assert post_url.endswith("/messages")
    assert post_kw["json"]["system"] == "Be concise."
    assert post_kw["json"]["messages"][0]["content"][0]["type"] == "image"

    # Anthropic SSE stream sync
    stream_resp = MagicMock()
    stream_resp.status_code = 200
    stream_resp.headers = {}
    stream_resp.raise_for_status = MagicMock()
    stream_resp.iter_lines.return_value = [
        'data: {"type": "message_start", "message": {"usage": {"input_tokens": 9}}}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello "}}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "claude"}}',
        'data: {"type": "message_delta", "usage": {"output_tokens": 2}}',
    ]
    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=stream_resp)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    mock_client.stream.return_value = stream_ctx

    with patch("httpx.Client", return_value=mock_client):
        tokens = list(anth.generate_stream_sync("hi"))
    assert "".join(tokens) == "hello claude"

    # Gemini generate_with_history + generate_stream_sync
    gem = OpenAPIProvider({
        "provider": "openapi",
        "api_dialect": "gemini",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "model_name": "gemini-2.5-flash",
        "api_key": "AIza-test",
    })
    gem_hist_resp = MagicMock()
    gem_hist_resp.raise_for_status = MagicMock()
    gem_hist_resp.headers = {}
    gem_hist_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "gem-history-ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3},
    }
    mock_client.post.return_value = gem_hist_resp
    with patch("httpx.Client", return_value=mock_client):
        gem_res = gem.generate_with_history(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
            system_prompt="System rule",
        )
    assert gem_res.text == "gem-history-ok"
    gem_url, gem_kw = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
    assert ":generateContent" in gem_url
    assert gem_kw["json"]["systemInstruction"]["parts"][0]["text"] == "System rule"
    assert gem_kw["json"]["contents"][1]["role"] == "model"


def test_api_execute_task_context_images_and_execution_mode_parity(tmp_path):
    from aurarouter.api import APIConfig, AuraRouterAPI
    from aurarouter.routing import ReviewResult, TriageResult
    from aurarouter.savings.models import GenerateResult

    cfg_path = tmp_path / "auraconfig.yaml"
    cfg_path.write_text(
        "models:\n"
        "  local_vlm:\n"
        "    provider: ollama\n"
        "    model_name: qwen2.5vl\n"
        "    modalities: [text, vision]\n"
        "    hosting_tier: on-prem\n"
        "roles:\n"
        "  router: [local_vlm]\n"
        "  coding: [local_vlm]\n"
        "  vision: [local_vlm]\n"
        "  reviewer: [local_vlm]\n",
        encoding="utf-8",
    )
    api = AuraRouterAPI(APIConfig(config_path=str(cfg_path)))
    try:
        with (
            patch("aurarouter.routing.analyze_intent", return_value=TriageResult(intent="VISION_OCR", complexity=3)) as m_intent,
            patch.object(api._fabric, "execute", return_value=GenerateResult(text="ocr-out")) as m_exec,
        ):
            res = api.execute_task("Extract text", context="Classification: CUI", images=["a.png"])
        assert res.output == "ocr-out"
        assert m_intent.call_args.kwargs.get("context") == "Classification: CUI"
        assert "Classification: CUI" in m_exec.call_args[0][1]

        with (
            patch("aurarouter.routing.analyze_intent", return_value=TriageResult(intent="SIMPLE_CODE", complexity=5)),
            patch("aurarouter.routing.generate_plan", return_value=["step1"]),
            patch("aurarouter.routing.review_output", return_value=ReviewResult(verdict="PASS", feedback="ok", correction_hints=[])) as m_rev,
            patch.object(api._fabric, "execute", return_value=GenerateResult(text="done")),
        ):
            api.execute_task("Describe image", images=["a.png"])
        assert m_rev.call_args.kwargs.get("images") == ["a.png"]

        async def _fake_spec(task, context=None, **kw):
            return GenerateResult(text="spec-out")

        with (
            patch("aurarouter.routing.analyze_intent", return_value=TriageResult(intent="SIMPLE_CODE", complexity=7)),
            patch("aurarouter.routing.review_output", return_value=ReviewResult(verdict="PASS", feedback="ok", correction_hints=[])),
            patch.object(api._fabric, "execute_speculative", side_effect=_fake_spec) as m_spec,
        ):
            spec_res = api.execute_task("Solve", execution_mode="speculative", images=["a.png"])
        assert spec_res.output == "spec-out"
        assert m_spec.call_args.kwargs.get("images") == ["a.png"]
    finally:
        api.close()


def test_fabric_session_and_stream_run_vision_preflight_and_mcp_sync_images():
    import asyncio
    from aurarouter.fabric import ComputeFabric
    from aurarouter.providers.mcp_provider import McpProvider
    from aurarouter.savings.models import GenerateResult
    from aurarouter.sovereignty.gate import SovereigntyResult, SovereigntyVerdict

    cfg = _make_config({
        "system": {
            "sovereignty_enforcement": True,
            "vision_sovereignty_preflight": True,
        },
    })
    import base64
    sov_gate = SovereigntyGate(cfg)
    fabric = ComputeFabric(cfg, sovereignty_gate=sov_gate)
    png = base64.b64encode(_make_png_bytes()).decode("ascii")

    with patch.object(
        fabric,
        "_run_vision_sovereignty_preflight",
        return_value=SovereigntyResult(verdict=SovereigntyVerdict.SOVEREIGN, reason="local"),
    ) as m_pre:
        with patch.object(fabric, "_get_provider") as m_get_prov:
            prov = MagicMock()
            prov.generate_with_history.return_value = GenerateResult(text="sess-ok", model_id="local_vlm")
            m_get_prov.return_value = prov
            out = fabric.execute_session(
                "vision",
                [{"role": "user", "content": "hi", "images": [{"data": png, "mime_type": "image/png"}]}],
            )
            assert out.text == "sess-ok"
            assert m_pre.call_count == 1

            async def _fake_stream(*a, **kw):
                yield "tok"

            prov.generate_stream.side_effect = _fake_stream

            async def _collect():
                return [
                    t
                    async for t in fabric.execute_stream(
                        "vision",
                        "hi",
                        images=[{"data": png, "mime_type": "image/png"}],
                    )
                ]

            toks = asyncio.run(_collect())
            assert toks == ["tok"]
            assert m_pre.call_count == 2

    mcp = McpProvider({"model_name": "m", "mcp_endpoint": "http://localhost:9999"})
    with patch.object(
        mcp,
        "_generate_with_usage_detail",
        return_value=(GenerateResult(text="mcp-img-ok", model_id="m"), True),
    ) as m_det:
        chunks = list(mcp.generate_stream_sync("ocr", images=[{"data": png, "mime_type": "image/png"}]))
    assert chunks == ["mcp-img-ok"]
    assert m_det.call_args.kwargs.get("images") is not None
