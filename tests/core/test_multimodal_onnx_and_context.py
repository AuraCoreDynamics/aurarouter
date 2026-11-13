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
