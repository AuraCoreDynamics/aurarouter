"""Unit tests for VisionLM / multimodal image routing, sovereignty, providers, and extraction."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.intent_registry import IntentRegistry
from aurarouter.media import (
    VISION_EXTRACTION_SCHEMA,
    VISION_OCR_SCHEMA,
    ImageInput,
    format_openai_content,
    format_openai_messages,
    normalize_images,
    resolve_vision_schema,
    strip_image_metadata,
)
from aurarouter.providers.base import GenerateResult
from aurarouter.providers.llamacpp_server import LlamaCppServerProvider
from aurarouter.providers.mcp_provider import McpProvider
from aurarouter.providers.ollama import OllamaProvider
from aurarouter.routing import analyze_intent
from aurarouter.routing_stack.context import CandidateEvalContext
from aurarouter.routing_stack.filters import CapabilityFilter
from aurarouter.runtime.server_process import ServerProcess
from aurarouter.sessions.manager import SessionManager
from aurarouter.sessions.projector import SessionProjector
from aurarouter.sessions.store import SessionStore


def _make_png(width: int = 2, height: int = 2, with_exif: bool = False) -> bytes:
    """Build a minimal valid PNG, optionally with tEXt and eXIf chunks."""
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_scanline = b"\x00" + b"\xff\x00\x00" * width
    idat = zlib.compress(raw_scanline * height)
    chunks = [_chunk(b"IHDR", ihdr)]
    if with_exif:
        chunks.append(_chunk(b"tEXt", b"Software\x00SecretCamGPS"))
        chunks.append(_chunk(b"eXIf", b"MM\x00*\x00\x00\x00\x08GPS"))
    chunks.append(_chunk(b"IDAT", idat))
    chunks.append(_chunk(b"IEND", b""))
    return sig + b"".join(chunks)


def _make_jpeg_with_exif() -> bytes:
    """Build a minimal JPEG byte stream containing an APP1 EXIF segment and SOF0 header."""
    soi = b"\xff\xd8"
    # APP1 EXIF segment
    app1_payload = b"Exif\x00\x00FakeGPSLatLonSecret"
    app1 = b"\xff\xe1" + struct.pack(">H", len(app1_payload) + 2) + app1_payload
    # SOF0 (baseline DCT) 16x32
    sof0_payload = struct.pack(">BHHB", 8, 32, 16, 3) + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    sof0 = b"\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload
    sos_and_eoi = b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x00\xff\xd9"
    return soi + app1 + sof0 + sos_and_eoi


class TestImageInputAndMedia:
    def test_from_bytes_path_data_uri_and_tokens(self, tmp_path: Path):
        png_bytes = _make_png(width=512, height=1024, with_exif=True)
        img_path = tmp_path / "sample.png"
        img_path.write_bytes(png_bytes)

        from_bytes = ImageInput.from_bytes(png_bytes)
        assert from_bytes.mime_type == "image/png"
        assert (from_bytes.width, from_bytes.height) == (512, 1024)
        assert from_bytes.estimate_tokens() == 85 + 170 * (1 * 2)

        from_path = ImageInput.from_path(img_path)
        assert from_path.source_path == str(img_path)
        assert from_path.data == from_bytes.data

        from_uri = ImageInput.from_data_uri(from_bytes.to_data_uri())
        assert from_uri.mime_type == "image/png"
        assert from_uri.data == from_bytes.data

        normalized = normalize_images([img_path, from_uri.to_dict(), from_bytes])
        assert len(normalized) == 3
        assert all(isinstance(item, ImageInput) for item in normalized)

    def test_strip_metadata_png_and_jpeg(self):
        png_dirty = _make_png(with_exif=True)
        png_clean, stripped_png = strip_image_metadata(png_dirty, "image/png")
        assert stripped_png is True
        assert b"tEXt" not in png_clean
        assert b"eXIf" not in png_clean
        assert png_clean.startswith(b"\x89PNG\r\n\x1a\n")

        jpg_dirty = _make_jpeg_with_exif()
        img = ImageInput.from_bytes(jpg_dirty)
        assert (img.width, img.height) == (16, 32)
        stripped_img = img.stripped_copy()
        assert stripped_img.exif_stripped is True
        raw_clean = stripped_img.raw_bytes()
        assert b"FakeGPSLatLonSecret" not in raw_clean
        assert b"\xff\xe1" not in raw_clean
        assert raw_clean.startswith(b"\xff\xd8")

    def test_resolve_vision_schema_presets_and_custom(self):
        assert resolve_vision_schema("vision_ocr") == VISION_OCR_SCHEMA
        assert resolve_vision_schema("VISION_EXTRACTION") == VISION_EXTRACTION_SCHEMA
        assert resolve_vision_schema(None, intent="VISION_OCR") == VISION_OCR_SCHEMA
        custom = {"type": "object", "properties": {"total": {"type": "number"}}}
        assert resolve_vision_schema(custom) == custom
        assert resolve_vision_schema(None, intent="DIRECT") is None


class TestVisionIntentsAndFilters:
    def test_intent_registry_has_vision_intents(self):
        reg = IntentRegistry()
        assert "VISION_OCR" in reg.get_intent_names()
        assert "VISION_EXTRACTION" in reg.get_intent_names()
        assert reg.resolve_role("VISION_OCR") == "vision"
        assert reg.resolve_role("VISION_EXTRACTION") == "vision"

    def test_analyze_intent_with_images_overrides_non_vision_fallback(self):
        mock_fabric = MagicMock()
        mock_fabric.execute.return_value = GenerateResult(
            text=json.dumps({"intent": "DIRECT", "complexity": 1})
        )
        res_ocr = analyze_intent(mock_fabric, "Please OCR and extract text from this scan", has_images=True)
        assert res_ocr.intent == "VISION_OCR"
        assert res_ocr.complexity >= 4

        res_ext = analyze_intent(mock_fabric, "What activities and objects are in this photo?", has_images=True)
        assert res_ext.intent == "VISION_EXTRACTION"
        assert res_ext.complexity >= 4

    def test_capability_filter_vision_aliases_and_mmproj(self):
        flt = CapabilityFilter()
        ctx = CandidateEvalContext(role="vision", required_capabilities=frozenset({"vision"}))

        assert flt.evaluate("vlm-alias", {"capabilities": ["vlm"]}, ctx).passed is True
        assert flt.evaluate("mm-tag", {"tags": ["multimodal"]}, ctx).passed is True
        assert flt.evaluate("gguf-mmproj", {"mmproj_path": "/models/mmproj.gguf"}, ctx).passed is True

        v_fail = flt.evaluate("text-only", {"capabilities": ["coding"]}, ctx)
        assert v_fail.passed is False
        assert "missing_capabilities:vision" in v_fail.reason


class TestFabricSovereigntyAndRouting:
    def _make_config(self) -> ConfigLoader:
        cfg = ConfigLoader(allow_missing=True)
        cfg.config = {
            "system": {"sovereignty_enforcement": True},
            "models": {
                "local_vlm": {
                    "provider": "ollama",
                    "model_name": "qwen2.5vl:7b",
                    "hosting_tier": "local",
                    "capabilities": ["vision"],
                    "supported_data_categories": ["*"],
                },
                "cloud_vlm": {
                    "provider": "openapi",
                    "model_name": "gpt-4o",
                    "endpoint": "https://api.openai.com/v1",
                    "api_key": "sk-test",
                    "hosting_tier": "cloud",
                    "capabilities": ["vision"],
                    "supported_data_categories": ["*"],
                },
                "local_text": {
                    "provider": "ollama",
                    "model_name": "llama3:8b",
                    "hosting_tier": "local",
                    "supported_data_categories": ["*"],
                },
            },
            "roles": {
                "vision": ["local_text", "local_vlm"],
                "coding": ["local_text"],
            },
        }
        return cfg

    def test_wildcard_category_and_auto_imagery_sovereignty_filtering(self):
        cfg = self._make_config()
        fabric = ComputeFabric(cfg)

        img = ImageInput.from_bytes(_make_jpeg_with_exif())

        # Local VLM passes Sovereignty ("*" accepts "IMAGERY") and receives intact image
        # while local_text is filtered out by required vision capability when images are present.
        trace = fabric.explain_route(
            "vision",
            images=[img.to_dict()],
            required_capabilities={"vision"},
        )
        assert trace.ranked_chain == ["local_vlm"]
        assert "local_text" in trace.pruned

    def test_cloud_dispatch_strips_exif_while_local_preserves(self):
        cfg = self._make_config()
        fabric = ComputeFabric(cfg)
        dirty_imgs = normalize_images([_make_jpeg_with_exif()])

        local_prepped = fabric._prepare_images_for_model(dirty_imgs, cfg.get_model_config("local_vlm"), False)
        assert local_prepped[0]["exif_stripped"] is False

        cloud_prepped = fabric._prepare_images_for_model(dirty_imgs, cfg.get_model_config("cloud_vlm"), False)
        assert cloud_prepped[0]["exif_stripped"] is True
        raw_cloud = base64.b64decode(cloud_prepped[0]["data"])
        assert b"FakeGPSLatLonSecret" not in raw_cloud

    def test_execute_forwards_images_and_resolved_schema(self):
        cfg = self._make_config()
        fabric = ComputeFabric(cfg)
        captured: dict = {}

        class _MockVLM:
            def generate_with_usage(self, prompt, json_mode=False, response_schema=None, images=None):
                captured["prompt"] = prompt
                captured["json_mode"] = json_mode
                captured["response_schema"] = response_schema
                captured["images"] = images
                return GenerateResult(text='{"summary": "desk"}', input_tokens=300, output_tokens=20)

            def generate(self, prompt, json_mode=False, response_schema=None, images=None):
                return self.generate_with_usage(prompt, json_mode, response_schema, images).text

        img_bytes = _make_png()
        with patch.object(fabric, "_get_provider", return_value=_MockVLM()):
            res = fabric.execute(
                "vision",
                "Extract objects and activities",
                json_mode=True,
                options={"response_schema": "vision_extraction"},
                images=[img_bytes],
            )
        assert res is not None
        assert res.model_id == "local_vlm"
        assert captured["response_schema"] == VISION_EXTRACTION_SCHEMA
        assert len(captured["images"]) == 1
        assert captured["images"][0]["mime_type"] == "image/png"


class TestProvidersAndRuntime:
    def test_server_process_includes_mmproj_flag(self, tmp_path: Path):
        bin_path = tmp_path / "llama-server"
        bin_path.write_text("", encoding="utf-8")
        proc = ServerProcess(
            model_path=tmp_path / "model.gguf",
            mmproj_path=tmp_path / "mmproj.gguf",
            binary_path=bin_path,
            port=8080,
        )
        cmd = proc._build_command()
        assert "--mmproj" in cmd
        idx = cmd.index("--mmproj")
        assert cmd[idx + 1].endswith("mmproj.gguf")

    def test_llamacpp_server_uses_chat_completions_when_images_present(self):
        prov = LlamaCppServerProvider({"endpoint": "http://localhost:8080", "model_name": "qwen2.5-vl"})
        img = ImageInput.from_bytes(_make_png()).to_dict()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Receipt total $12"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 150, "completion_tokens": 12},
        }
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.Client") as mock_client_cls:
            client = mock_client_cls.return_value.__enter__.return_value
            client.post.return_value = mock_resp
            res = prov.generate_with_usage(
                "Read receipt",
                response_schema=VISION_OCR_SCHEMA,
                images=[img],
            )
            assert res.text == "Receipt total $12"
            post_url = client.post.call_args[0][0]
            payload = client.post.call_args[1]["json"]
            assert post_url.endswith("/v1/chat/completions")
            content = payload["messages"][0]["content"]
            assert any(block.get("type") == "image_url" for block in content)
            assert payload["response_format"] == {"type": "json_object", "schema": VISION_OCR_SCHEMA}

    def test_ollama_provider_attaches_images_and_format_schema(self):
        prov = OllamaProvider({"endpoint": "http://localhost:11434", "model_name": "llava:latest"})
        img = ImageInput.from_bytes(_make_png()).to_dict()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok", "prompt_eval_count": 50, "eval_count": 5}
        mock_resp.raise_for_status.return_value = None

        with patch("httpx.Client") as mock_client_cls:
            client = mock_client_cls.return_value.__enter__.return_value
            client.post.return_value = mock_resp
            res = prov.generate_with_usage("Describe", response_schema=VISION_OCR_SCHEMA, images=[img])
            assert res.text == "ok"
            payload = client.post.call_args[1]["json"]
            assert payload["images"] == [img["data"]]
            assert payload["format"] == VISION_OCR_SCHEMA

    def test_openapi_formatting_helpers_for_images(self):
        img = ImageInput.from_bytes(_make_png()).to_dict()
        blocks = format_openai_content("Analyze scene", [img])
        assert isinstance(blocks, list)
        assert blocks[0] == {"type": "text", "text": "Analyze scene"}
        assert blocks[1]["type"] == "image_url"
        assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")

        msgs = format_openai_messages([{"role": "user", "content": "Look", "images": [img]}])
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][1]["type"] == "image_url"

    def test_mcp_provider_forwards_images_in_tool_call(self):
        prov = McpProvider({"mcp_endpoint": "http://localhost:9001", "model_name": "gemini-2.5-flash"})
        img = ImageInput.from_bytes(_make_png()).to_dict()
        with patch.object(prov, "_ensure_connected", return_value=None), patch.object(
            prov._client,
            "call_tool",
            return_value=json.dumps({"text": "saw cat", "input_tokens": 100, "output_tokens": 5}),
        ) as mock_call:
            res = prov.generate_with_usage("Describe", images=[img])
            assert res.text == "saw cat"
            tool_kwargs = mock_call.call_args[1]
            assert tool_kwargs["images"] == [img]


class TestSessionsBenchmarksAndAPI:
    def test_session_preserves_images_and_counts_image_tokens(self, tmp_path: Path):
        from aurarouter.sessions.projector import TargetContextBudget

        store = SessionStore(db_path=tmp_path / "sessions.db")
        mgr = SessionManager(store=store)
        sess = mgr.create_session(context_limit=8000)
        img = ImageInput.from_bytes(_make_png(width=512, height=512)).to_dict()

        sess = mgr.add_user_message(sess, "What is this?", images=[img])
        assert len(sess.history) == 1
        assert len(sess.history[0].images) == 1
        assert sess.token_stats.input_tokens >= 255

        projector = SessionProjector()
        projected = projector.project(
            session=sess,
            budget=TargetContextBudget(model_id="test", total_context_limit=8000),
        )
        assert projected.messages[0]["images"][0]["mime_type"] == "image/png"

    def test_benchmark_vision_prior(self):
        from aurarouter import benchmarks

        benchmarks._PRIOR_CACHE.clear()
        benchmarks._RECORD_CACHE.clear()
        fake_record = MagicMock()
        fake_record.scores = {
            "mmmu": 72.0,
            "ocrbench": 82.0,
            "docvqa": 90.0,
            "mathvista": 68.0,
            "vision_arena_elo": 80.0,
        }
        with patch("aurarouter.benchmarks.resolve_model_record", return_value=fake_record):
            prior = benchmarks.get_benchmark_prior("qwen2.5-vl-7b", band=1, vision=True)
        assert prior is not None
        alpha, beta = prior
        assert alpha + beta == pytest.approx(5.0, abs=1e-4)
        assert alpha > beta  # composite ~78.7% -> alpha > 3.5

    def test_route_task_mcp_passthrough(self):
        from aurarouter.mcp_tools import route_task

        mock_fabric = MagicMock()
        mock_fabric._config.get_role_chain.return_value = ["vlm1"]
        mock_fabric._config.config = {"system": {"sovereignty_enforcement": True}}
        mock_fabric.config.get_role_chain.return_value = []
        mock_fabric.get_max_review_iterations.return_value = 0
        mock_fabric.execute.return_value = GenerateResult(text='{"text": "hello"}', model_id="vlm1")

        img = ImageInput.from_bytes(_make_png()).to_dict()
        with patch("aurarouter.mcp_tools.analyze_intent") as mock_ai:
            from aurarouter.routing import TriageResult

            mock_ai.return_value = TriageResult(intent="VISION_OCR", complexity=4)
            out = route_task(mock_fabric, None, task="Extract text", images=[img], response_schema="vision_ocr")
            assert '"text": "hello"' in out
            mock_ai.assert_called_once()
            assert mock_ai.call_args[1]["has_images"] is True
            assert mock_fabric.execute.call_args[0][0] == "vision"
            assert mock_fabric.execute.call_args[1]["options"]["images"] is not None
            assert mock_fabric.execute.call_args[1]["options"]["response_schema"] == "vision_ocr"
