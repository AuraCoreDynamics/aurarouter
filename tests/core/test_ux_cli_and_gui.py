"""Tests for CLI and GUI UX enhancements (doctor, multimodal CLI flags, ModelDialog, Settings, Routing)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aurarouter.api import APIConfig, AuraRouterAPI
from aurarouter.cli import main as cli_main
from aurarouter.routing import ReviewResult, TriageResult
from aurarouter.savings.models import GenerateResult


def test_cli_doctor_json_reports_vision_and_preflight_warnings(tmp_path, capsys):
    cfg_path = tmp_path / "auraconfig.yaml"
    cfg_path.write_text(
        "system:\n"
        "  vision_sovereignty_preflight: true\n"
        "models:\n"
        "  local_gguf_vlm:\n"
        "    provider: llamacpp\n"
        "    model_path: /models/llava.gguf\n"
        "    modalities: [text, vision]\n"
        "  cloud_anthropic:\n"
        "    provider: openapi\n"
        "    api_dialect: anthropic\n"
        "    endpoint: https://api.anthropic.com/v1\n"
        "    model_name: claude-sonnet-4-20250514\n"
        "    modalities: [text, vision]\n"
        "roles:\n"
        "  vision: [cloud_anthropic]\n",
        encoding="utf-8",
    )

    with patch("sys.argv", ["aurarouter", "--config", str(cfg_path), "--json", "doctor"]):
        cli_main()
    out = json.loads(capsys.readouterr().out)
    assert out["models_count"] == 2
    assert out["vision_sovereignty_preflight"] is True
    assert out["status"] == "warnings"
    assert any("mmproj_path" in w for w in out["warnings"])


def test_cli_model_add_and_run_with_image_and_schema(tmp_path, capsys):
    cfg_path = tmp_path / "auraconfig.yaml"
    cfg_path.write_text("models: {}\nroles: {}\n", encoding="utf-8")

    with patch("sys.argv", [
        "aurarouter",
        "--config", str(cfg_path),
        "model", "add", "claude_vlm",
        "--provider", "openapi",
        "--endpoint", "https://api.anthropic.com/v1",
        "--model-name", "claude-sonnet-4-20250514",
        "--api-dialect", "anthropic",
        "--modalities", "text,vision",
    ]):
        cli_main()
    capsys.readouterr()

    with AuraRouterAPI(APIConfig(config_path=str(cfg_path))) as api:
        m = api.get_model("claude_vlm")
        assert m is not None
        assert m.config.get("api_dialect") == "anthropic"
        assert m.config.get("modalities") == ["text", "vision"]

    img_file = tmp_path / "sample.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    with (
        patch("sys.argv", [
            "aurarouter",
            "--config", str(cfg_path),
            "--json",
            "run", "Extract text",
            "--image", str(img_file),
            "--response-schema", "vision_ocr",
        ]),
        patch("aurarouter.routing.analyze_intent", return_value=TriageResult(intent="VISION_OCR", complexity=3)),
        patch("aurarouter.routing.review_output", return_value=ReviewResult(verdict="PASS", feedback="ok", correction_hints=[])),
        patch("aurarouter.fabric.ComputeFabric.execute", return_value=GenerateResult(text='{"text":"ok"}')) as m_exec,
    ):
        cli_main()
    res_json = json.loads(capsys.readouterr().out)
    assert res_json["output"] == '{"text":"ok"}'
    opts = m_exec.call_args.kwargs.get("options", {})
    assert opts.get("images") == [str(img_file)]
    assert opts.get("response_schema") == "vision_ocr"


def test_gui_model_dialog_onnx_dialect_modalities_and_warnings():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QApplication = QtWidgets.QApplication
    from aurarouter.gui.model_dialog import ModelDialog

    app = QApplication.instance() or QApplication([])
    assert app is not None

    dlg = ModelDialog(
        model_id="onnx_cls",
        config={
            "provider": "onnx",
            "model_path": "/models/cls.onnx",
            "task": "vision_classification",
            "modalities": ["vision"],
        },
    )
    cfg = dlg.get_model_config()
    assert cfg["provider"] == "onnx"
    assert cfg["task"] == "vision_classification"
    assert cfg["modalities"] == ["vision"]
    assert not dlg._vision_warning_label.isHidden()

    dlg_anth = ModelDialog(
        model_id="claude_vlm",
        config={
            "provider": "openapi",
            "endpoint": "https://api.anthropic.com/v1",
            "model_name": "claude-sonnet-4-20250514",
            "api_dialect": "anthropic",
            "modalities": ["text", "vision"],
        },
    )
    cfg_anth = dlg_anth.get_model_config()
    assert cfg_anth["api_dialect"] == "anthropic"
    assert cfg_anth["modalities"] == ["text", "vision"]
    dlg.close()
    dlg_anth.close()


def test_gui_settings_and_routing_simulator_vision_ux(tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    QApplication = QtWidgets.QApplication
    from aurarouter.gui.routing_panel import RoutingPanel
    from aurarouter.gui.settings_panel import SettingsPanel

    app = QApplication.instance() or QApplication([])
    assert app is not None

    cfg_path = tmp_path / "auraconfig.yaml"
    cfg_path.write_text(
        "system:\n"
        "  sovereignty_enforcement: true\n"
        "  vision_sovereignty_preflight: true\n"
        "  vision_sovereignty_preflight_fail_closed: false\n"
        "models:\n"
        "  vlm1:\n"
        "    provider: ollama\n"
        "    model_name: qwen2.5vl\n"
        "    modalities: [text, vision]\n"
        "roles:\n"
        "  vision: [vlm1]\n",
        encoding="utf-8",
    )
    with AuraRouterAPI(APIConfig(config_path=str(cfg_path))) as api:
        settings = SettingsPanel(api)
        assert settings._vision_preflight_cb.isChecked() is True
        assert settings._vision_preflight_fail_closed_cb.isChecked() is False
        settings._vision_preflight_fail_closed_cb.setChecked(True)
        settings._collect_settings()
        assert api.get_sovereignty_config()["vision_sovereignty_preflight_fail_closed"] is True

        routing = RoutingPanel(api)
        routing._sim_input.setPlainText("Classify this document")
        routing._sim_vision_cb.setChecked(True)
        routing._on_simulate()
        sim_txt = routing._sim_output.toPlainText()
        assert "vision modality required" in sim_txt
        assert "vlm1" in sim_txt
        settings.close()
        routing.close()
