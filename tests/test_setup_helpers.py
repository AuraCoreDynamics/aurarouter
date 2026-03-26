"""Tests for setup wizard helper functions."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.gui.help.setup_helpers import (
    HardwareInfo,
    detect_cloud_providers,
    detect_hardware,
    detect_ollama,
    get_recommended_models,
    pip_install_sync,
    suggest_cuda_sidecar,
)


# ---------------------------------------------------------------------------
# TestDetectHardware
# ---------------------------------------------------------------------------


class TestDetectHardware:
    @patch("aurarouter.gui.help.setup_helpers.importlib.metadata.distributions", return_value=[])
    @patch("aurarouter.gui.help.setup_helpers.subprocess.run", side_effect=FileNotFoundError)
    @patch("aurarouter.gui.help.setup_helpers.os.cpu_count", return_value=4)
    def test_no_gpu(self, _cpu, _run, _dists):
        """nvidia-smi fails and _detect_vram_bytes returns 0 => no GPU."""
        with patch.dict(
            "sys.modules",
            {"aurarouter.tuning": MagicMock(_detect_vram_bytes=lambda: 0)},
        ):
            hw = detect_hardware()
        assert hw.vram_mb == 0
        assert hw.has_nvidia is False
        assert hw.cpu_cores == 4

    @patch("aurarouter.gui.help.setup_helpers.importlib.metadata.distributions", return_value=[])
    @patch("aurarouter.gui.help.setup_helpers.os.cpu_count", return_value=16)
    def test_nvidia_gpu(self, _cpu, _dists):
        """nvidia-smi succeeds => parses name, vram, has_nvidia=True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4090, 24564, 550.54\n"

        # nvidia-smi for CUDA detection + nvcc
        mock_driver = MagicMock(returncode=0, stdout="550.54\n")
        mock_nvcc = MagicMock(returncode=0, stdout="Cuda compilation tools, release 12.4, V12.4.131\n")

        def side_effect(cmd, **kw):
            if "--query-gpu=name,memory.total,driver_version" in cmd:
                return mock_result
            if "--query-gpu=driver_version" in cmd:
                return mock_driver
            if cmd[0] == "nvcc":
                return mock_nvcc
            return MagicMock(returncode=1, stdout="")

        with patch("aurarouter.gui.help.setup_helpers.subprocess.run", side_effect=side_effect):
            hw = detect_hardware()

        assert hw.gpu_name == "NVIDIA GeForce RTX 4090"
        assert hw.vram_mb == 24564
        assert hw.has_nvidia is True
        assert hw.cuda_version == "12.4"
        assert hw.cpu_cores == 16

    @patch("aurarouter.gui.help.setup_helpers.subprocess.run", side_effect=FileNotFoundError)
    @patch("aurarouter.gui.help.setup_helpers.os.cpu_count", return_value=8)
    def test_cuda_sidecars_detected(self, _cpu, _run):
        """importlib.metadata finds aurarouter-cuda13 sidecar."""
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "aurarouter-cuda13"}
        mock_other = MagicMock()
        mock_other.metadata = {"Name": "requests"}

        with patch(
            "aurarouter.gui.help.setup_helpers.importlib.metadata.distributions",
            return_value=[mock_dist, mock_other],
        ):
            with patch.dict(
                "sys.modules",
                {"aurarouter.tuning": MagicMock(_detect_vram_bytes=lambda: 0)},
            ):
                hw = detect_hardware()

        assert hw.installed_sidecars == ["aurarouter-cuda13"]

    @patch("aurarouter.gui.help.setup_helpers.importlib.metadata.distributions", return_value=[])
    @patch("aurarouter.gui.help.setup_helpers.subprocess.run", side_effect=FileNotFoundError)
    @patch("aurarouter.gui.help.setup_helpers.os.cpu_count", return_value=12)
    def test_cpu_cores(self, mock_cpu, _run, _dists):
        """os.cpu_count() is called and its value used."""
        with patch.dict(
            "sys.modules",
            {"aurarouter.tuning": MagicMock(_detect_vram_bytes=lambda: 0)},
        ):
            hw = detect_hardware()
        mock_cpu.assert_called()
        assert hw.cpu_cores == 12


# ---------------------------------------------------------------------------
# TestDetectOllama
# ---------------------------------------------------------------------------


class TestDetectOllama:
    def test_ollama_available(self):
        """httpx.get returns tags JSON => available=True, models populated."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:7b", "size": 4700000000},
                {"name": "llama3:8b", "size": 0},
            ]
        }
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = detect_ollama()

        assert result["available"] is True
        assert result["endpoint"] == "http://localhost:11434"
        assert len(result["models"]) == 2
        assert result["models"][0]["name"] == "qwen2.5-coder:7b"
        assert "4.4" in result["models"][0]["size"]  # 4700000000 / 1024^3 ~ 4.4
        assert result["models"][1]["size"] == "unknown"  # size=0

    def test_ollama_not_available(self):
        """httpx.get raises => available=False, empty models."""
        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = Exception("Connection refused")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch(
                "aurarouter.gui.help.setup_helpers.subprocess.run",
                side_effect=FileNotFoundError,
            ):
                result = detect_ollama()

        assert result["available"] is False
        assert result["models"] == []

    def test_ollama_cli_fallback(self):
        """httpx fails but subprocess succeeds => available=True."""
        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = Exception("Connection refused")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = (
            "NAME              ID            SIZE    MODIFIED\n"
            "qwen2.5-coder:7b  abc123        4.7GB   2 hours ago\n"
        )

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch(
                "aurarouter.gui.help.setup_helpers.subprocess.run",
                return_value=mock_proc,
            ):
                result = detect_ollama()

        assert result["available"] is True
        assert len(result["models"]) == 1
        assert result["models"][0]["name"] == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# TestDetectCloudProviders
# ---------------------------------------------------------------------------


class TestDetectCloudProviders:
    def test_none_installed(self):
        """No cloud providers installed => all installed=False."""
        with patch(
            "aurarouter.gui.help.setup_helpers.importlib.metadata.distributions",
            return_value=[],
        ):
            providers = detect_cloud_providers()
        assert all(not p["installed"] for p in providers)
        assert len(providers) == 2

    def test_gemini_installed(self):
        """aurarouter-gemini in distributions => gemini.installed=True."""
        mock_gemini = MagicMock()
        mock_gemini.metadata = {"Name": "aurarouter-gemini"}
        mock_other = MagicMock()
        mock_other.metadata = {"Name": "requests"}

        with patch(
            "aurarouter.gui.help.setup_helpers.importlib.metadata.distributions",
            return_value=[mock_gemini, mock_other],
        ):
            providers = detect_cloud_providers()

        gemini = next(p for p in providers if p["name"] == "gemini")
        claude = next(p for p in providers if p["name"] == "claude")
        assert gemini["installed"] is True
        assert gemini["display_name"] == "Google Gemini"
        assert claude["installed"] is False


# ---------------------------------------------------------------------------
# TestGetRecommendedModels
# ---------------------------------------------------------------------------


class TestGetRecommendedModels:
    def test_no_gpu(self):
        """vram_mb=0 => returns only 1.5B model, recommended=True."""
        hw = HardwareInfo(vram_mb=0)
        models = get_recommended_models(hw)
        assert len(models) == 1
        assert "1.5B" in models[0]["display_name"]
        assert models[0]["recommended"] is True

    def test_8gb_vram(self):
        """vram_mb=8192 => returns 7B recommended + 1.5B fallback."""
        hw = HardwareInfo(vram_mb=8192, has_nvidia=True)
        models = get_recommended_models(hw)
        recommended = [m for m in models if m["recommended"]]
        assert any("7B" in m["display_name"] for m in recommended)
        # 1.5B should be present as fallback
        assert any("1.5B" in m["display_name"] for m in models)

    def test_24gb_vram(self):
        """vram_mb=24576 => returns 32B recommended + others."""
        hw = HardwareInfo(vram_mb=24576, has_nvidia=True)
        models = get_recommended_models(hw)
        recommended = [m for m in models if m["recommended"]]
        assert any("32B" in m["display_name"] for m in recommended)
        assert len(models) >= 2  # at least 32B + fallback(s)

    def test_4gb_vram(self):
        """vram_mb=4096 => returns 1.5B and 7B (boundary of both ranges)."""
        hw = HardwareInfo(vram_mb=4096, has_nvidia=True)
        models = get_recommended_models(hw)
        assert any("1.5B" in m["display_name"] for m in models)
        assert any(m["recommended"] for m in models)


# ---------------------------------------------------------------------------
# TestSuggestCudaSidecar
# ---------------------------------------------------------------------------


class TestSuggestCudaSidecar:
    def test_no_nvidia(self):
        """has_nvidia=False => None."""
        hw = HardwareInfo(has_nvidia=False)
        assert suggest_cuda_sidecar(hw) is None

    def test_already_installed(self):
        """has_nvidia=True, installed_sidecars=["aurarouter-cuda13"] => None."""
        hw = HardwareInfo(has_nvidia=True, installed_sidecars=["aurarouter-cuda13"])
        assert suggest_cuda_sidecar(hw) is None

    def test_cuda13(self):
        """has_nvidia=True, cuda_version="13.1" => "aurarouter-cuda13"."""
        hw = HardwareInfo(has_nvidia=True, cuda_version="13.1")
        assert suggest_cuda_sidecar(hw) == "aurarouter-cuda13"

    def test_cuda12(self):
        """has_nvidia=True, cuda_version="12.4" => "aurarouter-cuda12"."""
        hw = HardwareInfo(has_nvidia=True, cuda_version="12.4")
        assert suggest_cuda_sidecar(hw) == "aurarouter-cuda12"

    def test_unknown_version(self):
        """has_nvidia=True, cuda_version=None => "aurarouter-cuda12"."""
        hw = HardwareInfo(has_nvidia=True, cuda_version=None)
        assert suggest_cuda_sidecar(hw) == "aurarouter-cuda12"


# ---------------------------------------------------------------------------
# TestPipInstallSync
# ---------------------------------------------------------------------------


class TestPipInstallSync:
    @patch("aurarouter.gui.help.setup_helpers.subprocess.run")
    def test_success(self, mock_run):
        """returncode 0 => (True, output)."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="Successfully installed foo\n", stderr=""
        )
        ok, output = pip_install_sync("foo")
        assert ok is True
        assert "Successfully installed" in output

    @patch("aurarouter.gui.help.setup_helpers.subprocess.run")
    def test_failure(self, mock_run):
        """returncode 1 => (False, output)."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ERROR: No matching distribution\n"
        )
        ok, output = pip_install_sync("nonexistent-pkg")
        assert ok is False
        assert "ERROR" in output

    @patch(
        "aurarouter.gui.help.setup_helpers.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=300),
    )
    def test_timeout(self, _mock_run):
        """TimeoutExpired => (False, "timed out")."""
        ok, output = pip_install_sync("slow-package")
        assert ok is False
        assert "timed out" in output.lower()
