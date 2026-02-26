"""Tests for aurarouter.runtime.binary_manager."""

import os
import sys
from pathlib import Path

import pytest

from aurarouter.runtime.binary_manager import BinaryManager


class TestDetectPlatform:
    def test_returns_valid_key(self):
        """detect_platform() returns one of the three supported keys."""
        try:
            key = BinaryManager.detect_platform()
            assert key in ("win-x64", "linux-x64", "macos-x64")
        except RuntimeError:
            # Unsupported platform is acceptable in CI
            pytest.skip("Running on unsupported platform")

    def test_win_platform(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("platform.machine", lambda: "AMD64")
        assert BinaryManager.detect_platform() == "win-x64"

    def test_linux_platform(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        assert BinaryManager.detect_platform() == "linux-x64"

    def test_darwin_platform(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        assert BinaryManager.detect_platform() == "macos-x64"

    def test_unsupported_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "freebsd")
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            BinaryManager.detect_platform()


class TestResolveServerBinary:
    def test_config_takes_priority(self, tmp_path):
        """Config key 'llamacpp_binary' takes priority over all others."""
        binary = tmp_path / "llama-server.exe"
        binary.write_bytes(b"fake")
        cfg = {"llamacpp_binary": str(binary)}
        result = BinaryManager.resolve_server_binary(cfg)
        assert result == binary.resolve()

    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        """AURAROUTER_LLAMACPP_BIN env var takes priority over bundled."""
        binary = tmp_path / "llama-server.exe"
        binary.write_bytes(b"fake")
        monkeypatch.setenv("AURAROUTER_LLAMACPP_BIN", str(binary))
        result = BinaryManager.resolve_server_binary()
        assert result == binary.resolve()

    def test_bundled_binary(self, tmp_path, monkeypatch):
        """Falls back to bundled binary in the package bin/ directory."""
        # Create a fake bundled binary
        plat_dir = tmp_path / "bin" / "win-x64"
        plat_dir.mkdir(parents=True)
        binary = plat_dir / "llama-server.exe"
        binary.write_bytes(b"fake")

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("platform.machine", lambda: "AMD64")
        monkeypatch.setattr(
            BinaryManager, "get_bundled_bin_dir", staticmethod(lambda: plat_dir)
        )
        monkeypatch.delenv("AURAROUTER_LLAMACPP_BIN", raising=False)
        result = BinaryManager.resolve_server_binary()
        assert result == binary.resolve()

    def test_system_path(self, monkeypatch, tmp_path):
        """Falls back to shutil.which('llama-server') on system PATH."""
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"fake")
        monkeypatch.delenv("AURAROUTER_LLAMACPP_BIN", raising=False)
        monkeypatch.setattr(
            BinaryManager,
            "get_bundled_bin_dir",
            staticmethod(lambda: tmp_path / "nonexistent"),
        )
        monkeypatch.setattr("shutil.which", lambda name: str(binary))
        result = BinaryManager.resolve_server_binary()
        assert result == binary.resolve()

    def test_not_found_raises(self, monkeypatch, tmp_path):
        """Raises FileNotFoundError when no binary can be found."""
        monkeypatch.delenv("AURAROUTER_LLAMACPP_BIN", raising=False)
        monkeypatch.setattr(
            BinaryManager,
            "get_bundled_bin_dir",
            staticmethod(lambda: tmp_path / "nonexistent"),
        )
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="llama-server binary not found"):
            BinaryManager.resolve_server_binary()


class TestValidateBinary:
    def test_existing_file(self, tmp_path):
        """Returns True for an existing file."""
        binary = tmp_path / "llama-server"
        binary.write_bytes(b"fake")
        if sys.platform != "win32":
            os.chmod(binary, 0o755)
        assert BinaryManager.validate_binary(binary) is True

    def test_missing_file(self, tmp_path):
        """Returns False for a non-existent path."""
        assert BinaryManager.validate_binary(tmp_path / "nope") is False
