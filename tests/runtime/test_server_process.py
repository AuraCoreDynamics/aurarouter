"""Tests for aurarouter.runtime.server_process."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aurarouter.runtime.server_process import ServerProcess


@pytest.fixture
def fake_binary(tmp_path):
    """Create a fake llama-server binary."""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fake-binary")
    return binary


@pytest.fixture
def fake_model(tmp_path):
    """Create a fake GGUF model file."""
    model = tmp_path / "test.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 100)
    return model


class TestBuildCommand:
    def test_includes_all_params(self, fake_binary, fake_model):
        """CLI args include -m, -c, -ngl, -t, --host, --port."""
        sp = ServerProcess(
            model_path=fake_model,
            binary_path=fake_binary,
            n_ctx=8192,
            n_gpu_layers=-1,
            n_threads=4,
            host="127.0.0.1",
            port=9999,
        )
        sp._port = 9999  # Ensure port is set
        cmd = sp._build_command()
        assert str(fake_binary) in cmd
        assert "-m" in cmd
        assert str(fake_model) in cmd
        assert "-c" in cmd
        assert "8192" in cmd
        assert "-ngl" in cmd
        assert "-1" in cmd
        assert "-t" in cmd
        assert "4" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--port" in cmd
        assert "9999" in cmd

    def test_omits_threads_when_none(self, fake_binary, fake_model):
        """If n_threads is None, -t flag is not included."""
        sp = ServerProcess(
            model_path=fake_model,
            binary_path=fake_binary,
            n_threads=None,
            port=8080,
        )
        sp._port = 8080
        cmd = sp._build_command()
        assert "-t" not in cmd

    def test_verbose_flag(self, fake_binary, fake_model):
        """--verbose flag is included when verbose=True."""
        sp = ServerProcess(
            model_path=fake_model,
            binary_path=fake_binary,
            verbose=True,
            port=8080,
        )
        sp._port = 8080
        cmd = sp._build_command()
        assert "--verbose" in cmd


class TestFindFreePort:
    def test_returns_valid_port(self, fake_binary, fake_model):
        """_find_free_port() returns a port in the ephemeral range."""
        sp = ServerProcess(
            model_path=fake_model, binary_path=fake_binary
        )
        port = sp._find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


class TestEndpoint:
    def test_endpoint_property(self, fake_binary, fake_model):
        """endpoint returns http://host:port string."""
        sp = ServerProcess(
            model_path=fake_model,
            binary_path=fake_binary,
            host="127.0.0.1",
            port=12345,
        )
        sp._port = 12345
        assert sp.endpoint == "http://127.0.0.1:12345"


class TestStartStop:
    @patch("aurarouter.runtime.server_process.subprocess.Popen")
    @patch("aurarouter.runtime.server_process.urllib.request.urlopen")
    def test_start_calls_popen(self, mock_urlopen, mock_popen, fake_binary, fake_model):
        """start() calls subprocess.Popen with the built command."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        sp = ServerProcess(
            model_path=fake_model, binary_path=fake_binary, port=0
        )
        sp.start(timeout=5.0)
        mock_popen.assert_called_once()
        assert sp.is_running

    @patch("aurarouter.runtime.server_process.subprocess.Popen")
    def test_stop_terminates_process(self, mock_popen, fake_binary, fake_model):
        """stop() calls terminate() on the subprocess."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        sp = ServerProcess(
            model_path=fake_model, binary_path=fake_binary, port=8080
        )
        sp._process = mock_proc
        sp.stop()
        mock_proc.terminate.assert_called_once()

    @patch("aurarouter.runtime.server_process.subprocess.Popen")
    @patch("aurarouter.runtime.server_process.urllib.request.urlopen")
    def test_start_timeout_raises(self, mock_urlopen, mock_popen, fake_binary, fake_model):
        """TimeoutError if server doesn't become healthy within timeout."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        mock_urlopen.side_effect = ConnectionRefusedError("refused")

        sp = ServerProcess(
            model_path=fake_model, binary_path=fake_binary, port=0
        )
        with pytest.raises(TimeoutError, match="did not become healthy"):
            sp.start(timeout=1.0)


class TestIsRunning:
    def test_not_running_initially(self, fake_binary, fake_model):
        """is_running is False before start()."""
        sp = ServerProcess(
            model_path=fake_model, binary_path=fake_binary
        )
        assert sp.is_running is False

    def test_running_with_live_process(self, fake_binary, fake_model):
        """is_running is True when process is alive."""
        sp = ServerProcess(
            model_path=fake_model, binary_path=fake_binary
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        sp._process = mock_proc
        assert sp.is_running is True

    def test_not_running_after_exit(self, fake_binary, fake_model):
        """is_running is False when process has exited."""
        sp = ServerProcess(
            model_path=fake_model, binary_path=fake_binary
        )
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        sp._process = mock_proc
        assert sp.is_running is False
