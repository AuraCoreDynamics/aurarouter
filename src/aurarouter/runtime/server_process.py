"""Managed llama-server subprocess lifecycle."""

import atexit
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Runtime")


class ServerProcess:
    """Manages a llama-server subprocess for a single GGUF model.

    Starts the server on demand, health-checks it, and stops it on
    shutdown.  Each instance manages exactly one model file.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        binary_path: Path | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        verbose: bool = False,
    ) -> None:
        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._host = host
        self._port = port
        self._verbose = verbose
        self._process: subprocess.Popen | None = None
        self._atexit_registered = False

        if binary_path is not None:
            self._binary_path = Path(binary_path)
        else:
            from aurarouter.runtime.binary_manager import BinaryManager
            self._binary_path = BinaryManager.resolve_server_binary()

    @property
    def endpoint(self) -> str:
        """Return the base URL: ``http://host:port``."""
        return f"http://{self._host}:{self._port}"

    @property
    def is_running(self) -> bool:
        """Check if the subprocess is alive."""
        return self._process is not None and self._process.poll() is None

    def start(self, timeout: float = 60.0) -> None:
        """Start llama-server and wait until healthy.

        1. Find a free port (if port was 0).
        2. Build the command line.
        3. Launch ``subprocess.Popen``.
        4. Poll ``GET /health`` every 0.5s until 200 or *timeout*.
        5. Register ``atexit`` shutdown handler.

        Raises
        ------
        TimeoutError
            If the server doesn't become healthy within *timeout* seconds.
        FileNotFoundError
            If the binary is missing.
        """
        if self.is_running:
            logger.debug("Server already running at %s", self.endpoint)
            return

        if not self._binary_path.is_file():
            raise FileNotFoundError(
                f"llama-server binary not found: {self._binary_path}"
            )

        if self._port == 0:
            self._port = self._find_free_port()

        cmd = self._build_command()
        logger.info("Starting llama-server: %s", " ".join(cmd))

        kwargs: dict = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._process = subprocess.Popen(cmd, **kwargs)

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

        self._wait_healthy(timeout)
        logger.info(
            "llama-server healthy at %s (model: %s)",
            self.endpoint,
            self._model_path.name,
        )

    def stop(self) -> None:
        """Stop the subprocess gracefully (terminate), then force-kill after 5s."""
        if self._process is None:
            return

        if self._process.poll() is not None:
            self._process = None
            return

        logger.info("Stopping llama-server (pid=%d)...", self._process.pid)
        try:
            self._process.terminate()
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server did not stop; force-killing.")
            self._process.kill()
            self._process.wait(timeout=5.0)
        except Exception:
            pass
        finally:
            self._process = None

    def _find_free_port(self) -> int:
        """Bind a TCP socket to port 0, read the assigned port, close."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _build_command(self) -> list[str]:
        """Build the llama-server CLI argument list."""
        cmd = [
            str(self._binary_path),
            "-m", str(self._model_path),
            "--host", self._host,
            "--port", str(self._port),
            "-c", str(self._n_ctx),
            "-ngl", str(self._n_gpu_layers),
        ]
        if self._n_threads is not None:
            cmd.extend(["-t", str(self._n_threads)])
        if self._verbose:
            cmd.append("--verbose")
        return cmd

    def _wait_healthy(self, timeout: float) -> None:
        """Poll ``GET http://host:port/health`` until 200 OK or timeout."""
        url = f"{self.endpoint}/health"
        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None

        while time.monotonic() < deadline:
            # Check if process died during startup
            if self._process is not None and self._process.poll() is not None:
                stderr_output = ""
                if self._process.stderr:
                    try:
                        stderr_output = self._process.stderr.read().decode(
                            errors="replace"
                        )
                    except Exception:
                        pass
                raise RuntimeError(
                    f"llama-server exited during startup "
                    f"(code={self._process.returncode}). "
                    f"stderr: {stderr_output[:500]}"
                )
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:
                last_exc = exc
            time.sleep(0.5)

        # Timeout — kill the process
        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy within {timeout}s. "
            f"Last error: {last_exc}"
        )
