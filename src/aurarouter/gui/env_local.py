"""Local (standalone / pure-python) environment context.

Manages configuration via the local ``auraconfig.yaml`` file, models via
``FileModelStorage``, and the MCP server as a subprocess.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from aurarouter.config import ConfigLoader
from aurarouter.gui.environment import EnvironmentContext, HealthStatus, ServiceState
from aurarouter.models.file_storage import FileModelStorage

logger = logging.getLogger(__name__)


class LocalEnvironmentContext(EnvironmentContext):
    """``EnvironmentContext`` for standalone / pure-python deployments."""

    def __init__(
        self,
        config: Optional[ConfigLoader] = None,
        config_path: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._config_path = config_path
        if config is not None:
            self._config = config
        else:
            try:
                self._config = ConfigLoader(config_path=config_path)
            except FileNotFoundError:
                self._config = ConfigLoader(allow_missing=True)
                self._config.config = {"models": {}, "roles": {}}

        self._state = ServiceState.STOPPED
        self._process: Optional[subprocess.Popen] = None
        self._storage = FileModelStorage()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Local"

    @property
    def description(self) -> str:
        return "Standalone deployment — configuration and models are local to this machine."

    # ------------------------------------------------------------------
    # Config CRUD
    # ------------------------------------------------------------------

    def get_config_loader(self) -> ConfigLoader:
        return self._config

    def save_config(self) -> Path:
        path = self._config.save()
        self.config_changed.emit()
        return path

    def reload_config(self) -> ConfigLoader:
        source = self._config.config_path
        if source and source.is_file():
            self._config = ConfigLoader(config_path=str(source))
        self.config_changed.emit()
        return self._config

    def config_affects_other_nodes(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[dict]:
        self._storage.scan()
        return self._storage.list_models()

    def list_remote_models(self) -> list[dict]:
        return []

    def remove_model(self, filename: str, delete_file: bool = True) -> bool:
        return self._storage.remove(filename, delete_file=delete_file)

    def get_storage_info(self) -> dict:
        models = self._storage.list_models()
        total_bytes = sum(m.get("size_bytes", 0) for m in models)
        return {
            "path": str(self._storage.models_dir),
            "count": len(models),
            "total_bytes": total_bytes,
        }

    # ------------------------------------------------------------------
    # Service lifecycle  (MCP server subprocess)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._state in (ServiceState.RUNNING, ServiceState.STARTING):
            return

        self._set_state(ServiceState.STARTING)
        try:
            cmd = [sys.executable, "-m", "aurarouter"]
            if self._config.config_path:
                cmd += ["--config", str(self._config.config_path)]

            # On Windows, use CREATE_NEW_PROCESS_GROUP so we can signal it.
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **kwargs,
            )
            logger.info("MCP server subprocess started (PID %d)", self._process.pid)

            # If local GPU models are configured, wait for them to be ready
            # before reporting RUNNING.
            if self._has_local_gpu_models():
                self._set_state(ServiceState.LOADING_MODEL)
                self._wait_for_model_ready()

            self._set_state(ServiceState.RUNNING)
        except Exception as exc:
            logger.error("Failed to start MCP server: %s", exc)
            self._set_state(ServiceState.ERROR)

    def _has_local_gpu_models(self) -> bool:
        """Check if any configured model uses a local GPU provider."""
        for model_id in self._config.get_all_model_ids():
            cfg = self._config.get_model_config(model_id)
            provider = cfg.get("provider", "")
            if provider in ("llamacpp", "llamacpp-server"):
                return True
        return False

    def _wait_for_model_ready(self, timeout: float = 120.0) -> None:
        """Poll local model providers until they report ready."""
        import time

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            # If the subprocess died, stop waiting.
            if self._process is not None and self._process.poll() is not None:
                logger.warning("MCP subprocess exited during model loading")
                return
            if self._check_local_models_ready():
                return
            time.sleep(2.0)
        logger.warning("Model loading timed out after %.0fs", timeout)

    def _check_local_models_ready(self) -> bool:
        """Check if all local GPU models are loaded and responsive."""
        for model_id in self._config.get_all_model_ids():
            cfg = self._config.get_model_config(model_id)
            provider = cfg.get("provider", "")
            if provider == "llamacpp-server":
                try:
                    import httpx

                    endpoint = cfg.get("endpoint", "http://localhost:8080")
                    resp = httpx.get(
                        f"{endpoint.rstrip('/')}/health", timeout=5.0
                    )
                    data = resp.json()
                    if data.get("status") != "ok":
                        return False
                except Exception:
                    return False
            elif provider == "llamacpp":
                # In-process llamacpp loads when the subprocess first
                # instantiates the provider. The model file existing is
                # the best pre-check we can do here.
                path = cfg.get("model_path", "")
                if path and not Path(path).is_file():
                    return False
        return True

    def stop(self) -> None:
        if self._state in (ServiceState.STOPPED, ServiceState.STOPPING):
            return

        self._set_state(ServiceState.STOPPING)
        self._terminate_process()
        self._set_state(ServiceState.STOPPED)

    def pause(self) -> None:
        if self._state != ServiceState.RUNNING:
            return

        self._set_state(ServiceState.PAUSING)
        self._terminate_process()
        self._set_state(ServiceState.PAUSED)

    def resume(self) -> None:
        if self._state != ServiceState.PAUSED:
            return
        self.start()

    def get_state(self) -> ServiceState:
        # Detect if the subprocess died unexpectedly.
        if (
            self._state == ServiceState.RUNNING
            and self._process is not None
            and self._process.poll() is not None
        ):
            logger.warning(
                "MCP server subprocess exited unexpectedly (rc=%d)",
                self._process.returncode,
            )
            self._set_state(ServiceState.ERROR)
        return self._state

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check_health(self) -> HealthStatus:
        # Early return for non-running states — provider checks are
        # meaningless when the service itself isn't active.
        if self._state in (ServiceState.STOPPED, ServiceState.ERROR):
            return HealthStatus(
                healthy=False,
                message=f"Service is {self._state.value}. Start the service to check health.",
                details={},
            )
        if self._state in (
            ServiceState.STARTING,
            ServiceState.STOPPING,
            ServiceState.PAUSING,
            ServiceState.LOADING_MODEL,
        ):
            return HealthStatus(
                healthy=False,
                message=f"Service is {self._state.value}...",
                details={},
            )
        if self._state == ServiceState.PAUSED:
            return HealthStatus(
                healthy=False,
                message="Service is paused.",
                details={},
            )

        details: dict[str, bool] = {}
        messages: list[str] = []

        # Check subprocess health.
        proc_alive = (
            self._process is not None and self._process.poll() is None
        )

        if not proc_alive:
            return HealthStatus(
                healthy=False,
                message="MCP server subprocess is not running.",
                details=details,
            )

        # Check configured model providers (lightweight reachability).
        for model_id in self._config.get_all_model_ids():
            cfg = self._config.get_model_config(model_id)
            provider = cfg.get("provider", "")
            ok = self._check_provider(provider, cfg)
            details[model_id] = ok
            if not ok:
                messages.append(f"{model_id} ({provider}): unreachable")

        healthy = all(details.values()) if details else True
        return HealthStatus(
            healthy=healthy,
            message="Service running. " + ("; ".join(messages) if messages else "All providers OK."),
            details=details,
        )

    # ------------------------------------------------------------------
    # GUI extension points
    # ------------------------------------------------------------------

    def get_extra_tabs(self) -> list[tuple[str, "QWidget"]]:
        return []

    def get_toolbar_widgets(self) -> list["QWidget"]:
        return []

    def get_config_warnings(self) -> list[str]:
        return []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        self._terminate_process()
        self._state = ServiceState.STOPPED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: ServiceState) -> None:
        self._state = state
        self.state_changed.emit(state.value)

    def _terminate_process(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                if sys.platform == "win32":
                    self._process.terminate()
                else:
                    os.kill(self._process.pid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3)
            logger.info("MCP server subprocess terminated.")
        except Exception as exc:
            logger.warning("Error terminating subprocess: %s", exc)
        finally:
            self._process = None

    @staticmethod
    def _check_provider(provider: str, cfg: dict) -> bool:
        """Lightweight reachability check for a single provider."""
        try:
            if provider == "ollama":
                import httpx

                endpoint = cfg.get(
                    "endpoint", "http://localhost:11434/api/generate"
                )
                base = (
                    endpoint.split("/api/")[0]
                    if "/api/" in endpoint
                    else endpoint.rstrip("/")
                )
                resp = httpx.get(f"{base}/api/tags", timeout=5.0)
                return resp.status_code == 200

            if provider == "llamacpp-server":
                import httpx

                endpoint = cfg.get("endpoint", "http://localhost:8080")
                resp = httpx.get(
                    f"{endpoint.rstrip('/')}/health", timeout=5.0
                )
                return resp.status_code == 200

            if provider == "llamacpp":
                path = cfg.get("model_path", "")
                return bool(path) and Path(path).is_file()

            if provider == "openapi":
                import httpx

                endpoint = cfg.get("endpoint", "http://localhost:8000/v1")
                resp = httpx.get(
                    f"{endpoint.rstrip('/')}/models", timeout=5.0
                )
                return resp.status_code == 200

            if provider in ("google", "claude"):
                key = cfg.get("api_key", "")
                if key and "YOUR_" not in str(key):
                    return True
                env_key = cfg.get("env_key")
                if env_key:
                    return bool(os.environ.get(env_key))
                return False

            return True
        except Exception:
            return False
