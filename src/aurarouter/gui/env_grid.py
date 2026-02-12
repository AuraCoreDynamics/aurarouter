"""AuraGrid environment context.

Manages configuration, models, and service lifecycle when AuraRouter is
deployed as a Managed Application Service (MAS) on AuraGrid.  Configuration
changes may propagate cell-wide depending on user permissions.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from aurarouter.config import ConfigLoader
from aurarouter.gui.environment import EnvironmentContext, HealthStatus, ServiceState
from aurarouter.models.file_storage import FileModelStorage

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# Optional AuraGrid imports — guarded so the module loads even without the SDK.
try:
    from aurarouter.auragrid.config_loader import ConfigLoader as GridConfigLoader
except ImportError:
    GridConfigLoader = None

try:
    from aurarouter.auragrid.lifecycle import LifecycleCallbacks
except ImportError:
    LifecycleCallbacks = None

try:
    from aurarouter.auragrid.model_storage import GridModelStorage
except ImportError:
    GridModelStorage = None


class AuraGridEnvironmentContext(EnvironmentContext):
    """``EnvironmentContext`` for AuraGrid MAS deployments."""

    def __init__(
        self,
        config: Optional[ConfigLoader] = None,
        config_path: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._state = ServiceState.STOPPED
        self._local_storage = FileModelStorage()

        # Grid-aware config loader (merges env → manifest → file → defaults).
        self._grid_config_loader: Optional[object] = None
        if GridConfigLoader is not None:
            self._grid_config_loader = GridConfigLoader(
                config_file_path=Path(config_path) if config_path else None,
                allow_missing=True,
            )
            self._config = self._grid_config_loader.load()
        elif config is not None:
            self._config = config
        else:
            try:
                self._config = ConfigLoader(config_path=config_path)
            except FileNotFoundError:
                self._config = ConfigLoader(allow_missing=True)
                self._config.config = {"models": {}, "roles": {}}

        # Lifecycle callbacks (startup / shutdown / health).
        self._lifecycle: Optional[object] = None
        self._grid_model_storage: Optional[object] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "AuraGrid"

    @property
    def description(self) -> str:
        return (
            "AuraGrid deployment — configuration and models may be shared "
            "across all nodes on the cell."
        )

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
        if self._grid_config_loader is not None and hasattr(
            self._grid_config_loader, "load"
        ):
            self._config = self._grid_config_loader.load()
        else:
            source = self._config.config_path
            if source and source.is_file():
                self._config = ConfigLoader(config_path=str(source))
        self.config_changed.emit()
        return self._config

    def config_affects_other_nodes(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[dict]:
        self._local_storage.scan()
        return self._local_storage.list_models()

    def list_remote_models(self) -> list[dict]:
        if GridModelStorage is None:
            return []
        try:
            if self._grid_model_storage is None:
                self._grid_model_storage = GridModelStorage()
            model_ids = asyncio.run(self._grid_model_storage.list_models())
            return [{"model_id": mid} for mid in model_ids]
        except Exception as exc:
            logger.warning("Failed to list grid models: %s", exc)
            return []

    def remove_model(self, filename: str, delete_file: bool = True) -> bool:
        return self._local_storage.remove(filename, delete_file=delete_file)

    def get_storage_info(self) -> dict:
        models = self._local_storage.list_models()
        total_bytes = sum(m.get("size_bytes", 0) for m in models)
        info: dict = {
            "path": str(self._local_storage.models_dir),
            "count": len(models),
            "total_bytes": total_bytes,
        }
        if self._grid_model_storage is not None:
            info["grid_available"] = True
        return info

    # ------------------------------------------------------------------
    # Service lifecycle  (AuraGrid MAS via LifecycleCallbacks)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._state in (ServiceState.RUNNING, ServiceState.STARTING):
            return

        self._set_state(ServiceState.STARTING)
        try:
            if LifecycleCallbacks is not None:
                self._lifecycle = LifecycleCallbacks(
                    config_loader=self._config,
                    grid_config_loader=self._grid_config_loader,
                )
                asyncio.run(self._lifecycle.startup())
                logger.info("AuraGrid MAS lifecycle started.")
                self._set_state(ServiceState.RUNNING)
            else:
                logger.error("AuraGrid SDK not available — cannot start MAS.")
                self._set_state(ServiceState.ERROR)
        except Exception as exc:
            logger.error("Failed to start AuraGrid MAS: %s", exc)
            self._set_state(ServiceState.ERROR)

    def stop(self) -> None:
        if self._state in (ServiceState.STOPPED, ServiceState.STOPPING):
            return

        self._set_state(ServiceState.STOPPING)
        try:
            if self._lifecycle is not None and hasattr(self._lifecycle, "shutdown"):
                asyncio.run(self._lifecycle.shutdown())
            logger.info("AuraGrid MAS lifecycle stopped.")
        except Exception as exc:
            logger.warning("Error during AuraGrid shutdown: %s", exc)
        finally:
            self._lifecycle = None
            self._set_state(ServiceState.STOPPED)

    def pause(self) -> None:
        if self._state != ServiceState.RUNNING:
            return

        self._set_state(ServiceState.PAUSING)
        # Pause by setting the fabric to reject new requests.
        if (
            self._lifecycle is not None
            and hasattr(self._lifecycle, "fabric")
            and self._lifecycle.fabric is not None
        ):
            self._lifecycle.is_healthy = False
        self._set_state(ServiceState.PAUSED)

    def resume(self) -> None:
        if self._state != ServiceState.PAUSED:
            return
        if (
            self._lifecycle is not None
            and hasattr(self._lifecycle, "fabric")
            and self._lifecycle.fabric is not None
        ):
            self._lifecycle.is_healthy = True
        self._set_state(ServiceState.RUNNING)

    def get_state(self) -> ServiceState:
        return self._state

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check_health(self) -> HealthStatus:
        if self._lifecycle is not None and hasattr(self._lifecycle, "health_check"):
            try:
                ok = asyncio.run(self._lifecycle.health_check())
                return HealthStatus(
                    healthy=ok,
                    message="AuraGrid health check passed." if ok else "Health check failed.",
                )
            except Exception as exc:
                return HealthStatus(healthy=False, message=str(exc))
        return HealthStatus(healthy=False, message="Lifecycle not initialized.")

    # ------------------------------------------------------------------
    # GUI extension points
    # ------------------------------------------------------------------

    def get_extra_tabs(self) -> list[tuple[str, QWidget]]:
        tabs: list[tuple[str, QWidget]] = []
        try:
            from aurarouter.gui.grid_deployment_panel import GridDeploymentPanel

            tabs.append(("Deployment", GridDeploymentPanel()))
        except ImportError:
            pass
        try:
            from aurarouter.gui.grid_status_panel import GridStatusPanel

            tabs.append(("Cell Status", GridStatusPanel()))
        except ImportError:
            pass
        return tabs

    def get_toolbar_widgets(self) -> list[QWidget]:
        return []

    def get_config_warnings(self) -> list[str]:
        return ["Changes will propagate to all nodes on this cell."]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        if self._state not in (ServiceState.STOPPED, ServiceState.ERROR):
            self.stop()

        if self._grid_config_loader is not None and hasattr(
            self._grid_config_loader, "close"
        ):
            try:
                self._grid_config_loader.close()
            except Exception:
                pass

        self._lifecycle = None
        self._grid_model_storage = None
        self._state = ServiceState.STOPPED

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_state(self, state: ServiceState) -> None:
        self._state = state
        self.state_changed.emit(state.value)
