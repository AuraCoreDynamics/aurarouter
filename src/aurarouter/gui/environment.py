"""Core abstractions for environment-aware GUI behavior.

Defines the EnvironmentContext base class that separates deployment-specific
concerns (Local vs AuraGrid) from the GUI, along with ServiceState
and HealthStatus types used by the service lifecycle controls.

Note: EnvironmentContext inherits from QObject (for Qt signals) but cannot
also inherit from ABC due to metaclass conflicts. Abstract enforcement is
done via NotImplementedError instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

    from aurarouter.config import ConfigLoader


class ServiceState(Enum):
    """State machine for the AuraRouter service lifecycle."""

    STOPPED = "stopped"
    STARTING = "starting"
    LOADING_MODEL = "loading_model"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class HealthStatus:
    """Result of a health check across configured model providers."""

    healthy: bool
    message: str = ""
    details: dict[str, bool] = field(default_factory=dict)


class EnvironmentContext(QObject):
    """Abstracts deployment-specific behavior for the GUI.

    Concrete implementations:
    - ``LocalEnvironmentContext``  — standalone / pure-python mode
    - ``AuraGridEnvironmentContext`` — AuraGrid MAS mode

    The GUI consults the active context for config CRUD, model management,
    service lifecycle, health checks, and any extra tabs/widgets that are
    specific to the deployment environment.

    Subclasses must override all methods that raise ``NotImplementedError``.
    """

    # ------------------------------------------------------------------
    # Qt signals
    # ------------------------------------------------------------------
    state_changed = Signal(str)       # ServiceState.value
    health_updated = Signal(object)   # HealthStatus
    config_changed = Signal()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier, e.g. ``'Local'`` or ``'AuraGrid'``."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Human-readable one-liner shown in the UI."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Config CRUD
    # ------------------------------------------------------------------

    def get_config_loader(self) -> ConfigLoader:
        """Return the current ``ConfigLoader`` instance."""
        raise NotImplementedError

    def save_config(self) -> Path:
        """Persist configuration and return the path written to."""
        raise NotImplementedError

    def reload_config(self) -> ConfigLoader:
        """Reload configuration from its source and return the new loader."""
        raise NotImplementedError

    def config_affects_other_nodes(self) -> bool:
        """True if saving config will propagate changes beyond this machine."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def list_local_models(self) -> list[dict]:
        """Return metadata dicts for locally-stored GGUF models."""
        raise NotImplementedError

    def list_remote_models(self) -> list[dict]:
        """Return metadata dicts for remotely-stored models (grid, etc.).

        Implementations that have no remote storage should return ``[]``.
        """
        raise NotImplementedError

    def remove_model(self, filename: str, delete_file: bool = True) -> bool:
        """Remove a model from the registry (and optionally from disk)."""
        raise NotImplementedError

    def get_storage_info(self) -> dict:
        """Return storage metadata (path, model count, total size, etc.)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the routing service (MCP server subprocess or MAS)."""
        raise NotImplementedError

    def stop(self) -> None:
        """Gracefully stop the routing service."""
        raise NotImplementedError

    def pause(self) -> None:
        """Pause the service (stop accepting new requests, finish in-flight)."""
        raise NotImplementedError

    def resume(self) -> None:
        """Resume a paused service."""
        raise NotImplementedError

    def get_state(self) -> ServiceState:
        """Return the current ``ServiceState``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check_health(self) -> HealthStatus:
        """Run health checks against configured providers.

        This may be called from a background thread.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # GUI extension points
    # ------------------------------------------------------------------

    def get_extra_tabs(self) -> list[tuple[str, QWidget]]:
        """Return ``(label, widget)`` pairs for environment-specific tabs."""
        raise NotImplementedError

    def get_toolbar_widgets(self) -> list[QWidget]:
        """Return extra widgets to embed in the service toolbar."""
        raise NotImplementedError

    def get_config_warnings(self) -> list[str]:
        """Return warning strings to display on the config panel (e.g.
        'Changes will propagate to all nodes on this cell').
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Release all resources held by this context.

        Called before switching environments at runtime.  Implementations
        must stop any running service, terminate subprocesses, cancel
        background tasks, etc.
        """
        raise NotImplementedError
