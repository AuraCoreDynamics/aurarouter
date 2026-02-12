"""Core abstractions for environment-aware GUI behavior.

Defines the EnvironmentContext ABC that separates deployment-specific
concerns (Local vs AuraGrid) from the GUI, along with ServiceState
and HealthStatus types used by the service lifecycle controls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class EnvironmentContext(QObject, ABC):
    """Abstracts deployment-specific behavior for the GUI.

    Concrete implementations:
    - ``LocalEnvironmentContext``  — standalone / pure-python mode
    - ``AuraGridEnvironmentContext`` — AuraGrid MAS mode

    The GUI consults the active context for config CRUD, model management,
    service lifecycle, health checks, and any extra tabs/widgets that are
    specific to the deployment environment.
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
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``'Local'`` or ``'AuraGrid'``."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable one-liner shown in the UI."""

    # ------------------------------------------------------------------
    # Config CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def get_config_loader(self) -> ConfigLoader:
        """Return the current ``ConfigLoader`` instance."""

    @abstractmethod
    def save_config(self) -> Path:
        """Persist configuration and return the path written to."""

    @abstractmethod
    def reload_config(self) -> ConfigLoader:
        """Reload configuration from its source and return the new loader."""

    @abstractmethod
    def config_affects_other_nodes(self) -> bool:
        """True if saving config will propagate changes beyond this machine."""

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    @abstractmethod
    def list_local_models(self) -> list[dict]:
        """Return metadata dicts for locally-stored GGUF models."""

    @abstractmethod
    def list_remote_models(self) -> list[dict]:
        """Return metadata dicts for remotely-stored models (grid, etc.).

        Implementations that have no remote storage should return ``[]``.
        """

    @abstractmethod
    def remove_model(self, filename: str, delete_file: bool = True) -> bool:
        """Remove a model from the registry (and optionally from disk)."""

    @abstractmethod
    def get_storage_info(self) -> dict:
        """Return storage metadata (path, model count, total size, etc.)."""

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """Start the routing service (MCP server subprocess or MAS)."""

    @abstractmethod
    def stop(self) -> None:
        """Gracefully stop the routing service."""

    @abstractmethod
    def pause(self) -> None:
        """Pause the service (stop accepting new requests, finish in-flight)."""

    @abstractmethod
    def resume(self) -> None:
        """Resume a paused service."""

    @abstractmethod
    def get_state(self) -> ServiceState:
        """Return the current ``ServiceState``."""

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @abstractmethod
    def check_health(self) -> HealthStatus:
        """Run health checks against configured providers.

        This may be called from a background thread.
        """

    # ------------------------------------------------------------------
    # GUI extension points
    # ------------------------------------------------------------------

    @abstractmethod
    def get_extra_tabs(self) -> list[tuple[str, QWidget]]:
        """Return ``(label, widget)`` pairs for environment-specific tabs."""

    @abstractmethod
    def get_toolbar_widgets(self) -> list[QWidget]:
        """Return extra widgets to embed in the service toolbar."""

    @abstractmethod
    def get_config_warnings(self) -> list[str]:
        """Return warning strings to display on the config panel (e.g.
        'Changes will propagate to all nodes on this cell').
        """

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @abstractmethod
    def dispose(self) -> None:
        """Release all resources held by this context.

        Called before switching environments at runtime.  Implementations
        must stop any running service, terminate subprocesses, cancel
        background tasks, etc.
        """
