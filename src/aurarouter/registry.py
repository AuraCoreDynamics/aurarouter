"""Runtime model registry — unified view of model availability across all providers."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurarouter.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class RuntimeModelRegistry:
    """Unified view of model availability across all providers.

    Thread-safe. Aggregates telemetry from all configured providers
    via background polling on a daemon thread.

    Follows the IPC server daemon-thread pattern (see ``ipc.py``).
    """

    def __init__(self, providers: dict[str, BaseProvider], poll_interval: float = 15.0):
        self._providers = providers
        self._poll_interval = poll_interval
        self._telemetry: dict[str, object] = {}  # model_id -> ModelTelemetry
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_all_telemetry(self) -> dict:
        """Return current telemetry snapshot for all models. Key = model_id."""
        with self._lock:
            return dict(self._telemetry)

    def get_online_models(self) -> list[str]:
        """Return model_ids with state WARM or LOADING."""
        try:
            from aurarouter.auragrid.contracts import ModelState
        except ImportError:
            return []
        with self._lock:
            return [
                mid
                for mid, t in self._telemetry.items()
                if hasattr(t, "state")
                and t.state in (ModelState.WARM, ModelState.LOADING)
            ]

    def get_model_state(self, model_id: str):
        """Return current ModelState for a specific model, or UNKNOWN if not tracked."""
        try:
            from aurarouter.auragrid.contracts import ModelState
        except ImportError:
            return None
        with self._lock:
            t = self._telemetry.get(model_id)
            if t is None or not hasattr(t, "state"):
                return ModelState.UNKNOWN
            return t.state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_polling(self) -> None:
        """Start the background telemetry polling loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="model-registry-poll"
        )
        self._thread.start()
        logger.info("Model registry polling started (interval=%.1fs)", self._poll_interval)

    def stop_polling(self) -> None:
        """Stop background polling and wait for the thread to exit."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 2)
            self._thread = None
        logger.info("Model registry polling stopped")

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background loop — polls providers then sleeps until stop or timeout."""
        while self._running:
            self._poll_once()
            self._stop_event.wait(timeout=self._poll_interval)

    def _poll_once(self) -> None:
        """Single poll cycle — query every provider for telemetry."""
        for name, provider in self._providers.items():
            try:
                if not hasattr(provider, "get_telemetry"):
                    continue
                telemetry = provider.get_telemetry()
                if telemetry is None:
                    continue
                model_id = getattr(telemetry, "model_id", name)
                with self._lock:
                    self._telemetry[model_id] = telemetry
                logger.debug(
                    "Telemetry updated: %s -> %s",
                    model_id,
                    getattr(telemetry, "state", "unknown"),
                )
            except Exception as ex:  # noqa: F841
                logger.debug("registry._poll_once_error", exc_info=True)
                logger.debug(
                    "Telemetry poll failed for provider %s", name, exc_info=True
                )
                # Preserve last-known state if available; only set UNKNOWN for
                # providers that have never reported telemetry.
                with self._lock:
                    if name not in self._telemetry:
                        try:
                            from aurarouter.auragrid.contracts import ModelTelemetry, ModelState

                            self._telemetry[name] = ModelTelemetry(
                                model_id=name,
                                provider_name=name,
                                state=ModelState.UNKNOWN,
                            )
                        except ImportError:
                            pass
