"""Thread-safe controller for environment lifecycle operations.

Wraps ``EnvironmentContext.start/stop/pause/resume/check_health`` in
``QThread`` workers so the GUI remains responsive during potentially
long-running operations (subprocess spawn, network health checks, etc.).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from aurarouter.gui.environment import EnvironmentContext, HealthStatus, ServiceState


# ------------------------------------------------------------------
# Generic one-shot worker
# ------------------------------------------------------------------

class _Worker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self._fn()
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))


class _HealthWorker(QObject):
    finished = Signal(object)  # HealthStatus
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# ServiceController
# ------------------------------------------------------------------

class ServiceController(QObject):
    """Manages ``EnvironmentContext`` lifecycle from the GUI main thread.

    Emits:
        state_changed(str) — forwarded from the context
        health_result(object) — ``HealthStatus`` from a background check
        error(str) — any exception message
    """

    state_changed = Signal(str)
    health_result = Signal(object)   # HealthStatus
    error = Signal(str)

    def __init__(self, context: EnvironmentContext, parent=None):
        super().__init__(parent)
        self._context = context
        self._thread: Optional[QThread] = None
        self._worker: Optional[QObject] = None

        # Forward context signals.
        self._context.state_changed.connect(self.state_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_context(self, context: EnvironmentContext) -> None:
        """Replace the active context (e.g. after environment switch)."""
        self._cleanup()
        try:
            self._context.state_changed.disconnect(self.state_changed)
        except RuntimeError:
            pass
        self._context = context
        self._context.state_changed.connect(self.state_changed)

    def start_service(self) -> None:
        self._run_in_thread(self._context.start)

    def stop_service(self) -> None:
        self._run_in_thread(self._context.stop)

    def pause_service(self) -> None:
        # Pause is lightweight — safe on main thread.
        try:
            self._context.pause()
        except Exception as exc:
            self.error.emit(str(exc))

    def resume_service(self) -> None:
        self._run_in_thread(self._context.resume)

    def run_health_check(self) -> None:
        self._run_health_in_thread()

    def current_state(self) -> ServiceState:
        return self._context.get_state()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_in_thread(self, fn) -> None:
        self._cleanup()

        worker = _Worker(fn)
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(lambda: self._on_thread_done(thread, worker))

        self._thread = thread
        self._worker = worker
        thread.start()

    def _run_health_in_thread(self) -> None:
        self._cleanup()

        worker = _HealthWorker(self._context.check_health)
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_health_done)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(lambda: self._on_thread_done(thread, worker))

        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_health_done(self, status: HealthStatus) -> None:
        self.health_result.emit(status)

    def _on_error(self, message: str) -> None:
        self.error.emit(message)

    def _on_thread_done(self, thread: QThread, worker: QObject) -> None:
        thread.deleteLater()
        worker.deleteLater()
        if self._thread is thread:
            self._thread = None
            self._worker = None

    def _cleanup(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
