"""AuraGrid deployment strategy panel.

Shown as an extra tab when the AuraGrid environment is active.  Allows
administrators to view and manage model replica counts, always-on
models, and available compute resources across the cell.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ------------------------------------------------------------------
# Background worker for fetching deployment info
# ------------------------------------------------------------------

class _DeploymentWorker(QObject):
    finished = Signal(dict)  # {"models": [...], "resources": {...}}
    error = Signal(str)

    def __init__(self, context):
        super().__init__()
        self._context = context

    def run(self) -> None:
        try:
            info: dict = {"models": [], "resources": {}}

            # Gather model info from grid storage if available.
            try:
                remote_models = self._context.list_remote_models()
                info["models"] = remote_models
            except Exception:
                info["models"] = []

            # Gather resource info from discovery if available.
            try:
                from aurarouter.auragrid.discovery import OllamaDiscovery

                discovery = OllamaDiscovery()
                endpoints = discovery.get_available_endpoints()
                info["resources"] = {
                    "endpoints": len(endpoints),
                    "details": endpoints,
                }
            except Exception:
                info["resources"] = {"endpoints": 0, "details": []}

            self.finished.emit(info)
        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# Panel
# ------------------------------------------------------------------

class GridDeploymentPanel(QWidget):
    """Deployment strategy editor for AuraGrid environments."""

    def __init__(self, context=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._context = context
        self._thread: Optional[QThread] = None
        self._worker: Optional[_DeploymentWorker] = None

        layout = QVBoxLayout(self)

        # ---- Model Deployment section ----
        layout.addWidget(self._build_models_section())

        # ---- Resources section ----
        layout.addWidget(self._build_resources_section())

        # ---- Refresh controls ----
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(refresh_btn)

        apply_btn = QPushButton("Apply Strategy")
        apply_btn.setStyleSheet("font-weight: bold;")
        apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(apply_btn)

        btn_row.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        btn_row.addWidget(self._status_label)
        layout.addLayout(btn_row)

        # Initial fetch.
        QTimer.singleShot(500, self._refresh)

    def _build_models_section(self) -> QGroupBox:
        group = QGroupBox("Model Deployment Strategy")
        layout = QVBoxLayout(group)

        self._model_table = QTableWidget(0, 4)
        self._model_table.setHorizontalHeaderLabels(
            ["Model ID", "Current Replicas", "Desired Replicas", "Status"]
        )
        self._model_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._model_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        layout.addWidget(self._model_table)

        return group

    def _build_resources_section(self) -> QGroupBox:
        group = QGroupBox("Cell Resources")
        layout = QVBoxLayout(group)

        self._resource_label = QLabel("Endpoints: --")
        self._resource_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self._resource_label)

        self._resource_table = QTableWidget(0, 2)
        self._resource_table.setHorizontalHeaderLabels(["Endpoint", "Status"])
        self._resource_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._resource_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._resource_table.setMaximumHeight(150)
        layout.addWidget(self._resource_table)

        return group

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._context is None:
            return
        self._cleanup_thread()

        self._worker = _DeploymentWorker(self._context)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_data)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._status_label.setText("Refreshing...")
        self._thread.start()

    def _on_data(self, info: dict) -> None:
        # Populate model table.
        models = info.get("models", [])
        self._model_table.setRowCount(0)
        for m in models:
            row = self._model_table.rowCount()
            self._model_table.insertRow(row)
            model_id = m if isinstance(m, str) else m.get("model_id", str(m))
            self._model_table.setItem(row, 0, QTableWidgetItem(model_id))
            self._model_table.setItem(row, 1, QTableWidgetItem("1"))

            # Desired replicas spinner.
            spinner = QSpinBox()
            spinner.setMinimum(0)
            spinner.setMaximum(10)
            spinner.setValue(1)
            self._model_table.setCellWidget(row, 2, spinner)

            self._model_table.setItem(row, 3, QTableWidgetItem("active"))

        # Populate resource table.
        resources = info.get("resources", {})
        endpoint_count = resources.get("endpoints", 0)
        self._resource_label.setText(f"Endpoints: {endpoint_count}")

        details = resources.get("details", [])
        self._resource_table.setRowCount(0)
        for ep in details:
            row = self._resource_table.rowCount()
            self._resource_table.insertRow(row)
            ep_str = ep if isinstance(ep, str) else str(ep)
            self._resource_table.setItem(row, 0, QTableWidgetItem(ep_str))
            self._resource_table.setItem(row, 1, QTableWidgetItem("available"))

        self._status_label.setText(f"Last refreshed. {len(models)} models, {endpoint_count} endpoints.")

    def _on_error(self, message: str) -> None:
        self._status_label.setText(f"Error: {message}")

    def _on_apply(self) -> None:
        QMessageBox.information(
            self,
            "Apply Strategy",
            "Deployment strategy changes will be sent to the AuraGrid cell.\n"
            "(This feature requires the AuraGrid orchestration API.)",
        )

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(3000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
