"""AuraGrid cell/node status panel.

Shown as an extra tab when the AuraGrid environment is active.  Displays
the health and configuration of nodes in the cell, the model distribution
map, and recent routing events.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ------------------------------------------------------------------
# Background worker for cell status
# ------------------------------------------------------------------

class _StatusWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, context):
        super().__init__()
        self._context = context

    def run(self) -> None:
        try:
            result: dict = {"nodes": [], "events": []}

            # Try to gather endpoint info from discovery.
            try:
                from aurarouter.auragrid.discovery import OllamaDiscovery

                discovery = OllamaDiscovery()
                endpoints = discovery.get_available_endpoints()
                for i, ep in enumerate(endpoints):
                    ep_str = ep if isinstance(ep, str) else str(ep)
                    result["nodes"].append({
                        "id": f"node-{i + 1}",
                        "address": ep_str,
                        "status": "healthy",
                        "models": "—",
                        "last_seen": "just now",
                    })
            except Exception:
                pass

            # Try to gather recent events from EventBridge.
            try:
                from auragrid.sdk.event_bridge import get_recent_events

                events = get_recent_events("aurarouter.*", limit=20)
                result["events"] = [
                    f"[{e.get('timestamp', '?')}] {e.get('type', '?')}: {e.get('summary', '')}"
                    for e in events
                ]
            except Exception:
                result["events"] = ["(Event log requires AuraGrid EventBridge SDK)"]

            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# Panel
# ------------------------------------------------------------------

class GridStatusPanel(QWidget):
    """Cell and node status overview for AuraGrid environments."""

    # Auto-refresh interval (milliseconds).
    _REFRESH_INTERVAL_MS = 30_000

    def __init__(self, context=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._context = context
        self._thread: Optional[QThread] = None
        self._worker: Optional[_StatusWorker] = None

        layout = QVBoxLayout(self)

        # ---- Node list ----
        layout.addWidget(self._build_node_section())

        # ---- Event log ----
        layout.addWidget(self._build_event_section())

        # ---- Controls ----
        ctrl_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Now")
        refresh_btn.clicked.connect(self._refresh)
        ctrl_row.addWidget(refresh_btn)

        ctrl_row.addStretch()
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        ctrl_row.addWidget(self._status_label)
        layout.addLayout(ctrl_row)

        # Auto-refresh timer.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(self._REFRESH_INTERVAL_MS)

        # Initial fetch.
        QTimer.singleShot(500, self._refresh)

    def _build_node_section(self) -> QGroupBox:
        group = QGroupBox("Cell Nodes")
        layout = QVBoxLayout(group)

        self._node_table = QTableWidget(0, 5)
        self._node_table.setHorizontalHeaderLabels(
            ["Node ID", "Address", "Status", "Models Loaded", "Last Seen"]
        )
        self._node_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._node_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._node_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        layout.addWidget(self._node_table)

        return group

    def _build_event_section(self) -> QGroupBox:
        group = QGroupBox("Event Log")
        layout = QVBoxLayout(group)

        self._event_display = QTextEdit()
        self._event_display.setReadOnly(True)
        self._event_display.setMaximumHeight(150)
        self._event_display.setPlaceholderText("Events will appear here...")
        layout.addWidget(self._event_display)

        return group

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._context is None:
            return
        self._cleanup_thread()

        self._worker = _StatusWorker(self._context)
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
        # Populate node table.
        nodes = info.get("nodes", [])
        self._node_table.setRowCount(0)
        for node in nodes:
            row = self._node_table.rowCount()
            self._node_table.insertRow(row)
            self._node_table.setItem(row, 0, QTableWidgetItem(node.get("id", "?")))
            self._node_table.setItem(row, 1, QTableWidgetItem(node.get("address", "?")))

            status_item = QTableWidgetItem(node.get("status", "unknown"))
            status = node.get("status", "")
            if status == "healthy":
                status_item.setForeground(status_item.foreground())
            self._node_table.setItem(row, 2, status_item)

            self._node_table.setItem(row, 3, QTableWidgetItem(node.get("models", "—")))
            self._node_table.setItem(row, 4, QTableWidgetItem(node.get("last_seen", "?")))

        # Populate event log.
        events = info.get("events", [])
        self._event_display.setPlainText("\n".join(events) if events else "(no events)")

        self._status_label.setText(f"Last refreshed. {len(nodes)} node(s).")

    def _on_error(self, message: str) -> None:
        self._status_label.setText(f"Error: {message}")

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
