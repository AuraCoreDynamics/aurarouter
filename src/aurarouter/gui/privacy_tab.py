"""Privacy Audit Dashboard tab — security event log and summary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aurarouter.savings.privacy import PrivacyStore

_TIME_RANGES = [
    "Last Hour",
    "Today",
    "This Week",
    "This Month",
    "All Time",
]

_SEVERITY_FILTERS = ["All", "High", "Medium", "Low"]

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

_SEVERITY_COLORS = {
    "high": "#dc3545",
    "medium": "#fd7e14",
    "low": "#ffc107",
}


def _time_range_bounds(label: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(start, end)`` ISO-8601 strings for the chosen range."""
    now = datetime.now(timezone.utc)
    if label == "Last Hour":
        start = now - timedelta(hours=1)
    elif label == "Today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif label == "This Week":
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    elif label == "This Month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:  # All Time
        return None, None
    return start.isoformat(), None


def _max_severity(severities: list[str]) -> str:
    """Return the highest severity in a list."""
    if not severities:
        return "low"
    return max(severities, key=lambda s: _SEVERITY_RANK.get(s, 0))


# ------------------------------------------------------------------
# Background worker
# ------------------------------------------------------------------


class _PrivacyWorker(QObject):
    """Queries PrivacyStore off the main thread."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        store: PrivacyStore,
        time_range: str,
        severity_filter: str,
    ) -> None:
        super().__init__()
        self._store = store
        self._time_range = time_range
        self._severity_filter = severity_filter

    def run(self) -> None:
        try:
            start, end = _time_range_bounds(self._time_range)
            min_sev = (
                self._severity_filter.lower()
                if self._severity_filter != "All"
                else None
            )

            events = self._store.query(start=start, end=end, min_severity=min_sev)

            # Classify events by max severity.
            high_count = 0
            medium_count = 0
            low_count = 0
            for ev in events:
                ms = _max_severity(ev["severities"])
                if ms == "high":
                    high_count += 1
                elif ms == "medium":
                    medium_count += 1
                else:
                    low_count += 1

            # Pattern breakdown.
            pattern_data: dict[str, dict] = {}
            for ev in events:
                for i, name in enumerate(ev["pattern_names"]):
                    sev = (
                        ev["severities"][i]
                        if i < len(ev["severities"])
                        else "low"
                    )
                    bucket = pattern_data.setdefault(
                        name,
                        {"count": 0, "most_recent": "", "severity": sev},
                    )
                    bucket["count"] += 1
                    if ev["timestamp"] > bucket["most_recent"]:
                        bucket["most_recent"] = ev["timestamp"]
                    # Keep highest severity seen for this pattern.
                    if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(
                        bucket["severity"], 0
                    ):
                        bucket["severity"] = sev

            pattern_rows = sorted(
                [{"name": k, **v} for k, v in pattern_data.items()],
                key=lambda r: r["count"],
                reverse=True,
            )

            # Event rows (newest first).
            event_rows = sorted(
                events, key=lambda e: e["timestamp"], reverse=True
            )

            self.finished.emit(
                {
                    "total": len(events),
                    "high": high_count,
                    "medium": medium_count,
                    "low": low_count,
                    "pattern_rows": pattern_rows,
                    "event_rows": event_rows,
                }
            )

        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# Privacy Audit Dashboard tab
# ------------------------------------------------------------------


class PrivacyAuditTab(QWidget):
    """Dashboard showing privacy audit events with severity coding and badges."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._store: Optional[PrivacyStore] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_PrivacyWorker] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data_source(self, privacy_store: PrivacyStore) -> None:
        """Inject data layer reference (called after construction)."""
        self._store = privacy_store

    def refresh(self) -> None:
        """Kick off a background query with current filters."""
        if self._store is None:
            return
        if self._thread is not None and self._thread.isRunning():
            return

        time_range = self._range_combo.currentText()
        severity_filter = self._severity_combo.currentText()

        self._worker = _PrivacyWorker(self._store, time_range, severity_filter)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._apply_data)
        self._worker.error.connect(self._on_worker_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._refresh_btn.setEnabled(False)
        self._thread.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ---- Top bar ----
        top = QHBoxLayout()
        top.addWidget(QLabel("Time Range:"))
        self._range_combo = QComboBox()
        self._range_combo.addItems(_TIME_RANGES)
        self._range_combo.currentIndexChanged.connect(lambda _: self.refresh())
        top.addWidget(self._range_combo)

        top.addWidget(QLabel("Severity:"))
        self._severity_combo = QComboBox()
        self._severity_combo.addItems(_SEVERITY_FILTERS)
        self._severity_combo.currentIndexChanged.connect(lambda _: self.refresh())
        top.addWidget(self._severity_combo)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self._refresh_btn)
        top.addStretch()
        root.addLayout(top)

        # ---- Summary row ----
        summary_group = QGroupBox("Summary")
        summary_layout = QHBoxLayout(summary_group)
        self._lbl_total = self._stat_card(
            "Total Events", "0", None, summary_layout
        )
        self._lbl_high = self._stat_card(
            "High Severity", "0", _SEVERITY_COLORS["high"], summary_layout
        )
        self._lbl_medium = self._stat_card(
            "Medium Severity", "0", _SEVERITY_COLORS["medium"], summary_layout
        )
        self._lbl_low = self._stat_card(
            "Low Severity", "0", _SEVERITY_COLORS["low"], summary_layout
        )
        root.addWidget(summary_group)

        # ---- Pattern Breakdown ----
        pattern_group = QGroupBox("Pattern Breakdown")
        pattern_layout = QVBoxLayout(pattern_group)
        self._pattern_table = self._make_table(
            ["Pattern Name", "Count", "Most Recent", "Severity"],
            pattern_layout,
        )
        root.addWidget(pattern_group)

        # ---- Event Log ----
        event_group = QGroupBox("Event Log")
        event_layout = QVBoxLayout(event_group)
        self._event_table = self._make_table(
            ["Timestamp", "Model", "Provider", "Matches", "Severity", "Badge"],
            event_layout,
        )
        self._event_table.currentCellChanged.connect(self._on_event_selected)

        # Detail area for selected event.
        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(
            "background: #f8f9fa; border: 1px solid #dee2e6; "
            "padding: 8px; border-radius: 4px;"
        )
        self._detail_label.setVisible(False)
        event_layout.addWidget(self._detail_label)
        root.addWidget(event_group)

        # ---- Recommendation banner ----
        self._banner = QLabel("")
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(
            "background: #fff3cd; color: #856404; border: 1px solid #ffc107; "
            "padding: 10px; border-radius: 4px; font-weight: bold;"
        )
        self._banner.setVisible(False)
        root.addWidget(self._banner)

    @staticmethod
    def _stat_card(
        title: str,
        initial: str,
        color: Optional[str],
        parent_layout: QHBoxLayout,
    ) -> QLabel:
        """Create a labelled stat card and return the value label."""
        box = QVBoxLayout()
        header = QLabel(title)
        header.setStyleSheet("color: gray; font-weight: bold;")
        box.addWidget(header)
        value = QLabel(initial)
        style = "font-size: 16px; font-weight: bold;"
        if color:
            style += f" color: {color};"
        value.setStyleSheet(style)
        box.addWidget(value)
        parent_layout.addLayout(box)
        return value

    @staticmethod
    def _make_table(
        columns: list[str], parent_layout: QVBoxLayout
    ) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        parent_layout.addWidget(table)
        return table

    # ------------------------------------------------------------------
    # Data application (runs on main thread via signal)
    # ------------------------------------------------------------------

    def _apply_data(self, data: dict) -> None:
        # Summary cards.
        self._lbl_total.setText(str(data["total"]))
        self._lbl_high.setText(str(data["high"]))
        self._lbl_medium.setText(str(data["medium"]))
        self._lbl_low.setText(str(data["low"]))

        # Pattern breakdown.
        self._fill_pattern_table(data["pattern_rows"])

        # Event log.
        self._fill_event_table(data["event_rows"])

        # Recommendation banner.
        high_count = data["high"]
        if high_count > 0:
            self._banner.setText(
                f"{high_count} request{'s' if high_count != 1 else ''} "
                f"contained sensitive data sent to cloud providers. "
                f"Consider configuring local-first routing for sensitive "
                f"workloads."
            )
            self._banner.setVisible(True)
        else:
            self._banner.setVisible(False)

        self._detail_label.setVisible(False)
        self._refresh_btn.setEnabled(True)

    def _fill_pattern_table(self, rows: list[dict]) -> None:
        t = self._pattern_table
        t.setRowCount(0)
        for row_data in rows:
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(row_data["name"]))
            t.setItem(r, 1, QTableWidgetItem(str(row_data["count"])))
            t.setItem(r, 2, QTableWidgetItem(row_data["most_recent"]))
            sev = row_data["severity"]
            sev_item = QTableWidgetItem(sev.capitalize())
            color = _SEVERITY_COLORS.get(sev)
            if color:
                sev_item.setForeground(QColor(color))
            t.setItem(r, 3, sev_item)

    def _fill_event_table(self, rows: list[dict]) -> None:
        t = self._event_table
        t.setRowCount(0)
        for row_data in rows:
            r = t.rowCount()
            t.insertRow(r)

            # Store event data on the timestamp item for detail view.
            ts_item = QTableWidgetItem(row_data["timestamp"])
            ts_item.setData(Qt.ItemDataRole.UserRole, row_data)
            t.setItem(r, 0, ts_item)

            t.setItem(r, 1, QTableWidgetItem(row_data["model_id"]))
            t.setItem(r, 2, QTableWidgetItem(row_data["provider"]))
            t.setItem(r, 3, QTableWidgetItem(str(row_data["match_count"])))

            max_sev = _max_severity(row_data["severities"])
            sev_item = QTableWidgetItem(max_sev.capitalize())
            color = _SEVERITY_COLORS.get(max_sev)
            if color:
                sev_item.setForeground(QColor(color))
            t.setItem(r, 4, sev_item)

            # Badge column.
            if max_sev == "high":
                badge = QLabel("Consider Local Routing")
                badge.setStyleSheet(
                    f"background: {_SEVERITY_COLORS['high']}; color: white; "
                    "padding: 2px 6px; border-radius: 3px; font-size: 11px; "
                    "font-weight: bold;"
                )
                t.setCellWidget(r, 5, badge)
            else:
                t.setItem(r, 5, QTableWidgetItem(""))

    def _on_event_selected(
        self, row: int, _col: int, _prev_row: int, _prev_col: int
    ) -> None:
        """Show match details when an event row is selected."""
        if row < 0:
            self._detail_label.setVisible(False)
            return

        item = self._event_table.item(row, 0)
        if item is None:
            self._detail_label.setVisible(False)
            return

        ev = item.data(Qt.ItemDataRole.UserRole)
        if ev is None:
            self._detail_label.setVisible(False)
            return

        lines = []
        for i, name in enumerate(ev["pattern_names"]):
            sev = (
                ev["severities"][i]
                if i < len(ev["severities"])
                else "unknown"
            )
            lines.append(f"  \u2022 {name} ({sev})")

        self._detail_label.setText(
            f"Matches ({ev['match_count']}):\n" + "\n".join(lines)
        )
        self._detail_label.setVisible(True)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _on_worker_error(self, message: str) -> None:
        self._refresh_btn.setEnabled(True)

    def _cleanup_thread(self) -> None:
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
