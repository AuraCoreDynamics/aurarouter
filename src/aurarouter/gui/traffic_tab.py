"""Token Traffic Monitor tab — real-time token usage dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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

from aurarouter.config import ConfigLoader
from aurarouter.gui._format import format_cost, format_duration, format_tokens
from aurarouter.savings.pricing import CostEngine, resolve_hosting_tier
from aurarouter.savings.usage_store import UsageStore

# Map combo-box labels → callables that return (start_iso, end_iso | None).
_TIME_RANGES = [
    "Last Hour",
    "Today",
    "This Week",
    "This Month",
    "All Time",
]


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


# ------------------------------------------------------------------
# Background worker
# ------------------------------------------------------------------

class _TrafficWorker(QObject):
    """Queries UsageStore / CostEngine off the main thread."""

    finished = Signal(dict)  # payload with all computed data
    error = Signal(str)

    def __init__(
        self,
        store: UsageStore,
        engine: CostEngine,
        time_range: str,
        config: Optional[ConfigLoader] = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._engine = engine
        self._time_range = time_range
        self._config = config

    def run(self) -> None:
        try:
            start, end = _time_range_bounds(self._time_range)

            records = self._store.query(start=start, end=end)
            totals = self._store.total_tokens(start=start, end=end)
            total_cost = self._engine.total_spend(start=start, end=end)

            # --- Provider × Model breakdown ---
            provider_model: dict[tuple[str, str], dict] = {}
            for r in records:
                key = (r.provider, r.model_id)
                bucket = provider_model.setdefault(
                    key,
                    {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0,
                        "total_elapsed": 0.0,
                    },
                )
                bucket["requests"] += 1
                bucket["input_tokens"] += r.input_tokens
                bucket["output_tokens"] += r.output_tokens
                bucket["cost"] += self._engine.calculate_cost(
                    r.input_tokens, r.output_tokens, r.model_id, r.provider
                )
                bucket["total_elapsed"] += r.elapsed_s

            provider_rows = []
            for (prov, model), stats in sorted(provider_model.items()):
                avg_latency = (
                    stats["total_elapsed"] / stats["requests"]
                    if stats["requests"]
                    else 0.0
                )
                # Resolve hosting tier from config if available.
                tier_explicit = None
                if self._config is not None:
                    tier_explicit = self._config.get_model_hosting_tier(model)
                tier = resolve_hosting_tier(tier_explicit, prov)
                provider_rows.append(
                    {
                        "provider": prov,
                        "model": model,
                        "requests": stats["requests"],
                        "input_tokens": stats["input_tokens"],
                        "output_tokens": stats["output_tokens"],
                        "cost": stats["cost"],
                        "avg_latency": avg_latency,
                        "tier": tier,
                    }
                )

            # --- Intent breakdown ---
            intent_map: dict[str, dict] = {}
            for r in records:
                bucket = intent_map.setdefault(
                    r.intent, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
                )
                bucket["requests"] += 1
                bucket["input_tokens"] += r.input_tokens
                bucket["output_tokens"] += r.output_tokens

            intent_rows = [
                {"intent": k, **v} for k, v in sorted(intent_map.items())
            ]

            # --- Role breakdown ---
            role_map: dict[str, dict] = {}
            for r in records:
                bucket = role_map.setdefault(
                    r.role, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
                )
                bucket["requests"] += 1
                bucket["input_tokens"] += r.input_tokens
                bucket["output_tokens"] += r.output_tokens

            role_rows = [
                {"role": k, **v} for k, v in sorted(role_map.items())
            ]

            self.finished.emit(
                {
                    "total_requests": len(records),
                    "input_tokens": totals["input_tokens"],
                    "output_tokens": totals["output_tokens"],
                    "total_cost": total_cost,
                    "provider_rows": provider_rows,
                    "intent_rows": intent_rows,
                    "role_rows": role_rows,
                }
            )

        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# Token Traffic Monitor tab
# ------------------------------------------------------------------

class TokenTrafficTab(QWidget):
    """Dashboard showing token usage broken down by provider, intent, and role."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._store: Optional[UsageStore] = None
        self._engine: Optional[CostEngine] = None
        self._config: Optional[ConfigLoader] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_TrafficWorker] = None

        self._build_ui()

        # Auto-refresh timer (30 s).
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(30_000)
        self._auto_timer.timeout.connect(self._on_auto_tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data_sources(
        self,
        usage_store: UsageStore,
        cost_engine: CostEngine,
        config: Optional[ConfigLoader] = None,
    ) -> None:
        """Inject data layer references (called after construction)."""
        self._store = usage_store
        self._engine = cost_engine
        self._config = config

    def refresh(self) -> None:
        """Kick off a background query with the current time-range filter."""
        if self._store is None or self._engine is None:
            return

        # Don't stack concurrent refreshes.
        if self._thread is not None and self._thread.isRunning():
            return

        time_range = self._range_combo.currentText()
        self._worker = _TrafficWorker(self._store, self._engine, time_range, self._config)
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

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self._refresh_btn)

        self._auto_cb = QCheckBox("Auto-refresh")
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        top.addWidget(self._auto_cb)
        top.addStretch()
        root.addLayout(top)

        # ---- Summary row ----
        summary_group = QGroupBox("Summary")
        summary_layout = QHBoxLayout(summary_group)
        self._lbl_requests = self._stat_card("Total Requests", "0", summary_layout)
        self._lbl_input = self._stat_card("Input Tokens", "0", summary_layout)
        self._lbl_output = self._stat_card("Output Tokens", "0", summary_layout)
        self._lbl_cost = self._stat_card("Total Cost", "$0.00", summary_layout)
        root.addWidget(summary_group)

        # ---- Provider breakdown ----
        prov_group = QGroupBox("Provider Breakdown")
        prov_layout = QVBoxLayout(prov_group)
        self._provider_table = self._make_table(
            ["Provider", "Model", "Tier", "Requests", "Input Tokens", "Output Tokens", "Cost", "Avg Latency"],
            prov_layout,
        )
        root.addWidget(prov_group)

        # ---- Intent breakdown ----
        intent_group = QGroupBox("Intent Breakdown")
        intent_layout = QVBoxLayout(intent_group)
        self._intent_table = self._make_table(
            ["Intent", "Requests", "Input Tokens", "Output Tokens"],
            intent_layout,
        )
        root.addWidget(intent_group)

        # ---- Role breakdown ----
        role_group = QGroupBox("Role Breakdown")
        role_layout = QVBoxLayout(role_group)
        self._role_table = self._make_table(
            ["Role", "Requests", "Input Tokens", "Output Tokens"],
            role_layout,
        )
        root.addWidget(role_group)

    @staticmethod
    def _stat_card(
        title: str, initial: str, parent_layout: QHBoxLayout
    ) -> QLabel:
        """Create a labelled stat card and return the value label."""
        box = QVBoxLayout()
        header = QLabel(title)
        header.setStyleSheet("color: gray; font-weight: bold;")
        box.addWidget(header)
        value = QLabel(initial)
        value.setStyleSheet("font-size: 16px; font-weight: bold;")
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
        self._lbl_requests.setText(format_tokens(data["total_requests"]))
        self._lbl_input.setText(format_tokens(data["input_tokens"]))
        self._lbl_output.setText(format_tokens(data["output_tokens"]))
        self._lbl_cost.setText(format_cost(data["total_cost"]))

        # Provider table.
        self._fill_provider_table(data["provider_rows"])

        # Intent table.
        self._fill_simple_table(
            self._intent_table,
            data["intent_rows"],
            ["intent", "requests", "input_tokens", "output_tokens"],
        )

        # Role table.
        self._fill_simple_table(
            self._role_table,
            data["role_rows"],
            ["role", "requests", "input_tokens", "output_tokens"],
        )

        self._refresh_btn.setEnabled(True)

    def _fill_provider_table(self, rows: list[dict]) -> None:
        t = self._provider_table
        t.setRowCount(0)
        for row_data in rows:
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(row_data["provider"]))
            t.setItem(r, 1, QTableWidgetItem(row_data["model"]))
            t.setItem(r, 2, QTableWidgetItem(row_data.get("tier", "")))
            t.setItem(r, 3, QTableWidgetItem(str(row_data["requests"])))
            t.setItem(r, 4, QTableWidgetItem(format_tokens(row_data["input_tokens"])))
            t.setItem(r, 5, QTableWidgetItem(format_tokens(row_data["output_tokens"])))
            t.setItem(r, 6, QTableWidgetItem(format_cost(row_data["cost"])))
            t.setItem(r, 7, QTableWidgetItem(format_duration(row_data["avg_latency"])))

    def _fill_simple_table(
        self,
        table: QTableWidget,
        rows: list[dict],
        keys: list[str],
    ) -> None:
        table.setRowCount(0)
        for row_data in rows:
            r = table.rowCount()
            table.insertRow(r)
            for col, key in enumerate(keys):
                val = row_data[key]
                if key in ("input_tokens", "output_tokens"):
                    text = format_tokens(val)
                else:
                    text = str(val)
                table.setItem(r, col, QTableWidgetItem(text))

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------

    def _on_auto_toggled(self, checked: bool) -> None:
        if checked:
            self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _on_auto_tick(self) -> None:
        """Only refresh when the tab is actually visible."""
        if self.isVisible():
            self.refresh()

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
