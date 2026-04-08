"""Unified observability dashboard -- traffic, privacy, health.

Consolidates the previously separate traffic, privacy, and health views
into a single ``MonitorPanel`` with shared time-range controls, summary
cards that are always visible, and four sub-panels accessible via a left
mini-nav: Overview, Traffic, Privacy, and Health.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aurarouter.api import AuraRouterAPI, HealthReport
from aurarouter.gui._format import format_cost, format_duration, format_tokens
from aurarouter.gui.theme import DARK_PALETTE, RADIUS, SPACING, TYPOGRAPHY
from aurarouter.gui.widgets.help_tooltip import HelpTooltip
from aurarouter.gui.widgets.stat_card import StatCard
from aurarouter.gui.widgets.status_badge import StatusBadge


# ======================================================================
# Constants
# ======================================================================

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
    "high": DARK_PALETTE.error,
    "medium": DARK_PALETTE.warning,
    "low": DARK_PALETTE.info,
}

_NAV_ITEMS = ["Overview", "Traffic", "Privacy", "Health", "ROI & Telemetry"]


# ======================================================================
# Helpers
# ======================================================================


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


# ======================================================================
# Background workers
# ======================================================================


class _MonitorWorker(QObject):
    """Fetches traffic, privacy, and budget data off the main thread."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        api: AuraRouterAPI,
        time_range: str,
        severity_filter: str,
    ) -> None:
        super().__init__()
        self._api = api
        self._time_range = time_range
        self._severity_filter = severity_filter

    def run(self) -> None:
        try:
            start, end = _time_range_bounds(self._time_range)
            tr = (start, end) if start else None

            # Traffic
            traffic = self._api.get_traffic(time_range=tr)

            # Privacy
            sev = (
                self._severity_filter.lower()
                if self._severity_filter != "All"
                else None
            )
            privacy = self._api.get_privacy_events(time_range=tr, severity=sev)

            # Budget
            budget = self._api.get_budget_status()

            # ROI
            if self._time_range == "Last Hour":
                days = 1
            elif self._time_range == "Today":
                days = 1
            elif self._time_range == "This Week":
                days = 7
            elif self._time_range == "This Month":
                days = 30
            else:
                days = 365
            import dataclasses
            roi = self._api.get_roi_metrics(days)
            roi_dict = dataclasses.asdict(roi) if roi else {}

            self.finished.emit({
                "traffic": traffic,
                "privacy": privacy,
                "budget": budget,
                "roi": roi_dict,
            })
        except Exception as exc:
            self.error.emit(str(exc))


class _HealthWorker(QObject):
    """Runs health checks off the main thread."""

    finished = Signal(list)  # list[HealthReport]
    error = Signal(str)

    def __init__(self, api: AuraRouterAPI, model_id: Optional[str] = None) -> None:
        super().__init__()
        self._api = api
        self._model_id = model_id

    def run(self) -> None:
        try:
            reports = self._api.check_health(model_id=self._model_id)
            self.finished.emit(reports)
        except Exception as exc:
            self.error.emit(str(exc))


# ======================================================================
# Helper: sortable table
# ======================================================================


def _make_sortable_table(
    columns: list[str], parent_layout: QVBoxLayout
) -> QTableWidget:
    """Create a QTableWidget with sortable headers."""
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSortingEnabled(True)
    parent_layout.addWidget(table)
    return table


# ======================================================================
# Sub-panels
# ======================================================================


class _OverviewSubPanel(QWidget):
    """Overview: top models, budget, active alerts."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- Top N models by request count --
        models_group = QGroupBox("Top Models by Request Count")
        self._models_layout = QVBoxLayout(models_group)
        self._model_bars: list[tuple[QLabel, QProgressBar]] = []
        layout.addWidget(models_group)

        # -- Budget status --
        self._budget_group = QGroupBox("Budget Status")
        self._budget_layout = QVBoxLayout(self._budget_group)
        self._daily_label = QLabel("Daily: --")
        self._monthly_label = QLabel("Monthly: --")
        self._daily_bar = QProgressBar()
        self._daily_bar.setRange(0, 100)
        self._monthly_bar = QProgressBar()
        self._monthly_bar.setRange(0, 100)
        self._budget_layout.addWidget(self._daily_label)
        self._budget_layout.addWidget(self._daily_bar)
        self._budget_layout.addWidget(self._monthly_label)
        self._budget_layout.addWidget(self._monthly_bar)
        self._budget_group.setVisible(False)
        layout.addWidget(self._budget_group)

        # -- Active alerts --
        alerts_group = QGroupBox("Active Alerts")
        self._alerts_layout = QVBoxLayout(alerts_group)
        self._alerts_label = QLabel("No active alerts.")
        self._alerts_label.setWordWrap(True)
        self._alerts_layout.addWidget(self._alerts_label)
        layout.addWidget(alerts_group)

        layout.addStretch()

    def update_data(
        self,
        traffic: object,
        privacy: object,
        budget: Optional[dict],
        health_reports: list[HealthReport],
    ) -> None:
        """Refresh overview from combined data."""
        # -- Top models --
        # Clear previous bars.
        for lbl, bar in self._model_bars:
            self._models_layout.removeWidget(lbl)
            lbl.deleteLater()
            self._models_layout.removeWidget(bar)
            bar.deleteLater()
        self._model_bars.clear()

        by_model = sorted(
            traffic.by_model, key=lambda m: m.get("total_tokens", 0), reverse=True
        )[:10]
        max_tok = by_model[0].get("total_tokens", 1) if by_model else 1

        for entry in by_model:
            model_id = entry.get("model_id", entry.get("group", "unknown"))
            total = entry.get("total_tokens", 0)
            lbl = QLabel(f"{model_id}  ({format_tokens(total)} tokens)")
            bar = QProgressBar()
            bar.setRange(0, max_tok)
            bar.setValue(total)
            bar.setTextVisible(False)
            bar.setMaximumHeight(14)
            self._models_layout.addWidget(lbl)
            self._models_layout.addWidget(bar)
            self._model_bars.append((lbl, bar))

        # -- Budget --
        if budget is not None:
            self._budget_group.setVisible(True)
            daily_spend = budget.get("daily_spend", 0.0)
            monthly_spend = budget.get("monthly_spend", 0.0)
            daily_limit = budget.get("daily_limit", 0.0)
            monthly_limit = budget.get("monthly_limit", 0.0)

            self._daily_label.setText(
                f"Daily: {format_cost(daily_spend)} / {format_cost(daily_limit)}"
            )
            self._monthly_label.setText(
                f"Monthly: {format_cost(monthly_spend)} / {format_cost(monthly_limit)}"
            )

            if daily_limit > 0:
                self._daily_bar.setValue(int(min(daily_spend / daily_limit, 1.0) * 100))
            else:
                self._daily_bar.setValue(0)

            if monthly_limit > 0:
                self._monthly_bar.setValue(int(min(monthly_spend / monthly_limit, 1.0) * 100))
            else:
                self._monthly_bar.setValue(0)
        else:
            self._budget_group.setVisible(False)

        # -- Active alerts --
        alert_lines: list[str] = []
        high_count = privacy.by_severity.get("high", 0)
        if high_count > 0:
            alert_lines.append(
                f"\u26a0 {high_count} high-severity privacy "
                f"event{'s' if high_count != 1 else ''}"
            )

        for report in health_reports:
            if not report.healthy:
                alert_lines.append(f"\u274c {report.model_id}: {report.message}")

        if alert_lines:
            self._alerts_label.setText("\n".join(alert_lines))
            self._alerts_label.setStyleSheet(
                f"color: {DARK_PALETTE.error}; font-weight: bold;"
            )
        else:
            self._alerts_label.setText("No active alerts.")
            self._alerts_label.setStyleSheet("")


class _TrafficSubPanel(QWidget):
    """Traffic: provider, intent, and role breakdown tables."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Provider breakdown
        prov_group = QGroupBox("Provider Breakdown")
        prov_layout = QVBoxLayout(prov_group)
        self._provider_table = _make_sortable_table(
            ["Provider", "Model", "Tier", "Requests", "Input Tokens",
             "Output Tokens", "Cost", "Avg Latency"],
            prov_layout,
        )
        layout.addWidget(prov_group)

        # Intent breakdown
        intent_group = QGroupBox("Intent Breakdown")
        intent_layout = QVBoxLayout(intent_group)
        self._intent_table = _make_sortable_table(
            ["Intent", "Requests", "Input Tokens", "Output Tokens"],
            intent_layout,
        )
        layout.addWidget(intent_group)

        # Role breakdown
        role_group = QGroupBox("Role Breakdown")
        role_layout = QVBoxLayout(role_group)
        self._role_table = _make_sortable_table(
            ["Role", "Requests", "Input Tokens", "Output Tokens"],
            role_layout,
        )
        layout.addWidget(role_group)

    def update_data(self, traffic: object) -> None:
        """Populate tables from TrafficSummary."""
        # Provider table — by_model entries contain per-model breakdowns.
        t = self._provider_table
        t.setSortingEnabled(False)
        t.setRowCount(0)
        for entry in traffic.by_model:
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(entry.get("provider", "")))
            t.setItem(r, 1, QTableWidgetItem(
                entry.get("model_id", entry.get("group", ""))
            ))
            t.setItem(r, 2, QTableWidgetItem(entry.get("tier", "")))

            req_item = QTableWidgetItem()
            req_item.setData(Qt.ItemDataRole.DisplayRole, entry.get("requests", 0))
            t.setItem(r, 3, req_item)

            inp_item = QTableWidgetItem(format_tokens(entry.get("input_tokens", 0)))
            inp_item.setData(Qt.ItemDataRole.UserRole, entry.get("input_tokens", 0))
            t.setItem(r, 4, inp_item)

            out_item = QTableWidgetItem(format_tokens(entry.get("output_tokens", 0)))
            out_item.setData(Qt.ItemDataRole.UserRole, entry.get("output_tokens", 0))
            t.setItem(r, 5, out_item)

            cost_item = QTableWidgetItem(format_cost(entry.get("cost", 0.0)))
            cost_item.setData(Qt.ItemDataRole.UserRole, entry.get("cost", 0.0))
            t.setItem(r, 6, cost_item)

            lat = entry.get("avg_latency", 0.0)
            lat_item = QTableWidgetItem(format_duration(lat))
            lat_item.setData(Qt.ItemDataRole.UserRole, lat)
            t.setItem(r, 7, lat_item)
        t.setSortingEnabled(True)

        # Intent table
        self._fill_breakdown(
            self._intent_table,
            traffic.by_model,
            group_key="intent",
        )

        # Role table
        self._fill_breakdown(
            self._role_table,
            traffic.by_model,
            group_key="role",
        )

    def _fill_breakdown(
        self, table: QTableWidget, entries: list[dict], group_key: str
    ) -> None:
        """Aggregate entries by *group_key* and fill the table."""
        agg: dict[str, dict] = {}
        for entry in entries:
            key = entry.get(group_key, "unknown")
            bucket = agg.setdefault(
                key, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
            )
            bucket["requests"] += entry.get("requests", 0)
            bucket["input_tokens"] += entry.get("input_tokens", 0)
            bucket["output_tokens"] += entry.get("output_tokens", 0)

        table.setSortingEnabled(False)
        table.setRowCount(0)
        for key, stats in sorted(agg.items()):
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(key))

            req_item = QTableWidgetItem()
            req_item.setData(Qt.ItemDataRole.DisplayRole, stats["requests"])
            table.setItem(r, 1, req_item)

            inp_item = QTableWidgetItem(format_tokens(stats["input_tokens"]))
            inp_item.setData(Qt.ItemDataRole.UserRole, stats["input_tokens"])
            table.setItem(r, 2, inp_item)

            out_item = QTableWidgetItem(format_tokens(stats["output_tokens"]))
            out_item.setData(Qt.ItemDataRole.UserRole, stats["output_tokens"])
            table.setItem(r, 3, out_item)
        table.setSortingEnabled(True)


class _PrivacySubPanel(QWidget):
    """Privacy: severity filter, pattern breakdown, event log, detail view."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._events: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Severity filter row (panel-level time range is shared via parent)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Severity:"))
        self._severity_combo = QComboBox()
        self._severity_combo.addItems(_SEVERITY_FILTERS)
        filter_row.addWidget(self._severity_combo)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        # Pattern breakdown
        pattern_group = QGroupBox("Pattern Breakdown")
        pattern_layout = QVBoxLayout(pattern_group)
        self._pattern_table = _make_sortable_table(
            ["Pattern Name", "Count", "Most Recent", "Severity"],
            pattern_layout,
        )
        layout.addWidget(pattern_group)

        # Event log
        event_group = QGroupBox("Event Log")
        event_layout = QVBoxLayout(event_group)
        self._event_table = _make_sortable_table(
            ["Timestamp", "Model", "Provider", "Matches", "Severity", "Badge"],
            event_layout,
        )
        self._event_table.currentCellChanged.connect(self._on_event_selected)

        # Detail area
        self._detail_label = QLabel("")
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(
            f"background: {DARK_PALETTE.bg_tertiary}; "
            f"border: 1px solid {DARK_PALETTE.border}; "
            f"padding: 8px; border-radius: {RADIUS.sm}px;"
        )
        self._detail_label.setVisible(False)
        event_layout.addWidget(self._detail_label)
        layout.addWidget(event_group)

        # Recommendation banner
        self._banner = QLabel("")
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(
            f"background: {DARK_PALETTE.bg_tertiary}; color: {DARK_PALETTE.warning}; "
            f"border: 1px solid {DARK_PALETTE.warning}; "
            f"padding: 10px; border-radius: {RADIUS.sm}px; font-weight: bold;"
        )
        self._banner.setVisible(False)
        layout.addWidget(self._banner)

    @property
    def severity_combo(self) -> QComboBox:
        """Expose severity combo for parent to connect signals."""
        return self._severity_combo

    def update_data(self, privacy: object) -> None:
        """Populate tables from PrivacySummary."""
        events = privacy.events
        self._events = events

        # -- Pattern breakdown --
        pattern_data: dict[str, dict] = {}
        for ev in events:
            for i, name in enumerate(ev.get("pattern_names", [])):
                sev = (
                    ev["severities"][i]
                    if i < len(ev.get("severities", []))
                    else "low"
                )
                bucket = pattern_data.setdefault(
                    name,
                    {"count": 0, "most_recent": "", "severity": sev},
                )
                bucket["count"] += 1
                ts = ev.get("timestamp", "")
                if ts > bucket["most_recent"]:
                    bucket["most_recent"] = ts
                if _SEVERITY_RANK.get(sev, 0) > _SEVERITY_RANK.get(
                    bucket["severity"], 0
                ):
                    bucket["severity"] = sev

        pattern_rows = sorted(
            [{"name": k, **v} for k, v in pattern_data.items()],
            key=lambda r: r["count"],
            reverse=True,
        )

        t = self._pattern_table
        t.setSortingEnabled(False)
        t.setRowCount(0)
        for row_data in pattern_rows:
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(row_data["name"]))
            count_item = QTableWidgetItem()
            count_item.setData(Qt.ItemDataRole.DisplayRole, row_data["count"])
            t.setItem(r, 1, count_item)
            t.setItem(r, 2, QTableWidgetItem(row_data["most_recent"]))
            sev = row_data["severity"]
            sev_item = QTableWidgetItem(sev.capitalize())
            color = _SEVERITY_COLORS.get(sev)
            if color:
                sev_item.setForeground(QColor(color))
            t.setItem(r, 3, sev_item)
        t.setSortingEnabled(True)

        # -- Event log --
        event_rows = sorted(events, key=lambda e: e.get("timestamp", ""), reverse=True)

        et = self._event_table
        et.setSortingEnabled(False)
        et.setRowCount(0)
        for row_data in event_rows:
            r = et.rowCount()
            et.insertRow(r)

            ts_item = QTableWidgetItem(row_data.get("timestamp", ""))
            ts_item.setData(Qt.ItemDataRole.UserRole, row_data)
            et.setItem(r, 0, ts_item)
            et.setItem(r, 1, QTableWidgetItem(row_data.get("model_id", "")))
            et.setItem(r, 2, QTableWidgetItem(row_data.get("provider", "")))

            match_item = QTableWidgetItem()
            match_item.setData(
                Qt.ItemDataRole.DisplayRole, row_data.get("match_count", 0)
            )
            et.setItem(r, 3, match_item)

            max_sev = _max_severity(row_data.get("severities", []))
            sev_item = QTableWidgetItem(max_sev.capitalize())
            color = _SEVERITY_COLORS.get(max_sev)
            if color:
                sev_item.setForeground(QColor(color))
            et.setItem(r, 4, sev_item)

            if max_sev == "high":
                badge = QLabel("Consider Local Routing")
                badge.setStyleSheet(
                    f"background: {DARK_PALETTE.error}; color: {DARK_PALETTE.text_inverse}; "
                    f"padding: 2px 6px; border-radius: 3px; font-size: {TYPOGRAPHY.size_body}px; "
                    "font-weight: bold;"
                )
                et.setCellWidget(r, 5, badge)
            else:
                et.setItem(r, 5, QTableWidgetItem(""))
        et.setSortingEnabled(True)

        # -- Recommendation banner --
        high_count = privacy.by_severity.get("high", 0)
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

    def _on_event_selected(
        self, row: int, _col: int, _prev_row: int, _prev_col: int
    ) -> None:
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
        for i, name in enumerate(ev.get("pattern_names", [])):
            sev = (
                ev["severities"][i]
                if i < len(ev.get("severities", []))
                else "unknown"
            )
            lines.append(f"  \u2022 {name} ({sev})")

        self._detail_label.setText(
            f"Matches ({ev.get('match_count', 0)}):\n" + "\n".join(lines)
        )
        self._detail_label.setVisible(True)


class _HealthSubPanel(QWidget):
    """Health: per-model cards with status badges and check buttons."""

    check_requested = Signal(str)   # model_id (empty string = all)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._card_widgets: list[QWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Check All button
        btn_row = QHBoxLayout()
        self._check_all_btn = QPushButton("Check All")
        self._check_all_btn.clicked.connect(lambda: self.check_requested.emit(""))
        btn_row.addWidget(self._check_all_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Scrollable card area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.addStretch()

        self._scroll.setWidget(self._cards_container)
        root.addWidget(self._scroll)

    def update_data(self, reports: list[HealthReport]) -> None:
        """Rebuild health cards from a list of HealthReport."""
        # Remove old cards.
        for w in self._card_widgets:
            self._cards_layout.removeWidget(w)
            w.deleteLater()
        self._card_widgets.clear()

        for report in reports:
            card = self._make_card(report)
            # Insert before the stretch.
            self._cards_layout.insertWidget(
                self._cards_layout.count() - 1, card
            )
            self._card_widgets.append(card)

    def _make_card(self, report: HealthReport) -> QWidget:
        """Build a single health card widget."""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {DARK_PALETTE.bg_secondary};"
            f"  border: 1px solid {DARK_PALETTE.border};"
            f"  border-radius: {RADIUS.md}px;"
            f"}}"
        )

        layout = QHBoxLayout(card)
        layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)

        # Info column
        info = QVBoxLayout()
        model_label = QLabel(report.model_id)
        model_label.setStyleSheet(
            f"font-weight: bold; font-size: {TYPOGRAPHY.size_h2}px; border: none;"
        )
        info.addWidget(model_label)

        detail_text = ""
        if report.message and report.message != "OK":
            detail_text = report.message
        if report.latency > 0:
            detail_text += f"  |  Latency: {format_duration(report.latency)}"
        if detail_text:
            detail = QLabel(detail_text.strip(" |"))
            detail.setStyleSheet(
                f"color: {DARK_PALETTE.text_secondary}; "
                f"font-size: {TYPOGRAPHY.size_small}px; border: none;"
            )
            detail.setWordWrap(True)
            info.addWidget(detail)

        layout.addLayout(info, stretch=1)

        # Status badge
        if report.healthy:
            badge = StatusBadge(mode="healthy")
        else:
            if report.message:
                badge = StatusBadge(mode="unhealthy")
            else:
                badge = StatusBadge(mode="stopped", text="Untested")
        layout.addWidget(badge)

        # Individual check button
        check_btn = QPushButton("Check")
        check_btn.setFixedWidth(60)
        mid = report.model_id
        check_btn.clicked.connect(lambda _checked, m=mid: self.check_requested.emit(m))
        layout.addWidget(check_btn)

        return card


class _RoiSubPanel(QWidget):
    """ROI & Telemetry: visualizing cost savings and local routing."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING)

        # -- Top row: Summary Cards --
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(SPACING)

        self.lbl_dollars_saved = QLabel("$0.00")
        self.lbl_dollars_saved.setFont(TYPOGRAPHY.h1)
        self.lbl_dollars_saved.setStyleSheet(f"color: {DARK_PALETTE.success};")
        card_dollars = StatCard(
            title="Total Dollars Saved",
            value_widget=self.lbl_dollars_saved,
            tooltip="Total simulated USD cost avoided by routing to local models.",
        )
        cards_layout.addWidget(card_dollars)

        self.lbl_ratio = QLabel("0.0%")
        self.lbl_ratio.setFont(TYPOGRAPHY.h1)
        card_ratio = StatCard(
            title="Hard-Route Ratio",
            value_widget=self.lbl_ratio,
            tooltip="Percentage of traffic routed to local models.",
        )
        cards_layout.addWidget(card_ratio)

        layout.addLayout(cards_layout)

        # -- Middle row: Complexity Chart --
        gb_complexity = QGroupBox("Complexity Distribution")
        gb_layout = QVBoxLayout(gb_complexity)
        
        self.prog_complexity = QProgressBar()
        self.prog_complexity.setTextVisible(True)
        self.prog_complexity.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {DARK_PALETTE.border};
                border-radius: {RADIUS}px;
                background-color: {DARK_PALETTE.surface};
                text-align: center;
                color: {DARK_PALETTE.text};
            }}
            QProgressBar::chunk {{
                background-color: {DARK_PALETTE.accent};
                border-radius: {RADIUS - 1}px;
            }}
        """)
        self.prog_complexity.setFormat("Local Routing Rate: %p%")
        gb_layout.addWidget(self.prog_complexity)
        layout.addWidget(gb_complexity)

        # -- Bottom row: Recent Hard-Routed Tasks --
        gb_table = QGroupBox("Recent Hard-Routed Tasks (Top 50)")
        table_layout = QVBoxLayout(gb_table)
        self.table = _make_sortable_table(
            ["Timestamp", "Model", "Complexity", "Savings", "Latency"], table_layout
        )
        layout.addWidget(gb_table)

    def populate(self, metrics: dict) -> None:
        """Update ROI stats."""
        saved = metrics.get("total_simulated_cost_avoided", 0.0)
        ratio = metrics.get("hard_route_percentage", 0.0)

        self.lbl_dollars_saved.setText(format_cost(saved))
        self.lbl_ratio.setText(f"{ratio:.1f}%")
        self.prog_complexity.setValue(int(ratio))

        recent = metrics.get("recent_hard_routed", [])
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(recent))
        for row, item in enumerate(recent):
            self.table.setItem(row, 0, QTableWidgetItem(item["timestamp"]))
            self.table.setItem(row, 1, QTableWidgetItem(item["model_id"]))
            
            c_item = QTableWidgetItem()
            c_item.setData(Qt.ItemDataRole.DisplayRole, item["complexity"])
            self.table.setItem(row, 2, c_item)

            s_item = QTableWidgetItem()
            s_item.setData(Qt.ItemDataRole.DisplayRole, item["savings"])
            self.table.setItem(row, 3, s_item)

            l_item = QTableWidgetItem()
            l_item.setData(Qt.ItemDataRole.DisplayRole, item["latency"])
            self.table.setItem(row, 4, l_item)

        self.table.setSortingEnabled(True)

# ======================================================================
# Main panel
# ======================================================================


class MonitorPanel(QWidget):
    """Unified observability dashboard -- traffic, privacy, health."""

    def __init__(
        self,
        api: AuraRouterAPI,
        help_registry=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._api = api
        self._help_registry = help_registry

        # Worker state
        self._monitor_thread: Optional[QThread] = None
        self._monitor_worker: Optional[_MonitorWorker] = None
        self._health_thread: Optional[QThread] = None
        self._health_worker: Optional[_HealthWorker] = None

        # Latest health reports for overview cross-referencing
        self._health_reports: list[HealthReport] = []

        # Latest traffic/privacy for overview cross-referencing
        self._last_traffic = None
        self._last_privacy = None
        self._last_budget = None

        self._build_ui()

        # Auto-refresh timer (30 s).
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(30_000)
        self._auto_timer.timeout.connect(self._on_auto_tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Kick off a background data fetch with current filters."""
        if self._monitor_thread is not None and self._monitor_thread.isRunning():
            return

        time_range = self._range_combo.currentText()
        severity_filter = self._privacy_panel.severity_combo.currentText()

        self._monitor_worker = _MonitorWorker(self._api, time_range, severity_filter)
        self._monitor_thread = QThread()
        self._monitor_worker.moveToThread(self._monitor_thread)

        self._monitor_thread.started.connect(self._monitor_worker.run)
        self._monitor_worker.finished.connect(self._apply_monitor_data)
        self._monitor_worker.error.connect(self._on_monitor_error)

        self._monitor_worker.finished.connect(self._monitor_thread.quit)
        self._monitor_worker.error.connect(self._monitor_thread.quit)
        self._monitor_thread.finished.connect(self._cleanup_monitor_thread)

        self._refresh_btn.setEnabled(False)
        self._monitor_thread.start()

    def check_health(self, model_id: Optional[str] = None) -> None:
        """Run health checks in the background."""
        if self._health_thread is not None and self._health_thread.isRunning():
            return

        mid = model_id if model_id else None
        self._health_worker = _HealthWorker(self._api, model_id=mid)
        self._health_thread = QThread()
        self._health_worker.moveToThread(self._health_thread)

        self._health_thread.started.connect(self._health_worker.run)
        self._health_worker.finished.connect(self._apply_health_data)
        self._health_worker.error.connect(self._on_health_error)

        self._health_worker.finished.connect(self._health_thread.quit)
        self._health_worker.error.connect(self._health_thread.quit)
        self._health_thread.finished.connect(self._cleanup_health_thread)

        self._health_thread.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(SPACING.sm)

        # ---- Top control bar ----
        top = QHBoxLayout()
        top.addWidget(QLabel("Time Range:"))
        self._range_combo = QComboBox()
        self._range_combo.addItems(_TIME_RANGES)
        self._range_combo.currentIndexChanged.connect(lambda _: self.refresh())
        top.addWidget(self._range_combo)

        self._auto_cb = QCheckBox("Auto-refresh")
        self._auto_cb.toggled.connect(self._on_auto_toggled)
        top.addWidget(self._auto_cb)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self._refresh_btn)

        # Help button
        help_text = "Unified monitoring dashboard for traffic, privacy, and health."
        if self._help_registry:
            topic = self._help_registry.get("panel.monitor")
            if topic:
                help_text = topic.body
        top.addWidget(HelpTooltip(help_text))

        top.addStretch()
        root.addLayout(top)

        # ---- Summary cards row ----
        cards_row = QHBoxLayout()
        self._card_requests = StatCard(
            title="Total Requests", value="0", accent_color="accent"
        )
        self._card_tokens = StatCard(
            title="Total Tokens", value="0", accent_color="info"
        )
        self._card_cost = StatCard(
            title="Total Cost", value="$0.00", accent_color="warning"
        )
        self._card_health = StatCard(
            title="Health Status", value="--", accent_color="success"
        )
        for card in (
            self._card_requests,
            self._card_tokens,
            self._card_cost,
            self._card_health,
        ):
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            cards_row.addWidget(card)
        root.addLayout(cards_row)

        # ---- Alert banner ----
        self._alert_banner = QLabel("")
        self._alert_banner.setWordWrap(True)
        self._alert_banner.setStyleSheet(
            f"background: {DARK_PALETTE.bg_tertiary}; color: {DARK_PALETTE.error}; "
            f"border: 1px solid {DARK_PALETTE.error}; "
            f"padding: 10px; border-radius: {RADIUS.sm}px; font-weight: bold;"
        )
        self._alert_banner.setVisible(False)
        root.addWidget(self._alert_banner)

        # ---- Sub-tab area: left mini-nav + stacked content ----
        body = QHBoxLayout()

        # Left mini-nav
        self._nav_list = QListWidget()
        self._nav_list.setFixedWidth(120)
        self._nav_list.setStyleSheet(
            f"QListWidget {{"
            f"  background-color: {DARK_PALETTE.bg_secondary};"
            f"  border: 1px solid {DARK_PALETTE.border};"
            f"  border-radius: {RADIUS.sm}px;"
            f"}}"
            f"QListWidget::item {{"
            f"  padding: {SPACING.sm}px;"
            f"  color: {DARK_PALETTE.text_secondary};"
            f"}}"
            f"QListWidget::item:selected {{"
            f"  background-color: {DARK_PALETTE.bg_selected};"
            f"  color: {DARK_PALETTE.text_primary};"
            f"  font-weight: bold;"
            f"}}"
        )
        for name in _NAV_ITEMS:
            self._nav_list.addItem(QListWidgetItem(name))
        self._nav_list.setCurrentRow(0)
        self._nav_list.currentRowChanged.connect(self._on_nav_changed)
        body.addWidget(self._nav_list)

        # Stacked content
        self._stack = QStackedWidget()

        self._overview_panel = _OverviewSubPanel()
        self._traffic_panel = _TrafficSubPanel()
        self._privacy_panel = _PrivacySubPanel()
        self._health_panel = _HealthSubPanel()
        self._roi_panel = _RoiSubPanel()

        # Wrap each in a scroll area for overflow
        for panel in (
            self._overview_panel,
            self._traffic_panel,
            self._privacy_panel,
            self._health_panel,
            self._roi_panel,
        ):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(panel)
            self._stack.addWidget(scroll)

        body.addWidget(self._stack, stretch=1)
        root.addLayout(body, stretch=1)

        # Connect sub-panel signals
        self._privacy_panel.severity_combo.currentIndexChanged.connect(
            lambda _: self.refresh()
        )
        self._health_panel.check_requested.connect(self._on_health_check_requested)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_nav_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Data application (main thread, via signals)
    # ------------------------------------------------------------------

    def _apply_monitor_data(self, data: dict) -> None:
        traffic = data["traffic"]
        privacy = data["privacy"]
        budget = data["budget"]

        self._last_traffic = traffic
        self._last_privacy = privacy
        self._last_budget = budget

        # Summary cards
        total_requests = len(privacy.events) + traffic.total_tokens  # approximate
        # Use by_model to sum requests if available
        total_req = sum(m.get("requests", 0) for m in traffic.by_model)
        total_tokens = traffic.input_tokens + traffic.output_tokens

        self._card_requests.set_value(format_tokens(total_req))
        self._card_tokens.set_value(format_tokens(total_tokens))
        self._card_cost.set_value(format_cost(traffic.total_spend))

        # Health card shows latest cached state
        self._update_health_card()

        # Alert banner for high-severity privacy events
        high_count = privacy.by_severity.get("high", 0)
        if high_count > 0:
            self._alert_banner.setText(
                f"\u26a0 {high_count} high-severity privacy "
                f"event{'s' if high_count != 1 else ''} detected. "
                f"Review the Privacy tab for details."
            )
            self._alert_banner.setVisible(True)
        else:
            self._alert_banner.setVisible(False)

        # Update sub-panels
        self._overview_panel.update_data(
            traffic, privacy, budget, self._health_reports
        )
        self._traffic_panel.update_data(traffic)
        self._privacy_panel.update_data(privacy)
        self._roi_panel.populate(data.get("roi", {}))

        self._refresh_btn.setEnabled(True)

    def _apply_health_data(self, reports: list) -> None:
        self._health_reports = reports
        self._health_panel.update_data(reports)
        self._update_health_card()

        # Also refresh overview alerts if we have cached data
        if self._last_traffic is not None and self._last_privacy is not None:
            self._overview_panel.update_data(
                self._last_traffic,
                self._last_privacy,
                self._last_budget,
                self._health_reports,
            )

    def _update_health_card(self) -> None:
        """Update the Health Status summary card from cached reports."""
        if not self._health_reports:
            self._card_health.set_value("--")
            self._card_health.set_subtitle("")
            return

        healthy = sum(1 for r in self._health_reports if r.healthy)
        total = len(self._health_reports)
        if healthy == total:
            self._card_health.set_value("All Healthy")
            self._card_health.set_subtitle(f"{total}/{total} models OK")
        else:
            unhealthy = total - healthy
            self._card_health.set_value(f"{unhealthy} Unhealthy")
            self._card_health.set_subtitle(f"{healthy}/{total} models OK")

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------

    def _on_auto_toggled(self, checked: bool) -> None:
        if checked:
            self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _on_auto_tick(self) -> None:
        """Only refresh when the panel is actually visible."""
        if self.isVisible():
            self.refresh()
            # Piggyback health check on auto-refresh
            self.check_health()

    # ------------------------------------------------------------------
    # Health check trigger
    # ------------------------------------------------------------------

    def _on_health_check_requested(self, model_id: str) -> None:
        """Handle check request from health sub-panel."""
        mid = model_id if model_id else None
        self.check_health(model_id=mid)

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    def _on_monitor_error(self, message: str) -> None:
        self._refresh_btn.setEnabled(True)

    def _on_health_error(self, message: str) -> None:
        pass  # Silently handle; user can retry via Check All

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _cleanup_monitor_thread(self) -> None:
        if self._monitor_thread:
            self._monitor_thread.deleteLater()
            self._monitor_thread = None
        if self._monitor_worker:
            self._monitor_worker.deleteLater()
            self._monitor_worker = None

    def _cleanup_health_thread(self) -> None:
        if self._health_thread:
            self._health_thread.deleteLater()
            self._health_thread = None
        if self._health_worker:
            self._health_worker.deleteLater()
            self._health_worker = None
