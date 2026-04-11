"""Monologue reasoning monitor tab for AuraRouter GUI."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QVBoxLayout, QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Spacing, Typography
from aurarouter.gui.widgets.confidence_bar import ConfidenceBar
from aurarouter.gui.widgets.stat_card import StatCard
from aurarouter.gui.widgets.timeline import TimelineEntry, TimelineWidget


class MonologueTab(QWidget):
    """Monitor tab for monologue reasoning sessions."""

    def __init__(self, api, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._palette = palette or DARK_PALETTE
        self._sessions: list[dict] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.md, Spacing.md, Spacing.md, Spacing.md)
        layout.setSpacing(Spacing.md)

        title = QLabel("Monologue Reasoning", self)
        title.setStyleSheet(
            f"color: {self._palette.text_primary}; "
            f"font-size: {Typography.size_h2}pt; font-weight: bold;"
        )
        layout.addWidget(title)

        # Summary cards
        cards_row = QHBoxLayout()
        self._card_sessions = StatCard("Sessions", "—", palette=self._palette, parent=self)
        self._card_avg_iters = StatCard("Avg Iterations", "—", palette=self._palette, parent=self)
        self._card_converged = StatCard("Converged", "—", palette=self._palette, parent=self)
        for c in [self._card_sessions, self._card_avg_iters, self._card_converged]:
            cards_row.addWidget(c)
        layout.addLayout(cards_row)

        # Config
        self._config_label = QLabel("Config: —", self)
        self._config_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;"
        )
        layout.addWidget(self._config_label)

        # Splitter: session list + trace
        splitter = QSplitter(Qt.Horizontal, self)

        # Sessions list
        sessions_widget = QWidget()
        sl_layout = QVBoxLayout(sessions_widget)
        sl_layout.setContentsMargins(0, 0, 0, 0)
        sl_label = QLabel("Sessions:", sessions_widget)
        sl_label.setStyleSheet(f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;")
        sl_layout.addWidget(sl_label)
        self._session_list = QListWidget(sessions_widget)
        self._session_list.setMaximumWidth(200)
        self._session_list.setStyleSheet(
            f"QListWidget {{ background: {self._palette.bg_secondary}; "
            f"color: {self._palette.text_primary}; "
            f"font-size: {Typography.size_small}pt; border: none; }}"
        )
        self._session_list.currentItemChanged.connect(self._on_session_selected)
        sl_layout.addWidget(self._session_list)
        splitter.addWidget(sessions_widget)

        # Trace view
        trace_widget = QWidget()
        trace_layout = QVBoxLayout(trace_widget)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_label = QLabel("Reasoning Trace:", trace_widget)
        trace_label.setStyleSheet(f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;")
        trace_layout.addWidget(trace_label)
        self._trace_timeline = TimelineWidget(self._palette, trace_widget)
        trace_layout.addWidget(self._trace_timeline)
        splitter.addWidget(trace_widget)

        splitter.setSizes([200, 600])
        layout.addWidget(splitter)

        # Footer: convergence info
        self._footer_label = QLabel("", self)
        self._footer_label.setStyleSheet(
            f"color: {self._palette.text_disabled}; font-size: {Typography.size_small}pt;"
        )
        layout.addWidget(self._footer_label)

    def update_data(self, sessions: list[dict], mono_config: dict) -> None:
        """Refresh display with new API data."""
        self._sessions = sessions

        enabled = mono_config.get("enabled", False)
        max_iters = mono_config.get("max_iterations_default", 5)
        threshold = mono_config.get("convergence_threshold_default", 0.85)
        self._config_label.setText(
            f"{'Enabled ✓' if enabled else 'Disabled ✗'}  |  "
            f"Max iterations: {max_iters}  |  Convergence threshold: {threshold}"
        )

        self._card_sessions.set_value(str(len(sessions)))
        if sessions:
            avg_iters = sum(s.get("iteration_count", 0) for s in sessions) / len(sessions)
            self._card_avg_iters.set_value(f"{avg_iters:.1f}")
            converged = sum(1 for s in sessions if s.get("convergence_reason"))
            self._card_converged.set_value(str(converged))
        else:
            self._card_avg_iters.set_value("—")
            self._card_converged.set_value("—")

        self._session_list.clear()
        for s in sessions:
            sid = str(s.get("session_id", ""))
            iters = s.get("iteration_count", 0)
            item = QListWidgetItem(f"● {sid[:8]}… ({iters} iter)")
            item.setData(Qt.UserRole, sid)
            self._session_list.addItem(item)

    def _on_session_selected(self, current: QListWidgetItem, previous) -> None:
        if current is None:
            return
        session_id = current.data(Qt.UserRole)
        if not session_id:
            return
        try:
            trace = self._api.get_monologue_trace(session_id)
        except Exception:
            trace = None

        self._trace_timeline.clear()
        if not trace or "error" in trace:
            return

        entries = []
        for step in trace.get("steps", []):
            role = step.get("role", "")
            model = step.get("model_id", "")
            preview = step.get("output_preview", "")
            mas = step.get("mas_relevancy_score", 0.0)
            conf = step.get("confidence", 0.0)
            iteration = step.get("iteration", 0)

            # Expert-role color mapping (dual-coded with timeline icon)
            role_color = ""
            if role == "generator":
                role_color = self._palette.monologue_generator
            elif role == "critic":
                role_color = self._palette.monologue_critic
            elif role == "refiner":
                role_color = self._palette.monologue_refiner

            entry = TimelineEntry(
                timestamp=f"iter {iteration}",
                title=f"{role.capitalize()} [{model}]  MAS: {mas:.2f}  conf: {conf:.2f}",
                status="success" if conf >= 0.8 else "running",
                detail=preview,
                role_color=role_color,
            )
            entries.append(entry)

        self._trace_timeline.set_entries(entries)
        conv = trace.get("convergence_reason", "")
        iters = trace.get("iteration_count", 0)
        self._footer_label.setText(f"Convergence: {conv or 'n/a'}  |  Iterations: {iters}")
