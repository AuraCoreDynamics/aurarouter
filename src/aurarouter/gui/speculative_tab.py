"""Speculative decoding monitor tab for AuraRouter GUI."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QHeaderView, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Spacing, Typography
from aurarouter.gui.widgets.stat_card import StatCard
from aurarouter.gui.widgets.timeline import TimelineEntry, TimelineWidget


class SpeculativeTab(QWidget):
    """Monitor tab for speculative decoding activity."""

    def __init__(self, api, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._palette = palette or DARK_PALETTE
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.md, Spacing.md, Spacing.md, Spacing.md)
        layout.setSpacing(Spacing.md)

        # Title
        title = QLabel("Speculative Decoding", self)
        title.setStyleSheet(
            f"color: {self._palette.text_primary}; "
            f"font-size: {Typography.size_h2}pt; font-weight: bold;"
        )
        layout.addWidget(title)

        # Summary cards
        cards_row = QHBoxLayout()
        self._card_sessions = StatCard("Sessions", "—", palette=self._palette, parent=self)
        self._card_acceptance = StatCard("Acceptance Rate", "—", palette=self._palette, parent=self)
        self._card_notional = StatCard("Notional Emitted", "—", palette=self._palette, parent=self)
        self._card_corrections = StatCard("Corrections", "—", palette=self._palette, parent=self)
        for c in [self._card_sessions, self._card_acceptance, self._card_notional, self._card_corrections]:
            cards_row.addWidget(c)
        layout.addLayout(cards_row)

        # Config row
        self._config_label = QLabel("Config: —", self)
        self._config_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;"
        )
        layout.addWidget(self._config_label)

        # Active sessions table
        sessions_label = QLabel("Active Sessions:", self)
        sessions_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt; font-weight: bold;"
        )
        layout.addWidget(sessions_label)

        self._sessions_table = QTableWidget(0, 5, self)
        self._sessions_table.setHorizontalHeaderLabels(
            ["Session", "Drafter", "Verifier", "Acceptance", "Status"]
        )
        self._sessions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._sessions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._sessions_table.setMaximumHeight(180)
        self._sessions_table.setStyleSheet(
            f"QTableWidget {{ background: {self._palette.bg_secondary}; "
            f"color: {self._palette.text_primary}; font-size: {Typography.size_small}pt; border: none; }}"
            f"QHeaderView::section {{ background: {self._palette.bg_tertiary}; "
            f"color: {self._palette.text_secondary}; padding: 4px; border: none; }}"
        )
        layout.addWidget(self._sessions_table)

        # Notional timeline
        notional_label = QLabel("Notional Response Timeline:", self)
        notional_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt; font-weight: bold;"
        )
        layout.addWidget(notional_label)

        self._timeline = TimelineWidget(self._palette, self)
        layout.addWidget(self._timeline)

    def update_data(self, sessions: list[dict], spec_config: dict) -> None:
        """Refresh display with new API data."""
        # Config label
        enabled = spec_config.get("enabled", False)
        threshold = spec_config.get("complexity_threshold", 7)
        conf_threshold = spec_config.get("notional_confidence_threshold", 0.85)
        self._config_label.setText(
            f"{'Enabled ✓' if enabled else 'Disabled ✗'}  |  "
            f"Complexity threshold: {threshold}  |  "
            f"Confidence gate: {conf_threshold}"
        )

        # Summary cards
        self._card_sessions.set_value(str(len(sessions)))
        if sessions:
            avg_acc = sum(s.get("acceptance_rate", 0.0) for s in sessions) / len(sessions)
            self._card_acceptance.set_value(f"{avg_acc:.0%}")
        else:
            self._card_acceptance.set_value("—")
        self._card_notional.set_value("—")  # Would need notional emitter stats
        self._card_corrections.set_value("—")

        # Sessions table
        self._sessions_table.setRowCount(0)
        for s in sessions:
            row = self._sessions_table.rowCount()
            self._sessions_table.insertRow(row)
            self._sessions_table.setItem(row, 0, QTableWidgetItem(str(s.get("session_id", ""))[:12]))
            self._sessions_table.setItem(row, 1, QTableWidgetItem(s.get("drafter_model", "—")))
            self._sessions_table.setItem(row, 2, QTableWidgetItem(s.get("verifier_model", "—")))
            acc = s.get("acceptance_rate", 0.0)
            self._sessions_table.setItem(row, 3, QTableWidgetItem(f"{acc:.0%}"))
            self._sessions_table.setItem(row, 4, QTableWidgetItem(s.get("status", "—")))

        # Timeline entries (simulated from session data)
        entries = []
        for s in sessions[:10]:
            acc = s.get("acceptance_rate", 0.0)
            status_val = "success" if acc >= 0.8 else "failed"
            entry = TimelineEntry(
                timestamp="",
                title=f"Session {str(s.get('session_id', ''))[:8]}: {acc:.0%} acceptance",
                status=status_val,
                detail=f"Drafter: {s.get('drafter_model', '?')} → Verifier: {s.get('verifier_model', '?')}",
            )
            entries.append(entry)
        self._timeline.set_entries(entries)
