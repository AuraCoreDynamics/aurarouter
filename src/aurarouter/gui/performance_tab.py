"""Model performance monitor tab for AuraRouter GUI."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Spacing, Typography


class PerformanceTab(QWidget):
    """Monitor tab showing per-model success rate × complexity heatmap."""

    COMPLEXITY_BANDS = ["1-3", "4-6", "7-8", "9-10"]

    def __init__(self, api, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._palette = palette or DARK_PALETTE
        self._window_days = 7
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.md, Spacing.md, Spacing.md, Spacing.md)
        layout.setSpacing(Spacing.md)

        # Header
        header = QHBoxLayout()
        title = QLabel("Model Performance", self)
        title.setStyleSheet(
            f"color: {self._palette.text_primary}; "
            f"font-size: {Typography.size_h2}pt; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()

        header.addWidget(QLabel("Window:", self))
        self._window_combo = QComboBox(self)
        for d in [7, 14, 30, 90]:
            self._window_combo.addItem(f"{d} days", d)
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        header.addWidget(self._window_combo)
        layout.addLayout(header)

        subtitle = QLabel("Success Rate by Model × Complexity Band:", self)
        subtitle.setStyleSheet(
            f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;"
        )
        layout.addWidget(subtitle)

        # Heatmap table
        cols = ["Model"] + self.COMPLEXITY_BANDS + ["Avg", "Calls"]
        self._table = QTableWidget(0, len(cols), self)
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(cols)):
            self._table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setStyleSheet(
            f"QTableWidget {{ background: {self._palette.bg_secondary}; "
            f"color: {self._palette.text_primary}; font-size: {Typography.size_small}pt; border: none; }}"
            f"QHeaderView::section {{ background: {self._palette.bg_tertiary}; "
            f"color: {self._palette.text_secondary}; padding: 4px; border: none; }}"
        )
        self._table.currentCellChanged.connect(self._on_row_selected)
        layout.addWidget(self._table)

        # Legend
        legend = QLabel("Green ≥80%  Yellow ≥60%  Red <60%  — = no data", self)
        legend.setStyleSheet(
            f"color: {self._palette.text_disabled}; font-size: {Typography.size_small}pt;"
        )
        layout.addWidget(legend)

        # Detail panel
        self._detail = QLabel("", self)
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(
            f"background: {self._palette.bg_secondary}; "
            f"color: {self._palette.text_primary}; "
            f"padding: 8px; font-size: {Typography.size_small}pt;"
        )
        self._detail.hide()
        layout.addWidget(self._detail)

    def update_data(self, model_stats: list[dict]) -> None:
        """Refresh heatmap with per-model performance data."""
        self._table.setRowCount(0)
        for stat in model_stats:
            row = self._table.rowCount()
            self._table.insertRow(row)
            model_id = stat.get("model_id", "—")
            self._table.setItem(row, 0, QTableWidgetItem(model_id))

            success_rate = stat.get("success_rate", None)
            call_count = stat.get("call_count", 0)

            # Fill complexity bands (we only have aggregate data, so fill Avg)
            for i, band in enumerate(self.COMPLEXITY_BANDS):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(row, i + 1, item)

            # Avg column
            if success_rate is not None:
                pct = f"{success_rate:.0%}"
                avg_item = QTableWidgetItem(pct)
                avg_item.setTextAlignment(Qt.AlignCenter)
                color = self._cell_color(success_rate)
                avg_item.setBackground(QColor(color + "40"))
                self._table.setItem(row, 5, avg_item)
            else:
                self._table.setItem(row, 5, QTableWidgetItem("—"))

            calls_item = QTableWidgetItem(str(call_count))
            calls_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 6, calls_item)

    def _cell_color(self, rate: float) -> str:
        if rate >= 0.80:
            return self._palette.success
        elif rate >= 0.60:
            return self._palette.warning
        return self._palette.error

    def _on_window_changed(self, index: int) -> None:
        self._window_days = self._window_combo.itemData(index)
        # The parent MonitorPanel will call update_data on next refresh

    def _on_row_selected(self, row: int, col: int, prev_row: int, prev_col: int) -> None:
        if row < 0:
            self._detail.hide()
            return
        model_item = self._table.item(row, 0)
        if not model_item:
            return
        model_id = model_item.text()
        avg_item = self._table.item(row, 5)
        calls_item = self._table.item(row, 6)
        avg = avg_item.text() if avg_item else "—"
        calls = calls_item.text() if calls_item else "—"
        self._detail.setText(
            f"<b>{model_id}</b>: {calls} calls, {avg} success rate"
        )
        self._detail.setVisible(True)

    def get_window_days(self) -> int:
        return self._window_days
