"""Chat bubble and routing insight pill widgets."""
from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Radius, Typography
from aurarouter.gui.widgets.status_badge import StatusBadge


class RoutingInsightPill(QWidget):
    """Compact clickable pill showing routing decision summary."""

    clicked = Signal()

    def __init__(self, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._context: dict = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"background: {self._palette.bg_tertiary}; "
            f"border-radius: {Radius.lg}px;"
        )

        self._pill_label = QLabel("", self)
        self._pill_label.setStyleSheet(
            f"font-size: {Typography.size_small}pt; "
            f"color: {self._palette.text_secondary};"
        )
        layout.addWidget(self._pill_label)
        self.hide()

        # Popover reference
        self._popover: Optional[QFrame] = None

    def set_context(self, routing_context: dict) -> None:
        self._context = routing_context
        summary = self._build_summary(routing_context)
        self._pill_label.setText(summary)
        self.setVisible(bool(summary))

    def _build_summary(self, ctx: dict) -> str:
        if not ctx:
            return ""
        intent = ctx.get("intent", "")
        conf = ctx.get("confidence", 0.0)
        hard_routed = ctx.get("hard_routed", False)
        speculative = ctx.get("speculative", False)
        monologue = ctx.get("monologue", False)
        cost = ctx.get("simulated_cost_avoided", 0.0)

        tier = "Local" if hard_routed else ctx.get("provider", "Cloud")
        savings_str = f" · ${cost:.2f} saved" if cost > 0 else ""

        if speculative:
            return f"{tier} · {intent} · {conf:.2f} conf · ⚡ Speculative"
        if monologue:
            iters = ctx.get("iterations", 0)
            return f"{tier} · {intent} · {iters} iters · 🔄 Monologue"
        return f"{tier} · {intent} · {conf:.2f} conf{savings_str}"

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        self._show_popover()

    def _show_popover(self) -> None:
        if self._popover and self._popover.isVisible():
            self._popover.hide()
            return
        if not self._context:
            return

        from aurarouter.gui.widgets.confidence_bar import ConfidenceBar
        from aurarouter.gui.widgets.complexity_badge import ComplexityBadge

        pop = QFrame(self.window(), Qt.Popup)
        pop.setFrameShape(QFrame.StyledPanel)
        pop.setStyleSheet(
            f"QFrame {{ background: {self._palette.bg_secondary}; "
            f"border: 1px solid {self._palette.border}; "
            f"border-radius: {Radius.md}px; }}"
        )
        layout = QVBoxLayout(pop)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        def _row(label: str, value: str) -> None:
            row = QHBoxLayout()
            lbl = QLabel(f"<b>{label}</b>", pop)
            lbl.setStyleSheet(f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;")
            row.addWidget(lbl)
            val = QLabel(value, pop)
            val.setStyleSheet(f"color: {self._palette.text_primary}; font-size: {Typography.size_small}pt;")
            row.addWidget(val)
            layout.addLayout(row)

        ctx = self._context
        _row("Intent:", ctx.get("intent", "—"))

        # Complexity badge
        if "complexity" in ctx:
            row = QHBoxLayout()
            lbl = QLabel("<b>Complexity:</b>", pop)
            lbl.setStyleSheet(f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;")
            row.addWidget(lbl)
            badge = ComplexityBadge(self._palette, pop)
            badge.set_score(ctx["complexity"])
            row.addWidget(badge)
            row.addStretch()
            layout.addLayout(row)

        _row("Strategy:", ctx.get("strategy", "—"))

        # Confidence bar
        if "confidence" in ctx:
            row = QHBoxLayout()
            lbl = QLabel("<b>Confidence:</b>", pop)
            lbl.setStyleSheet(f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;")
            row.addWidget(lbl)
            cbar = ConfidenceBar(self._palette, pop)
            cbar.set_score(ctx["confidence"])
            row.addWidget(cbar)
            layout.addLayout(row)

        hard = ctx.get("hard_routed", False)
        cost = ctx.get("simulated_cost_avoided", 0.0)
        _row("Hard-routed:", f"{'Yes' if hard else 'No'}{f' — ${cost:.2f} saved' if hard and cost > 0 else ''}")
        _row("Analyzers:", ctx.get("analyzer_id", "—"))
        _row("Mode:", ctx.get("execution_mode", "Standard"))

        pop.adjustSize()
        global_pos = self.mapToGlobal(self.rect().bottomLeft())
        pop.move(global_pos)
        pop.show()
        self._popover = pop

        # Install event filter to close on outside click
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.MouseButtonPress and self._popover and self._popover.isVisible():
            if not self._popover.geometry().contains(
                self._popover.mapFromGlobal(event.globalPos()) + self._popover.pos()
            ):
                self._popover.hide()
                QApplication.instance().removeEventFilter(self)
        return False


class ChatBubble(QFrame):
    """Single conversation message bubble."""

    routing_pill_clicked = Signal(dict)

    def __init__(
        self,
        role: str,
        content: str,
        model_id: str = "",
        timestamp: str = "",
        is_verified: bool = False,
        routing_context: dict | None = None,
        palette: ColorPalette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette or DARK_PALETTE
        self._role = role
        self.setFrameShape(QFrame.StyledPanel)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # Header: model badge + verified icon
        if role == "assistant":
            header = QHBoxLayout()
            if model_id:
                badge = StatusBadge(model_id, palette=self._palette, parent=self)
                header.addWidget(badge)
            if is_verified:
                shield = QLabel(self)
                try:
                    import os
                    root = os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )))
                    pix = QPixmap(os.path.join(root, "auracore_green.png"))
                    if not pix.isNull():
                        shield.setPixmap(pix.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        shield.setText("✓")
                except Exception:
                    shield.setText("✓")
                shield.setToolTip("Speculative draft — verified by a stronger model before delivery.")
                header.addWidget(shield)
            header.addStretch()
            if timestamp:
                ts_lbl = QLabel(timestamp, self)
                ts_lbl.setStyleSheet(
                    f"font-size: {Typography.size_small}pt; color: {self._palette.text_disabled};"
                )
                header.addWidget(ts_lbl)
            outer.addLayout(header)

            # Routing insight pill
            self._pill = RoutingInsightPill(self._palette, self)
            if routing_context:
                self._pill.set_context(routing_context)
                self._pill.clicked.connect(lambda: self.routing_pill_clicked.emit(routing_context))
            outer.addWidget(self._pill)

        elif role == "user":
            if timestamp:
                header = QHBoxLayout()
                header.addStretch()
                ts_lbl = QLabel(timestamp, self)
                ts_lbl.setStyleSheet(
                    f"font-size: {Typography.size_small}pt; color: {self._palette.text_disabled};"
                )
                header.addWidget(ts_lbl)
                outer.addLayout(header)
            self._pill = None
        else:
            self._pill = None

        # Content
        self._content = QTextBrowser(self)
        self._content.setReadOnly(True)
        self._content.setOpenExternalLinks(False)
        self._content.setPlainText(content)
        self._content.setMaximumWidth(int(self.parent().width() * 0.70) if self.parent() else 600)
        self._content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._content.setStyleSheet(
            f"background: transparent; border: none; "
            f"color: {self._palette.text_primary}; "
            f"font-size: {Typography.size_body}pt;"
        )
        outer.addWidget(self._content)

        # Alignment & styling
        if role == "user":
            self.setStyleSheet(
                f"background: {self._palette.accent}22; "
                f"border-radius: {Radius.md}px; "
                f"border: 1px solid {self._palette.accent}44;"
            )
        elif role == "assistant":
            self.setStyleSheet(
                f"background: {self._palette.bg_secondary}; "
                f"border-radius: {Radius.md}px; "
                f"border: 1px solid {self._palette.border};"
            )
        else:  # system
            self.setStyleSheet(
                f"background: transparent; "
                f"border-radius: {Radius.sm}px; "
                f"border: 1px solid {self._palette.separator};"
            )

    def set_routing_context(self, ctx: dict) -> None:
        if self._pill:
            self._pill.set_context(ctx)

    def append_text(self, text: str) -> None:
        current = self._content.toPlainText()
        self._content.setPlainText(current + text)
