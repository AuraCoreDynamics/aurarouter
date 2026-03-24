"""Interactive visual routing editor panel.

Provides a flowchart-style canvas for editing role -> model fallback
chains, a properties sidebar for node details, and a triage preview
section.  All changes require explicit Save to persist.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from aurarouter.api import AuraRouterAPI
    from aurarouter.gui.help.content import HelpRegistry


# ======================================================================
# Layout constants
# ======================================================================

_ROLE_W = 160
_ROLE_H = 64
_MODEL_W = 150
_MODEL_H = 56
_H_GAP = 50
_V_GAP = 24
_MARGIN = 20
_TIER_STRIPE_W = 6
_ADD_BTN_SIZE = 28
_ARROW_SIZE = 8


# ======================================================================
# Node types used by the canvas
# ======================================================================

class _RoleNode:
    """In-memory representation of a role on the canvas."""

    __slots__ = ("name", "description", "required", "synonyms", "chain", "rect")

    def __init__(
        self,
        name: str,
        description: str = "",
        required: bool = False,
        synonyms: list[str] | None = None,
        chain: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.required = required
        self.synonyms: list[str] = synonyms or []
        self.chain: list[str] = chain or []
        self.rect = QRectF()


class _ModelNode:
    """In-memory representation of a model in a chain on the canvas."""

    __slots__ = ("model_id", "provider", "tier", "role_name", "index", "rect")

    def __init__(
        self,
        model_id: str,
        provider: str = "",
        tier: str = "",
        role_name: str = "",
        index: int = 0,
    ):
        self.model_id = model_id
        self.provider = provider
        self.tier = tier
        self.role_name = role_name
        self.index = index
        self.rect = QRectF()


# ======================================================================
# Flowchart canvas
# ======================================================================

class _RoutingCanvas(QWidget):
    """Custom QPainter widget that renders the role/model flowchart."""

    role_clicked = Signal(str)          # role name
    model_clicked = Signal(str, str)    # role name, model_id
    add_model_clicked = Signal(str)     # role name (the [+] at end of chain)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._role_nodes: list[_RoleNode] = []
        self._model_nodes: list[_ModelNode] = []
        self._add_btn_rects: dict[str, QRectF] = {}  # role -> rect of [+]

        self._selected_role: Optional[str] = None
        self._selected_model: Optional[tuple[str, str]] = None  # (role, model_id)
        self._hover_item: Optional[str] = None  # "role:NAME" or "model:ROLE:ID"

        self.setMouseTracking(True)
        self.setMinimumSize(400, 200)

        # Import palette lazily so the module can be imported without QApp.
        from aurarouter.gui.theme import get_palette
        self._palette = get_palette("dark")

    def set_data(
        self,
        role_nodes: list[_RoleNode],
        model_nodes: list[_ModelNode],
    ) -> None:
        self._role_nodes = role_nodes
        self._model_nodes = model_nodes
        self._layout_nodes()
        self.update()

    def set_selection(
        self,
        role: Optional[str] = None,
        model: Optional[tuple[str, str]] = None,
    ) -> None:
        self._selected_role = role
        self._selected_model = model
        self.update()

    # ---- layout -------------------------------------------------------

    def _layout_nodes(self) -> None:
        """Position roles in a left column, models flowing right."""
        self._add_btn_rects.clear()
        y = _MARGIN
        for rn in self._role_nodes:
            rn.rect = QRectF(_MARGIN, y, _ROLE_W, _ROLE_H)
            # Position model nodes
            mx = _MARGIN + _ROLE_W + _H_GAP
            for mn in self._model_nodes:
                if mn.role_name == rn.name:
                    mn.rect = QRectF(mx, y + (_ROLE_H - _MODEL_H) / 2, _MODEL_W, _MODEL_H)
                    mx += _MODEL_W + _H_GAP * 0.6
            # [+] button at end of chain
            self._add_btn_rects[rn.name] = QRectF(
                mx, y + (_ROLE_H - _ADD_BTN_SIZE) / 2, _ADD_BTN_SIZE, _ADD_BTN_SIZE
            )
            y += _ROLE_H + _V_GAP

        # Compute minimum size
        max_x = _MARGIN + _ROLE_W + _H_GAP
        for mn in self._model_nodes:
            if mn.rect.right() + _H_GAP + _ADD_BTN_SIZE > max_x:
                max_x = mn.rect.right() + _H_GAP + _ADD_BTN_SIZE
        for r in self._add_btn_rects.values():
            if r.right() + _MARGIN > max_x:
                max_x = r.right() + _MARGIN
        max_y = y + _MARGIN if self._role_nodes else _MARGIN * 2
        self.setMinimumSize(int(max_x), int(max_y))

    # ---- painting -----------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = self._palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw edges first (behind nodes).
        self._draw_edges(painter)

        # Draw role nodes.
        for rn in self._role_nodes:
            self._draw_role(painter, rn)

        # Draw model nodes.
        for mn in self._model_nodes:
            self._draw_model(painter, mn)

        # Draw [+] buttons at end of each chain.
        for role_name, rect in self._add_btn_rects.items():
            self._draw_add_button(painter, rect)

        painter.end()

    def _draw_role(self, painter: QPainter, rn: _RoleNode) -> None:
        p = self._palette
        is_selected = (
            self._selected_role == rn.name and self._selected_model is None
        )
        is_hover = self._hover_item == f"role:{rn.name}"

        # Background
        bg = QColor(p.bg_selected if is_selected else p.bg_tertiary)
        path = QPainterPath()
        path.addRoundedRect(rn.rect, 8, 8)
        painter.fillPath(path, bg)

        # Border — accent for required roles, normal otherwise
        border_color = QColor(p.accent) if rn.required else QColor(p.border)
        pen_w = 2.5 if is_selected or is_hover else (2.0 if rn.required else 1.5)
        painter.setPen(QPen(border_color, pen_w))
        painter.drawRoundedRect(rn.rect, 8, 8)

        # Role name
        painter.setPen(QPen(QColor(p.text_primary)))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        name_rect = QRectF(
            rn.rect.x() + 8, rn.rect.y() + 6,
            rn.rect.width() - 16, 20,
        )
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, rn.name)

        # Badge
        badge_text = "required" if rn.required else "optional"
        badge_color = QColor(p.accent) if rn.required else QColor(p.text_disabled)
        font.setPointSize(7)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(badge_color))
        badge_rect = QRectF(
            rn.rect.right() - 60, rn.rect.y() + 6, 52, 14,
        )
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, badge_text)

        # Description
        painter.setPen(QPen(QColor(p.text_secondary)))
        font.setPointSize(8)
        painter.setFont(font)
        desc_rect = QRectF(
            rn.rect.x() + 8, rn.rect.y() + 28,
            rn.rect.width() - 16, 16,
        )
        desc = rn.description
        if len(desc) > 28:
            desc = desc[:26] + ".."
        painter.drawText(desc_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, desc)

        # Chain count
        painter.setPen(QPen(QColor(p.text_disabled)))
        font.setPointSize(7)
        painter.setFont(font)
        count_rect = QRectF(
            rn.rect.x() + 8, rn.rect.y() + 46,
            rn.rect.width() - 16, 14,
        )
        painter.drawText(
            count_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{len(rn.chain)} model(s)",
        )

    def _draw_model(self, painter: QPainter, mn: _ModelNode) -> None:
        p = self._palette
        is_selected = (
            self._selected_model is not None
            and self._selected_model[0] == mn.role_name
            and self._selected_model[1] == mn.model_id
        )
        is_hover = self._hover_item == f"model:{mn.role_name}:{mn.model_id}:{mn.index}"

        # Background
        bg = QColor(p.bg_selected if is_selected else p.bg_secondary)
        path = QPainterPath()
        path.addRoundedRect(mn.rect, 6, 6)
        painter.fillPath(path, bg)

        # Border
        border_color = QColor(p.accent) if is_selected else QColor(p.border)
        pen_w = 2.0 if is_selected or is_hover else 1.0
        painter.setPen(QPen(border_color, pen_w))
        painter.drawRoundedRect(mn.rect, 6, 6)

        # Tier colour stripe on the left edge
        tier_color = self._tier_color(mn.tier)
        stripe_rect = QRectF(mn.rect.x(), mn.rect.y(), _TIER_STRIPE_W, mn.rect.height())
        stripe_path = QPainterPath()
        stripe_path.addRoundedRect(stripe_rect, 3, 3)
        painter.fillPath(stripe_path, tier_color)

        # Model ID
        painter.setPen(QPen(QColor(p.text_primary)))
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        id_rect = QRectF(
            mn.rect.x() + _TIER_STRIPE_W + 6, mn.rect.y() + 6,
            mn.rect.width() - _TIER_STRIPE_W - 12, 18,
        )
        mid_text = mn.model_id
        if len(mid_text) > 18:
            mid_text = mid_text[:16] + ".."
        painter.drawText(id_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, mid_text)

        # Provider + tier
        painter.setPen(QPen(QColor(p.text_secondary)))
        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        sub_rect = QRectF(
            mn.rect.x() + _TIER_STRIPE_W + 6, mn.rect.y() + 26,
            mn.rect.width() - _TIER_STRIPE_W - 12, 14,
        )
        sub_text = mn.provider
        if mn.tier:
            sub_text += f" ({mn.tier})"
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sub_text)

        # Priority number
        painter.setPen(QPen(QColor(p.text_disabled)))
        font.setPointSize(7)
        painter.setFont(font)
        pri_rect = QRectF(
            mn.rect.x() + _TIER_STRIPE_W + 6, mn.rect.y() + 40,
            mn.rect.width() - _TIER_STRIPE_W - 12, 12,
        )
        painter.drawText(
            pri_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"priority #{mn.index + 1}",
        )

    def _draw_add_button(self, painter: QPainter, rect: QRectF) -> None:
        p = self._palette
        path = QPainterPath()
        path.addRoundedRect(rect, 4, 4)
        painter.fillPath(path, QColor(p.bg_tertiary))
        painter.setPen(QPen(QColor(p.border), 1.0))
        painter.drawRoundedRect(rect, 4, 4)
        # Plus sign
        painter.setPen(QPen(QColor(p.text_secondary), 2.0))
        cx, cy = rect.center().x(), rect.center().y()
        half = 6
        painter.drawLine(QPointF(cx - half, cy), QPointF(cx + half, cy))
        painter.drawLine(QPointF(cx, cy - half), QPointF(cx, cy + half))

    def _draw_edges(self, painter: QPainter) -> None:
        """Draw bezier arrows: role -> first model, then model -> model."""
        p = self._palette
        pen = QPen(QColor(p.text_disabled), 1.5)
        painter.setPen(pen)

        for rn in self._role_nodes:
            chain_models = [
                mn for mn in self._model_nodes if mn.role_name == rn.name
            ]
            chain_models.sort(key=lambda m: m.index)
            if not chain_models:
                continue

            # Role -> first model
            self._draw_bezier_arrow(
                painter,
                rn.rect.right(), rn.rect.center().y(),
                chain_models[0].rect.left(), chain_models[0].rect.center().y(),
            )

            # Model -> next model
            for i in range(len(chain_models) - 1):
                self._draw_bezier_arrow(
                    painter,
                    chain_models[i].rect.right(), chain_models[i].rect.center().y(),
                    chain_models[i + 1].rect.left(), chain_models[i + 1].rect.center().y(),
                )

    def _draw_bezier_arrow(
        self, painter: QPainter, x1: float, y1: float, x2: float, y2: float,
    ) -> None:
        """Draw a smooth bezier curve with an arrowhead."""
        p = self._palette
        path = QPainterPath()
        path.moveTo(x1, y1)
        ctrl_offset = abs(x2 - x1) * 0.4
        path.cubicTo(x1 + ctrl_offset, y1, x2 - ctrl_offset, y2, x2, y2)
        painter.drawPath(path)
        self._draw_arrowhead(painter, x2 - ctrl_offset, y2, x2, y2)

    @staticmethod
    def _draw_arrowhead(
        painter: QPainter, x1: float, y1: float, x2: float, y2: float,
    ) -> None:
        angle = math.atan2(y2 - y1, x2 - x1)
        p1x = x2 - _ARROW_SIZE * math.cos(angle - 0.4)
        p1y = y2 - _ARROW_SIZE * math.sin(angle - 0.4)
        p2x = x2 - _ARROW_SIZE * math.cos(angle + 0.4)
        p2y = y2 - _ARROW_SIZE * math.sin(angle + 0.4)

        path = QPainterPath()
        path.moveTo(x2, y2)
        path.lineTo(p1x, p1y)
        path.lineTo(p2x, p2y)
        path.closeSubpath()
        painter.fillPath(path, painter.pen().color())

    def _tier_color(self, tier: str) -> QColor:
        p = self._palette
        tier_lower = tier.lower() if tier else ""
        if tier_lower in ("on-prem", "local"):
            return QColor(p.tier_local)
        elif tier_lower == "cloud":
            return QColor(p.tier_cloud)
        elif tier_lower in ("dedicated-tenant", "grid"):
            return QColor(p.tier_grid)
        return QColor(p.text_disabled)

    # ---- mouse interaction --------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        # Check [+] buttons first
        for role_name, rect in self._add_btn_rects.items():
            if rect.contains(pos):
                self.add_model_clicked.emit(role_name)
                return
        # Check model nodes
        for mn in self._model_nodes:
            if mn.rect.contains(pos):
                self.model_clicked.emit(mn.role_name, mn.model_id)
                return
        # Check role nodes
        for rn in self._role_nodes:
            if rn.rect.contains(pos):
                self.role_clicked.emit(rn.name)
                return
        # Click on empty space clears selection
        self.role_clicked.emit("")

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        old_hover = self._hover_item
        self._hover_item = None
        for role_name, rect in self._add_btn_rects.items():
            if rect.contains(pos):
                self._hover_item = f"add:{role_name}"
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                break
        if self._hover_item is None:
            for mn in self._model_nodes:
                if mn.rect.contains(pos):
                    self._hover_item = f"model:{mn.role_name}:{mn.model_id}:{mn.index}"
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    break
        if self._hover_item is None:
            for rn in self._role_nodes:
                if rn.rect.contains(pos):
                    self._hover_item = f"role:{rn.name}"
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    break
        if self._hover_item is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if old_hover != self._hover_item:
            self.update()

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """Right-click context menu for model and role nodes."""
        pos = event.pos()
        # Check model nodes
        for mn in self._model_nodes:
            if mn.rect.contains(QPointF(pos)):
                menu = QMenu(self)
                remove_action = menu.addAction(f"Remove '{mn.model_id}' from chain")
                action = menu.exec(event.globalPos())
                if action == remove_action:
                    # Emit model click first so the panel knows which node
                    self.model_clicked.emit(mn.role_name, mn.model_id)
                    # Fire a custom signal — we'll handle removal in the panel
                    self._request_remove_model(mn.role_name, mn.model_id, mn.index)
                return
        # Check role nodes
        for rn in self._role_nodes:
            if rn.rect.contains(QPointF(pos)):
                if not rn.required:
                    menu = QMenu(self)
                    delete_action = menu.addAction(f"Delete role '{rn.name}'")
                    action = menu.exec(event.globalPos())
                    if action == delete_action:
                        self._request_delete_role(rn.name)
                return

    # These will be connected by RoutingPanel
    _remove_model_callback: Optional[object] = None
    _delete_role_callback: Optional[object] = None

    def _request_remove_model(self, role: str, model_id: str, index: int) -> None:
        if callable(self._remove_model_callback):
            self._remove_model_callback(role, model_id, index)

    def _request_delete_role(self, role: str) -> None:
        if callable(self._delete_role_callback):
            self._delete_role_callback(role)


# ======================================================================
# Properties sidebar
# ======================================================================

class _PropertiesSidebar(QWidget):
    """Right-side panel showing details of the selected node."""

    test_connection_requested = Signal(str)  # model_id
    edit_model_requested = Signal(str)       # model_id
    move_up_requested = Signal(str, str, int)     # role, model_id, index
    move_down_requested = Signal(str, str, int)   # role, model_id, index

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedWidth(250)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._content_widget: Optional[QWidget] = None
        self.show_overview(0, 0, [])

    def _clear(self) -> None:
        if self._content_widget is not None:
            self._content_widget.setParent(None)  # type: ignore[arg-type]
            self._content_widget.deleteLater()
            self._content_widget = None

    def show_overview(
        self, n_roles: int, n_models: int, missing: list[str],
    ) -> None:
        self._clear()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(self._heading("Overview"))
        lay.addWidget(self._info_label(f"Roles configured: {n_roles}"))
        lay.addWidget(self._info_label(f"Total models: {n_models}"))
        if missing:
            lbl = QLabel(f"Missing required: {', '.join(missing)}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #f38ba8; font-weight: bold; font-size: 10px;")
            lay.addWidget(lbl)
        else:
            lay.addWidget(self._info_label("All required roles configured"))
        lay.addStretch()

        self._content_widget = w
        self._layout.addWidget(w)

    def show_role(self, rn: _RoleNode) -> None:
        self._clear()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(self._heading(f"Role: {rn.name}"))
        if rn.required:
            badge = QLabel("REQUIRED")
            badge.setStyleSheet(
                "color: #1e1e2e; background-color: #89b4fa; "
                "padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
            )
            badge.setFixedHeight(18)
            lay.addWidget(badge)
        else:
            badge = QLabel("OPTIONAL")
            badge.setStyleSheet(
                "color: #cdd6f4; background-color: #45475a; "
                "padding: 2px 6px; border-radius: 3px; font-size: 9px;"
            )
            badge.setFixedHeight(18)
            lay.addWidget(badge)

        if rn.description:
            desc = QLabel(rn.description)
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #a6adc8; font-size: 10px; margin-top: 4px;")
            lay.addWidget(desc)

        if rn.synonyms:
            syn = QLabel(f"Synonyms: {', '.join(rn.synonyms)}")
            syn.setWordWrap(True)
            syn.setStyleSheet("color: #6c7086; font-size: 9px; margin-top: 2px;")
            lay.addWidget(syn)

        lay.addWidget(self._sub_heading("Chain"))
        if rn.chain:
            for i, mid in enumerate(rn.chain):
                lbl = QLabel(f"  {i + 1}. {mid}")
                lbl.setStyleSheet("font-size: 10px;")
                lay.addWidget(lbl)
        else:
            lay.addWidget(self._info_label("(empty chain)"))

        lay.addStretch()
        self._content_widget = w
        self._layout.addWidget(w)

    def show_model(
        self,
        mn: _ModelNode,
        model_config: dict,
        chain_length: int,
    ) -> None:
        self._clear()
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(self._heading(f"Model: {mn.model_id}"))

        # Provider
        lay.addWidget(self._info_label(f"Provider: {mn.provider}"))

        # Tier badge
        if mn.tier:
            tier_lbl = QLabel(mn.tier.upper())
            tier_colors = {
                "on-prem": "#a6e3a1", "local": "#a6e3a1",
                "cloud": "#cba6f7",
                "dedicated-tenant": "#89dceb", "grid": "#89dceb",
            }
            tc = tier_colors.get(mn.tier.lower(), "#6c7086")
            tier_lbl.setStyleSheet(
                f"color: #1e1e2e; background-color: {tc}; "
                "padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
            )
            tier_lbl.setFixedHeight(18)
            lay.addWidget(tier_lbl)

        # Endpoint
        endpoint = model_config.get(
            "endpoint", model_config.get("model_name", model_config.get("model_path", ""))
        )
        if endpoint:
            ep_lbl = QLabel(f"Endpoint: {endpoint}")
            ep_lbl.setWordWrap(True)
            ep_lbl.setStyleSheet("color: #a6adc8; font-size: 9px;")
            lay.addWidget(ep_lbl)

        # In role / priority
        lay.addWidget(self._info_label(f"In role: {mn.role_name}"))
        lay.addWidget(self._info_label(f"Priority: #{mn.index + 1} of {chain_length}"))

        # Reorder buttons
        btn_row = QHBoxLayout()
        up_btn = QPushButton("Move Up")
        up_btn.setEnabled(mn.index > 0)
        up_btn.clicked.connect(
            lambda: self.move_up_requested.emit(mn.role_name, mn.model_id, mn.index)
        )
        btn_row.addWidget(up_btn)

        down_btn = QPushButton("Move Down")
        down_btn.setEnabled(mn.index < chain_length - 1)
        down_btn.clicked.connect(
            lambda: self.move_down_requested.emit(mn.role_name, mn.model_id, mn.index)
        )
        btn_row.addWidget(down_btn)
        lay.addLayout(btn_row)

        # Test connection
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(lambda: self.test_connection_requested.emit(mn.model_id))
        lay.addWidget(test_btn)

        # Edit model
        edit_btn = QPushButton("Edit Model...")
        edit_btn.clicked.connect(lambda: self.edit_model_requested.emit(mn.model_id))
        lay.addWidget(edit_btn)

        lay.addStretch()
        self._content_widget = w
        self._layout.addWidget(w)

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _heading(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 13px; font-weight: bold; margin-bottom: 4px;")
        return lbl

    @staticmethod
    def _sub_heading(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 11px; font-weight: bold; margin-top: 8px;")
        return lbl

    @staticmethod
    def _info_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size: 10px; color: #a6adc8;")
        return lbl


# ======================================================================
# Triage preview
# ======================================================================

class _TriagePreview(QWidget):
    """Collapsible section showing complexity -> role mappings."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._expanded = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QHBoxLayout()
        self._toggle_btn = QPushButton("\u25b6 Triage Preview")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setStyleSheet("text-align: left; font-weight: bold; font-size: 10px;")
        self._toggle_btn.clicked.connect(self._toggle)
        header.addWidget(self._toggle_btn)
        header.addStretch()
        layout.addLayout(header)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 8, 4)
        self._body.setVisible(False)
        layout.addWidget(self._body)

    def set_rules(self, rules: list[dict]) -> None:
        """Populate triage rules or show an informational note."""
        # Clear
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not rules:
            note = QLabel(
                "Triage is not enabled. Enable the savings subsystem in "
                "Settings to activate complexity-based routing."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: #6c7086; font-size: 10px; font-style: italic;")
            self._body_layout.addWidget(note)
            return

        for rule in rules:
            text = (
                f"Complexity <= {rule.get('max_complexity', '?')}  "
                f"-> {rule.get('preferred_role', '?')}  "
                f"({rule.get('description', '')})"
            )
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 10px; color: #a6adc8;")
            self._body_layout.addWidget(lbl)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        arrow = "\u25bc" if self._expanded else "\u25b6"
        self._toggle_btn.setText(f"{arrow} Triage Preview")


# ======================================================================
# Main panel
# ======================================================================

class RoutingPanel(QWidget):
    """Interactive visual editor for role -> model fallback chains."""

    config_changed = Signal()  # emitted after save

    def __init__(
        self,
        api: AuraRouterAPI,
        help_registry: Optional[HelpRegistry] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._api = api
        self._help_registry = help_registry

        # Working copies — modifications go here until Save
        self._working_roles: dict[str, list[str]] = {}  # role -> chain
        self._dirty = False

        self._selected_role: Optional[str] = None
        self._selected_model: Optional[tuple[str, str]] = None
        self._selected_model_index: Optional[int] = None

        self._build_ui()
        self._load_from_api()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---- Toolbar --------------------------------------------------
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 6, 8, 6)

        add_role_btn = QPushButton("Add Role")
        add_role_btn.clicked.connect(self._on_add_role)
        toolbar.addWidget(add_role_btn)

        toolbar.addStretch()

        self._dirty_label = QLabel("")
        toolbar.addWidget(self._dirty_label)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("primary")
        self._save_btn.clicked.connect(self._on_save)
        toolbar.addWidget(self._save_btn)

        revert_btn = QPushButton("Revert")
        revert_btn.clicked.connect(self._on_revert)
        toolbar.addWidget(revert_btn)

        if self._help_registry is not None:
            help_btn = QPushButton("?")
            help_btn.setFixedSize(28, 28)
            help_btn.setToolTip("Routing help")
            help_btn.clicked.connect(self._on_help)
            toolbar.addWidget(help_btn)

        root_layout.addLayout(toolbar)

        # ---- Main area (canvas + sidebar) -----------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Canvas in a scroll area
        self._canvas = _RoutingCanvas()
        self._canvas.role_clicked.connect(self._on_role_clicked)
        self._canvas.model_clicked.connect(self._on_model_clicked)
        self._canvas.add_model_clicked.connect(self._on_add_model_to_chain)
        self._canvas._remove_model_callback = self._on_remove_model_from_chain
        self._canvas._delete_role_callback = self._on_delete_role

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._canvas)
        splitter.addWidget(scroll)

        # Properties sidebar
        self._sidebar = _PropertiesSidebar()
        self._sidebar.test_connection_requested.connect(self._on_test_connection)
        self._sidebar.edit_model_requested.connect(self._on_edit_model)
        self._sidebar.move_up_requested.connect(self._on_move_up)
        self._sidebar.move_down_requested.connect(self._on_move_down)
        splitter.addWidget(self._sidebar)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root_layout.addWidget(splitter, 1)

        # ---- Triage preview (collapsible below canvas) ----------------
        self._triage_preview = _TriagePreview()
        root_layout.addWidget(self._triage_preview)

    # ==================================================================
    # Data loading
    # ==================================================================

    def _load_from_api(self) -> None:
        """Pull current state from the API into working copies."""
        self._working_roles.clear()
        for rc in self._api.list_roles():
            self._working_roles[rc.role] = list(rc.chain)

        self._dirty = False
        self._update_dirty_label()
        self._rebuild_canvas()
        self._refresh_sidebar()
        self._refresh_triage()

    def _rebuild_canvas(self) -> None:
        """Rebuild the canvas node data from working copies."""
        from aurarouter.semantic_verbs import BUILTIN_VERBS

        verb_map = {v.role: v for v in BUILTIN_VERBS}
        models_cache: dict[str, dict] = {}
        for m in self._api.list_models():
            models_cache[m.model_id] = m.config

        role_nodes: list[_RoleNode] = []
        model_nodes: list[_ModelNode] = []

        for role_name, chain in self._working_roles.items():
            verb = verb_map.get(role_name)
            rn = _RoleNode(
                name=role_name,
                description=verb.description if verb else "",
                required=verb.required if verb else False,
                synonyms=verb.synonyms if verb else [],
                chain=list(chain),
            )
            role_nodes.append(rn)

            for idx, mid in enumerate(chain):
                cfg = models_cache.get(mid, {})
                mn = _ModelNode(
                    model_id=mid,
                    provider=cfg.get("provider", "unknown"),
                    tier=cfg.get("hosting_tier", ""),
                    role_name=role_name,
                    index=idx,
                )
                model_nodes.append(mn)

        self._canvas.set_data(role_nodes, model_nodes)
        self._canvas.set_selection(
            role=self._selected_role if self._selected_model is None else None,
            model=self._selected_model,
        )

    def _refresh_sidebar(self) -> None:
        """Update the properties sidebar based on current selection."""
        from aurarouter.semantic_verbs import BUILTIN_VERBS

        if self._selected_model is not None:
            role, mid = self._selected_model
            chain = self._working_roles.get(role, [])
            idx = self._selected_model_index or 0
            cfg = {}
            m = self._api.get_model(mid)
            if m is not None:
                cfg = m.config
            verb_map = {v.role: v for v in BUILTIN_VERBS}
            mn = _ModelNode(
                model_id=mid,
                provider=cfg.get("provider", "unknown"),
                tier=cfg.get("hosting_tier", ""),
                role_name=role,
                index=idx,
            )
            self._sidebar.show_model(mn, cfg, len(chain))
        elif self._selected_role:
            verb_map = {v.role: v for v in BUILTIN_VERBS}
            verb = verb_map.get(self._selected_role)
            chain = self._working_roles.get(self._selected_role, [])
            rn = _RoleNode(
                name=self._selected_role,
                description=verb.description if verb else "",
                required=verb.required if verb else False,
                synonyms=verb.synonyms if verb else [],
                chain=list(chain),
            )
            self._sidebar.show_role(rn)
        else:
            missing = self._api.get_missing_required_roles()
            # Also check working copy for missing required
            from aurarouter.semantic_verbs import get_required_roles
            configured = set(self._working_roles.keys())
            working_missing = [r for r in get_required_roles() if r not in configured]
            n_models = len(set(
                mid for chain in self._working_roles.values() for mid in chain
            ))
            self._sidebar.show_overview(
                len(self._working_roles), n_models, working_missing,
            )

    def _refresh_triage(self) -> None:
        rules = self._api.get_triage_rules()
        self._triage_preview.set_rules(rules)

    # ==================================================================
    # Dirty state
    # ==================================================================

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        if self._dirty:
            self._dirty_label.setText("Unsaved changes")
            self._dirty_label.setStyleSheet(
                "color: #f9e2af; font-weight: bold; font-size: 10px;"
            )
            self._save_btn.setEnabled(True)
        else:
            self._dirty_label.setText("")
            self._save_btn.setEnabled(True)

    def _validate_for_save(self) -> Optional[str]:
        """Return an error message if save is not allowed, or None if ok."""
        from aurarouter.semantic_verbs import get_required_roles

        for role in get_required_roles():
            chain = self._working_roles.get(role, [])
            if not chain:
                return (
                    f"Required role '{role}' has no models in its chain. "
                    f"Add at least one model before saving."
                )
        return None

    def has_unsaved_changes(self) -> bool:
        """Public accessor for dirty state (for navigation guards)."""
        return self._dirty

    def confirm_discard(self) -> bool:
        """Show a confirmation dialog if there are unsaved changes.

        Returns True if it is safe to proceed (no changes or user confirmed).
        """
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved routing changes. Discard them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # ==================================================================
    # Selection handlers
    # ==================================================================

    def _on_role_clicked(self, role: str) -> None:
        if role:
            self._selected_role = role
            self._selected_model = None
            self._selected_model_index = None
        else:
            self._selected_role = None
            self._selected_model = None
            self._selected_model_index = None
        self._canvas.set_selection(
            role=self._selected_role if self._selected_model is None else None,
            model=self._selected_model,
        )
        self._refresh_sidebar()

    def _on_model_clicked(self, role: str, model_id: str) -> None:
        chain = self._working_roles.get(role, [])
        idx = 0
        for i, mid in enumerate(chain):
            if mid == model_id:
                idx = i
                break
        self._selected_role = role
        self._selected_model = (role, model_id)
        self._selected_model_index = idx
        self._canvas.set_selection(model=self._selected_model)
        self._refresh_sidebar()

    # ==================================================================
    # Chain editing
    # ==================================================================

    def _on_add_model_to_chain(self, role: str) -> None:
        """Show dropdown to pick a model and append it to the role's chain."""
        available = self._api.list_models()
        if not available:
            QMessageBox.information(
                self, "No Models",
                "No models are configured. Add models in the Configuration panel first.",
            )
            return

        # Build a popup to select from available models
        items = [m.model_id for m in available]
        combo_dialog = QWidget(self, Qt.WindowType.Popup)
        combo_layout = QVBoxLayout(combo_dialog)
        combo_layout.setContentsMargins(4, 4, 4, 4)
        combo = QComboBox()
        combo.addItems(items)
        combo_layout.addWidget(combo)
        ok_btn = QPushButton("Add")
        combo_layout.addWidget(ok_btn)

        def _accept() -> None:
            model_id = combo.currentText()
            if model_id:
                chain = self._working_roles.setdefault(role, [])
                chain.append(model_id)
                self._mark_dirty()
                self._rebuild_canvas()
                self._refresh_sidebar()
            combo_dialog.close()

        ok_btn.clicked.connect(_accept)

        # Position near the [+] button
        add_rect = self._canvas._add_btn_rects.get(role)
        if add_rect is not None:
            global_pos = self._canvas.mapToGlobal(add_rect.bottomLeft().toPoint())
            combo_dialog.move(global_pos)

        combo_dialog.show()

    def _on_remove_model_from_chain(
        self, role: str, model_id: str, index: int,
    ) -> None:
        """Remove a model from a role's chain by index."""
        chain = self._working_roles.get(role, [])
        if 0 <= index < len(chain) and chain[index] == model_id:
            chain.pop(index)
        elif model_id in chain:
            chain.remove(model_id)
        self._mark_dirty()
        # Clear selection if removed the selected model
        if self._selected_model == (role, model_id):
            self._selected_model = None
            self._selected_model_index = None
        self._rebuild_canvas()
        self._refresh_sidebar()

    def _on_add_role(self) -> None:
        """Dialog to add a new role with autocomplete from known roles."""
        from aurarouter.semantic_verbs import get_known_roles

        combo_dialog = QWidget(self, Qt.WindowType.Dialog)
        combo_dialog.setWindowTitle("Add Role")
        combo_dialog.setMinimumWidth(300)
        lay = QVBoxLayout(combo_dialog)

        lay.addWidget(QLabel("Role name:"))
        combo = QComboBox()
        combo.setEditable(True)
        known = get_known_roles()
        # Filter out already-configured roles
        existing = set(self._working_roles.keys())
        suggestions = [r for r in known if r not in existing]
        combo.addItems(suggestions)
        combo.setCurrentText("")
        lay.addWidget(combo)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Add")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        def _accept() -> None:
            name = combo.currentText().strip()
            if not name:
                return
            if name in self._working_roles:
                QMessageBox.warning(
                    combo_dialog, "Duplicate", f"Role '{name}' already exists."
                )
                return
            self._working_roles[name] = []
            self._mark_dirty()
            self._selected_role = name
            self._selected_model = None
            self._selected_model_index = None
            self._rebuild_canvas()
            self._refresh_sidebar()
            combo_dialog.close()

        ok_btn.clicked.connect(_accept)
        cancel_btn.clicked.connect(combo_dialog.close)
        combo_dialog.show()

    def _on_delete_role(self, role: str) -> None:
        """Delete a role (from context menu)."""
        from aurarouter.semantic_verbs import get_required_roles

        if role in get_required_roles():
            QMessageBox.warning(
                self, "Cannot Delete",
                f"Role '{role}' is required and cannot be deleted.",
            )
            return

        reply = QMessageBox.question(
            self, "Delete Role",
            f"Delete role '{role}' and its chain?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._working_roles.pop(role, None)
            self._mark_dirty()
            if self._selected_role == role:
                self._selected_role = None
                self._selected_model = None
                self._selected_model_index = None
            self._rebuild_canvas()
            self._refresh_sidebar()

    def _on_move_up(self, role: str, model_id: str, index: int) -> None:
        chain = self._working_roles.get(role, [])
        if index > 0 and index < len(chain):
            chain[index], chain[index - 1] = chain[index - 1], chain[index]
            self._selected_model_index = index - 1
            self._mark_dirty()
            self._rebuild_canvas()
            self._refresh_sidebar()

    def _on_move_down(self, role: str, model_id: str, index: int) -> None:
        chain = self._working_roles.get(role, [])
        if index < len(chain) - 1:
            chain[index], chain[index + 1] = chain[index + 1], chain[index]
            self._selected_model_index = index + 1
            self._mark_dirty()
            self._rebuild_canvas()
            self._refresh_sidebar()

    # ==================================================================
    # Keyboard shortcut
    # ==================================================================

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_model is not None:
                role, mid = self._selected_model
                idx = self._selected_model_index or 0
                self._on_remove_model_from_chain(role, mid, idx)
                return
        super().keyPressEvent(event)

    # ==================================================================
    # Save / Revert
    # ==================================================================

    def _on_save(self) -> None:
        error = self._validate_for_save()
        if error:
            QMessageBox.warning(self, "Validation Error", error)
            return

        try:
            # Apply working changes to the API
            for role, chain in self._working_roles.items():
                self._api.set_role_chain(role, chain)

            # Remove roles that were deleted from working copy
            current_api_roles = {rc.role for rc in self._api.list_roles()}
            for role in current_api_roles:
                if role not in self._working_roles:
                    self._api.remove_role(role)

            saved_path = self._api.save_config()
            self._dirty = False
            self._update_dirty_label()
            self.config_changed.emit()
            QMessageBox.information(
                self, "Saved", f"Routing configuration saved to:\n{saved_path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_revert(self) -> None:
        if not self._dirty:
            return
        reply = QMessageBox.question(
            self, "Revert",
            "Discard all unsaved routing changes and reload from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._api.reload_config()
            self._selected_role = None
            self._selected_model = None
            self._selected_model_index = None
            self._load_from_api()

    # ==================================================================
    # Model actions (sidebar)
    # ==================================================================

    def _on_test_connection(self, model_id: str) -> None:
        """Run health check on a model (blocking for simplicity)."""
        try:
            reports = self._api.check_health(model_id)
            if reports:
                r = reports[0]
                if r.healthy:
                    QMessageBox.information(
                        self, "Connection OK",
                        f"Model '{model_id}' is reachable.\n"
                        f"Latency: {r.latency:.2f}s\n{r.message}",
                    )
                else:
                    QMessageBox.warning(
                        self, "Connection Failed",
                        f"Model '{model_id}' is unreachable.\n{r.message}",
                    )
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _on_edit_model(self, model_id: str) -> None:
        """Open the ModelDialog to edit a model's configuration."""
        from aurarouter.gui.model_dialog import ModelDialog

        model_cfg = {}
        m = self._api.get_model(model_id)
        if m is not None:
            model_cfg = m.config

        dlg = ModelDialog(parent=self, model_id=model_id, model_config=model_cfg)
        if dlg.exec() == ModelDialog.DialogCode.Accepted:
            self._api.add_model(model_id, dlg.get_model_config())
            self._mark_dirty()
            self._rebuild_canvas()
            self._refresh_sidebar()

    # ==================================================================
    # Help
    # ==================================================================

    def _on_help(self) -> None:
        if self._help_registry is None:
            return
        roles_topic = self._help_registry.get("concept.roles")
        fallback_topic = self._help_registry.get("concept.fallback")

        body_parts: list[str] = []
        if roles_topic:
            body_parts.append(roles_topic.body)
        if fallback_topic:
            body_parts.append(fallback_topic.body)

        if not body_parts:
            body_parts.append("<p>No help content available for routing.</p>")

        from PySide6.QtWidgets import QDialog, QTextBrowser

        dlg = QDialog(self)
        dlg.setWindowTitle("Routing Help")
        dlg.setMinimumSize(500, 400)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setHtml("<br>".join(body_parts))
        lay.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()
