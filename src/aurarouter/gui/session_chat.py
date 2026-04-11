"""Session chat widget for multi-turn conversational mode."""
from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Spacing, Typography
from aurarouter.gui.widgets.chat_bubble import ChatBubble
from aurarouter.gui.widgets.token_pressure import TokenPressureGauge


class SessionChatWidget(QWidget):
    """Multi-turn chat interface with session persistence.

    Contains:
    - Left sidebar: session list with search
    - Center: scrollable conversation with ChatBubble widgets
    - TokenPressureGauge in header
    - Bottom: text input with Send button
    """

    message_submitted = Signal(str)  # user typed message to send
    session_changed = Signal(str)    # session_id changed

    def __init__(self, api, palette: ColorPalette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._palette = palette or DARK_PALETTE
        self._current_session_id: str = ""
        self._pending_assistant_bubble: Optional[ChatBubble] = None
        self._token_timer = QTimer(self)
        self._token_timer.setSingleShot(True)
        self._token_timer.timeout.connect(self._flush_pending_text)
        self._pending_text: str = ""

        self._build_ui()
        self._load_sessions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with token pressure gauge
        header = QWidget(self)
        header.setMaximumHeight(52)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(Spacing.md, Spacing.sm, Spacing.md, Spacing.sm)

        self._session_label = QLabel("No active session", header)
        self._session_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;"
        )
        header_layout.addWidget(self._session_label)
        header_layout.addStretch()

        self._pressure_gauge = TokenPressureGauge(self._palette, header)
        self._pressure_gauge.condense_requested.connect(self._on_condense_requested)
        header_layout.addWidget(self._pressure_gauge)

        new_btn = QPushButton("+ New Session", header)
        new_btn.setStyleSheet(
            f"color: {self._palette.accent}; border: none; "
            f"font-size: {Typography.size_small}pt; background: transparent;"
        )
        new_btn.clicked.connect(self._create_new_session)
        header_layout.addWidget(new_btn)
        layout.addWidget(header)

        # Main area: session list + conversation
        splitter = QSplitter(Qt.Horizontal, self)

        # Session list sidebar
        sidebar = QWidget()
        sidebar.setMaximumWidth(180)
        sidebar.setMinimumWidth(120)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(4, 4, 4, 4)
        sb_layout.setSpacing(4)

        self._session_list = QListWidget(sidebar)
        self._session_list.setStyleSheet(
            f"QListWidget {{ background: {self._palette.bg_secondary}; border: none; "
            f"font-size: {Typography.size_small}pt; color: {self._palette.text_primary}; }}"
            f"QListWidget::item:selected {{ background: {self._palette.bg_selected}; }}"
            f"QListWidget::item:hover {{ background: {self._palette.bg_hover}; }}"
        )
        self._session_list.currentItemChanged.connect(self._on_session_selected)
        sb_layout.addWidget(self._session_list)
        splitter.addWidget(sidebar)

        # Conversation area
        conv_widget = QWidget()
        conv_layout = QVBoxLayout(conv_widget)
        conv_layout.setContentsMargins(0, 0, 0, 0)
        conv_layout.setSpacing(0)

        self._scroll = QScrollArea(conv_widget)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._conv_container = QWidget()
        self._conv_layout = QVBoxLayout(self._conv_container)
        self._conv_layout.setContentsMargins(Spacing.md, Spacing.md, Spacing.md, Spacing.md)
        self._conv_layout.setSpacing(Spacing.sm)
        self._conv_layout.addStretch()
        self._scroll.setWidget(self._conv_container)
        conv_layout.addWidget(self._scroll)

        # Input area
        input_frame = QFrame(conv_widget)
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_frame.setStyleSheet(f"background: {self._palette.bg_secondary}; "
                                   f"border-top: 1px solid {self._palette.border};")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(Spacing.sm, Spacing.sm, Spacing.sm, Spacing.sm)

        self._input = QTextEdit(input_frame)
        self._input.setMaximumHeight(80)
        self._input.setPlaceholderText("Type a message... (Ctrl+Return to send)")
        self._input.setStyleSheet(
            f"background: {self._palette.bg_primary}; "
            f"color: {self._palette.text_primary}; "
            f"border: 1px solid {self._palette.border}; border-radius: 4px;"
            f"font-size: {Typography.size_body}pt;"
        )
        input_layout.addWidget(self._input)

        send_btn = QPushButton("\u23ce Send", input_frame)
        send_btn.setFixedWidth(75)
        send_btn.setStyleSheet(
            f"background: {self._palette.accent}; color: {self._palette.text_inverse}; "
            f"border: none; border-radius: 4px; font-size: {Typography.size_body}pt; "
            f"padding: 4px 8px;"
        )
        send_btn.clicked.connect(self._on_send)
        input_layout.addWidget(send_btn)
        conv_layout.addWidget(input_frame)

        splitter.addWidget(conv_widget)
        splitter.setSizes([150, 500])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _load_sessions(self) -> None:
        self._session_list.clear()
        try:
            sessions = self._api.list_sessions()
        except Exception:
            return
        if not isinstance(sessions, list):
            return
        for s in sessions:
            if isinstance(s, dict) and "error" not in s:
                sid = s.get("session_id", "")
                count = s.get("message_count", 0)
                item = QListWidgetItem(f"{sid[:8]}\u2026 ({count} msgs)")
                item.setData(Qt.UserRole, sid)
                self._session_list.addItem(item)

    def _create_new_session(self) -> None:
        try:
            result = self._api.create_session()
        except Exception:
            return
        if not isinstance(result, dict) or "error" in result:
            return
        sid = result.get("session_id", "")
        if sid:
            self._switch_session(sid)
            self._load_sessions()

    def _on_session_selected(self, current: QListWidgetItem, previous) -> None:
        if current is None:
            return
        sid = current.data(Qt.UserRole)
        if sid and sid != self._current_session_id:
            self._switch_session(sid)

    def _switch_session(self, session_id: str) -> None:
        self._current_session_id = session_id
        self._pressure_gauge.set_session_id(session_id)
        self._session_label.setText(f"Session: {session_id[:12]}\u2026")
        self._clear_conversation()
        # Load existing messages
        try:
            session = self._api.get_session(session_id)
        except Exception:
            session = None
        if session and isinstance(session, dict) and "error" not in session:
            for msg in session.get("messages", []):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant", "system"):
                    self._add_bubble(role, content, msg.get("model_id", ""))
        self.session_changed.emit(session_id)

    def _clear_conversation(self) -> None:
        while self._conv_layout.count() > 1:
            item = self._conv_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ------------------------------------------------------------------
    # Chat rendering
    # ------------------------------------------------------------------

    def _add_bubble(self, role: str, content: str, model_id: str = "",
                    routing_context: dict | None = None, is_verified: bool = False) -> ChatBubble:
        bubble = ChatBubble(
            role=role, content=content, model_id=model_id,
            routing_context=routing_context, is_verified=is_verified,
            palette=self._palette, parent=self._conv_container,
        )
        # Insert before the trailing stretch
        count = self._conv_layout.count()
        self._conv_layout.insertWidget(count - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)
        return bubble

    def start_assistant_response(self, model_id: str = "") -> None:
        """Begin a streaming assistant response (creates empty bubble)."""
        self._pending_assistant_bubble = self._add_bubble("assistant", "", model_id)
        self._pending_text = ""

    def append_token(self, token: str) -> None:
        """Buffer incoming token and flush with 50ms debounce."""
        self._pending_text += token
        if not self._token_timer.isActive():
            self._token_timer.start(50)

    def _flush_pending_text(self) -> None:
        if self._pending_assistant_bubble and self._pending_text:
            self._pending_assistant_bubble.append_text(self._pending_text)
            self._pending_text = ""
            self._scroll_to_bottom()

    def finalize_response(self, routing_context: dict | None = None, is_verified: bool = False) -> None:
        """Finalize the current streaming response and update routing pill."""
        self._token_timer.stop()
        if self._pending_text and self._pending_assistant_bubble:
            self._pending_assistant_bubble.append_text(self._pending_text)
            self._pending_text = ""
        if self._pending_assistant_bubble and routing_context:
            self._pending_assistant_bubble.set_routing_context(routing_context)
        self._pending_assistant_bubble = None
        self._scroll_to_bottom()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        if not self._current_session_id:
            self._create_new_session()
        self._input.clear()
        self._add_bubble("user", text)
        self.message_submitted.emit(text)
        try:
            self._api.add_session_message(self._current_session_id, "user", text)
        except Exception:
            pass

    def keyPressEvent(self, event) -> None:
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Return:
            self._on_send()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Token pressure
    # ------------------------------------------------------------------

    def update_token_pressure(self, input_tokens: int, output_tokens: int, context_limit: int) -> None:
        self._pressure_gauge.set_stats(input_tokens, output_tokens, context_limit)

    def _on_condense_requested(self) -> None:
        if self._current_session_id:
            try:
                mgr = self._api._get_session_manager()
                if hasattr(mgr, "ensure_context_pressure"):
                    mgr.ensure_context_pressure(self._current_session_id, force=True)
            except Exception:
                pass
            self._pressure_gauge.set_pressure(0.5)  # Reset visual after condense

    def get_current_session_id(self) -> str:
        return self._current_session_id
