"""Composite widget for task context input with file attachment support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# File types supported for text extraction.
_SUPPORTED_EXTENSIONS = {
    ".txt", ".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml",
    ".csv", ".xml", ".html", ".css", ".go", ".rs", ".java", ".cpp",
    ".c", ".h", ".hpp", ".sh", ".bat", ".ps1", ".toml", ".ini", ".cfg",
    ".log", ".sql", ".r", ".rb",
}

_MAX_FILE_SIZE = 512 * 1024  # 512 KB per file


class _FileChip(QWidget):
    """A small removable chip showing an attached filename."""

    remove_clicked = Signal(str)  # emits the file path

    def __init__(self, file_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.file_path = file_path
        name = Path(file_path).name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        label = QLabel(name)
        label.setStyleSheet(
            "background-color: #e0e0e0; border-radius: 3px; padding: 2px 6px;"
        )
        layout.addWidget(label)

        remove_btn = QPushButton("\u00d7")  # multiplication sign as close icon
        remove_btn.setFixedSize(18, 18)
        remove_btn.setStyleSheet(
            "border: none; color: #666; font-weight: bold; font-size: 12px;"
        )
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.file_path))
        layout.addWidget(remove_btn)


class DocumentInputWidget(QWidget):
    """Context input with optional file attachments.

    Combines a free-text ``QLineEdit`` with a file attachment zone.
    Attached files are read as text and concatenated into the context
    string returned by :meth:`get_context`.
    """

    context_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._attached_files: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ---- Text input ----
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("Paste relevant context...")
        self._text_input.textChanged.connect(self.context_changed)
        layout.addWidget(self._text_input)

        # ---- Attachment area ----
        attach_row = QHBoxLayout()

        self._attach_btn = QPushButton("Attach Files...")
        self._attach_btn.setFixedWidth(120)
        self._attach_btn.clicked.connect(self._on_attach)
        attach_row.addWidget(self._attach_btn)

        self._chips_container = QHBoxLayout()
        self._chips_container.setSpacing(4)
        attach_row.addLayout(self._chips_container)

        self._size_label = QLabel("")
        self._size_label.setStyleSheet("color: gray; font-size: 11px;")
        attach_row.addStretch()
        attach_row.addWidget(self._size_label)

        layout.addLayout(attach_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_context(self) -> str:
        """Return combined context: manual text + contents of attached files."""
        parts: list[str] = []

        manual = self._text_input.text().strip()
        if manual:
            parts.append(manual)

        for fp in self._attached_files:
            try:
                text = Path(fp).read_text(encoding="utf-8", errors="replace")
                if len(text) > _MAX_FILE_SIZE:
                    text = text[:_MAX_FILE_SIZE] + "\n... (truncated)"
                parts.append(f"--- {Path(fp).name} ---\n{text}")
            except Exception:
                parts.append(f"--- {Path(fp).name} --- (could not read)")

        return "\n\n".join(parts)

    def clear(self) -> None:
        """Clear all text and attachments."""
        self._text_input.clear()
        self._remove_all_chips()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_attach(self) -> None:
        filter_str = "Text files ({});;All files (*)".format(
            " ".join(f"*{ext}" for ext in sorted(_SUPPORTED_EXTENSIONS))
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach Files", "", filter_str
        )
        for path in paths:
            if path not in self._attached_files:
                self._attached_files.append(path)
                self._add_chip(path)

        self._update_size_label()
        self.context_changed.emit()

    def _add_chip(self, file_path: str) -> None:
        chip = _FileChip(file_path, parent=self)
        chip.remove_clicked.connect(self._on_remove_chip)
        self._chips_container.addWidget(chip)

    def _on_remove_chip(self, file_path: str) -> None:
        if file_path in self._attached_files:
            self._attached_files.remove(file_path)

        # Remove the chip widget.
        for i in range(self._chips_container.count()):
            item = self._chips_container.itemAt(i)
            widget = item.widget() if item else None
            if isinstance(widget, _FileChip) and widget.file_path == file_path:
                self._chips_container.removeWidget(widget)
                widget.deleteLater()
                break

        self._update_size_label()
        self.context_changed.emit()

    def _remove_all_chips(self) -> None:
        self._attached_files.clear()
        while self._chips_container.count():
            item = self._chips_container.takeAt(0)
            widget = item.widget() if item else None
            if widget:
                widget.deleteLater()
        self._update_size_label()

    def _update_size_label(self) -> None:
        if not self._attached_files:
            self._size_label.setText("")
            return

        total = 0
        for fp in self._attached_files:
            try:
                total += Path(fp).stat().st_size
            except OSError:
                pass

        # Rough token estimate: ~4 chars per token.
        tokens_est = total // 4
        if tokens_est > 1000:
            token_str = f"~{tokens_est // 1000}k tokens"
        else:
            token_str = f"~{tokens_est} tokens"

        count = len(self._attached_files)
        self._size_label.setText(
            f"{count} file{'s' if count != 1 else ''} attached ({token_str})"
        )
