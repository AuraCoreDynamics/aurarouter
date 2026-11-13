"""Composite widget for task context input with file attachment support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QBuffer, QIODevice, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aurarouter._logging import get_logger
from aurarouter.media import IMAGE_EXTENSIONS, ImageInput

logger = get_logger("AuraRouter.GUI.DocumentInput")

# File types supported for text extraction.
_TEXT_EXTENSIONS = {
    ".txt", ".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml",
    ".csv", ".xml", ".html", ".css", ".go", ".rs", ".java", ".cpp",
    ".c", ".h", ".hpp", ".sh", ".bat", ".ps1", ".toml", ".ini", ".cfg",
    ".log", ".sql", ".r", ".rb",
}
_SUPPORTED_EXTENSIONS = _TEXT_EXTENSIONS | IMAGE_EXTENSIONS

_MAX_FILE_SIZE = 512 * 1024  # 512 KB per text file
_MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB per image file


class _FileChip(QWidget):
    """A small removable chip showing an attached filename."""

    remove_clicked = Signal(str)  # emits the file path

    def __init__(self, file_path: str, parent: Optional[QWidget] = None, is_image: bool = False):
        super().__init__(parent)
        self.file_path = file_path
        self.is_image = is_image
        name = Path(file_path).name
        prefix = "[IMG] " if is_image else ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        label = QLabel(f"{prefix}{name}")
        bg = "#1e3a5f" if is_image else "#313244"
        label.setStyleSheet(
            f"background-color: {bg}; color: #cdd6f4; "
            "border-radius: 3px; padding: 2px 6px;"
        )
        layout.addWidget(label)

        remove_btn = QPushButton("\u00d7")  # multiplication sign as close icon
        remove_btn.setFixedSize(18, 18)
        remove_btn.setStyleSheet(
            "border: none; color: #f38ba8; font-weight: bold; font-size: 12px; "
            "background: transparent;"
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
        self._attached_images: dict[str, tuple[ImageInput, int]] = {}
        self._clipboard_counter = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # ---- Text input ----
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("Paste relevant context or attach files/images...")
        self._text_input.textChanged.connect(self.context_changed)
        layout.addWidget(self._text_input)

        # ---- Attachment area ----
        attach_row = QHBoxLayout()

        self._attach_btn = QPushButton("Attach Files...")
        self._attach_btn.setFixedWidth(120)
        self._attach_btn.clicked.connect(self._on_attach)
        attach_row.addWidget(self._attach_btn)

        self._paste_img_btn = QPushButton("Paste Image")
        self._paste_img_btn.setFixedWidth(95)
        self._paste_img_btn.setToolTip("Paste an image from the system clipboard")
        self._paste_img_btn.clicked.connect(self._on_paste_image)
        attach_row.addWidget(self._paste_img_btn)

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
            except Exception as ex:  # noqa: F841
                logger.debug("gui.document_input.get_context_error", exc_info=True)
                parts.append(f"--- {Path(fp).name} --- (could not read)")

        return "\n\n".join(parts)

    def has_images(self) -> bool:
        """Return True if at least one image is attached."""
        return bool(self._attached_images)

    def get_images(self) -> list[dict]:
        """Return normalized image payload dicts for all attached images."""
        return [img.to_dict() for img, _ in self._attached_images.values()]

    def clear(self) -> None:
        """Clear all text and attachments."""
        self._text_input.clear()
        self._remove_all_chips()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_attach(self) -> None:
        img_exts = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))
        all_exts = " ".join(f"*{ext}" for ext in sorted(_SUPPORTED_EXTENSIONS))
        filter_str = (
            f"All supported ({all_exts});;"
            f"Images ({img_exts});;"
            f"All files (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Attach Files or Images", "", filter_str
        )
        for path in paths:
            p = Path(path)
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                if path not in self._attached_images:
                    try:
                        img_obj = ImageInput.from_path(p)
                        raw_len = len(img_obj.raw_bytes())
                        if raw_len > _MAX_IMAGE_SIZE:
                            continue
                        self._attached_images[path] = (img_obj, raw_len)
                        self._add_chip(path, is_image=True)
                    except Exception:
                        logger.debug("gui.document_input.attach_image_error", exc_info=True)
            elif path not in self._attached_files:
                self._attached_files.append(path)
                self._add_chip(path, is_image=False)

        self._update_size_label()
        self.context_changed.emit()

    def _on_paste_image(self) -> None:
        cb = QApplication.clipboard()
        if cb is None:
            return
        qimg = cb.image()
        if qimg.isNull():
            return
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        qimg.save(buf, "PNG")
        raw_bytes = bytes(buf.data())
        buf.close()
        if not raw_bytes or len(raw_bytes) > _MAX_IMAGE_SIZE:
            return
        self._clipboard_counter += 1
        key = f"clipboard-{self._clipboard_counter}.png"
        img_obj = ImageInput.from_bytes(raw_bytes, mime_type="image/png", filename=key)
        self._attached_images[key] = (img_obj, len(raw_bytes))
        self._add_chip(key, is_image=True)
        self._update_size_label()
        self.context_changed.emit()

    def _add_chip(self, file_path: str, is_image: bool = False) -> None:
        chip = _FileChip(file_path, parent=self, is_image=is_image)
        chip.remove_clicked.connect(self._on_remove_chip)
        self._chips_container.addWidget(chip)

    def _on_remove_chip(self, file_path: str) -> None:
        if file_path in self._attached_files:
            self._attached_files.remove(file_path)
        self._attached_images.pop(file_path, None)

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
        self._attached_images.clear()
        while self._chips_container.count():
            item = self._chips_container.takeAt(0)
            widget = item.widget() if item else None
            if widget:
                widget.deleteLater()
        self._update_size_label()

    def _update_size_label(self) -> None:
        if not self._attached_files and not self._attached_images:
            self._size_label.setText("")
            return

        total = sum(sz for _, sz in self._attached_images.values())
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
