"""Intent registry editor panel for AuraRouter GUI."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, ColorPalette, Spacing, Typography


class IntentEditorPanel(QWidget):
    """Panel for browsing and managing the intent registry and active analyzer."""

    analyzer_changed = Signal(str)

    def __init__(
        self,
        api,
        palette: ColorPalette | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._api = api
        self._palette = palette or DARK_PALETTE
        self._intents: list[dict] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.sm)

        # Section header
        header = QHBoxLayout()
        title = QLabel("Intent Registry", self)
        title.setStyleSheet(
            f"color: {self._palette.text_primary}; "
            f"font-size: {Typography.size_h2}pt; font-weight: bold;"
        )
        header.addWidget(title)
        header.addStretch()
        refresh_btn = QPushButton("\u27f3", self)
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.setToolTip("Refresh intent list")
        refresh_btn.clicked.connect(self.refresh)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Intent table
        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ["Intent", "Target Role", "Source", "Priority", "Actions"]
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setStyleSheet(
            f"QTableWidget {{ background: {self._palette.bg_secondary}; "
            f"color: {self._palette.text_primary}; "
            f"font-size: {Typography.size_small}pt; border: none; }}"
            f"QHeaderView::section {{ background: {self._palette.bg_tertiary}; "
            f"color: {self._palette.text_secondary}; "
            f"padding: 4px; border: none; }}"
        )
        layout.addWidget(self._table)

        # Add custom intent button
        add_btn = QPushButton("+ Add Custom Intent", self)
        add_btn.setStyleSheet(
            f"color: {self._palette.accent}; border: 1px solid {self._palette.accent}; "
            f"border-radius: 4px; padding: 4px 8px; font-size: {Typography.size_small}pt; "
            f"background: transparent;"
        )
        add_btn.clicked.connect(self._on_add_intent)
        layout.addWidget(add_btn)

        # Add intent form (hidden by default)
        self._add_form = QFrame(self)
        self._add_form.setFrameShape(QFrame.Shape.StyledPanel)
        self._add_form.setStyleSheet(f"background: {self._palette.bg_secondary};")
        form_layout = QVBoxLayout(self._add_form)
        form_layout.setContentsMargins(Spacing.sm, Spacing.sm, Spacing.sm, Spacing.sm)
        form_layout.setSpacing(Spacing.xs)

        for label, attr in [
            ("Intent Name:", "_form_name"),
            ("Description:", "_form_desc"),
            ("Target Role:", "_form_role"),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label, self._add_form)
            lbl.setFixedWidth(110)
            lbl.setStyleSheet(
                f"color: {self._palette.text_secondary}; font-size: {Typography.size_small}pt;"
            )
            row.addWidget(lbl)
            edit = QLineEdit(self._add_form)
            edit.setStyleSheet(
                f"background: {self._palette.bg_primary}; color: {self._palette.text_primary}; "
                f"border: 1px solid {self._palette.border}; border-radius: 3px; "
                f"font-size: {Typography.size_small}pt;"
            )
            setattr(self, attr, edit)
            row.addWidget(edit)
            form_layout.addLayout(row)

        form_btns = QHBoxLayout()
        save_btn = QPushButton("Save", self._add_form)
        save_btn.clicked.connect(self._on_save_intent)
        save_btn.setStyleSheet(
            f"background: {self._palette.accent}; color: {self._palette.text_inverse}; "
            f"border: none; border-radius: 3px; padding: 3px 10px;"
        )
        cancel_btn = QPushButton("Cancel", self._add_form)
        cancel_btn.clicked.connect(lambda: self._add_form.hide())
        cancel_btn.setStyleSheet(
            f"color: {self._palette.text_secondary}; border: none; background: transparent;"
        )
        form_btns.addWidget(save_btn)
        form_btns.addWidget(cancel_btn)
        form_btns.addStretch()
        form_layout.addLayout(form_btns)
        self._add_form.hide()
        layout.addWidget(self._add_form)

        # Analyzer section
        analyzer_frame = QFrame(self)
        analyzer_frame.setFrameShape(QFrame.Shape.StyledPanel)
        analyzer_frame.setStyleSheet(f"background: {self._palette.bg_secondary};")
        al_layout = QHBoxLayout(analyzer_frame)
        al_layout.setContentsMargins(Spacing.sm, Spacing.xs, Spacing.sm, Spacing.xs)

        al_label = QLabel("Active Analyzer:", analyzer_frame)
        al_label.setStyleSheet(f"color: {self._palette.text_secondary};")
        al_layout.addWidget(al_label)

        self._analyzer_combo = QComboBox(analyzer_frame)
        self._analyzer_combo.setStyleSheet(
            f"background: {self._palette.bg_primary}; color: {self._palette.text_primary}; "
            f"border: 1px solid {self._palette.border}; border-radius: 3px;"
        )
        self._analyzer_combo.currentTextChanged.connect(self._on_analyzer_changed)
        al_layout.addWidget(self._analyzer_combo)

        validate_btn = QPushButton("Validate", analyzer_frame)
        validate_btn.clicked.connect(self._on_validate)
        al_layout.addWidget(validate_btn)
        layout.addWidget(analyzer_frame)

        # Validation result label
        self._validation_label = QLabel("", self)
        self._validation_label.setWordWrap(True)
        self._validation_label.setStyleSheet(f"font-size: {Typography.size_small}pt;")
        self._validation_label.hide()
        layout.addWidget(self._validation_label)

    def refresh(self) -> None:
        """Reload intents and analyzer list from API."""
        try:
            self._intents = self._api.list_intents()
        except Exception:
            self._intents = []
        self._populate_table()
        self._load_analyzers()

    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        for intent in self._intents:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(intent.get("name", "")))
            self._table.setItem(row, 1, QTableWidgetItem(intent.get("target_role", "")))
            source = intent.get("source", "builtin")
            self._table.setItem(row, 2, QTableWidgetItem(source))
            self._table.setItem(row, 3, QTableWidgetItem(str(intent.get("priority", 0))))
            if source != "builtin":
                edit_btn = QPushButton("\u270e", self._table)
                edit_btn.setFixedSize(24, 20)
                edit_btn.setStyleSheet(
                    f"color: {self._palette.accent}; border: none; background: transparent;"
                )
                self._table.setCellWidget(row, 4, edit_btn)
            else:
                self._table.setItem(row, 4, QTableWidgetItem("\u2014"))

    def _load_analyzers(self) -> None:
        try:
            analyzers = self._api.catalog_list(kind="analyzer")
            names = [a.get("artifact_id", "") for a in analyzers if a.get("artifact_id")]
        except Exception:
            names = []

        current = ""
        try:
            current = self._api.get_active_analyzer() or ""
        except Exception:
            pass

        self._analyzer_combo.blockSignals(True)
        self._analyzer_combo.clear()
        self._analyzer_combo.addItems(names)
        if current in names:
            self._analyzer_combo.setCurrentText(current)
        self._analyzer_combo.blockSignals(False)

    def _on_add_intent(self) -> None:
        self._form_name.clear()
        self._form_desc.clear()
        self._form_role.clear()
        self._add_form.setVisible(not self._add_form.isVisible())

    def _on_save_intent(self) -> None:
        name = self._form_name.text().strip()
        role = self._form_role.text().strip()
        if not name or not role:
            QMessageBox.warning(self, "Validation", "Intent name and target role are required.")
            return
        # Custom intents are added through analyzer spec — show placeholder message
        QMessageBox.information(
            self,
            "Intent Added",
            f"Custom intent '{name}' \u2192 '{role}' recorded.\n"
            "To persist, add it to your active analyzer's role_bindings and reload.",
        )
        self._add_form.hide()
        self.refresh()

    def _on_analyzer_changed(self, name: str) -> None:
        if not name:
            return
        try:
            self._api.set_active_analyzer(name)
            self.analyzer_changed.emit(name)
            self.refresh()
        except Exception:
            pass

    def _on_validate(self) -> None:
        try:
            from aurarouter.analyzer_schema import validate_analyzer_spec

            active = self._api.get_active_analyzer()
            if not active:
                self._show_validation("No active analyzer selected.", error=False)
                return
            artifact = self._api.catalog_get(active)
            if not artifact:
                self._show_validation(
                    f"Analyzer '{active}' not found in catalog.", error=True
                )
                return
            result = validate_analyzer_spec(artifact)
            if result.valid:
                msg = "\u2713 Analyzer spec is valid."
                if result.warnings:
                    msg += f"\n\u26a0 Warnings: {'; '.join(result.warnings)}"
                self._show_validation(msg, error=False)
            else:
                self._show_validation(
                    f"\u2717 Errors: {'; '.join(result.errors)}", error=True
                )
        except Exception as exc:
            self._show_validation(f"Validation error: {exc}", error=True)

    def _show_validation(self, msg: str, error: bool = False) -> None:
        color = self._palette.error if error else self._palette.success
        self._validation_label.setStyleSheet(
            f"font-size: {Typography.size_small}pt; color: {color};"
        )
        self._validation_label.setText(msg)
        self._validation_label.setVisible(True)

    def highlight(self, context: dict) -> None:
        """Highlight a specific intent (called by cross-panel navigation)."""
        intent_name = context.get("intent_name", "")
        if not intent_name:
            return
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.text() == intent_name:
                self._table.selectRow(row)
                self._table.scrollToItem(item)
                break
