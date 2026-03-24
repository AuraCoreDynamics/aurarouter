"""Settings panel — system configuration for MCP tools, budget, privacy, and YAML editor.

Provides :class:`SettingsPanel`, a unified configuration widget built from
:class:`~aurarouter.gui.widgets.collapsible_section.CollapsibleSection`
sections.  Uses :class:`~aurarouter.api.AuraRouterAPI` as its data source.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.theme import DARK_PALETTE, SPACING, TYPOGRAPHY
from aurarouter.gui.widgets.collapsible_section import CollapsibleSection
from aurarouter.gui.widgets.help_tooltip import HelpTooltip

if TYPE_CHECKING:
    from aurarouter.api import AuraRouterAPI
    from aurarouter.gui.help.content import HelpRegistry


class SettingsPanel(QWidget):
    """System configuration — MCP tools, budget, privacy, YAML editor."""

    settings_saved = Signal()  # emitted after a successful save

    def __init__(
        self,
        api: "AuraRouterAPI",
        help_registry: Optional["HelpRegistry"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._api = api
        self._help_registry = help_registry
        self._dirty = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scroll area wrapping all sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setSpacing(SPACING.sm)

        # ---- Section 1: MCP Tools ----
        self._mcp_section = CollapsibleSection(
            "MCP Tools", initially_expanded=True,
        )
        self._mcp_container = QWidget()
        self._mcp_layout = QVBoxLayout(self._mcp_container)
        self._mcp_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        self._mcp_checkboxes: dict[str, QCheckBox] = {}
        self._build_mcp_tools_section()
        self._mcp_section.add_widget(self._mcp_container)
        self._layout.addWidget(self._mcp_section)

        # ---- Section 2: Budget & Cost ----
        self._budget_section = CollapsibleSection(
            "Budget & Cost", initially_expanded=False,
        )
        self._budget_container = QWidget()
        self._budget_layout = QVBoxLayout(self._budget_container)
        self._budget_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        self._build_budget_section()
        self._budget_section.add_widget(self._budget_container)
        self._layout.addWidget(self._budget_section)

        # ---- Section 3: Privacy ----
        self._privacy_section = CollapsibleSection(
            "Privacy", initially_expanded=False,
        )
        self._privacy_container = QWidget()
        self._privacy_layout = QVBoxLayout(self._privacy_container)
        self._privacy_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        self._build_privacy_section()
        self._privacy_section.add_widget(self._privacy_container)
        self._layout.addWidget(self._privacy_section)

        # ---- Section 4: System ----
        self._system_section = CollapsibleSection(
            "System", initially_expanded=False,
        )
        self._system_container = QWidget()
        self._system_layout = QVBoxLayout(self._system_container)
        self._system_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        self._build_system_section()
        self._system_section.add_widget(self._system_container)
        self._layout.addWidget(self._system_section)

        # ---- Section 5: YAML Editor ----
        self._yaml_section = CollapsibleSection(
            "YAML Editor", initially_expanded=False,
        )
        self._yaml_container = QWidget()
        self._yaml_layout = QVBoxLayout(self._yaml_container)
        self._yaml_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        self._build_yaml_section()
        self._yaml_section.add_widget(self._yaml_container)
        self._layout.addWidget(self._yaml_section)

        self._layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

        # ---- Bottom toolbar ----
        toolbar = QHBoxLayout()
        self._dirty_label = QLabel("")
        toolbar.addWidget(self._dirty_label)
        toolbar.addStretch()

        revert_all_btn = QPushButton("Revert All")
        revert_all_btn.clicked.connect(self._on_revert_all)
        toolbar.addWidget(revert_all_btn)

        save_all_btn = QPushButton("Save All")
        save_all_btn.setObjectName("primary")
        save_all_btn.clicked.connect(self._on_save_all)
        toolbar.addWidget(save_all_btn)

        outer.addLayout(toolbar)

        # Initial data load
        self._refresh_all()

    # ==================================================================
    # Section builders
    # ==================================================================

    def _build_mcp_tools_section(self) -> None:
        """Populate the MCP Tools collapsible section."""
        tools = self._api.get_mcp_tools()
        for tool in tools:
            cb = QCheckBox(f"{tool.name} \u2014 {tool.description}" if tool.description else tool.name)
            cb.setChecked(tool.enabled)
            cb.stateChanged.connect(
                lambda state, name=tool.name: self._on_mcp_tool_toggled(name, state)
            )
            self._mcp_checkboxes[tool.name] = cb
            self._mcp_layout.addWidget(cb)

        note = QLabel("Changes take effect after saving and restarting the MCP server.")
        note.setStyleSheet(f"color: {DARK_PALETTE.text_disabled}; font-size: {TYPOGRAPHY.size_small}px;")
        note.setWordWrap(True)
        self._mcp_layout.addWidget(note)

    def _on_mcp_tool_toggled(self, name: str, state: int) -> None:
        enabled = state == Qt.CheckState.Checked.value
        self._api.set_mcp_tool(name, enabled)
        self._mark_dirty()

    def _build_budget_section(self) -> None:
        """Populate the Budget & Cost collapsible section."""
        lay = self._budget_layout

        # Enable checkbox
        self._budget_enabled_cb = QCheckBox("Enable budget enforcement")
        status = self._api.get_budget_status()
        self._budget_enabled_cb.setChecked(status is not None and status.get("allowed") is not None)
        self._budget_enabled_cb.stateChanged.connect(lambda _: self._mark_dirty())
        lay.addWidget(self._budget_enabled_cb)

        # Daily limit
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Daily limit ($):"))
        self._daily_limit_spin = QDoubleSpinBox()
        self._daily_limit_spin.setRange(0.0, 99999.99)
        self._daily_limit_spin.setDecimals(2)
        self._daily_limit_spin.setSpecialValueText("No limit")
        if status and status.get("daily_limit") is not None:
            self._daily_limit_spin.setValue(status["daily_limit"])
        self._daily_limit_spin.valueChanged.connect(lambda _: self._mark_dirty())
        row1.addWidget(self._daily_limit_spin)
        row1.addStretch()
        lay.addLayout(row1)

        # Monthly limit
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Monthly limit ($):"))
        self._monthly_limit_spin = QDoubleSpinBox()
        self._monthly_limit_spin.setRange(0.0, 99999.99)
        self._monthly_limit_spin.setDecimals(2)
        self._monthly_limit_spin.setSpecialValueText("No limit")
        if status and status.get("monthly_limit") is not None:
            self._monthly_limit_spin.setValue(status["monthly_limit"])
        self._monthly_limit_spin.valueChanged.connect(lambda _: self._mark_dirty())
        row2.addWidget(self._monthly_limit_spin)
        row2.addStretch()
        lay.addLayout(row2)

        # Current spend display
        self._spend_label = QLabel()
        self._update_spend_label(status)
        lay.addWidget(self._spend_label)

        # Pricing overrides table
        lay.addWidget(QLabel("Pricing overrides (per 1M tokens):"))
        self._pricing_table = QTableWidget(0, 3)
        self._pricing_table.setHorizontalHeaderLabels(["Model", "Input ($)", "Output ($)"])
        self._pricing_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._pricing_table.setMaximumHeight(150)
        lay.addWidget(self._pricing_table)
        self._refresh_pricing_overrides()

    def _update_spend_label(self, status: Optional[dict]) -> None:
        if status:
            daily = status.get("daily_spend", 0.0)
            monthly = status.get("monthly_spend", 0.0)
            self._spend_label.setText(
                f"Current spend: ${daily:.2f} today / ${monthly:.2f} this month"
            )
        else:
            self._spend_label.setText("Budget tracking is disabled.")
        self._spend_label.setStyleSheet(
            f"color: {DARK_PALETTE.text_secondary}; font-size: {TYPOGRAPHY.size_small}px;"
        )

    def _refresh_pricing_overrides(self) -> None:
        """Load pricing overrides from model configs into the table."""
        self._pricing_table.setRowCount(0)
        models = self._api.list_models()
        for m in models:
            inp = m.config.get("cost_per_1m_input")
            out = m.config.get("cost_per_1m_output")
            if inp is not None or out is not None:
                row = self._pricing_table.rowCount()
                self._pricing_table.insertRow(row)
                self._pricing_table.setItem(row, 0, QTableWidgetItem(m.model_id))
                self._pricing_table.setItem(row, 1, QTableWidgetItem(str(inp or "")))
                self._pricing_table.setItem(row, 2, QTableWidgetItem(str(out or "")))

    def _build_privacy_section(self) -> None:
        """Populate the Privacy collapsible section."""
        lay = self._privacy_layout

        # Enable checkbox
        self._privacy_enabled_cb = QCheckBox("Enable privacy auditing")
        settings = self._api.get_system_settings()
        self._privacy_enabled_cb.setChecked(settings.get("enable_privacy", True))
        self._privacy_enabled_cb.stateChanged.connect(lambda _: self._mark_dirty())
        lay.addWidget(self._privacy_enabled_cb)

        # Built-in patterns (read-only)
        lay.addWidget(QLabel("Built-in patterns (read-only):"))
        self._builtin_patterns_table = QTableWidget(0, 3)
        self._builtin_patterns_table.setHorizontalHeaderLabels(["Name", "Severity", "Description"])
        self._builtin_patterns_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._builtin_patterns_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._builtin_patterns_table.setMaximumHeight(180)
        lay.addWidget(self._builtin_patterns_table)
        self._refresh_builtin_patterns()

        # Custom patterns (editable)
        lay.addWidget(QLabel("Custom patterns:"))
        self._custom_patterns_table = QTableWidget(0, 3)
        self._custom_patterns_table.setHorizontalHeaderLabels(["Name", "Severity", "Regex Pattern"])
        self._custom_patterns_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._custom_patterns_table.setMaximumHeight(150)
        lay.addWidget(self._custom_patterns_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Pattern")
        add_btn.clicked.connect(self._on_add_custom_pattern)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove Pattern")
        remove_btn.clicked.connect(self._on_remove_custom_pattern)
        btn_row.addWidget(remove_btn)

        test_btn = QPushButton("Test Pattern")
        test_btn.clicked.connect(self._on_test_pattern)
        btn_row.addWidget(test_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)

    def _refresh_builtin_patterns(self) -> None:
        """Load built-in privacy patterns into the read-only table."""
        from aurarouter.savings.privacy import _BUILTIN_PATTERNS

        self._builtin_patterns_table.setRowCount(0)
        for pat in _BUILTIN_PATTERNS:
            row = self._builtin_patterns_table.rowCount()
            self._builtin_patterns_table.insertRow(row)
            self._builtin_patterns_table.setItem(row, 0, QTableWidgetItem(pat.name))
            self._builtin_patterns_table.setItem(row, 1, QTableWidgetItem(pat.severity))
            self._builtin_patterns_table.setItem(row, 2, QTableWidgetItem(pat.description))

    def _on_add_custom_pattern(self) -> None:
        row = self._custom_patterns_table.rowCount()
        self._custom_patterns_table.insertRow(row)
        self._custom_patterns_table.setItem(row, 0, QTableWidgetItem("New Pattern"))
        sev_combo = QComboBox()
        sev_combo.addItems(["low", "medium", "high"])
        self._custom_patterns_table.setCellWidget(row, 1, sev_combo)
        self._custom_patterns_table.setItem(row, 2, QTableWidgetItem(r"\b...regex...\b"))
        self._mark_dirty()

    def _on_remove_custom_pattern(self) -> None:
        row = self._custom_patterns_table.currentRow()
        if row >= 0:
            self._custom_patterns_table.removeRow(row)
            self._mark_dirty()

    def _on_test_pattern(self) -> None:
        """Test the selected custom pattern against user-entered text."""
        row = self._custom_patterns_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Test Pattern", "Select a custom pattern row first.")
            return
        pattern_item = self._custom_patterns_table.item(row, 2)
        if not pattern_item:
            return
        pattern_text = pattern_item.text()
        try:
            compiled = re.compile(pattern_text)
        except re.error as exc:
            QMessageBox.warning(self, "Invalid Regex", f"Pattern error: {exc}")
            return

        # Simple test against a sample string
        from PySide6.QtWidgets import QInputDialog

        test_string, ok = QInputDialog.getText(
            self, "Test Pattern", "Enter text to test against:"
        )
        if not ok or not test_string:
            return
        matches = compiled.findall(test_string)
        if matches:
            QMessageBox.information(
                self, "Pattern Matches",
                f"Found {len(matches)} match(es):\n" + "\n".join(str(m) for m in matches[:10]),
            )
        else:
            QMessageBox.information(self, "No Matches", "Pattern did not match any text.")

    def _build_system_section(self) -> None:
        """Populate the System collapsible section."""
        lay = self._system_layout

        # Log level
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Log level:"))
        self._log_level_combo = QComboBox()
        self._log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        # Read current from config
        cfg = self._api._config.config  # noqa: SLF001
        log_cfg = cfg.get("logging", {})
        current_level = log_cfg.get("level", "INFO").upper()
        idx = self._log_level_combo.findText(current_level)
        if idx >= 0:
            self._log_level_combo.setCurrentIndex(idx)
        self._log_level_combo.currentTextChanged.connect(lambda _: self._mark_dirty())
        row1.addWidget(self._log_level_combo)
        row1.addStretch()
        lay.addLayout(row1)

        # Default timeout
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Default timeout (s):"))
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(5, 600)
        server_cfg = cfg.get("server", {})
        self._timeout_spin.setValue(server_cfg.get("timeout", 60))
        self._timeout_spin.valueChanged.connect(lambda _: self._mark_dirty())
        row2.addWidget(self._timeout_spin)
        row2.addStretch()
        lay.addLayout(row2)

        # Max review iterations
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Max review iterations:"))
        self._max_review_spin = QSpinBox()
        self._max_review_spin.setRange(0, 10)
        self._max_review_spin.setValue(cfg.get("max_review_iterations", 2))
        self._max_review_spin.valueChanged.connect(lambda _: self._mark_dirty())
        row3.addWidget(self._max_review_spin)
        row3.addStretch()
        lay.addLayout(row3)

        # Session toggle
        self._session_cb = QCheckBox("Enable sessions (experimental)")
        settings = self._api.get_system_settings()
        self._session_cb.setChecked(settings.get("enable_sessions", False))
        self._session_cb.stateChanged.connect(lambda _: self._mark_dirty())
        lay.addWidget(self._session_cb)

    def _build_yaml_section(self) -> None:
        """Populate the YAML Editor collapsible section."""
        lay = self._yaml_layout

        warn = QLabel(
            "Warning: Editing YAML directly bypasses validation. "
            "Invalid changes may prevent AuraRouter from starting."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(
            f"color: {DARK_PALETTE.warning}; font-size: {TYPOGRAPHY.size_small}px;"
        )
        lay.addWidget(warn)

        self._yaml_editor = QTextEdit()
        self._yaml_editor.setFont(QFont("Consolas", 10))
        self._yaml_editor.setMinimumHeight(250)
        self._yaml_editor.textChanged.connect(self._on_yaml_text_changed)
        lay.addWidget(self._yaml_editor)

        # Syntax error indicator
        self._yaml_error_label = QLabel("")
        self._yaml_error_label.setStyleSheet(
            f"color: {DARK_PALETTE.error}; font-size: {TYPOGRAPHY.size_small}px;"
        )
        self._yaml_error_label.setWordWrap(True)
        self._yaml_error_label.setVisible(False)
        lay.addWidget(self._yaml_error_label)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._on_copy_yaml)
        btn_row.addWidget(copy_btn)

        save_yaml_btn = QPushButton("Save YAML")
        save_yaml_btn.clicked.connect(self._on_save_yaml)
        btn_row.addWidget(save_yaml_btn)

        revert_yaml_btn = QPushButton("Revert YAML")
        revert_yaml_btn.clicked.connect(self._on_revert_yaml)
        btn_row.addWidget(revert_yaml_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)

    # ==================================================================
    # YAML editor handlers
    # ==================================================================

    def _on_yaml_text_changed(self) -> None:
        """Validate YAML on every edit."""
        import yaml

        text = self._yaml_editor.toPlainText()
        try:
            yaml.safe_load(text)
            self._yaml_error_label.setVisible(False)
        except yaml.YAMLError as exc:
            self._yaml_error_label.setText(f"YAML error: {exc}")
            self._yaml_error_label.setVisible(True)

    def _on_copy_yaml(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self._yaml_editor.toPlainText())

    def _on_save_yaml(self) -> None:
        """Parse and save YAML directly, bypassing section widgets."""
        import yaml

        text = self._yaml_editor.toPlainText()
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            QMessageBox.warning(self, "Invalid YAML", str(exc))
            return
        confirm = QMessageBox.question(
            self, "Save Raw YAML",
            "This will overwrite the configuration with the raw YAML. Continue?",
        )
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                # Write directly through config loader
                self._api._config.config = yaml.safe_load(text)  # noqa: SLF001
                self._api.save_config()
                self._dirty = False
                self._update_dirty_label()
                self._refresh_all()
                QMessageBox.information(self, "Saved", "YAML configuration saved.")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", str(exc))

    def _on_revert_yaml(self) -> None:
        self._yaml_editor.blockSignals(True)
        self._yaml_editor.setPlainText(self._api.get_config_yaml())
        self._yaml_editor.blockSignals(False)
        self._yaml_error_label.setVisible(False)

    # ==================================================================
    # Refresh / state management
    # ==================================================================

    def _refresh_all(self) -> None:
        """Reload all sections from the API."""
        self._refresh_mcp_tools()
        self._refresh_budget()
        self._refresh_privacy()
        self._refresh_system()
        self._refresh_yaml()
        self._update_dirty_label()

    def _refresh_mcp_tools(self) -> None:
        tools = self._api.get_mcp_tools()
        tool_map = {t.name: t.enabled for t in tools}
        for name, cb in self._mcp_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(tool_map.get(name, True))
            cb.blockSignals(False)

    def _refresh_budget(self) -> None:
        status = self._api.get_budget_status()
        self._update_spend_label(status)

    def _refresh_privacy(self) -> None:
        self._refresh_builtin_patterns()

    def _refresh_system(self) -> None:
        cfg = self._api._config.config  # noqa: SLF001
        log_cfg = cfg.get("logging", {})
        level = log_cfg.get("level", "INFO").upper()
        idx = self._log_level_combo.findText(level)
        if idx >= 0:
            self._log_level_combo.blockSignals(True)
            self._log_level_combo.setCurrentIndex(idx)
            self._log_level_combo.blockSignals(False)

    def _refresh_yaml(self) -> None:
        self._yaml_editor.blockSignals(True)
        self._yaml_editor.setPlainText(self._api.get_config_yaml())
        self._yaml_editor.blockSignals(False)
        self._yaml_error_label.setVisible(False)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        if self._dirty:
            self._dirty_label.setText("Unsaved changes")
            self._dirty_label.setStyleSheet(
                f"color: {DARK_PALETTE.warning}; font-weight: bold;"
            )
        else:
            self._dirty_label.setText("")

    # ==================================================================
    # Save / Revert
    # ==================================================================

    def _collect_settings(self) -> None:
        """Push widget state into the API before saving."""
        # System settings
        cfg = self._api._config.config  # noqa: SLF001
        cfg.setdefault("logging", {})["level"] = self._log_level_combo.currentText()
        cfg.setdefault("server", {})["timeout"] = self._timeout_spin.value()
        cfg["max_review_iterations"] = self._max_review_spin.value()

        # Budget settings
        budget = cfg.setdefault("savings", {}).setdefault("budget", {})
        budget["enabled"] = self._budget_enabled_cb.isChecked()
        daily_val = self._daily_limit_spin.value()
        budget["daily_limit"] = daily_val if daily_val > 0 else None
        monthly_val = self._monthly_limit_spin.value()
        budget["monthly_limit"] = monthly_val if monthly_val > 0 else None

    def _on_save_all(self) -> None:
        """Save all settings to disk."""
        if self._api.config_affects_other_nodes():
            reply = QMessageBox.warning(
                self,
                "Cell-Wide Configuration Change",
                "This configuration change will propagate to all nodes "
                "on your AuraGrid cell. Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            self._collect_settings()
            saved_to = self._api.save_config()
            self._dirty = False
            self._update_dirty_label()
            self._refresh_yaml()
            self.settings_saved.emit()
            QMessageBox.information(
                self, "Saved", f"Configuration saved to:\n{saved_to}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_revert_all(self) -> None:
        """Discard unsaved changes and reload from disk."""
        if not self._dirty:
            return
        confirm = QMessageBox.question(
            self, "Confirm", "Discard unsaved changes and reload from disk?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._api.reload_config()
            self._dirty = False
            self._refresh_all()
