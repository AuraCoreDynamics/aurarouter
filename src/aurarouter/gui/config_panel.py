from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aurarouter.config import ConfigLoader
from aurarouter.gui.model_dialog import ModelDialog

if TYPE_CHECKING:
    from aurarouter.gui.environment import EnvironmentContext


class ConfigPanel(QWidget):
    """Configuration management panel with models, routing, and YAML preview."""

    config_saved = Signal()  # emitted after a successful save

    def __init__(
        self,
        config: Optional[ConfigLoader] = None,
        *,
        context: Optional[EnvironmentContext] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._context: Optional[EnvironmentContext] = context
        self._config: ConfigLoader = (
            context.get_config_loader() if context is not None else config  # type: ignore[assignment]
        )
        self._dirty = False

        layout = QVBoxLayout(self)

        # ---- Warning banner (shown for environments that propagate config) ----
        self._warning_banner = QLabel()
        self._warning_banner.setStyleSheet(
            "background-color: #fff3cd; color: #856404; padding: 6px; "
            "border: 1px solid #ffc107; border-radius: 3px; font-weight: bold;"
        )
        self._warning_banner.setWordWrap(True)
        self._warning_banner.setVisible(False)
        layout.addWidget(self._warning_banner)
        self._update_warning_banner()

        splitter = QSplitter()

        # ---- Left side: Models + Routing ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_models_section())
        left_layout.addWidget(self._build_routing_section())
        left_layout.addWidget(self._build_mcp_tools_section())

        splitter.addWidget(left)

        # ---- Right side: YAML preview ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_yaml_section())
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # ---- Save / Revert toolbar ----
        toolbar = QHBoxLayout()
        self._dirty_label = QLabel("")
        toolbar.addWidget(self._dirty_label)
        toolbar.addStretch()

        revert_btn = QPushButton("Revert")
        revert_btn.clicked.connect(self._on_revert)
        toolbar.addWidget(revert_btn)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("font-weight: bold;")
        save_btn.clicked.connect(self._on_save)
        toolbar.addWidget(save_btn)

        layout.addLayout(toolbar)

        # Initial population
        self._refresh_all()

    # ==================================================================
    # Context management
    # ==================================================================

    def set_context(self, context: EnvironmentContext) -> None:
        """Switch to a new environment context (e.g. after environment change)."""
        self._context = context
        self._config = context.get_config_loader()
        self._dirty = False
        self._update_warning_banner()
        self._refresh_all()

    def _update_warning_banner(self) -> None:
        if self._context is not None:
            warnings = self._context.get_config_warnings()
            if warnings:
                self._warning_banner.setText("\n".join(warnings))
                self._warning_banner.setVisible(True)
                return
        self._warning_banner.setVisible(False)

    # ==================================================================
    # Section builders
    # ==================================================================

    def _build_models_section(self) -> QGroupBox:
        group = QGroupBox("Models")
        layout = QVBoxLayout(group)

        self._models_table = QTableWidget(0, 4)
        self._models_table.setHorizontalHeaderLabels(["Model ID", "Provider", "Endpoint / Model", "Tags"])
        self._models_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._models_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_model)
        btn_row.addWidget(add_btn)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._on_edit_model)
        btn_row.addWidget(edit_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_model)
        btn_row.addWidget(remove_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        return group

    def _build_routing_section(self) -> QGroupBox:
        group = QGroupBox("Routing (Fallback Chains)")
        layout = QVBoxLayout(group)

        # Explanation
        explain = QLabel(
            "Each role has a priority-ordered fallback chain. "
            "The router tries the first model; if it fails, it falls back to "
            "the next. Only one model handles each request."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color: #616161; font-style: italic; font-size: 11px;")
        layout.addWidget(explain)

        self._roles_table = QTableWidget(0, 2)
        self._roles_table.setHorizontalHeaderLabels(
            ["Role", "Fallback Order (first = highest priority)"]
        )
        self._roles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._roles_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._roles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._roles_table)

        # Missing required roles warning
        self._missing_roles_label = QLabel()
        self._missing_roles_label.setStyleSheet(
            "color: #d32f2f; font-weight: bold; font-size: 11px;"
        )
        self._missing_roles_label.setWordWrap(True)
        self._missing_roles_label.setVisible(False)
        layout.addWidget(self._missing_roles_label)

        # Edit row: role name (editable combo) + model selector + buttons
        edit_row = QHBoxLayout()

        edit_row.addWidget(QLabel("Role:"))
        self._role_name_input = QComboBox()
        self._role_name_input.setEditable(True)
        self._role_name_input.setMaximumWidth(150)
        from aurarouter.semantic_verbs import get_known_roles
        self._role_name_input.addItems(get_known_roles())
        self._role_name_input.setCurrentText("")
        edit_row.addWidget(self._role_name_input)

        edit_row.addWidget(QLabel("Add model:"))
        self._role_model_combo = QComboBox()
        self._role_model_combo.setMinimumWidth(150)
        edit_row.addWidget(self._role_model_combo)

        add_to_chain_btn = QPushButton("Append")
        add_to_chain_btn.clicked.connect(self._on_append_to_chain)
        edit_row.addWidget(add_to_chain_btn)

        edit_row.addStretch()

        up_btn = QPushButton("Up")
        up_btn.setMaximumWidth(40)
        up_btn.clicked.connect(self._on_move_up)
        edit_row.addWidget(up_btn)

        down_btn = QPushButton("Down")
        down_btn.setMaximumWidth(50)
        down_btn.clicked.connect(self._on_move_down)
        edit_row.addWidget(down_btn)

        remove_from_chain_btn = QPushButton("Remove from Chain")
        remove_from_chain_btn.clicked.connect(self._on_remove_from_chain)
        edit_row.addWidget(remove_from_chain_btn)

        remove_role_btn = QPushButton("Delete Role")
        remove_role_btn.clicked.connect(self._on_remove_role)
        edit_row.addWidget(remove_role_btn)

        layout.addLayout(edit_row)

        return group

    def _build_mcp_tools_section(self) -> QGroupBox:
        """Build the MCP Tools toggle section."""
        group = QGroupBox("MCP Tools (Exposed to Host Models)")
        layout = QVBoxLayout(group)

        # (config_key, label, description, default_enabled)
        self._mcp_tool_definitions = [
            ("route_task", "Route Task",
             "General-purpose router for local/cloud models with fallback", True),
            ("local_inference", "Local Inference",
             "Privacy-preserving execution on local models only", True),
            ("generate_code", "Generate Code",
             "Multi-step code generation with planning", True),
            ("compare_models", "Compare Models",
             "Run prompt across multiple models for comparison", False),
        ]

        self._mcp_tool_checkboxes: dict[str, QCheckBox] = {}

        for key, label, description, default in self._mcp_tool_definitions:
            cb = QCheckBox(f"{label} \u2014 {description}")
            cb.setChecked(self._config.is_mcp_tool_enabled(key, default=default))
            cb.stateChanged.connect(
                lambda state, k=key: self._on_mcp_tool_toggled(k, state)
            )
            self._mcp_tool_checkboxes[key] = cb
            layout.addWidget(cb)

        note = QLabel("Changes take effect after saving and restarting the MCP server.")
        note.setStyleSheet("color: gray; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        return group

    def _on_mcp_tool_toggled(self, tool_name: str, state: int) -> None:
        """Handle MCP tool checkbox toggle."""
        from PySide6.QtCore import Qt

        enabled = state == Qt.CheckState.Checked.value
        self._config.set_mcp_tool_enabled(tool_name, enabled)
        self._mark_dirty()

    def _build_yaml_section(self) -> QGroupBox:
        group = QGroupBox("YAML Preview")
        layout = QVBoxLayout(group)

        self._yaml_preview = QTextEdit()
        self._yaml_preview.setReadOnly(True)
        self._yaml_preview.setFont(QFont("Consolas", 9))
        layout.addWidget(self._yaml_preview)

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self._on_copy_yaml)
        layout.addWidget(copy_btn)

        return group

    # ==================================================================
    # Refresh / populate
    # ==================================================================

    def _refresh_all(self) -> None:
        self._refresh_models_table()
        self._refresh_roles_table()
        self._refresh_model_combo()
        self._refresh_mcp_tools()
        self._refresh_yaml_preview()
        self._update_dirty_label()

    def _refresh_models_table(self) -> None:
        self._models_table.setRowCount(0)
        for model_id in self._config.get_all_model_ids():
            cfg = self._config.get_model_config(model_id)
            row = self._models_table.rowCount()
            self._models_table.insertRow(row)
            self._models_table.setItem(row, 0, QTableWidgetItem(model_id))
            self._models_table.setItem(row, 1, QTableWidgetItem(cfg.get("provider", "")))
            detail = cfg.get("endpoint", cfg.get("model_name", cfg.get("model_path", "")))
            self._models_table.setItem(row, 2, QTableWidgetItem(str(detail)))
            tags = cfg.get("tags", [])
            self._models_table.setItem(row, 3, QTableWidgetItem(", ".join(tags) if tags else ""))

    def _refresh_roles_table(self) -> None:
        self._roles_table.setRowCount(0)
        for role in self._config.get_all_roles():
            chain = self._config.get_role_chain(role)
            row = self._roles_table.rowCount()
            self._roles_table.insertRow(row)
            self._roles_table.setItem(row, 0, QTableWidgetItem(role))
            self._roles_table.setItem(row, 1, QTableWidgetItem(" > ".join(chain)))
        self._update_missing_roles_warning()

    def _refresh_model_combo(self) -> None:
        self._role_model_combo.clear()
        self._role_model_combo.addItems(self._config.get_all_model_ids())

    def _refresh_mcp_tools(self) -> None:
        """Sync MCP tool checkboxes with current config."""
        for key, _label, _desc, default in self._mcp_tool_definitions:
            cb = self._mcp_tool_checkboxes.get(key)
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(self._config.is_mcp_tool_enabled(key, default=default))
                cb.blockSignals(False)

    def _refresh_yaml_preview(self) -> None:
        self._yaml_preview.setPlainText(self._config.to_yaml())

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_dirty_label()
        self._refresh_yaml_preview()

    def _update_dirty_label(self) -> None:
        if self._dirty:
            self._dirty_label.setText("Unsaved changes")
            self._dirty_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self._dirty_label.setText("")

    # ==================================================================
    # Model CRUD
    # ==================================================================

    def _on_add_model(self) -> None:
        dlg = ModelDialog(parent=self)
        if dlg.exec() == ModelDialog.DialogCode.Accepted:
            model_id = dlg.get_model_id()
            if model_id in self._config.get_all_model_ids():
                QMessageBox.warning(self, "Duplicate", f"Model '{model_id}' already exists.")
                return
            self._config.set_model(model_id, dlg.get_model_config())
            self._refresh_models_table()
            self._refresh_model_combo()
            self._mark_dirty()

    def _on_edit_model(self) -> None:
        row = self._models_table.currentRow()
        if row < 0:
            return
        model_id = self._models_table.item(row, 0).text()
        model_cfg = self._config.get_model_config(model_id)
        dlg = ModelDialog(parent=self, model_id=model_id, model_config=model_cfg)
        if dlg.exec() == ModelDialog.DialogCode.Accepted:
            self._config.set_model(model_id, dlg.get_model_config())
            self._refresh_models_table()
            self._mark_dirty()

    def _on_remove_model(self) -> None:
        row = self._models_table.currentRow()
        if row < 0:
            return
        model_id = self._models_table.item(row, 0).text()
        confirm = QMessageBox.question(
            self, "Confirm", f"Remove model '{model_id}'?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._config.remove_model(model_id)
            self._refresh_models_table()
            self._refresh_model_combo()
            self._mark_dirty()

    # ==================================================================
    # Routing management
    # ==================================================================

    def _get_selected_role(self) -> Optional[str]:
        row = self._roles_table.currentRow()
        if row < 0:
            return None
        return self._roles_table.item(row, 0).text()

    def _update_missing_roles_warning(self) -> None:
        """Show a warning if required roles (router, reasoning, coding) are missing."""
        from aurarouter.semantic_verbs import get_required_roles

        configured = set(self._config.get_all_roles())
        missing = [r for r in get_required_roles() if r not in configured]
        if missing:
            self._missing_roles_label.setText(
                "Missing required roles: " + ", ".join(missing)
            )
            self._missing_roles_label.setVisible(True)
        else:
            self._missing_roles_label.setVisible(False)

    def _on_append_to_chain(self) -> None:
        role = self._role_name_input.currentText().strip()
        model_id = self._role_model_combo.currentText()
        if not role:
            # Try to use the selected role from the table
            role = self._get_selected_role() or ""
        if not role or not model_id:
            return
        chain = self._config.get_role_chain(role)
        chain.append(model_id)
        self._config.set_role_chain(role, chain)
        self._refresh_roles_table()
        self._mark_dirty()

    def _on_move_up(self) -> None:
        role = self._get_selected_role()
        if not role:
            return
        chain = self._config.get_role_chain(role)
        # We need to know which model in the chain to move — use a simple approach:
        # Move the last item up (swap with previous). For a more advanced UI, we'd
        # need a sub-selection within the chain. For now, prompt for index.
        if len(chain) < 2:
            return
        # Move the selected role's chain: rotate last element to second-to-last
        # Actually, let's just swap the last two for simplicity in this table-based UI
        # A more sophisticated approach would use a QListWidget for the chain
        chain[-1], chain[-2] = chain[-2], chain[-1]
        self._config.set_role_chain(role, chain)
        self._refresh_roles_table()
        self._mark_dirty()

    def _on_move_down(self) -> None:
        role = self._get_selected_role()
        if not role:
            return
        chain = self._config.get_role_chain(role)
        if len(chain) < 2:
            return
        chain[0], chain[1] = chain[1], chain[0]
        self._config.set_role_chain(role, chain)
        self._refresh_roles_table()
        self._mark_dirty()

    def _on_remove_from_chain(self) -> None:
        role = self._get_selected_role()
        if not role:
            return
        chain = self._config.get_role_chain(role)
        if not chain:
            return
        # Remove last model from chain
        chain.pop()
        self._config.set_role_chain(role, chain)
        self._refresh_roles_table()
        self._mark_dirty()

    def _on_remove_role(self) -> None:
        role = self._get_selected_role()
        if not role:
            return
        confirm = QMessageBox.question(
            self, "Confirm", f"Delete role '{role}'?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._config.remove_role(role)
            self._refresh_roles_table()
            self._mark_dirty()

    # ==================================================================
    # YAML preview
    # ==================================================================

    def _on_copy_yaml(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self._yaml_preview.toPlainText())

    # ==================================================================
    # Save / Revert
    # ==================================================================

    def _on_save(self) -> None:
        # Warn about cell-wide propagation when applicable.
        if self._context is not None and self._context.config_affects_other_nodes():
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
            if self._context is not None:
                saved_to = self._context.save_config()
            else:
                saved_to = self._config.save()
            self._dirty = False
            self._update_dirty_label()
            self.config_saved.emit()
            QMessageBox.information(
                self, "Saved", f"Configuration saved to:\n{saved_to}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_revert(self) -> None:
        if not self._dirty:
            return
        confirm = QMessageBox.question(
            self, "Confirm", "Discard unsaved changes and reload from disk?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            if self._context is not None:
                self._config = self._context.reload_config()
            else:
                path = self._config.config_path
                if path and path.is_file():
                    reloaded = ConfigLoader(config_path=str(path))
                    self._config.config = reloaded.config
            self._dirty = False
            self._refresh_all()
