from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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


class ConfigPanel(QWidget):
    """Configuration management panel with models, routing, and YAML preview."""

    config_saved = Signal()  # emitted after a successful save

    def __init__(self, config: ConfigLoader, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._dirty = False

        layout = QVBoxLayout(self)

        splitter = QSplitter()

        # ---- Left side: Models + Routing ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_models_section())
        left_layout.addWidget(self._build_routing_section())

        splitter.addWidget(left)

        # ---- Right side: YAML preview + Local Models ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_yaml_section())
        right_layout.addWidget(self._build_local_models_section())
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
    # Section builders
    # ==================================================================

    def _build_models_section(self) -> QGroupBox:
        group = QGroupBox("Models")
        layout = QVBoxLayout(group)

        self._models_table = QTableWidget(0, 3)
        self._models_table.setHorizontalHeaderLabels(["Model ID", "Provider", "Endpoint / Model"])
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
        group = QGroupBox("Routing (Role Chains)")
        layout = QVBoxLayout(group)

        self._roles_table = QTableWidget(0, 2)
        self._roles_table.setHorizontalHeaderLabels(["Role", "Model Chain"])
        self._roles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._roles_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._roles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._roles_table)

        # Edit row: role name + model selector + buttons
        edit_row = QHBoxLayout()

        edit_row.addWidget(QLabel("Role:"))
        self._role_name_input = QLineEdit()
        self._role_name_input.setPlaceholderText("e.g. router, reasoning, coding")
        self._role_name_input.setMaximumWidth(150)
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

    def _build_local_models_section(self) -> QGroupBox:
        group = QGroupBox("Local Models")
        layout = QVBoxLayout(group)

        self._local_models_table = QTableWidget(0, 3)
        self._local_models_table.setHorizontalHeaderLabels(["Filename", "Size (MB)", "Repo"])
        self._local_models_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._local_models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._local_models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._local_models_table.setMaximumHeight(150)
        layout.addWidget(self._local_models_table)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_local_models)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return group

    # ==================================================================
    # Refresh / populate
    # ==================================================================

    def _refresh_all(self) -> None:
        self._refresh_models_table()
        self._refresh_roles_table()
        self._refresh_model_combo()
        self._refresh_yaml_preview()
        self._refresh_local_models()
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

    def _refresh_roles_table(self) -> None:
        self._roles_table.setRowCount(0)
        for role in self._config.get_all_roles():
            chain = self._config.get_role_chain(role)
            row = self._roles_table.rowCount()
            self._roles_table.insertRow(row)
            self._roles_table.setItem(row, 0, QTableWidgetItem(role))
            self._roles_table.setItem(row, 1, QTableWidgetItem(" -> ".join(chain)))

    def _refresh_model_combo(self) -> None:
        self._role_model_combo.clear()
        self._role_model_combo.addItems(self._config.get_all_model_ids())

    def _refresh_yaml_preview(self) -> None:
        self._yaml_preview.setPlainText(self._config.to_yaml())

    def _refresh_local_models(self) -> None:
        from aurarouter.models.file_storage import FileModelStorage

        self._local_models_table.setRowCount(0)
        try:
            storage = FileModelStorage()
            storage.scan()
            for m in storage.list_models():
                row = self._local_models_table.rowCount()
                self._local_models_table.insertRow(row)
                self._local_models_table.setItem(row, 0, QTableWidgetItem(m["filename"]))
                size_mb = m.get("size_bytes", 0) / (1024 * 1024)
                self._local_models_table.setItem(row, 1, QTableWidgetItem(f"{size_mb:.0f}"))
                self._local_models_table.setItem(row, 2, QTableWidgetItem(m.get("repo", "unknown")))
        except Exception:
            pass  # graceful — local models section is informational

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

    def _on_append_to_chain(self) -> None:
        role = self._role_name_input.text().strip()
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
        try:
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
            path = self._config.config_path
            if path and path.is_file():
                reloaded = ConfigLoader(config_path=str(path))
                self._config.config = reloaded.config
            self._dirty = False
            self._refresh_all()
