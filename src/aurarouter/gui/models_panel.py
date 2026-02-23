from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from aurarouter.gui.environment import EnvironmentContext


# ------------------------------------------------------------------
# Grid refresh worker (async -> sync bridge)
# ------------------------------------------------------------------

class _GridRefreshWorker(QObject):
    """Fetches model IDs from GridModelStorage in a background thread."""

    finished = Signal(list)  # list of model ID strings
    error = Signal(str)

    def __init__(self, grid_storage):
        super().__init__()
        self._grid_storage = grid_storage

    def run(self) -> None:
        try:
            model_ids = asyncio.run(self._grid_storage.list_models())
            self.finished.emit(model_ids)
        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# Models panel
# ------------------------------------------------------------------

class ModelsPanel(QWidget):
    """Dedicated panel for managing local and grid model files."""

    def __init__(
        self,
        *,
        context: Optional[EnvironmentContext] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._context: Optional[EnvironmentContext] = context
        self._grid_thread: Optional[QThread] = None
        self._grid_worker: Optional[_GridRefreshWorker] = None

        layout = QVBoxLayout(self)

        # ---- Local Models section ----
        layout.addWidget(self._build_local_section())

        # ---- Grid Models section (conditional) ----
        self._grid_group = self._build_grid_section()
        layout.addWidget(self._grid_group)

        # Show grid section only when the context reports remote models.
        self._grid_available = self._has_remote_models()
        self._grid_group.setVisible(self._grid_available)

        # Initial population
        self._refresh_local_models()

    # ==================================================================
    # Context management
    # ==================================================================

    def set_context(self, context: EnvironmentContext) -> None:
        """Switch to a new environment context (e.g. after environment change)."""
        self._context = context
        self._grid_available = self._has_remote_models()
        self._grid_group.setVisible(self._grid_available)
        self._refresh_local_models()
        if self._grid_available:
            self._refresh_grid_models()

    def _has_remote_models(self) -> bool:
        """Check if the current context supports remote model listing."""
        if self._context is None:
            # Fallback to import-based detection for backward compatibility.
            try:
                from aurarouter.auragrid.model_storage import GridModelStorage  # noqa: F401
                return True
            except ImportError:
                return False
        # Context-based: if the context returns remote models, show the section.
        # We also check the environment name as a hint (avoids calling list_remote_models
        # which may be expensive).
        return self._context.name != "Local"

    # ==================================================================
    # Section builders
    # ==================================================================

    def _build_local_section(self) -> QGroupBox:
        group = QGroupBox("Local Models")
        layout = QVBoxLayout(group)

        self._local_table = QTableWidget(0, 4)
        self._local_table.setHorizontalHeaderLabels(
            ["Filename", "Size (MB)", "Repo", "Downloaded"]
        )
        self._local_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._local_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._local_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        layout.addWidget(self._local_table)

        # Action buttons
        btn_row = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_local_models)
        btn_row.addWidget(refresh_btn)

        download_btn = QPushButton("Download from HuggingFace...")
        download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(download_btn)

        import_btn = QPushButton("Import Local File...")
        import_btn.clicked.connect(self._on_import_local)
        btn_row.addWidget(import_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(remove_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Storage info
        self._storage_info = QLabel("")
        self._storage_info.setStyleSheet("color: gray;")
        layout.addWidget(self._storage_info)

        return group

    def _build_grid_section(self) -> QGroupBox:
        group = QGroupBox("Grid Models (AuraGrid)")
        layout = QVBoxLayout(group)

        self._grid_table = QTableWidget(0, 1)
        self._grid_table.setHorizontalHeaderLabels(["Model ID"])
        self._grid_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._grid_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._grid_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        layout.addWidget(self._grid_table)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Grid")
        refresh_btn.clicked.connect(self._refresh_grid_models)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return group

    # ==================================================================
    # Local models
    # ==================================================================

    def _refresh_local_models(self) -> None:
        self._local_table.setRowCount(0)
        try:
            if self._context is not None:
                models = self._context.list_local_models()
            else:
                from aurarouter.models.file_storage import FileModelStorage
                storage = FileModelStorage()
                storage.scan()
                models = storage.list_models()

            for m in models:
                row = self._local_table.rowCount()
                self._local_table.insertRow(row)
                self._local_table.setItem(
                    row, 0, QTableWidgetItem(m["filename"])
                )
                size_mb = m.get("size_bytes", 0) / (1024 * 1024)
                self._local_table.setItem(
                    row, 1, QTableWidgetItem(f"{size_mb:.0f}")
                )
                self._local_table.setItem(
                    row, 2, QTableWidgetItem(m.get("repo", "unknown"))
                )
                downloaded = m.get("downloaded_at", "")
                if downloaded:
                    # Show just the date portion
                    downloaded = downloaded[:10]
                self._local_table.setItem(
                    row, 3, QTableWidgetItem(downloaded)
                )

            self._update_storage_info(models)
        except Exception:
            self._storage_info.setText("Could not read model storage.")

    def _update_storage_info(self, models: list[dict]) -> None:
        total_bytes = sum(m.get("size_bytes", 0) for m in models)
        total_gb = total_bytes / (1024 * 1024 * 1024)
        count = len(models)

        if self._context is not None:
            info = self._context.get_storage_info()
            path = info.get("path", "unknown")
        else:
            path = "~/.auracore/models"

        self._storage_info.setText(
            f"Storage: {path}  "
            f"({count} model{'s' if count != 1 else ''}, {total_gb:.1f} GB total)"
        )

    def _on_download(self) -> None:
        from aurarouter.gui.download_dialog import DownloadDialog

        dlg = DownloadDialog(parent=self)
        dlg.download_complete.connect(self._refresh_local_models)
        dlg.exec()

    def _on_import_local(self) -> None:
        """Import a GGUF model file from a local path."""
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model File",
            str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)",
        )
        if not path:
            return

        path_obj = Path(path)
        if not path_obj.is_file():
            QMessageBox.warning(self, "Invalid File", f"File not found: {path}")
            return

        size_bytes = path_obj.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        reply = QMessageBox.question(
            self,
            "Import Model",
            f"Register '{path_obj.name}' ({size_mb:.0f} MB) in the model registry?\n\n"
            f"The file will remain at its current location:\n{path}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from aurarouter.models.file_storage import FileModelStorage

        storage = FileModelStorage()
        storage.register(
            repo="local-import",
            filename=path_obj.name,
            path=path_obj,
        )
        self._refresh_local_models()

    def _on_remove(self) -> None:
        row = self._local_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No Selection", "Select a model to remove.")
            return

        filename = self._local_table.item(row, 0).text()
        confirm = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove '{filename}' and delete the file from disk?",
        )
        if confirm == QMessageBox.StandardButton.Yes:
            if self._context is not None:
                self._context.remove_model(filename, delete_file=True)
            else:
                from aurarouter.models.file_storage import FileModelStorage
                storage = FileModelStorage()
                storage.remove(filename, delete_file=True)
            self._refresh_local_models()

    # ==================================================================
    # Grid models
    # ==================================================================

    def _refresh_grid_models(self) -> None:
        if not self._grid_available:
            return

        try:
            from aurarouter.auragrid.model_storage import GridModelStorage

            grid_storage = GridModelStorage()

            self._grid_worker = _GridRefreshWorker(grid_storage)
            self._grid_thread = QThread()
            self._grid_worker.moveToThread(self._grid_thread)

            self._grid_thread.started.connect(self._grid_worker.run)
            self._grid_worker.finished.connect(self._on_grid_models_loaded)
            self._grid_worker.error.connect(self._on_grid_error)
            self._grid_worker.finished.connect(self._grid_thread.quit)
            self._grid_worker.error.connect(self._grid_thread.quit)
            self._grid_thread.finished.connect(self._cleanup_grid_thread)

            self._grid_thread.start()
        except Exception:
            pass

    def _on_grid_models_loaded(self, model_ids: list) -> None:
        self._grid_table.setRowCount(0)
        for model_id in model_ids:
            row = self._grid_table.rowCount()
            self._grid_table.insertRow(row)
            self._grid_table.setItem(row, 0, QTableWidgetItem(model_id))

    def _on_grid_error(self, message: str) -> None:
        self._grid_table.setRowCount(0)
        self._grid_table.insertRow(0)
        self._grid_table.setItem(
            0, 0, QTableWidgetItem(f"Error: {message}")
        )

    def _cleanup_grid_thread(self) -> None:
        if self._grid_thread:
            self._grid_thread.deleteLater()
            self._grid_thread = None
        if self._grid_worker:
            self._grid_worker.deleteLater()
            self._grid_worker = None
