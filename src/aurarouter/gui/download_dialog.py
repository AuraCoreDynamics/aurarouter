from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


# ------------------------------------------------------------------
# Background workers
# ------------------------------------------------------------------

class _SearchWorker(QObject):
    """Searches HuggingFace Hub for GGUF model repos."""

    finished = Signal(list)  # list of dicts: {id, downloads, likes}
    error = Signal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            results = []
            for model in api.list_models(
                search=self.query,
                filter="gguf",
                sort="downloads",
                limit=25,
            ):
                results.append({
                    "id": model.id,
                    "downloads": getattr(model, "downloads", 0) or 0,
                    "likes": getattr(model, "likes", 0) or 0,
                })
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class _FileListWorker(QObject):
    """Lists .gguf files in a HuggingFace repo."""

    finished = Signal(list)  # list of dicts: {filename, size}
    error = Signal(str)

    def __init__(self, repo_id: str):
        super().__init__()
        self.repo_id = repo_id

    def run(self) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            repo_info = api.repo_info(self.repo_id, files_metadata=True)
            files = []
            for sibling in repo_info.siblings or []:
                if sibling.rfilename.endswith(".gguf"):
                    files.append({
                        "filename": sibling.rfilename,
                        "size": getattr(sibling, "size", 0) or 0,
                    })
            self.finished.emit(files)
        except Exception as exc:
            self.error.emit(str(exc))


class _DownloadWorker(QObject):
    """Runs download_model() off the main thread."""

    finished = Signal(str)   # path to downloaded file
    error = Signal(str)
    progress = Signal(float, float)  # (downloaded_bytes, total_bytes)

    def __init__(self, repo: str, filename: str):
        super().__init__()
        self.repo = repo
        self.filename = filename

    def run(self) -> None:
        try:
            from aurarouter.models.downloader import download_model

            path = download_model(
                repo=self.repo,
                filename=self.filename,
                progress_callback=lambda downloaded, total: self.progress.emit(downloaded, total),
            )
            self.finished.emit(str(path))
        except Exception as exc:
            self.error.emit(str(exc))


# ------------------------------------------------------------------
# Download dialog
# ------------------------------------------------------------------

class DownloadDialog(QDialog):
    """Dialog for searching and downloading GGUF models from HuggingFace."""

    download_complete = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[QObject] = None
        self._selected_repo: Optional[str] = None

        self.setWindowTitle("Download Model from HuggingFace")
        self.setMinimumSize(650, 500)

        layout = QVBoxLayout(self)

        # --- Search bar ---
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("e.g. qwen2.5 coder 7b")
        self._search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self._search_input)
        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(self._on_search)
        search_row.addWidget(self._search_btn)
        layout.addLayout(search_row)

        # --- Repo results table ---
        layout.addWidget(QLabel("Repositories:"))
        self._repo_table = QTableWidget(0, 3)
        self._repo_table.setHorizontalHeaderLabels(["Repository", "Downloads", "Likes"])
        self._repo_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._repo_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._repo_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._repo_table.currentCellChanged.connect(self._on_repo_selected)
        layout.addWidget(self._repo_table)

        # --- File list table ---
        layout.addWidget(QLabel("GGUF files in selected repo:"))
        self._file_table = QTableWidget(0, 2)
        self._file_table.setHorizontalHeaderLabels(["Filename", "Size"])
        self._file_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._file_table)

        # --- Progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # --- Status ---
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._download_btn = QPushButton("Download")
        self._download_btn.setEnabled(False)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search(self) -> None:
        query = self._search_input.text().strip()
        if not query:
            return

        self._search_btn.setEnabled(False)
        self._repo_table.setRowCount(0)
        self._file_table.setRowCount(0)
        self._download_btn.setEnabled(False)
        self._selected_repo = None
        self._status_label.setText("Searching HuggingFace...")
        self._status_label.setStyleSheet("")

        self._start_worker(_SearchWorker(query), self._on_search_finished, self._on_search_error)

    def _on_search_finished(self, results: list) -> None:
        self._search_btn.setEnabled(True)
        self._repo_table.setRowCount(0)

        if not results:
            self._status_label.setText("No GGUF repositories found.")
            return

        for r in results:
            row = self._repo_table.rowCount()
            self._repo_table.insertRow(row)
            self._repo_table.setItem(row, 0, QTableWidgetItem(r["id"]))
            self._repo_table.setItem(row, 1, QTableWidgetItem(f"{r['downloads']:,}"))
            self._repo_table.setItem(row, 2, QTableWidgetItem(f"{r['likes']:,}"))

        self._status_label.setText(f"Found {len(results)} repositories. Select one to see available files.")

    def _on_search_error(self, message: str) -> None:
        self._search_btn.setEnabled(True)
        self._status_label.setText(f"Search error: {message}")
        self._status_label.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # Repo selection -> file listing
    # ------------------------------------------------------------------

    def _on_repo_selected(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row < 0:
            return
        item = self._repo_table.item(row, 0)
        if not item:
            return

        repo_id = item.text()
        if repo_id == self._selected_repo:
            return

        self._selected_repo = repo_id
        self._file_table.setRowCount(0)
        self._download_btn.setEnabled(False)
        self._status_label.setText(f"Loading files from {repo_id}...")
        self._status_label.setStyleSheet("")

        self._start_worker(_FileListWorker(repo_id), self._on_files_loaded, self._on_files_error)

    def _on_files_loaded(self, files: list) -> None:
        self._file_table.setRowCount(0)

        if not files:
            self._status_label.setText("No .gguf files found in this repository.")
            return

        for f in files:
            row = self._file_table.rowCount()
            self._file_table.insertRow(row)
            self._file_table.setItem(row, 0, QTableWidgetItem(f["filename"]))
            size_bytes = f["size"]
            if size_bytes > 0:
                size_gb = size_bytes / (1024 * 1024 * 1024)
                self._file_table.setItem(row, 1, QTableWidgetItem(f"{size_gb:.1f} GB"))
            else:
                self._file_table.setItem(row, 1, QTableWidgetItem("--"))

        self._download_btn.setEnabled(True)
        self._status_label.setText(f"{len(files)} GGUF file(s) available. Select one and click Download.")

    def _on_files_error(self, message: str) -> None:
        self._status_label.setText(f"Error loading files: {message}")
        self._status_label.setStyleSheet("color: red;")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _on_download(self) -> None:
        file_row = self._file_table.currentRow()
        if file_row < 0:
            QMessageBox.warning(self, "No File Selected", "Select a GGUF file to download.")
            return
        if not self._selected_repo:
            return

        filename = self._file_table.item(file_row, 0).text()

        self._download_btn.setEnabled(False)
        self._search_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # indeterminate until first progress
        self._status_label.setText(f"Downloading {filename}...")
        self._status_label.setStyleSheet("")

        worker = _DownloadWorker(repo=self._selected_repo, filename=filename)
        worker.progress.connect(self._on_download_progress)
        self._start_worker(worker, self._on_download_finished, self._on_download_error)

    def _on_download_progress(self, downloaded: float, total: float) -> None:
        if total > 0:
            # Use MB scale for QProgressBar to avoid 32-bit int overflow
            total_mb = int(total / (1024 * 1024))
            dl_mb = int(downloaded / (1024 * 1024))
            self._progress_bar.setRange(0, max(total_mb, 1))
            self._progress_bar.setValue(dl_mb)
            pct = int(downloaded * 100 / total)
            self._status_label.setText(f"Downloading... {dl_mb} / {total_mb} MB ({pct}%)")

    def _on_download_finished(self, path: str) -> None:
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Downloaded to: {path}")
        self._status_label.setStyleSheet("color: green; font-weight: bold;")
        self.download_complete.emit()
        QMessageBox.information(
            self,
            "Download Complete",
            f"Model downloaded successfully:\n{path}",
        )
        self.accept()

    def _on_download_error(self, message: str) -> None:
        self._progress_bar.setVisible(False)
        self._download_btn.setEnabled(True)
        self._search_btn.setEnabled(True)
        self._status_label.setText(f"Download error: {message}")
        self._status_label.setStyleSheet("color: red; font-weight: bold;")

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _start_worker(self, worker: QObject, on_finished, on_error) -> None:
        """Start a worker in a background thread, cleaning up any previous one."""
        # Wait for any previous thread to finish before starting a new one
        if self._thread is not None:
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(3000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

        self._worker = worker
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_finished)
        self._worker.error.connect(on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(3000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def closeEvent(self, event) -> None:
        self._cleanup_thread()
        event.accept()
