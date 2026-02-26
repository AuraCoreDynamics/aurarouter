from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

PROVIDERS = ["ollama", "google", "claude", "llamacpp-server", "llamacpp", "openapi"]

# Fields that appear for each provider type
_PROVIDER_FIELDS: dict[str, list[str]] = {
    "ollama": ["endpoint", "model_name"],
    "google": ["model_name", "api_key", "env_key"],
    "claude": ["model_name", "api_key", "env_key"],
    "llamacpp-server": ["endpoint"],
    "llamacpp": ["model_path"],
    "openapi": ["endpoint", "model_name", "api_key", "env_key"],
}

_FIELD_DEFAULTS: dict[str, str] = {
    "endpoint": "http://localhost:11434/api/generate",
    "model_name": "",
    "api_key": "",
    "env_key": "",
    "model_path": "",
}

_LLAMACPP_SERVER_ENDPOINT_DEFAULT = "http://localhost:8080"
_OPENAPI_ENDPOINT_DEFAULT = "http://localhost:8000/v1"


# ------------------------------------------------------------------
# Background connection test worker
# ------------------------------------------------------------------

class _ConnectionTestWorker(QObject):
    finished = Signal(bool, str)  # (success, message)

    def __init__(self, provider: str, config: dict):
        super().__init__()
        self.provider = provider
        self.config = config

    def run(self) -> None:
        try:
            if self.provider == "ollama":
                self._test_ollama()
            elif self.provider == "google":
                self._test_google()
            elif self.provider == "claude":
                self._test_claude()
            elif self.provider == "llamacpp-server":
                self._test_llamacpp_server()
            elif self.provider == "llamacpp":
                self._test_llamacpp()
            elif self.provider == "openapi":
                self._test_openapi()
            else:
                self.finished.emit(False, f"Unknown provider: {self.provider}")
        except Exception as exc:
            self.finished.emit(False, str(exc))

    def _test_ollama(self) -> None:
        import httpx

        endpoint = self.config.get("endpoint", "http://localhost:11434/api/generate")
        base = endpoint.split("/api/")[0] if "/api/" in endpoint else endpoint.rstrip("/")
        url = base + "/api/tags"
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model_name = self.config.get("model_name", "")
        if model_name and model_name not in models:
            self.finished.emit(
                False,
                f"Server reachable but model '{model_name}' not found.\n"
                f"Available: {', '.join(models[:10])}",
            )
        else:
            self.finished.emit(True, f"Connected. {len(models)} model(s) available.")

    def _test_google(self) -> None:
        import os
        key = self.config.get("api_key", "")
        if not key or "YOUR_" in key:
            env_key = self.config.get("env_key", "GOOGLE_API_KEY")
            key = os.environ.get(env_key, "")
        if not key:
            self.finished.emit(False, "No API key configured.")
            return
        from google import genai
        client = genai.Client(api_key=key)
        models = list(client.models.list())
        self.finished.emit(True, f"Authenticated. {len(models)} model(s) available.")

    def _test_claude(self) -> None:
        import os
        key = self.config.get("api_key", "")
        if not key or "YOUR_" in key:
            env_key = self.config.get("env_key", "ANTHROPIC_API_KEY")
            key = os.environ.get(env_key, "")
        if not key:
            self.finished.emit(False, "No API key configured.")
            return
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        models = client.models.list()
        self.finished.emit(True, f"Authenticated. Connection successful.")

    def _test_llamacpp_server(self) -> None:
        import httpx

        endpoint = self.config.get("endpoint", "http://localhost:8080")
        url = endpoint.rstrip("/") + "/health"
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        self.finished.emit(True, "llama-server is reachable.")

    def _test_llamacpp(self) -> None:
        model_path = self.config.get("model_path", "")
        if not model_path:
            self.finished.emit(False, "No model_path configured.")
            return
        if Path(model_path).is_file():
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            self.finished.emit(True, f"Model file exists ({size_mb:.0f} MB).")
        else:
            self.finished.emit(False, f"File not found: {model_path}")

    def _test_openapi(self) -> None:
        import httpx

        endpoint = self.config.get("endpoint", _OPENAPI_ENDPOINT_DEFAULT)
        url = endpoint.rstrip("/") + "/models"
        headers: dict[str, str] = {}
        api_key = self.config.get("api_key", "")
        if not api_key or "YOUR_" in api_key:
            import os
            env_key = self.config.get("env_key", "")
            if env_key:
                api_key = os.environ.get(env_key, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = httpx.get(url, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        model_names = [m.get("id", "") for m in models]
        model_name = self.config.get("model_name", "")
        if model_name and model_name not in model_names:
            self.finished.emit(
                False,
                f"Server reachable but model '{model_name}' not listed.\n"
                f"Available: {', '.join(model_names[:10])}",
            )
        else:
            self.finished.emit(True, f"Connected. {len(models)} model(s) available.")


# ------------------------------------------------------------------
# Background auto-tune worker
# ------------------------------------------------------------------

class _AutoTuneWorker(QObject):
    finished = Signal(bool, str, dict)  # (success, message, params)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self) -> None:
        try:
            from aurarouter.tuning import extract_gguf_metadata, recommend_llamacpp_params

            metadata = extract_gguf_metadata(self.model_path)
            params = recommend_llamacpp_params(self.model_path, metadata)

            ctx = metadata.get("context_length", 0)
            arch = metadata.get("architecture", "unknown")
            chat = "yes" if metadata.get("has_chat_template") else "no"
            msg = f"Architecture: {arch}, context: {ctx}, chat template: {chat}"
            self.finished.emit(True, msg, params)
        except Exception as exc:
            self.finished.emit(False, str(exc), {})


# ------------------------------------------------------------------
# Model edit dialog
# ------------------------------------------------------------------

class ModelDialog(QDialog):
    """Dialog for adding or editing a model definition."""

    def __init__(
        self,
        parent=None,
        model_id: str = "",
        model_config: Optional[dict] = None,
    ):
        super().__init__(parent)
        self._editing = bool(model_id)
        self._test_thread: Optional[QThread] = None
        self._test_worker: Optional[_ConnectionTestWorker] = None
        self._tune_thread: Optional[QThread] = None
        self._tune_worker: Optional[_AutoTuneWorker] = None

        self.setWindowTitle("Edit Model" if self._editing else "Add Model")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)

        # --- Form ---
        self._form = QFormLayout()

        self._id_input = QLineEdit(model_id)
        if self._editing:
            self._id_input.setReadOnly(True)
        self._form.addRow("Model ID:", self._id_input)

        self._provider_combo = QComboBox()
        self._provider_combo.addItems(PROVIDERS)
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self._form.addRow("Provider:", self._provider_combo)

        # Dynamic fields — created once, shown/hidden per provider
        self._field_inputs: dict[str, QLineEdit] = {}
        for field in ("endpoint", "model_name", "api_key", "env_key", "model_path"):
            inp = QLineEdit()
            inp.setPlaceholderText(_FIELD_DEFAULTS.get(field, ""))
            self._field_inputs[field] = inp
            if field == "model_path":
                # Add a browse button next to model_path
                mp_row = QHBoxLayout()
                mp_row.addWidget(inp)
                self._browse_btn = QPushButton("Browse Local")
                self._browse_btn.clicked.connect(self._on_browse_local_models)
                mp_row.addWidget(self._browse_btn)
                self._form.addRow(f"{field}:", mp_row)
            else:
                self._form.addRow(f"{field}:", inp)

        # Tags (comma-separated, shown for all providers)
        self._tags_input = QLineEdit()
        self._tags_input.setPlaceholderText("e.g. private, fast, coding")
        self._form.addRow("Tags:", self._tags_input)

        # Parameters (free-form YAML-ish key: value)
        self._params_input = QTextEdit()
        self._params_input.setPlaceholderText(
            "temperature: 0.1\nn_ctx: 4096\nmax_tokens: 2048"
        )
        self._params_input.setMaximumHeight(100)
        self._form.addRow("Parameters:", self._params_input)

        layout.addLayout(self._form)

        # --- Test Connection ---
        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test_connection)
        test_row.addWidget(self._test_btn)
        self._test_label = QLabel("")
        test_row.addWidget(self._test_label)
        test_row.addStretch()
        layout.addLayout(test_row)

        # --- Auto-Tune (llamacpp only) ---
        tune_row = QHBoxLayout()
        self._tune_btn = QPushButton("Auto-Tune")
        self._tune_btn.setToolTip(
            "Analyze the GGUF model and populate recommended parameters."
        )
        self._tune_btn.clicked.connect(self._on_auto_tune)
        tune_row.addWidget(self._tune_btn)
        self._tune_label = QLabel("")
        tune_row.addWidget(self._tune_label)
        tune_row.addStretch()
        layout.addLayout(tune_row)

        # --- Buttons ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Populate from existing config
        if model_config:
            self._populate(model_config)

        # Trigger initial field visibility
        self._on_provider_changed(self._provider_combo.currentText())

    # ------------------------------------------------------------------

    def _populate(self, cfg: dict) -> None:
        provider = cfg.get("provider", "ollama")
        idx = self._provider_combo.findText(provider)
        if idx >= 0:
            self._provider_combo.setCurrentIndex(idx)

        for field, inp in self._field_inputs.items():
            value = cfg.get(field, "")
            if value:
                inp.setText(str(value))

        tags = cfg.get("tags", [])
        if tags:
            self._tags_input.setText(", ".join(tags))

        params = cfg.get("parameters", {})
        if params:
            lines = [f"{k}: {v}" for k, v in params.items()]
            self._params_input.setPlainText("\n".join(lines))

    def _on_provider_changed(self, provider: str) -> None:
        visible = set(_PROVIDER_FIELDS.get(provider, []))
        for field, inp in self._field_inputs.items():
            row_visible = field in visible
            inp.setVisible(row_visible)
            # Also hide the label
            if field == "model_path":
                # model_path uses a QHBoxLayout wrapper — find label for the layout
                self._browse_btn.setVisible(row_visible)
                label = self._form.labelForField(inp.parent().layout() if inp.parent() else inp)
                if not label:
                    label = self._form.labelForField(inp)
            else:
                label = self._form.labelForField(inp)
            if label:
                label.setVisible(row_visible)

        # Show Auto-Tune button only for llamacpp provider
        is_llamacpp = provider == "llamacpp"
        self._tune_btn.setVisible(is_llamacpp)
        self._tune_label.setVisible(is_llamacpp)

        # Set a sensible default endpoint per provider
        if provider == "llamacpp-server" and not self._field_inputs["endpoint"].text():
            self._field_inputs["endpoint"].setText(_LLAMACPP_SERVER_ENDPOINT_DEFAULT)
        elif provider == "ollama" and not self._field_inputs["endpoint"].text():
            self._field_inputs["endpoint"].setText("http://localhost:11434/api/generate")
        elif provider == "openapi" and not self._field_inputs["endpoint"].text():
            self._field_inputs["endpoint"].setText(_OPENAPI_ENDPOINT_DEFAULT)

    def _on_accept(self) -> None:
        if not self._id_input.text().strip():
            QMessageBox.warning(self, "Validation", "Model ID is required.")
            return
        self.accept()

    def get_model_id(self) -> str:
        return self._id_input.text().strip()

    def get_model_config(self) -> dict:
        provider = self._provider_combo.currentText()
        cfg: dict = {"provider": provider}
        visible = set(_PROVIDER_FIELDS.get(provider, []))
        for field, inp in self._field_inputs.items():
            if field in visible and inp.text().strip():
                cfg[field] = inp.text().strip()

        # Tags
        tags_text = self._tags_input.text().strip()
        if tags_text:
            cfg["tags"] = [t.strip() for t in tags_text.split(",") if t.strip()]

        # Parse parameters
        params_text = self._params_input.toPlainText().strip()
        if params_text:
            params: dict = {}
            for line in params_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # Try to parse as number
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            if v.lower() in ("true", "false"):
                                v = v.lower() == "true"
                    params[k] = v
            if params:
                cfg["parameters"] = params

        return cfg

    # ------------------------------------------------------------------
    # Browse local models
    # ------------------------------------------------------------------

    def _on_browse_local_models(self) -> None:
        from aurarouter.models.file_storage import FileModelStorage

        storage = FileModelStorage()
        storage.scan()
        models = storage.list_models()

        if not models:
            QMessageBox.information(
                self,
                "No Local Models",
                f"No GGUF models found in {storage.models_dir}\n\n"
                "Download one with:\n"
                "  aurarouter download-model --repo <repo> --file <name>",
            )
            return

        items = [m["filename"] for m in models]
        from PySide6.QtWidgets import QInputDialog

        chosen, ok = QInputDialog.getItem(
            self, "Select Model", "Available local models:", items, 0, False
        )
        if ok and chosen:
            path = storage.get_model_path(chosen)
            if path:
                self._field_inputs["model_path"].setText(str(path))

    # ------------------------------------------------------------------
    # Connection testing
    # ------------------------------------------------------------------

    def _on_test_connection(self) -> None:
        self._test_btn.setEnabled(False)
        self._test_label.setText("Testing...")
        self._test_label.setStyleSheet("")

        config = self.get_model_config()
        self._test_worker = _ConnectionTestWorker(
            provider=config.get("provider", ""), config=config
        )
        self._test_thread = QThread()
        self._test_worker.moveToThread(self._test_thread)

        self._test_thread.started.connect(self._test_worker.run)
        self._test_worker.finished.connect(self._on_test_result)
        self._test_worker.finished.connect(self._test_thread.quit)
        self._test_thread.finished.connect(self._cleanup_test_thread)

        self._test_thread.start()

    def _on_test_result(self, success: bool, message: str) -> None:
        self._test_btn.setEnabled(True)
        if success:
            self._test_label.setText(f"OK: {message}")
            self._test_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self._test_label.setText(f"FAIL: {message}")
            self._test_label.setStyleSheet("color: red; font-weight: bold;")

    def _cleanup_test_thread(self) -> None:
        if self._test_thread:
            self._test_thread.deleteLater()
            self._test_thread = None
        if self._test_worker:
            self._test_worker.deleteLater()
            self._test_worker = None

    # ------------------------------------------------------------------
    # Auto-Tune
    # ------------------------------------------------------------------

    def _on_auto_tune(self) -> None:
        model_path = self._field_inputs.get("model_path")
        if not model_path or not model_path.text().strip():
            QMessageBox.warning(
                self, "Auto-Tune", "Set a model_path before auto-tuning."
            )
            return

        path_str = model_path.text().strip()
        if not Path(path_str).is_file():
            QMessageBox.warning(
                self, "Auto-Tune", f"File not found: {path_str}"
            )
            return

        self._tune_btn.setEnabled(False)
        self._tune_label.setText("Analyzing model...")
        self._tune_label.setStyleSheet("")

        self._tune_worker = _AutoTuneWorker(path_str)
        self._tune_thread = QThread()
        self._tune_worker.moveToThread(self._tune_thread)

        self._tune_thread.started.connect(self._tune_worker.run)
        self._tune_worker.finished.connect(self._on_auto_tune_result)
        self._tune_worker.finished.connect(self._tune_thread.quit)
        self._tune_thread.finished.connect(self._cleanup_tune_thread)

        self._tune_thread.start()

    def _on_auto_tune_result(self, success: bool, message: str, params: dict) -> None:
        self._tune_btn.setEnabled(True)
        if success:
            self._tune_label.setText(f"OK: {message}")
            self._tune_label.setStyleSheet("color: green; font-weight: bold;")
            # Populate the parameters text field
            lines = [f"{k}: {v}" for k, v in params.items()]
            self._params_input.setPlainText("\n".join(lines))
        else:
            self._tune_label.setText(f"FAIL: {message}")
            self._tune_label.setStyleSheet("color: red; font-weight: bold;")

    def _cleanup_tune_thread(self) -> None:
        if self._tune_thread:
            self._tune_thread.deleteLater()
            self._tune_thread = None
        if self._tune_worker:
            self._tune_worker.deleteLater()
            self._tune_worker = None
