from __future__ import annotations

from pathlib import Path
from typing import Optional

import re
import webbrowser
from PySide6.QtCore import QObject, QThread, Signal, QTimer
from PySide6.QtGui import QClipboard, QGuiApplication
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

from aurarouter.gui.theme import DARK_PALETTE, TYPOGRAPHY, get_palette
from aurarouter.gui.widgets import StatusBadge, TagChips
from aurarouter.auth.registry import get_auth_metadata, AuthMetadata, AUTH_REGISTRY

PROVIDERS = ["ollama", "llamacpp-server", "llamacpp", "openapi", "mcp"]

# Fields that appear for each provider type
_PROVIDER_FIELDS: dict[str, list[str]] = {
    "ollama": ["endpoint", "model_name"],
    "llamacpp-server": ["endpoint"],
    "llamacpp": ["model_path"],
    "openapi": ["endpoint", "model_name", "api_key", "env_key"],
    "mcp": ["mcp_endpoint", "model_name"],
}

_FIELD_DEFAULTS: dict[str, str] = {
    "endpoint": "http://localhost:11434/api/generate",
    "model_name": "",
    "api_key": "",
    "env_key": "",
    "model_path": "",
    "mcp_endpoint": "http://localhost:8080",
}

_LLAMACPP_SERVER_ENDPOINT_DEFAULT = "http://localhost:8080"
_OPENAPI_ENDPOINT_DEFAULT = "http://localhost:8000/v1"
_MCP_ENDPOINT_DEFAULT = "http://localhost:8080"


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
            elif self.provider == "llamacpp-server":
                self._test_llamacpp_server()
            elif self.provider == "llamacpp":
                self._test_llamacpp()
            elif self.provider == "openapi":
                self._test_openapi()
            elif self.provider == "mcp":
                self._test_mcp()
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

    def _test_mcp(self) -> None:
        import httpx

        endpoint = self.config.get("mcp_endpoint", _MCP_ENDPOINT_DEFAULT)
        url = endpoint.rstrip("/") + "/health"
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        self.finished.emit(True, "MCP endpoint is reachable.")


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
        self._palette = get_palette("dark")
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
        for field in ("endpoint", "model_name", "api_key", "env_key", "model_path", "mcp_endpoint"):
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
                label = field.replace("_", " ").replace("mcp endpoint", "MCP Endpoint")
                self._form.addRow(f"{label}:", inp)

        # Tags — TagChips widget with inline add
        tags_row = QHBoxLayout()
        self._tag_chips = TagChips(editable=True, palette=self._palette)
        tags_row.addWidget(self._tag_chips, 1)
        self._tag_add_input = QLineEdit()
        self._tag_add_input.setPlaceholderText("Add tag...")
        self._tag_add_input.setMaximumWidth(120)
        self._tag_add_input.returnPressed.connect(self._on_add_tag)
        tags_row.addWidget(self._tag_add_input)
        add_tag_btn = QPushButton("+")
        add_tag_btn.setFixedSize(24, 24)
        add_tag_btn.clicked.connect(self._on_add_tag)
        tags_row.addWidget(add_tag_btn)
        self._form.addRow("Tags:", tags_row)

        # Hosting tier (shown for all providers)
        self._hosting_tier_combo = QComboBox()
        self._hosting_tier_combo.addItems(["", "on-prem", "cloud", "dedicated-tenant"])
        self._hosting_tier_combo.setToolTip(
            "Override hosting classification. Leave blank to auto-detect from provider."
        )
        self._form.addRow("Hosting Tier:", self._hosting_tier_combo)

        # Cost per 1M tokens (shown for all providers)
        self._cost_input = QLineEdit()
        self._cost_input.setPlaceholderText("0.00")
        self._cost_input.setToolTip("Cost per 1M input tokens (USD). Leave blank for auto.")
        self._form.addRow("Cost/1M Input:", self._cost_input)

        self._cost_output = QLineEdit()
        self._cost_output.setPlaceholderText("0.00")
        self._cost_output.setToolTip("Cost per 1M output tokens (USD). Leave blank for auto.")
        self._form.addRow("Cost/1M Output:", self._cost_output)

        # Parameters (free-form YAML-ish key: value)
        self._params_input = QTextEdit()
        self._params_input.setPlaceholderText(
            "temperature: 0.1\nn_ctx: 4096\nmax_tokens: 2048"
        )
        self._params_input.setMaximumHeight(100)
        self._form.addRow("Parameters:", self._params_input)

        layout.addLayout(self._form)

        # --- Test Connection (with StatusBadge) ---
        test_row = QHBoxLayout()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test_connection)
        test_row.addWidget(self._test_btn)
        self._test_badge = StatusBadge("stopped", text="Not tested", palette=self._palette)
        test_row.addWidget(self._test_badge)
        self._test_label = QLabel("")
        self._test_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_small}px;"
        )
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

        # --- Quick Connect Wizard (for cloud providers) ---
        self._quick_connect_row = QHBoxLayout()
        self._quick_connect_btn = QPushButton("Quick Connect")
        self._quick_connect_btn.setToolTip("Open browser to get API key and auto-capture from clipboard.")
        self._quick_connect_btn.clicked.connect(self._on_quick_connect)
        self._quick_connect_row.addWidget(self._quick_connect_btn)
        
        self._quick_connect_badge = StatusBadge("stopped", text="Not listening", palette=self._palette)
        self._quick_connect_row.addWidget(self._quick_connect_badge)
        
        self._quick_connect_label = QLabel("")
        self._quick_connect_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_small}px;"
        )
        self._quick_connect_row.addWidget(self._quick_connect_label)
        self._quick_connect_row.addStretch()
        layout.addLayout(self._quick_connect_row)

        # Clipboard polling timer
        self._clipboard_timer = QTimer(self)
        self._clipboard_timer.setInterval(1000)
        self._clipboard_timer.timeout.connect(self._check_clipboard)
        self._last_clipboard_text = ""

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
            self._tag_chips.clear_tags()
            for t in tags:
                self._tag_chips.add_tag(t)

        hosting_tier = cfg.get("hosting_tier", "")
        idx = self._hosting_tier_combo.findText(hosting_tier)
        if idx >= 0:
            self._hosting_tier_combo.setCurrentIndex(idx)

        cost_in = cfg.get("cost_per_1m_input")
        if cost_in is not None:
            self._cost_input.setText(str(cost_in))
        cost_out = cfg.get("cost_per_1m_output")
        if cost_out is not None:
            self._cost_output.setText(str(cost_out))

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

        # Show Quick Connect only for openapi (cloud) providers
        is_cloud = provider == "openapi"
        self._quick_connect_btn.setVisible(is_cloud)
        self._quick_connect_badge.setVisible(is_cloud)
        self._quick_connect_label.setVisible(is_cloud)

        # Set a sensible default endpoint per provider
        if provider == "llamacpp-server" and not self._field_inputs["endpoint"].text():
            self._field_inputs["endpoint"].setText(_LLAMACPP_SERVER_ENDPOINT_DEFAULT)
        elif provider == "ollama" and not self._field_inputs["endpoint"].text():
            self._field_inputs["endpoint"].setText("http://localhost:11434/api/generate")
        elif provider == "openapi" and not self._field_inputs["endpoint"].text():
            self._field_inputs["endpoint"].setText(_OPENAPI_ENDPOINT_DEFAULT)
        elif provider == "mcp" and not self._field_inputs["mcp_endpoint"].text():
            self._field_inputs["mcp_endpoint"].setText(_MCP_ENDPOINT_DEFAULT)

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

        # Tags (from TagChips widget)
        tags = self._tag_chips.tags()
        if tags:
            cfg["tags"] = tags

        # Hosting tier
        tier = self._hosting_tier_combo.currentText()
        if tier:
            cfg["hosting_tier"] = tier

        # Cost per 1M tokens
        cost_in_text = self._cost_input.text().strip()
        if cost_in_text:
            try:
                cfg["cost_per_1m_input"] = float(cost_in_text)
            except ValueError:
                pass
        cost_out_text = self._cost_output.text().strip()
        if cost_out_text:
            try:
                cfg["cost_per_1m_output"] = float(cost_out_text)
            except ValueError:
                pass

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
    # Tag management
    # ------------------------------------------------------------------

    def _on_add_tag(self) -> None:
        """Add a tag from the input field to the TagChips widget."""
        text = self._tag_add_input.text().strip()
        if text and text not in self._tag_chips.tags():
            self._tag_chips.add_tag(text)
        self._tag_add_input.clear()

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
        self._test_badge.set_mode("loading", text="Testing...")
        self._test_label.setText("")
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
            self._test_badge.set_mode("healthy", text="Connected")
            self._test_label.setText(message)
            self._test_label.setStyleSheet(
                f"color: {self._palette.success}; font-weight: bold;"
            )
        else:
            self._test_badge.set_mode("error", text="Failed")
            self._test_label.setText(message)
            self._test_label.setStyleSheet(
                f"color: {self._palette.error}; font-weight: bold;"
            )

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
            self._tune_label.setStyleSheet(
                f"color: {self._palette.success}; font-weight: bold;"
            )
            # Populate the parameters text field
            lines = [f"{k}: {v}" for k, v in params.items()]
            self._params_input.setPlainText("\n".join(lines))
        else:
            self._tune_label.setText(f"FAIL: {message}")
            self._tune_label.setStyleSheet(
                f"color: {self._palette.error}; font-weight: bold;"
            )

    def _cleanup_tune_thread(self) -> None:
        if self._tune_thread:
            self._tune_thread.deleteLater()
            self._tune_thread = None
        if self._tune_worker:
            self._tune_worker.deleteLater()
            self._tune_worker = None

    # ------------------------------------------------------------------
    # Quick Connect Wizard
    # ------------------------------------------------------------------

    def _on_quick_connect(self) -> None:
        """Start the quick connect sequence for the current model name/provider."""
        model_name = self._field_inputs["model_name"].text().strip().lower()
        
        # Try to find metadata by model name first (e.g. "gpt-4o"), then provider
        provider_id = self._provider_combo.currentText().lower()
        meta = get_auth_metadata(provider_id)
        if not meta:
            # Try to infer provider from model name
            if "gpt" in model_name: meta = get_auth_metadata("openai")
            elif "claude" in model_name: meta = get_auth_metadata("anthropic")
            elif "gemini" in model_name: meta = get_auth_metadata("google")
            
        if not meta:
            QMessageBox.information(self, "Quick Connect", 
                                    "No automated connector found for this provider. "
                                    "Please paste your API key manually.")
            return

        # Start listening
        self._quick_connect_badge.set_mode("loading", text="Listening...")
        self._quick_connect_label.setText(f"Waiting for key from {meta.display_name}...")
        self._quick_connect_btn.setEnabled(False)
        self._clipboard_timer.start()
        
        # Open browser
        webbrowser.open(meta.auth_url)

    def _check_clipboard(self) -> None:
        """Poll the clipboard for strings matching known API key patterns."""
        clipboard = QGuiApplication.clipboard()
        text = clipboard.text().strip()
        
        if text == self._last_clipboard_text or not text:
            return
            
        self._last_clipboard_text = text
        
        # Check against all known patterns (user might have open multiple tabs)
        for provider_id, meta in AUTH_REGISTRY.items():
            if re.match(meta.key_regex, text):
                # Found a match!
                self._field_inputs["api_key"].setText(text)
                self._quick_connect_badge.set_mode("healthy", text="Captured!")
                self._quick_connect_label.setText(f"Successfully captured {meta.display_name} key.")
                self._quick_connect_btn.setEnabled(True)
                self._clipboard_timer.stop()
                
                # Highlight the field briefly
                self._field_inputs["api_key"].setStyleSheet(
                    f"background-color: rgba(166, 227, 161, 0.2); border: 1px solid {self._palette.success};"
                )
                QTimer.singleShot(2000, lambda: self._field_inputs["api_key"].setStyleSheet(""))
                return
