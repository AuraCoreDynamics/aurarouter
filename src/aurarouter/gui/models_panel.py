"""Unified model management panel — local, cloud, external, and grid models.

Replaces the old split between ModelsPanel (files) and ConfigPanel (CRUD)
with a single two-column layout: category/provider sidebar on the left,
scrollable model cards on the right, toolbar on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from aurarouter.gui.theme import (
    DARK_PALETTE,
    RADIUS,
    SPACING,
    TYPOGRAPHY,
    ColorPalette,
    get_palette,
)
from aurarouter.gui.widgets import (
    CollapsibleSection,
    HelpTooltip,
    SearchInput,
    StatusBadge,
    TagChips,
)

if TYPE_CHECKING:
    from aurarouter.api import AuraRouterAPI, CatalogEntry, ModelInfo


# ======================================================================
# Tier helpers
# ======================================================================

_TIER_CATEGORY: dict[str, str] = {
    "ollama": "local",
    "llamacpp": "local",
    "llamacpp-server": "local",
    "openapi": "cloud",
    "mcp": "external",
}

_TIER_SORT_ORDER: dict[str, int] = {
    "local": 0,
    "cloud": 1,
    "external": 2,
    "grid": 3,
}


def _model_category(info: "ModelInfo") -> str:
    """Classify a model into local / cloud / external / grid."""
    cfg = info.config or {}
    tier = cfg.get("hosting_tier", "")
    if tier == "on-prem":
        return "local"
    if tier == "cloud":
        return "cloud"
    if tier == "dedicated-tenant":
        return "external"
    tags = cfg.get("tags", [])
    if "grid" in tags or "remote" in tags:
        return "grid"
    return _TIER_CATEGORY.get(info.provider, "external")


def _tier_color(category: str, palette: ColorPalette) -> str:
    """Return the stripe colour for a model category."""
    return {
        "local": palette.tier_local,
        "cloud": palette.tier_cloud,
        "external": palette.warning,
        "grid": palette.tier_grid,
    }.get(category, palette.border)


# ======================================================================
# Background workers
# ======================================================================

class _ConnectionTestWorker(QObject):
    """Run api.test_model_connection() off the GUI thread."""

    finished = Signal(str, bool, str)  # model_id, success, message

    def __init__(self, api: "AuraRouterAPI", model_id: str) -> None:
        super().__init__()
        self._api = api
        self._model_id = model_id

    def run(self) -> None:
        try:
            ok, msg = self._api.test_model_connection(self._model_id)
            self.finished.emit(self._model_id, ok, msg)
        except Exception as exc:
            self.finished.emit(self._model_id, False, str(exc))


class _CatalogRefreshWorker(QObject):
    """Discover catalog providers in background."""

    finished = Signal(list)  # list[CatalogEntry]
    error = Signal(str)

    def __init__(self, api: "AuraRouterAPI") -> None:
        super().__init__()
        self._api = api

    def run(self) -> None:
        try:
            entries = self._api.list_catalog()
            self.finished.emit(entries)
        except Exception as exc:
            self.error.emit(str(exc))


class _AutoRegisterWorker(QObject):
    """Auto-register models from a catalog provider in background."""

    finished = Signal(str, int)  # provider_name, count_added
    error = Signal(str, str)    # provider_name, message

    def __init__(self, api: "AuraRouterAPI", provider_name: str) -> None:
        super().__init__()
        self._api = api
        self._provider_name = provider_name

    def run(self) -> None:
        try:
            count = self._api.auto_register_catalog_models(self._provider_name)
            self.finished.emit(self._provider_name, count)
        except Exception as exc:
            self.error.emit(self._provider_name, str(exc))


# ======================================================================
# Model card widget
# ======================================================================

class _ModelCard(QFrame):
    """A single model card in the main scrollable area."""

    edit_requested = Signal(str)      # model_id
    remove_requested = Signal(str)    # model_id
    test_requested = Signal(str)      # model_id

    def __init__(
        self,
        model_info: "ModelInfo",
        roles: list[str],
        palette: ColorPalette,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._model_id = model_info.model_id
        self._palette = palette
        self._category = _model_category(model_info)
        cfg = model_info.config or {}

        stripe_color = _tier_color(self._category, palette)
        self.setStyleSheet(
            f"_ModelCard {{"
            f"  background-color: {palette.bg_secondary};"
            f"  border: 1px solid {palette.border};"
            f"  border-left: 3px solid {stripe_color};"
            f"  border-radius: {RADIUS.md}px;"
            f"}}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        root.setSpacing(SPACING.xs)

        # ---- Header row ----
        header = QHBoxLayout()
        header.setSpacing(SPACING.sm)

        id_label = QLabel(model_info.model_id)
        id_label.setStyleSheet(
            f"font-weight: bold; font-size: {TYPOGRAPHY.size_h2}px; "
            f"color: {palette.text_primary}; background: transparent;"
        )
        header.addWidget(id_label)

        provider_badge = StatusBadge(
            mode="running" if self._category == "local" else "loading",
            text=model_info.provider or "?",
            palette=palette,
        )
        header.addWidget(provider_badge)

        # Health icon placeholder
        self._health_icon = QLabel("")
        self._health_icon.setStyleSheet("background: transparent;")
        self._health_icon.setFixedWidth(20)
        header.addWidget(self._health_icon)

        header.addStretch()

        edit_btn = QPushButton("Edit")
        edit_btn.setFixedHeight(24)
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(self._model_id))
        header.addWidget(edit_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("danger")
        remove_btn.setFixedHeight(24)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._model_id))
        header.addWidget(remove_btn)

        root.addLayout(header)

        # ---- Detail row ----
        detail_parts: list[str] = []
        if cfg.get("endpoint"):
            detail_parts.append(f"Endpoint: {cfg['endpoint']}")
        if cfg.get("model_name"):
            detail_parts.append(f"Model: {cfg['model_name']}")
        if cfg.get("model_path"):
            detail_parts.append(f"File: {cfg['model_path']}")
        if cfg.get("mcp_endpoint"):
            detail_parts.append(f"MCP: {cfg['mcp_endpoint']}")

        if detail_parts:
            detail_label = QLabel("  |  ".join(detail_parts))
            detail_label.setStyleSheet(
                f"color: {palette.text_secondary}; "
                f"font-size: {TYPOGRAPHY.size_small}px; "
                f"background: transparent;"
            )
            detail_label.setWordWrap(True)
            root.addWidget(detail_label)

        # ---- Tags ----
        tags = cfg.get("tags", [])
        if tags:
            chips = TagChips(editable=False, palette=palette, parent=self)
            for t in tags:
                chips.add_tag(t)
            root.addWidget(chips)

        # ---- Roles ----
        if roles:
            roles_label = QLabel(f"Roles: {', '.join(roles)}")
            roles_label.setStyleSheet(
                f"color: {palette.text_secondary}; "
                f"font-size: {TYPOGRAPHY.size_small}px; "
                f"font-style: italic; background: transparent;"
            )
            root.addWidget(roles_label)

        # ---- Action row (conditional) ----
        actions = QHBoxLayout()
        actions.setSpacing(SPACING.sm)

        test_btn = QPushButton("Test Connection")
        test_btn.setFixedHeight(22)
        test_btn.clicked.connect(lambda: self.test_requested.emit(self._model_id))
        actions.addWidget(test_btn)

        self._test_status = QLabel("")
        self._test_status.setStyleSheet("background: transparent;")
        actions.addWidget(self._test_status)

        if model_info.provider == "llamacpp":
            tune_btn = QPushButton("Auto-Tune")
            tune_btn.setFixedHeight(22)
            tune_btn.setToolTip("Analyze GGUF and recommend parameters")
            tune_btn.clicked.connect(lambda: self._on_auto_tune())
            actions.addWidget(tune_btn)

        actions.addStretch()
        root.addLayout(actions)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def category(self) -> str:
        return self._category

    def set_health(self, healthy: bool) -> None:
        """Update the inline health icon."""
        if healthy:
            self._health_icon.setText("\u2705")
        else:
            self._health_icon.setText("\u274c")

    def set_test_result(self, success: bool, message: str) -> None:
        """Show the connection test result inline."""
        if success:
            self._test_status.setText(f"\u2705 {message}")
            self._test_status.setStyleSheet(
                f"color: {self._palette.success}; font-size: {TYPOGRAPHY.size_small}px; "
                f"font-weight: bold; background: transparent;"
            )
        else:
            self._test_status.setText(f"\u274c {message}")
            self._test_status.setStyleSheet(
                f"color: {self._palette.error}; font-size: {TYPOGRAPHY.size_small}px; "
                f"font-weight: bold; background: transparent;"
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_auto_tune(self) -> None:
        """Placeholder — auto-tune opens the edit dialog with tune pre-triggered."""
        self.edit_requested.emit(self._model_id)


# ======================================================================
# Main panel
# ======================================================================

class ModelsPanel(QWidget):
    """Unified model management — local, cloud, external, grid models."""

    def __init__(
        self,
        api: "AuraRouterAPI",
        help_registry: Optional[dict] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._api = api
        self._help_registry = help_registry or {}
        self._palette = get_palette("dark")
        self._cards: list[_ModelCard] = []
        self._catalog_entries: list["CatalogEntry"] = []
        self._filter_text = ""
        self._filter_category = "all"

        # Thread management
        self._bg_thread: Optional[QThread] = None
        self._bg_worker: Optional[QObject] = None

        self._build_ui()
        self._refresh_models()
        self._refresh_catalog()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Left sidebar (160px) ----
        sidebar = QWidget()
        sidebar.setFixedWidth(160)
        sidebar.setStyleSheet(
            f"background-color: {self._palette.bg_secondary}; "
            f"border-right: 1px solid {self._palette.border};"
        )
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(SPACING.sm, SPACING.sm, SPACING.sm, SPACING.sm)
        sb_layout.setSpacing(SPACING.sm)

        # Category filters
        cat_label = QLabel("Categories")
        cat_label.setStyleSheet(
            f"font-weight: bold; font-size: {TYPOGRAPHY.size_small}px; "
            f"color: {self._palette.text_secondary}; background: transparent;"
        )
        sb_layout.addWidget(cat_label)

        self._category_list = QListWidget()
        self._category_list.setMaximumHeight(140)
        self._category_list.setStyleSheet(
            f"QListWidget {{"
            f"  background-color: {self._palette.bg_secondary};"
            f"  border: none;"
            f"  color: {self._palette.text_primary};"
            f"  font-size: {TYPOGRAPHY.size_body}px;"
            f"}}"
            f"QListWidget::item {{"
            f"  padding: {SPACING.xs}px {SPACING.sm}px;"
            f"  border-radius: {RADIUS.sm}px;"
            f"}}"
            f"QListWidget::item:selected {{"
            f"  background-color: {self._palette.bg_hover};"
            f"  color: {self._palette.accent};"
            f"}}"
        )
        for key, label in [
            ("all", "All"),
            ("local", "Local"),
            ("cloud", "Cloud"),
            ("external", "External"),
            ("grid", "Grid"),
        ]:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._category_list.addItem(item)
        self._category_list.setCurrentRow(0)
        self._category_list.currentItemChanged.connect(self._on_category_changed)
        sb_layout.addWidget(self._category_list)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"background-color: {self._palette.separator};")
        sep1.setFixedHeight(1)
        sb_layout.addWidget(sep1)

        # Provider Catalog section
        self._catalog_section = CollapsibleSection(
            "Provider Catalog", initially_expanded=False, palette=self._palette
        )
        self._catalog_content = QWidget()
        self._catalog_layout = QVBoxLayout(self._catalog_content)
        self._catalog_layout.setContentsMargins(0, 0, 0, 0)
        self._catalog_layout.setSpacing(SPACING.xs)

        self._catalog_status_label = QLabel("Discovering...")
        self._catalog_status_label.setStyleSheet(
            f"color: {self._palette.text_secondary}; "
            f"font-size: {TYPOGRAPHY.size_small}px; background: transparent;"
        )
        self._catalog_layout.addWidget(self._catalog_status_label)

        # Provider list container (populated dynamically)
        self._provider_list_widget = QWidget()
        self._provider_list_layout = QVBoxLayout(self._provider_list_widget)
        self._provider_list_layout.setContentsMargins(0, 0, 0, 0)
        self._provider_list_layout.setSpacing(SPACING.xs)
        self._catalog_layout.addWidget(self._provider_list_widget)

        # Catalog action buttons
        auto_reg_btn = QPushButton("Auto-Register Models")
        auto_reg_btn.setFixedHeight(22)
        auto_reg_btn.setToolTip("Discover and register models from all running providers")
        auto_reg_btn.clicked.connect(self._on_auto_register_all)
        self._catalog_layout.addWidget(auto_reg_btn)

        add_provider_btn = QPushButton("Add Provider...")
        add_provider_btn.setFixedHeight(22)
        add_provider_btn.setToolTip("Manually register an MCP provider endpoint")
        add_provider_btn.clicked.connect(self._on_add_provider)
        self._catalog_layout.addWidget(add_provider_btn)

        self._catalog_section.add_widget(self._catalog_content)
        sb_layout.addWidget(self._catalog_section)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"background-color: {self._palette.separator};")
        sep2.setFixedHeight(1)
        sb_layout.addWidget(sep2)

        # Actions
        dl_btn = QPushButton("Download from HF")
        dl_btn.setFixedHeight(26)
        dl_btn.setToolTip("Download a GGUF model from HuggingFace Hub")
        dl_btn.clicked.connect(self._on_download)
        sb_layout.addWidget(dl_btn)

        import_btn = QPushButton("Import Local File")
        import_btn.setFixedHeight(26)
        import_btn.setToolTip("Register an existing GGUF file in the model storage")
        import_btn.clicked.connect(self._on_import_local)
        sb_layout.addWidget(import_btn)

        sb_layout.addStretch()

        # Storage info
        self._storage_label = QLabel("")
        self._storage_label.setWordWrap(True)
        self._storage_label.setStyleSheet(
            f"color: {self._palette.text_disabled}; "
            f"font-size: {TYPOGRAPHY.size_small}px; background: transparent;"
        )
        sb_layout.addWidget(self._storage_label)

        root.addWidget(sidebar)

        # ---- Main area ----
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(SPACING.md, SPACING.sm, SPACING.md, SPACING.sm)
        main_layout.setSpacing(SPACING.sm)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACING.sm)

        add_btn = QPushButton("+ Add Model")
        add_btn.setObjectName("primary")
        add_btn.setFixedHeight(28)
        add_btn.clicked.connect(self._on_add_model)
        toolbar.addWidget(add_btn)

        self._search = SearchInput(
            placeholder="Filter by model ID, provider, tags, tier...",
            palette=self._palette,
        )
        self._search.search_changed.connect(self._on_search_changed)
        toolbar.addWidget(self._search, 1)

        help_btn = HelpTooltip(
            help_text=(
                "Manage all configured models in one place. "
                "Use categories to filter, the search bar to find specific models, "
                "and the provider catalog to discover new ones."
            ),
            palette=self._palette,
        )
        toolbar.addWidget(help_btn)

        main_layout.addLayout(toolbar)

        # Scrollable card area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._cards_container = QWidget()
        self._cards_container.setStyleSheet("background: transparent;")
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(SPACING.sm)
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._scroll.setWidget(self._cards_container)
        main_layout.addWidget(self._scroll, 1)

        # Empty state label
        self._empty_label = QLabel("No models configured. Click '+ Add Model' to get started.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {self._palette.text_disabled}; "
            f"font-size: {TYPOGRAPHY.size_h2}px; "
            f"padding: {SPACING.xxl}px; background: transparent;"
        )
        self._empty_label.setVisible(False)
        main_layout.addWidget(self._empty_label)

        root.addWidget(main, 1)

    # ==================================================================
    # Model list
    # ==================================================================

    def _refresh_models(self) -> None:
        """Reload all models from the API and rebuild cards."""
        # Clear existing cards
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        models = self._api.list_models()

        # Build role lookup: model_id -> list of role names
        role_lookup: dict[str, list[str]] = {}
        for rc in self._api.list_roles():
            for mid in rc.chain:
                role_lookup.setdefault(mid, []).append(rc.role)

        # Sort: local first, then cloud, external, grid; alpha within groups
        models.sort(key=lambda m: (
            _TIER_SORT_ORDER.get(_model_category(m), 99),
            m.model_id.lower(),
        ))

        # Build cards
        for m in models:
            roles = role_lookup.get(m.model_id, [])
            card = _ModelCard(m, roles, self._palette, parent=self._cards_container)
            card.edit_requested.connect(self._on_edit_model)
            card.remove_requested.connect(self._on_remove_model)
            card.test_requested.connect(self._on_test_connection)
            self._cards.append(card)
            self._cards_layout.addWidget(card)

        self._update_category_counts(models)
        self._update_storage_info()
        self._apply_filters()

    def _update_category_counts(self, models: list["ModelInfo"]) -> None:
        """Update the sidebar category item labels with counts."""
        counts: dict[str, int] = {"all": len(models), "local": 0, "cloud": 0, "external": 0, "grid": 0}
        for m in models:
            cat = _model_category(m)
            counts[cat] = counts.get(cat, 0) + 1

        for i in range(self._category_list.count()):
            item = self._category_list.item(i)
            if item is None:
                continue
            key = item.data(Qt.ItemDataRole.UserRole)
            n = counts.get(key, 0)
            base_label = {"all": "All", "local": "Local", "cloud": "Cloud",
                          "external": "External", "grid": "Grid"}.get(key, key)
            item.setText(f"{base_label} ({n})")

    def _update_storage_info(self) -> None:
        """Update the sidebar storage label."""
        try:
            info = self._api.get_storage_info()
            gb = info.total_bytes / (1024 * 1024 * 1024)
            self._storage_label.setText(
                f"{info.models_dir}\n"
                f"{info.total_files} file{'s' if info.total_files != 1 else ''}, "
                f"{gb:.1f} GB"
            )
        except Exception:
            self._storage_label.setText("Storage info unavailable")

    def _apply_filters(self) -> None:
        """Show/hide cards based on current category and search text."""
        query = self._filter_text.lower()
        visible_count = 0

        for card in self._cards:
            # Category filter
            if self._filter_category != "all" and card.category != self._filter_category:
                card.setVisible(False)
                continue

            # Text filter
            if query:
                searchable = card.model_id.lower()
                # Also search provider, tags from model info
                model = self._api.get_model(card.model_id)
                if model:
                    cfg = model.config or {}
                    searchable += f" {model.provider} "
                    searchable += " ".join(cfg.get("tags", []))
                    searchable += f" {cfg.get('hosting_tier', '')}"
                if query not in searchable:
                    card.setVisible(False)
                    continue

            card.setVisible(True)
            visible_count += 1

        self._empty_label.setVisible(visible_count == 0 and len(self._cards) == 0)

    # ==================================================================
    # Category / search
    # ==================================================================

    def _on_category_changed(self, current: QListWidgetItem, _previous: QListWidgetItem) -> None:
        if current is None:
            return
        self._filter_category = current.data(Qt.ItemDataRole.UserRole)
        self._apply_filters()

    def _on_search_changed(self, text: str) -> None:
        self._filter_text = text
        self._apply_filters()

    # ==================================================================
    # CRUD operations
    # ==================================================================

    def _on_add_model(self) -> None:
        from aurarouter.gui.model_dialog import ModelDialog

        dlg = ModelDialog(parent=self)
        if dlg.exec() == ModelDialog.DialogCode.Accepted:
            model_id = dlg.get_model_id()
            existing = self._api.get_model(model_id)
            if existing is not None:
                QMessageBox.warning(self, "Duplicate", f"Model '{model_id}' already exists.")
                return
            self._api.add_model(model_id, dlg.get_model_config())
            self._refresh_models()

    def _on_edit_model(self, model_id: str) -> None:
        from aurarouter.gui.model_dialog import ModelDialog

        info = self._api.get_model(model_id)
        if info is None:
            return
        dlg = ModelDialog(parent=self, model_id=model_id, model_config=info.config)
        if dlg.exec() == ModelDialog.DialogCode.Accepted:
            self._api.update_model(model_id, dlg.get_model_config())
            self._refresh_models()

    def _on_remove_model(self, model_id: str) -> None:
        confirm = QMessageBox.question(
            self, "Confirm Removal", f"Remove model '{model_id}' from configuration?"
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._api.remove_model(model_id)
            self._refresh_models()

    # ==================================================================
    # Connection testing
    # ==================================================================

    def _on_test_connection(self, model_id: str) -> None:
        """Run a connection test in a background thread."""
        # Find the card and show loading state
        for card in self._cards:
            if card.model_id == model_id:
                card.set_test_result(True, "Testing...")
                break

        self._cleanup_bg_thread()

        worker = _ConnectionTestWorker(self._api, model_id)
        self._bg_thread = QThread()
        self._bg_worker = worker
        worker.moveToThread(self._bg_thread)

        self._bg_thread.started.connect(worker.run)
        worker.finished.connect(self._on_test_result)
        worker.finished.connect(self._bg_thread.quit)
        self._bg_thread.finished.connect(self._cleanup_bg_thread)

        self._bg_thread.start()

    def _on_test_result(self, model_id: str, success: bool, message: str) -> None:
        for card in self._cards:
            if card.model_id == model_id:
                card.set_test_result(success, message)
                card.set_health(success)
                break

    # ==================================================================
    # Download / Import
    # ==================================================================

    def _on_download(self) -> None:
        from aurarouter.gui.download_dialog import DownloadDialog

        dlg = DownloadDialog(parent=self)
        dlg.download_complete.connect(self._on_download_complete)
        dlg.exec()

    def _on_download_complete(self) -> None:
        """After a download completes, refresh to pick up any auto-registered models."""
        self._refresh_models()

    def _on_import_local(self) -> None:
        """Import a GGUF model file from a local path via file browser."""
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

        size_mb = path_obj.stat().st_size / (1024 * 1024)
        reply = QMessageBox.question(
            self,
            "Import Model",
            f"Register '{path_obj.name}' ({size_mb:.0f} MB) in the model storage?\n\n"
            f"The file will remain at its current location:\n{path}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._api.import_asset(str(path_obj), repo="local-import")
        self._refresh_models()

    # ==================================================================
    # Provider Catalog
    # ==================================================================

    def _refresh_catalog(self) -> None:
        """Discover catalog providers in background."""
        self._cleanup_bg_thread()

        worker = _CatalogRefreshWorker(self._api)
        self._bg_thread = QThread()
        self._bg_worker = worker
        worker.moveToThread(self._bg_thread)

        self._bg_thread.started.connect(worker.run)
        worker.finished.connect(self._on_catalog_loaded)
        worker.error.connect(self._on_catalog_error)
        worker.finished.connect(self._bg_thread.quit)
        worker.error.connect(self._bg_thread.quit)
        self._bg_thread.finished.connect(self._cleanup_bg_thread)

        self._bg_thread.start()

    def _on_catalog_loaded(self, entries: list) -> None:
        self._catalog_entries = entries
        self._catalog_status_label.setText(f"{len(entries)} provider(s)")

        # Clear old provider widgets
        while self._provider_list_layout.count():
            item = self._provider_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Build provider rows
        for entry in entries:
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)

            # Name + badge
            name_row = QHBoxLayout()
            name_row.setSpacing(SPACING.xs)

            name_lbl = QLabel(entry.name)
            name_lbl.setStyleSheet(
                f"font-size: {TYPOGRAPHY.size_small}px; "
                f"color: {self._palette.text_primary}; "
                f"font-weight: bold; background: transparent;"
            )
            name_row.addWidget(name_lbl)

            if entry.running:
                badge = StatusBadge("running", text="Running", palette=self._palette)
            elif entry.installed:
                badge = StatusBadge("stopped", text="Installed", palette=self._palette)
            else:
                badge = StatusBadge("stopped", text="Not Installed", palette=self._palette)
            name_row.addWidget(badge)
            name_row.addStretch()
            row_layout.addLayout(name_row)

            # Action buttons for external providers
            if entry.installed:
                btn_row = QHBoxLayout()
                btn_row.setSpacing(SPACING.xs)
                if entry.running:
                    stop_btn = QPushButton("Stop")
                    stop_btn.setFixedHeight(18)
                    stop_btn.setFixedWidth(40)
                    provider_name = entry.name
                    stop_btn.clicked.connect(
                        lambda _checked=False, n=provider_name: self._on_stop_provider(n)
                    )
                    btn_row.addWidget(stop_btn)
                else:
                    start_btn = QPushButton("Start")
                    start_btn.setFixedHeight(18)
                    start_btn.setFixedWidth(40)
                    provider_name = entry.name
                    start_btn.clicked.connect(
                        lambda _checked=False, n=provider_name: self._on_start_provider(n)
                    )
                    btn_row.addWidget(start_btn)
                btn_row.addStretch()
                row_layout.addLayout(btn_row)

            self._provider_list_layout.addWidget(row_widget)

    def _on_catalog_error(self, message: str) -> None:
        self._catalog_status_label.setText(f"Error: {message}")

    def _on_start_provider(self, name: str) -> None:
        try:
            ok = self._api.start_catalog_provider(name)
            if ok:
                self._refresh_catalog()
            else:
                QMessageBox.warning(self, "Start Failed", f"Could not start provider '{name}'.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _on_stop_provider(self, name: str) -> None:
        try:
            ok = self._api.stop_catalog_provider(name)
            if ok:
                self._refresh_catalog()
            else:
                QMessageBox.warning(self, "Stop Failed", f"Could not stop provider '{name}'.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def _on_auto_register_all(self) -> None:
        """Auto-register models from all running catalog providers."""
        running = [e for e in self._catalog_entries if e.running]
        if not running:
            QMessageBox.information(
                self, "No Running Providers",
                "No providers are currently running. Start a provider first.",
            )
            return

        total_added = 0
        for entry in running:
            try:
                count = self._api.auto_register_catalog_models(entry.name)
                total_added += count
            except Exception:
                pass

        if total_added > 0:
            self._refresh_models()
            QMessageBox.information(
                self, "Auto-Register",
                f"Registered {total_added} new model(s) from {len(running)} provider(s).",
            )
        else:
            QMessageBox.information(
                self, "Auto-Register",
                "No new models discovered from running providers.",
            )

    def _on_add_provider(self) -> None:
        """Manually add an MCP provider by name and endpoint."""
        name, ok = QInputDialog.getText(
            self, "Add Provider", "Provider name:"
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        endpoint, ok2 = QInputDialog.getText(
            self, "Add Provider", "MCP endpoint URL:",
            QLineEdit.EchoMode.Normal,
            "http://localhost:8080",
        )
        if not ok2 or not endpoint.strip():
            return
        endpoint = endpoint.strip()

        try:
            self._api.add_catalog_provider(name, endpoint)
            self._refresh_catalog()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to add provider: {exc}")

    # ==================================================================
    # Thread management
    # ==================================================================

    def _cleanup_bg_thread(self) -> None:
        if self._bg_thread is not None:
            if self._bg_thread.isRunning():
                self._bg_thread.quit()
                self._bg_thread.wait(3000)
            self._bg_thread.deleteLater()
            self._bg_thread = None
        if self._bg_worker is not None:
            self._bg_worker.deleteLater()
            self._bg_worker = None
