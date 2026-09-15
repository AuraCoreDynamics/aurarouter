"""Tests for the GUI intent selector (workspace panel) and intent display (settings panel).

Uses ``pytest-qt`` for widget testing.  Tests skip gracefully when
``pytestqt`` is not installed or no display is available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestqt = pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QApplication

from aurarouter.intent_registry import IntentDefinition, IntentRegistry


# ======================================================================
# Helpers
# ======================================================================


def _make_mock_api(
    active_analyzer: str = "aurarouter-default",
    analyzers: list[dict] | None = None,
    role_bindings: dict[str, str] | None = None,
):
    """Create a minimal mock AuraRouterAPI with config scaffolding."""
    api = MagicMock()

    config = MagicMock()
    api._config = config

    # Provide a real dict for config.config so system_section can read it
    config.config = {
        "system": {"log_level": "INFO", "default_timeout": 120.0},
        "logging": {"level": "INFO"},
        "server": {"timeout": 60},
        "max_review_iterations": 2,
    }

    config.get_active_analyzer.return_value = active_analyzer

    # catalog_get returns analyzer data
    default_data = {
        "kind": "analyzer",
        "display_name": "AuraRouter Default",
        "description": "Intent classification with complexity-based triage routing",
        "analyzer_kind": "intent_triage",
        "capabilities": ["code", "reasoning", "review", "planning"],
        "role_bindings": role_bindings or {
            "simple_code": "coding",
            "complex_reasoning": "reasoning",
            "review": "reviewer",
        },
    }
    config.catalog_get.return_value = default_data

    if analyzers is None:
        analyzers = [
            {
                "artifact_id": "aurarouter-default",
                "display_name": "AuraRouter Default",
                "description": "Default analyzer",
                "analyzer_kind": "intent_triage",
                "capabilities": ["code", "reasoning"],
                "role_bindings": role_bindings or {
                    "simple_code": "coding",
                    "complex_reasoning": "reasoning",
                    "review": "reviewer",
                },
            }
        ]
    config.catalog_query.return_value = analyzers

    # Mock other API methods used by settings panel
    api.get_mcp_tools.return_value = []
    api.get_budget_status.return_value = None
    api.get_system_settings.return_value = {}
    api.list_models.return_value = []
    api.get_config_yaml.return_value = "system:\n  log_level: INFO\n"
    api.config_affects_other_nodes.return_value = False

    return api


# ======================================================================
# T5.5a: Intent combobox creation and default
# ======================================================================


class TestIntentComboboxCreation:
    """Test the intent combobox is created with correct defaults."""

    def test_intent_combo_exists_with_auto_default(self, qtbot):
        """The intent combobox should exist and default to 'Auto (classify)'."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api()
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        assert hasattr(panel, "_intent_combo")
        assert panel._intent_combo.currentText() == "Auto (classify)"
        assert panel._intent_combo.currentData() is None

    def test_auto_selection_returns_none(self, qtbot):
        """Selecting 'Auto (classify)' should return None as intent."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api()
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        panel._intent_combo.setCurrentIndex(0)
        assert panel.get_selected_intent() is None


# ======================================================================
# T5.5b: Built-in intents populated
# ======================================================================


class TestIntentComboboxBuiltins:
    """Test that built-in intents appear in the combobox."""

    def test_builtin_intents_present(self, qtbot):
        """All built-in intents should appear in the combobox."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api()
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        # Collect all items with non-None user data
        combo = panel._intent_combo
        items = {}
        for i in range(combo.count()):
            data = combo.itemData(i)
            if data is not None:
                items[data] = combo.itemText(i)

        # All built-in intents should be present
        for defn in IntentRegistry.BUILTIN_INTENTS:
            assert defn.name in items, f"Built-in intent '{defn.name}' not found"


# ======================================================================
# T5.5c: Analyzer-declared intents populated
# ======================================================================


class TestIntentComboboxAnalyzerIntents:
    """Test that analyzer-declared intents appear in the combobox."""

    def test_analyzer_intents_present(self, qtbot):
        """Analyzer role_bindings should produce entries in the combobox."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api(role_bindings={
            "simple_code": "coding",
            "complex_reasoning": "reasoning",
            "review": "reviewer",
            "sar_processing": "coding",  # custom intent
        })
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        combo = panel._intent_combo
        all_data = [combo.itemData(i) for i in range(combo.count())]

        # The custom "sar_processing" intent should appear
        assert "sar_processing" in all_data

    def test_no_analyzer_intents_when_no_custom(self, qtbot):
        """When analyzer only has standard role bindings matching built-ins,
        the analyzer section may or may not appear (overlapping names
        with built-ins get higher priority via registry)."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api(role_bindings={
            "simple_code": "coding",
            "complex_reasoning": "reasoning",
        })
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        # Should still have Auto + built-in entries at minimum
        combo = panel._intent_combo
        assert combo.count() >= 4  # Auto + separator + header + 3 built-ins


# ======================================================================
# T5.5d: Analyzer change refreshes intent combobox
# ======================================================================


class TestAnalyzerChangeRefreshesIntents:
    """Test that changing the analyzer refreshes the intent combobox."""

    def test_refresh_resets_to_auto(self, qtbot):
        """After refresh_intents(), selection should reset to Auto."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api()
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        # Select a non-auto item
        combo = panel._intent_combo
        for i in range(combo.count()):
            if combo.itemData(i) is not None:
                combo.setCurrentIndex(i)
                break

        # Refresh should reset to Auto
        panel.refresh_intents("some-new-analyzer")
        assert combo.currentIndex() == 0
        assert combo.currentData() is None

    def test_refresh_repopulates(self, qtbot):
        """refresh_intents() should repopulate from the current config state."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api()
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        old_count = panel._intent_combo.count()

        # Change the config mock to return extra intents
        new_data = {
            "kind": "analyzer",
            "display_name": "Custom Analyzer",
            "role_bindings": {
                "simple_code": "coding",
                "complex_reasoning": "reasoning",
                "custom_new_intent": "reasoning",
            },
        }
        api._config.catalog_get.return_value = new_data

        panel.refresh_intents("custom-analyzer")

        # The "custom_new_intent" should now be present
        combo = panel._intent_combo
        all_data = [combo.itemData(i) for i in range(combo.count())]
        assert "custom_new_intent" in all_data


# ======================================================================
# T5.5e: Specific intent selection returns intent name
# ======================================================================


class TestIntentSelectionValues:
    """Test that selecting specific intents returns correct values."""

    def test_specific_intent_returns_name(self, qtbot):
        """Selecting a specific intent should return its name string."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api()
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        combo = panel._intent_combo
        # Find and select DIRECT
        for i in range(combo.count()):
            if combo.itemData(i) == "DIRECT":
                combo.setCurrentIndex(i)
                break

        assert panel.get_selected_intent() == "DIRECT"

    def test_simple_code_intent_returns_name(self, qtbot):
        """Selecting SIMPLE_CODE should return 'SIMPLE_CODE'."""
        from aurarouter.gui.workspace_panel import WorkspacePanel

        api = _make_mock_api()
        panel = WorkspacePanel(api=api)
        qtbot.addWidget(panel)

        combo = panel._intent_combo
        for i in range(combo.count()):
            if combo.itemData(i) == "SIMPLE_CODE":
                combo.setCurrentIndex(i)
                break

        assert panel.get_selected_intent() == "SIMPLE_CODE"


# ======================================================================
# T5.5f: Settings panel shows declared intents
# ======================================================================


class TestSettingsPanelIntents:
    """Test that the settings panel displays declared intents."""

    def test_analyzer_section_shows_intents(self, qtbot):
        """The analyzer details should include intent chip display."""
        from aurarouter.gui.settings_panel import SettingsPanel

        api = _make_mock_api()
        panel = SettingsPanel(api=api)
        qtbot.addWidget(panel)

        # Check that intent chips widget was created
        assert hasattr(panel, "_builtin_intent_chips")
        builtin_tags = panel._builtin_intent_chips.tags()
        assert len(builtin_tags) == len(IntentRegistry.BUILTIN_INTENTS)

        # Check that each built-in intent is represented
        for defn in IntentRegistry.BUILTIN_INTENTS:
            expected = f"{defn.name} \u2192 {defn.target_role}"
            assert expected in builtin_tags, (
                f"Expected '{expected}' in built-in intent chips"
            )

    def test_analyzer_custom_intents_shown(self, qtbot):
        """Analyzer role_bindings should appear as intent chips."""
        from aurarouter.gui.settings_panel import SettingsPanel

        api = _make_mock_api(role_bindings={
            "simple_code": "coding",
            "complex_reasoning": "reasoning",
            "review": "reviewer",
        })
        panel = SettingsPanel(api=api)
        qtbot.addWidget(panel)

        assert hasattr(panel, "_intent_chips")
        tags = panel._intent_chips.tags()
        assert "simple_code \u2192 coding" in tags
        assert "complex_reasoning \u2192 reasoning" in tags
        assert "review \u2192 reviewer" in tags

    def test_analyzer_refresh_updates_intents(self, qtbot):
        """Refreshing the analyzer section should update intent chips."""
        from aurarouter.gui.settings_panel import SettingsPanel

        api = _make_mock_api()
        panel = SettingsPanel(api=api)
        qtbot.addWidget(panel)

        # Change role_bindings via mock and refresh
        new_data = {
            "kind": "analyzer",
            "display_name": "AuraRouter Default",
            "description": "Updated",
            "analyzer_kind": "intent_triage",
            "capabilities": ["code"],
            "role_bindings": {"new_intent": "reasoning"},
        }
        # Update catalog_get to return new data
        api._config.catalog_get.return_value = new_data
        api._config.catalog_query.return_value = [
            {**new_data, "artifact_id": "aurarouter-default"},
        ]

        panel._refresh_analyzer_section()

        # The new intent should be displayed
        assert hasattr(panel, "_intent_chips")
        tags = panel._intent_chips.tags()
        assert "new_intent \u2192 reasoning" in tags
