"""Integration tests for the AuraRouter GUI."""
import pytest


def test_panel_imports() -> None:
    """Verify all panel modules import without error."""
    from aurarouter.gui.workspace_panel import WorkspacePanel
    from aurarouter.gui.routing_panel import RoutingPanel
    from aurarouter.gui.models_panel import ModelsPanel
    from aurarouter.gui.monitor_panel import MonitorPanel
    from aurarouter.gui.settings_panel import SettingsPanel
    from aurarouter.gui.help.help_panel import HelpPanel
    from aurarouter.gui.config_panel import ConfigPanel  # still importable
    from aurarouter.gui.session_chat import SessionChatWidget
    from aurarouter.gui.intent_editor import IntentEditorPanel
    from aurarouter.gui.speculative_tab import SpeculativeTab
    from aurarouter.gui.monologue_tab import MonologueTab
    from aurarouter.gui.performance_tab import PerformanceTab


def test_widget_imports() -> None:
    """Verify all new widgets import without error."""
    from aurarouter.gui.widgets import (
        TokenPressureGauge,
        TimelineWidget,
        TimelineEntry,
        ConfidenceBar,
        ComplexityBadge,
        ChatBubble,
        RoutingInsightPill,
    )


def test_help_topic_count() -> None:
    """Verify help registry has at least 43 topics (28 original + 15 net new from TG7)."""
    from aurarouter.gui.help.content import HELP
    count = len(HELP._topics) if hasattr(HELP, '_topics') else 0
    if count == 0:
        try:
            count = len(list(HELP.all_topics()))
        except Exception:
            pass
    assert count >= 43, f"Expected \u226543 help topics, got {count}"


def test_help_topic_related_references() -> None:
    """Verify all 'related' topic IDs in help content point to existing topics."""
    from aurarouter.gui.help.content import HELP

    all_ids: set[str] = set(HELP._topics.keys())
    all_topics = list(HELP._topics.values())

    dangling: list[str] = []
    for topic in all_topics:
        for ref in (topic.related or []):
            if ref not in all_ids:
                dangling.append(f"{topic.id} \u2192 {ref}")

    assert not dangling, f"Dangling related references: {dangling}"


def test_wizard_page_count() -> None:
    """Verify onboarding wizard has exactly 6 pages."""
    import inspect
    from aurarouter.gui.help import onboarding as owiz

    page_classes = [
        name for name, obj in inspect.getmembers(owiz, inspect.isclass)
        if name.endswith("Page")
    ]
    assert len(page_classes) == 6, (
        f"Expected 6 page classes, found {len(page_classes)}: {page_classes}"
    )


def test_persona_chooser_replaces_welcome() -> None:
    """Verify WelcomePage was replaced by PersonaChooserPage."""
    from aurarouter.gui.help import onboarding as owiz
    import inspect

    names = {
        name for name, obj in inspect.getmembers(owiz, inspect.isclass)
        if name.endswith("Page")
    }
    assert "PersonaChooserPage" in names, "PersonaChooserPage not found"
    assert "WelcomePage" not in names, "WelcomePage should have been removed"


def test_persona_settings_applied() -> None:
    """Verify _apply_persona writes into WizardState correctly."""
    from aurarouter.gui.help.onboarding import WizardState, _apply_persona, PERSONA_SETTINGS

    for persona_key in ("performance", "privacy", "researcher"):
        state = WizardState()
        _apply_persona(state, persona_key)
        assert state.persona == persona_key
        expected = PERSONA_SETTINGS[persona_key]
        for k, v in expected.items():
            assert state.persona_settings.get(k) == v, (
                f"Persona {persona_key!r}: expected {k}={v!r}, "
                f"got {state.persona_settings.get(k)!r}"
            )


def test_new_api_methods() -> None:
    """Verify all new API methods exist on AuraRouterAPI."""
    from aurarouter.api import AuraRouterAPI
    expected_methods = [
        'list_sessions', 'create_session', 'get_session',
        'add_session_message', 'delete_session', 'execute_in_session',
        'get_speculative_config', 'get_speculative_sessions', 'get_speculative_session',
        'get_monologue_config', 'get_monologue_sessions', 'get_monologue_trace',
        'evaluate_sovereignty', 'get_sovereignty_config',
        'list_intents', 'get_intent',
        'get_model_performance', 'get_savings_summary', 'get_rag_status',
    ]
    missing = [m for m in expected_methods if not hasattr(AuraRouterAPI, m)]
    assert not missing, f"AuraRouterAPI missing methods: {missing}"


def test_theme_new_colors() -> None:
    """Verify new theme color fields exist in both palettes."""
    from aurarouter.gui.theme import DARK_PALETTE, LIGHT_PALETTE
    new_fields = [
        'sovereignty_open', 'sovereignty_local', 'sovereignty_blocked',
        'speculative_draft', 'speculative_verified',
        'monologue_generator', 'monologue_critic', 'monologue_refiner',
    ]
    missing_dark = [f for f in new_fields if not hasattr(DARK_PALETTE, f)]
    missing_light = [f for f in new_fields if not hasattr(LIGHT_PALETTE, f)]
    assert not missing_dark, f"DARK_PALETTE missing fields: {missing_dark}"
    assert not missing_light, f"LIGHT_PALETTE missing fields: {missing_light}"
