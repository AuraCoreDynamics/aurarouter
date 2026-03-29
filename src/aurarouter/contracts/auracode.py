"""AuraCode domain analyzer contract.

Defines the intents, role mappings, and routing hints that AuraCode expects
from its AuraRouter integration.
"""

AURACODE_INTENTS: dict[str, str] = {
    "generate_code": "coding",
    "edit_code": "coding",
    "complete_code": "coding",
    "explain_code": "reasoning",
    "review": "reasoning",
    "chat": "reasoning",
    "plan": "reasoning",
}

AURACODE_CAPABILITIES = ["code", "reasoning", "review", "planning"]


def create_auracode_analyzer_spec() -> dict:
    """Return the canonical analyzer spec for an AuraCode-compatible analyzer."""
    return {
        "analyzer_kind": "intent_triage",
        "role_bindings": AURACODE_INTENTS,
        "capabilities": AURACODE_CAPABILITIES,
    }
