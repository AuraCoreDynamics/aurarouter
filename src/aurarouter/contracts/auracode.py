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

# TG4: AuraCode must parse this block from GenerateResult.routing_context
# and surface it through ExecutionMetadata on EngineResponse.
AURACODE_ROUTING_CONTEXT_SCHEMA = {
    "strategy": "string — classification method used (vector/complexity/slm/federated)",
    "confidence_score": "float — pipeline confidence 0.0-1.0",
    "complexity_score": "int — prompt complexity 1-10",
    "selected_route": "string — role name routed to",
    "analyzer_chain": "list[str] — ordered analyzer IDs that ran",
    "intent": "string — final resolved intent",
    "hard_routed": "bool — whether cloud was bypassed",
    "simulated_cost_avoided": "float — estimated USD avoided by hard-routing; 0.0 if not hard-routed",
    "metadata": "dict — additional analyzer metadata",
}


def create_auracode_analyzer_spec() -> dict:
    """Return the canonical analyzer spec for an AuraCode-compatible analyzer."""
    return {
        "analyzer_kind": "intent_triage",
        "role_bindings": AURACODE_INTENTS,
        "capabilities": AURACODE_CAPABILITIES,
    }
