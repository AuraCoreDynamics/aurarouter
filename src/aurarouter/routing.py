import json
from dataclasses import dataclass
from typing import Optional

from aurarouter._logging import get_logger
from aurarouter.fabric import ComputeFabric
from aurarouter.semantic_verbs import resolve_synonym

logger = get_logger("AuraRouter.Routing")


@dataclass
class TriageResult:
    """Result of intent analysis including complexity scoring."""

    intent: str
    complexity: int


def analyze_intent(
    fabric: ComputeFabric,
    task: str,
    custom_verbs: Optional[dict[str, list[str]]] = None,
) -> TriageResult:
    """Classify a task and estimate complexity via the router role.

    Returns a ``TriageResult`` with *intent* (``SIMPLE_CODE`` or
    ``COMPLEX_REASONING``) and *complexity* (1\u201310, default 5).

    If *custom_verbs* is provided, the classifier prompt includes known
    roles and synonyms for context, and the returned intent is normalised
    via ``resolve_synonym``.
    """
    # Build available-roles context for the classifier.
    roles_hint = ""
    if custom_verbs:
        parts = [f"{role} (synonyms: {', '.join(syns)})" for role, syns in custom_verbs.items() if syns]
        if parts:
            roles_hint = "Available roles: " + "; ".join(parts) + "\n"

    prompt = (
        "CLASSIFY intent.\n"
        f'Task: "{task}"\n'
        'Options: ["SIMPLE_CODE", "COMPLEX_REASONING"]\n'
        f"{roles_hint}"
        'Return JSON: {"intent": "...", "complexity": 5}\n'
        "Where complexity is 1-10 (1=trivial, 10=very complex)."
    )
    res = fabric.execute("router", prompt, json_mode=True)
    try:
        data = json.loads(res)
        raw_intent = data.get("intent", "SIMPLE_CODE")
        # Normalise through synonym resolution.
        intent = resolve_synonym(raw_intent, custom_verbs)
        return TriageResult(
            intent=intent,
            complexity=data.get("complexity", 5),
        )
    except Exception:
        return TriageResult(intent="SIMPLE_CODE", complexity=5)


def generate_plan(
    fabric: ComputeFabric, task: str, context: str
) -> list[str]:
    """Ask the reasoning role to produce an ordered list of atomic steps."""
    prompt = (
        "You are a Lead Software Architect.\n"
        f"TASK: {task}\n"
        f"CONTEXT: {context}\n\n"
        "Create a strictly sequential JSON list of atomic coding steps.\n"
        'Example: ["Create utils.py", "Implement class in utils.py", "Update main.py"]\n'
        "Return JSON List only."
    )
    res = fabric.execute("reasoning", prompt)
    if not res:
        return [task]

    clean = res.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        return [task]
