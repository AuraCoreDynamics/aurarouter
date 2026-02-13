import json
from dataclasses import dataclass

from aurarouter._logging import get_logger
from aurarouter.fabric import ComputeFabric

logger = get_logger("AuraRouter.Routing")


@dataclass
class TriageResult:
    """Result of intent analysis including complexity scoring."""

    intent: str
    complexity: int


def analyze_intent(fabric: ComputeFabric, task: str) -> TriageResult:
    """Classify a task and estimate complexity via the router role.

    Returns a ``TriageResult`` with *intent* (``SIMPLE_CODE`` or
    ``COMPLEX_REASONING``) and *complexity* (1–10, default 5).

    Backwards compatible: if the model returns the old format without a
    ``complexity`` key, the score defaults to 5.
    """
    prompt = (
        'CLASSIFY intent.\n'
        f'Task: "{task}"\n'
        'Options: ["SIMPLE_CODE", "COMPLEX_REASONING"]\n'
        'Return JSON: {"intent": "...", "complexity": 5}\n'
        'Where complexity is 1-10 (1=trivial, 10=very complex).'
    )
    res = fabric.execute("router", prompt, json_mode=True)
    try:
        data = json.loads(res)
        return TriageResult(
            intent=data.get("intent", "SIMPLE_CODE"),
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
