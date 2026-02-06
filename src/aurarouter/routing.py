import json

from aurarouter._logging import get_logger
from aurarouter.fabric import ComputeFabric

logger = get_logger("AuraRouter.Routing")


def analyze_intent(fabric: ComputeFabric, task: str) -> str:
    """Classify a task as SIMPLE_CODE or COMPLEX_REASONING via the router role."""
    prompt = (
        'CLASSIFY intent.\n'
        f'Task: "{task}"\n'
        'Options: ["SIMPLE_CODE", "COMPLEX_REASONING"]\n'
        'Return JSON: {"intent": "..."}'
    )
    res = fabric.execute("router", prompt, json_mode=True)
    try:
        return json.loads(res).get("intent", "SIMPLE_CODE")
    except Exception:
        return "SIMPLE_CODE"


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
