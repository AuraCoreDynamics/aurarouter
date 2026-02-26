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


@dataclass
class ReviewResult:
    """Structured verdict from the reviewer role."""

    verdict: str  # "PASS" or "FAIL"
    feedback: str  # Human-readable assessment
    correction_hints: list[str]  # Actionable suggestions for correction

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "feedback": self.feedback,
            "correction_hints": self.correction_hints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewResult":
        return cls(
            verdict=data.get("verdict", "PASS"),
            feedback=data.get("feedback", ""),
            correction_hints=data.get("correction_hints", []),
        )


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


def review_output(
    fabric: ComputeFabric,
    task: str,
    output: str,
    iteration: int = 1,
) -> ReviewResult:
    """Send output to the reviewer role for quality assessment.

    Returns a structured ReviewResult. If the reviewer role is not
    configured or the call fails, returns a PASS verdict (fail-open).
    """
    review_prompt = (
        "You are a senior code reviewer. Assess the following output "
        "against the original task.\n\n"
        f"ORIGINAL TASK:\n{task}\n\n"
        f"OUTPUT (iteration {iteration}):\n{output}\n\n"
        "Return JSON only:\n"
        '{"verdict": "PASS" or "FAIL", '
        '"feedback": "brief assessment", '
        '"correction_hints": ["actionable hint 1", ...]}\n'
        "If the output correctly addresses the task, verdict is PASS."
    )
    try:
        raw = fabric.execute("reviewer", review_prompt, json_mode=True)
        if not raw:
            return ReviewResult(
                verdict="PASS",
                feedback="Reviewer unavailable",
                correction_hints=[],
            )
        data = json.loads(raw)
        return ReviewResult.from_dict(data)
    except Exception as exc:
        logger.warning("Review failed (fail-open): %s", exc)
        return ReviewResult(
            verdict="PASS",
            feedback=f"Review error: {exc}",
            correction_hints=[],
        )


def generate_correction_plan(
    fabric: ComputeFabric,
    task: str,
    output: str,
    review: ReviewResult,
) -> list[str]:
    """Generate a correction plan based on reviewer feedback.

    Uses the reasoning role to produce a list of correction steps.
    Falls back to a single re-execution step if planning fails.
    """
    hints = "\n".join(f"- {h}" for h in review.correction_hints) or "No specific hints."
    correction_prompt = (
        "You are a Lead Software Architect.\n"
        f"ORIGINAL TASK:\n{task}\n\n"
        f"PREVIOUS OUTPUT (rejected by reviewer):\n{output}\n\n"
        f"REVIEWER FEEDBACK:\n{review.feedback}\n\n"
        f"CORRECTION HINTS:\n{hints}\n\n"
        "Create a strictly sequential JSON list of atomic correction steps.\n"
        "Return JSON List only."
    )
    try:
        raw = fabric.execute("reasoning", correction_prompt, json_mode=True)
        if raw:
            steps = json.loads(raw)
            if isinstance(steps, list) and steps:
                return [str(s) for s in steps]
    except Exception:
        pass
    # Fallback: single re-do step with feedback
    return [f"Redo: {task}. Reviewer feedback: {review.feedback}"]
