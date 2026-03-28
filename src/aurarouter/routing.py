from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from aurarouter._logging import get_logger
from aurarouter.semantic_verbs import resolve_synonym

if TYPE_CHECKING:
    from aurarouter.broker import AnalyzerBid, BrokerResult
    from aurarouter.fabric import ComputeFabric

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
        'Options: ["DIRECT", "SIMPLE_CODE", "COMPLEX_REASONING"]\n'
        f"{roles_hint}"
        'Return JSON: {"intent": "...", "complexity": 5}\n'
        "Where complexity is 1-10 (1=trivial, 10=very complex).\n"
        "Use DIRECT for simple questions, jokes, or single-turn tasks that don't require code or multi-step reasoning."
    )
    res = fabric.execute("router", prompt, json_mode=True)
    try:
        data = json.loads(res.text if res else "")
        raw_intent = data.get("intent", "DIRECT")
        # Normalise through synonym resolution.
        intent = resolve_synonym(raw_intent, custom_verbs)
        return TriageResult(
            intent=intent,
            complexity=data.get("complexity", 1 if intent == "DIRECT" else 5),
        )
    except Exception:
        return TriageResult(intent="DIRECT", complexity=1)


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
    if not res or not res.text:
        return [task]

    clean = res.text.replace("```json", "").replace("```", "").strip()
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
        if not raw or not raw.text:
            return ReviewResult(
                verdict="PASS",
                feedback="Reviewer unavailable",
                correction_hints=[],
            )
        data = json.loads(raw.text)
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
        if raw and raw.text:
            steps = json.loads(raw.text)
            if isinstance(steps, list) and steps:
                return [str(s) for s in steps]
    except Exception:
        pass
    # Fallback: single re-do step with feedback
    return [f"Redo: {task}. Reviewer feedback: {review.feedback}"]


# ---------------------------------------------------------------------------
# Arbiter: collision resolution via the reasoning role
# ---------------------------------------------------------------------------

@dataclass
class ArbiterDecision:
    """Resolved execution plan from the reasoning arbiter."""

    execution_order: list[dict] = field(default_factory=list)
    reasoning: str = ""
    strategy: str = "winner_takes_all"  # "sequential" | "winner_takes_all" | "split"

    @classmethod
    def from_dict(cls, data: dict) -> ArbiterDecision:
        return cls(
            execution_order=data.get("execution_order", []),
            reasoning=data.get("reasoning", ""),
            strategy=data.get("strategy", "winner_takes_all"),
        )


def build_arbiter_prompt(
    user_request: str,
    collisions: list[tuple[AnalyzerBid, AnalyzerBid]],
    file_context: list[dict[str, str]] | None = None,
) -> str:
    """Build a prompt for the reasoning role to resolve routing collisions.

    The prompt includes the user request verbatim, formatted collision pairs
    with analyzer IDs / confidence / claimed files, and an optional
    WORKSPACE FILES section when *file_context* is provided.
    """
    sections: list[str] = []

    sections.append(
        "You are a routing arbiter. Two or more analyzers have claimed "
        "overlapping files with high confidence. Resolve the conflict by "
        "producing a JSON execution plan.\n"
    )

    # User request verbatim
    sections.append(f"USER REQUEST:\n{user_request}\n")

    # Collision pairs
    sections.append("COLLISIONS:")
    for idx, (a, b) in enumerate(collisions, 1):
        shared = sorted(set(a.claimed_files) & set(b.claimed_files))
        sections.append(
            f"  Collision {idx}:\n"
            f"    Analyzer A: {a.analyzer_id} (confidence={a.confidence}, "
            f"files={a.claimed_files})\n"
            f"    Analyzer B: {b.analyzer_id} (confidence={b.confidence}, "
            f"files={b.claimed_files})\n"
            f"    Shared files: {shared}"
        )
    sections.append("")

    # Optional workspace files section
    if file_context:
        sections.append("WORKSPACE FILES:")
        for fc in file_context:
            lang = fc.get("language", "")
            lang_str = f" ({lang})" if lang else ""
            sections.append(f"  - {fc['path']}{lang_str}")
        sections.append("")

    # Required response format
    sections.append(
        "Return JSON only with this structure:\n"
        "{\n"
        '  "execution_order": [\n'
        '    {"analyzer_id": "...", "role": "...", "tasks": [...], "files": [...]}\n'
        "  ],\n"
        '  "reasoning": "brief explanation",\n'
        '  "strategy": "sequential" | "winner_takes_all" | "split"\n'
        "}"
    )

    return "\n".join(sections)


def resolve_collisions(
    fabric: ComputeFabric,
    user_request: str,
    broker_result: BrokerResult,
    file_context: list[dict[str, str]] | None = None,
) -> ArbiterDecision:
    """Invoke the reasoning role to resolve routing collisions.

    Calls ``fabric.execute("reasoning", ..., json_mode=True)`` with a
    prompt built from the collisions.  On parse failure, falls back to the
    highest-confidence bid (fail-safe).

    Appends trace entries to ``broker_result.execution_trace`` for
    collision, resolution, and fallback events.
    """
    broker_result.execution_trace.append(
        f"Arbiter: resolving {len(broker_result.collisions)} collision(s)"
    )

    prompt = build_arbiter_prompt(user_request, broker_result.collisions, file_context)

    try:
        raw = fabric.execute("reasoning", prompt, json_mode=True)
        text = raw.text if hasattr(raw, "text") else (raw if isinstance(raw, str) else "")
        if not text:
            raise ValueError("Empty response from reasoning role")

        data = json.loads(text)
        decision = ArbiterDecision.from_dict(data)
        broker_result.execution_trace.append(
            f"Arbiter: resolved — strategy={decision.strategy}, "
            f"steps={len(decision.execution_order)}"
        )
        return decision

    except Exception as exc:
        logger.warning("Arbiter failed (%s); falling back to highest-confidence bid", exc)
        broker_result.execution_trace.append(
            f"Arbiter: fallback — parse failure ({exc})"
        )

        # Fail-safe: pick the highest-confidence bid
        best = max(broker_result.bids, key=lambda b: b.confidence)
        return ArbiterDecision(
            execution_order=[{
                "analyzer_id": best.analyzer_id,
                "role": best.role,
                "tasks": best.proposed_tasks,
                "files": best.claimed_files,
            }],
            reasoning=f"Fallback: selected highest-confidence bid from {best.analyzer_id}",
            strategy="winner_takes_all",
        )
