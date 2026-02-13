"""MCP tool implementations for AuraRouter.

Each function is a standalone implementation that takes a ComputeFabric
(and optional TriageRouter) and returns a string result.  The @mcp.tool()
decoration and registration happens in server.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from aurarouter._logging import get_logger
from aurarouter.routing import analyze_intent, generate_plan
from aurarouter.savings.pricing import _LOCAL_PROVIDERS

if TYPE_CHECKING:
    from aurarouter.fabric import ComputeFabric
    from aurarouter.savings.triage import TriageRouter

logger = get_logger("AuraRouter.MCPTools")


# ---------------------------------------------------------------------------
# route_task
# ---------------------------------------------------------------------------

def route_task(
    fabric: ComputeFabric,
    triage_router: Optional[TriageRouter],
    *,
    task: str,
    context: str = "",
    format: str = "text",
) -> str:
    """Route a task to local or specialized AI models with automatic fallback."""
    triage = analyze_intent(fabric, task)
    intent = triage.intent
    complexity = triage.complexity
    logger.info(f"[route_task] Intent: {intent}  Complexity: {complexity}")

    role = "coding"
    if triage_router is not None:
        role = triage_router.select_role(complexity)
        logger.info(f"[route_task] Triage selected role: {role}")

    full_prompt = f"TASK: {task}"
    if context:
        full_prompt += f"\nCONTEXT: {context}"
    if format != "text":
        full_prompt += f"\nFORMAT: {format}"

    if intent == "SIMPLE_CODE":
        full_prompt += "\nRESPOND WITH OUTPUT ONLY."
        return fabric.execute(role, full_prompt) or "Error: All models failed."

    # Complex path
    logger.info("[route_task] Complex task detected. Generating plan...")
    plan = generate_plan(fabric, task, context)
    logger.info(f"[route_task] Plan: {len(plan)} steps")

    output: list[str] = []
    for i, step in enumerate(plan):
        logger.info(f"[route_task] Step {i + 1}: {step}")
        step_prompt = (
            f"GOAL: {step}\n"
            f"CONTEXT: {context}\n"
            f"PREVIOUS_OUTPUT: {output}\n"
            "Return ONLY the requested output."
        )
        if format != "text":
            step_prompt += f"\nFORMAT: {format}"
        result = fabric.execute(role, step_prompt)
        if result:
            output.append(f"\n# --- Step {i + 1}: {step} ---\n{result}")
        else:
            output.append(f"\n# Step {i + 1} Failed.")

    return "\n".join(output)


# ---------------------------------------------------------------------------
# local_inference
# ---------------------------------------------------------------------------

def local_inference(
    fabric: ComputeFabric,
    *,
    prompt: str,
    context: str = "",
) -> str:
    """Execute a prompt on local/private AI models without cloud API calls."""
    full_prompt = prompt
    if context:
        full_prompt = f"{prompt}\n\nCONTEXT:\n{context}"

    # Filter the coding role chain to local providers only.
    chain = fabric._config.get_role_chain("coding")
    local_chain = [
        model_id
        for model_id in chain
        if fabric._config.get_model_config(model_id).get("provider") in _LOCAL_PROVIDERS
    ]

    if not local_chain:
        return "Error: No local models configured. Add Ollama or llama.cpp models to the 'coding' role."

    result = fabric.execute("coding", full_prompt, chain_override=local_chain)
    return result or "Error: All local models failed to generate a response."


# ---------------------------------------------------------------------------
# generate_code
# ---------------------------------------------------------------------------

def generate_code(
    fabric: ComputeFabric,
    triage_router: Optional[TriageRouter],
    *,
    task_description: str,
    file_context: str = "",
    language: str = "python",
) -> str:
    """Multi-step code generation with automatic planning."""
    triage = analyze_intent(fabric, task_description)
    intent = triage.intent
    complexity = triage.complexity
    logger.info(f"[generate_code] Intent: {intent}  Complexity: {complexity}")

    coding_role = "coding"
    if triage_router is not None:
        coding_role = triage_router.select_role(complexity)
        logger.info(f"[generate_code] Triage selected role: {coding_role}")

    if intent == "SIMPLE_CODE":
        prompt = (
            f"TASK: {task_description}\n"
            f"LANG: {language}\n"
            f"CONTEXT: {file_context}\n"
            "CODE ONLY."
        )
        return fabric.execute(coding_role, prompt) or "Error: Generation failed."

    # Complex path
    logger.info("[generate_code] Complexity detected. Generating plan...")
    plan = generate_plan(fabric, task_description, file_context)
    logger.info(f"[generate_code] Plan: {len(plan)} steps")

    output: list[str] = []
    for i, step in enumerate(plan):
        logger.info(f"[generate_code] Step {i + 1}: {step}")
        prompt = (
            f"GOAL: {step}\n"
            f"LANG: {language}\n"
            f"CONTEXT: {file_context}\n"
            f"PREVIOUS_CODE: {output}\n"
            "Return ONLY valid code."
        )
        code = fabric.execute(coding_role, prompt)
        if code:
            output.append(f"\n# --- Step {i + 1}: {step} ---\n{code}")
        else:
            output.append(f"\n# Step {i + 1} Failed.")

    return "\n".join(output)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

def compare_models(
    fabric: ComputeFabric,
    *,
    prompt: str,
    models: str = "",
) -> str:
    """Run a prompt across multiple models and return all responses."""
    model_ids = None
    if models.strip():
        model_ids = [m.strip() for m in models.split(",") if m.strip()]

    results = fabric.execute_all("coding", prompt, model_ids=model_ids)

    if not results:
        return "Error: No models available for comparison."

    output_parts: list[str] = []
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        header = (
            f"=== {r['model_id']} ({r['provider']}) [{status}] "
            f"({r['elapsed_s']}s, {r['input_tokens']}in/{r['output_tokens']}out) ==="
        )
        output_parts.append(header)
        output_parts.append(r["text"])
        output_parts.append("")

    return "\n".join(output_parts)
