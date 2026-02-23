"""MCP tool implementations for AuraRouter.

Each function is a standalone implementation that takes a ComputeFabric
(and optional TriageRouter) and returns a string result.  The @mcp.tool()
decoration and registration happens in server.py.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from aurarouter._logging import get_logger
from aurarouter.routing import analyze_intent, generate_plan
from aurarouter.savings.pricing import _LOCAL_PROVIDERS

if TYPE_CHECKING:
    from aurarouter.fabric import ComputeFabric
    from aurarouter.savings.triage import TriageRouter
    from aurarouter.sessions.manager import SessionManager

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


# ---------------------------------------------------------------------------
# Session tools (registered only when sessions are enabled)
# ---------------------------------------------------------------------------

def register_session_tools(mcp, fabric: ComputeFabric, session_manager: SessionManager) -> None:
    """Register session-related MCP tools (only if sessions enabled)."""

    @mcp.tool()
    def create_session(role: str = "coding") -> str:
        """Create a new stateful session for multi-turn interaction.

        Args:
            role: The primary role for this session.

        Returns:
            JSON with session_id.
        """
        # Get context limit from first model in role chain
        chain = fabric._config.get_role_chain(role)
        context_limit = 0
        if chain:
            model_cfg = fabric._config.get_model_config(chain[0])
            if model_cfg:
                provider = fabric._get_provider(chain[0], model_cfg)
                context_limit = provider.get_context_limit()

        session = session_manager.create_session(role=role, context_limit=context_limit)
        return json.dumps({
            "session_id": session.session_id,
            "context_limit": context_limit,
        })

    @mcp.tool()
    def session_message(session_id: str, message: str, role: str = "") -> str:
        """Send a message in an existing session.

        Args:
            session_id: The session ID from create_session.
            message: The user's message.
            role: Override role (optional, defaults to session's role).

        Returns:
            The model's response text.
        """
        session = session_manager.get_session(session_id)
        if session is None:
            return f"Session {session_id} not found"

        active_role = role or session.metadata.get("active_role", "coding")

        # Check context pressure and condense if needed
        if session_manager.check_pressure(session):
            session = session_manager.condense(session)

        result = fabric.execute_session(
            role=active_role,
            session=session,
            message=message,
            inject_gist=session_manager._auto_gist,
        )

        # Persist updated session
        session_manager._store.save(session)

        # Generate fallback gist if model didn't provide one
        if session_manager._auto_gist and result.gist is None:
            session_manager.generate_fallback_gist(session, result.text, result.model_id)

        return result.text

    @mcp.tool()
    def session_status(session_id: str) -> str:
        """Get the status of a session including token usage and pressure.

        Args:
            session_id: The session ID.

        Returns:
            JSON with session status information.
        """
        session = session_manager.get_session(session_id)
        if session is None:
            return json.dumps({"error": f"Session {session_id} not found"})

        return json.dumps({
            "session_id": session.session_id,
            "message_count": len(session.history),
            "gist_count": len(session.shared_context),
            "token_stats": session.token_stats.to_dict(),
            "pressure": round(session.token_stats.pressure, 3),
            "needs_condensation": session_manager.check_pressure(session),
            "active_role": session.metadata.get("active_role", ""),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        })

    @mcp.tool()
    def list_sessions() -> str:
        """List recent sessions.

        Returns:
            JSON array of session summaries.
        """
        sessions = session_manager.list_sessions(limit=20)
        return json.dumps(sessions)

    @mcp.tool()
    def delete_session(session_id: str) -> str:
        """Delete a session and its history.

        Args:
            session_id: The session ID to delete.

        Returns:
            JSON with deletion result.
        """
        deleted = session_manager.delete_session(session_id)
        return json.dumps({"deleted": deleted})
