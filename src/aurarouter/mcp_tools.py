"""MCP tool implementations for AuraRouter.

Each function is a standalone implementation that takes a ComputeFabric
(and optional TriageRouter) and returns a string result.  The @mcp.tool()
decoration and registration happens in server.py.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from aurarouter._logging import get_logger
from aurarouter.routing import (
    analyze_intent,
    generate_correction_plan,
    generate_plan,
    review_output,
)
from aurarouter.savings.pricing import is_cloud_tier

if TYPE_CHECKING:
    from aurarouter.fabric import ComputeFabric
    from aurarouter.savings.triage import TriageRouter
    from aurarouter.sessions.manager import SessionManager
    from aurarouter.mcp_client.registry import McpClientRegistry

logger = get_logger("AuraRouter.MCPTools")


# ---------------------------------------------------------------------------
# Review loop helper (shared by route_task and generate_code)
# ---------------------------------------------------------------------------

def _apply_review_loop(
    fabric: ComputeFabric,
    task: str,
    context: str,
    output: str,
    role: str,
) -> str:
    """Run the review-correct loop if a reviewer role is configured.

    Returns the (possibly corrected) output. Skips the loop entirely when
    no reviewer chain is configured or ``max_review_iterations`` is 0.
    """
    max_iterations = fabric._config.get_max_review_iterations()
    reviewer_chain = fabric._config.get_role_chain("reviewer")

    if max_iterations <= 0 or not reviewer_chain:
        return output

    for iteration in range(1, max_iterations + 1):
        review = review_output(fabric, task, output, iteration=iteration)
        logger.info(
            "[review] Iteration %d: %s", iteration, review.verdict,
        )
        if review.verdict.upper() == "PASS":
            break
        if iteration == max_iterations:
            logger.warning(
                "[review] Max iterations (%d) reached. Returning best output.",
                max_iterations,
            )
            break
        # Generate and execute correction plan
        correction_steps = generate_correction_plan(fabric, task, output, review)
        logger.info("[review] Correction plan: %d steps", len(correction_steps))
        corrected: list[str] = []
        for i, step in enumerate(correction_steps):
            step_prompt = (
                f"GOAL: {step}\nCONTEXT: {context}\n"
                f"PREVIOUS_OUTPUT:\n{output}\n"
                f"REVIEWER_FEEDBACK: {review.feedback}"
            )
            chunk = fabric.execute(role, step_prompt)
            corrected.append(chunk or f"\n# Correction Step {i + 1} Failed.")
        output = "\n".join(corrected)

    return output


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
        output = fabric.execute(role, full_prompt) or "Error: All models failed."
    else:
        # Complex path
        logger.info("[route_task] Complex task detected. Generating plan...")
        plan = generate_plan(fabric, task, context)
        logger.info(f"[route_task] Plan: {len(plan)} steps")

        parts: list[str] = []
        for i, step in enumerate(plan):
            logger.info(f"[route_task] Step {i + 1}: {step}")
            step_prompt = (
                f"GOAL: {step}\n"
                f"CONTEXT: {context}\n"
                f"PREVIOUS_OUTPUT: {parts}\n"
                "Return ONLY the requested output."
            )
            if format != "text":
                step_prompt += f"\nFORMAT: {format}"
            result = fabric.execute(role, step_prompt)
            if result:
                parts.append(f"\n# --- Step {i + 1}: {step} ---\n{result}")
            else:
                parts.append(f"\n# Step {i + 1} Failed.")
        output = "\n".join(parts)

    # --- Review loop (closed-loop execution) ---
    output = _apply_review_loop(fabric, task, context, output, role)

    return output


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

    # Filter the coding role chain to local (non-cloud) models only.
    chain = fabric._config.get_role_chain("coding")
    local_chain = [
        model_id
        for model_id in chain
        if not is_cloud_tier(
            fabric._config.get_model_hosting_tier(model_id),
            fabric._config.get_model_config(model_id).get("provider", ""),
        )
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
        output = fabric.execute(coding_role, prompt) or "Error: Generation failed."
    else:
        # Complex path
        logger.info("[generate_code] Complexity detected. Generating plan...")
        plan = generate_plan(fabric, task_description, file_context)
        logger.info(f"[generate_code] Plan: {len(plan)} steps")

        parts: list[str] = []
        for i, step in enumerate(plan):
            logger.info(f"[generate_code] Step {i + 1}: {step}")
            prompt = (
                f"GOAL: {step}\n"
                f"LANG: {language}\n"
                f"CONTEXT: {file_context}\n"
                f"PREVIOUS_CODE: {parts}\n"
                "Return ONLY valid code."
            )
            code = fabric.execute(coding_role, prompt)
            if code:
                parts.append(f"\n# --- Step {i + 1}: {step} ---\n{code}")
            else:
                parts.append(f"\n# Step {i + 1} Failed.")
        output = "\n".join(parts)

    # --- Review loop (closed-loop execution) ---
    output = _apply_review_loop(
        fabric, task_description, file_context, output, coding_role,
    )

    return output


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
# list_models
# ---------------------------------------------------------------------------

def list_models(fabric: ComputeFabric) -> str:
    """List all configured model IDs with their provider and endpoint info."""
    config = fabric._config
    models_info: list[dict] = []

    for model_id in config.get_all_model_ids():
        cfg = config.get_model_config(model_id)
        models_info.append({
            "model_id": model_id,
            "provider": cfg.get("provider", ""),
            "endpoint": cfg.get("endpoint", ""),
            "model_name": cfg.get("model_name", ""),
            "tags": cfg.get("tags", []),
        })

    return json.dumps(models_info, indent=2)


# ---------------------------------------------------------------------------
# list_assets
# ---------------------------------------------------------------------------

def list_assets() -> str:
    """List all physical GGUF model files in the local asset storage.

    Returns JSON array of asset entries with:
    - repo: HuggingFace repository ID
    - filename: GGUF file name
    - path: Full filesystem path
    - size_bytes: File size in bytes
    - downloaded_at: ISO timestamp
    - gguf_metadata: Optional metadata dict (if available)
    """
    try:
        from aurarouter.models.file_storage import FileModelStorage

        storage = FileModelStorage()
        entries = storage.list_models()
        return json.dumps(entries, indent=2)
    except Exception as exc:
        logger.error(f"[list_assets] Failed to list assets: {exc}")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# register_asset
# ---------------------------------------------------------------------------

def register_asset(
    fabric: ComputeFabric,
    config: "ConfigLoader",
    *,
    model_id: str,
    file_path: str,
    repo: str = "local",
    tags: str = "",
    cost_per_1m_input: float = -1.0,
    cost_per_1m_output: float = -1.0,
    hosting_tier: str = "",
) -> str:
    """Register a new GGUF model file and add it to routing config.

    Parameters:
    - fabric: Live ComputeFabric instance for immediate routing updates
    - config: Live ConfigLoader instance for config mutation
    - model_id: Unique identifier for routing (e.g., "my-fine-tuned-qwen")
    - file_path: Absolute path to the .gguf file
    - repo: HuggingFace repo ID or "local" (default: "local")
    - tags: Comma-separated capability tags (e.g., "coding,local,private")
    - cost_per_1m_input: Cost per 1M input tokens in USD (-1.0 = not set, uses fallback pricing)
    - cost_per_1m_output: Cost per 1M output tokens in USD (-1.0 = not set, uses fallback pricing)
    - hosting_tier: Hosting classification ("on-prem", "cloud", "dedicated-tenant"; empty = infer from provider)

    Returns:
    - JSON with {"success": true, "model_id": "...", "path": "...", "roles_joined": [...],
      "cost_per_1m_input": ..., "cost_per_1m_output": ..., "hosting_tier": ...}
    - Or {"error": "..."} on failure

    Side effects:
    - Registers file with FileModelStorage
    - Adds model to auraconfig.yaml with llamacpp provider
    - Auto-joins role chains when tags match role names or semantic verb synonyms
    - Updates the live ComputeFabric for immediate routing
    """
    try:
        from pathlib import Path

        from aurarouter.models.file_storage import FileModelStorage

        # 1. Validate file_path exists and is a .gguf file
        p = Path(file_path)
        if not p.is_file():
            return json.dumps({"error": f"File not found: {file_path}"})
        if p.suffix.lower() != ".gguf":
            return json.dumps({"error": f"Invalid file type '{p.suffix}'. Only .gguf files are supported."})

        # 2. Check if model_id already exists
        if config.get_model_config(model_id):
            return json.dumps({"error": f"Model ID '{model_id}' already exists in config."})

        # 3. Parse tags
        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        # 4. Extract GGUF metadata (non-fatal on failure)
        metadata = None
        try:
            from aurarouter.tuning import extract_gguf_metadata
            metadata = extract_gguf_metadata(str(p))
        except Exception as exc:
            logger.debug(f"[register_asset] GGUF metadata extraction failed: {exc}")

        # 5. Create model config
        model_config: dict = {
            "provider": "llamacpp",
            "model_path": file_path,
            "tags": tags_list,
        }

        # Cost fields (only set if explicitly provided)
        if cost_per_1m_input >= 0:
            model_config["cost_per_1m_input"] = cost_per_1m_input
        if cost_per_1m_output >= 0:
            model_config["cost_per_1m_output"] = cost_per_1m_output

        # Hosting tier (only set if explicitly provided)
        if hosting_tier:
            model_config["hosting_tier"] = hosting_tier

        # Inject metadata-derived parameters if available
        if metadata:
            if metadata.get("context_length"):
                model_config.setdefault("parameters", {})["n_ctx"] = metadata["context_length"]

        # 6. Add model to config
        config.set_model(model_id, model_config)

        # 7. Tag-to-role auto-integration
        roles_joined: list[str] = []
        known_roles = config.get_all_roles()

        # Build synonym reverse lookup from semantic verbs
        semantic_verbs = config.get_semantic_verbs()
        synonym_to_role: dict[str, str] = {}
        for role, synonyms in semantic_verbs.items():
            for syn in synonyms:
                synonym_to_role[syn.lower()] = role

        for tag in tags_list:
            tag_lower = tag.lower()
            matched_role = None
            if tag_lower in known_roles:
                matched_role = tag_lower
            elif tag_lower in synonym_to_role:
                matched_role = synonym_to_role[tag_lower]

            if matched_role and model_id not in config.get_role_chain(matched_role):
                chain = config.get_role_chain(matched_role)
                config.set_role_chain(matched_role, chain + [model_id])
                roles_joined.append(matched_role)

        # 8. Save config and update live fabric
        config.save()
        fabric.update_config(config)

        # 9. Register with FileModelStorage
        storage = FileModelStorage()
        storage.register(repo, p.name, p, metadata=metadata)

        # 10. Return success
        return json.dumps({
            "success": True,
            "model_id": model_id,
            "path": file_path,
            "roles_joined": roles_joined,
            "cost_per_1m_input": model_config.get("cost_per_1m_input"),
            "cost_per_1m_output": model_config.get("cost_per_1m_output"),
            "hosting_tier": model_config.get("hosting_tier"),
        })
    except Exception as exc:
        logger.error(f"[register_asset] Failed to register asset: {exc}")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# unregister_asset
# ---------------------------------------------------------------------------

def unregister_asset(
    fabric: ComputeFabric,
    config: "ConfigLoader",
    *,
    model_id: str,
    remove_from_roles: bool = True,
    delete_file: bool = False,
) -> str:
    """Unregister a model from routing config and optionally delete the file.

    Parameters:
    - fabric: Live ComputeFabric instance for immediate routing updates
    - config: Live ConfigLoader instance for config mutation
    - model_id: The model ID to remove
    - remove_from_roles: Remove from all role chains (default: True)
    - delete_file: Delete the physical GGUF file (default: False)

    Returns:
    - JSON with {"success": true, "model_id": "...", "roles_left": [...], "file_deleted": ...}
    - Or {"error": "..."} on failure
    """
    try:
        from pathlib import Path

        from aurarouter.models.file_storage import FileModelStorage

        # 1. Verify model exists
        model_cfg = config.get_model_config(model_id)
        if not model_cfg:
            return json.dumps({"error": f"Model ID '{model_id}' not found in config."})

        # 2. Remove from role chains if requested
        roles_left: list[str] = []
        if remove_from_roles:
            for role in config.get_all_roles():
                chain = config.get_role_chain(role)
                if model_id in chain:
                    config.set_role_chain(role, [m for m in chain if m != model_id])
                    roles_left.append(role)

        # 3. Get model_path before removing from config
        model_path = model_cfg.get("model_path", "")

        # 4. Remove from config
        config.remove_model(model_id)

        # 5. Save config and update live fabric
        config.save()
        fabric.update_config(config)

        # 6. Remove from FileModelStorage
        file_deleted = False
        if model_path:
            p = Path(model_path)
            storage = FileModelStorage()
            removed = storage.remove(p.name, delete_file=delete_file)
            file_deleted = delete_file and removed

        # 7. Return success
        return json.dumps({
            "success": True,
            "model_id": model_id,
            "roles_left": roles_left,
            "file_deleted": file_deleted,
        })
    except Exception as exc:
        logger.error(f"[unregister_asset] Failed to unregister asset: {exc}")
        return json.dumps({"error": str(exc)})


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


# ---------------------------------------------------------------------------
# Grid service tools (registered only when grid services are configured)
# ---------------------------------------------------------------------------

def register_grid_tools(
    mcp,
    fabric: ComputeFabric,
    registry: McpClientRegistry,
) -> None:
    """Register tools discovered from external grid service clients."""

    @mcp.tool()
    def list_grid_services() -> str:
        """List connected external grid services and their capabilities."""
        services: list[dict] = []
        for name, client in registry.get_clients().items():
            services.append({
                "name": name,
                "url": client.base_url,
                "connected": client.connected,
                "capabilities": sorted(client.get_capabilities()),
                "tool_count": len(client.get_tools()),
                "model_count": len(client.get_models()),
            })
        return json.dumps(services, indent=2)

    @mcp.tool()
    def list_remote_tools() -> str:
        """List all tools available from connected grid services."""
        tools = registry.get_all_remote_tools()
        return json.dumps(tools, indent=2)

    @mcp.tool()
    def call_remote_tool(service_name: str, tool_name: str, arguments: str = "{}") -> str:
        """Call a tool on a remote grid service.

        Args:
            service_name: Name of the grid service.
            tool_name: Name of the tool to invoke.
            arguments: JSON string of tool arguments.
        """
        clients = registry.get_clients()
        client = clients.get(service_name)
        if client is None:
            return json.dumps({"error": f"Service '{service_name}' not found"})
        if not client.connected:
            return json.dumps({"error": f"Service '{service_name}' is not connected"})

        try:
            kwargs = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON arguments: {exc}"})

        try:
            result = client.call_tool(tool_name, **kwargs)
            return json.dumps(result, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
