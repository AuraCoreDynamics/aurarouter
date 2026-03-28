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

if TYPE_CHECKING:
    from aurarouter.config import ConfigLoader
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
    max_iterations = fabric.get_max_review_iterations()
    reviewer_chain = fabric.config.get_role_chain("reviewer")

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
            chunk_result = fabric.execute(role, step_prompt)
            chunk = chunk_result.text if chunk_result else ""
            corrected.append(chunk or f"\n# Correction Step {i + 1} Failed.")
        output = "\n".join(corrected)

    return output


# ---------------------------------------------------------------------------
# Remote analyzer helper
# ---------------------------------------------------------------------------

async def _call_remote_analyzer(
    endpoint: str,
    tool_name: str,
    task: str,
    context: str | None,
) -> dict | None:
    """Call a remote analyzer via MCP JSON-RPC and return the routing decision."""
    import httpx

    payload: dict = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": {"prompt": task},
        },
        "id": 1,
    }
    if context:
        payload["params"]["arguments"]["context"] = context

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(endpoint, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("result", {})
            if isinstance(result, str):
                result = json.loads(result)
            return result
    return None


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
    config: Optional[ConfigLoader] = None,
    options: Optional[dict] = None,
) -> str:
    """Route a task to local or specialized AI models with automatic fallback.

    When *config* is provided the active analyzer is consulted first.  If the
    active analyzer is a remote MCP endpoint the request is delegated there
    and the returned ``ranked_models`` list drives model selection.  On any
    failure the built-in intent-triage path is used as fallback.

    When *options* contains ``routing_hints`` (a list of language/domain
    strings), the federated broker is invoked to collect bids from all
    registered analyzers before falling back to the built-in pipeline.
    """
    options = options or {}

    # --- Federated broker path: when routing hints are present ---
    routing_hints = options.get("routing_hints")
    if config is not None and routing_hints:
        try:
            import asyncio
            from aurarouter.broker import broadcast_to_analyzers, merge_bids

            loop = asyncio.new_event_loop()
            try:
                bids = loop.run_until_complete(
                    broadcast_to_analyzers(config, task, options)
                )
            finally:
                loop.close()

            broker_result = merge_bids(bids, routing_hints=routing_hints)
            logger.info(
                "[route_task] Broker returned %d bids, %d collisions, mismatch=%s",
                len(broker_result.bids),
                len(broker_result.collisions),
                broker_result.mismatch,
            )

            if broker_result.collisions:
                from aurarouter.routing import resolve_collisions
                file_ctx = [
                    {"path": fc["path"], "language": fc.get("language", "")}
                    for fc in (options or {}).get("file_constraints", [])
                ] if (options or {}).get("file_constraints") else None
                decision = resolve_collisions(fabric, task, broker_result, file_context=file_ctx)
                logger.info(
                    "[route_task] Arbiter resolved collisions: strategy=%s, steps=%d",
                    decision.strategy,
                    len(decision.execution_order),
                )
                # Execute decision.execution_order through fabric
                results = []
                for step in decision.execution_order:
                    role = step.get("role", "coding")
                    step_prompt = f"TASK: {task}\nFOCUS: {json.dumps(step.get('tasks', []))}"
                    output = fabric.execute(role, step_prompt)
                    if output:
                        text = output.text if hasattr(output, "text") else str(output)
                        results.append(text)
                if results:
                    return "\n\n".join(results)
            elif broker_result.merged_plan and not broker_result.mismatch:
                # Use the highest-confidence bid's role for execution
                top = broker_result.merged_plan[0]
                role = top.get("role", "coding")
                full_prompt = f"TASK: {task}"
                if context:
                    full_prompt += f"\nCONTEXT: {context}"
                result = fabric.execute(role, full_prompt)
                if result and result.text:
                    return result.text
                # Fall through on execution failure
        except Exception:
            logger.debug(
                "Federated broker failed; falling back to built-in pipeline",
                exc_info=True,
            )

    # --- Check active analyzer when config is available ---
    if config is not None:
        active_id = config.get_active_analyzer()
        if active_id and active_id != "aurarouter-default":
            analyzer_data = config.catalog_get(active_id)
            if analyzer_data and analyzer_data.get("mcp_endpoint"):
                try:
                    import asyncio

                    result = asyncio.get_event_loop().run_until_complete(
                        _call_remote_analyzer(
                            analyzer_data["mcp_endpoint"],
                            analyzer_data.get("mcp_tool_name", ""),
                            task,
                            context or None,
                        )
                    )
                    if result and result.get("ranked_models"):
                        role = result.get("role", "coding")
                        for _model_id in result["ranked_models"]:
                            output = fabric.execute(role, f"TASK: {task}")
                            text = output.text if hasattr(output, "text") else output
                            if text:
                                return text
                except Exception:
                    logger.debug(
                        "Remote analyzer '%s' failed; falling back to built-in",
                        active_id,
                    )

    # --- Built-in analyzer / legacy behaviour ---
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

    if intent in ("SIMPLE_CODE", "DIRECT"):
        full_prompt += "\nRESPOND WITH OUTPUT ONLY."
        result = fabric.execute(role, full_prompt)
        output = result.text if result else "Error: All models failed."
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
            step_result = fabric.execute(role, step_prompt)
            step_text = step_result.text if step_result else ""
            if step_text:
                parts.append(f"\n# --- Step {i + 1}: {step} ---\n{step_text}")
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
    local_chain = fabric.get_local_chain("coding")

    if not local_chain:
        return "Error: No local models configured. Add Ollama or llama.cpp models to the 'coding' role."

    result = fabric.execute("coding", full_prompt, chain_override=local_chain)
    return result.text if result else "Error: All local models failed to generate a response."


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
        result = fabric.execute(coding_role, prompt)
        output = result.text if result else "Error: Generation failed."
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
            code_result = fabric.execute(coding_role, prompt)
            code_text = code_result.text if code_result else ""
            if code_text:
                parts.append(f"\n# --- Step {i + 1}: {step} ---\n{code_text}")
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
    config = fabric.config
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
        roles_joined = config.auto_join_roles(model_id, tags_list)

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
# register_remote_asset
# ---------------------------------------------------------------------------

_VALID_HOSTING_TIERS = frozenset({"on-prem", "cloud", "dedicated-tenant"})


def register_remote_asset(
    fabric: ComputeFabric,
    config: "ConfigLoader",
    *,
    model_id: str,
    endpoint_url: str,
    provider: str = "openapi",
    tags: str = "",
    capabilities: str = "",
    context_window: int = 0,
    cost_per_1m_input: float = -1.0,
    cost_per_1m_output: float = -1.0,
    hosting_tier: str = "",
    node_id: str = "",
) -> str:
    """Register a remote model endpoint for routing (no local file required).

    Parameters:
    - fabric: Live ComputeFabric instance for immediate routing updates
    - config: Live ConfigLoader instance for config mutation
    - model_id: Unique identifier for routing (e.g., "xlm/mistral-7b")
    - endpoint_url: URL of the remote inference endpoint
    - provider: Provider backend type (default: "openapi")
    - tags: Comma-separated capability tags (e.g., "coding,reasoning")
    - capabilities: Comma-separated capability list (e.g., "code,chat")
    - context_window: Maximum context window in tokens (0 = not set)
    - cost_per_1m_input: Cost per 1M input tokens in USD (-1.0 = not set)
    - cost_per_1m_output: Cost per 1M output tokens in USD (-1.0 = not set)
    - hosting_tier: Hosting classification ("on-prem", "cloud", "dedicated-tenant"; empty = infer)
    - node_id: AuraGrid node identifier hosting this model (optional)

    Returns:
    - JSON with {"success": true, "model_id": "...", "endpoint": "...", "roles_joined": [...], ...}
    - Or {"error": "..."} on failure
    """
    try:
        # 1. Validate required fields
        if not model_id or not model_id.strip():
            return json.dumps({"error": "model_id is required."})
        if not endpoint_url or not endpoint_url.strip():
            return json.dumps({"error": "endpoint_url is required."})

        # 2. Validate hosting_tier if provided
        if hosting_tier and hosting_tier not in _VALID_HOSTING_TIERS:
            return json.dumps({
                "error": f"Invalid hosting_tier '{hosting_tier}'. "
                         f"Must be one of: {', '.join(sorted(_VALID_HOSTING_TIERS))}"
            })

        # 3. Check if model_id already exists
        if config.get_model_config(model_id):
            return json.dumps({"error": f"Model ID '{model_id}' already exists in config."})

        # 4. Parse tags and capabilities
        tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        caps_list = [c.strip() for c in capabilities.split(",") if c.strip()] if capabilities else []

        # 5. Create model config
        model_config: dict = {
            "provider": provider,
            "endpoint": endpoint_url,
            "model_name": model_id,
            "tags": tags_list,
        }

        if caps_list:
            model_config["capabilities"] = caps_list

        if context_window > 0:
            model_config.setdefault("parameters", {})["n_ctx"] = context_window

        # Cost fields (only set if explicitly provided)
        if cost_per_1m_input >= 0:
            model_config["cost_per_1m_input"] = cost_per_1m_input
        if cost_per_1m_output >= 0:
            model_config["cost_per_1m_output"] = cost_per_1m_output

        # Hosting tier (only set if explicitly provided)
        if hosting_tier:
            model_config["hosting_tier"] = hosting_tier

        # Node ID
        if node_id:
            model_config["node_id"] = node_id

        # 6. Add model to config
        config.set_model(model_id, model_config)

        # 7. Tag-to-role auto-integration
        roles_joined = config.auto_join_roles(model_id, tags_list)

        # 8. Save config and update live fabric
        config.save()
        fabric.update_config(config)

        # 9. Return success
        return json.dumps({
            "success": True,
            "model_id": model_id,
            "endpoint": endpoint_url,
            "provider": provider,
            "roles_joined": roles_joined,
            "cost_per_1m_input": model_config.get("cost_per_1m_input"),
            "cost_per_1m_output": model_config.get("cost_per_1m_output"),
            "hosting_tier": model_config.get("hosting_tier"),
            "node_id": node_id or None,
        })
    except Exception as exc:
        logger.error(f"[register_remote_asset] Failed to register remote asset: {exc}")
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
        chain = fabric.config.get_role_chain(role)
        context_limit = 0
        if chain:
            context_limit = fabric.get_context_limit(chain[0])

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

        if session_manager.check_pressure(session):
            session = session_manager.condense(session)

        result = session_manager.send_message(session, message, fabric, role=role)
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


# ---------------------------------------------------------------------------
# Catalog artifact tools
# ---------------------------------------------------------------------------

def catalog_list_artifacts(config: "ConfigLoader", kind: str | None = None) -> str:
    """List catalog artifacts, optionally filtered by kind.

    Returns JSON array of artifact dicts, each enriched with artifact_id.
    """
    from aurarouter.catalog_model import CatalogArtifact

    ids = config.catalog_list(kind=kind or None)
    artifacts: list[dict] = []
    for aid in ids:
        data = config.catalog_get(aid)
        if data is not None:
            entry = dict(data)
            entry["artifact_id"] = aid
            artifacts.append(entry)
    return json.dumps(artifacts, indent=2)


def catalog_get_artifact(config: "ConfigLoader", artifact_id: str) -> str:
    """Get a single catalog artifact by ID. Returns JSON."""
    data = config.catalog_get(artifact_id)
    if data is None:
        return json.dumps({"error": f"Artifact '{artifact_id}' not found"})
    result = dict(data)
    result["artifact_id"] = artifact_id
    return json.dumps(result, indent=2)


def catalog_register_artifact(
    config: "ConfigLoader",
    artifact_id: str,
    kind: str,
    display_name: str,
    **kwargs,
) -> str:
    """Register a new artifact in the catalog.

    Accepts optional keyword arguments: description, provider, version,
    tags (list), capabilities (list), status, and any spec fields.
    """
    from aurarouter.catalog_model import ArtifactKind, CatalogArtifact

    try:
        artifact_kind = ArtifactKind(kind)
    except ValueError:
        return json.dumps({"error": f"Invalid kind '{kind}'. Must be one of: model, service, analyzer"})

    data: dict = {"kind": kind, "display_name": display_name}
    for key in ("description", "provider", "version", "tags", "capabilities", "status"):
        if key in kwargs:
            data[key] = kwargs[key]
    # Remaining kwargs go into spec (merged at top level in storage)
    spec_keys = set(kwargs.keys()) - {"description", "provider", "version", "tags", "capabilities", "status"}
    for key in spec_keys:
        data[key] = kwargs[key]

    config.catalog_set(artifact_id, data)
    return json.dumps({"success": True, "artifact_id": artifact_id, "kind": kind})


def catalog_remove_artifact(config: "ConfigLoader", artifact_id: str) -> str:
    """Remove an artifact from the catalog."""
    removed = config.catalog_remove(artifact_id)
    if removed:
        return json.dumps({"success": True, "artifact_id": artifact_id})
    return json.dumps({"error": f"Artifact '{artifact_id}' not found in catalog"})


def set_active_analyzer(config: "ConfigLoader", analyzer_id: str | None = None) -> str:
    """Set (or clear) the active analyzer."""
    config.set_active_analyzer(analyzer_id)
    return json.dumps({"success": True, "active_analyzer": analyzer_id})


def get_active_analyzer(config: "ConfigLoader") -> str:
    """Get the currently active analyzer ID."""
    analyzer_id = config.get_active_analyzer()
    return json.dumps({"active_analyzer": analyzer_id})
