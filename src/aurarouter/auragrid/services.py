"""
AuraRouter services exposed as AuraGrid services.

This module wraps aurarouter's routing roles as AuraGrid-compatible services
that can be discovered and called by other grid applications.
"""

import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

from aurarouter.fabric import ComputeFabric
from aurarouter.routing import analyze_intent, generate_plan

logger = logging.getLogger(__name__)

try:
    from auragrid import auragrid_method, auragrid_service
except ImportError:
    logger.warning(
        "AuraGrid SDK not found — running in standalone mode. "
        "Grid integration will not be available."
    )

    # Fallback decorators if auragrid-sdk not installed
    def auragrid_service(name: str):
        return lambda cls: cls

    def auragrid_method():
        return lambda fn: fn


@auragrid_service(name="RouterService")
class RouterService:
    """
    Intent classification service.
    
    Analyzes incoming tasks to determine routing strategy.
    """

    def __init__(self, fabric: ComputeFabric, storage_client: Optional[Any] = None):
        """Initialize with ComputeFabric and optional AuraGrid AsyncStorageClient."""
        self.fabric = fabric
        self._storage_client = storage_client

    def set_storage_client(self, storage_client: Optional[Any]) -> None:
        """Attach an AuraGrid storage client for resolving CAS blob URIs."""
        self._storage_client = storage_client

    async def _resolve_grid_images(self, images: Optional[List[Any]]) -> List[Any]:
        if not images:
            return []
        resolved: List[Any] = []
        for item in images:
            if isinstance(item, dict) and "storage_uri" in item:
                if self._storage_client is None:
                    raise RuntimeError("Grid storage_client is required to resolve storage_uri blobs")
                uri = str(item["storage_uri"])
                raw_bytes = await self._storage_client.read(uri)
                if raw_bytes is None:
                    raise FileNotFoundError(f"Grid blob not found at {uri}")
                expected_sha = item.get("sha256")
                if expected_sha:
                    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
                    if actual_sha.lower() != str(expected_sha).lower():
                        raise ValueError(
                            f"Grid blob SHA-256 mismatch for {uri}: expected {expected_sha}, got {actual_sha}"
                        )
                resolved.append(raw_bytes)
            else:
                resolved.append(item)
        return resolved

    @auragrid_method()
    async def execute_multimodal(
        self,
        task: str,
        context: str = "",
        images: Optional[List[Any]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        execution_mode: Optional[str] = None,
        prefer_local: bool = False,
        role_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a multimodal or structured extraction request on this AuraRouter MAS node."""
        resolved_images = await self._resolve_grid_images(images)
        effective_intent = intent or ("VISION_EXTRACTION" if resolved_images else "EXTRACTION")
        target_role = role_override or (
            "vision" if effective_intent in ("VISION_OCR", "VISION_EXTRACTION") else "coding"
        )
        cfg = getattr(self.fabric, "_config", None)
        if not role_override and target_role == "vision" and cfg and not cfg.get_role_chain("vision"):
            target_role = "coding"

        exec_opts: Dict[str, Any] = {"intent": effective_intent}
        if resolved_images:
            exec_opts["images"] = resolved_images
        if response_schema is not None:
            exec_opts["response_schema"] = response_schema
        if execution_mode:
            exec_opts["execution_mode"] = execution_mode

        chain_override: Optional[List[str]] = None
        if prefer_local and hasattr(self.fabric, "get_local_chain") and cfg:
            local_chain = self.fabric.get_local_chain(target_role) or []
            full_chain = cfg.get_role_chain(target_role) or []
            if local_chain:
                chain_override = local_chain + [m for m in full_chain if m not in local_chain]

        prompt = f"TASK: {task}\nCONTEXT:\n{context}" if context else task
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: self.fabric.execute(
                target_role,
                prompt,
                json_mode=response_schema is not None,
                options=exec_opts,
                chain_override=chain_override,
            ),
        )
        text = res.text if hasattr(res, "text") else (res or "")
        return {
            "output": text,
            "intent": effective_intent,
            "model_id": getattr(res, "model_id", "") if res else "",
            "provider": getattr(res, "provider", "") if res else "",
            "input_tokens": int(getattr(res, "input_tokens", 0) or 0) if res else 0,
            "output_tokens": int(getattr(res, "output_tokens", 0) or 0) if res else 0,
            "success": bool(text),
        }

    @auragrid_method()
    async def classify_intent(
        self, task_description: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a task intent to determine routing.

        Args:
            task_description: The task to classify
            context: Additional context (optional)

        Returns:
            Routing decision with classification and metadata
        """
        # Use existing aurarouter routing logic (convert to async)
        loop = asyncio.get_running_loop()
        routing_decision = await loop.run_in_executor(
            None, 
            self.fabric.execute,
            "router",
            task_description,
            False
        )
        
        return {
            "classification": routing_decision,
            "task": task_description,
            "success": routing_decision is not None,
        }


@auragrid_service(name="ReasoningService")
class ReasoningService:
    """
    Architectural planning service.

    Generates execution plans for complex tasks.
    """

    def __init__(self, fabric: ComputeFabric):
        """Initialize with ComputeFabric instance."""
        self.fabric = fabric

    @auragrid_method()
    async def generate_plan(
        self, intent: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an execution plan for a given intent.

        Args:
            intent: The classified intent/task
            context: Additional context (optional)

        Returns:
            Dict with plan steps and metadata
        """
        # Use existing aurarouter planning logic (convert to async)
        loop = asyncio.get_running_loop()
        plan = await loop.run_in_executor(
            None,
            self.fabric.execute,
            "reasoning",
            intent,
            True  # json_mode for better structure
        )
        
        steps = []
        if plan:
            # Try to parse as JSON list, fallback to treating as single step
            try:
                import json
                steps = json.loads(plan) if isinstance(plan, str) else plan
                if not isinstance(steps, list):
                    steps = [steps]
            except (json.JSONDecodeError, TypeError):
                steps = [plan]
        
        return {
            "steps": steps,
            "intent": intent,
            "step_count": len(steps),
            "success": plan is not None,
        }


@auragrid_service(name="CodingService")
class CodingService:
    """
    Code generation service.

    Generates code based on a plan step.
    """

    def __init__(self, fabric: ComputeFabric):
        """Initialize with ComputeFabric instance."""
        self.fabric = fabric

    @auragrid_method()
    async def generate_code(
        self, plan_step: str, language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate code for a plan step.

        Args:
            plan_step: Description of the step to code
            language: Target programming language

        Returns:
            Generated code and metadata
        """
        # Use existing aurarouter coding logic (convert to async)
        prompt = f"Generate {language} code for: {plan_step}"
        
        loop = asyncio.get_running_loop()
        code = await loop.run_in_executor(
            None,
            self.fabric.execute,
            "coding",
            prompt,
            False
        )
        
        return {
            "code": code,
            "language": language,
            "plan_step": plan_step,
            "success": code is not None,
        }


@auragrid_service(name="UnifiedRouterService")
class UnifiedRouterService:
    """
    Unified intelligent code generation service.

    Coordinates routing, planning, and code generation.
    """

    def __init__(self, fabric: ComputeFabric):
        """Initialize with ComputeFabric instance."""
        self._fabric = fabric

    @auragrid_method()
    async def intelligent_code_gen(
        self,
        task_description: str,
        file_context: str = "",
        language: str = "python",
    ) -> str:
        """AuraRouter V3: Multi-model routing with Intent Classification and Auto-Planning."""
        loop = asyncio.get_running_loop()
        intent = await loop.run_in_executor(None, analyze_intent, self._fabric, task_description)

        if intent == "SIMPLE_CODE":
            prompt = (
                f"TASK: {task_description}\n"
                f"LANG: {language}\n"
                f"CONTEXT: {file_context}\n"
                "CODE ONLY."
            )
            return await loop.run_in_executor(None, self._fabric.execute, "coding", prompt) or "Error: Generation failed."

        # COMPLEX_REASONING path
        plan = await loop.run_in_executor(None, generate_plan, self._fabric, task_description, file_context)

        output: list[str] = []
        for i, step in enumerate(plan):
            prompt = (
                f"GOAL: {step}\n"
                f"LANG: {language}\n"
                f"CONTEXT: {file_context}\n"
                f"PREVIOUS_CODE: {output}\n"
                "Return ONLY valid code."
            )
            code = await loop.run_in_executor(None, self._fabric.execute, "coding", prompt)
            if code:
                output.append(f"\n# --- Step {i + 1}: {step} ---\n{code}")
            else:
                output.append(f"\n# Step {i + 1} Failed.")

        return "\n".join(output)


@auragrid_service(name="AuraCatalog")
class AuraCatalogService:
    """Catalog management service exposed to AuraGrid MAS."""

    def __init__(self, config=None, registration_store=None):
        self.config = config
        self.registration_store = registration_store

    @auragrid_method()
    async def Register(
        self,
        artifact_id: str,
        kind: str = "service",
        display_name: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Register an artifact in the unified catalog."""
        import json
        from aurarouter.mcp_tools import catalog_register_artifact

        if not display_name:
            display_name = artifact_id
        res_str = catalog_register_artifact(
            self.config,
            artifact_id=artifact_id,
            kind=kind,
            display_name=display_name,
            **kwargs,
        )
        try:
            return json.loads(res_str)
        except Exception:
            return {"success": True, "artifact_id": artifact_id, "kind": kind}

    @auragrid_method()
    async def Remove(self, artifact_id: str) -> Dict[str, Any]:
        """Remove an artifact from the unified catalog."""
        import json
        from aurarouter.mcp_tools import catalog_remove_artifact

        res_str = catalog_remove_artifact(self.config, artifact_id=artifact_id)
        try:
            return json.loads(res_str)
        except Exception:
            return {"success": True, "artifact_id": artifact_id}

    @auragrid_method()
    async def List(self, kind: Optional[str] = None) -> Any:
        """List artifacts from the unified catalog."""
        import json
        from aurarouter.mcp_tools import catalog_list_artifacts

        res_str = catalog_list_artifacts(self.config, kind=kind)
        try:
            return json.loads(res_str)
        except Exception:
            return []

