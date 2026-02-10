"""
AuraRouter services exposed as AuraGrid services.

This module wraps aurarouter's routing roles as AuraGrid-compatible services
that can be discovered and called by other grid applications.
"""

import asyncio
from typing import Any, Dict, Optional

try:
    from auragrid import auragrid_method, auragrid_service
except ImportError:
    # Fallback decorators if auragrid-sdk not installed
    def auragrid_service(name: str):
        return lambda cls: cls

    def auragrid_method():
        return lambda fn: fn


from aurarouter.fabric import ComputeFabric
from aurarouter.routing import analyze_intent, generate_plan


@auragrid_service(name="RouterService")
class RouterService:
    """
    Intent classification service.
    
    Analyzes incoming tasks to determine routing strategy.
    """

    def __init__(self, fabric: ComputeFabric):
        """Initialize with ComputeFabric instance."""
        self.fabric = fabric

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
