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
        loop = asyncio.get_event_loop()
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
        loop = asyncio.get_event_loop()
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
        
        loop = asyncio.get_event_loop()
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
        self.fabric = fabric
        self.router = RouterService(fabric)
        self.reasoner = ReasoningService(fabric)
        self.coder = CodingService(fabric)

    @auragrid_method()
    async def intelligent_code_gen(
        self,
        task: str,
        language: str = "python",
        file_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate code intelligently via routing, planning, and execution.

        This is the unified endpoint that orchestrates:
        1. Intent classification (router role)
        2. Plan generation (reasoning role)
        3. Code generation (coding role)

        Args:
            task: The task description
            language: Target programming language
            file_context: Existing file context (optional)

        Returns:
            Generated code and execution details
        """
        try:
            # Build full prompt
            prompt = f"Task: {task}\nLanguage: {language}"
            if file_context:
                prompt += f"\nExisting Code Context:\n{file_context}"

            # Execute through the unified pipeline
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.fabric.execute,
                "router",  # Start with router, which cascades through pipeline
                prompt,
                False
            )

            return {
                "result": result,
                "task": task,
                "language": language,
                "context_provided": file_context is not None,
                "success": result is not None,
            }
        
        except Exception as e:
            return {
                "result": None,
                "task": task,
                "language": language,
                "error": str(e),
                "success": False,
            }
