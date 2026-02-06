from mcp.server.fastmcp import FastMCP

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import analyze_intent, generate_plan

logger = get_logger("AuraRouter.Server")


def create_mcp_server(config: ConfigLoader) -> FastMCP:
    """Factory that builds a fully-wired FastMCP server instance."""
    mcp = FastMCP("AuraRouter")
    fabric = ComputeFabric(config)

    @mcp.tool()
    def intelligent_code_gen(
        task_description: str,
        file_context: str = "",
        language: str = "python",
    ) -> str:
        """AuraRouter V3: Multi-model routing with Intent Classification and Auto-Planning."""
        intent = analyze_intent(fabric, task_description)
        logger.info(f"Intent: {intent}")

        if intent == "SIMPLE_CODE":
            prompt = (
                f"TASK: {task_description}\n"
                f"LANG: {language}\n"
                f"CONTEXT: {file_context}\n"
                "CODE ONLY."
            )
            return fabric.execute("coding", prompt) or "Error: Generation failed."

        # COMPLEX_REASONING path
        logger.info("Complexity detected. Generating plan...")
        plan = generate_plan(fabric, task_description, file_context)
        logger.info(f"Plan: {len(plan)} steps")

        output: list[str] = []
        for i, step in enumerate(plan):
            logger.info(f"Step {i + 1}: {step}")
            prompt = (
                f"GOAL: {step}\n"
                f"LANG: {language}\n"
                f"CONTEXT: {file_context}\n"
                f"PREVIOUS_CODE: {output}\n"
                "Return ONLY valid code."
            )
            code = fabric.execute("coding", prompt)
            if code:
                output.append(f"\n# --- Step {i + 1}: {step} ---\n{code}")
            else:
                output.append(f"\n# Step {i + 1} Failed.")

        return "\n".join(output)

    return mcp
