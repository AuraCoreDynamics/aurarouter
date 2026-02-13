from pathlib import Path

from mcp.server.fastmcp import FastMCP

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.routing import analyze_intent, generate_plan
from aurarouter.savings.budget import BudgetManager
from aurarouter.savings.pricing import CostEngine, ModelPrice, PricingCatalog
from aurarouter.savings.privacy import PrivacyAuditor, PrivacyPattern, PrivacyStore
from aurarouter.savings.triage import TriageRouter
from aurarouter.savings.usage_store import UsageStore

logger = get_logger("AuraRouter.Server")


def _build_savings_components(config: ConfigLoader):
    """Instantiate savings components from config. Returns kwargs for ComputeFabric."""
    if not config.is_savings_enabled():
        return {}

    savings_cfg = config.get_savings_config()

    # UsageStore
    db_path_raw = savings_cfg.get("db_path")
    db_path = Path(db_path_raw) if db_path_raw else None
    usage_store = UsageStore(db_path=db_path)

    # PricingCatalog
    overrides_raw = config.get_pricing_overrides()
    overrides = None
    if overrides_raw:
        overrides = {
            k: ModelPrice(v["input_per_million"], v["output_per_million"])
            for k, v in overrides_raw.items()
        }
    pricing_catalog = PricingCatalog(overrides=overrides)

    # PrivacyAuditor + PrivacyStore
    privacy_cfg = config.get_privacy_config()
    privacy_auditor = None
    privacy_store = None
    if privacy_cfg.get("enabled", True):
        custom_raw = privacy_cfg.get("custom_patterns", [])
        custom = [
            PrivacyPattern(
                name=p["name"],
                pattern=p["pattern"],
                severity=p.get("severity", "medium"),
                description=p.get("description", ""),
            )
            for p in custom_raw
        ]
        privacy_auditor = PrivacyAuditor(custom_patterns=custom or None)
        privacy_store = PrivacyStore(db_path=db_path)

    # BudgetManager
    budget_cfg = config.get_budget_config()
    budget_manager = None
    if budget_cfg.get("enabled", False):
        cost_engine = CostEngine(pricing_catalog, usage_store)
        budget_manager = BudgetManager(cost_engine, budget_cfg)

    return {
        "usage_store": usage_store,
        "pricing_catalog": pricing_catalog,
        "privacy_auditor": privacy_auditor,
        "privacy_store": privacy_store,
        "budget_manager": budget_manager,
    }


def _build_triage_router(config: ConfigLoader) -> TriageRouter | None:
    """Build a TriageRouter from config, or None if triage is not enabled."""
    triage_cfg = config.get_triage_config()
    if not triage_cfg.get("enabled", False):
        return None
    return TriageRouter.from_config(triage_cfg)


def create_mcp_server(config: ConfigLoader) -> FastMCP:
    """Factory that builds a fully-wired FastMCP server instance."""
    mcp = FastMCP("AuraRouter")

    savings_kwargs = _build_savings_components(config)
    fabric = ComputeFabric(config, **savings_kwargs)
    triage_router = _build_triage_router(config)

    @mcp.tool()
    def intelligent_code_gen(
        task_description: str,
        file_context: str = "",
        language: str = "python",
    ) -> str:
        """AuraRouter: Multi-model task routing with intent classification and auto-planning.

        Routes tasks (code generation, summarization, analysis, etc.) across local
        and cloud models with automatic fallback.
        """
        triage = analyze_intent(fabric, task_description)
        intent = triage.intent
        complexity = triage.complexity
        logger.info(f"Intent: {intent}  Complexity: {complexity}")

        # Select coding role via triage (or default to "coding")
        coding_role = "coding"
        if triage_router is not None:
            coding_role = triage_router.select_role(complexity)
            logger.info(f"Triage selected role: {coding_role}")

        if intent == "SIMPLE_CODE":
            prompt = (
                f"TASK: {task_description}\n"
                f"LANG: {language}\n"
                f"CONTEXT: {file_context}\n"
                "CODE ONLY."
            )
            return fabric.execute(coding_role, prompt) or "Error: Generation failed."

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
            code = fabric.execute(coding_role, prompt)
            if code:
                output.append(f"\n# --- Step {i + 1}: {step} ---\n{code}")
            else:
                output.append(f"\n# Step {i + 1} Failed.")

        return "\n".join(output)

    return mcp
