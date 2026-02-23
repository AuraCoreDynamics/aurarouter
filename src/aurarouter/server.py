from pathlib import Path

from mcp.server.fastmcp import FastMCP

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import (
    compare_models as _compare_models,
    generate_code as _generate_code,
    local_inference as _local_inference,
    route_task as _route_task,
)
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


# Default enabled state for each MCP tool.
_MCP_TOOL_DEFAULTS: dict[str, bool] = {
    "route_task": True,
    "local_inference": True,
    "generate_code": True,
    "compare_models": False,
}


def create_mcp_server(config: ConfigLoader) -> FastMCP:
    """Factory that builds a fully-wired FastMCP server instance."""
    mcp = FastMCP("AuraRouter")

    savings_kwargs = _build_savings_components(config)
    fabric = ComputeFabric(config, **savings_kwargs)
    triage_router = _build_triage_router(config)

    def _is_enabled(tool_name: str) -> bool:
        default = _MCP_TOOL_DEFAULTS.get(tool_name, True)
        return config.is_mcp_tool_enabled(tool_name, default=default)

    # --- Conditionally register tools ---

    if _is_enabled("route_task"):
        @mcp.tool()
        def route_task(task: str, context: str = "", format: str = "text") -> str:
            """Route a task to local or specialized AI models with automatic
            fallback. Provides access to local LLMs and multi-model
            orchestration not available in this environment. Use for any task
            that benefits from local inference, privacy-preserving processing,
            or multi-model routing."""
            return _route_task(
                fabric, triage_router, task=task, context=context, format=format,
            )

    if _is_enabled("local_inference"):
        @mcp.tool()
        def local_inference(prompt: str, context: str = "") -> str:
            """Execute a prompt on local/private AI models (Ollama, llama.cpp)
            without sending data to cloud APIs. Use for privacy-sensitive
            tasks, offline processing, or when data must not leave the local
            network."""
            return _local_inference(fabric, prompt=prompt, context=context)

    if _is_enabled("generate_code"):
        @mcp.tool()
        def generate_code(
            task_description: str, file_context: str = "", language: str = "python",
        ) -> str:
            """Multi-step code generation with automatic planning. Breaks
            complex coding tasks into atomic steps and executes sequentially
            across specialized local and cloud models with fallback."""
            return _generate_code(
                fabric, triage_router,
                task_description=task_description,
                file_context=file_context,
                language=language,
            )

    if _is_enabled("compare_models"):
        @mcp.tool()
        def compare_models(prompt: str, models: str = "") -> str:
            """Run a prompt across multiple AI models and return all responses
            for comparison. Useful for evaluating model quality, testing
            prompts, or choosing the best response. Provide comma-separated
            model IDs, or leave empty to use all configured models."""
            return _compare_models(fabric, prompt=prompt, models=models)

    # --- Deprecated alias for backwards compatibility ---
    @mcp.tool()
    def intelligent_code_gen(
        task_description: str, file_context: str = "", language: str = "python",
    ) -> str:
        """[DEPRECATED - use generate_code instead] Multi-model code generation
        with intent classification and auto-planning."""
        return _generate_code(
            fabric, triage_router,
            task_description=task_description,
            file_context=file_context,
            language=language,
        )

    # --- Session management (opt-in) ---
    session_manager = None
    sessions_cfg = config.get_sessions_config()
    if sessions_cfg.get("enabled", False):
        from aurarouter.sessions import SessionStore, SessionManager

        store_path = sessions_cfg.get("store_path")
        store = SessionStore(
            db_path=Path(store_path) if store_path else None
        )
        session_manager = SessionManager(
            store=store,
            condensation_threshold=sessions_cfg.get("condensation_threshold", 0.8),
            auto_gist=sessions_cfg.get("auto_gist", True),
            generate_fn=lambda role, prompt: fabric.execute(role, prompt),
        )

    if session_manager is not None:
        from aurarouter.mcp_tools import register_session_tools
        register_session_tools(mcp, fabric, session_manager)

    return mcp
