from pathlib import Path

from mcp.server.fastmcp import FastMCP

from aurarouter._logging import get_logger
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.mcp_tools import (
    compare_models as _compare_models,
    generate_code as _generate_code,
    list_assets as _list_assets,
    list_models as _list_models,
    local_inference as _local_inference,
    register_asset as _register_asset,
    route_task as _route_task,
    unregister_asset as _unregister_asset,
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
    "list_models": True,
    "aurarouter.assets.list": True,
    "aurarouter.assets.register": True,
    "aurarouter.assets.unregister": True,
}


def create_mcp_server(config: ConfigLoader) -> FastMCP:
    """Factory that builds a fully-wired FastMCP server instance."""
    mcp = FastMCP("AuraRouter")

    savings_kwargs = _build_savings_components(config)
    fabric = ComputeFabric(config, **savings_kwargs)
    triage_router = _build_triage_router(config)

    # --- Grid services (opt-in) ---
    registry = None
    grid_cfg = config.get_grid_services_config()
    if grid_cfg.get("endpoints"):
        from aurarouter.mcp_client import GridMcpClient, McpClientRegistry

        registry = McpClientRegistry()
        for ep in grid_cfg["endpoints"]:
            url = ep.get("url", "")
            name = ep.get("name", url)
            if not url:
                continue
            client = GridMcpClient(base_url=url, name=name)
            connected = client.connect()
            registry.register(name, client)
            if not connected:
                logger.warning(f"Grid service '{name}' at {url} not reachable")

        # Auto-sync discovered models into config
        if grid_cfg.get("auto_sync_models", True):
            added = registry.sync_models(config)
            if added:
                logger.info(f"Auto-registered {added} remote model(s) from grid services")

        # Inject routing advisors into fabric
        fabric._routing_advisors = registry

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

    if _is_enabled("list_models"):
        @mcp.tool()
        def list_models() -> str:
            """List all configured model IDs with their provider, endpoint,
            and tags. Includes both local and auto-discovered remote models."""
            return _list_models(fabric)

    if _is_enabled("aurarouter.assets.list"):
        @mcp.tool(name="aurarouter.assets.list")
        def list_assets() -> str:
            """List physical GGUF model files in local storage. Returns
            JSON array with repo, filename, path, size, and metadata for
            each downloaded model. Use this to discover available local
            assets for inference."""
            return _list_assets()

    if _is_enabled("aurarouter.assets.register"):
        @mcp.tool(name="aurarouter.assets.register")
        def register_asset(
            model_id: str,
            file_path: str,
            repo: str = "local",
            tags: str = "",
        ) -> str:
            """Register a new GGUF model file for immediate routing. Adds
            the model to both the physical asset registry and the routing
            configuration with the specified capability tags. Tags matching
            existing role names or semantic verb synonyms automatically add
            the model to those role chains. The model becomes routable
            immediately without server restart."""
            return _register_asset(
                fabric, config,
                model_id=model_id,
                file_path=file_path,
                repo=repo,
                tags=tags,
            )

    if _is_enabled("aurarouter.assets.unregister"):
        @mcp.tool(name="aurarouter.assets.unregister")
        def unregister_asset(
            model_id: str,
            remove_from_roles: bool = True,
            delete_file: bool = False,
        ) -> str:
            """Unregister a model from routing config and optionally delete
            the physical file. Removes the model from all role chains and
            updates routing immediately."""
            return _unregister_asset(
                fabric, config,
                model_id=model_id,
                remove_from_roles=remove_from_roles,
                delete_file=delete_file,
            )

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

    # --- Grid service tools (opt-in, only when grid services exist) ---
    if registry is not None:
        from aurarouter.mcp_tools import register_grid_tools
        register_grid_tools(mcp, fabric, registry)

    return mcp
