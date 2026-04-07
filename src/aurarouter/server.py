from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from aurarouter._logging import get_logger
from aurarouter.budget_sync import (
    BudgetSyncStore,
    get_global_budget_fn as _get_global_budget,
    report_budget_sync_fn as _report_budget_sync,
)
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric
from aurarouter.registration import (
    RegistrationStore,
    discovery_status_fn as _discovery_status,
    registration_ready_fn as _registration_ready,
)
from aurarouter.mcp_tools import (
    catalog_get_artifact as _catalog_get_artifact,
    catalog_list_artifacts as _catalog_list_artifacts,
    catalog_register_artifact as _catalog_register_artifact,
    catalog_remove_artifact as _catalog_remove_artifact,
    compare_models as _compare_models,
    generate_code as _generate_code,
    get_active_analyzer as _get_active_analyzer,
    list_assets as _list_assets,
    list_intents as _list_intents,
    list_models as _list_models,
    local_inference as _local_inference,
    rag_status as _rag_status,
    register_asset as _register_asset,
    register_remote_asset as _register_remote_asset,
    route_task as _route_task,
    set_active_analyzer as _set_active_analyzer,
    sovereignty_status as _sovereignty_status,
    speculative_execute as _speculative_execute,
    speculative_status as _speculative_status,
    unregister_asset as _unregister_asset,
    monologue_execute as _monologue_execute,
    monologue_status as _monologue_status,
    monologue_trace as _monologue_trace,
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
    pricing_catalog = PricingCatalog(
        overrides=overrides,
        config_resolver=config.get_model_pricing,
    )

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
    "aurarouter.assets.register_remote": True,
    "aurarouter.assets.unregister": True,
    "intelligent_code_gen": False,
    "aurarouter.budget.report_sync": True,
    "aurarouter.budget.global": True,
}


def create_mcp_server(config: ConfigLoader) -> FastMCP:
    """Factory that builds a fully-wired FastMCP server instance."""
    import json as _json

    mcp = FastMCP("AuraRouter")

    # T9.1 — discovery handshake store (shared by catalog.register + HTTP routes)
    registration_store = RegistrationStore()
    # T9.2 — global budget sync store
    budget_sync_store = BudgetSyncStore()

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
            discovery_tool = grid_cfg.get("model_discovery_tool")
            if not discovery_tool:
                logger.info("No model_discovery_tool configured; relying on push registration")
            added = registry.sync_models(config, model_discovery_tool=discovery_tool)
            if added:
                logger.info(f"Auto-registered {added} remote model(s) from grid services")

        # Inject routing advisors into fabric
        fabric.set_routing_advisors(registry)

    # --- Provider catalog (discover entry-point & manual providers) ---
    from aurarouter.catalog import ProviderCatalog

    catalog = ProviderCatalog(config)
    catalog.discover()

    # Auto-start providers that have auto_start: true
    for manual_entry in config.get_catalog_manual_entries():
        entry_name = manual_entry.get("name", "")
        if manual_entry.get("auto_start", False) and entry_name:
            catalog.start_provider(entry_name)

    # Auto-start entry-point providers if configured
    if config.get_catalog_auto_start_entrypoints():
        for entry in catalog.get_entrypoint_providers():
            if entry.metadata and entry.metadata.command:
                catalog.start_provider(entry.name)

    # Auto-register models from running providers
    for entry_name, entry in catalog._entries.items():
        if entry.source in ("entrypoint", "manual"):
            added = catalog.auto_register_models(entry_name, config)
            if added:
                logger.info(
                    "Auto-registered %d model(s) from provider '%s'",
                    added, entry_name,
                )

    def _is_enabled(tool_name: str) -> bool:
        default = _MCP_TOOL_DEFAULTS.get(tool_name, True)
        return config.is_mcp_tool_enabled(tool_name, default=default)

    # --- Conditionally register tools ---

    if _is_enabled("route_task"):
        @mcp.tool()
        def route_task(
            task: str, context: str = "", format: str = "text",
            permissions: str = "",
        ) -> str:
            """Route a task to local or specialized AI models with automatic
            fallback. Provides access to local LLMs and multi-model
            orchestration not available in this environment. Use for any task
            that benefits from local inference, privacy-preserving processing,
            or multi-model routing."""
            perms_dict = None
            if permissions:
                try:
                    import json
                    perms_dict = json.loads(permissions)
                except Exception:
                    logger.warning(f"Invalid permissions JSON: {permissions}")

            return _route_task(
                fabric, triage_router, task=task, context=context, format=format,
                config=config, options={"permissions": perms_dict} if perms_dict else None,
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
            cost_per_1m_input: float = -1.0,
            cost_per_1m_output: float = -1.0,
            hosting_tier: str = "",
        ) -> str:
            """Register a new GGUF model file for immediate routing. Adds
            the model to both the physical asset registry and the routing
            configuration with the specified capability tags. Tags matching
            existing role names or semantic verb synonyms automatically add
            the model to those role chains. The model becomes routable
            immediately without server restart.

            Optional cost and tier metadata:
            - cost_per_1m_input: Cost per 1M input tokens in USD (-1.0 = not set)
            - cost_per_1m_output: Cost per 1M output tokens in USD (-1.0 = not set)
            - hosting_tier: "on-prem", "cloud", or "dedicated-tenant" (empty = infer)
            """
            return _register_asset(
                fabric, config,
                model_id=model_id,
                file_path=file_path,
                repo=repo,
                tags=tags,
                cost_per_1m_input=cost_per_1m_input,
                cost_per_1m_output=cost_per_1m_output,
                hosting_tier=hosting_tier,
            )

    if _is_enabled("aurarouter.assets.register_remote"):
        @mcp.tool(name="aurarouter.assets.register_remote")
        def register_remote_asset(
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
            """Register a remote model endpoint for immediate routing without
            requiring a local file. Creates a routing entry pointing to the
            remote inference endpoint. Tags matching existing role names or
            semantic verb synonyms automatically add the model to those role
            chains. Use this for models hosted on AuraGrid nodes or external
            inference servers."""
            return _register_remote_asset(
                fabric, config,
                model_id=model_id,
                endpoint_url=endpoint_url,
                provider=provider,
                tags=tags,
                capabilities=capabilities,
                context_window=context_window,
                cost_per_1m_input=cost_per_1m_input,
                cost_per_1m_output=cost_per_1m_output,
                hosting_tier=hosting_tier,
                node_id=node_id,
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
    if _is_enabled("intelligent_code_gen"):
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

    # --- Unified artifact catalog & default analyzer ---
    from aurarouter.analyzers import create_default_analyzer

    default_analyzer = create_default_analyzer()
    if config.catalog_get(default_analyzer.artifact_id) is None:
        config.catalog_set(default_analyzer.artifact_id, default_analyzer.to_dict())

    if config.get_active_analyzer() is None:
        config.set_active_analyzer(default_analyzer.artifact_id)

    # --- Catalog MCP tools ---
    @mcp.tool(name="aurarouter.catalog.list")
    def catalog_list_artifacts(kind: str = "") -> str:
        """List all artifacts in the unified catalog, optionally filtered
        by kind (model, service, analyzer)."""
        return _catalog_list_artifacts(config, kind=kind or None)

    @mcp.tool(name="aurarouter.catalog.get")
    def catalog_get_artifact(artifact_id: str) -> str:
        """Get details for a single catalog artifact by ID."""
        return _catalog_get_artifact(config, artifact_id)

    @mcp.tool(name="aurarouter.catalog.register")
    def catalog_register_artifact(
        artifact_id: str,
        kind: str,
        display_name: str,
        description: str = "",
        provider: str = "",
        version: str = "",
        tags: str = "",
        capabilities: str = "",
        handshake_version: int = 0,
    ) -> str:
        """Register a new artifact (model, service, or analyzer) in the
        unified catalog.  When handshake_version=1 is supplied the response
        includes a catalog_id for the three-phase discovery handshake."""
        kwargs: dict = {}
        if description:
            kwargs["description"] = description
        if provider:
            kwargs["provider"] = provider
        if version:
            kwargs["version"] = version
        if tags:
            # tags may arrive as a comma-separated string or a list
            if isinstance(tags, list):
                kwargs["tags"] = tags
            else:
                kwargs["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        # capabilities may arrive as a comma-separated string or a JSON array
        caps_list: list[str] = []
        if capabilities:
            if isinstance(capabilities, list):
                caps_list = [str(c) for c in capabilities]
            else:
                caps_list = [c.strip() for c in capabilities.split(",") if c.strip()]
            kwargs["capabilities"] = caps_list

        _catalog_register_artifact(
            config, artifact_id=artifact_id, kind=kind,
            display_name=display_name, **kwargs,
        )

        # T9.1: v1 handshake — return ACK with catalog_id
        if handshake_version == 1:
            catalog_id, accepted = registration_store.announce(artifact_id, caps_list)
            return _json.dumps({
                "catalog_id": catalog_id,
                "accepted_capabilities": accepted,
                "handshake_version": 1,
            })

        # Legacy: plain success (backward compat with old callers)
        return _json.dumps({"success": True, "artifact_id": artifact_id, "kind": kind})

    @mcp.tool(name="aurarouter.catalog.remove")
    def catalog_remove_artifact(artifact_id: str) -> str:
        """Remove an artifact from the unified catalog."""
        return _catalog_remove_artifact(config, artifact_id)

    @mcp.tool(name="aurarouter.analyzer.set_active")
    def set_active_analyzer(analyzer_id: str = "") -> str:
        """Set or clear the active analyzer for routing."""
        return _set_active_analyzer(config, analyzer_id=analyzer_id or None)

    @mcp.tool(name="aurarouter.analyzer.get_active")
    def get_active_analyzer() -> str:
        """Get the currently active analyzer ID."""
        return _get_active_analyzer(config)

    @mcp.tool(name="aurarouter.intents.list")
    def list_intents() -> str:
        """List all available intents (built-in and analyzer-declared) with
        their target roles and sources. Useful for discovering what intent
        classifications are available for routing decisions."""
        return _list_intents(config)

    @mcp.tool(name="aurarouter.sovereignty.status")
    def sovereignty_status() -> str:
        """Return sovereignty enforcement status, custom pattern count,
        and available local models."""
        return _sovereignty_status(fabric)

    @mcp.tool(name="aurarouter.rag.status")
    def rag_status() -> str:
        """Return RAG enrichment status and XLM endpoint configuration."""
        return _rag_status(fabric)

    @mcp.tool(name="aurarouter.budget.status")
    def budget_status() -> str:
        """Return current daily and monthly spend and limits."""
        if not fabric._budget_manager:
            return "Budget manager not configured"
        
        status = fabric._budget_manager.check_budget("cloud-generic") # just to get status
        return json.dumps({
            "enabled": fabric._budget_manager.is_enabled(),
            "daily_spend": status.daily_spend,
            "monthly_spend": status.monthly_spend,
            "daily_limit": status.daily_limit,
            "monthly_limit": status.monthly_limit,
            "daily_remaining": fabric._budget_manager.get_daily_remaining(),
            "monthly_remaining": fabric._budget_manager.get_monthly_remaining(),
        }, indent=2)

    # --- TG7: Speculative decoding tools ---
    @mcp.tool(name="aurarouter.speculative.execute")
    def speculative_execute(
        task: str, context: str = "",
        permissions: str = "",
    ) -> str:
        """Execute a task using speculative decoding with drafter/verifier."""
        return _speculative_execute(
            fabric, task=task, context=context, permissions=permissions
        )

    @mcp.tool(name="aurarouter.speculative.status")
    def speculative_status_tool() -> str:
        """Return speculative decoding status and active sessions."""
        return _speculative_status(fabric)

    # --- TG10: AuraMonologue reasoning tools ---
    @mcp.tool(name="aurarouter.monologue.execute")
    def monologue_execute(
        task: str,
        context: str = "",
        max_iterations: int = 5,
        convergence_threshold: float = 0.85,
        mas_relevancy_threshold: float = 0.4,
        permissions: str = "",
    ) -> str:
        """Execute a task using recursive multi-expert reasoning (AuraMonologue)."""
        return _monologue_execute(
            fabric, task=task, context=context,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            mas_relevancy_threshold=mas_relevancy_threshold,
            permissions=permissions,
        )

    @mcp.tool(name="aurarouter.monologue.status")
    def monologue_status_tool() -> str:
        """Return AuraMonologue status and active sessions."""
        return _monologue_status(fabric)

    @mcp.tool(name="aurarouter.monologue.trace")
    def monologue_trace(session_id: str) -> str:
        """Retrieve the full reasoning trace for a monologue session."""
        return _monologue_trace(fabric, session_id=session_id)

    # --- T9.1: Service discovery handshake tools ---

    @mcp.tool(name="aurarouter.registration.ready")
    def registration_ready(catalog_id: str) -> str:
        """Confirm service is operational after ANNOUNCE ACK (discovery handshake phase 3).

        Args:
            catalog_id: The catalog_id received in the ANNOUNCE ACK response.

        Returns:
            JSON: {"ok": true, "catalog_id": "..."} or {"error": "..."}.
        """
        return _registration_ready(registration_store, catalog_id)

    @mcp.tool(name="aurarouter.discovery.status")
    def discovery_status() -> str:
        """Return registration status for all services in the discovery registry.

        Returns:
            JSON: {"services": [{"name": "auraxlm", "status": "operational|pending|unregistered",
                   "catalog_id": "..."}, ...]}
        """
        return _discovery_status(registration_store)

    # --- T9.1: Custom HTTP routes (direct HTTP, not MCP JSON-RPC) ---

    @mcp.custom_route("/api/registration/ready", methods=["POST"])
    async def http_registration_ready(request: Request) -> Response:
        """HTTP endpoint for the READY phase of the discovery handshake.

        Accepts: {"catalog_id": "...", "status": "operational", "handshake_version": 1}
        Returns: 200 {"ok": true, "catalog_id": "..."} or 404 {"error": "..."}
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        catalog_id = body.get("catalog_id", "")
        result_json = _registration_ready(registration_store, catalog_id)
        result = _json.loads(result_json)
        status_code = 200 if result.get("ok") else 404
        return JSONResponse(result, status_code=status_code)

    @mcp.custom_route("/api/discovery/status", methods=["GET"])
    async def http_discovery_status(request: Request) -> Response:
        """HTTP endpoint returning registration status for all services."""
        return JSONResponse(_json.loads(_discovery_status(registration_store)))

    # --- T9.2: Budget synchronization tools ---

    if _is_enabled("aurarouter.budget.report_sync"):
        @mcp.tool(name="aurarouter.budget.report_sync")
        def report_budget_sync(payload_json: str) -> str:
            """Accept a cross-project budget sync report and store it.

            Args:
                payload_json: JSON string conforming to BudgetSyncMessage:
                    {"source": "aurarouter|auraxlm|auragrid",
                     "period_start": "<ISO datetime>",
                     "period_end": "<ISO datetime>",
                     "token_spend": {"input": 0, "output": 0},
                     "inference_cost_usd": 0.0,
                     "compute_cost_usd": 0.0}

            Returns:
                JSON: {"ok": true, "source": "..."} or {"error": "..."}.
            """
            return _report_budget_sync(budget_sync_store, payload_json)

    if _is_enabled("aurarouter.budget.global"):
        @mcp.tool(name="aurarouter.budget.global")
        def get_global_budget() -> str:
            """Return a merged view of all cross-project budget sync reports.

            Aggregates token spend and costs across aurarouter, auraxlm, and auragrid.
            Last-write-wins per source; resets on AuraRouter restart.

            Returns:
                JSON: {"period_start": "...", "period_end": "...",
                       "total_input_tokens": 0, "total_output_tokens": 0,
                       "total_inference_cost_usd": 0.0, "total_compute_cost_usd": 0.0,
                       "sources_reported": [...]}
            """
            return _get_global_budget(budget_sync_store)

    return mcp
