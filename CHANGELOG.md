# Changelog

All notable changes to AuraRouter are documented here.

## [0.5.2] — 2026-03-24

### Added
- **External provider packages**: `aurarouter-claude` (Anthropic Claude) and `aurarouter-gemini` (Google Gemini) as separate MCP server packages, discoverable via `aurarouter.providers` entry-point group
- **XLM integration**: `xlm:` config section with `prompt_augmentation` and `usage_reporting` feature flags; `ComputeFabric._augment_prompt()` calls `auraxlm.query` for RAG context injection with fail-safe fallback
- **ConfigLoader XLM accessors**: `get_xlm_config()`, `is_xlm_augmentation_enabled()`, `is_xlm_usage_reporting_enabled()`, `get_xlm_endpoint()`
- **Feedback loop**: `FeedbackStore` (SQLite-backed routing outcome store), `ComputeFabric._record_feedback()` for asynchronous recording of model success/failure/latency; adaptive triage weight adjustment based on historical outcomes
- **Cross-project integration tests**: `test_cross_project_integration.py` validating entry-point discovery, XLM augmentation hooks, and feedback store integration

### Changed
- `ComputeFabric` constructor accepts optional `xlm_client` and `feedback_store` parameters
- `savings.feedback:` config section added for feedback loop configuration
- Provider count updated: 4 built-in + 2 external MCP = 6 total

## [0.5.1] — 2026-03-24

### Added
- **Unified Artifact Catalog**: Three-kind typed registry supporting models, services, and analyzers in a single `catalog` section of `auraconfig.yaml`
- **Route Analyzers**: First-class orchestration primitives that represent multi-model routing strategies (intent triage, MoE ranking, pipeline, custom)
- **Built-in `aurarouter-default` analyzer**: Auto-registered on startup, wraps the existing intent→triage→execute pipeline
- **Catalog MCP tools**: `aurarouter.catalog.list`, `.get`, `.register`, `.remove`, `aurarouter.analyzer.set_active`, `.get_active`
- **Analyzer-aware `route_task`**: When an analyzer is active, delegates routing decisions via MCP callback to the analyzer provider
- **Remote analyzer callback**: `_call_remote_analyzer()` calls external analyzers (e.g., AuraXLM) over MCP JSON-RPC
- **Config migration CLI**: `aurarouter migrate-config [--dry-run]` converts old-format configs to include catalog entries
- **ConfigLoader catalog CRUD**: `catalog_get`, `catalog_set`, `catalog_list`, `catalog_remove`, `catalog_query`
- **ConfigLoader missing methods**: Implemented stub methods referenced by server.py (`is_savings_enabled`, `get_grid_services_config`, `is_mcp_tool_enabled`, `auto_join_roles`, etc.)

### Changed
- `route_task()` now accepts optional `config` parameter for analyzer-aware routing
- `auraconfig.yaml` updated with `system.active_analyzer` and `catalog` section
- MCP service connections from `grid_services.endpoints` can now be migrated to catalog service entries

### Backward Compatibility
- Legacy `models` section continues to work unchanged
- `catalog_list(kind="model")` includes entries from both `catalog` and `models` sections
- All existing MCP tools (`list_models`, `register_asset`, `route_task`) work identically
- Old configs without a `catalog` section load without error

## [0.3.0] — 2026-03-01

### Added
- Multi-model MCP routing fabric with automatic fallback
- Intent → Plan → Execute loop
- Provider support: Ollama, llama.cpp, Claude, Gemini, OpenAI-compatible
- PySide6 desktop GUI
- Savings subsystem: usage tracking, pricing, privacy audit, budget management
- Triage routing: complexity-based role selection
- MCP server with configurable tool enablement
- Session management with auto-gisting
- AuraGrid MAS deployment support
- Python SDK for grid integration
