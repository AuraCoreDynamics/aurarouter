# Changelog

All notable changes to AuraRouter are documented here.

## [0.5.5] — 2026-04-08

### Added
- **Savings Telemetry Persistence**: `UsageStore` now persists `simulated_cost_avoided` and `complexity_score` for all locally-routed tasks.
- **ROI & Telemetry Dashboard**: New high-visibility tab in the desktop GUI `MonitorPanel` visualizing cumulative counterfactual savings, hard-route ratios, and task complexity.
- **ROI Metrics API**: `AuraRouterAPI.get_roi_metrics()` provides aggregated return-on-investment statistics over configurable timeframes.
- **Complexity-Aware Usage Recording**: `ComputeFabric` now extracts complexity and savings metadata from `RoutingContext` and `GenerateResult` for persistent auditing.
- **Automatic Schema Migration**: SQLite `UsageStore` automatically evolves existing databases to support ROI fields via runtime `ALTER TABLE` checks.

### Changed
- `UsageRecord` model updated with ROI metadata fields.
- `MonitorPanel` UI expanded to five navigation tabs.
- `UsageStore.record()` and `query()` updated to handle ROI metadata.

### Fixed
- Ephemeral ROI data loss: Savings and complexity metrics are no longer lost after the inference loop completes.

## [0.5.4] — 2026-04-03

### Added
- **Artifact Discovery Service**: Unified Artifact Catalog now serves as a central discovery registry for downstream projects and test suites.
- **JSON CLI Querying**: `aurarouter catalog artifacts --json` provides machine-readable discovery of all registered compute resources.
- **AuraXLM Test Integration**: Practical implementation of dynamic ONNX model discovery for AuraXLM unit and integration tests, eliminating brittle hardcoded paths.
- **RAG enrichment pipeline** with AuraXLM-backed context injection and graceful timeout fallback
- **Sovereignty gate** for prompt evaluation, local-only enforcement, and blocked execution when no compliant local models exist
- **Response sanitizer** with built-in PII patterns, configurable sovereignty patterns, and unified sovereignty audit logging
- **Speculative decoding orchestration**: `SpeculativeOrchestrator`, speculative session/state tracking, and notional streaming support
- **Notional response protocol** with correction events for verifier-driven rewind/replay
- **AuraMonologue** recursive generator / critic / refiner reasoning loop with MAS-score-gated node idling
- **New MCP tools** for `rag_status`, `sovereignty_status`, `speculative_execute`, `speculative_status`, `monologue_execute`, `monologue_status`, and `monologue_trace`
- **Cross-language serialization tests** covering DraftTokenBatch, VerificationResult, LatentAnchor metadata, and unified sovereignty audit payloads

### Changed
- `route_task()` execution path now supports monologue and speculative execution mode selection in addition to standard routing
- sovereignty decisions are now audited in a shared cross-project schema
- full test suite now runs clean with optional provider discovery tests skipped when `aurarouter-claude` / `aurarouter-gemini` are not installed

### Test Status
- `pytest tests/ -q` -> 1499 passed, 8 skipped

## [0.5.3] — 2026-03-28

### Added
- **Intent Registry**: `IntentRegistry` and `IntentDefinition` classes for central management of all known intents (built-in + analyzer-declared). `build_intent_registry()` factory populates the registry from the active analyzer's `role_bindings`.
- **Custom domain intents**: Analyzers can declare domain-specific intents via `role_bindings` in their catalog spec. Each key becomes a registered intent with higher priority than built-in intents.
- **Analyzer spec validation**: `validate_analyzer_spec()` and `AnalyzerSpecValidation` dataclass for validating analyzer spec fields (required fields, role binding targets, MCP endpoint format). Warn-only for backwards compatibility.
- **`list_intents` MCP tool**: Returns all available intents (built-in + analyzer-declared) with target roles and sources as JSON.
- **`route_task` intent parameter**: Optional `intent` parameter to bypass auto-classification and force a specific intent.
- **CLI `intent` subcommand**: `aurarouter intent list` and `aurarouter intent describe NAME` for intent discovery and inspection.
- **CLI `--intent` flag**: `aurarouter run TASK --intent NAME` to force a specific intent during task execution.
- **Intent combobox (GUI)**: Workspace panel combobox for selecting intents before execution. Shows Auto, built-in intents, and analyzer-declared intents in grouped sections. Auto-refreshes on analyzer change.
- **Intent display in Settings**: Settings panel analyzer section shows declared intents as tag chips alongside built-in intents.
- **`supported_intents` on models**: `CatalogArtifact.supported_intents` field allows models to declare which intents they are suited for. `ComputeFabric.filter_chain_by_intent()` narrows model chains based on declared support.
- **Routing advisors**: `register_routing_advisor()`, `unregister_routing_advisor()`, `list_routing_advisors()`, and `consult_routing_advisors()` on `ComputeFabric` for intent-aware chain reordering by external MCP services.
- **Auto-registration of catalog advisors**: Services with `routing_advisor` capability are auto-registered on startup.
- **Reference contracts**: `contracts/auracode.py` (AuraCode intents and `create_auracode_analyzer_spec()`) and `contracts/auraxlm.py` (AuraXLM MoE advisor interface with `ANALYZE_ROUTE_PARAMS` and response schema).
- **Analyzer Developer Guide**: Comprehensive documentation at `docs/ANALYZER_GUIDE.md` covering analyzer types, registration, spec schema, role bindings, MCP endpoint contract, intent lifecycle, routing advisors, and a worked SAR processing example.
- **In-app help topics**: Four new help topics: `custom-intents` (concept), `intent-selection` (how-to), `analyzer-intents` (reference), `routing-advisors` (concept).

### Changed
- `ComputeFabric` constructor accepts optional `routing_advisors` parameter
- `ComputeFabric.execute()` consults routing advisors for chain reordering before model execution
- Intent classification prompt now includes custom intents via `IntentRegistry.build_classifier_choices()`
- `CatalogArtifact.from_dict()` and `to_dict()` handle `supported_intents` field
- GUI workspace panel now includes intent selector in the button row
- Settings panel analyzer section displays intent-to-role mappings as tag chips
- README updated with Intent Classification section, `list_intents` tool, and `intent` CLI commands
- GUI Guide updated with intent combobox and settings panel documentation
- Deployment Guide updated with semantic_verbs/role_bindings/supported_intents relationship
- Backend Plugins Guide updated with analyzer plugins comparison section

### Backward Compatibility
- All changes are additive. Existing configs without `role_bindings` or `supported_intents` work unchanged.
- Intent auto-classification is the default; explicit intent selection is opt-in.
- `filter_chain_by_intent()` returns the full chain when no models declare `supported_intents`.
- Built-in intents (DIRECT, SIMPLE_CODE, COMPLEX_REASONING) are always available.

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
