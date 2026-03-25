# AuraRouter -- Developer Reference

Multi-model MCP routing fabric for local and cloud LLMs with automatic fallback. Routes tasks across Ollama, llama.cpp, Gemini, Claude, and OpenAI-compatible endpoints using an Intent -> Plan -> Execute loop. Runs as an MCP server, desktop GUI (PySide6), or AuraGrid managed service.

## Architecture

```
MCP Client / GUI / AuraGrid MAS
        |
   FastMCP Server (server.py)
        |
   MCP Tool Layer (mcp_tools.py)
        |-- route_task         (IPE loop + active analyzer delegation)
        |-- generate_code      (multi-step code gen with review)
        |-- local_inference    (privacy-preserving local-only execution)
        |-- catalog tools      (unified artifact CRUD)
        |-- analyzer tools     (active analyzer management)
        |-- session tools      (opt-in stateful conversations)
        |-- grid tools         (opt-in remote MCP service calls)
        |
   ComputeFabric (fabric.py)
        |-- ConfigLoader       (auraconfig.yaml read/write)
        |-- ProviderCatalog    (provider discovery + lifecycle)
        |-- TriageRouter       (complexity-based role selection)
        |-- UsageStore         (token accounting)
        |-- BudgetManager      (cost guardrails)
        |-- PrivacyAuditor     (PII detection)
        |
   Provider Layer
        |-- Ollama             (local HTTP)
        |-- llamacpp-server    (local HTTP)
        |-- llamacpp           (embedded, requires [local])
        |-- openapi            (OpenAI-compatible HTTP)
        |-- MCPProvider        (wraps external MCP server packages)
```

### Key Design Decisions

- **Unified artifact catalog.** Models, services, and analyzers share a single typed registry (`catalog` section in YAML). Legacy `models` entries are surfaced as `kind: model` artifacts transparently.
- **Route analyzers as first-class primitives.** Analyzers control routing decisions. A built-in `aurarouter-default` wraps the IPE pipeline. Remote analyzers (e.g., AuraXLM) can take over via MCP callback. Fallback to built-in on failure.
- **Provider separation.** Built-in providers (Ollama, llama.cpp, OpenAPI) ship with the core package. Cloud providers (Claude, Gemini) are external MCP server packages connected via `MCPProvider`.
- **Atomic config persistence.** `ConfigLoader.save()` writes to a temp file then renames. No partial writes.
- **Convention-over-configuration MCP tools.** Each tool can be individually enabled/disabled in `mcp.tools`. Sensible defaults in `_MCP_TOOL_DEFAULTS`.

## Project Structure

```
src/aurarouter/
  __init__.py
  server.py              # FastMCP server factory (create_mcp_server)
  config.py              # ConfigLoader — YAML read/write, catalog CRUD, active analyzer
  fabric.py              # ComputeFabric — model execution with fallback chains
  routing.py             # Intent analysis, plan generation, review loop
  mcp_tools.py           # MCP tool implementations (stateless functions)
  catalog_model.py       # CatalogArtifact, ArtifactKind domain model
  analyzers.py           # Built-in analyzer factory (create_default_analyzer)
  migration.py           # Config migration: old format -> catalog format
  cli.py                 # Click CLI (model, route, config, catalog, migrate-config, gui)
  catalog.py             # ProviderCatalog — provider discovery and lifecycle
  semantic_verbs.py      # Synonym resolution for tag-to-role auto-join
  tuning.py              # GGUF metadata extraction, auto-tune
  _logging.py            # Structured logging setup
  providers/
    protocol.py          # ProviderProtocol ABC
    ollama.py            # Ollama provider
    llamacpp_server.py   # llama.cpp HTTP server provider
    llamacpp_embedded.py # Embedded llama.cpp (optional dep)
    openapi.py           # OpenAI-compatible provider
    mcp_provider.py      # MCPProvider — wraps external MCP servers
    template/            # Starter template for custom providers
  savings/
    triage.py            # TriageRouter — complexity-based role selection
    pricing.py           # CostEngine, PricingCatalog, ModelPrice
    usage_store.py       # SQLite-backed token accounting
    budget.py            # BudgetManager — per-period cost limits
    privacy.py           # PrivacyAuditor, PrivacyStore, PII pattern matching
  sessions/
    manager.py           # SessionManager — stateful multi-turn conversations
  mcp_client/
    registry.py          # McpClientRegistry — grid service client management
  models/
    file_storage.py      # FileModelStorage — local GGUF file registry
  auragrid/
    mas_host.py          # AuraGrid MAS host entry point
  gui/
    app.py               # PySide6 desktop GUI entry point
```

## Domain Model: Unified Artifact Catalog

Three artifact kinds managed through a single `catalog` section in `auraconfig.yaml`:

### CatalogArtifact (catalog_model.py)

```python
class ArtifactKind(str, Enum):
    MODEL = "model"
    SERVICE = "service"
    ANALYZER = "analyzer"

@dataclass
class CatalogArtifact:
    artifact_id: str
    kind: ArtifactKind
    display_name: str
    description: str = ""
    provider: str = ""
    version: str = ""
    tags: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    status: str = "registered"
    spec: dict[str, Any] = field(default_factory=dict)   # Kind-specific fields
```

- `to_dict()` serializes; spec fields merge at top level (flat YAML).
- `from_dict(artifact_id, data)` deserializes from YAML config dict.
- `is_remote` property: True if `spec["mcp_endpoint"]` is set.

### ConfigLoader Catalog Methods (config.py)

| Method | Description |
|--------|-------------|
| `catalog_get(artifact_id)` | Look up by ID. Falls back to `models` section (legacy). |
| `catalog_list(kind=None)` | All artifact IDs. Merges `catalog` + legacy `models`. |
| `catalog_set(artifact_id, data)` | Write artifact to `config["catalog"]`. |
| `catalog_remove(artifact_id)` | Remove from `config["catalog"]`. Returns bool. |
| `catalog_query(kind, tags, capabilities, provider)` | Filtered query. All filter params optional. Results enriched with `artifact_id`. |
| `get_active_analyzer()` | Read `system.active_analyzer`. Returns str or None. |
| `set_active_analyzer(analyzer_id)` | Write or clear `system.active_analyzer`. |

### Built-in Analyzer (analyzers.py)

`create_default_analyzer()` returns a `CatalogArtifact` with:
- `artifact_id`: `"aurarouter-default"`
- `kind`: `ANALYZER`
- `analyzer_kind`: `"intent_triage"` (in spec)
- `role_bindings`: `{simple_code: coding, complex_reasoning: reasoning, review: reviewer}` (in spec)
- `capabilities`: `[code, reasoning, review, planning]`

Auto-registered in catalog on server startup if not already present.

### Config Migration (migration.py)

`migrate_config(config_data)` transforms old-format configs:
1. Adds `catalog` section if missing.
2. Converts `grid_services.endpoints` to catalog service entries.
3. Sets `system.active_analyzer` to `"aurarouter-default"` if missing.
4. Never modifies `models` or `grid_services` sections.

`migrate_config_file(path, dry_run=False)` reads YAML, migrates, writes back.

CLI: `aurarouter migrate-config [--dry-run]`

## MCP Tools

### Core Routing
- `route_task(task, context, format)` -- IPE loop; consults active analyzer first
- `local_inference(prompt, context)` -- Local-only execution
- `generate_code(task_description, file_context, language)` -- Multi-step code gen
- `compare_models(prompt, models)` -- Side-by-side model comparison
- `list_models()` -- All configured models as JSON

### Asset Management
- `aurarouter.assets.list` -- Physical GGUF files
- `aurarouter.assets.register` -- Register local GGUF
- `aurarouter.assets.register_remote` -- Register remote endpoint
- `aurarouter.assets.unregister` -- Remove model from routing

### Catalog (new in 0.5.1)
- `aurarouter.catalog.list(kind?)` -- List artifacts (model/service/analyzer)
- `aurarouter.catalog.get(artifact_id)` -- Get single artifact
- `aurarouter.catalog.register(artifact_id, kind, display_name, ...)` -- Register artifact
- `aurarouter.catalog.remove(artifact_id)` -- Remove artifact

### Analyzer (new in 0.5.1)
- `aurarouter.analyzer.set_active(analyzer_id?)` -- Set/clear active analyzer
- `aurarouter.analyzer.get_active()` -- Get active analyzer ID

### Session (opt-in, `sessions.enabled: true`)
- `create_session`, `session_message`, `session_status`, `list_sessions`, `delete_session`

### Grid (opt-in, when `grid_services.endpoints` configured)
- `list_grid_services`, `list_remote_tools`, `call_remote_tool`

## auraconfig.yaml Schema

```yaml
system:
  log_level: INFO                        # DEBUG, INFO, WARNING, ERROR
  default_timeout: 120.0                 # Seconds
  active_analyzer: aurarouter-default    # Catalog artifact ID of active route analyzer

models:                                  # Legacy model definitions (still supported)
  model_id:
    provider: ollama | llamacpp-server | llamacpp | openapi
    endpoint: http://...                 # For HTTP providers
    model_path: /path/to.gguf            # For llamacpp
    model_name: ...
    tags: [coding, local]
    capabilities: [code, chat]
    cost_per_1m_input: 0.0
    cost_per_1m_output: 0.0
    hosting_tier: on-prem | cloud | dedicated-tenant
    parameters:
      temperature: 0.1
      num_ctx: 4096                      # Ollama
      n_ctx: 4096                        # llama.cpp

roles:                                   # Ordered fallback chains
  router: [model_a]
  reasoning: [model_b, model_a]
  coding: [model_a]
  reviewer: [model_a]                    # Optional — enables review loop

execution:
  max_review_iterations: 3               # 0 = disable review loop

catalog:                                 # Unified artifact catalog
  artifact_id:
    kind: model | service | analyzer
    display_name: ...
    description: ...
    provider: ...
    version: ...
    tags: [...]
    capabilities: [...]
    status: registered                   # Default
    # Kind-specific spec fields merged at top level:
    # Analyzer: analyzer_kind, role_bindings, mcp_endpoint, mcp_tool_name
    # Service: endpoint, protocol, auto_sync_models, health_check

xlm:                                     # AuraXLM integration (opt-in)
  endpoint: http://xlm-host:8080         # AuraXLM MCP endpoint
  features:
    prompt_augmentation: true             # Prepend RAG context via auraxlm.query
    usage_reporting: false                # Report token usage back to XLM

savings:                                 # Opt-in cost/privacy subsystem
  enabled: false
  db_path: ...
  triage:
    enabled: false
  budget:
    enabled: false
  privacy:
    enabled: true
  feedback:                              # Routing outcome feedback loop
    enabled: false
    db_path: ~/.auracore/aurarouter/feedback.db

sessions:
  enabled: false
  store_path: ...

grid_services:
  endpoints:
    - name: ...
      url: http://...
  auto_sync_models: true

mcp:
  tools:
    tool_name:
      enabled: true | false

provider_catalog:
  auto_start_entrypoints: true
  manual:
    - name: gemini
      endpoint: http://localhost:9001
      auto_start: true

semantic_verbs:                          # Custom tag-to-role synonym mappings
  coding: [code, program, develop]
```

## External Provider Packages

Cloud providers are distributed as separate MCP server packages, discovered via the `aurarouter.providers` entry-point group:

| Package | Provider | Install | Entry Point |
|---------|----------|---------|-------------|
| `aurarouter-claude` | Anthropic Claude (Opus 4, Sonnet 4, Haiku 4.5) | `pip install aurarouter-claude` | `claude = "aurarouter_claude"` |
| `aurarouter-gemini` | Google Gemini (2.5 Pro, 2.5 Flash, 2.0 Flash) | `pip install aurarouter-gemini` | `gemini = "aurarouter_gemini"` |

Both packages register under the `aurarouter.providers` entry-point group. `ProviderCatalog.discover()` finds them automatically when installed. Each exposes a `get_provider_metadata()` callable that returns a `ProviderMetadata`-compatible object.

Provider count: 4 built-in + 2 external MCP = 6 providers total.

## Conventions

- **Python 3.12+**, src-layout.
- Apache-2.0 license (open source under AuraCore Dynamics).
- `mcp[cli]`, `httpx`, `pyyaml`, `PySide6` are core deps. Everything else is optional.
- MCP tool functions in `mcp_tools.py` are stateless — they take `ComputeFabric`/`ConfigLoader` and return strings.
- Tool registration and wiring happens in `server.py` (`create_mcp_server`).
- `ConfigLoader` is the single source of truth for all YAML config access. No direct `config["key"]` outside of it.
- Providers implement `ProviderProtocol`.
- Test with `pytest tests/ -v --tb=short`. Comprehensive test suite with 130+ tests.

## Testing

```bash
# Full suite
pytest tests/ -v --tb=short

# Catalog tests specifically
pytest tests/test_catalog*.py -v --tb=short

# Config and migration
pytest tests/test_config*.py tests/test_migration*.py -v --tb=short

# MCP tools
pytest tests/test_mcp_tools*.py -v --tb=short

# XLM integration
pytest tests/test_xlm_integration.py -v --tb=short

# Feedback loop
pytest tests/test_feedback.py -v --tb=short

# Cross-project integration
pytest tests/test_cross_project_integration.py -v --tb=short
```

All catalog, config, and migration tests are self-contained with no external dependencies.
