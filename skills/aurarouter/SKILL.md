---
name: aurarouter
description: >-
  Manage, configure, test, and deploy AuraRouter — the multi-model MCP routing fabric
  for local and cloud LLMs. Use when routing prompts, managing models/catalogs,
  obtaining BLACK Task Group FMoE routing policies, running diagnostics, or
  installing/cleaning up AuraRouter.
---

# AuraRouter: Multi-Model MCP Routing Fabric

**AuraRouter** routes inference tasks across local and cloud LLMs, exposes an MCP server (including deterministic BLACK Task Group FMoE policy derivation), and includes a PySide6 GUI and AuraGrid MAS host.

---

## 1. Zero-Configuration Bootstrappers (No `PYTHONPATH` Required)

Run directly from the source checkout on any OS:

```powershell
# Windows PowerShell
& ".\aurarouter.ps1" --help
```

```bash
# Linux / macOS
./aurarouter.sh --help
```

### Installing as a User-Space App (via AuraSprout)
AuraRouter includes an `app.manifest.json` and can be installed/uninstalled cleanly via AuraSprout:
```powershell
aurasprout install <aurarouter_repo_dir> --shortcut --json
aurasprout uninstall aurarouter --remove-mvenv --json
```

---

## 2. Common Agent Workflows & CLI Subcommands (Use `--json`)

`aurarouter` supports `--json` globally before the subcommand for machine-readable output:

```powershell
# Readiness, diagnostics, GUI, and IPC server
aurarouter --json doctor
aurarouter gui
aurarouter serve-ipc

# Model lifecycle: list, add, edit, remove, test, auto-tune (plus legacy GGUF helpers)
aurarouter --json model list
aurarouter --json model add <id> --provider ollama --endpoint http://127.0.0.1:11434/api/generate
aurarouter --json model edit <id> --endpoint http://127.0.0.1:11434/api/generate
aurarouter --json model remove <id>
aurarouter --json model test <id>
aurarouter --json model auto-tune <id>
aurarouter --json download-model --repo <repo> --file <file>
aurarouter --json list-models
aurarouter --json remove-model --file <file>

# Role chains: list, set, append, remove-model, delete
aurarouter --json route list
aurarouter --json route set coding <model_id>
aurarouter --json route append coding <model_id>
aurarouter --json route remove-model coding <model_id>
aurarouter --json route delete <role>

# Execution, comparison, and intent inspection
aurarouter --json run "<prompt>" --intent generate_code
aurarouter --json compare "<prompt>" --models <m1>,<m2>
aurarouter --json intent list
aurarouter --json intent describe generate_code

# Analyzers: list, active, set, clear
aurarouter --json analyzer list
aurarouter --json analyzer active
aurarouter --json analyzer set <analyzer_id>
aurarouter --json analyzer clear

# Unified catalog & provider lifecycle
aurarouter --json catalog list
aurarouter --json catalog artifacts
aurarouter --json catalog get <artifact_id>
aurarouter --json catalog register <artifact_id> --file <spec.json>
aurarouter --json catalog unregister <artifact_id>
aurarouter --json catalog add <name> --endpoint <url>
aurarouter --json catalog remove <name>
aurarouter --json catalog start <name>
aurarouter --json catalog stop <name>
aurarouter --json catalog health
aurarouter --json catalog discover <name>

# Telemetry, privacy, health, budget, and configuration management
aurarouter --json health
aurarouter --json traffic
aurarouter --json privacy
aurarouter --json budget
aurarouter --json config show
aurarouter --json config set <key> <value>
aurarouter --json config mcp-tool <tool_name> --enable
aurarouter --json config save
aurarouter --json config reload
aurarouter --json migrate-config --dry-run
```

<!-- BLACK CLI command parity index:
black active, black add, black analyzer, black append, black artifacts, black auto-tune,
black budget, black catalog, black clear, black compare, black config, black delete,
black describe, black discover, black doctor, black download-model, black edit, black get,
black gui, black health, black intent, black list, black list-models, black mcp-tool,
black migrate-config, black model, black privacy, black register, black reload, black remove,
black remove-model, black route, black run, black save, black serve-ipc, black set,
black show, black start, black stop, black test, black traffic, black unregister
-->

---

## 3. Key MCP Tools (Routing, Skills & BLACK FMoE Policy)

- **Core Routing**: `route_task`, `local_inference`, `generate_code`, `compare_models`, `list_models`
- **BLACK FMoE Routing Policy**: `aurarouter.routing.recommend_for_taskgroup` (maps a BLACK `scoring.json` Task Group entry to deterministic `recommended_intent`, `recommended_role`, `required_capabilities`, `hard_routed`/`local_only`, and `max_remediation_attempts`)
- **Stacked Routing & Rate Limits**: `aurarouter.routing.stack_status`, `aurarouter.routing.explain`, `aurarouter.ratelimit.status`
- **Companion Skills**: `aurarouter.skills.list`, `aurarouter.skills.read`
- **Assets & Catalog**: `aurarouter.assets.list`, `aurarouter.assets.register`, `aurarouter.assets.register_remote`, `aurarouter.assets.unregister`, `aurarouter.catalog.list`, `aurarouter.catalog.get`, `aurarouter.catalog.register`, `aurarouter.catalog.remove`
- **Intents, Analyzers & Governance**: `aurarouter.intents.list` (`list_intents`), `aurarouter.analyzer.get_active`, `aurarouter.analyzer.set_active`, `aurarouter.sovereignty.status`, `aurarouter.rag.status`, `aurarouter.budget.status`, `aurarouter.budget.report_sync`, `aurarouter.budget.global`
- **Benchmarks, Speculative & Monologue**: `aurarouter.benchmarks.model`, `aurarouter.benchmarks.compare`, `aurarouter.benchmarks.pareto`, `aurarouter.speculative.execute`, `aurarouter.speculative.status`, `aurarouter.monologue.execute`, `aurarouter.monologue.status`, `aurarouter.monologue.trace`

---

## 4. State & Cleanup Locations

- **User config / state**: `~/.auracore/` (e.g., `~/.auracore/aurarouter/`, `~/.auracore/models/`)
- **Config file**: `auraconfig.yaml` (or `--config <path>`)
- **Clean uninstall**: `aurasprout uninstall aurarouter --remove-mvenv --json` (if installed via AuraSprout) or `pip uninstall -y aurarouter` and remove `~/.auracore/aurarouter`.
