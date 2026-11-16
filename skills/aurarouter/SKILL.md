---
name: aurarouter
description: >-
  Manage, configure, test, and deploy AuraRouter — the multi-model MCP routing fabric
  for local and cloud LLMs. Use when routing prompts, managing models/catalogs,
  running diagnostics, or installing/cleaning up AuraRouter.
---

# AuraRouter: Multi-Model MCP Routing Fabric

**AuraRouter** routes inference tasks across local and cloud LLMs, exposes an MCP server, and includes a PySide6 GUI and AuraGrid MAS host.

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

## 2. Common Agent Workflows (Use `--json`)

`aurarouter` supports `--json` globally before the subcommand for machine-readable output:

```powershell
# Check config, runtime dependencies, and readiness
aurarouter --json doctor

# List configured models and routing roles
aurarouter --json model list
aurarouter --json route list

# Check model health, traffic, and budget
aurarouter --json health
aurarouter --json traffic
aurarouter --json budget

# Run a task through the routing loop
aurarouter --json run "<prompt>"

# Manage catalog providers & artifacts
aurarouter --json catalog list
aurarouter --json catalog artifacts
```

---

## 3. State & Cleanup Locations

- **User config / state**: `~/.auracore/` (e.g., `~/.auracore/aurarouter/`, `~/.auracore/models/`)
- **Config file**: `auraconfig.yaml` (or `--config <path>`)
- **Clean uninstall**: `aurasprout uninstall aurarouter --remove-mvenv --json` (if installed via AuraSprout) or `pip uninstall -y aurarouter` and remove `~/.auracore/aurarouter`.

