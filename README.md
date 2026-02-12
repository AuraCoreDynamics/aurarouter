# AuraRouter: The AuraXLM-Lite Compute Fabric

**Current Status:** Production Prototype v3 (Feb 2026)
**Maintainer:** Steven Siebert / AuraCore Dynamics

## Overview

AuraRouter implements a role-based configurable xLM (SLM/TLM/LLM) prompt routing fabric. It acts as intelligent middleware that routes code generation tasks across local and cloud models with automatic fallback. AuraRouter can run as an MCP server, a desktop GUI application, or a managed service on AuraGrid.

It implements an **Intent -> Plan -> Execute** loop:
1.  **Router:** A fast local model classifies the task (Simple vs. Complex).
2.  **Architect:** If complex, a reasoning model generates a sequential execution plan.
3.  **Worker:** A coding model executes the plan step-by-step.

## Architecture

```mermaid
graph TD
    User[MCP Client / GUI] -->|Task| Router{Intent Analysis}
    Router -->|Simple| Worker[Coding Node]
    Router -->|Complex| Architect[Reasoning Node]
    Architect -->|Plan JSON| Worker

    subgraph Compute Fabric [auraconfig.yaml]
        Worker -->|Try| Node1[Local Model]
        Node1 -->|Fail| Node2[Cloud Fallback]
    end
```

## Installation

### PyPI (Recommended)

```bash
# Core install (MCP server + GUI + cloud providers + llamacpp-server HTTP provider)
pip install aurarouter

# With embedded llama.cpp + HuggingFace model downloading
pip install aurarouter[local]

# Everything (local + AuraGrid + dev tools)
pip install aurarouter[all]
```

### Source Install

```bash
git clone https://github.com/auracoredynamics/aurarouter.git
cd aurarouter
pip install -r requirements.txt        # Core dependencies
pip install -r requirements-local.txt   # Optional: local inference deps
pip install -e .                        # Editable install
```

### Conda

```bash
conda env create -f environment.yaml
conda activate aurarouter
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment and configuration guide.

## Quick Start

### 1. Configuration

Run the interactive installer to create a config template:

```bash
aurarouter --install
```

Or manually create `~/.auracore/aurarouter/auraconfig.yaml`:

```yaml
models:
  local_qwen:
    provider: ollama
    endpoint: http://localhost:11434/api/generate
    model_name: qwen2.5-coder:7b

  cloud_gemini:
    provider: google
    model_name: gemini-2.0-flash
    api_key: "AIzaSy..."

roles:
  router:   [local_qwen, cloud_gemini]
  reasoning: [cloud_gemini]
  coding:   [local_qwen, cloud_gemini]
```

### 2. Run

```bash
# MCP server (default)
aurarouter

# Desktop GUI
aurarouter gui

# With explicit config
aurarouter --config /path/to/auraconfig.yaml
```

## Providers

| Provider | Type | Config Key | Dependencies |
|----------|------|------------|--------------|
| Ollama | Local HTTP | `ollama` | None (uses httpx) |
| llama.cpp Server | Local HTTP | `llamacpp-server` | None (uses httpx) |
| llama.cpp Embedded | Local Native | `llamacpp` | `pip install aurarouter[local]` |
| Google Gemini | Cloud | `google` | Included |
| Anthropic Claude | Cloud | `claude` | Included |

## GUI

The desktop GUI (included in the base install) provides:

- **Execute tab** - Task input, intent analysis, plan visualization, code output
- **Configuration tab** - Model management, routing rule editor, connection testing, live YAML preview

All configuration changes made in the GUI are persisted to `auraconfig.yaml`.

## CLI Commands

| Command | Description |
|---------|-------------|
| `aurarouter` | Run MCP server (default) |
| `aurarouter gui` | Launch desktop GUI |
| `aurarouter download-model --repo REPO --file FILE` | Download GGUF model from HuggingFace |
| `aurarouter --install` | Interactive installer for MCP clients |
| `aurarouter --install-gemini` | Register for Gemini CLI |
| `aurarouter --install-claude` | Register for Claude |

## AuraGrid Integration (Optional)

AuraRouter can be deployed as a **Managed Application Service (MAS)** on [AuraGrid](https://github.com/auracoredynamics/auragrid) for distributed access to routing services.

```bash
pip install aurarouter[auragrid]
```

See [AURAGRID.md](AURAGRID.md) for the complete integration guide.

## Scaling Guide

When you add new on-prem xLM resources:

1. Open `auraconfig.yaml` (or use the GUI Configuration tab).
2. Add the new model under `models`.
3. Add it to the appropriate role chain under `roles`.
4. Restart the router (or save from the GUI). **No code changes required.**

## Troubleshooting

* **"Empty response received":** The local model is likely OOMing or timing out. Check the `timeout` setting in `auraconfig.yaml`.
* **"Model not found":** Ensure the `model_name` in YAML matches `ollama list` exactly.
* **"huggingface-hub is required":** Run `pip install aurarouter[local]` to enable model downloading and embedded llama.cpp.
* **PySide6 issues on headless servers:** PySide6 is a core dependency. On headless/server-only deployments, use the MCP server mode (`aurarouter`) which does not launch the GUI.

## License

Copyright 2026 AuraCore Dynamics Inc.
