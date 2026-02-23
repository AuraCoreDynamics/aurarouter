# AuraRouter Deployment Guide

## Deployment Options

AuraRouter supports three deployment contexts:

1. **Standalone Python** - Install via PyPI or from source. Feature-complete with durable configuration.
2. **AuraGrid MAS** - Deploy as a managed application service on AuraGrid. See [AURAGRID.md](AURAGRID.md).
3. **Conda** - Isolated environment via conda/mamba. Uses the same package.

This guide focuses on **standalone Python deployment**.

---

## Prerequisites

- Python 3.12+
- For local inference: [Ollama](https://ollama.ai) and/or [llama.cpp](https://github.com/ggerganov/llama.cpp)
- For cloud providers: API keys for Google AI Studio and/or Anthropic

## Installation

### Option A: PyPI Install

```bash
# Core (MCP server + GUI + cloud providers + llamacpp-server HTTP provider)
pip install aurarouter

# Add embedded llama.cpp inference and HuggingFace model downloads
pip install aurarouter[local]

# Install everything (local + AuraGrid + dev tools)
pip install aurarouter[all]
```

**GPU acceleration for embedded llama.cpp:**
Pre-built CPU wheels install automatically. For CUDA GPU support:

```bash
# CUDA 12.4
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# CUDA 11.8
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118
```

### Option B: Source Install

```bash
git clone https://github.com/auracoredynamics/aurarouter.git
cd aurarouter

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install local inference dependencies
pip install -r requirements-local.txt

# Install the package in editable mode
pip install -e .
```

### Option C: Conda

```bash
conda env create -f environment.yaml
conda activate aurarouter
```

---

## Configuration

### Config File Location

AuraRouter searches for `auraconfig.yaml` in this order:

1. `--config` CLI argument
2. `AURACORE_ROUTER_CONFIG` environment variable
3. `~/.auracore/aurarouter/auraconfig.yaml`

### Creating Initial Configuration

**Interactive:**
```bash
aurarouter --install
```
This creates `~/.auracore/aurarouter/auraconfig.yaml` (if it doesn't exist) and offers to register AuraRouter with MCP clients.

**Manual:** Create `~/.auracore/aurarouter/auraconfig.yaml`:

```yaml
system:
  log_level: INFO
  default_timeout: 120.0

models:
  # Local models (via Ollama)
  local_qwen:
    provider: ollama
    endpoint: http://localhost:11434/api/generate
    model_name: qwen2.5-coder:7b
    parameters:
      temperature: 0.1
      num_ctx: 4096

  # Local models (via llama-server HTTP - no native deps)
  # local_llama_server:
  #   provider: llamacpp-server
  #   endpoint: http://localhost:8080
  #   parameters:
  #     temperature: 0.1
  #     n_predict: 2048

  # Local models (embedded llama.cpp - requires aurarouter[local])
  # local_llama_embedded:
  #   provider: llamacpp
  #   model_path: "/path/to/model.gguf"
  #   parameters:
  #     n_ctx: 4096
  #     n_gpu_layers: -1
  #     temperature: 0.1

  # Cloud models
  cloud_gemini_flash:
    provider: google
    model_name: gemini-2.0-flash
    api_key: YOUR_GOOGLE_API_KEY
    tags: [coding, reasoning]
    # Or use env var: env_key: GOOGLE_API_KEY

  # cloud_claude:
  #   provider: claude
  #   model_name: claude-sonnet-4-5-20250929
  #   env_key: ANTHROPIC_API_KEY

  # OpenAPI-compatible endpoint (vLLM, LocalAI, LM Studio, etc.)
  # my_vllm:
  #   provider: openapi
  #   endpoint: http://localhost:8000/v1
  #   model_name: meta-llama/Llama-3-8B
  #   tags: [private, coding]

roles:
  router:    [local_qwen, cloud_gemini_flash]
  reasoning: [cloud_gemini_flash]
  coding:    [local_qwen, cloud_gemini_flash]

# Optional: map synonyms to canonical role names for intent classification
# semantic_verbs:
#   coding:
#     synonyms: [programming, code generation, developer]
#   reasoning:
#     synonyms: [planner, architect, planning]
```

### Configuration Persistence

Configuration changes made through the GUI or the `ConfigLoader` API are saved atomically back to the YAML file. Changes survive restarts.

### Environment Variable Overrides

Any config key can be overridden via `AURAROUTER_*` environment variables using `__` for nesting:

```bash
# Override a model endpoint
export AURAROUTER_MODELS__LOCAL_QWEN__ENDPOINT=http://192.168.1.100:11434/api/generate

# Override API keys
export GOOGLE_API_KEY=AIzaSy...
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Provider Setup

### Ollama (Local HTTP)

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull qwen2.5-coder:7b`
3. Configure in YAML:
   ```yaml
   local_model:
     provider: ollama
     endpoint: http://localhost:11434/api/generate
     model_name: qwen2.5-coder:7b
   ```

### llama.cpp Server (Local HTTP, Zero Native Deps)

1. Download `llama-server` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
2. Download a GGUF model (e.g., via `aurarouter download-model`)
3. Start the server: `llama-server -m model.gguf --port 8080`
4. Configure in YAML:
   ```yaml
   local_llama:
     provider: llamacpp-server
     endpoint: http://localhost:8080
   ```

### llama.cpp Embedded (Local Native)

Requires `pip install aurarouter[local]`.

1. Download a GGUF model:
   ```bash
   aurarouter download-model \
     --repo Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
     --file qwen2.5-coder-7b-instruct-q4_k_m.gguf
   ```
2. Configure in YAML:
   ```yaml
   local_embedded:
     provider: llamacpp
     model_path: ~/.auracore/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
     parameters:
       n_ctx: 4096
       n_gpu_layers: -1
   ```

### Google Gemini (Cloud)

1. Get an API key from [Google AI Studio](https://aistudio.google.com)
2. Configure in YAML:
   ```yaml
   cloud_gemini:
     provider: google
     model_name: gemini-2.0-flash
     api_key: YOUR_KEY
   ```
   Or use an environment variable: `env_key: GOOGLE_API_KEY`

### Anthropic Claude (Cloud)

1. Get an API key from [Anthropic Console](https://console.anthropic.com)
2. Configure in YAML:
   ```yaml
   cloud_claude:
     provider: claude
     model_name: claude-sonnet-4-5-20250929
     env_key: ANTHROPIC_API_KEY
   ```

### OpenAPI-Compatible (Local/Cloud HTTP)

Works with any endpoint implementing the OpenAI chat completions API: vLLM, text-generation-inference, LocalAI, LM Studio, etc.

1. Start your OpenAI-compatible server (e.g., `vllm serve meta-llama/Llama-3-8B`)
2. Configure in YAML:
   ```yaml
   my_vllm:
     provider: openapi
     endpoint: http://localhost:8000/v1
     model_name: meta-llama/Llama-3-8B
     # api_key: optional-key  # or env_key: VLLM_API_KEY
     parameters:
       temperature: 0.7
       max_tokens: 2048
   ```

The provider sends requests to `{endpoint}/chat/completions` using the standard OpenAI request/response format. Locality (local vs cloud) is inferred from the endpoint address -- `localhost`/`127.0.0.1` endpoints are treated as local; others as cloud. You can override this with an explicit `locality: local` or `locality: cloud` field.

---

## Running

### MCP Server (Default)

```bash
aurarouter                                    # Uses default config location
aurarouter --config /path/to/auraconfig.yaml  # Explicit config
python -m aurarouter                          # Module invocation
```

### Desktop GUI

```bash
aurarouter gui                                         # Via main CLI
aurarouter gui --config /path/to/auraconfig.yaml       # With config
aurarouter gui --environment auragrid                  # Start in AuraGrid mode
aurarouter-gui                                         # Standalone entry point
aurarouter-gui --config /path/to/auraconfig.yaml       # Standalone with config
```

#### GUI Administration

The GUI provides full service lifecycle management:

- **Environment selector**: Switch between Local and AuraGrid at runtime (toolbar dropdown).
- **Service controls**: Start/Stop/Pause buttons manage the MCP server subprocess (Local) or MAS lifecycle (AuraGrid).
- **Health dashboard**: Click the health indicator to see per-model status. Use "Check" to run diagnostics.
- **Document upload**: Attach files as context for tasks via the Execute tab.
- **DAG visualization**: Expandable execution trace showing the classify/plan/execute pipeline with per-node telemetry.
- **Privacy-aware routing**: Automatically re-routes PII-containing prompts to local/private-tagged models.
- **Prompt history**: Recent tasks are saved and restorable from a dropdown.
- **Keyboard shortcuts**: Ctrl+Enter (execute), Ctrl+N (new prompt), Escape (cancel).

See [GUI_GUIDE.md](GUI_GUIDE.md) for the complete GUI reference.

### Model Management

```bash
# Download a GGUF model from HuggingFace
aurarouter download-model \
  --repo Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  --file qwen2.5-coder-7b-instruct-q4_k_m.gguf

# List locally downloaded models
aurarouter list-models

# Remove a downloaded model (deletes file)
aurarouter remove-model --file qwen2.5-coder-7b-instruct-q4_k_m.gguf

# Remove from registry only, keep the file
aurarouter remove-model --file model.gguf --keep-file
```

Downloaded models are stored in `~/.auracore/models/` by default, with a `models.json` registry tracking metadata (repo, filename, size, download date).

---

## Data Directories

| Path | Purpose |
|------|---------|
| `~/.auracore/aurarouter/auraconfig.yaml` | Runtime configuration |
| `~/.auracore/aurarouter/aurarouter.pid` | Singleton PID lock file |
| `~/.auracore/aurarouter/history.json` | GUI prompt history (last 20 tasks + results) |
| `~/.auracore/models/` | Downloaded GGUF model files |
| `~/.auracore/models/models.json` | Model registry (auto-managed) |

---

## Programmatic Usage

```python
from aurarouter.config import ConfigLoader
from aurarouter.fabric import ComputeFabric

# Load config
config = ConfigLoader(config_path="auraconfig.yaml")

# Execute tasks (AuraRouter is content-agnostic)
fabric = ComputeFabric(config)
result = fabric.execute("coding", "Write a hello world function in Python")
result = fabric.execute("coding", "Summarize the key findings from the attached report")

# Modify and save config
config.set_model("new_model", {"provider": "ollama", "model_name": "llama3"})
config.set_role_chain("coding", ["new_model", "cloud_gemini"])
config.save()
```
