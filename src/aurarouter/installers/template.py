from pathlib import Path

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Template")

_TEMPLATE = """\
# --- SYSTEM SETTINGS ---
system:
  log_level: INFO
  default_timeout: 120.0

# --- HARDWARE INVENTORY ---
models:
  # ----- Ollama (local HTTP) -----
  local_3070_qwen:
    provider: ollama
    endpoint: http://localhost:11434/api/generate
    model_name: qwen2.5-coder:7b
    parameters:
      temperature: 0.1
      num_ctx: 4096

  # ----- llama.cpp via HTTP server (no native deps needed) -----
  # Requires llama-server running externally: llama-server -m model.gguf
  # local_llama_server:
  #   provider: llamacpp-server
  #   endpoint: http://localhost:8080
  #   parameters:
  #     temperature: 0.1
  #     n_predict: 2048

  # ----- llama.cpp embedded (requires pip install aurarouter[local]) -----
  # local_llama_qwen:
  #   provider: llamacpp
  #   model_path: "C:/models/qwen2.5-coder-7b-q4_k_m.gguf"
  #   parameters:
  #     n_ctx: 4096
  #     n_gpu_layers: -1      # -1 = all layers to GPU
  #     n_batch: 512
  #     temperature: 0.1
  #     max_tokens: 2048

  # ----- OpenAPI-compatible (vLLM, LocalAI, LM Studio, etc.) -----
  # my_openapi:
  #   provider: openapi
  #   endpoint: http://localhost:8000/v1
  #   model_name: my-model
  #   # api_key: YOUR_API_KEY
  #   # env_key: MY_API_KEY

# --- ROLES & ROUTING ---
# The router iterates each list until one model succeeds.
roles:
  router:
    - local_3070_qwen

  reasoning:
    - local_3070_qwen

  coding:
    - local_3070_qwen

# --- MCP TOOL SURFACE ---
# Controls which tools AuraRouter exposes to MCP host models.
# Each tool can be independently enabled or disabled.
mcp:
  tools:
    route_task:
      enabled: true        # General-purpose router — local/cloud with fallback
    local_inference:
      enabled: true        # Privacy-preserving — local models only, no cloud
    generate_code:
      enabled: true        # Multi-step code gen with planning
    compare_models:
      enabled: false       # Run prompt across all models for comparison
"""


def create_config_template() -> None:
    """Write a starter auraconfig.yaml to ~/.auracore/aurarouter/ if missing."""
    target_dir = Path.home() / ".auracore" / "aurarouter"
    target = target_dir / "auraconfig.yaml"

    print("\n   Looking for configuration file...")

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(_TEMPLATE)
            print(f"   Config created at: {target}")
            print("   Edit it to add your API keys and model endpoints.")
        else:
            print(f"   Config already exists at: {target}, skipping.")
    except Exception as e:
        print(f"   Error creating config file: {e}")
