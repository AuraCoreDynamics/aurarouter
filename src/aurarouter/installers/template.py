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

  # ----- llama.cpp (embedded, no Ollama needed) -----
  # local_llama_qwen:
  #   provider: llamacpp
  #   model_path: "C:/models/qwen2.5-coder-7b-q4_k_m.gguf"
  #   parameters:
  #     n_ctx: 4096
  #     n_gpu_layers: -1      # -1 = all layers to GPU
  #     n_batch: 512
  #     temperature: 0.1
  #     max_tokens: 2048

  # ----- Google Gemini (cloud) -----
  cloud_gemini_flash:
    provider: google
    model_name: gemini-2.0-flash
    api_key: YOUR_API_KEY
    # env_key: GOOGLE_API_KEY

  cloud_gemini_pro:
    provider: google
    model_name: gemini-2.0-pro-exp
    api_key: YOUR_API_KEY

  # ----- Anthropic Claude (cloud) -----
  # cloud_claude_sonnet:
  #   provider: claude
  #   model_name: claude-sonnet-4-5-20250929
  #   env_key: ANTHROPIC_API_KEY
  #   parameters:
  #     max_tokens: 4096
  #     temperature: 0.1

# --- ROLES & ROUTING ---
# The router iterates each list until one model succeeds.
roles:
  router:
    - local_3070_qwen
    - cloud_gemini_flash

  reasoning:
    - cloud_gemini_pro
    - cloud_gemini_flash

  coding:
    - local_3070_qwen
    - cloud_gemini_flash
"""


def create_config_template() -> None:
    """Write a template auraconfig.yaml to ~/.auracore/aurarouter/ if missing."""
    target_dir = Path.home() / ".auracore" / "aurarouter"
    target = target_dir / "auraconfig_template.yaml"

    print("\n   Looking for configuration template...")

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(_TEMPLATE)
            print(f"   Template created at: {target}")
            print("   Rename to 'auraconfig.yaml' and edit it to configure your models.")
        else:
            print(f"   Template already exists at: {target}, skipping.")
    except Exception as e:
        print(f"   Error handling template file: {e}")
