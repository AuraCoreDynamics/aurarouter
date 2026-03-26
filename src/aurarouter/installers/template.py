from pathlib import Path

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Template")

_TEMPLATE = """\
# AuraRouter Configuration
# Run the Setup Wizard (GUI) or edit this file manually.

system:
  log_level: INFO
  default_timeout: 120.0
  active_analyzer: aurarouter-default

# Add your models here. The Setup Wizard can help.
models: {}

# Assign models to roles. Each role uses an ordered fallback chain.
roles:
  router: []
  reasoning: []
  coding: []

execution:
  max_review_iterations: 3

mcp:
  tools:
    route_task:
      enabled: true
    local_inference:
      enabled: true
    generate_code:
      enabled: true
    compare_models:
      enabled: false

catalog:
  aurarouter-default:
    kind: analyzer
    display_name: AuraRouter Default
    description: Built-in intent classifier with complexity-based triage
    provider: aurarouter
    analyzer_kind: intent_triage
    capabilities: [code, reasoning, review, planning]
    role_bindings:
      simple_code: coding
      complex_reasoning: reasoning
      review: reviewer
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
