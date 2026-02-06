import os
import sys
import yaml
import json
import httpx
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from google import genai

# --- Boot Sequence ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AuraRouter")
logger.setLevel(logging.INFO)

def get_config_path_arg():
    """A lightweight parser to find the --config argument before full setup."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to the auraconfig.yaml file.")
    known_args, _ = parser.parse_known_args()
    return known_args.config


# --- Configuration Manager ---
class ConfigLoader:
    def __init__(self, config_path_arg: Optional[str] = None):
        # If we are just installing, we don't crash if config is missing
        if any(arg in sys.argv for arg in ["--install", "--install-gemini", "--install-claude"]):
            self.config = {}
            return

        config_path = self._find_config(config_path_arg)

        if not config_path:
            searched_paths = []
            if config_path_arg:
                searched_paths.append(f"  - Command line (--config): {Path(config_path_arg).resolve()}")
            
            env_var = "AURACORE_ROUTER_CONFIG"
            env_path_str = os.environ.get(env_var)
            if env_path_str:
                searched_paths.append(f"  - Environment variable ({env_var}): {Path(env_path_str).resolve()}")
            
            searched_paths.append(f"  - User home directory: {Path.home() / '.auracore' / 'aurarouter' / 'auraconfig.yaml'}")

            error_msg = "Could not find 'auraconfig.yaml'. Searched in the following locations:\n" + "\n".join(searched_paths)
            raise FileNotFoundError(error_msg)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from: {config_path.resolve()}")

    def _find_config(self, config_path_arg: Optional[str]) -> Optional[Path]:
        """Finds config by searching in a prioritized sequence."""
        # 1. Check path from --config argument
        if config_path_arg:
            path = Path(config_path_arg)
            logger.info(f"Attempting to load config from command line path: {path.resolve()}")
            if path.is_file():
                return path
            else:
                logger.warning(f"Config file not found at path specified by --config: {path.resolve()}")

        # 2. Check path from AURACORE_ROUTER_CONFIG environment variable
        env_path_str = os.environ.get("AURACORE_ROUTER_CONFIG")
        if env_path_str:
            path = Path(env_path_str)
            logger.info(f"Attempting to load config from AURACORE_ROUTER_CONFIG env var: {path.resolve()}")
            if path.is_file():
                return path
            else:
                logger.warning(f"Config file not found at path specified by AURACORE_ROUTER_CONFIG: {path.resolve()}")

        # 3. Check user's home directory
        home_path = Path.home() / ".auracore" / "aurarouter" / "auraconfig.yaml"
        logger.info(f"Attempting to load config from user home directory: {home_path.resolve()}")
        if home_path.is_file():
            return home_path

        return None

    def get_role_chain(self, role: str) -> List[str]:
        return self.config.get('roles', {}).get(role, [])
        
    def get_model_config(self, model_id: str) -> Dict:
        return self.config.get('models', {}).get(model_id, {})

try:
    # Parse the --config argument before initializing anything else
    config_path_from_arg = get_config_path_arg()
    config = ConfigLoader(config_path_arg=config_path_from_arg)
except Exception as e:
    logger.critical(f"Failed to load config: {e}")
    sys.exit(1)

# Initialize MCP only if not installing (prevents side effects during setup)
IS_INSTALL_MODE = any(arg in sys.argv for arg in ["--install", "--install-gemini", "--install-claude"])
if not IS_INSTALL_MODE:
    mcp = FastMCP("AuraRouter")

# --- Installer Logic (Robust & Interactive) ---

def _generic_mcp_installer(server_name: str, display_name: str, extra_args: Optional[List[str]] = None):
    """
    A generic installer to register an MCP server in the Gemini CLI's settings.json.
    """
    if extra_args is None:
        extra_args = []

    print(f"\n🔧 AuraRouter ({display_name}) Installer")
    print("=======================")

    # 1. Detect Environment
    current_python = sys.executable
    script_path = os.path.abspath(__file__)
    
    print(f"   📍 Python Interpreter: {current_python}")
    print(f"   📍 Router Script:      {script_path}")

    # 2. Intelligent Path Detection for Gemini CLI's settings.json
    home = Path.home()
    
    # Priority list of common locations
    candidates = [
        home / ".gemini" / "settings.json",      # User preference
        home / ".geminichat" / "settings.json",  # Standard Node CLI
        home / ".geminichat" / "config.json",    # Legacy
        home / "gemini-cli" / "settings.json",   # Manual install
    ]
    
    detected_path = None
    for p in candidates:
        if p.exists():
            detected_path = p
            print(f"   ✅ Auto-detected Gemini CLI config file at: {p}")
            break
            
    # Default fallback if nothing found
    default_path = detected_path if detected_path else (home / ".gemini" / "settings.json")

    # 3. User Confirmation
    print("\n   Where is your Gemini CLI settings.json file?")
    user_input = input(f"   [Press ENTER to use: {default_path}]: ").strip()
    
    target_path = Path(os.path.expanduser(user_input)) if user_input else default_path
    
    # 4. Validation
    if not target_path.parent.exists():
        print(f"\n   ❌ Error: The directory '{target_path.parent}' does not exist.")
        print("   Please run the Gemini CLI once to generate its folders, or create the directory manually.")
        return

    print(f"   📂 Targeting: {target_path}")

    # 5. Injection Logic
    try:
        data = {}
        if target_path.exists():
            try:
                with open(target_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
            except json.JSONDecodeError:
                print("   ⚠️  File exists but contains invalid JSON. Backing up and starting fresh.")
                target_path.rename(target_path.with_suffix(".json.bak"))
                data = {}
        
        # Ensure block exists
        if "mcpServers" not in data:
            data["mcpServers"] = {}
            
        # The Payload
        data["mcpServers"][server_name] = {
            "command": current_python,
            "args": [script_path] + extra_args,
            "env": {
                "PYTHONUNBUFFERED": "1" # Ensure logs flush immediately
            }
        }
        
        # Write
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n   ✅ SUCCESS: AuraRouter ({display_name}) registered successfully.")
        print("   🔁 Action Required: Restart your Gemini CLI terminal session.")
        
    except Exception as e:
        print(f"\n   ❌ FATAL: {display_name} Installation Failed: {e}")


def install_gemini_router():
    """Auto-registers this script into the Gemini CLI settings.json for Gemini routing."""
    _generic_mcp_installer(server_name="aurarouter", display_name="Gemini")


def install_claude_router():
    """Auto-registers this script into the Gemini CLI settings.json for Claude routing."""
    _generic_mcp_installer(
        server_name="clauderouter",
        display_name="Claude",
        extra_args=["--claude-mode"]
    )


def install_all():
    """
    Iterates through all available installers, allowing the user to select.
    """
    print("\n🔧 AuraRouter Interactive Installer")
    print("==================================")
    
    installers = {
        'Gemini': install_gemini_router,
        'Claude': install_claude_router
    }

    for name, installer_func in installers.items():
        while True:
            choice = input(f"\n   Install support for {name}? [Y]es, [N]o, [Q]uit: ").lower().strip()
            if choice in ['y', 'yes']:
                installer_func()
                break
            elif choice in ['n', 'no', 'skip']:
                print(f"   Skipping {name} installation.")
                break
            elif choice in ['q', 'quit']:
                print("   Aborting installation.")
                return
            else:
                print("   Invalid choice. Please enter Y, N, or Q.")


# --- The Compute Fabric ---

class ComputeFabric:
    """
    Handles N-Model routing with graceful degradation.
    """
    
    def _resolve_api_key(self, cfg: Dict) -> str:
        if cfg.get('api_key') and "YOUR_PASTED_KEY" not in cfg['api_key']:
            return cfg['api_key']
        if cfg.get('env_key'):
            return os.environ.get(cfg['env_key'])
        return None

    def _call_ollama(self, cfg: Dict, prompt: str, json_mode: bool) -> str:
        url = cfg.get('endpoint', 'http://localhost:11434/api/generate')
        payload = {
            "model": cfg['model_name'],
            "prompt": prompt,
            "stream": False,
            "options": cfg.get('parameters', {})
        }
        if json_mode: payload['format'] = 'json'
        
        timeout = cfg.get('timeout', 120.0)
        
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    def _call_google(self, cfg: Dict, prompt: str) -> str:
        api_key = self._resolve_api_key(cfg)
        if not api_key: 
            raise ValueError(f"No API Key found for {cfg.get('model_name')}")
        
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=cfg['model_name'],
            contents=prompt
        )
        return resp.text

    def _call_claude(self, cfg: Dict, prompt: str, json_mode: bool) -> str:
        # Placeholder for Claude API call
        raise NotImplementedError("Claude API integration is not yet implemented.")

    def execute(self, role: str, prompt: str, json_mode: bool = False) -> str:
        chain = config.get_role_chain(role)
        if not chain:
            return f"ERROR: No models defined for role '{role}' in YAML."

        errors = []
        for model_id in chain:
            model_cfg = config.get_model_config(model_id)
            if not model_cfg: continue

            provider = model_cfg.get('provider')
            logger.info(f"🔄 [{role.upper()}] Routing to: {model_id} ({provider})")
            
            try:
                result = None
                if provider == 'ollama':
                    result = self._call_ollama(model_cfg, prompt, json_mode)
                elif provider == 'google':
                    result = self._call_google(model_cfg, prompt)
                elif provider == 'claude':
                    result = self._call_claude(model_cfg, prompt, json_mode)
                
                if result and len(str(result).strip()) > 5:
                    logger.info(f"✅ [{role.upper()}] Success.")
                    return result
                else:
                    raise ValueError("Response was empty or invalid.")
                    
            except Exception as e:
                err_msg = f"{model_id} failed: {str(e)}"
                logger.warning(f"⚠️ {err_msg}")
                errors.append(err_msg)
                continue

        logger.critical(f"💀 FATAL: All nodes failed for role {role}. Errors: {errors}")
        return None

# Only init fabric if not installing
if not IS_INSTALL_MODE:
    fabric = ComputeFabric()


# --- AuraXLM Logic ---

def analyze_intent(task: str) -> str:
    prompt = f"""
    CLASSIFY intent.
    Task: "{task}"
    Options: ["SIMPLE_CODE", "COMPLEX_REASONING"]
    Return JSON: {{"intent": "..."}}
    """
    res = fabric.execute("router", prompt, json_mode=True)
    try:
        return json.loads(res).get("intent", "SIMPLE_CODE")
    except:
        return "SIMPLE_CODE"

def generate_plan(task: str, context: str) -> List[str]:
    prompt = f"""
    You are a Lead Software Architect.
    TASK: {task}
    CONTEXT: {context}
    
    Create a strictly sequential JSON list of atomic coding steps.
    Example: ["Create utils.py", "Implement class in utils.py", "Update main.py"]
    Return JSON List only.
    """
    res = fabric.execute("reasoning", prompt)
    if not res: return [task]
    
    clean = res.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except:
        return [task]

# --- MCP Entry Point ---

if not IS_INSTALL_MODE:
    @mcp.tool()
    def intelligent_code_gen(task_description: str, file_context: str = "", language: str = "python") -> str:
        """
        AuraRouter V3: Multi-model routing with Intent Classification and Auto-Planning.
        """
        intent = analyze_intent(task_description)
        logger.info(f"🚦 Intent: {intent}")
        
        if intent == "SIMPLE_CODE":
            prompt = f"TASK: {task_description}\nLANG: {language}\nCONTEXT: {file_context}\nCODE ONLY."
            return fabric.execute("coding", prompt) or "Error: Generation failed."

        logger.info("📋 Complexity Detected. Generating Plan...")
        plan = generate_plan(task_description, file_context)
        logger.info(f"📋 Plan: {len(plan)} steps")
        
        output = []
        for i, step in enumerate(plan):
            logger.info(f"🔨 Step {i+1}: {step}")
            prompt = f"""
            GOAL: {step}
            LANG: {language}
            CONTEXT: {file_context}
            PREVIOUS_CODE: {output}
            Return ONLY valid code.
            """
            code = fabric.execute("coding", prompt)
            if code:
                output.append(f"\n# --- Step {i+1}: {step} ---\n{code}")
            else:
                output.append(f"\n# Step {i+1} Failed.")
        
        return "\n".join(output)

def create_config_template():
    """
    Copies a template configuration file to the user's home directory
    during installation, if one doesn't already exist.
    """
    template_content = """# --- SYSTEM SETTINGS ---
system:
  log_level: INFO
  default_timeout: 120.0

# --- HARDWARE INVENTORY (mobile1) ---
models:
  # Node 1: The 3070 Laptop (Code Monkey)
  local_3070_qwen:
    provider: ollama
    endpoint: http://localhost:11434/api/generate
    model_name: qwen2.5-coder:7b
    parameters:
      temperature: 0.1
      num_ctx: 4096

  # Node 2: The 3090 Server (space heater) 
  # local_3090_deepseek:
  #   provider: ollama
  #   endpoint: http://192.168.1.50:11434/api/generate
  #   model_name: deepseek-coder-v2:lite
  #   parameters:
  #     num_ctx: 16384

  # Node 3: Cloud Fallback (Fast)
  cloud_gemini_flash:
    provider: google
    model_name: gemini-2.0-flash
    # You can now paste the key directly here, or use env_key to read from shell
    api_key: YOUR_API_KEY
    # env_key: GOOGLE_API_KEY 

  # Node 4: Cloud Architect (Reasoning)
  cloud_gemini_pro:
    provider: google
    model_name: gemini-2.0-pro-exp
    api_key: YOUR_API_KEY

# --- ROLES & ROUTING (The "Tags") ---
# The Router will iterate these lists until one works.
roles:
  # Tag: Who decides intent?
  router:
    - local_3070_qwen
    - cloud_gemini_flash

  # Tag: Who creates the architectural plan?
  reasoning:
    # - local_3090_deepseek (Preferred when online)
    - cloud_gemini_pro
    - cloud_gemini_flash

  # Tag: Who writes the code?
  coding:
    - local_3070_qwen
    - cloud_gemini_flash
"""
    
    target_dir = Path.home() / ".auracore" / "aurarouter"
    target_path = target_dir / "auraconfig_template.yaml"

    print(f"\n   📄 Looking for configuration template...")

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            with open(target_path, 'w') as f:
                f.write(template_content)
            print(f"   ✅ Template created at: {target_path}")
            print(f"      Rename this file to 'auraconfig.yaml' and edit it to configure your models.")
        else:
            print(f"   ℹ️  Template file already exists at: {target_path}, skipping creation.")

    except Exception as e:
        print(f"   ❌ Error handling template file: {e}")


def main():
    parser = argparse.ArgumentParser(description="AuraRouter MCP Server")
    parser.add_argument("--config", help="Path to the auraconfig.yaml file. If not provided, it will be searched in the AURACORE_ROUTER_CONFIG env var, and then in ~/.auracore/aurarouter/.")
    parser.add_argument("--install", action="store_true", help="Run interactive installer for all supported models.")
    parser.add_argument("--install-gemini", action="store_true", help="Register AuraRouter for Gemini.")
    parser.add_argument("--install-claude", action="store_true", help="Register AuraRouter for Claude.")
    args = parser.parse_args()

    if IS_INSTALL_MODE:
        create_config_template()

    if args.install:
        install_all()
    elif args.install_gemini:
        install_gemini_router()
    elif args.install_claude:
        install_claude_router()
    else:
        mcp.run()

if __name__ == "__main__":
    main()