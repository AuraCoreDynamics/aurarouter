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

# --- Configuration Manager ---
class ConfigLoader:
    def __init__(self, path="auraconfig.yaml"):
        # If we are just installing, we don't crash if config is missing
        if "--install" in sys.argv:
            self.config = {}
            return

        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file {path} missing!")
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_role_chain(self, role: str) -> List[str]:
        return self.config.get('roles', {}).get(role, [])
        
    def get_model_config(self, model_id: str) -> Dict:
        return self.config.get('models', {}).get(model_id, {})

try:
    config = ConfigLoader()
except Exception as e:
    logger.critical(f"Failed to load config: {e}")
    sys.exit(1)

# Initialize MCP only if not installing (prevents side effects during setup)
if "--install" not in sys.argv:
    mcp = FastMCP("AuraRouter")

# --- Installer Logic (Robust & Interactive) ---

def install_router():
    """
    Auto-registers this script into the Gemini CLI settings.json.
    """
    print("\n🔧 AuraRouter Installer")
    print("=======================")

    # 1. Detect Environment
    current_python = sys.executable
    script_path = os.path.abspath(__file__)
    
    print(f"   📍 Python Interpreter: {current_python}")
    print(f"   📍 Router Script:      {script_path}")

    # 2. Intelligent Path Detection
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
            print(f"   ✅ Auto-detected config file at: {p}")
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
        data["mcpServers"]["aurarouter"] = {
            "command": current_python,
            "args": [script_path],
            "env": {
                "PYTHONUNBUFFERED": "1" # Ensure logs flush immediately
            }
        }
        
        # Write
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print("\n   ✅ SUCCESS: AuraRouter registered successfully.")
        print("   🔁 Action Required: Restart your Gemini CLI terminal session.")
        
    except Exception as e:
        print(f"\n   ❌ FATAL: Installation Failed: {e}")


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
if "--install" not in sys.argv:
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

if "--install" not in sys.argv:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraRouter MCP Server")
    parser.add_argument("--install", action="store_true", help="Register AuraRouter with Gemini CLI")
    args = parser.parse_args()

    if args.install:
        install_router()
    else:
        mcp.run()