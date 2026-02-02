import os
import yaml
import json
import httpx
import logging
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
    exit(1)

mcp = FastMCP("AuraRouter")

# --- The Compute Fabric ---

class ComputeFabric:
    """
    Handles N-Model routing with graceful degradation.
    """
    
    def _resolve_api_key(self, cfg: Dict) -> str:
        # 1. Try direct key in YAML
        if cfg.get('api_key') and "YOUR_PASTED_KEY" not in cfg['api_key']:
            return cfg['api_key']
        # 2. Try environment variable reference
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
        """
        Executes a prompt against the configured priority chain for a role.
        """
        chain = config.get_role_chain(role)
        if not chain:
            return f"ERROR: No models defined for role '{role}' in YAML."

        errors = []
        
        for model_id in chain:
            model_cfg = config.get_model_config(model_id)
            if not model_cfg:
                logger.warning(f"⚠️ Model ID '{model_id}' listed in role '{role}' but not defined in models.")
                continue

            provider = model_cfg.get('provider')
            logger.info(f"🔄 [{role.upper()}] Routing to: {model_id} ({provider})")
            
            try:
                result = None
                if provider == 'ollama':
                    result = self._call_ollama(model_cfg, prompt, json_mode)
                elif provider == 'google':
                    result = self._call_google(model_cfg, prompt)
                
                # Quality Gate
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

@mcp.tool()
def intelligent_code_gen(task_description: str, file_context: str = "", language: str = "python") -> str:
    """
    AuraRouter V2: 
    Multi-model routing with Intent Classification and Auto-Planning.
    """
    
    # 1. Intent Analysis
    intent = analyze_intent(task_description)
    logger.info(f"🚦 Intent: {intent}")
    
    # 2. Simple Path
    if intent == "SIMPLE_CODE":
        prompt = f"TASK: {task_description}\nLANG: {language}\nCONTEXT: {file_context}\nCODE ONLY."
        return fabric.execute("coding", prompt) or "Error: Generation failed."

    # 3. Complex Path (Auto-Planning)
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
    mcp.run()