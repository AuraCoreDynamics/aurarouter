import os
import httpx
import logging
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. Boot Sequence ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AuraRouter")

# --- 2. Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "deepseek-coder-v2:lite"
FALLBACK_MODEL = "gemini-2.0-flash"

mcp = FastMCP("AuraRouter")

# --- 3. The Engines ---

def call_ollama(prompt: str, temperature: float = 0.1) -> str:
    """Attempts to generate code using the local DeepSeek model on the 3070."""
    logger.info(f"⚡ [LOCAL] Spinning up {LOCAL_MODEL} on GPU...")
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                OLLAMA_URL,
                json={
                    "model": LOCAL_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "options": {
                        "num_ctx": 4096, 
                        "num_predict": 2048
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
    except Exception as e:
        logger.error(f"❌ [LOCAL FAIL] {e}")
        return None

def call_gemini_fallback(prompt: str) -> str:
    """Fallback to Google Gemini API if local hardware fails."""
    logger.warning(f"☁️ [CLOUD] engaging fallback: {FALLBACK_MODEL}")
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not found in .env or environment."
            
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=FALLBACK_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.critical(f"💀 [TOTAL FAILURE] Cloud fallback also failed: {e}")
        return f"CRITICAL SYSTEM FAILURE: {e}"

# --- 4. The Router Logic ---

@mcp.tool()
def intelligent_code_gen(task_description: str, file_context: str = "", language: str = "python") -> str:
    """
    Generates robust, functional code by routing to DeepSeek-Coder locally.
    Automatically falls back to Gemini Cloud if the local model fails.
    
    Args:
        task_description: The specific coding task or function to implement.
        file_context: The existing code or file content to modify.
        language: The target programming language.
    """
    
    # Robust Prompt Engineering (License requirement removed)
    prompt = f"""
    You are an expert Senior Software Engineer specializing in robust, functional code.
    
    LANGUAGE: {language}
    
    CONTEXT:
    {file_context}
    
    TASK:
    {task_description}
    
    MANDATORY REQUIREMENTS:
    1. ROBUSTNESS: Bias towards over-engineering (error handling, logging, type safety).
    2. FORMAT: Return ONLY the valid code block. No markdown backticks, no conversational text.
    3. FUNCTIONALITY: If modifying, return the COMPLETE file content.
    """
    
    # Step 1: Attempt Local Inference
    result = call_ollama(prompt)
    
    # Step 2: Quality Gate
    if result and len(result) > 15:
        logger.info("✅ [LOCAL SUCCESS] DeepSeek returned valid output.")
        return result
        
    # Step 3: Cloud Fallback
    logger.info("⚠️ [QUALITY GATE] Local output invalid. Switching to Cloud.")
    return call_gemini_fallback(prompt)

if __name__ == "__main__":
    mcp.run()