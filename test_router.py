import sys
import logging
import time
import aurarouter  # Imports the running instance of 'fabric' and 'config'

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TestHarness")

def run_test(name, validation_func):
    """Wrapper to run tests and print status."""
    print(f"\n{'='*10} TEST: {name} {'='*10}")
    try:
        start = time.time()
        result = validation_func()
        duration = time.time() - start
        
        if result and "Error" not in result:
            print(f"✅ PASS | Duration: {duration:.2f}s")
            print(f"   Output Preview: {str(result)[:80].replace(chr(10), ' ')}...")
        else:
            print(f"❌ FAIL | Duration: {duration:.2f}s")
            print(f"   Result: {result}")
    except Exception as e:
        print(f"💥 CRASH | {e}")

def test_happy_path():
    """
    Test 1: Standard Operation
    Uses the default configuration (should hit Local Qwen).
    """
    print("📋 Goal: Verify the primary coding model is working.")
    
    task = "Write a python function to add two numbers."
    
    # We call the MCP tool function directly
    return aurarouter.intelligent_code_gen(
        task_description=task,
        language="python"
    )

def test_fallback_logic():
    """
    Test 2: Disaster Simulation
    Dynamically finds the primary local model in the config and breaks its URL.
    """
    print("📋 Goal: Verify Fabric fails over to Cloud when Local dies.")
    
    # 1. Inspect the Config to find the victim
    primary_model_id = aurarouter.config.get_role_chain('coding')[0]
    original_config = aurarouter.config.config['models'][primary_model_id].copy()
    
    print(f"   🎯 Sabotaging Node: {primary_model_id}")
    
    # 2. Break the endpoint
    aurarouter.config.config['models'][primary_model_id]['endpoint'] = "http://localhost:9999/dead"
    
    try:
        task = "Write a python hello world."
        # This should trigger the "Warning" logs in aurarouter and switch to the next node
        return aurarouter.intelligent_code_gen(
            task_description=task,
            language="python"
        )
    finally:
        # 3. Heal the configuration so we don't break future tests
        print(f"   ❤️  Healing Node: {primary_model_id}")
        aurarouter.config.config['models'][primary_model_id] = original_config

if __name__ == "__main__":
    print("🚀 AURAROUTER V2 DIAGNOSTICS")
    
    # Run Happy Path
    run_test("Local Execution", test_happy_path)
    
    # Run Failure Simulation
    run_test("Graceful Degradation", test_fallback_logic)
    
    print("\n🏁 DONE")