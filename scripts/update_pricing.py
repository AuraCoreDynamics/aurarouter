import json
import re
from pathlib import Path

# Add src to path so we can import aurarouter
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aurarouter.savings.cost_estimator import AzureRetailCostEstimator, BedrockCostEstimator

class DummyConfig:
    def __init__(self):
        self.config = {}

def update_builtin_table(azure_pricing: dict, bedrock_pricing: dict):
    target_path = Path(__file__).parent.parent / "src" / "aurarouter" / "savings" / "cost_estimator.py"
    
    print(f"Updating {target_path}...")
    content = target_path.read_text(encoding="utf-8")
    
    match = re.search(r"(_BUILTIN_TABLE\s*=\s*)(\{[\s\S]*?\n    \})", content)
    if not match:
        print("Could not find _BUILTIN_TABLE in cost_estimator.py")
        return
        
    existing_table_str = match.group(2)
    try:
        existing_table = eval(existing_table_str)
    except Exception as e:
        print("Failed to parse existing table:", e)
        return
        
    # Update with new providers
    if azure_pricing:
        existing_table["azure_openai"] = azure_pricing
    if bedrock_pricing:
        existing_table["bedrock"] = bedrock_pricing
    
    new_table_str = json.dumps(existing_table, indent=8)
    lines = new_table_str.split("\n")
    lines = [lines[0]] + ["    " + line for line in lines[1:-1]] + ["    }"]
    new_table_str = "\n".join(lines)
    
    new_content = content[:match.start(2)] + new_table_str + content[match.end(2):]
    
    target_path.write_text(new_content, encoding="utf-8")
    print("Update complete!")

def main():
    dummy = DummyConfig()
    
    print("Fetching Bedrock pricing...")
    bedrock = BedrockCostEstimator(dummy).update_cache()
    print("Bedrock models found:", len(bedrock))
    
    print("Fetching Azure pricing...")
    azure = AzureRetailCostEstimator(dummy).update_cache()
    print("Azure models found:", len(azure))
    
    update_builtin_table(azure, bedrock)

if __name__ == "__main__":
    main()
