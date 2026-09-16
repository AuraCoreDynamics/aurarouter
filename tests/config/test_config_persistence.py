import os
import yaml
from pathlib import Path
from aurarouter.config import ConfigLoader

def test_config_save(tmp_path):
    config_file = tmp_path / "auraconfig.yaml"
    with open(config_file, "w") as f:
        f.write("api_version: '1.0'\n")
    
    # Initialize with default config
    loader = ConfigLoader(config_path=str(config_file))
    
    # Mutate and save
    loader.config["savings"] = {"cost_estimator": "azure_retail"}
    loader.save()
    
    # Verify file was written correctly
    with open(config_file, "r") as f:
        saved_data = yaml.safe_load(f)
        
    assert saved_data["savings"]["cost_estimator"] == "azure_retail"
