import yaml
from pathlib import Path
from aurarouter.config import ConfigLoader

def test_load_flat_roles(tmp_path: Path):
    """Test the original flat list format for roles."""
    config_dict = {
        "models": {"m1": {"provider": "p1"}},
        "roles": {
            "coder": ["m1"]
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    loader = ConfigLoader(config_path=str(config_path))
    chain = loader.get_role_chain("coder")
    
    assert chain == ["m1"]

def test_load_nested_roles(tmp_path: Path):
    """Test the new nested dictionary format for roles."""
    config_dict = {
        "models": {"m1": {"provider": "p1"}, "m2": {"provider": "p2"}},
        "roles": {
            "coder": {
                "chain": ["m1", "m2"],
                "description": "This is a test role"
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    loader = ConfigLoader(config_path=str(config_path))
    chain = loader.get_role_chain("coder")
    
    assert chain == ["m1", "m2"]

def test_get_role_chain_mixed_formats(tmp_path: Path):
    """Test a config with both flat and nested role formats."""
    config_dict = {
        "models": {"m1": {"provider": "p1"}, "m2": {"provider": "p2"}},
        "roles": {
            "flat_role": ["m1"],
            "nested_role": {
                "chain": ["m2"],
                "description": "Nested"
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    loader = ConfigLoader(config_path=str(config_path))
    
    assert loader.get_role_chain("flat_role") == ["m1"]
    assert loader.get_role_chain("nested_role") == ["m2"]

def test_get_role_chain_nonexistent(config):
    """Test that getting a nonexistent role returns an empty list."""
    chain = config.get_role_chain("nonexistent_role")
    assert chain == []

