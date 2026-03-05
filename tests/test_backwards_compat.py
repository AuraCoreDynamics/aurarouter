"""
Backwards compatibility tests for AuraRouter.

Ensures that aurarouter works identically when deployed standalone
without AuraGrid SDK installed.
"""

import subprocess
import sys
from pathlib import Path


def test_standalone_import():
    """Test that aurarouter imports without AuraGrid SDK."""
    # This test runs in current environment
    import aurarouter
    from aurarouter import ConfigLoader, ComputeFabric
    
    assert aurarouter.__version__ == "0.3.0"
    assert ConfigLoader is not None
    assert ComputeFabric is not None


def test_auragrid_conditional_import():
    """Test that aurarouter.auragrid submodule is always available."""
    import importlib

    # After namespace consolidation, aurarouter.auragrid is always available
    spec = importlib.util.find_spec("aurarouter.auragrid")
    assert spec is not None, "aurarouter.auragrid should always be importable"
    
    # Can import submodules
    from aurarouter.auragrid import config_loader
    assert config_loader is not None


def test_cli_works_without_auragrid():
    """Test that CLI command works without AuraGrid SDK."""
    # This would require auragrid SDK not to be installed
    # We'll just check that the entry point is defined
    from aurarouter.cli import main
    assert callable(main)


def test_server_works_without_auragrid():
    """Test that MCP server works without AuraGrid SDK."""
    from aurarouter.server import create_mcp_server
    assert callable(create_mcp_server)


def test_config_loader_independence():
    """Test that ConfigLoader works independently of AuraGrid."""
    from aurarouter.config import ConfigLoader
    
    # Should work with allow_missing=True
    loader = ConfigLoader(allow_missing=True)
    assert loader.config is not None or loader.config == {}


def test_compute_fabric_independence():
    """Test that ComputeFabric can be initialized without AuraGrid."""
    from aurarouter.config import ConfigLoader
    from aurarouter.fabric import ComputeFabric
    
    # Create loader with allow_missing to avoid file requirement
    config = ConfigLoader(allow_missing=True)
    
    # This should work even with minimal config
    fabric = ComputeFabric(config=config)
    assert fabric is not None


def test_auragrid_module_not_in_main_all():
    """Test that auragrid submodule is not exported in main __all__."""
    import aurarouter
    
    # After namespace consolidation, auragrid is a submodule but not in main __all__
    # Main exports are for the core standalone functionality
    assert "auragrid" not in aurarouter.__all__, "auragrid should not be in main __all__"
    
    # But the submodule itself should still be importable
    from aurarouter import auragrid
    assert auragrid is not None


def test_existing_functionality_unchanged():
    """
    Test that existing aurarouter functionality is unchanged.
    
    This is a meta-test that verifies the public API surface.
    """
    import aurarouter
    
    # Check that key exports are still available
    expected_exports = ["ConfigLoader", "ComputeFabric", "__version__"]
    
    for export in expected_exports:
        assert hasattr(aurarouter, export), f"Missing export: {export}"


class TestNoAuraGridDependency:
    """Tests verifying aurarouter has no hard dependency on AuraGrid."""

    def test_pyproject_auragrid_is_optional(self):
        """Test that auragrid-sdk is in optional-dependencies, not dependencies."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # auragrid-sdk should be in optional-dependencies, not dependencies
            assert "optional-dependencies" in content
            
            # Check that auragrid is optional
            optional_section = content.split("[project.optional-dependencies]")[1]
            assert "auragrid" in optional_section.lower()

    def test_no_hard_auragrid_imports_in_init(self):
        """Test that __init__.py has no conditional import logic."""
        init_path = Path(__file__).parent.parent / "src" / "aurarouter" / "__init__.py"
        
        if init_path.exists():
            content = init_path.read_text()
            
            # After namespace consolidation, no conditional import logic needed
            assert "importlib.util.find_spec" not in content, "Should not have conditional import logic"
            assert "sys.modules" not in content, "Should not have sys.modules hack"
            
            # auragrid is a submodule, not imported in main __init__
            assert "from aurarouter.auragrid" not in content, "Should not import auragrid in main __init__"

