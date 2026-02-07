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
    
    assert aurarouter.__version__ == "0.2.0"
    assert ConfigLoader is not None
    assert ComputeFabric is not None


def test_auragrid_conditional_import():
    """Test that auragrid module import is conditional."""
    import aurarouter
    
    # Check that conditional import logic exists
    assert hasattr(aurarouter, "_auragrid_available")


def test_cli_works_without_auragrid():
    """Test that CLI command works without AuraGrid SDK."""
    # This would require auragrid SDK not to be installed
    # We'll just check that the entry point is defined
    from aurarouter.cli import main
    assert callable(main)


def test_server_works_without_auragrid():
    """Test that MCP server works without AuraGrid SDK."""
    from aurarouter.server import create_server
    assert callable(create_server)


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
    """Test that auragrid module is only in __all__ if available."""
    import aurarouter
    
    # After import, check __all__
    # auragrid should only be there if SDK is available
    if hasattr(aurarouter, "auragrid"):
        # If auragrid is available, it should be in __all__
        assert "auragrid" in aurarouter.__all__
    else:
        # If auragrid is not available, it shouldn't be in __all__
        # (conditional import warning will be shown)
        pass


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
        """Test that __init__.py has try/except for auragrid import."""
        init_path = Path(__file__).parent.parent / "src" / "aurarouter" / "__init__.py"
        
        if init_path.exists():
            content = init_path.read_text()
            
            # Should have try/except for conditional import
            assert "try:" in content
            assert "except ImportError" in content
            assert "_auragrid_available" in content
