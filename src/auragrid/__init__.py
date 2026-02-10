"""
AuraRouter integration with AuraGrid.

This module provides optional AuraGrid integration for deploying aurarouter
as a Managed Application Service (MAS) on AuraGrid. It is only available
when the auragrid-sdk package is installed.

The module remains optional for backwards compatibility—aurarouter works
identically when deployed standalone without AuraGrid.
"""

try:
    from .config_loader import ConfigLoader
    from .events import EventBridge
    from .lifecycle import LifecycleCallbacks
    from .mas_host import AuraRouterMasHost
    from .services import (
        CodingService,
        ReasoningService,
        RouterService,
        UnifiedRouterService,
    )

    __all__ = [
        "AuraRouterMasHost",
        "RouterService",
        "ReasoningService",
        "CodingService",
        "UnifiedRouterService",
        "ConfigLoader",
        "LifecycleCallbacks",
        "EventBridge",
    ]

except ImportError:
    import warnings

    warnings.warn(
        "AuraGrid SDK not installed. AuraRouter can be deployed standalone. "
        "To enable AuraGrid integration, install: pip install aurarouter[auragrid]",
        ImportWarning,
        stacklevel=2,
    )
    __all__ = []
