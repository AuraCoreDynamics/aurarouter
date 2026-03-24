"""Help framework for AuraRouter GUI.

Provides the HelpTopic dataclass, HelpRegistry, and a module-level
HELP singleton populated with all built-in topics on import.
"""

from __future__ import annotations

from aurarouter.gui.help.content import HELP, HelpRegistry, HelpTopic

__all__ = ["HELP", "HelpRegistry", "HelpTopic"]
