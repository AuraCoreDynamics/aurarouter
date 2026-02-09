import json
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.Installer")


class BaseInstaller(ABC):
    """Abstract base for MCP client installers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown during installation (e.g. 'Gemini')."""
        ...

    @property
    @abstractmethod
    def server_name(self) -> str:
        """Key used in the mcpServers JSON block."""
        ...

    @abstractmethod
    def config_candidates(self) -> list[Path]:
        """Return an ordered list of paths to search for the client config."""
        ...

    def default_config_path(self) -> Path:
        """Fallback path when no candidate is found on disk."""
        candidates = self.config_candidates()
        return candidates[0] if candidates else Path.home() / ".mcp" / "settings.json"

    def extra_args(self) -> list[str]:
        """Additional CLI args to pass when launching this router variant."""
        return []

    # ------------------------------------------------------------------
    def detect_config_path(self) -> Optional[Path]:
        """Find the first existing config file from the candidate list."""
        for p in self.config_candidates():
            if p.exists():
                return p
        return None

    def build_payload(self) -> dict:
        """Build the mcpServers entry for this installer."""
        python = sys.executable
        return {
            "command": python,
            "args": ["-m", "aurarouter"] + self.extra_args(),
            "env": {"PYTHONUNBUFFERED": "1"},
        }

    def install(self) -> None:
        """Detect config → prompt user → inject mcpServers entry → write."""
        print(f"\n  AuraRouter ({self.name}) Installer")
        print("  =======================")
        print(f"   Python Interpreter: {sys.executable}")

        detected = self.detect_config_path()
        default = detected if detected else self.default_config_path()

        if detected:
            print(f"   Auto-detected config at: {detected}")
        else:
            print(f"   No existing config found; will use: {default}")

        print(f"\n   Where is your {self.name} CLI settings file?")
        user_input = input(f"   [Press ENTER to use: {default}]: ").strip()
        target = Path(os.path.expanduser(user_input)) if user_input else default

        if not target.parent.exists():
            print(f"\n   Error: Directory '{target.parent}' does not exist.")
            print("   Create it manually or run the CLI once to generate its folders.")
            return

        print(f"   Targeting: {target}")

        try:
            data: dict = {}
            if target.exists():
                try:
                    with open(target, "r") as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                except json.JSONDecodeError:
                    print("   File contains invalid JSON. Backing up and starting fresh.")
                    target.rename(target.with_suffix(".json.bak"))

            data.setdefault("mcpServers", {})
            data["mcpServers"][self.server_name] = self.build_payload()

            with open(target, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\n   SUCCESS: AuraRouter ({self.name}) registered.")
            print("   Restart your CLI session to pick up the change.")

        except Exception as e:
            print(f"\n   FATAL: {self.name} installation failed: {e}")


# ------------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------------

def _get_all_installers() -> list[BaseInstaller]:
    """Return instances of every registered installer."""
    from aurarouter.installers.gemini import GeminiInstaller
    from aurarouter.installers.claude_inst import ClaudeInstaller

    return [GeminiInstaller(), ClaudeInstaller()]


def install_all() -> None:
    """Interactive loop that offers to run each installer."""
    print("\n  AuraRouter Interactive Installer")
    print("  ==================================")

    for installer in _get_all_installers():
        while True:
            choice = (
                input(f"\n   Install support for {installer.name}? [Y]es, [N]o, [Q]uit: ")
                .lower()
                .strip()
            )
            if choice in ("y", "yes"):
                installer.install()
                break
            elif choice in ("n", "no", "skip"):
                print(f"   Skipping {installer.name}.")
                break
            elif choice in ("q", "quit"):
                print("   Aborting installation.")
                return
            else:
                print("   Invalid choice. Please enter Y, N, or Q.")
