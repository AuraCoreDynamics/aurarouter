from pathlib import Path

from aurarouter.installers.registry import BaseInstaller


class ClaudeInstaller(BaseInstaller):
    """Registers AuraRouter as an MCP server for Claude."""

    @property
    def name(self) -> str:
        return "Claude"

    @property
    def server_name(self) -> str:
        return "clauderouter"

    def config_candidates(self) -> list[Path]:
        home = Path.home()
        return [
            home / ".claude" / "settings.json",
            home / ".claude" / "config.json",
        ]

    def extra_args(self) -> list[str]:
        return ["--claude-mode"]
