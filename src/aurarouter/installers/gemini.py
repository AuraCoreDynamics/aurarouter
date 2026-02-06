from pathlib import Path

from aurarouter.installers.registry import BaseInstaller


class GeminiInstaller(BaseInstaller):
    """Registers AuraRouter as an MCP server in the Gemini CLI."""

    @property
    def name(self) -> str:
        return "Gemini"

    @property
    def server_name(self) -> str:
        return "aurarouter"

    def config_candidates(self) -> list[Path]:
        home = Path.home()
        return [
            home / ".gemini" / "settings.json",
            home / ".geminichat" / "settings.json",
            home / ".geminichat" / "config.json",
            home / "gemini-cli" / "settings.json",
        ]
