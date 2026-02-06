import json
from pathlib import Path
from unittest.mock import patch

from aurarouter.installers.gemini import GeminiInstaller
from aurarouter.installers.claude_inst import ClaudeInstaller


def test_gemini_installer_properties():
    inst = GeminiInstaller()
    assert inst.name == "Gemini"
    assert inst.server_name == "aurarouter"
    assert len(inst.config_candidates()) >= 2
    assert inst.extra_args() == []


def test_claude_installer_properties():
    inst = ClaudeInstaller()
    assert inst.name == "Claude"
    assert inst.server_name == "clauderouter"
    assert "--claude-mode" in inst.extra_args()


def test_gemini_detect_finds_existing(tmp_path):
    settings = tmp_path / ".gemini" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{}")

    inst = GeminiInstaller()
    with patch.object(inst, "config_candidates", return_value=[settings]):
        assert inst.detect_config_path() == settings


def test_detect_returns_none_when_missing():
    inst = GeminiInstaller()
    with patch.object(inst, "config_candidates", return_value=[Path("/does/not/exist.json")]):
        assert inst.detect_config_path() is None


def test_install_writes_mcp_entry(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text("{}")

    inst = GeminiInstaller()
    with (
        patch.object(inst, "detect_config_path", return_value=settings),
        patch("builtins.input", return_value=""),
    ):
        inst.install()

    data = json.loads(settings.read_text())
    assert "mcpServers" in data
    assert "aurarouter" in data["mcpServers"]
    entry = data["mcpServers"]["aurarouter"]
    assert "command" in entry
    assert "args" in entry
