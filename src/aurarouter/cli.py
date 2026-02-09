import argparse
import logging
import sys

from aurarouter._logging import get_logger

logger = get_logger("AuraRouter.CLI")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="AuraRouter MCP Server")
    parser.add_argument(
        "--config",
        help="Path to auraconfig.yaml. Falls back to AURACORE_ROUTER_CONFIG env var, "
        "then ~/.auracore/aurarouter/auraconfig.yaml.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run interactive installer for all supported MCP clients.",
    )
    parser.add_argument(
        "--install-gemini",
        action="store_true",
        help="Register AuraRouter for the Gemini CLI.",
    )
    parser.add_argument(
        "--install-claude",
        action="store_true",
        help="Register AuraRouter for Claude.",
    )
    parser.add_argument(
        "--claude-mode",
        action="store_true",
        help="Run in Claude-compatible mode (used by Claude installer).",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- download-model subcommand ---
    dl = subparsers.add_parser(
        "download-model",
        help="Download a GGUF model from HuggingFace Hub.",
    )
    dl.add_argument("--repo", required=True, help="HuggingFace repo ID")
    dl.add_argument("--file", required=True, help="GGUF filename in the repo")
    dl.add_argument(
        "--dest",
        default=None,
        help="Destination directory (default: ~/.auracore/models/)",
    )

    # --- gui subcommand ---
    subparsers.add_parser("gui", help="Launch the AuraRouter GUI.")

    args = parser.parse_args()

    # ---- Install mode (no config needed) ----
    is_install = args.install or args.install_gemini or args.install_claude

    if is_install:
        from aurarouter.installers.template import create_config_template

        create_config_template()

        if args.install:
            from aurarouter.installers.registry import install_all

            install_all()
        elif args.install_gemini:
            from aurarouter.installers.gemini import GeminiInstaller

            GeminiInstaller().install()
        elif args.install_claude:
            from aurarouter.installers.claude_inst import ClaudeInstaller

            ClaudeInstaller().install()
        return

    # ---- download-model subcommand (no config needed) ----
    if args.command == "download-model":
        from aurarouter.models.downloader import download_model

        download_model(repo=args.repo, filename=args.file, dest=args.dest)
        return

    # ---- GUI subcommand ----
    if args.command == "gui":
        try:
            from aurarouter.gui.app import launch_gui
        except ImportError:
            print(
                "PySide6 is required for the GUI.\n"
                "Install with:  pip install aurarouter[gui]"
            )
            sys.exit(1)

        from aurarouter.config import ConfigLoader

        config = ConfigLoader(config_path=args.config)
        launch_gui(config)
        return

    # ---- Default: run MCP server ----
    from aurarouter.config import ConfigLoader
    from aurarouter.server import create_mcp_server

    config = ConfigLoader(config_path=args.config)
    mcp = create_mcp_server(config)
    mcp.run()
