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

    # --- list-models subcommand ---
    lm = subparsers.add_parser(
        "list-models",
        help="List locally downloaded GGUF models.",
    )
    lm.add_argument(
        "--dir",
        default=None,
        help="Model storage directory (default: ~/.auracore/models/)",
    )

    # --- remove-model subcommand ---
    rm = subparsers.add_parser(
        "remove-model",
        help="Remove a downloaded GGUF model.",
    )
    rm.add_argument("--file", required=True, help="GGUF filename to remove")
    rm.add_argument(
        "--keep-file",
        action="store_true",
        help="Remove from registry only, keep the file on disk.",
    )
    rm.add_argument(
        "--dir",
        default=None,
        help="Model storage directory (default: ~/.auracore/models/)",
    )

    # --- gui subcommand ---
    gui_parser = subparsers.add_parser("gui", help="Launch the AuraRouter GUI.")
    gui_parser.add_argument(
        "--environment",
        choices=["local", "auragrid"],
        default=None,
        help="Deployment environment to start in (default: local).",
    )

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
        try:
            from aurarouter.models.downloader import download_model
        except ImportError:
            print(
                "huggingface-hub is required for model downloading.\n"
                "Install with:  pip install aurarouter[local]"
            )
            sys.exit(1)

        download_model(repo=args.repo, filename=args.file, dest=args.dest)
        return

    # ---- list-models subcommand (no config needed) ----
    if args.command == "list-models":
        from aurarouter.models.file_storage import FileModelStorage

        storage = FileModelStorage(args.dir)
        storage.scan()
        models = storage.list_models()

        if not models:
            print("No models found in:", storage.models_dir)
            print("Download one with:  aurarouter download-model --repo <repo> --file <name>")
            return

        print(f"Models in {storage.models_dir}:\n")
        for m in models:
            size_mb = m.get("size_bytes", 0) / (1024 * 1024)
            repo = m.get("repo", "unknown")
            print(f"  {m['filename']}  ({size_mb:.0f} MB)  repo: {repo}")
        print(f"\n{len(models)} model(s) total.")
        return

    # ---- remove-model subcommand (no config needed) ----
    if args.command == "remove-model":
        from aurarouter.models.file_storage import FileModelStorage

        storage = FileModelStorage(args.dir)
        delete_file = not args.keep_file
        removed = storage.remove(args.file, delete_file=delete_file)

        if removed:
            action = "Removed and deleted" if delete_file else "Unregistered"
            print(f"{action}: {args.file}")
        else:
            print(f"Model not found in registry: {args.file}")
            sys.exit(1)
        return

    # ---- GUI subcommand ----
    if args.command == "gui":
        from aurarouter.gui.app import _create_context, launch_gui

        environment = getattr(args, "environment", None)
        context = _create_context(
            environment=environment,
            config_path=args.config,
        )
        launch_gui(context)
        return

    # ---- Default: run MCP server ----
    from aurarouter.config import ConfigLoader
    from aurarouter.server import create_mcp_server

    try:
        config = ConfigLoader(config_path=args.config)
    except FileNotFoundError:
        print(
            "No configuration file found.\n"
            "Run 'aurarouter --install' to create one, or\n"
            "run 'aurarouter gui' to configure interactively."
        )
        sys.exit(1)

    mcp = create_mcp_server(config)
    mcp.run()
