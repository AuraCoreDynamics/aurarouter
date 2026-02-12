import sys

from aurarouter.gui import check_pyside6

check_pyside6()

from PySide6.QtWidgets import QApplication  # noqa: E402

from aurarouter.config import ConfigLoader  # noqa: E402
from aurarouter.gui.main_window import AuraRouterWindow  # noqa: E402


def launch_gui(config: ConfigLoader) -> None:
    """Create the QApplication and show the main window."""
    app = QApplication(sys.argv)
    app.setApplicationName("AuraRouter")

    window = AuraRouterWindow(config)
    window.show()

    sys.exit(app.exec())


def main() -> None:
    """Standalone entry-point (aurarouter-gui script)."""
    import argparse
    import logging
    from aurarouter.config import ConfigLoader

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="AuraRouter GUI")
    parser.add_argument(
        "--config",
        help="Path to auraconfig.yaml. Falls back to AURACORE_ROUTER_CONFIG env var, "
        "then ~/.auracore/aurarouter/auraconfig.yaml.",
    )
    args = parser.parse_args()

    try:
        config = ConfigLoader(config_path=args.config)
    except FileNotFoundError:
        config = ConfigLoader(allow_missing=True)
        config.config = {"models": {}, "roles": {}}

    launch_gui(config)
