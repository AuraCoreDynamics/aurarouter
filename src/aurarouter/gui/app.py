import sys

from aurarouter.gui import check_pyside6

check_pyside6()

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from aurarouter.gui.environment import EnvironmentContext  # noqa: E402
from aurarouter.gui.main_window import AuraRouterWindow  # noqa: E402


def _create_context(
    environment: str | None = None,
    config_path: str | None = None,
) -> EnvironmentContext:
    """Create the appropriate ``EnvironmentContext`` for the selected environment.

    Args:
        environment: ``"local"`` or ``"auragrid"``.  If *None*, defaults to
            ``"local"`` (AuraGrid is only used when explicitly requested).
        config_path: Optional path to ``auraconfig.yaml``.
    """
    env = (environment or "local").lower()

    if env == "auragrid":
        try:
            from aurarouter.gui.env_grid import AuraGridEnvironmentContext

            return AuraGridEnvironmentContext(config_path=config_path)
        except ImportError:
            # auragrid SDK not installed — fall back to local.
            pass

    from aurarouter.gui.env_local import LocalEnvironmentContext

    return LocalEnvironmentContext(config_path=config_path)


def launch_gui(context: EnvironmentContext, config_path: str | None = None) -> None:
    """Create the QApplication and show the main window."""
    from aurarouter.singleton import SingletonLock

    lock = SingletonLock()

    # Check for an already-running instance.
    existing = lock.get_existing_instance()
    if existing:
        # Need QApplication to show the dialog.
        app = QApplication(sys.argv)
        app.setApplicationName("AuraRouter")

        from PySide6.QtWidgets import QMessageBox

        pid = existing.get("pid", "?")
        reply = QMessageBox.question(
            None,
            "AuraRouter Already Running",
            f"Another AuraRouter instance is running (PID {pid}).\n\n"
            "Only one instance should run at a time.\n"
            "Would you like to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            sys.exit(0)
        # User chose No — continue anyway (they may know what they're doing).

    if not lock.acquire():
        # Rare: PID file gone between get_existing and acquire.
        pass  # Continue — best effort.

    app = QApplication(sys.argv)
    app.setApplicationName("AuraRouter")
    app.aboutToQuit.connect(lock.release)

    # Apply design-system theme.
    from aurarouter.gui.theme import apply_theme

    apply_theme(app, "dark")

    # Create AuraRouterAPI.
    from aurarouter.api import APIConfig, AuraRouterAPI

    api_config = APIConfig(
        config_path=config_path,
        environment=context.name.lower(),
    )
    api = AuraRouterAPI(api_config)

    window = AuraRouterWindow(api=api, env_context=context)
    window.show()

    # Trigger onboarding wizard on first launch (after event loop starts).
    QTimer.singleShot(0, window.trigger_onboarding_if_needed)

    sys.exit(app.exec())


def main() -> None:
    """Standalone entry-point (aurarouter-gui script)."""
    import argparse
    import logging

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
    parser.add_argument(
        "--environment",
        choices=["local", "auragrid"],
        default=None,
        help="Deployment environment to start in (default: local).",
    )
    args = parser.parse_args()

    context = _create_context(
        environment=args.environment,
        config_path=args.config,
    )

    launch_gui(context, config_path=args.config)
