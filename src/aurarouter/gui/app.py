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
    from aurarouter.config import ConfigLoader

    config = ConfigLoader()
    launch_gui(config)
