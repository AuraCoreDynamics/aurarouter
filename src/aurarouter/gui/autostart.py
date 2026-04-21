"""Windows auto-start management for AuraRouter.

Adds or removes AuraRouter from the Windows startup registry.
No-op on non-Windows platforms.
"""

from __future__ import annotations

import sys


def set_autostart(enabled: bool) -> None:
    """Add/remove AuraRouter from Windows startup via registry. No-op on non-Windows."""
    if sys.platform != "win32":
        return
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "AuraRouter"
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE,
        )
        if enabled:
            import shutil

            exe_path = shutil.which("aurarouter-gui") or f"{sys.executable} -m aurarouter.gui.app"
            winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, exe_path)
        else:
            try:
                winreg.DeleteValue(key, app_name)
            except FileNotFoundError:
                pass
        winreg.CloseKey(key)
    except OSError:
        pass


def get_autostart() -> bool:
    """Check if AuraRouter is registered for auto-start. Returns False on non-Windows."""
    if sys.platform != "win32":
        return False
    import winreg

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_READ,
        )
        winreg.QueryValueEx(key, "AuraRouter")
        winreg.CloseKey(key)
        return True
    except (FileNotFoundError, OSError):
        return False
