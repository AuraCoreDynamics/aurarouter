"""Cross-process singleton enforcement for AuraRouter.

Uses a PID file with liveness check as the primary mechanism.
On Windows, also acquires a named kernel mutex for robustness.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("AuraRouter.Singleton")

_LOCK_DIR = Path.home() / ".auracore" / "aurarouter"
_PID_FILE = _LOCK_DIR / "aurarouter.pid"


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't have permission to signal it.
            return True


class SingletonLock:
    """Prevents multiple AuraRouter instances from running simultaneously.

    Usage::

        lock = SingletonLock()
        existing = lock.get_existing_instance()
        if existing:
            print(f"Already running: PID {existing['pid']}")
        elif lock.acquire():
            try:
                ...  # run application
            finally:
                lock.release()
    """

    def __init__(self) -> None:
        self._acquired = False
        self._mutex_handle = None  # Windows kernel mutex

    def acquire(self) -> bool:
        """Attempt to acquire the singleton lock.

        Returns True if this process now holds the lock, False if another
        instance is already running.
        """
        if self._acquired:
            return True

        _LOCK_DIR.mkdir(parents=True, exist_ok=True)

        # Check for stale PID file.
        existing = self.get_existing_instance()
        if existing is not None:
            return False

        # On Windows, also try a named kernel mutex for robustness.
        if sys.platform == "win32":
            if not self._acquire_win_mutex():
                return False

        # Write our PID file.
        info = {"pid": os.getpid()}
        try:
            _PID_FILE.write_text(json.dumps(info), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write PID file: %s", exc)
            return False

        self._acquired = True
        logger.info("Singleton lock acquired (PID %d)", os.getpid())
        return True

    def get_existing_instance(self) -> Optional[dict]:
        """If another AuraRouter instance is running, return its info.

        Returns a dict with ``pid`` key, or ``None`` if no live instance
        is detected.
        """
        if not _PID_FILE.is_file():
            return None

        try:
            data = json.loads(_PID_FILE.read_text(encoding="utf-8"))
            pid = data.get("pid")
            if pid is None:
                return None

            # Don't treat our own PID as a conflict.
            if pid == os.getpid():
                return None

            if _is_pid_alive(pid):
                return data

            # Stale PID file — process is dead.
            logger.info("Removing stale PID file (PID %d is dead)", pid)
            _PID_FILE.unlink(missing_ok=True)
            return None
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt PID file, removing: %s", exc)
            _PID_FILE.unlink(missing_ok=True)
            return None

    def release(self) -> None:
        """Release the singleton lock."""
        if not self._acquired:
            return

        try:
            if _PID_FILE.is_file():
                _PID_FILE.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove PID file: %s", exc)

        if sys.platform == "win32":
            self._release_win_mutex()

        self._acquired = False
        logger.info("Singleton lock released")

    # ------------------------------------------------------------------
    # Windows named mutex helpers
    # ------------------------------------------------------------------

    def _acquire_win_mutex(self) -> bool:
        """Try to create a Windows named mutex. Returns False if already held."""
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            ERROR_ALREADY_EXISTS = 183

            handle = kernel32.CreateMutexW(None, True, "Global\\AuraRouter_Singleton")
            last_err = ctypes.get_last_error()

            if handle == 0:
                return False

            if last_err == ERROR_ALREADY_EXISTS:
                kernel32.CloseHandle(handle)
                return False

            self._mutex_handle = handle
            return True
        except Exception:
            # ctypes may not work in all environments; fall back to PID-only.
            return True

    def _release_win_mutex(self) -> None:
        """Release the Windows named mutex."""
        if self._mutex_handle is not None:
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                kernel32.ReleaseMutex(self._mutex_handle)
                kernel32.CloseHandle(self._mutex_handle)
            except Exception:
                pass
            self._mutex_handle = None
