"""Tests for singleton lock and IPC server/client."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from aurarouter.singleton import SingletonLock, _PID_FILE


@pytest.fixture(autouse=True)
def clean_pid_file():
    """Ensure no stale PID file from previous test runs."""
    if _PID_FILE.is_file():
        _PID_FILE.unlink()
    yield
    if _PID_FILE.is_file():
        _PID_FILE.unlink()


class TestSingletonLock:
    def test_acquire_and_release(self):
        lock = SingletonLock()
        assert lock.acquire()
        assert _PID_FILE.is_file()
        data = json.loads(_PID_FILE.read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid()
        lock.release()
        assert not _PID_FILE.is_file()

    def test_double_acquire_same_instance(self):
        lock = SingletonLock()
        assert lock.acquire()
        assert lock.acquire()  # idempotent
        lock.release()

    def test_get_existing_returns_none_when_no_pid_file(self):
        lock = SingletonLock()
        assert lock.get_existing_instance() is None

    def test_get_existing_detects_live_process(self):
        # Write a PID file with the current process PID but from a
        # "different" perspective (our PID won't conflict with itself
        # in get_existing_instance because it checks pid != os.getpid()).
        # Use PID 1 (always alive on Unix, usually alive on Windows).
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(json.dumps({"pid": 1}), encoding="utf-8")

        lock = SingletonLock()
        # PID 1 should be alive, so get_existing should return it.
        result = lock.get_existing_instance()
        # On Windows, PID 1 may not be accessible, so this test is
        # platform-dependent. We just check the function doesn't crash.
        if result is not None:
            assert result["pid"] == 1

    def test_stale_pid_file_is_cleaned(self):
        # Write a PID for a process that definitely doesn't exist.
        dead_pid = 2_000_000_000  # very unlikely to be a real PID
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(json.dumps({"pid": dead_pid}), encoding="utf-8")

        lock = SingletonLock()
        result = lock.get_existing_instance()
        assert result is None  # stale file should be cleaned
        assert not _PID_FILE.is_file()

    def test_corrupt_pid_file_is_cleaned(self):
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text("not valid json", encoding="utf-8")

        lock = SingletonLock()
        result = lock.get_existing_instance()
        assert result is None
        assert not _PID_FILE.is_file()

    def test_acquire_fails_when_live_instance_exists(self):
        # Write a PID file for a process we know is alive (our own,
        # but pretend it's a different PID by writing PID 1).
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(json.dumps({"pid": 1}), encoding="utf-8")

        lock = SingletonLock()
        # This should fail on systems where PID 1 is alive.
        result = lock.acquire()
        if lock.get_existing_instance() is not None:
            assert not result
        lock.release()


class TestIPCServerClient:
    def test_ping_when_no_server(self):
        from aurarouter.ipc import IPCClient

        # Use an ephemeral port that nothing is listening on
        client = IPCClient(port=0)
        # port=0 won't connect to anything; use a high unlikely port instead
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        unused_port = s.getsockname()[1]
        s.close()
        client = IPCClient(port=unused_port)
        assert not client.ping(timeout=0.5)

    def test_server_client_roundtrip(self):
        from aurarouter.ipc import IPCClient, IPCServer

        server = IPCServer(port=0)  # OS-assigned ephemeral port
        server.register("health", lambda: {"status": "ok"})
        server.register("echo", lambda message="": {"echo": message})
        server.start()

        try:
            # Give server time to bind.
            time.sleep(0.3)

            client = IPCClient(port=server.port)
            assert client.ping(timeout=2.0)

            result = client.call("health", timeout=2.0)
            assert result == {"status": "ok"}

            result = client.call("echo", params={"message": "hello"}, timeout=2.0)
            assert result == {"echo": "hello"}
        finally:
            server.stop()

    def test_call_unknown_method(self):
        from aurarouter.ipc import IPCClient, IPCServer

        server = IPCServer(port=0)  # OS-assigned ephemeral port
        server.register("health", lambda: {"status": "ok"})
        server.start()

        try:
            time.sleep(0.3)
            client = IPCClient(port=server.port)
            with pytest.raises(RuntimeError, match="Unknown method"):
                client.call("nonexistent", timeout=2.0)
        finally:
            server.stop()
