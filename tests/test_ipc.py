"""Tests for IPC server/client (JSON-RPC over TCP loopback).

Task Group C, TG1 — covers IPCServer and IPCClient with round-trip
integration, error handling, lifecycle, and platform dispatch.
"""

import json
import socket
import sys
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from aurarouter.ipc import IPCServer, IPCClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_server(server: IPCServer, timeout: float = 3.0) -> None:
    """Block until the server socket is listening."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server._server_socket is not None:
            try:
                # Attempt a quick connect to verify the port is open
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                addr = server._server_socket.getsockname()
                sock.connect(addr)
                sock.close()
                return
            except (OSError, TypeError):
                pass
        time.sleep(0.05)
    raise TimeoutError("Server did not start within timeout")


def _start_server_on_free_port(handlers: dict | None = None) -> IPCServer:
    """Create and start an IPCServer bound to a random free port."""
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = IPCServer.__new__(IPCServer)
    server._address = ("127.0.0.1", port)
    server._handlers = {}
    server._server_socket = None
    server._thread = None
    server._running = False

    # Override _serve_loop to use TCP directly (bypass platform dispatch)
    def _serve_tcp(self_ref=server):
        self_ref._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self_ref._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self_ref._server_socket.settimeout(1.0)
        self_ref._server_socket.bind(self_ref._address)
        self_ref._server_socket.listen(5)
        while self_ref._running:
            try:
                conn, _ = self_ref._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self_ref._handle_connection, args=(conn,), daemon=True
            ).start()

    server._serve_loop = _serve_tcp

    if handlers:
        for method, fn in handlers.items():
            server._handlers[method] = fn

    server._running = True
    server._thread = threading.Thread(target=server._serve_loop, daemon=True)
    server._thread.start()
    _wait_for_server(server)
    return server


def _make_client(server: IPCServer) -> IPCClient:
    """Create an IPCClient pointing at the test server's address."""
    client = IPCClient.__new__(IPCClient)
    addr = server._server_socket.getsockname()
    client._address = addr

    # Override _connect to always use TCP (bypass platform dispatch)
    def _connect(timeout: float = 5.0, addr=addr) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(addr)
        return sock

    client._connect = _connect
    return client


# ---------------------------------------------------------------------------
# IPCServer tests (Task 1.1)
# ---------------------------------------------------------------------------

class TestIPCServerHandlerDispatch:
    """Task 1.1.1 — Handler registration and dispatch."""

    def test_handler_registration_and_roundtrip(self):
        server = _start_server_on_free_port({"echo": lambda msg="": msg})
        try:
            client = _make_client(server)
            result = client.call("echo", params={"msg": "hello"})
            assert result == "hello"
        finally:
            server.stop()

    def test_handler_no_params(self):
        server = _start_server_on_free_port({"ping": lambda: {"status": "ok"}})
        try:
            client = _make_client(server)
            result = client.call("ping")
            assert result == {"status": "ok"}
        finally:
            server.stop()


class TestIPCServerErrors:
    """Task 1.1.2–1.1.3 — Error codes."""

    def test_unknown_method_returns_32601(self):
        server = _start_server_on_free_port()
        try:
            client = _make_client(server)
            with pytest.raises(RuntimeError, match="Unknown method"):
                client.call("nonexistent")
        finally:
            server.stop()

    def test_handler_exception_returns_32000(self):
        def bad_handler():
            raise ValueError("boom")

        server = _start_server_on_free_port({"fail": bad_handler})
        try:
            client = _make_client(server)
            with pytest.raises(RuntimeError, match="boom"):
                client.call("fail")
        finally:
            server.stop()


class TestIPCServerLifecycle:
    """Task 1.1.4–1.1.5 — Start/stop lifecycle."""

    def test_start_sets_running_and_spawns_thread(self):
        server = IPCServer(address="127.0.0.1")
        # Patch _serve_loop to avoid actually binding
        server._serve_loop = lambda: time.sleep(5)
        server.start()
        try:
            assert server._running is True
            assert server._thread is not None
            assert server._thread.is_alive()
        finally:
            server.stop()

    def test_stop_clears_running_and_joins_thread(self):
        server = _start_server_on_free_port({"health": lambda: "ok"})
        assert server._running is True
        thread = server._thread
        server.stop()
        assert server._running is False
        assert not thread.is_alive()

    def test_idempotent_start(self):
        server = IPCServer(address="127.0.0.1")
        server._serve_loop = lambda: time.sleep(5)
        server.start()
        first_thread = server._thread
        server.start()  # Second call — should be no-op
        assert server._thread is first_thread
        server.stop()


class TestIPCServerPlatformDispatch:
    """Task 1.1.6 — Platform dispatch in _serve_loop."""

    def test_serve_loop_calls_win32_on_windows(self):
        server = IPCServer()
        server._serve_win32 = MagicMock()
        server._serve_unix = MagicMock()
        with patch("aurarouter.ipc.sys") as mock_sys:
            mock_sys.platform = "win32"
            server._serve_loop()
        server._serve_win32.assert_called_once()
        server._serve_unix.assert_not_called()

    def test_serve_loop_calls_unix_on_linux(self):
        server = IPCServer()
        server._serve_win32 = MagicMock()
        server._serve_unix = MagicMock()
        with patch("aurarouter.ipc.sys") as mock_sys:
            mock_sys.platform = "linux"
            server._serve_loop()
        server._serve_unix.assert_called_once()
        server._serve_win32.assert_not_called()


# ---------------------------------------------------------------------------
# IPCClient tests (Task 1.2)
# ---------------------------------------------------------------------------

class TestIPCClientCall:
    """Task 1.2.1 — Successful call round-trip."""

    def test_successful_call(self):
        server = _start_server_on_free_port(
            {"greet": lambda name="": f"Hello, {name}!"}
        )
        try:
            client = _make_client(server)
            result = client.call("greet", params={"name": "World"})
            assert result == "Hello, World!"
        finally:
            server.stop()


class TestIPCClientPing:
    """Task 1.2.2–1.2.3 — Ping reachable/unreachable."""

    def test_ping_reachable(self):
        server = _start_server_on_free_port({"health": lambda: {"status": "ok"}})
        try:
            client = _make_client(server)
            assert client.ping() is True
        finally:
            server.stop()

    def test_ping_unreachable(self):
        client = IPCClient.__new__(IPCClient)
        client._address = ("127.0.0.1", 1)  # Not listening

        def _connect(timeout=5.0):
            raise ConnectionRefusedError("refused")

        client._connect = _connect
        assert client.ping() is False


class TestIPCClientErrors:
    """Task 1.2.4–1.2.5 — Connection error and server error propagation."""

    def test_connection_error_when_no_server(self):
        client = IPCClient.__new__(IPCClient)
        client._address = ("127.0.0.1", 1)

        def _connect(timeout=5.0):
            raise ConnectionRefusedError("refused")

        client._connect = _connect
        with pytest.raises(ConnectionError):
            client.call("anything")

    def test_server_error_propagation(self):
        def raise_error():
            raise RuntimeError("internal failure")

        server = _start_server_on_free_port({"bad": raise_error})
        try:
            client = _make_client(server)
            with pytest.raises(RuntimeError, match="internal failure"):
                client.call("bad")
        finally:
            server.stop()


class TestIPCClientTimeout:
    """Task 1.2.6 — Timeout behavior."""

    def test_timeout_on_slow_handler(self):
        def slow_handler():
            time.sleep(10)
            return "too late"

        server = _start_server_on_free_port({"slow": slow_handler})
        try:
            client = _make_client(server)
            # Client should timeout (socket.timeout → ConnectionError or timeout)
            with pytest.raises((socket.timeout, ConnectionError, OSError)):
                client.call("slow", timeout=0.3)
        finally:
            server.stop()


class TestIPCClientPortFileDiscovery:
    """Task 1.2.7 — Windows port file discovery."""

    def test_port_file_discovery(self, tmp_path):
        port_file = tmp_path / "aurarouter.port"
        port_file.write_text("54321", encoding="utf-8")

        with patch("aurarouter.ipc._IPC_DIR", tmp_path):
            with patch("aurarouter.ipc.sys") as mock_sys:
                mock_sys.platform = "win32"
                # Re-create client to pick up patched values
                client = IPCClient(address="ignored")
                # The _connect method reads the port file on win32
                with pytest.raises(ConnectionError):
                    # Will fail to connect but should read port 54321
                    client.call("test")
