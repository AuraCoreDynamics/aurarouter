"""Lightweight IPC server/client for cross-process AuraRouter communication.

Uses named pipes on Windows and Unix domain sockets on Linux/macOS.
Protocol is newline-delimited JSON-RPC (request/response).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("AuraRouter.IPC")

_IPC_DIR = Path.home() / ".auracore" / "aurarouter"

# On Windows we use a named pipe path; on Unix a domain socket file.
if sys.platform == "win32":
    IPC_ADDRESS = r"\\.\pipe\AuraRouter"
else:
    IPC_ADDRESS = str(_IPC_DIR / "aurarouter.sock")


class IPCServer:
    """JSON-RPC server over named pipe (Windows) or Unix socket.

    Register handlers, then call ``start()`` to begin accepting connections
    in a background daemon thread.

    Usage::

        server = IPCServer()
        server.register("health", lambda: {"status": "ok"})
        server.register("get_state", lambda: {"state": "running"})
        server.start()
        ...
        server.stop()
    """

    def __init__(self, address: str = IPC_ADDRESS, *, port: int = 19470) -> None:
        self._address = address
        self._port = port
        self._bound_port: int | None = None
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def port(self) -> int | None:
        """Return the TCP port the server is bound to (Windows), or ``None``."""
        return self._bound_port

    def register(self, method: str, handler: Callable[..., Any]) -> None:
        """Register a JSON-RPC method handler."""
        self._handlers[method] = handler

    def start(self) -> None:
        """Start accepting connections in a background daemon thread."""
        if self._running:
            return

        if sys.platform != "win32":
            # Clean up stale socket file.
            sock_path = Path(self._address)
            if sock_path.exists():
                sock_path.unlink()
            _IPC_DIR.mkdir(parents=True, exist_ok=True)

        self._running = True
        self._thread = threading.Thread(
            target=self._serve_loop, name="AuraRouter-IPC", daemon=True
        )
        self._thread.start()
        logger.info("IPC server started at %s", self._address)

    def stop(self) -> None:
        """Stop the IPC server."""
        self._running = False
        # Close the server socket to unblock accept().
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)
        logger.info("IPC server stopped")

    def _serve_loop(self) -> None:
        """Accept connections and handle requests."""
        try:
            if sys.platform == "win32":
                self._serve_win32()
            else:
                self._serve_unix()
        except Exception:
            if self._running:
                logger.exception("IPC server error")

    def _serve_unix(self) -> None:
        """Serve on a Unix domain socket."""
        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.settimeout(1.0)
        self._server_socket.bind(self._address)
        self._server_socket.listen(5)

        while self._running:
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
            ).start()

    def _serve_win32(self) -> None:
        """Serve on a TCP loopback socket (Windows named pipes require
        win32pipe which isn't always available; TCP localhost is simpler
        and equally effective for single-machine IPC).
        """
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        # Bind to localhost. Port 0 lets the OS assign an ephemeral port.
        self._server_socket.bind(("127.0.0.1", self._port))
        self._server_socket.listen(5)
        self._bound_port = self._server_socket.getsockname()[1]

        # Write the port to a file so clients can discover it.
        port_file = _IPC_DIR / "aurarouter.port"
        _IPC_DIR.mkdir(parents=True, exist_ok=True)
        port_file.write_text(str(self._bound_port), encoding="utf-8")

        while self._running:
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
            ).start()

        # Clean up port file.
        try:
            port_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _handle_connection(self, conn: socket.socket) -> None:
        """Handle a single client connection (one request/response)."""
        try:
            conn.settimeout(10.0)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if not data:
                return

            request = json.loads(data.decode("utf-8").strip())
            method = request.get("method", "")
            req_id = request.get("id")
            params = request.get("params", {})

            handler = self._handlers.get(method)
            if handler is None:
                response = {
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }
            else:
                try:
                    result = handler(**params) if params else handler()
                    response = {"id": req_id, "result": result}
                except Exception as exc:
                    response = {
                        "id": req_id,
                        "error": {"code": -32000, "message": str(exc)},
                    }

            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception:
            logger.debug("IPC connection error", exc_info=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass


class IPCClient:
    """JSON-RPC client for communicating with a running AuraRouter instance.

    Usage::

        client = IPCClient()
        if client.ping():
            state = client.call("get_state")
            print(state)
    """

    def __init__(self, address: str = IPC_ADDRESS, *, port: int | None = None) -> None:
        self._address = address
        self._port_override = port

    def ping(self, timeout: float = 2.0) -> bool:
        """Check if the IPC server is reachable."""
        try:
            result = self.call("health", timeout=timeout)
            return result is not None
        except Exception:
            return False

    def call(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: float = 5.0,
    ) -> Any:
        """Send a JSON-RPC request and return the result.

        Raises ``ConnectionError`` if the server is not reachable.
        Raises ``RuntimeError`` if the server returns an error response.
        """
        request = {"method": method, "id": 1}
        if params:
            request["params"] = params

        payload = (json.dumps(request) + "\n").encode("utf-8")

        try:
            sock = self._connect(timeout)
        except Exception as exc:
            raise ConnectionError(f"Cannot connect to AuraRouter IPC: {exc}") from exc

        try:
            sock.settimeout(timeout)
            sock.sendall(payload)

            # Read response.
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if not data:
                raise ConnectionError("Empty response from IPC server")

            response = json.loads(data.decode("utf-8").strip())
            if "error" in response:
                err = response["error"]
                raise RuntimeError(f"IPC error: {err.get('message', err)}")

            return response.get("result")
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _connect(self, timeout: float) -> socket.socket:
        """Create a connected socket to the IPC server."""
        if sys.platform == "win32":
            # Connect to TCP localhost.
            if self._port_override is not None:
                port = self._port_override
            else:
                port_file = _IPC_DIR / "aurarouter.port"
                if port_file.is_file():
                    port = int(port_file.read_text(encoding="utf-8").strip())
                else:
                    port = 19470
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(("127.0.0.1", port))
            return sock
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(self._address)
            return sock
