"""
Client utilities for communicating with the OmniMemory CLI daemon.
"""

from multiprocessing.connection import Client
import socket
from typing import Any, Dict, Optional

from .daemon_constants import DAEMON_HOST, DAEMON_PORT, DAEMON_AUTH_KEY


class DaemonNotRunningError(Exception):
    """Raised when the daemon is not running."""


class DaemonResponseError(Exception):
    """Raised when the daemon returns an error."""


def _connect(timeout: Optional[float] = None):
    return Client(
        (DAEMON_HOST, DAEMON_PORT),
        authkey=DAEMON_AUTH_KEY,
    )


def call_daemon(method: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    """Send a request to the daemon and return the response result."""
    try:
        conn = _connect()
    except (ConnectionRefusedError, FileNotFoundError, socket.error) as exc:
        raise DaemonNotRunningError from exc

    try:
        conn.send({"method": method, "payload": payload})
        response = conn.recv()
    finally:
        conn.close()

    if not isinstance(response, dict):
        raise DaemonResponseError("Invalid response from daemon")

    status = response.get("status")
    if status != "ok":
        raise DaemonResponseError(response.get("error", "Unknown error"))

    return response.get("result")


def is_daemon_running() -> bool:
    """Return True if the daemon is running."""
    try:
        call_daemon("ping", {})
        return True
    except (DaemonNotRunningError, DaemonResponseError):
        return False
