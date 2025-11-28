"""
Client utilities for communicating with the OmniMemory CLI daemon.
"""

import socket
from multiprocessing.connection import Client, Connection
from typing import Any, Dict, Optional

from .daemon_constants import DAEMON_HOST, DAEMON_PORT, DAEMON_AUTH_KEY


class DaemonNotRunningError(Exception):
    """Raised when the daemon is not running."""


class DaemonResponseError(Exception):
    """Raised when the daemon returns an error."""


def _connect(timeout: Optional[float] = None) -> Connection:
    """
    Create a connection to the daemon.

    Args:
        timeout: Optional connection timeout in seconds.

    Returns:
        Multiprocessing Connection object bound to the daemon endpoint.
    """
    return Client(
        (DAEMON_HOST, DAEMON_PORT),
        authkey=DAEMON_AUTH_KEY,
        family="AF_INET",
    )


def call_daemon(method: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    """
    Send a request to the daemon and return the response payload.

    Args:
        method: RPC method name understood by the daemon.
        payload: Optional dictionary containing method parameters.

    Returns:
        Deserialized response data returned by the daemon.

    Raises:
        DaemonNotRunningError: If the daemon socket cannot be reached.
        DaemonResponseError: If the daemon responds with an error or malformed data.
    """
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
    """
    Check if the daemon process is reachable.

    Returns:
        True if the daemon responded to a ping request, False otherwise.
    """
    try:
        call_daemon("ping", {})
        return True
    except (DaemonNotRunningError, DaemonResponseError):
        return False
