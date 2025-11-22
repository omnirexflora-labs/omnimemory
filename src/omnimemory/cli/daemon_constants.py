"""
Constants and helpers for the OmniMemory CLI daemon.
"""

from pathlib import Path

DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 59611
DAEMON_AUTH_KEY = b"omnimemory-daemon"

STATE_DIR = Path.home() / ".omnimemory"
PID_FILE = STATE_DIR / "daemon.pid"
LOG_FILE = STATE_DIR / "daemon.log"


def ensure_state_dir() -> None:
    """Ensure the daemon state directory exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
