"""
Background daemon that keeps OmniMemorySDK initialized for CLI commands (async version).
"""

from multiprocessing.connection import Listener
import os
import signal
import traceback
import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Coroutine

from omnimemory.sdk import OmniMemorySDK
from omnimemory.core.schemas import (
    AddUserMessageRequest,
    ConversationSummaryRequest,
    AgentMemoryRequest,
)

from .daemon_constants import (
    DAEMON_HOST,
    DAEMON_PORT,
    DAEMON_AUTH_KEY,
    ensure_state_dir,
    PID_FILE,
    LOG_FILE,
)


class DaemonServer:
    """Persistent server that executes OmniMemorySDK operations (async version)."""

    def __init__(self):
        ensure_state_dir()
        self._sdk: Optional[OmniMemorySDK] = None
        self.listener: Optional[Listener] = None
        self.running = True
        self._memory_manager_ready = False
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="omnimemory-daemon-loop", daemon=True
        )
        self._loop_thread.start()
        self._loop_ready.wait()

    @property
    def sdk(self) -> OmniMemorySDK:
        if self._sdk is None:
            self._write_log("Initializing SDK (lazy)...")
            self._sdk = OmniMemorySDK()
            self._write_log("SDK initialized")
        return self._sdk

    def start(self):
        """Start the listener loop with robust error handling."""
        listener = None
        try:
            listener = Listener(
                (DAEMON_HOST, DAEMON_PORT),
                authkey=DAEMON_AUTH_KEY,
            )
            self.listener = listener
            self._write_pid()
            self._write_log("Daemon started")

            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

            while self.running:
                try:
                    conn = listener.accept()
                except (OSError, EOFError) as e:
                    if not self.running:
                        break
                    self._write_log(f"Connection accept error: {e}")
                    continue
                except Exception as e:
                    if not self.running:
                        break
                    self._write_log(f"Unexpected error accepting connection: {e}")
                    self._write_log(traceback.format_exc())
                    time.sleep(0.1)
                    continue

                try:
                    request = conn.recv()
                    response = self._handle_request(request)
                    conn.send({"status": "ok", "result": response})
                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                    self._write_log(f"Error handling request: {error_msg}")
                    self._write_log(traceback.format_exc())
                    try:
                        conn.send({"status": "error", "error": error_msg})
                    except Exception:
                        pass
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

        except OSError as e:
            error_msg = f"Failed to start daemon listener: {e}"
            self._write_log(error_msg)
            self._write_log(traceback.format_exc())
            raise
        except Exception as e:
            error_msg = f"Unexpected error in daemon start: {e}"
            self._write_log(error_msg)
            self._write_log(traceback.format_exc())
            raise
        finally:
            self._cleanup()

    def _handle_signal(self, signum, frame):
        """Signal handler to shut down gracefully."""
        self._write_log(f"Received signal {signum}, shutting down.")
        self.running = False
        if self.listener:
            self.listener.close()

    def _initialize_memory_manager_async(self):
        """Initialize memory manager in background thread."""

        def _init():
            try:
                manager = self.sdk.memory_manager
                self._memory_manager_ready = manager is not None
                if self._memory_manager_ready:
                    self._write_log("MemoryManager initialized successfully")
                    try:
                        warm_up_ok = self._run_async(self.sdk.warm_up())
                        if warm_up_ok:
                            self._write_log("Connection pool warm-up completed")
                        else:
                            self._write_log("Connection pool warm-up failed")
                    except Exception as e:
                        self._write_log(f"Connection pool warm-up error: {e}")
            except Exception as exc:
                self._memory_manager_ready = False
                self._write_log(f"MemoryManager initialization failed: {exc}")

        thread = threading.Thread(target=_init, daemon=True, name="daemon-init")
        thread.start()

    def _run_loop(self):
        """Background event-loop runner."""
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _run_async(self, coro: Coroutine[Any, Any, Any]):
        """Submit a coroutine to the background loop and wait for the result."""
        if not self._loop or not self._loop_thread.is_alive():
            raise RuntimeError("Daemon event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _cancel_pending_tasks(self):
        """Cancel all pending tasks running inside the daemon loop."""
        tasks = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _shutdown_loop(self):
        """Stop the background event loop and join the thread."""
        if not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._cancel_pending_tasks(), self._loop
            ).result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    def _ensure_memory_manager_ready(self):
        """Ensure memory manager is initialized (lazy initialization)."""
        if not self._memory_manager_ready:
            try:
                manager = self.sdk.memory_manager
                self._memory_manager_ready = manager is not None
            except Exception:
                pass

    def _update_memory_manager_flag(self):
        """Update cached readiness flag based on SDK state."""
        self._memory_manager_ready = (
            getattr(self.sdk, "_memory_manager", None) is not None
        )

    def _handle_request(self, request: Dict[str, Any]) -> Any:
        """Handle request synchronously, running async SDK methods in event loop."""
        method = request.get("method")
        payload = request.get("payload") or {}

        if method == "ping":
            return {"timestamp": datetime.now(timezone.utc).isoformat()}
        if method == "status":
            _ = self.sdk
            try:
                manager = self.sdk.memory_manager
                self._memory_manager_ready = manager is not None
            except Exception:
                self._memory_manager_ready = False
                if not self._memory_manager_ready:
                    self._initialize_memory_manager_async()

            pool_stats = {}
            if self._memory_manager_ready:
                try:
                    pool_stats = self._run_async(self.sdk.get_connection_pool_stats())
                except Exception:
                    pass
            return {
                "sdk_initialized": True,
                "memory_manager_initialized": self._memory_manager_ready,
                "connection_pool": pool_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        if method == "add_memory":
            user_message = AddUserMessageRequest(**payload["user_message"])
            result = self._run_async(self.sdk.add_memory(user_message))
            self._ensure_memory_manager_ready()
            return result
        if method == "query_memory":
            self._ensure_memory_manager_ready()
            result = self._run_async(self.sdk.query_memory(**payload))
            return result
        if method == "get_memory":
            self._ensure_memory_manager_ready()
            result = self._run_async(self.sdk.get_memory(**payload))
            return result
        if method == "traverse_memory_evolution_chain":
            self._ensure_memory_manager_ready()
            result = self._run_async(
                self.sdk.traverse_memory_evolution_chain(**payload)
            )
            return result
        if method == "delete_memory":
            self._ensure_memory_manager_ready()
            result = self._run_async(self.sdk.delete_memory(**payload))
            return result
        if method == "summarize_conversation":
            summary_request = ConversationSummaryRequest(**payload["summary_request"])
            result = self._run_async(self.sdk.summarize_conversation(summary_request))
            self._ensure_memory_manager_ready()
            return result
        if method == "add_agent_memory":
            agent_request = AgentMemoryRequest(**payload["agent_request"])
            result = self._run_async(self.sdk.add_agent_memory(agent_request))
            self._ensure_memory_manager_ready()
            return result
        if method == "shutdown":
            self.running = False
            return {"message": "Shutting down"}
        if method == "generate_evolution_graph":
            chain = payload.get("chain", [])
            fmt = payload.get("format", "mermaid")
            self._ensure_memory_manager_ready()
            return self.sdk.generate_evolution_graph(chain=chain, format=fmt)

        raise ValueError(f"Unsupported daemon method: {method}")

    def _write_pid(self):
        PID_FILE.write_text(str(os.getpid()))

    def _cleanup(self):
        """Cleanup resources with robust error handling."""
        try:
            if self.listener:
                try:
                    self.listener.close()
                except Exception as e:
                    self._write_log(f"Error closing listener: {e}")
                finally:
                    self.listener = None
        except Exception as e:
            self._write_log(f"Error in listener cleanup: {e}")

        try:
            self._shutdown_loop()
        except Exception as e:
            self._write_log(f"Error shutting down event loop: {e}")

        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception as e:
            self._write_log(f"Error removing PID file: {e}")

        self._write_log("Daemon stopped")

    def _write_log(self, message: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")


def main():
    server = DaemonServer()
    server.start()


if __name__ == "__main__":
    main()
