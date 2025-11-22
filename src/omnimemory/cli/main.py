"""
OmniMemory CLI - Command Line Interface

Beautiful CLI for interacting with OmniMemory SDK.
"""

import json
import typer
from pathlib import Path
from typing import Optional, List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.tree import Tree
from rich.syntax import Syntax
from rich.columns import Columns
from rich.align import Align
from rich.text import Text
from datetime import datetime

from omnimemory.core.schemas import (
    AddUserMessageRequest,
    DEFAULT_MAX_MESSAGES,
)
from importlib import metadata as importlib_metadata
from omnimemory.cli.banner import OMNIMEMORY_BANNER
from omnimemory.cli.daemon_client import (
    call_daemon,
    DaemonNotRunningError,
    DaemonResponseError,
    is_daemon_running,
)
from omnimemory.cli.daemon_constants import ensure_state_dir, LOG_FILE, PID_FILE
import subprocess
import sys
import time
import os
import signal

console = Console(record=True, force_terminal=True)

DAEMON_START_TIMEOUT = 30.0
DAEMON_STOP_TIMEOUT = 10.0
DAEMON_POLL_INTERVAL = 0.25

app = typer.Typer(
    name="omnimemory",
    help="[bold cyan]🧠 OmniMemory CLI[/] - Advanced Memory Management System",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
)

memory_app = typer.Typer(
    name="memory",
    help="[bold green]💾 Memory Operations[/] - Add, query, get, and delete memories",
    rich_markup_mode="rich",
)

daemon_app = typer.Typer(
    name="daemon",
    help="[bold yellow]⚙️  Daemon Management[/] - Start/stop the persistent SDK daemon",
    rich_markup_mode="rich",
)

agent_app = typer.Typer(
    name="agent",
    help="[bold magenta]🤖 Agent Operations[/] - Single-agent workflows and summaries",
    rich_markup_mode="rich",
)

app.add_typer(memory_app, name="memory")
app.add_typer(daemon_app, name="daemon")
app.add_typer(agent_app, name="agent")


def get_version() -> str:
    """Get the package version from metadata."""
    try:
        return importlib_metadata.version("omnimemory")
    except importlib_metadata.PackageNotFoundError:
        try:
            version_file = Path(__file__).parent.parent.parent.parent / "_version.py"
            if version_file.exists():
                with open(version_file) as f:
                    for line in f:
                        if line.startswith("__version__"):
                            return line.split("=")[1].strip().strip('"').strip("'")
        except Exception:
            pass
        return "dev"


def _show_welcome_screen():
    """Display welcome screen with feature overview."""
    version_str = get_version()

    console.print()
    console.print(
        create_header_panel(
            "OmniMemory CLI",
            f"Version {version_str} - Advanced Memory Management System",
        )
    )
    console.print()

    features_table = Table(
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        header_style="bold cyan",
        show_header=True,
        title="[bold white]Core Features[/]",
    )
    features_table.add_column("Feature", style="bold white", width=30)
    features_table.add_column("Description", style="dim", width=60)

    features_table.add_row(
        "[bold green]💾 Memory Operations[/]",
        "Full dual-agent pipeline: create, query, get, and delete memories with intelligent conflict resolution",
    )
    features_table.add_row(
        "[bold magenta]🤖 Agent Memory[/]",
        "Fast single-agent path: agents can store memories directly (<10s) with flexible input (string or array)",
    )
    features_table.add_row(
        "[bold blue]📝 Conversation Summary[/]",
        "Smart summarization: sync mode for instant results, async with webhook for rich metadata",
    )
    features_table.add_row(
        "[bold yellow]⚙️  Daemon Service[/]",
        "Persistent background daemon for fast CLI operations with connection pooling",
    )
    features_table.add_row(
        "[bold cyan]🔍 Intelligent Query[/]",
        "Multi-dimensional ranking: relevance × (recency + importance) for optimal retrieval",
    )
    features_table.add_row(
        "[bold red]🔄 Self-Evolution[/]",
        "AI-powered conflict resolution: automatically consolidates, updates, or deletes related memories",
    )

    console.print(features_table)
    console.print()

    commands_table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        title="[bold white]Quick Start Commands[/]",
    )
    commands_table.add_column("Command", style="bold cyan", width=40)
    commands_table.add_column("Description", style="dim", width=50)

    commands_table.add_row(
        "[bold]omnimemory memory add[/]",
        "Create memory from user messages (full pipeline)",
    )
    commands_table.add_row(
        "[bold]omnimemory agent add-memory[/]",
        "Create memory from agent messages (fast path)",
    )
    commands_table.add_row(
        "[bold]omnimemory agent summarize[/]",
        "Generate conversation summary (sync or async)",
    )
    commands_table.add_row(
        "[bold]omnimemory memory query[/]", "Query memories with intelligent ranking"
    )
    commands_table.add_row(
        "[bold]omnimemory daemon start[/]",
        "Start persistent daemon for faster operations",
    )
    commands_table.add_row(
        "[bold]omnimemory --help[/]", "View detailed help for any command"
    )

    console.print(commands_table)
    console.print()

    info_panel = Panel(
        Align.center(
            "[bold cyan]🚀 Production-Ready Features[/]\n\n"
            "[dim]• Asynchronous processing with background tasks[/]\n"
            "[dim]• Connection pooling for optimal performance[/]\n"
            "[dim]• Smart retry logic with exponential backoff[/]\n"
            "[dim]• Flexible input formats (string or structured arrays)[/]\n"
            "[dim]• Webhook support with intelligent error handling[/]\n"
            "[dim]• Comprehensive error handling and logging[/]"
        ),
        title="[bold white]Enterprise Features[/]",
        border_style="bright_cyan",
        box=box.DOUBLE,
    )
    console.print(info_panel)
    console.print()

    console.print(
        "[dim]For detailed documentation, visit: [/][bold cyan]https://github.com/your-repo/omnimemory[/]"
    )
    console.print()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
):
    """OmniMemory CLI - Advanced Memory Management System."""
    if version_flag:
        version_str = get_version()
        console.print(f"[bold cyan]OmniMemory[/] version [bold green]{version_str}[/]")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        _show_welcome_screen()


def success_message(message: str):
    """Display a success message with icon."""
    console.print(f"[bold green]✓[/] {message}")


def error_message(message: str):
    """Display an error message with icon."""
    console.print(f"[bold red]✗[/] {message}")


def warning_message(message: str):
    """Display a warning message with icon."""
    console.print(f"[bold yellow]⚠[/] {message}")


def info_message(message: str):
    """Display an info message with icon."""
    console.print(f"[bold cyan]ℹ[/] {message}")


def create_header_panel(title: str, subtitle: str = "") -> Panel:
    """Create a beautiful header panel."""
    content = f"[bold white]{title}[/]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/]"
    return Panel(
        Align.center(content),
        style="bold cyan",
        border_style="bright_cyan",
        box=box.DOUBLE_EDGE,
    )


def create_metric_card(title: str, value: str, icon: str = "📊") -> Panel:
    """Create a beautiful metric card."""
    content = Align.center(f"{icon}\n[bold bright_white]{value}[/]\n[dim]{title}[/]")
    return Panel(
        content,
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def daemon_request(method: str, payload: Optional[dict] = None):
    """Send a request to the daemon and handle errors."""
    try:
        return call_daemon(method, payload or {})
    except DaemonNotRunningError:
        error_message(
            "OmniMemory daemon is not running.\n"
            "Start it once with [bold]omnimemory daemon start[/] and try again."
        )
        raise typer.Exit(1) from None
    except DaemonResponseError as exc:
        error_message(f"Daemon error: {exc}")
        raise typer.Exit(1) from None


def _kill_process_by_port(port: int) -> bool:
    """Kill any process using the specified port."""
    try:
        import socket

        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(pid) for pid in result.stdout.strip().split("\n") if pid]
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(1)
                    try:
                        os.kill(pid, 0)
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(0.5)
                    except OSError:
                        pass
                except (OSError, ProcessLookupError):
                    pass
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        try:
            result = subprocess.run(
                ["ss", "-lptn", f"sport = :{port}"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                import re

                match = re.search(r"pid=(\d+)", result.stdout)
                if match:
                    pid = int(match.group(1))
                    try:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(1)
                        try:
                            os.kill(pid, 0)
                            os.kill(pid, signal.SIGKILL)
                            time.sleep(0.5)
                        except OSError:
                            pass
                    except (OSError, ProcessLookupError):
                        pass
                    return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return False


def _kill_stale_daemon():
    """Kill stale daemon process if PID file exists but process is not responding."""
    killed = False

    from omnimemory.cli.daemon_constants import DAEMON_PORT

    if _kill_process_by_port(DAEMON_PORT):
        killed = True
        time.sleep(1)

    if not PID_FILE.exists():
        return killed

    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        PID_FILE.unlink()
        return killed

    try:
        os.kill(pid, 0)
    except OSError:
        PID_FILE.unlink()
        return killed

    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            killed = True
        except OSError:
            killed = True
    except OSError:
        pass

    if PID_FILE.exists():
        PID_FILE.unlink()

    return killed


def _wait_for_daemon(
    expected_running: bool,
    timeout: float = DAEMON_START_TIMEOUT,
    poll_interval: float = DAEMON_POLL_INTERVAL,
) -> bool:
    """Wait for the daemon to reach the desired running state."""
    start = time.time()
    while time.time() - start < timeout:
        if is_daemon_running() == expected_running:
            return True
        time.sleep(poll_interval)
    return False


def _load_conversation_payload(
    messages_file: Optional[str],
    message: Optional[str],
    text: Optional[str],
):
    """Load conversation data from file, single message, or raw text."""
    if text:
        if not text.strip():
            error_message("Conversation text cannot be empty.")
            raise typer.Exit(1)
        return text

    if messages_file:
        path = Path(messages_file)
        if not path.exists():
            error_message(f"File not found: {messages_file}")
            raise typer.Exit(1)
        try:
            with path.open("r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
        except json.JSONDecodeError:
            error_message(f"Invalid JSON in {messages_file}")
            raise typer.Exit(1)

        if isinstance(data, dict):
            if isinstance(data.get("messages"), list):
                return data["messages"]
            if isinstance(data.get("text"), str):
                return data["text"]
        if isinstance(data, list):
            return data
        if isinstance(data, str):
            return data

        error_message(
            "Conversation file must contain either 'messages' array or 'text' string."
        )
        raise typer.Exit(1)

    if message:
        parts = message.split(":", 2)
        if len(parts) != 3:
            error_message("Message format must be: role:content:timestamp")
            raise typer.Exit(1)
        return [{"role": parts[0], "content": parts[1], "timestamp": parts[2]}]

    error_message(
        "Provide conversation input via --messages-file, --message, or --text."
    )
    raise typer.Exit(1)


@daemon_app.command("start")
def daemon_start():
    """Start the OmniMemory daemon."""
    ensure_state_dir()

    if is_daemon_running():
        success_message("OmniMemory daemon is already running.")
        return

    from omnimemory.cli.daemon_constants import DAEMON_PORT

    if _kill_stale_daemon():
        warning_message("Killed stale daemon process or process using port.")
        time.sleep(2)

    if PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except OSError:
            pass

    command = [sys.executable, "-m", "omnimemory.cli.daemon_service"]
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=log_file,
                close_fds=True,
                start_new_session=True,
            )
    except Exception as e:
        error_message(f"Failed to start daemon process: {e}")
        raise typer.Exit(1)

    if _wait_for_daemon(expected_running=True, timeout=DAEMON_START_TIMEOUT):
        success_message("OmniMemory daemon started successfully.")
        info_message(f"Logs: {LOG_FILE}")
    else:
        error_message(
            f"Failed to confirm daemon startup within {int(DAEMON_START_TIMEOUT)}s."
        )
        info_message(f"Check logs for details: {LOG_FILE}")
        _kill_stale_daemon()
        raise typer.Exit(1)


@daemon_app.command("stop")
def daemon_stop():
    """Stop the OmniMemory daemon."""
    if not is_daemon_running():
        if PID_FILE.exists():
            warning_message("Daemon not responding but PID file exists. Cleaning up...")
            _kill_stale_daemon()
        else:
            warning_message("OmniMemory daemon is not running.")
        return

    try:
        call_daemon("shutdown", {})
    except (DaemonNotRunningError, DaemonResponseError):
        warning_message("Daemon already stopped.")
        _kill_stale_daemon()
        return

    if _wait_for_daemon(expected_running=False, timeout=DAEMON_STOP_TIMEOUT):
        success_message("OmniMemory daemon stopped.")
    else:
        warning_message("Daemon did not stop gracefully. Force killing...")
        _kill_stale_daemon()
        if not is_daemon_running():
            success_message("OmniMemory daemon stopped.")
        else:
            error_message("Failed to stop daemon. Check logs.")


@daemon_app.command("status")
def daemon_status():
    """Show daemon status."""
    if not is_daemon_running():
        warning_message("OmniMemory daemon is not running.")
        info_message("Start it with `omnimemory daemon start`.")
        return
    try:
        status = call_daemon("status", {})
    except (DaemonNotRunningError, DaemonResponseError):
        warning_message("Unable to contact daemon.")
        return

    table = Table(title="[bold white]Daemon Status[/]", box=box.ROUNDED)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("SDK Initialized", "Yes")
    table.add_row(
        "Memory Manager Initialized",
        "Yes" if status.get("memory_manager_initialized") else "Not yet",
    )
    pool_stats = status.get("connection_pool") or {}
    table.add_row(
        "Pool Active/Max",
        f"{pool_stats.get('active_handlers', 0)}/{pool_stats.get('max_connections', '?')}",
    )
    table.add_row(
        "Pool Available",
        str(pool_stats.get("available_handlers", 0)),
    )
    table.add_row(
        "Pool Created",
        str(pool_stats.get("created_handlers", 0)),
    )
    table.add_row("Timestamp", status.get("timestamp", "N/A"))
    console.print(table)
    success_message("Daemon is running.")


@app.command()
def info():
    """Show OmniMemory welcome screen and comprehensive feature guide."""
    console.print()
    console.print(OMNIMEMORY_BANNER)
    console.print()

    welcome_panel = Panel(
        Align.center(
            "[bold white]Welcome to OmniMemory![/]\n\n"
            "[dim]Self-Evolving Composite Memory Synthesis Architecture (SECMSA)[/]\n"
            "[dim cyan]Dual-Agent Construction • Persistent Memory Storage • Self-Evolution [/]"
        ),
        border_style="bright_cyan",
        box=box.DOUBLE,
    )
    console.print(welcome_panel)
    console.print()

    quickstart = Table(
        box=box.ROUNDED,
        border_style="green",
        show_header=True,
        header_style="bold green",
        title="[bold white]⚡ Quick Start Guide[/]",
    )
    quickstart.add_column("Command", style="cyan bold", no_wrap=True)
    quickstart.add_column("Description", style="white")

    quickstart.add_row(
        "[bold]omnimemory memory add[/]",
        "Create memory from user messages (full dual-agent pipeline)",
    )
    quickstart.add_row(
        "[bold]omnimemory memory query[/]",
        "Query memories with intelligent composite scoring",
    )
    quickstart.add_row(
        "[bold]omnimemory memory get[/]", "Retrieve a specific memory by ID"
    )
    quickstart.add_row(
        "[bold]omnimemory memory delete[/]",
        "Delete a memory (marks as deleted with status)",
    )
    quickstart.add_row(
        "[bold]omnimemory agent add-memory[/]",
        "Fast agent memory storage (<10s, no conflict resolution)",
    )
    quickstart.add_row(
        "[bold]omnimemory agent summarize[/]",
        "Generate conversation summary (sync or async with webhook)",
    )
    quickstart.add_row(
        "[bold]omnimemory daemon start[/]",
        "Start persistent background daemon for faster operations",
    )
    quickstart.add_row(
        "[bold]omnimemory daemon status[/]",
        "Check daemon status and connection pool stats",
    )
    quickstart.add_row("[bold]omnimemory daemon stop[/]", "Stop the background daemon")
    quickstart.add_row(
        "[bold]omnimemory health[/]", "Comprehensive system health check"
    )

    console.print(quickstart)
    console.print()

    features = Tree(
        "✨ [bold magenta]Core Capabilities[/]",
        guide_style="dim",
    )

    memory_branch = features.add("[bold cyan]💾 Memory Operations[/]")
    memory_branch.add(
        "[dim]• Standard Memory Creation: Dual-agent parallel construction (Episodic + Summarizer)[/]"
    )
    memory_branch.add(
        "[dim]• Agent Memory Creation: Fast single-agent path for direct agent message storage[/]"
    )
    memory_branch.add(
        "[dim]• Intelligent Query: Multi-dimensional ranking (relevance × (1 + recency + importance))[/]"
    )
    memory_branch.add(
        "[dim]• Conflict Resolution: AI-powered status-based evolution (UPDATE/DELETE/SKIP/CREATE)[/]"
    )
    memory_branch.add(
        "[dim]• Status Tracking: Simple status-based updates (active/updated/deleted) with reasons[/]"
    )
    memory_branch.add(
        "[dim]• Flexible Filtering: Query by app_id, user_id, session_id with similarity thresholds[/]"
    )

    agent_branch = features.add("[bold magenta]🤖 Agent Operations[/]")
    agent_branch.add(
        "[dim]• Conversation Summary: Fast text-only (<10s) or full structured with metadata[/]"
    )
    agent_branch.add(
        "[dim]• Webhook Support: Async delivery with retry logic (exponential backoff)[/]"
    )
    agent_branch.add(
        "[dim]• Flexible Input: Accept messages as string or structured array[/]"
    )
    agent_branch.add(
        "[dim]• Direct Storage: Agent memories bypass conflict resolution for speed[/]"
    )

    storage_branch = features.add("[bold yellow]🔍 Vector Database & Storage[/]")
    storage_branch.add("[dim]• Vector Database: Qdrant with async client support[/]")
    storage_branch.add(
        "[dim]• Connection Pooling: Configurable pool size with retry/backoff[/]"
    )
    storage_branch.add(
        "[dim]• Automatic Embeddings: Voyage AI embeddings with configurable dimensions[/]"
    )
    storage_branch.add(
        "[dim]• Semantic Search: Approximate nearest neighbor with similarity thresholds[/]"
    )
    storage_branch.add(
        "[dim]• Metadata Storage: Status, timestamps, and reasons stored alongside vectors[/]"
    )

    system_branch = features.add("[bold green]⚙️  System Features[/]")
    system_branch.add(
        "[dim]• Asynchronous Processing: Fully async with asyncio (no blocking operations)[/]"
    )
    system_branch.add(
        "[dim]• Background Tasks: Non-blocking memory creation and webhook delivery[/]"
    )
    system_branch.add(
        "[dim]• Connection Pooling: Efficient resource management with warm-up support[/]"
    )
    system_branch.add(
        "[dim]• Prometheus Metrics: In-memory metrics with HTTP endpoint[/]"
    )
    system_branch.add(
        "[dim]• Comprehensive Logging: Structured logging with context and error tracking[/]"
    )
    system_branch.add(
        "[dim]• Error Handling: Graceful degradation with retry logic and timeouts[/]"
    )

    console.print(features)
    console.print()

    architecture_panel = Panel(
        Align.center(
            "[bold cyan]🏗️  Architecture Highlights[/]\n\n"
            "[dim]• Status-Based Evolution: Simple status + reason tracking (no complex versioning)[/]\n"
            "[dim]• Three Memory Paths: Standard (full), Agent (fast), Summary (standalone)[/]\n"
            "[dim]• Fully Asynchronous: All I/O operations are non-blocking with asyncio[/]\n"
            "[dim]• Connection Pooling: Configurable pool with automatic retry and backoff[/]\n"
            "[dim]• In-Memory Metrics: Prometheus client library (no external dependencies)[/]\n"
            "[dim]• Production Ready: Comprehensive error handling, logging, and observability[/]"
        ),
        title="[bold white]System Architecture[/]",
        border_style="bright_cyan",
        box=box.DOUBLE,
    )
    console.print(architecture_panel)
    console.print()

    help_panel = Panel(
        "[cyan]📚 Documentation:[/] See API_SPECIFICATION.md and ARCHITECTURE_NAME_PROPOSAL.md\n"
        "[cyan]💬 Support:[/] Check GitHub issues\n"
        "[cyan]📖 Full CLI Help:[/] [bold]omnimemory --help[/]\n"
        "[cyan]🔍 Command Help:[/] [bold]omnimemory <command> --help[/]",
        title="[bold white]Need Help?[/]",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print(help_panel)
    console.print()

    info_message("Ready to manage memories with OmniMemory!")
    console.print()


@app.command()
def health():
    """Show comprehensive OmniMemory system health."""
    console.print()
    console.print(OMNIMEMORY_BANNER)
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running health check...", total=None)
        try:
            status = daemon_request("status", {})
            sdk_healthy = status.get("sdk_initialized", False)
            memory_manager_healthy = status.get("memory_manager_initialized", False)
            connectivity_ok = True
            progress.update(task, completed=True)

        except Exception as e:
            error_message(f"Failed to check health: {e}")
            raise typer.Exit(1)

    console.print()

    if sdk_healthy and memory_manager_healthy and connectivity_ok:
        overall_status = "[bold green]● HEALTHY[/]"
        status_color = "green"
        status_desc = "All systems operational"
    else:
        overall_status = "[bold red]● UNHEALTHY[/]"
        status_color = "red"
        status_desc = "Some components unavailable"

    header = Panel(
        Align.center(
            f"{overall_status}\n"
            f"[dim]{status_desc}[/]\n\n"
            f"[dim]SDK Version:[/] [cyan]{get_version()}[/]"
        ),
        border_style=status_color,
        box=box.DOUBLE,
        title="[bold white]❤️  System Health[/]",
    )
    console.print(header)
    console.print()

    cards = [
        create_metric_card("SDK", "✓ Initialized" if sdk_healthy else "✗ Failed", "🧠"),
        create_metric_card(
            "Memory Manager", "✓ Ready" if memory_manager_healthy else "✗ Failed", "💾"
        ),
        create_metric_card(
            "Connectivity", "✓ OK" if connectivity_ok else "✗ Failed", "🔌"
        ),
    ]

    console.print(Columns(cards, equal=True, expand=True))
    console.print()

    status_table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    status_table.add_column("Component", style="white")
    status_table.add_column("Status", justify="center")
    status_table.add_column("Details", style="dim")

    status_table.add_row(
        "OmniMemorySDK",
        "[green]✓ Healthy[/]" if sdk_healthy else "[red]✗ Unhealthy[/]",
        "Main SDK instance",
    )
    status_table.add_row(
        "MemoryManager",
        "[green]✓ Ready[/]" if memory_manager_healthy else "[red]✗ Failed[/]",
        "Memory operations handler",
    )
    status_table.add_row(
        "Connection Pool",
        "[green]✓ Active[/]" if connectivity_ok else "[red]✗ Inactive[/]",
        "Vector DB connection pool",
    )

    status_panel = Panel(
        status_table,
        title="[bold white]🔌 Component Status[/]",
        border_style="bright_cyan",
        box=box.ROUNDED,
    )
    console.print(status_panel)
    console.print()

    success_message("Health check completed successfully")
    console.print()


@memory_app.command("add")
def memory_add(
    app_id: str = typer.Option(..., "--app-id", "-a", help="Application ID"),
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID"),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID"
    ),
    messages_file: Optional[str] = typer.Option(
        None, "--messages-file", "-f", help="Path to JSON file with messages"
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="Single message content (role:content:timestamp)"
    ),
):
    """
    Add a new memory from user messages.

    Messages can be provided via:
    - JSON file with messages array (--messages-file)
    - Single message (--message) in format "role:content:timestamp"

    Maximum {DEFAULT_MAX_MESSAGES} messages allowed.
    """
    console.print()

    try:
        if messages_file:
            try:
                with open(messages_file, "r") as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
            except FileNotFoundError:
                error_message(f"File not found: {messages_file}")
                raise typer.Exit(1)
            except json.JSONDecodeError:
                error_message(f"Invalid JSON in {messages_file}")
                raise typer.Exit(1)
        elif message:
            parts = message.split(":", 2)
            if len(parts) != 3:
                error_message("Message format must be: role:content:timestamp")
                raise typer.Exit(1)
            messages = [{"role": parts[0], "content": parts[1], "timestamp": parts[2]}]
        else:
            error_message("Either --messages-file or --message must be provided")
            raise typer.Exit(1)

        if len(messages) < 1:
            error_message(
                f"At least 1 message is required. You provided {len(messages)} messages."
            )
            raise typer.Exit(1)
        if len(messages) > DEFAULT_MAX_MESSAGES:
            error_message(
                f"Maximum {DEFAULT_MAX_MESSAGES} messages allowed. "
                f"You provided {len(messages)} messages. "
                f"Please reduce to {DEFAULT_MAX_MESSAGES} or fewer."
            )
            raise typer.Exit(1)

        request = AddUserMessageRequest(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            messages=messages,
        )

        info_message(
            f"Adding memory: app_id={app_id}, user_id={user_id}, message_count={len(messages)}"
        )
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Adding memory...", total=None)
            try:
                user_message = request.to_user_messages()
                result = daemon_request(
                    "add_memory", {"user_message": user_message.model_dump()}
                )
                progress.update(task, completed=True)
            except ValueError as e:
                error_message(f"Validation failed: {e}")
                raise typer.Exit(1)
            except Exception as e:
                error_message(f"Failed to add memory: {e}")
                raise typer.Exit(1)

        console.print()

        result_panel = Panel(
            f"[bold cyan]{result.get('task_id', 'N/A')}[/]",
            title="[bold green]✓ Memory Added Successfully[/]",
            subtitle="[dim]Task ID[/]",
            border_style="green",
            box=box.DOUBLE,
        )
        console.print(result_panel)
        console.print()

        console.print("[dim]Next steps:[/]")
        console.print(f"  [cyan]• Memory is being processed asynchronously[/]")
        console.print(
            f"  [cyan]• Use[/] omnimemory memory query [dim]to search for memories[/]"
        )
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        error_message(f"Error: {e}")
        raise typer.Exit(1)


@memory_app.command("query")
def memory_query(
    app_id: str = typer.Option(..., "--app-id", "-a", help="Application ID"),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Natural language query"
    ),
    user_id: Optional[str] = typer.Option(
        None, "--user-id", "-u", help="User ID filter"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID filter"
    ),
    n_results: int = typer.Option(
        10, "--limit", "-n", help="Maximum number of results", min=1, max=100
    ),
    similarity_threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t", help="Similarity threshold (0.0-1.0)"
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    query_words: List[str] = typer.Argument(
        None,
        help=(
            "Optional query text without --query flag. "
            "Useful when you forget to wrap the query in quotes."
        ),
    ),
):
    """Query memories with intelligent multi-dimensional ranking."""
    console.print()

    if query and query_words:
        raise typer.BadParameter(
            "Provide the query either with --query or as a positional argument, not both."
        )

    final_query = (query or "").strip()
    if not final_query and query_words:
        final_query = " ".join(query_words).strip()

    if not final_query:
        final_query = typer.prompt("Enter natural language query").strip()

    if not final_query:
        error_message("Query text is required.")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Querying memories...", total=None)
        try:
            results = daemon_request(
                "query_memory",
                {
                    "app_id": app_id,
                    "query": final_query,
                    "user_id": user_id,
                    "session_id": session_id,
                    "n_results": n_results,
                    "similarity_threshold": similarity_threshold,
                },
            )
            progress.update(task, completed=True)
        except Exception as e:
            error_message(f"Failed to query memories: {e}")
            raise typer.Exit(1)

    console.print()

    if output_json:
        console.print(json.dumps(results, indent=2, default=str))
        return

    if not results:
        warning_message("No memories found matching your query")
        console.print()
        return

    header = Panel(
        Align.center(
            f"[bold cyan]{len(results)}[/] result{'s' if len(results) > 1 else ''} found"
        ),
        title="[bold white]🔍 Query Results[/]",
        border_style="cyan",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    table = Table(
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Rank", justify="right", style="yellow")
    table.add_column("Memory ID", style="cyan", no_wrap=True)
    table.add_column("Composite Score", justify="right", style="green")
    table.add_column("Similarity", justify="right", style="cyan")
    table.add_column("Preview", style="white")

    for idx, result in enumerate(results, 1):
        memory_id = result.get("id", "N/A")
        composite = result.get("composite_score", 0)
        similarity = result.get("similarity_score", 0)
        document = result.get("document", "")
        preview = document[:50] + "..." if len(document) > 50 else document

        table.add_row(
            str(idx),
            memory_id[:20] + "..." if len(memory_id) > 20 else memory_id,
            f"{composite:.3f}",
            f"{similarity:.3f}",
            preview,
        )

    console.print(table)
    console.print()
    success_message(f"Query completed: {len(results)} results")
    console.print()


@memory_app.command("get")
def memory_get(
    memory_id: str = typer.Option(..., "--memory-id", "-m", help="Memory ID"),
    app_id: str = typer.Option(..., "--app-id", "-a", help="Application ID"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get a single memory by its ID."""
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Fetching memory {memory_id[:16]}...", total=None
        )
        try:
            memory = daemon_request(
                "get_memory", {"memory_id": memory_id, "app_id": app_id}
            )
            progress.update(task, completed=True)
        except Exception as e:
            error_message(f"Failed to get memory: {e}")
            raise typer.Exit(1)

    console.print()

    if memory is None:
        error_message(f"Memory not found: {memory_id}")
        raise typer.Exit(1)

    if output_json:
        console.print(json.dumps(memory, indent=2, default=str))
        return

    header = Panel(
        Align.center(
            f"[bold green]💾 Memory Details[/]\n[dim]ID:[/] [cyan]{memory_id}[/]"
        ),
        border_style="green",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    if memory.get("document"):
        doc_panel = Panel(
            memory["document"],
            title="[bold white]📄 Document[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
        console.print(doc_panel)
        console.print()

    if memory.get("metadata"):
        metadata_json = json.dumps(memory["metadata"], indent=2)
        syntax = Syntax(metadata_json, "json", theme="monokai", line_numbers=True)
        metadata_panel = Panel(
            syntax,
            title="[bold white]📋 Metadata[/]",
            border_style="magenta",
            box=box.ROUNDED,
        )
        console.print(metadata_panel)
        console.print()

    success_message("Memory retrieved successfully")
    console.print()


@memory_app.command("evolution")
def memory_evolution(
    memory_id: str = typer.Option(..., "--memory-id", "-m", help="Starting memory ID"),
    app_id: str = typer.Option(..., "--app-id", "-a", help="Application ID"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    graph: bool = typer.Option(
        False, "--graph", "-g", help="Generate graph visualization"
    ),
    graph_format: str = typer.Option(
        "mermaid", "--format", "-f", help="Graph format: mermaid, dot, or html"
    ),
    output_file: str = typer.Option(
        None, "--output", "-o", help="Output file path (for graph formats)"
    ),
):
    """Traverse the memory evolution chain using singly linked list algorithm."""
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Traversing evolution chain from {memory_id[:16]}...", total=None
        )
        try:
            chain = daemon_request(
                "traverse_memory_evolution_chain",
                {"memory_id": memory_id, "app_id": app_id},
            )
            progress.update(task, completed=True)
        except Exception as e:
            error_message(f"Failed to traverse evolution chain: {e}")
            raise typer.Exit(1)

    console.print()

    if not chain:
        warning_message(f"No evolution chain found starting from {memory_id}")
        console.print()
        return

    if graph:
        try:
            graph_output = daemon_request(
                "generate_evolution_graph",
                {"chain": chain, "format": graph_format},
            )

            if output_file:
                output_path = Path(output_file)
                output_path.write_text(graph_output, encoding="utf-8")
                success_message(f"Graph saved to: {output_path.absolute()}")
                console.print()

                if graph_format == "html":
                    console.print(
                        f"[dim]Open in browser:[/] [cyan]file://{output_path.absolute()}[/]"
                    )
                    console.print()
            else:
                if graph_format == "html":
                    warning_message(
                        "HTML format should be saved to a file. Use --output to specify file path."
                    )
                    console.print()
                else:
                    console.print()
                    graph_panel = Panel(
                        graph_output,
                        title=f"[bold cyan]📊 Evolution Graph ({graph_format.upper()})[/]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    )
                    console.print(graph_panel)
                    console.print()

                    if graph_format == "mermaid":
                        info_panel = Panel(
                            "[bold]How to view this graph:[/]\n\n"
                            "1. Copy the Mermaid code above\n"
                            "2. Go to: [cyan]https://mermaid.live[/]\n"
                            "3. Paste the code in the editor\n"
                            "4. The graph will render automatically\n\n"
                            "Or use the HTML format with --format html --output file.html",
                            title="[bold yellow]📖 Viewing Instructions[/]",
                            border_style="yellow",
                            box=box.ROUNDED,
                        )
                    elif graph_format == "dot":
                        info_panel = Panel(
                            "[bold]How to view this graph:[/]\n\n"
                            "1. Copy the DOT code above\n"
                            "2. Go to: [cyan]https://dreampuf.github.io/GraphvizOnline/[/]\n"
                            "3. Paste the code in the editor\n"
                            "4. Click 'Generate' to render\n\n"
                            "Or install Graphviz locally:\n"
                            "  - macOS: [cyan]brew install graphviz[/]\n"
                            "  - Linux: [cyan]sudo apt-get install graphviz[/]\n"
                            "  - Then: [cyan]dot -Tpng file.dot -o output.png[/]",
                            title="[bold yellow]📖 Viewing Instructions[/]",
                            border_style="yellow",
                            box=box.ROUNDED,
                        )
                    else:
                        info_panel = None

                    if info_panel:
                        console.print(info_panel)
                        console.print()

            if not output_file and graph_format != "html":
                return
        except Exception as e:
            error_message(f"Failed to generate graph: {e}")
            console.print()

    if output_json:
        console.print(json.dumps(chain, indent=2, default=str))
        return

    header = Panel(
        Align.center(
            f"[bold cyan]🔗 Memory Evolution Chain[/]\n"
            f"[dim]Starting from:[/] [cyan]{memory_id}[/]\n"
            f"[dim]Chain length:[/] [green]{len(chain)}[/] memories"
        ),
        border_style="cyan",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    table = Table(
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Order", justify="right", style="yellow")
    table.add_column("Memory ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Next ID", style="magenta", no_wrap=True)
    table.add_column("Created", style="dim")
    table.add_column("Preview", style="white")

    for idx, memory in enumerate(chain, 1):
        mem_id = memory.get("memory_id", "N/A")
        metadata = memory.get("metadata", {})
        status = metadata.get("status", "unknown")
        next_id = metadata.get("next_id")
        created_at = metadata.get("created_at", "N/A")
        document = memory.get("document", "")
        preview = document[:40] + "..." if len(document) > 40 else document

        next_display = (
            (next_id[:20] + "..." if len(next_id) > 20 else next_id)
            if next_id
            else "[dim]None (end)[/]"
        )
        created_display = (
            created_at.split("T")[0] if "T" in str(created_at) else str(created_at)
        )

        table.add_row(
            str(idx),
            mem_id[:24] + "..." if len(mem_id) > 24 else mem_id,
            status,
            next_display,
            created_display,
            preview,
        )

    console.print(table)
    console.print()
    success_message(f"Evolution chain: {len(chain)} memories traversed")
    console.print()


@memory_app.command("delete")
def memory_delete(
    memory_id: str = typer.Option(..., "--memory-id", "-m", help="Memory ID"),
    app_id: str = typer.Option(..., "--app-id", "-a", help="Application ID"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a memory from the collection."""
    console.print()

    if not confirm:
        warning_panel = Panel(
            Align.center(
                f"[bold yellow]⚠️  Warning[/]\n\n"
                f"You are about to delete memory:\n"
                f"[cyan]{memory_id}[/]\n\n"
                f"[dim]This action cannot be undone.[/]"
            ),
            border_style="yellow",
            box=box.DOUBLE,
        )
        console.print(warning_panel)
        console.print()

        confirmed = typer.confirm("Are you sure you want to proceed?")
        if not confirmed:
            warning_message("Deletion cancelled")
            raise typer.Exit(0)

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Deleting memory...", total=None)
        try:
            success = daemon_request(
                "delete_memory", {"app_id": app_id, "doc_id": memory_id}
            )
            progress.update(task, completed=True)
        except Exception as e:
            error_message(f"Failed to delete memory: {e}")
            raise typer.Exit(1)

    console.print()

    if success:
        success_message(f"Memory deleted: {memory_id}")
    else:
        error_message(f"Failed to delete memory: {memory_id}")
        raise typer.Exit(1)

    console.print()


@agent_app.command("summarize")
def agent_summarize(
    app_id: str = typer.Option(..., "--app-id", "-a", help="Application ID"),
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID"),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID"
    ),
    messages_file: Optional[str] = typer.Option(
        None, "--messages-file", "-f", help="Path to JSON conversation payload"
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="Single message entry role:content:timestamp"
    ),
    text: Optional[str] = typer.Option(
        None, "--text", "-t", help="Raw conversation text"
    ),
    callback_url: Optional[str] = typer.Option(
        None, "--callback-url", "-c", help="Webhook URL for async delivery"
    ),
    callback_header: List[str] = typer.Option(
        [],
        "--callback-header",
        "-H",
        help="Webhook headers (key=value). Can be provided multiple times.",
    ),
):
    """Generate a conversation summary using the single-agent pipeline."""
    console.print()

    payload_messages = _load_conversation_payload(messages_file, message, text)

    headers_dict: Dict[str, str] = {}
    for header in callback_header:
        if "=" not in header:
            error_message("Callback headers must be in key=value format.")
            raise typer.Exit(1)
        key, value = header.split("=", 1)
        headers_dict[key.strip()] = value.strip()

    summary_request = {
        "app_id": app_id,
        "user_id": user_id,
        "session_id": session_id,
        "messages": payload_messages,
    }

    if callback_url:
        summary_request["callback_url"] = callback_url
        if headers_dict:
            summary_request["callback_headers"] = headers_dict

    info_message("Requesting conversation summary from daemon...")
    result = daemon_request(
        "summarize_conversation", {"summary_request": summary_request}
    )

    if isinstance(result, dict) and result.get("status") == "accepted":
        success_message(
            f"Summary scheduled for callback delivery. Task ID: {result.get('task_id')}"
        )
        return

    summary_text = result.get("summary", "No summary returned.")
    summary_panel = Panel(
        summary_text,
        title="[bold magenta]Conversation Summary[/]",
        border_style="magenta",
        box=box.ROUNDED,
    )
    console.print(summary_panel)

    key_points = result.get("key_points")
    if key_points:
        console.print(
            Panel(
                key_points,
                title="[bold cyan]Key Points[/]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    tags = result.get("tags") or []
    keywords = result.get("keywords") or []
    semantic_queries = result.get("semantic_queries") or []

    if any([tags, keywords, semantic_queries]):
        retrieval_table = Table(
            title="Retrieval Suggestions",
            box=box.ROUNDED,
            border_style="bright_magenta",
        )
        retrieval_table.add_column("Type", style="bold white")
        retrieval_table.add_column("Values", style="dim")
        retrieval_table.add_row("Tags", ", ".join(tags) or "—")
        retrieval_table.add_row("Keywords", ", ".join(keywords) or "—")
        retrieval_table.add_row("Semantic Queries", ", ".join(semantic_queries) or "—")
        console.print(retrieval_table)

    metadata = result.get("metadata") or {}
    if metadata:
        meta_table = Table(
            title="Metadata",
            box=box.ROUNDED,
            border_style="bright_cyan",
        )
        meta_table.add_column("Field", style="bold white")
        meta_table.add_column("Value", style="dim")
        for key, value in metadata.items():
            meta_table.add_row(key, json.dumps(value, ensure_ascii=False))
        console.print(meta_table)

    generated_at = result.get("generated_at")
    if generated_at:
        info_message(f"Generated at: {generated_at}")

    success_message("Conversation summary completed.")


@agent_app.command("add-memory")
def agent_add_memory(
    app_id: str = typer.Option(..., "--app-id", "-a", help="Application ID"),
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID"),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID"
    ),
    messages_file: Optional[str] = typer.Option(
        None,
        "--messages-file",
        "-f",
        help="Path to JSON file with messages (string or list)",
    ),
    messages: Optional[str] = typer.Option(
        None, "--messages", "-m", help="Raw messages text"
    ),
):
    """Create memory from agent messages (async)."""
    console.print()

    payload_messages = _load_conversation_payload(messages_file, None, messages)
    if not payload_messages:
        error_message("Provide messages via --messages-file or --messages.")
        raise typer.Exit(1)

    agent_request = {
        "app_id": app_id,
        "user_id": user_id,
        "session_id": session_id,
        "messages": payload_messages,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Submitting agent memory task...", total=None)
        try:
            result = daemon_request(
                "add_agent_memory", {"agent_request": agent_request}
            )
            progress.update(task, completed=True)
        except Exception as e:
            error_message(f"Failed to submit agent memory task: {e}")
            raise typer.Exit(1)

    console.print()

    task_id = result.get("task_id")
    status = result.get("status", "unknown")

    header = Panel(
        Align.center(
            f"[bold green]Agent Memory Task Submitted[/]\n"
            f"[dim]Task ID:[/] [cyan]{task_id}[/]\n"
            f"[dim]Status:[/] [yellow]{status}[/]"
        ),
        title="[bold white]Task Accepted[/]",
        border_style="green",
        box=box.DOUBLE,
    )
    console.print(header)
    console.print()

    info_message(
        "Memory is being processed in the background. "
        "Use the task_id to check status if needed."
    )
    console.print()

    success_message("Agent memory task submitted successfully.")


if __name__ == "__main__":
    app()
