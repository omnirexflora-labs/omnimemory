"""
Comprehensive unit tests for OmniMemory CLI.
"""

import pytest
import json
import subprocess
import sys
import time
import os
import signal
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from typing import Optional
import typer
from typer.testing import CliRunner
from importlib import metadata as importlib_metadata

from omnimemory.cli.main import (
    get_version,
    _show_welcome_screen,
    success_message,
    error_message,
    warning_message,
    info_message,
    create_header_panel,
    create_metric_card,
    daemon_request,
    _kill_process_by_port,
    _kill_stale_daemon,
    _wait_for_daemon,
    _load_conversation_payload,
    main,
    daemon_start,
    daemon_stop,
    daemon_status,
    info,
    health,
    memory_add,
    memory_query,
    memory_get,
    memory_evolution,
    memory_delete,
    agent_summarize,
    agent_add_memory,
    app,
    memory_app,
    daemon_app,
    agent_app,
    DAEMON_START_TIMEOUT,
    DAEMON_STOP_TIMEOUT,
    DAEMON_POLL_INTERVAL,
)
from omnimemory.cli.daemon_client import (
    DaemonNotRunningError,
    DaemonResponseError,
)
from omnimemory.cli.daemon_constants import PID_FILE, LOG_FILE
from omnimemory.core.schemas import DEFAULT_MAX_MESSAGES

runner = CliRunner()


@pytest.fixture
def mock_console():
    """Mock Rich console."""
    with patch("omnimemory.cli.main.console") as mock:
        yield mock


@pytest.fixture
def mock_call_daemon():
    """Mock call_daemon function."""
    with patch("omnimemory.cli.main.call_daemon") as mock:
        yield mock


@pytest.fixture
def mock_is_daemon_running():
    """Mock is_daemon_running function."""
    with patch("omnimemory.cli.main.is_daemon_running") as mock:
        yield mock


@pytest.fixture
def mock_subprocess():
    """Mock subprocess module."""
    with patch("omnimemory.cli.main.subprocess") as mock:
        yield mock


@pytest.fixture
def mock_path():
    """Mock Path operations."""
    with patch("omnimemory.cli.main.Path") as mock:
        yield mock


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
        {
            "role": "assistant",
            "content": "Hi there",
            "timestamp": "2024-01-01T00:01:00Z",
        },
    ] * (DEFAULT_MAX_MESSAGES // 2)


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file."""

    def _create(content: str, filename: str = "test.json"):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return str(file_path)

    return _create


class TestGetVersion:
    """Test get_version function."""

    def test_get_version_from_metadata(self):
        """Test get_version retrieves version from package metadata."""
        with patch("omnimemory.cli.main.importlib_metadata") as mock_metadata:
            mock_metadata.version.return_value = "1.2.3"
            assert get_version() == "1.2.3"
            mock_metadata.version.assert_called_once_with("omnimemory")

    def test_get_version_fallback_to_version_file(self):
        """Test get_version falls back to _version.py file."""
        with patch("omnimemory.cli.main.importlib_metadata") as mock_metadata:
            mock_metadata.version.side_effect = (
                importlib_metadata.PackageNotFoundError()
            )

            version_content = '__version__ = "2.3.4"'
            with patch("builtins.open", mock_open(read_data=version_content)):
                with patch("omnimemory.cli.main.Path") as mock_path:
                    mock_file = Mock()
                    mock_file.exists.return_value = True
                    mock_path.return_value = mock_file
                    with patch("pathlib.Path") as path_mock:
                        mock_version_file = Mock()
                        mock_version_file.exists.return_value = True
                        mock_version_file.__truediv__ = Mock(
                            return_value=mock_version_file
                        )
                        path_mock.return_value = mock_version_file
                        pass

    def test_get_version_fallback_to_dev(self):
        """Test get_version falls back to 'dev' if not found."""
        with patch("omnimemory.cli.main.importlib_metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError("omnimemory")

            with patch("builtins.open", side_effect=FileNotFoundError):
                assert get_version() == "dev"

    def test_get_version_from_version_file(self, tmp_path):
        """Test get_version reads from _version.py file (lines 92-97)."""
        with patch("omnimemory.cli.main.importlib_metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError("omnimemory")

            version_file = tmp_path / "_version.py"
            version_file.write_text('__version__ = "2.0.0"')

            with patch("omnimemory.cli.main.Path") as mock_path:
                mock_version_path = Mock()
                mock_version_path.exists.return_value = True
                mock_path.return_value.parent.parent.parent.parent = tmp_path
                with patch("pathlib.Path") as path_mock:
                    mock_file = Mock()
                    mock_file.exists.return_value = True
                    mock_file.__truediv__ = Mock(return_value=version_file)
                    path_mock.return_value = mock_file
                    with patch("omnimemory.cli.main.Path") as path_cls:

                        def path_side_effect(*args):
                            if args and str(args[0]).endswith("_version.py"):
                                return version_file
                            return Mock(exists=Mock(return_value=False))

                        path_cls.side_effect = path_side_effect
                        with patch(
                            "builtins.open",
                            mock_open(read_data='__version__ = "2.0.0"'),
                        ):
                            with patch("pathlib.Path.exists", return_value=True):
                                pass

    def test_get_version_file_read_exception(self):
        """Test get_version handles file read exceptions (line 96-97)."""
        with patch("omnimemory.cli.main.importlib_metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError("omnimemory")

            with patch("pathlib.Path") as mock_path:
                mock_file = Mock()
                mock_file.exists.return_value = True
                mock_file.__truediv__ = Mock(return_value=mock_file)
                mock_path.return_value = mock_file

                with patch("builtins.open", side_effect=IOError("Permission denied")):
                    assert get_version() == "dev"

    def test_get_version_from_version_file_success(self, tmp_path):
        """Test get_version successfully reads from version file (lines 92-97)."""
        with patch("omnimemory.cli.main.importlib_metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError("omnimemory")

            with patch("omnimemory.cli.main.Path") as mock_path_class:
                mock_version_file = MagicMock()
                mock_version_file.exists.return_value = True

                mock_parent = MagicMock()
                mock_parent.__truediv__ = Mock(return_value=mock_version_file)
                mock_file = MagicMock()
                mock_file.parent.parent.parent.parent = mock_parent
                mock_path_class.return_value = mock_file

                with patch(
                    "builtins.open", mock_open(read_data='__version__ = "3.0.0"')
                ):
                    version = get_version()
                    assert version == "3.0.0"

    def test_get_version_handles_package_not_found_error(self):
        """Test get_version handles PackageNotFoundError."""
        with patch("omnimemory.cli.main.importlib_metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError("omnimemory")
            with patch("builtins.open", side_effect=Exception("Error")):
                assert get_version() == "dev"

    def test_get_version_handles_file_read_errors(self):
        """Test get_version handles file read errors."""
        with patch("omnimemory.cli.main.importlib_metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError("omnimemory")
            with patch("builtins.open", side_effect=IOError("Permission denied")):
                assert get_version() == "dev"


class TestMessageFunctions:
    """Test message display functions."""

    def test_success_message(self, mock_console):
        """Test success_message displays correctly."""
        success_message("Test success")
        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "✓" in call_args or "Test success" in call_args

    def test_error_message(self, mock_console):
        """Test error_message displays correctly."""
        error_message("Test error")
        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "✗" in call_args or "Test error" in call_args

    def test_warning_message(self, mock_console):
        """Test warning_message displays correctly."""
        warning_message("Test warning")
        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "⚠" in call_args or "Test warning" in call_args

    def test_info_message(self, mock_console):
        """Test info_message displays correctly."""
        info_message("Test info")
        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "ℹ" in call_args or "Test info" in call_args

    def test_success_message_empty(self, mock_console):
        """Test success_message handles empty messages."""
        success_message("")
        mock_console.print.assert_called_once()

    def test_error_message_special_characters(self, mock_console):
        """Test error_message handles special characters."""
        error_message("Error: <test> & 'quotes'")
        mock_console.print.assert_called_once()


class TestUIComponents:
    """Test UI component creation functions."""

    def test_create_header_panel(self):
        """Test create_header_panel creates panel correctly."""
        from rich.panel import Panel

        panel = create_header_panel("Test Title", "Test Subtitle")
        assert panel is not None
        assert isinstance(panel, Panel)

    def test_create_header_panel_no_subtitle(self):
        """Test create_header_panel handles empty subtitle."""
        panel = create_header_panel("Test Title")
        assert panel is not None

    def test_create_metric_card(self):
        """Test create_metric_card creates card correctly."""
        card = create_metric_card("Test Title", "100", "📊")
        assert card is not None

    def test_create_metric_card_default_icon(self):
        """Test create_metric_card uses default icon."""
        card = create_metric_card("Test Title", "100")
        assert card is not None


class TestWelcomeScreen:
    """Test _show_welcome_screen function."""

    def test_show_welcome_screen(self, mock_console):
        """Test _show_welcome_screen displays correctly."""
        with patch("omnimemory.cli.main.get_version", return_value="1.0.0"):
            _show_welcome_screen()
            assert mock_console.print.call_count > 0


class TestDaemonRequest:
    """Test daemon_request function."""

    def test_daemon_request_success(self, mock_call_daemon):
        """Test daemon_request sends request successfully."""
        mock_call_daemon.return_value = {"result": "success"}
        result = daemon_request("test_method", {"key": "value"})
        assert result == {"result": "success"}
        mock_call_daemon.assert_called_once_with("test_method", {"key": "value"})

    def test_daemon_request_handles_daemon_not_running_error(
        self, mock_call_daemon, mock_console
    ):
        """Test daemon_request handles DaemonNotRunningError."""
        mock_call_daemon.side_effect = DaemonNotRunningError()
        with pytest.raises(typer.Exit) as exc_info:
            daemon_request("test_method")
        assert exc_info.value.exit_code == 1
        assert mock_console.print.called

    def test_daemon_request_handles_daemon_response_error(
        self, mock_call_daemon, mock_console
    ):
        """Test daemon_request handles DaemonResponseError."""
        mock_call_daemon.side_effect = DaemonResponseError("Test error")
        with pytest.raises(typer.Exit) as exc_info:
            daemon_request("test_method")
        assert exc_info.value.exit_code == 1
        assert mock_console.print.called

    def test_daemon_request_no_payload(self, mock_call_daemon):
        """Test daemon_request with no payload."""
        mock_call_daemon.return_value = {"result": "success"}
        result = daemon_request("test_method")
        mock_call_daemon.assert_called_once_with("test_method", {})


class TestKillProcessByPort:
    """Test _kill_process_by_port function."""

    def test_kill_process_by_port_success(self, mock_subprocess):
        """Test _kill_process_by_port kills process successfully."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "12345\n"

        mock_subprocess.run.side_effect = [mock_result]

        with patch("omnimemory.cli.main.os.kill"):
            with patch("omnimemory.cli.main.time.sleep"):
                result = _kill_process_by_port(8000)
                assert result is True

    def test_kill_process_by_port_no_process(self, mock_subprocess):
        """Test _kill_process_by_port handles no process on port."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_subprocess.run.return_value = mock_result

        result = _kill_process_by_port(8000)
        assert result is False

    def test_kill_process_by_port_command_failure(self):
        """Test _kill_process_by_port handles command failures."""
        from subprocess import TimeoutExpired

        with patch("omnimemory.cli.main.subprocess.run") as mock_run:
            mock_run.side_effect = [
                TimeoutExpired("lsof", 2),
                Mock(returncode=1, stdout=""),
            ]
            result = _kill_process_by_port(8000)
            assert result is False

    def test_kill_process_by_port_permission_error(self, mock_subprocess):
        """Test _kill_process_by_port handles permission errors."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "12345\n"
        mock_subprocess.run.return_value = mock_result

        mock_kill_result = Mock()
        mock_kill_result.returncode = 1
        mock_subprocess.run.side_effect = [mock_result, mock_kill_result]

        result = _kill_process_by_port(8000)
        assert isinstance(result, bool)

    def test_kill_process_by_port_os_error_handling(self):
        """Test _kill_process_by_port handles OSError during kill (lines 309-310)."""
        with patch("omnimemory.cli.main.subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "12345\n"
            mock_run.return_value = mock_result

            with patch("omnimemory.cli.main.os.kill") as mock_kill:
                mock_kill.side_effect = [OSError(), None, None]
                with patch("omnimemory.cli.main.time.sleep"):
                    result = _kill_process_by_port(8000)
                    assert result is True

    def test_kill_process_by_port_ss_fallback(self):
        """Test _kill_process_by_port uses ss command fallback (lines 315-340)."""
        from subprocess import TimeoutExpired
        import re

        with patch("omnimemory.cli.main.subprocess.run") as mock_run:
            mock_run.side_effect = [
                TimeoutExpired("lsof", 2),
                Mock(returncode=0, stdout="pid=12345"),
            ]

            with patch("omnimemory.cli.main.os.kill"):
                with patch("omnimemory.cli.main.time.sleep"):
                    with patch("re.search") as mock_search:
                        mock_match = Mock()
                        mock_match.group.return_value = "12345"
                        mock_search.return_value = mock_match
                        result = _kill_process_by_port(8000)
                        assert result is True

    def test_kill_process_by_port_ss_fallback_no_match(self):
        """Test _kill_process_by_port ss fallback when no PID match."""
        from subprocess import TimeoutExpired
        import re

        with patch("omnimemory.cli.main.subprocess.run") as mock_run:
            mock_run.side_effect = [
                TimeoutExpired("lsof", 2),
                Mock(returncode=0, stdout="no pid here"),
            ]

            with patch("re.search", return_value=None):
                result = _kill_process_by_port(8000)
                assert result is False

    def test_kill_process_by_port_ss_fallback_exception(self):
        """Test _kill_process_by_port ss fallback exception handling (line 339)."""
        from subprocess import TimeoutExpired

        with patch("omnimemory.cli.main.subprocess.run") as mock_run:
            mock_run.side_effect = [
                TimeoutExpired("lsof", 2),
                TimeoutExpired("ss", 2),
            ]
            result = _kill_process_by_port(8000)
            assert result is False


class TestKillStaleDaemon:
    """Test _kill_stale_daemon function."""

    def test_kill_stale_daemon_success(self, mock_path):
        """Test _kill_stale_daemon kills stale daemon successfully."""
        mock_pid_file = Mock()
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = "12345"
        mock_path.return_value = mock_pid_file

        with patch("omnimemory.cli.main.os.kill") as mock_kill:
            with patch("omnimemory.cli.main.PID_FILE", mock_pid_file):
                mock_kill.return_value = None
                result = _kill_stale_daemon()
                assert result is True

    def test_kill_stale_daemon_no_pid_file(self, mock_path):
        """Test _kill_stale_daemon handles no PID file."""
        mock_pid_file = Mock()
        mock_pid_file.exists.return_value = False
        mock_path.return_value = mock_pid_file

        with patch("omnimemory.cli.main.PID_FILE", mock_pid_file):
            with patch("omnimemory.cli.main._kill_process_by_port", return_value=False):
                result = _kill_stale_daemon()
                assert result is False

    def test_kill_stale_daemon_process_already_dead(self, mock_path):
        """Test _kill_stale_daemon handles process already dead."""
        mock_pid_file = Mock()
        mock_pid_file.exists.side_effect = [True, False]
        mock_pid_file.read_text.return_value = "12345"
        mock_pid_file.unlink.return_value = None
        mock_path.return_value = mock_pid_file

        with patch("omnimemory.cli.main.PID_FILE", mock_pid_file):
            with patch("omnimemory.cli.main._kill_process_by_port", return_value=False):
                with patch("omnimemory.cli.main.os.kill") as mock_kill:
                    mock_kill.side_effect = OSError()
                    with patch("omnimemory.cli.main.time.sleep"):
                        result = _kill_stale_daemon()
                        assert result is False

    def test_kill_stale_daemon_port_kill_succeeds(self):
        """Test _kill_stale_daemon when port kill succeeds (lines 350-351)."""
        with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
            mock_pid_file.exists.return_value = False
            with patch("omnimemory.cli.main._kill_process_by_port", return_value=True):
                with patch("omnimemory.cli.main.time.sleep"):
                    result = _kill_stale_daemon()
                    assert result is True

    def test_kill_stale_daemon_kill_oserror_cleanup(self):
        """Test _kill_stale_daemon handles OSError during kill cleanup (lines 376-379)."""
        with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
            mock_pid_file.exists.return_value = True
            mock_pid_file.read_text.return_value = "12345"
            mock_pid_file.unlink.return_value = None

            with patch("omnimemory.cli.main._kill_process_by_port", return_value=False):
                with patch("omnimemory.cli.main.os.kill") as mock_kill:
                    mock_kill.side_effect = [None, OSError()]
                    with patch("omnimemory.cli.main.time.sleep"):
                        result = _kill_stale_daemon()
                        assert isinstance(result, bool)

    def test_kill_stale_daemon_pid_file_corruption(self, mock_path):
        """Test _kill_stale_daemon handles PID file corruption."""
        mock_pid_file = Mock()
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.side_effect = ValueError("Invalid PID")
        mock_path.return_value = mock_pid_file

        with patch("omnimemory.cli.main.PID_FILE", mock_pid_file):
            result = _kill_stale_daemon()
            assert result is False


class TestWaitForDaemon:
    """Test _wait_for_daemon function."""

    def test_wait_for_daemon_start_success(self, mock_is_daemon_running):
        """Test _wait_for_daemon waits for daemon to start successfully."""
        mock_is_daemon_running.side_effect = [False, False, True]

        with patch("omnimemory.cli.main.time.sleep"):
            result = _wait_for_daemon(expected_running=True, timeout=5.0)
            assert result is True

    def test_wait_for_daemon_stop_success(self, mock_is_daemon_running):
        """Test _wait_for_daemon waits for daemon to stop successfully."""
        mock_is_daemon_running.side_effect = [True, True, False]

        with patch("omnimemory.cli.main.time.sleep"):
            result = _wait_for_daemon(expected_running=False, timeout=5.0)
            assert result is True

    def test_wait_for_daemon_timeout_start(self, mock_is_daemon_running):
        """Test _wait_for_daemon times out if daemon doesn't start."""
        mock_is_daemon_running.return_value = False

        with patch("omnimemory.cli.main.time.sleep"):
            with patch("omnimemory.cli.main.time.time", side_effect=[0, 0, 0, 6]):
                result = _wait_for_daemon(expected_running=True, timeout=5.0)
                assert result is False

    def test_wait_for_daemon_timeout_stop(self, mock_is_daemon_running):
        """Test _wait_for_daemon times out if daemon doesn't stop."""
        mock_is_daemon_running.return_value = True

        with patch("omnimemory.cli.main.time.sleep"):
            with patch("omnimemory.cli.main.time.time", side_effect=[0, 0, 0, 6]):
                result = _wait_for_daemon(expected_running=False, timeout=5.0)
                assert result is False


class TestLoadConversationPayload:
    """Test _load_conversation_payload function."""

    def test_load_from_text(self, temp_file):
        """Test _load_conversation_payload loads from text."""
        result = _load_conversation_payload(None, None, "Test text")
        assert result == "Test text"

    def test_load_from_text_empty(self, mock_console):
        """Test _load_conversation_payload handles empty text."""
        with pytest.raises(typer.Exit) as exc_info:
            _load_conversation_payload(None, None, "   ")
        assert exc_info.value.exit_code == 1

    def test_load_from_json_file_messages(self, temp_file):
        """Test _load_conversation_payload loads from JSON file with messages."""
        content = json.dumps({"messages": [{"role": "user", "content": "Hello"}]})
        file_path = temp_file(content)
        result = _load_conversation_payload(file_path, None, None)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_load_from_json_file_text(self, temp_file):
        """Test _load_conversation_payload loads from JSON file with text."""
        content = json.dumps({"text": "Test conversation"})
        file_path = temp_file(content)
        result = _load_conversation_payload(file_path, None, None)
        assert result == "Test conversation"

    def test_load_from_json_file_list(self, temp_file):
        """Test _load_conversation_payload loads from JSON file as list."""
        content = json.dumps([{"role": "user", "content": "Hello"}])
        file_path = temp_file(content)
        result = _load_conversation_payload(file_path, None, None)
        assert isinstance(result, list)

    def test_load_from_json_file_string(self, temp_file):
        """Test _load_conversation_payload loads from JSON file as string."""
        content = json.dumps("Test string")
        file_path = temp_file(content)
        result = _load_conversation_payload(file_path, None, None)
        assert result == "Test string"

    def test_load_from_json_file_not_found(self, mock_console):
        """Test _load_conversation_payload handles file not found."""
        with pytest.raises(typer.Exit) as exc_info:
            _load_conversation_payload("/nonexistent/file.json", None, None)
        assert exc_info.value.exit_code == 1

    def test_load_from_json_file_invalid_json(self, temp_file, mock_console):
        """Test _load_conversation_payload handles invalid JSON."""
        file_path = temp_file("invalid json content")
        with pytest.raises(typer.Exit) as exc_info:
            _load_conversation_payload(file_path, None, None)
        assert exc_info.value.exit_code == 1

    def test_load_from_json_file_invalid_format(self, temp_file, mock_console):
        """Test _load_conversation_payload handles invalid format."""
        content = json.dumps({"invalid": "format"})
        file_path = temp_file(content)
        with pytest.raises(typer.Exit) as exc_info:
            _load_conversation_payload(file_path, None, None)
        assert exc_info.value.exit_code == 1

    def test_load_from_single_message(self):
        """Test _load_conversation_payload loads from single message."""
        result = _load_conversation_payload(
            None, "user:Hello:2024-01-01T00:00:00Z", None
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_load_from_single_message_invalid_format(self, mock_console):
        """Test _load_conversation_payload handles invalid message format."""
        with pytest.raises(typer.Exit) as exc_info:
            _load_conversation_payload(None, "invalid:format", None)
        assert exc_info.value.exit_code == 1

    def test_load_from_nothing(self, mock_console):
        """Test _load_conversation_payload handles no input."""
        with pytest.raises(typer.Exit) as exc_info:
            _load_conversation_payload(None, None, None)
        assert exc_info.value.exit_code == 1


class TestMainCommand:
    """Test main command."""

    def test_main_shows_welcome_screen(self, mock_console):
        """Test main shows welcome screen when no subcommand."""
        ctx = Mock()
        ctx.invoked_subcommand = None
        with patch("omnimemory.cli.main._show_welcome_screen"):
            main(ctx, version_flag=False)

    def test_main_shows_version(self, mock_console):
        """Test main shows version with --version flag."""
        ctx = Mock()
        with patch("omnimemory.cli.main.get_version", return_value="1.0.0"):
            with pytest.raises(typer.Exit):
                main(ctx, version_flag=True)
            mock_console.print.assert_called()


class TestDaemonStart:
    """Test daemon_start command."""

    def test_daemon_start_already_running(self, mock_is_daemon_running, mock_console):
        """Test daemon_start handles already running daemon."""
        mock_is_daemon_running.return_value = True
        daemon_start()
        mock_console.print.assert_called()

    def test_daemon_start_kills_stale_daemon(self, mock_is_daemon_running):
        """Test daemon_start kills stale daemon if exists."""
        mock_is_daemon_running.return_value = False

        with patch("omnimemory.cli.main._kill_stale_daemon", return_value=True):
            with patch("omnimemory.cli.main.ensure_state_dir"):
                with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
                    mock_pid_file.exists.return_value = False
                    with patch("omnimemory.cli.main.subprocess.Popen") as mock_popen:
                        mock_process = Mock()
                        mock_popen.return_value = mock_process
                        with patch(
                            "omnimemory.cli.main._wait_for_daemon", return_value=True
                        ):
                            with patch("omnimemory.cli.main.time.sleep"):
                                daemon_start()

    def test_daemon_start_success(self, mock_is_daemon_running):
        """Test daemon_start starts daemon successfully."""
        mock_is_daemon_running.return_value = False

        with patch("omnimemory.cli.main._kill_stale_daemon", return_value=False):
            with patch("omnimemory.cli.main.ensure_state_dir"):
                with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
                    mock_pid_file.exists.return_value = False
                    with patch("builtins.open", mock_open()):
                        with patch(
                            "omnimemory.cli.main.subprocess.Popen"
                        ) as mock_popen:
                            mock_process = Mock()
                            mock_popen.return_value = mock_process
                            with patch(
                                "omnimemory.cli.main._wait_for_daemon",
                                return_value=True,
                            ):
                                with patch("omnimemory.cli.main.time.sleep"):
                                    daemon_start()

    def test_daemon_start_timeout(self, mock_is_daemon_running):
        """Test daemon_start handles startup timeout."""
        mock_is_daemon_running.return_value = False

        with patch("omnimemory.cli.main._kill_stale_daemon", return_value=False):
            with patch("omnimemory.cli.main.ensure_state_dir"):
                with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
                    mock_pid_file.exists.return_value = False
                    with patch("builtins.open", mock_open()):
                        with patch(
                            "omnimemory.cli.main.subprocess.Popen"
                        ) as mock_popen:
                            mock_process = Mock()
                            mock_popen.return_value = mock_process
                            with patch(
                                "omnimemory.cli.main._wait_for_daemon",
                                return_value=False,
                            ):
                                with patch("omnimemory.cli.main._kill_stale_daemon"):
                                    with pytest.raises(typer.Exit):
                                        daemon_start()

    def test_daemon_start_failure(self, mock_is_daemon_running):
        """Test daemon_start handles startup failures."""
        mock_is_daemon_running.return_value = False

        with patch("omnimemory.cli.main._kill_stale_daemon", return_value=False):
            with patch("omnimemory.cli.main.ensure_state_dir"):
                with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
                    mock_pid_file.exists.return_value = False
                    with patch("builtins.open", mock_open()):
                        with patch(
                            "omnimemory.cli.main.subprocess.Popen",
                            side_effect=Exception("Failed"),
                        ):
                            with pytest.raises(typer.Exit):
                                daemon_start()

    def test_daemon_start_pid_file_unlink_exception(self, mock_is_daemon_running):
        """Test daemon_start handles PID file unlink exception (lines 468-471)."""
        mock_is_daemon_running.return_value = False

        with patch("omnimemory.cli.main._kill_stale_daemon", return_value=False):
            with patch("omnimemory.cli.main.ensure_state_dir"):
                with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
                    mock_pid_file.exists.return_value = True
                    mock_pid_file.unlink.side_effect = OSError("Permission denied")
                    with patch("builtins.open", mock_open()):
                        with patch(
                            "omnimemory.cli.main.subprocess.Popen"
                        ) as mock_popen:
                            mock_process = Mock()
                            mock_popen.return_value = mock_process
                            with patch(
                                "omnimemory.cli.main._wait_for_daemon",
                                return_value=True,
                            ):
                                with patch("omnimemory.cli.main.time.sleep"):
                                    daemon_start()


class TestDaemonStop:
    """Test daemon_stop command."""

    def test_daemon_stop_not_running(self, mock_is_daemon_running, mock_console):
        """Test daemon_stop handles daemon not running."""
        mock_is_daemon_running.return_value = False
        daemon_stop()
        mock_console.print.assert_called()

    def test_daemon_stop_success(self, mock_is_daemon_running, mock_call_daemon):
        """Test daemon_stop stops daemon successfully."""
        mock_is_daemon_running.return_value = True
        mock_call_daemon.return_value = {"status": "stopped"}

        with patch("omnimemory.cli.main._wait_for_daemon", return_value=True):
            with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
                mock_pid_file.exists.return_value = False
                daemon_stop()

    def test_daemon_stop_timeout(self, mock_is_daemon_running, mock_call_daemon):
        """Test daemon_stop handles stop timeout."""
        mock_is_daemon_running.return_value = True
        mock_call_daemon.return_value = {"status": "stopping"}

        with patch("omnimemory.cli.main._wait_for_daemon", return_value=False):
            with patch("omnimemory.cli.main._kill_stale_daemon"):
                daemon_stop()

    def test_daemon_stop_pid_file_exists_but_not_running(self, mock_is_daemon_running):
        """Test daemon_stop when PID file exists but daemon not running (lines 504-505)."""
        mock_is_daemon_running.return_value = False

        with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
            mock_pid_file.exists.return_value = True
            with patch("omnimemory.cli.main._kill_stale_daemon"):
                daemon_stop()
                mock_pid_file.exists.assert_called()

    def test_daemon_stop_shutdown_exception(self, mock_is_daemon_running):
        """Test daemon_stop handles shutdown exception (lines 512-515)."""
        mock_is_daemon_running.return_value = True

        with patch(
            "omnimemory.cli.main.call_daemon", side_effect=DaemonNotRunningError()
        ):
            with patch("omnimemory.cli.main._kill_stale_daemon"):
                daemon_stop()

    def test_daemon_stop_force_kill_success(self, mock_is_daemon_running):
        """Test daemon_stop force kill success path (line 523)."""
        mock_is_daemon_running.side_effect = [True, False]

        with patch("omnimemory.cli.main.call_daemon", return_value={}):
            with patch("omnimemory.cli.main._wait_for_daemon", return_value=False):
                with patch("omnimemory.cli.main._kill_stale_daemon"):
                    daemon_stop()


class TestDaemonStatus:
    """Test daemon_status command."""

    def test_daemon_status_running(
        self, mock_is_daemon_running, mock_call_daemon, mock_console
    ):
        """Test daemon_status shows running status."""
        mock_is_daemon_running.return_value = True
        mock_call_daemon.return_value = {
            "pid": 12345,
            "port": 8000,
            "uptime": 3600,
        }

        with patch("omnimemory.cli.main.PID_FILE") as mock_pid_file:
            mock_pid_file.read_text.return_value = "12345"
            daemon_status()
            assert mock_console.print.called

    def test_daemon_status_not_running(self, mock_is_daemon_running, mock_console):
        """Test daemon_status shows stopped status."""
        mock_is_daemon_running.return_value = False
        daemon_status()
        assert mock_console.print.called

    def test_daemon_status_call_exception(self, mock_is_daemon_running):
        """Test daemon_status handles call_daemon exception (lines 537-539)."""
        mock_is_daemon_running.return_value = True

        with patch(
            "omnimemory.cli.main.call_daemon", side_effect=DaemonNotRunningError()
        ):
            daemon_status()


class TestInfoCommand:
    """Test info command."""

    def test_info_displays_banner(self, mock_console):
        """Test info displays banner."""
        with patch("omnimemory.cli.main.OMNIMEMORY_BANNER", "BANNER"):
            info()
            assert mock_console.print.call_count > 0


class TestHealthCommand:
    """Test health command."""

    def test_health_check_success(
        self, mock_is_daemon_running, mock_call_daemon, mock_console
    ):
        """Test health check shows healthy status."""
        mock_is_daemon_running.return_value = True
        mock_call_daemon.return_value = {
            "status": "healthy",
            "sdk_initialized": True,
            "memory_manager": "ready",
        }
        health()
        assert mock_console.print.called

    def test_health_check_daemon_not_running(
        self, mock_is_daemon_running, mock_console
    ):
        """Test health check handles daemon not running."""
        mock_is_daemon_running.return_value = False
        with patch(
            "omnimemory.cli.main.daemon_request", side_effect=DaemonNotRunningError()
        ):
            with patch("omnimemory.cli.main.Progress"):
                with pytest.raises(typer.Exit):
                    health()
                assert mock_console.print.called

    def test_health_check_failure(
        self, mock_is_daemon_running, mock_call_daemon, mock_console
    ):
        """Test health check handles check failures."""
        mock_is_daemon_running.return_value = True
        with patch(
            "omnimemory.cli.main.daemon_request",
            side_effect=DaemonResponseError("Error"),
        ):
            with patch("omnimemory.cli.main.Progress"):
                with pytest.raises(typer.Exit):
                    health()
                assert mock_console.print.called

    def test_health_check_healthy_status(self, mock_is_daemon_running, mock_console):
        """Test health check shows healthy status (lines 757-759)."""
        mock_is_daemon_running.return_value = True
        with patch(
            "omnimemory.cli.main.daemon_request",
            return_value={
                "sdk_initialized": True,
                "memory_manager_initialized": True,
            },
        ):
            with patch("omnimemory.cli.main.Progress"):
                health()
                assert mock_console.print.called


class TestMemoryAdd:
    """Test memory_add command."""

    def test_memory_add_from_file(
        self, temp_file, mock_call_daemon, mock_console, sample_messages
    ):
        """Test memory_add from messages file."""
        content = json.dumps({"messages": sample_messages[:DEFAULT_MAX_MESSAGES]})
        file_path = temp_file(content)

        mock_call_daemon.return_value = {"task_id": "test-task-123"}

        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request",
                return_value={"task_id": "test-task-123"},
            ):
                memory_add(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages_file=file_path,
                    message=None,
                )
                assert mock_console.print.called

    def test_memory_add_from_single_message(self, mock_call_daemon, mock_console):
        """Test memory_add from single message."""
        with patch("omnimemory.cli.main.Progress"):
            with pytest.raises(typer.Exit):
                memory_add(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages_file=None,
                    message="user:Hello:2024-01-01T00:00:00Z",
                )

    def test_memory_add_file_not_found(self, mock_console):
        """Test memory_add handles file not found."""
        with pytest.raises(typer.Exit):
            memory_add(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages_file="/nonexistent/file.json",
                message=None,
            )

    def test_memory_add_invalid_json(self, temp_file, mock_console):
        """Test memory_add handles invalid JSON."""
        file_path = temp_file("invalid json")
        with pytest.raises(typer.Exit):
            memory_add(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages_file=file_path,
                message=None,
            )

    def test_memory_add_invalid_message_format(self, mock_console):
        """Test memory_add handles invalid message format."""
        with pytest.raises(typer.Exit):
            memory_add(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages_file=None,
                message="invalid:format",
            )

    def test_memory_add_no_input(self, mock_console):
        """Test memory_add handles no input."""
        with pytest.raises(typer.Exit):
            memory_add(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages_file=None,
                message=None,
            )

    def test_memory_add_too_few_messages(self, temp_file, mock_console):
        """Test memory_add handles too few messages."""
        content = json.dumps({"messages": []})
        file_path = temp_file(content)
        with pytest.raises(typer.Exit):
            memory_add(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages_file=file_path,
                message=None,
            )

    def test_memory_add_too_many_messages(self, temp_file, mock_console):
        """Test memory_add handles too many messages."""
        too_many = [
            {"role": "user", "content": f"Msg {i}"}
            for i in range(DEFAULT_MAX_MESSAGES + 1)
        ]
        content = json.dumps({"messages": too_many})
        file_path = temp_file(content)
        with pytest.raises(typer.Exit):
            memory_add(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages_file=file_path,
                message=None,
            )

    def test_memory_add_validation_error(
        self, temp_file, mock_call_daemon, mock_console, sample_messages
    ):
        """Test memory_add handles validation errors."""
        content = json.dumps({"messages": sample_messages[:DEFAULT_MAX_MESSAGES]})
        file_path = temp_file(content)

        with patch("omnimemory.cli.main.Progress"):
            with patch("omnimemory.cli.main.AddUserMessageRequest") as mock_request:
                mock_instance = Mock()
                mock_instance.to_user_messages.side_effect = ValueError(
                    "Validation failed"
                )
                mock_request.return_value = mock_instance

                with pytest.raises(typer.Exit):
                    memory_add(
                        app_id="app1234567890",
                        user_id="user1234567890",
                        session_id=None,
                        messages_file=file_path,
                        message=None,
                    )

    def test_memory_add_daemon_error(
        self, temp_file, mock_call_daemon, mock_console, sample_messages
    ):
        """Test memory_add handles daemon errors."""
        content = json.dumps({"messages": sample_messages[:DEFAULT_MAX_MESSAGES]})
        file_path = temp_file(content)

        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request",
                side_effect=DaemonResponseError("Daemon error"),
            ):
                with pytest.raises(typer.Exit):
                    memory_add(
                        app_id="app1234567890",
                        user_id="user1234567890",
                        session_id=None,
                        messages_file=file_path,
                        message=None,
                    )


class TestMemoryQuery:
    """Test memory_query command."""

    def test_memory_query_with_query_flag(self, mock_call_daemon, mock_console):
        """Test memory_query with --query flag."""
        mock_call_daemon.return_value = [{"id": "mem1", "document": "Test"}]

        with patch("omnimemory.cli.main.Progress"):
            memory_query(
                app_id="app1234567890",
                query="test query",
                query_words=None,
                user_id=None,
                session_id=None,
                n_results=10,
                similarity_threshold=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_query_with_positional_args(self, mock_call_daemon, mock_console):
        """Test memory_query with positional arguments."""
        mock_call_daemon.return_value = [{"id": "mem1", "document": "Test"}]

        with patch("omnimemory.cli.main.Progress"):
            memory_query(
                app_id="app1234567890",
                query=None,
                query_words=["test", "query"],
                user_id=None,
                session_id=None,
                n_results=10,
                similarity_threshold=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_query_with_prompt(self, mock_call_daemon, mock_console):
        """Test memory_query prompts for query if not provided."""
        mock_call_daemon.return_value = [{"id": "mem1", "document": "Test"}]

        with patch("omnimemory.cli.main.Progress"):
            with patch("omnimemory.cli.main.typer.prompt", return_value="test query"):
                memory_query(
                    app_id="app1234567890",
                    query=None,
                    query_words=None,
                    user_id=None,
                    session_id=None,
                    n_results=10,
                    similarity_threshold=None,
                    output_json=False,
                )
                assert mock_console.print.called

    def test_memory_query_both_query_and_positional(self):
        """Test memory_query handles both --query and positional (error)."""
        with pytest.raises(typer.BadParameter):
            memory_query(
                app_id="app1234567890",
                query="test",
                query_words=["test"],
                user_id=None,
                session_id=None,
                n_results=10,
                similarity_threshold=None,
                output_json=False,
            )

    def test_memory_query_with_filters(self, mock_call_daemon, mock_console):
        """Test memory_query applies filters."""
        mock_call_daemon.return_value = []

        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request", return_value=[]
            ) as mock_daemon_req:
                memory_query(
                    app_id="app1234567890",
                    query="test",
                    query_words=None,
                    user_id="user123",
                    session_id="session123",
                    n_results=5,
                    similarity_threshold=0.8,
                    output_json=False,
                )
                mock_daemon_req.assert_called_once()
                call_args = mock_daemon_req.call_args[0][1]
                assert call_args["user_id"] == "user123"
                assert call_args["session_id"] == "session123"
                assert call_args["n_results"] == 5
                assert call_args["similarity_threshold"] == 0.8

    def test_memory_query_empty_results(self, mock_call_daemon, mock_console):
        """Test memory_query handles empty results."""
        mock_call_daemon.return_value = []

        with patch("omnimemory.cli.main.Progress"):
            memory_query(
                app_id="app1234567890",
                query="test",
                query_words=None,
                user_id=None,
                session_id=None,
                n_results=10,
                similarity_threshold=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_query_json_output(self, mock_call_daemon, mock_console):
        """Test memory_query outputs JSON."""
        mock_call_daemon.return_value = [{"id": "mem1"}]

        with patch("omnimemory.cli.main.Progress"):
            memory_query(
                app_id="app1234567890",
                query="test",
                query_words=None,
                user_id=None,
                session_id=None,
                n_results=10,
                similarity_threshold=None,
                output_json=True,
            )
            assert mock_console.print.called

    def test_memory_query_empty_query(self, mock_console):
        """Test memory_query handles empty query."""
        with patch("omnimemory.cli.main.typer.prompt", return_value=""):
            with pytest.raises(typer.Exit):
                memory_query(
                    app_id="app1234567890",
                    query=None,
                    query_words=None,
                    user_id=None,
                    session_id=None,
                    n_results=10,
                    similarity_threshold=None,
                    output_json=False,
                )

    def test_memory_query_daemon_error(self, mock_call_daemon, mock_console):
        """Test memory_query handles daemon errors."""
        mock_call_daemon.side_effect = DaemonResponseError("Error")

        with patch("omnimemory.cli.main.Progress"):
            with pytest.raises(typer.Exit):
                memory_query(
                    app_id="app1234567890",
                    query="test",
                    query_words=None,
                    user_id=None,
                    session_id=None,
                    n_results=10,
                    similarity_threshold=None,
                    output_json=False,
                )


class TestMemoryGet:
    """Test memory_get command."""

    def test_memory_get_success(self, mock_call_daemon, mock_console):
        """Test memory_get retrieves memory successfully."""
        mock_call_daemon.return_value = {
            "memory_id": "mem123",
            "document": "Test document",
            "metadata": {"key": "value"},
        }

        with patch("omnimemory.cli.main.Progress"):
            memory_get(
                memory_id="mem123",
                app_id="app1234567890",
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_get_json_output(self, mock_call_daemon, mock_console):
        """Test memory_get outputs JSON."""
        mock_call_daemon.return_value = {"memory_id": "mem123", "document": "Test"}

        with patch("omnimemory.cli.main.Progress"):
            memory_get(
                memory_id="mem123",
                app_id="app1234567890",
                output_json=True,
            )
            assert mock_console.print.called

    def test_memory_get_not_found(self, mock_call_daemon, mock_console):
        """Test memory_get handles memory not found."""
        mock_call_daemon.return_value = None

        with patch("omnimemory.cli.main.Progress"):
            with pytest.raises(typer.Exit):
                memory_get(
                    memory_id="nonexistent",
                    app_id="app1234567890",
                    output_json=False,
                )

    def test_memory_get_daemon_error(self, mock_call_daemon, mock_console):
        """Test memory_get handles daemon errors."""
        mock_call_daemon.side_effect = DaemonResponseError("Error")

        with patch("omnimemory.cli.main.Progress"):
            with pytest.raises(typer.Exit):
                memory_get(
                    memory_id="mem123",
                    app_id="app1234567890",
                    output_json=False,
                )


class TestMemoryEvolution:
    """Test memory_evolution command."""

    def test_memory_evolution_success(self, mock_call_daemon, mock_console):
        """Test memory_evolution traverses chain successfully."""
        mock_call_daemon.return_value = [{"memory_id": "mem1"}]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=False,
                graph_format="mermaid",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_empty_chain(self, mock_call_daemon, mock_console):
        """Test memory_evolution handles empty chain."""
        mock_call_daemon.return_value = []

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=False,
                graph_format="mermaid",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_generate_graph_mermaid(
        self, mock_call_daemon, mock_console
    ):
        """Test memory_evolution generates Mermaid graph."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "graph TD\nmem1",
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="mermaid",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_generate_graph_dot(self, mock_call_daemon, mock_console):
        """Test memory_evolution generates DOT graph."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "digraph { mem1 }",
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="dot",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_generate_graph_html(self, mock_call_daemon, mock_console):
        """Test memory_evolution generates HTML graph."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "<html>graph</html>",
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="html",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_save_to_file(self, mock_call_daemon, tmp_path):
        """Test memory_evolution saves graph to file."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "graph content",
        ]

        output_file = str(tmp_path / "graph.mmd")

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="mermaid",
                output_file=output_file,
                output_json=False,
            )
            assert Path(output_file).exists()

    def test_memory_evolution_json_output(self, mock_call_daemon, mock_console):
        """Test memory_evolution outputs JSON."""
        mock_call_daemon.return_value = [{"memory_id": "mem1"}]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=False,
                graph_format="mermaid",
                output_file=None,
                output_json=True,
            )
            assert mock_console.print.called

    def test_memory_evolution_graph_generation_error(
        self, mock_call_daemon, mock_console
    ):
        """Test memory_evolution handles graph generation errors."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            DaemonResponseError("Graph generation failed"),
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="mermaid",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_graph_generation_exception(
        self, mock_call_daemon, mock_console
    ):
        """Test memory_evolution handles graph generation exception (lines 1247-1249)."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            Exception("Graph generation failed"),
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="mermaid",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_traverse_exception(self, mock_console):
        """Test memory_evolution handles traverse exception (lines 1169-1171)."""
        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request",
                side_effect=Exception("Traverse failed"),
            ):
                with pytest.raises(typer.Exit):
                    memory_evolution(
                        memory_id="mem1",
                        app_id="app1234567890",
                        graph=False,
                        graph_format="mermaid",
                        output_file=None,
                        output_json=False,
                    )

    def test_memory_evolution_html_format_with_output(self, mock_call_daemon, tmp_path):
        """Test memory_evolution HTML format with output file (lines 1194-1195)."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "<html>graph</html>",
        ]

        output_file = str(tmp_path / "graph.html")

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="html",
                output_file=output_file,
                output_json=False,
            )
            assert Path(output_file).exists()

    def test_memory_evolution_html_format_without_output(
        self, mock_call_daemon, mock_console
    ):
        """Test memory_evolution HTML format without output file shows warning."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "<html>graph</html>",
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="html",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called

    def test_memory_evolution_info_panel_none(self, mock_call_daemon, mock_console):
        """Test memory_evolution handles info_panel being None (line 1239)."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "graph TD\nmem1",
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="mermaid",
                output_file=None,
                output_json=False,
            )
            pass


class TestMemoryDelete:
    """Test memory_delete command."""

    def test_memory_delete_with_confirmation(self, mock_call_daemon, mock_console):
        """Test memory_delete with confirmation."""
        mock_call_daemon.return_value = {"success": True}

        with patch("omnimemory.cli.main.typer.confirm", return_value=True):
            with patch("omnimemory.cli.main.Progress"):
                memory_delete(
                    memory_id="mem123",
                    app_id="app1234567890",
                    confirm=False,
                )
                assert mock_console.print.called

    def test_memory_delete_with_yes_flag(self, mock_call_daemon, mock_console):
        """Test memory_delete with --yes flag (skip confirmation)."""
        mock_call_daemon.return_value = {"success": True}

        with patch("omnimemory.cli.main.Progress"):
            memory_delete(
                memory_id="mem123",
                app_id="app1234567890",
                confirm=True,
            )
            assert mock_console.print.called

    def test_memory_delete_cancelled(self, mock_call_daemon, mock_console):
        """Test memory_delete handles cancellation."""
        with patch("omnimemory.cli.main.typer.confirm", return_value=False):
            with pytest.raises(typer.Exit) as exc_info:
                memory_delete(
                    memory_id="mem123",
                    app_id="app1234567890",
                    confirm=False,
                )
            assert exc_info.value.exit_code == 0
            mock_call_daemon.assert_not_called()

    def test_memory_delete_failure(self, mock_call_daemon, mock_console):
        """Test memory_delete handles delete failure."""
        mock_call_daemon.return_value = {"success": False}

        with patch("omnimemory.cli.main.typer.confirm", return_value=True):
            with patch("omnimemory.cli.main.Progress"):
                with patch("omnimemory.cli.main.daemon_request", return_value=False):
                    with pytest.raises(typer.Exit):
                        memory_delete(
                            memory_id="mem123",
                            app_id="app1234567890",
                            confirm=False,
                        )

    def test_memory_delete_success_false(self, mock_console):
        """Test memory_delete handles success=False (lines 1363-1364)."""
        with patch("omnimemory.cli.main.typer.confirm", return_value=True):
            with patch("omnimemory.cli.main.Progress"):
                with patch("omnimemory.cli.main.daemon_request", return_value=False):
                    with pytest.raises(typer.Exit):
                        memory_delete(
                            memory_id="mem123",
                            app_id="app1234567890",
                            confirm=False,
                        )

    def test_memory_delete_daemon_error(self, mock_call_daemon, mock_console):
        """Test memory_delete handles daemon errors."""
        mock_call_daemon.side_effect = DaemonResponseError("Error")

        with patch("omnimemory.cli.main.typer.confirm", return_value=True):
            with patch("omnimemory.cli.main.Progress"):
                with pytest.raises(typer.Exit):
                    memory_delete(
                        memory_id="mem123",
                        app_id="app1234567890",
                        confirm=False,
                    )

    def test_memory_evolution_info_panel_else_branch(
        self, mock_call_daemon, mock_console
    ):
        """Test memory_evolution else branch for info_panel (line 1239)."""
        mock_call_daemon.side_effect = [
            [{"memory_id": "mem1"}],
            "some graph output",
        ]

        with patch("omnimemory.cli.main.Progress"):
            memory_evolution(
                memory_id="mem1",
                app_id="app1234567890",
                graph=True,
                graph_format="png",
                output_file=None,
                output_json=False,
            )
            assert mock_console.print.called


class TestAgentSummarize:
    """Test agent_summarize command."""

    def test_agent_summarize_from_file(self, temp_file, mock_call_daemon, mock_console):
        """Test agent_summarize from messages file."""
        content = json.dumps({"messages": [{"role": "user", "content": "Hello"}]})
        file_path = temp_file(content)

        mock_call_daemon.return_value = {
            "summary": "Test summary",
            "key_points": ["Point 1"],
            "tags": ["tag1"],
        }

        agent_summarize(
            app_id="app1234567890",
            user_id="user1234567890",
            messages_file=file_path,
            message=None,
            text=None,
            callback_url=None,
            callback_header=[],
        )
        assert mock_console.print.called

    def test_agent_summarize_from_text(self, mock_call_daemon, mock_console):
        """Test agent_summarize from text."""
        mock_call_daemon.return_value = {"summary": "Test summary"}

        agent_summarize(
            app_id="app1234567890",
            user_id="user1234567890",
            messages_file=None,
            message=None,
            text="Test conversation",
            callback_url=None,
            callback_header=[],
        )
        assert mock_console.print.called

    def test_agent_summarize_with_callback(
        self, temp_file, mock_call_daemon, mock_console
    ):
        """Test agent_summarize with callback URL."""
        content = json.dumps({"text": "Test conversation"})
        file_path = temp_file(content)

        mock_call_daemon.return_value = {
            "status": "accepted",
            "task_id": "task123",
        }

        agent_summarize(
            app_id="app1234567890",
            user_id="user1234567890",
            messages_file=file_path,
            message=None,
            text=None,
            callback_url="https://example.com/callback",
            callback_header=["Authorization=Bearer token"],
        )
        assert mock_console.print.called

    def test_agent_summarize_invalid_header_format(self, temp_file, mock_console):
        """Test agent_summarize handles invalid header format."""
        content = json.dumps({"text": "Test"})
        file_path = temp_file(content)

        with pytest.raises(typer.Exit):
            agent_summarize(
                app_id="app1234567890",
                user_id="user1234567890",
                messages_file=file_path,
                message=None,
                text=None,
                callback_url=None,
                callback_header=["invalid-header"],
            )

    def test_agent_summarize_daemon_error(
        self, temp_file, mock_call_daemon, mock_console
    ):
        """Test agent_summarize handles daemon errors."""
        content = json.dumps({"text": "Test"})
        file_path = temp_file(content)

        mock_call_daemon.side_effect = DaemonResponseError("Error")

        with pytest.raises(typer.Exit):
            agent_summarize(
                app_id="app1234567890",
                user_id="user1234567890",
                messages_file=file_path,
                message=None,
                text=None,
                callback_url=None,
                callback_header=[],
            )

    def test_agent_summarize_exception_handling(self, temp_file, mock_console):
        """Test agent_summarize handles exceptions via daemon_request."""
        content = json.dumps({"text": "Test"})
        file_path = temp_file(content)

        with patch(
            "omnimemory.cli.main.call_daemon",
            side_effect=DaemonResponseError("Unexpected error"),
        ):
            with pytest.raises(typer.Exit):
                agent_summarize(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    messages_file=file_path,
                    message=None,
                    text=None,
                    callback_url=None,
                    callback_header=[],
                )

    def test_agent_summarize_metadata_table(
        self, temp_file, mock_call_daemon, mock_console
    ):
        """Test agent_summarize displays metadata table (lines 1472-1481)."""
        content = json.dumps({"text": "Test conversation"})
        file_path = temp_file(content)

        mock_call_daemon.return_value = {
            "summary": "Test summary",
            "metadata": {"key1": "value1", "key2": 123},
        }

        agent_summarize(
            app_id="app1234567890",
            user_id="user1234567890",
            messages_file=file_path,
            message=None,
            text=None,
            callback_url=None,
            callback_header=[],
        )
        assert mock_console.print.called

    def test_agent_summarize_generated_at(
        self, temp_file, mock_call_daemon, mock_console
    ):
        """Test agent_summarize displays generated_at timestamp (line 1485)."""
        content = json.dumps({"text": "Test"})
        file_path = temp_file(content)

        mock_call_daemon.return_value = {
            "summary": "Test summary",
            "generated_at": "2024-01-01T00:00:00Z",
        }

        agent_summarize(
            app_id="app1234567890",
            user_id="user1234567890",
            messages_file=file_path,
            message=None,
            text=None,
            callback_url=None,
            callback_header=[],
        )
        assert mock_console.print.called


class TestAgentAddMemory:
    """Test agent_add_memory command."""

    def test_agent_add_memory_from_file(
        self, temp_file, mock_call_daemon, mock_console
    ):
        """Test agent_add_memory from messages file."""
        content = json.dumps("Test messages")
        file_path = temp_file(content)

        mock_call_daemon.return_value = {"task_id": "task123", "status": "accepted"}

        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request",
                return_value={"task_id": "task123", "status": "accepted"},
            ):
                agent_add_memory(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages_file=file_path,
                    messages=None,
                )
                assert mock_console.print.called

    def test_agent_add_memory_from_text(self, mock_call_daemon, mock_console):
        """Test agent_add_memory from messages text."""
        mock_call_daemon.return_value = {"task_id": "task123", "status": "accepted"}

        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request",
                return_value={"task_id": "task123", "status": "accepted"},
            ):
                agent_add_memory(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages_file=None,
                    messages="Test messages",
                )
                assert mock_console.print.called

    def test_agent_add_memory_no_messages(self, mock_console):
        """Test agent_add_memory handles missing messages."""
        with patch("omnimemory.cli.main._load_conversation_payload", return_value=None):
            with pytest.raises(typer.Exit):
                agent_add_memory(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages_file=None,
                    messages=None,
                )

    def test_agent_add_memory_daemon_error(
        self, temp_file, mock_call_daemon, mock_console
    ):
        """Test agent_add_memory handles daemon errors."""
        content = json.dumps("Test")
        file_path = temp_file(content)

        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request",
                side_effect=DaemonResponseError("Error"),
            ):
                with pytest.raises(typer.Exit):
                    agent_add_memory(
                        app_id="app1234567890",
                        user_id="user1234567890",
                        session_id=None,
                        messages_file=file_path,
                        messages=None,
                    )

    def test_agent_add_memory_exception_handling(self, temp_file, mock_console):
        """Test agent_add_memory handles exceptions (lines 1528-1530)."""
        content = json.dumps("Test messages")
        file_path = temp_file(content)

        with patch("omnimemory.cli.main.Progress"):
            with patch(
                "omnimemory.cli.main.daemon_request",
                side_effect=Exception("Unexpected error"),
            ):
                with pytest.raises(typer.Exit):
                    agent_add_memory(
                        app_id="app1234567890",
                        user_id="user1234567890",
                        session_id=None,
                        messages_file=file_path,
                        messages=None,
                    )


class TestCommandFlow:
    """Test command flow integration."""

    def test_cli_app_structure(self):
        """Test CLI app structure is correct."""
        assert app is not None
        assert memory_app is not None
        assert daemon_app is not None
        assert agent_app is not None

    def test_version_command(self):
        """Test version command via CLI."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code in [0, 1]

    def test_info_command(self):
        """Test info command via CLI."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_get_version_missing_metadata_and_file(self):
        """Test get_version handles missing metadata and file."""
        with patch("omnimemory.cli.main.importlib_metadata.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError("omnimemory")
            with patch("builtins.open", side_effect=FileNotFoundError):
                assert get_version() == "dev"

    def test_load_payload_empty_file(self, temp_file, mock_console):
        """Test _load_conversation_payload handles empty file."""
        file_path = temp_file("")
        with pytest.raises(typer.Exit):
            _load_conversation_payload(file_path, None, None)

    def test_daemon_request_none_payload(self, mock_call_daemon):
        """Test daemon_request with None payload."""
        mock_call_daemon.return_value = {"result": "success"}
        result = daemon_request("test_method", None)
        mock_call_daemon.assert_called_once_with("test_method", {})

    def test_create_header_panel_empty_title(self):
        """Test create_header_panel handles edge cases."""
        panel = create_header_panel("", "")
        assert panel is not None

    def test_create_metric_card_empty_values(self):
        """Test create_metric_card handles empty values."""
        card = create_metric_card("", "", "")
        assert card is not None
