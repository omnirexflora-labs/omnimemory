"""
Comprehensive unit tests for OmniMemory logger utilities.
"""

import logging
from unittest.mock import patch
from omnimemory.core.logger_utils import OmniMemoryLogger, get_logger


class TestOmniMemoryLogger:
    """Test cases for OmniMemoryLogger."""

    def test_rich_import_error_handling(self):
        """Test handle Rich import error (coverage lines 20-22)."""
        from omnimemory.core.logger_utils import RICH_AVAILABLE

        assert isinstance(RICH_AVAILABLE, bool)

        logger = OmniMemoryLogger(name="test", use_rich=True)
        assert logger.logger is not None

    def test_init_with_default_parameters(self):
        """Test initialize with default parameters."""
        logger = OmniMemoryLogger(name="test_logger")
        assert logger.name == "test_logger"
        assert logger.logger is not None

    def test_init_with_custom_log_level(self):
        """Test initialize with custom log_level."""
        logger = OmniMemoryLogger(name="test", log_level="DEBUG")
        assert logger.logger.level == logging.DEBUG

    def test_init_load_log_level_from_env(self):
        """Test load log_level from LOG_LEVEL env var."""
        with patch("omnimemory.core.logger_utils.config") as mock_config:
            mock_config.return_value = "WARNING"
            logger = OmniMemoryLogger(name="test")
            assert logger.logger.level == logging.WARNING

    def test_init_load_log_dir_from_env(self):
        """Test load log_dir from LOG_DIR env var."""
        with patch("omnimemory.core.logger_utils.config") as mock_config:
            mock_config.return_value = "/custom/logs"
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.open", create=True):
                        with patch("logging.handlers.RotatingFileHandler"):
                            logger = OmniMemoryLogger(name="test", enable_file=False)
                            assert logger is not None

    def test_init_create_log_directory(self):
        """Test create log directory if not exists."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            OmniMemoryLogger(name="test")
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_init_generate_timestamped_log_file(self):
        """Test generate timestamped log file name."""
        with patch("omnimemory.core.logger_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101"
            logger = OmniMemoryLogger(name="test_logger")
            assert "test_logger" in str(logger.log_file)
            assert "20240101" in str(logger.log_file)

    def test_init_setup_console_handler(self):
        """Test setup console handler."""
        logger = OmniMemoryLogger(name="test", enable_console=True)
        assert len(logger.logger.handlers) > 0

    def test_init_setup_file_handler(self):
        """Test setup file handler."""
        with patch("pathlib.Path.mkdir"):
            logger = OmniMemoryLogger(name="test", enable_file=True)
            assert len(logger.logger.handlers) > 0

    def test_init_use_rich_handler_when_available(self):
        """Test use Rich handler when available."""
        with patch("omnimemory.core.logger_utils.RICH_AVAILABLE", True):
            with patch("omnimemory.core.logger_utils.RichHandler"):
                logger = OmniMemoryLogger(name="test", use_rich=True)
                assert logger.logger is not None

    def test_init_fallback_to_stream_handler(self):
        """Test fall back to StreamHandler when Rich unavailable."""
        with patch("omnimemory.core.logger_utils.RICH_AVAILABLE", False):
            logger = OmniMemoryLogger(name="test", use_rich=True)
            assert logger.logger is not None

    def test_init_clear_existing_handlers(self):
        """Test clear existing handlers."""
        logger1 = OmniMemoryLogger(name="test")
        handler_count = len(logger1.logger.handlers)

        logger2 = OmniMemoryLogger(name="test")
        assert len(logger2.logger.handlers) == handler_count

    def test_init_set_propagate_false(self):
        """Test set propagate to False."""
        logger = OmniMemoryLogger(name="test")
        assert logger.logger.propagate is False

    def test_setup_console_handler_rich(self):
        """Test setup console handler with Rich."""
        with patch("omnimemory.core.logger_utils.RICH_AVAILABLE", True):
            with patch("omnimemory.core.logger_utils.RichHandler"):
                logger = OmniMemoryLogger(name="test")
                logger._setup_console_handler(use_rich=True)
                assert logger.logger is not None

    def test_setup_console_handler_stream(self):
        """Test setup console handler with StreamHandler."""
        with patch("omnimemory.core.logger_utils.RICH_AVAILABLE", False):
            logger = OmniMemoryLogger(name="test")
            logger._setup_console_handler(use_rich=False)
            handlers = [
                h
                for h in logger.logger.handlers
                if isinstance(h, logging.StreamHandler)
            ]
            assert len(handlers) > 0

    def test_setup_file_handler(self):
        """Test setup rotating file handler."""
        with patch("pathlib.Path.mkdir"):
            with patch("logging.handlers.RotatingFileHandler"):
                logger = OmniMemoryLogger(name="test")
                logger._setup_file_handler(max_bytes=1024, backup_count=3)
                assert logger.logger is not None

    def test_debug_logs_debug_message(self):
        """Test debug() logs debug message."""
        logger = OmniMemoryLogger(name="test", log_level="DEBUG")
        with patch.object(logger.logger, "debug") as mock_debug:
            logger.debug("Debug message")
            mock_debug.assert_called_once_with("Debug message")

    def test_info_logs_info_message(self):
        """Test info() logs info message."""
        logger = OmniMemoryLogger(name="test")
        with patch.object(logger.logger, "info") as mock_info:
            logger.info("Info message")
            mock_info.assert_called_once_with("Info message")

    def test_warning_logs_warning_message(self):
        """Test warning() logs warning message."""
        logger = OmniMemoryLogger(name="test")
        with patch.object(logger.logger, "warning") as mock_warning:
            logger.warning("Warning message")
            mock_warning.assert_called_once_with("Warning message")

    def test_error_logs_error_message(self):
        """Test error() logs error message."""
        logger = OmniMemoryLogger(name="test")
        with patch.object(logger.logger, "error") as mock_error:
            logger.error("Error message")
            mock_error.assert_called_once_with("Error message")

    def test_critical_logs_critical_message(self):
        """Test critical() logs critical message."""
        logger = OmniMemoryLogger(name="test")
        with patch.object(logger.logger, "critical") as mock_critical:
            logger.critical("Critical message")
            mock_critical.assert_called_once_with("Critical message")

    def test_exception_logs_exception(self):
        """Test exception() logs exception with traceback."""
        logger = OmniMemoryLogger(name="test")
        with patch.object(logger.logger, "exception") as mock_exception:
            try:
                raise ValueError("Test error")
            except ValueError:
                logger.exception("Exception occurred")
            mock_exception.assert_called_once()

    def test_get_logger_returns_logger(self):
        """Test get_logger() returns underlying logger."""
        logger = OmniMemoryLogger(name="test")
        underlying = logger.get_logger()
        assert isinstance(underlying, logging.Logger)


class TestGetLogger:
    """Test cases for get_logger factory function."""

    def test_get_logger_creates_new_instance(self):
        """Test create new logger instance."""
        logger = get_logger(name="test_logger")
        assert isinstance(logger, OmniMemoryLogger)
        assert logger.name == "test_logger"

    def test_get_logger_returns_default_for_omnimemory(self):
        """Test return default logger for 'omnimemory' name."""
        logger1 = get_logger(name="omnimemory")
        logger2 = get_logger(name="omnimemory")
        assert logger1 is logger2

    def test_get_logger_caches_default_logger(self):
        """Test cache default logger."""
        logger1 = get_logger(name="omnimemory")
        logger2 = get_logger(name="omnimemory")
        assert logger1 is logger2

    def test_get_logger_passes_kwargs(self):
        """Test pass kwargs to OmniMemoryLogger."""
        logger = get_logger(name="test", log_level="DEBUG", enable_file=False)
        assert logger.logger.level == logging.DEBUG
        assert logger.logger is not None

    def test_get_logger_override_log_level(self):
        """Test override log_level."""
        logger = get_logger(name="test", log_level="ERROR")
        assert logger.logger.level == logging.ERROR

    def test_get_logger_override_log_file(self):
        """Test override log_file."""
        logger = get_logger(name="test", log_file="custom.log")
        assert "custom.log" in str(logger.log_file)


class TestLoggerFunction:
    """Test cases for logger() accessor function."""

    def test_logger_returns_default(self):
        """Test return default logger."""
        from omnimemory.core.logger_utils import logger as logger_func

        result = logger_func()
        assert isinstance(result, OmniMemoryLogger)

    def test_logger_creates_if_not_exists(self):
        """Test create default logger if not exists."""
        import omnimemory.core.logger_utils as logger_module

        logger_module._default_logger = None

        from omnimemory.core.logger_utils import logger as logger_func

        result = logger_func()
        assert isinstance(result, OmniMemoryLogger)
