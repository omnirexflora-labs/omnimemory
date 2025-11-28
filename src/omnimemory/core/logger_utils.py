"""
Comprehensive logging utility for OmniMemory.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
from decouple import config

try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore[assignment, misc]


class OmniMemoryLogger:
    """
    Full-featured logger for OmniMemory with support for:
    - Console and file logging
    - Colored output (via Rich)
    - Log rotation
    - Environment-based configuration
    - Structured logging
    """

    def __init__(
        self,
        name: str = "omnimemory",
        log_level: Optional[str] = None,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        use_rich: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). If None, reads from LOG_LEVEL env var
            log_file: Log file name. If None, uses default
            log_dir: Directory for log files. If None, uses current directory or LOG_DIR env var
            enable_console: Enable console logging
            enable_file: Enable file logging
            use_rich: Use Rich for colored console output (if available)
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup log files to keep
        """
        self.name = name
        self.logger = logging.getLogger(name)

        if log_level is None:
            log_level = config("LOG_LEVEL", default="INFO").upper()

        level = getattr(logging, log_level, logging.INFO)
        self.logger.setLevel(level)

        if self.logger.handlers:
            self.logger.handlers.clear()

        self.logger.propagate = False

        if log_dir is None:
            log_dir = config("LOG_DIR", default="logs")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = f"{name}_{timestamp}.log"

        self.log_file = self.log_dir / log_file

        if enable_console:
            self._setup_console_handler(use_rich)

        if enable_file:
            self._setup_file_handler(max_bytes, backup_count)

    def _setup_console_handler(self, use_rich: bool) -> None:
        """Setup console handler with optional Rich formatting."""
        if use_rich and RICH_AVAILABLE:
            console_handler: logging.Handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True,
                rich_tracebacks=True,
                markup=True,
            )
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def _setup_file_handler(self, max_bytes: int, backup_count: int) -> None:
        """Setup rotating file handler.

        Handles permission errors gracefully by falling back to console-only logging.
        This is especially important during test runs where log directories may not be writable.
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            test_file = self.log_dir / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except (PermissionError, OSError):
                if self.logger.handlers:
                    self.logger.warning(
                        f"Cannot write to log directory {self.log_dir}. "
                        "File logging disabled. Continuing with console logging only."
                    )
                return

            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
        except (PermissionError, OSError) as e:
            if self.logger.handlers:
                self.logger.warning(
                    f"Cannot create log file {self.log_file}: {e}. "
                    "File logging disabled. Continuing with console logging only."
                )

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)

    def exception(
        self, message: str, *args: Any, exc_info: bool = True, **kwargs: Any
    ) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, *args, exc_info=exc_info, **kwargs)

    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self.logger


_default_logger: Optional[OmniMemoryLogger] = None


def get_logger(
    name: str = "omnimemory",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    **kwargs,
) -> OmniMemoryLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name
        log_level: Log level override
        log_file: Log file name override
        **kwargs: Additional arguments passed to OmniMemoryLogger

    Returns:
        OmniMemoryLogger instance
    """
    global _default_logger

    if name == "omnimemory" and _default_logger is not None:
        return _default_logger

    logger = OmniMemoryLogger(
        name=name, log_level=log_level, log_file=log_file, **kwargs
    )

    if name == "omnimemory":
        _default_logger = logger

    return logger


def logger() -> OmniMemoryLogger:
    """Get the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    return _default_logger
