"""
Prometheus-based metrics and observability system for OmniMemory.
"""

import time
import threading
from typing import Optional, Dict
from threading import Lock

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    REGISTRY,
    start_http_server,
)
from prometheus_client.core import CollectorRegistry

from omnimemory.core.logger_utils import get_logger
from omnimemory.core.config import METRICS_SERVER_PORT, ENABLE_METRICS_SERVER

logger = get_logger(name="omnimemory.core.metrics")


class MetricsCollector:
    """
    In-memory Prometheus metrics collector for OmniMemory.

    Uses prometheus_client library for thread-safe in-memory metrics.
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = Lock()

    def __new__(cls, enable: bool = True):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, enable: bool = True):
        if self._initialized:
            return

        self.enabled = enable
        self._server_started = False
        self._initialized = True

        if not self.enabled:
            return

        self._init_metrics()

    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        self.query_operations_total = Counter(
            "omnimemory_query_operations_total",
            "Total number of query operations",
            ["operation", "status"],
        )
        self.write_operations_total = Counter(
            "omnimemory_write_operations_total",
            "Total number of write operations",
            ["operation", "status"],
        )
        self.update_operations_total = Counter(
            "omnimemory_update_operations_total",
            "Total number of update operations",
            ["operation", "status"],
        )
        self.batch_operations_total = Counter(
            "omnimemory_batch_operations_total",
            "Total number of batch operations",
            ["operation", "status"],
        )
        self.errors_total = Counter(
            "omnimemory_errors_total",
            "Total number of errors",
            ["operation", "error_code"],
        )

        self.query_duration_seconds = Histogram(
            "omnimemory_query_duration_seconds",
            "Query operation duration in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )
        self.write_duration_seconds = Histogram(
            "omnimemory_write_duration_seconds",
            "Write operation duration in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )
        self.update_duration_seconds = Histogram(
            "omnimemory_update_duration_seconds",
            "Update operation duration in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )
        self.batch_duration_seconds = Histogram(
            "omnimemory_batch_duration_seconds",
            "Batch operation duration in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )
        self.query_results_count = Histogram(
            "omnimemory_query_results_count",
            "Number of results returned by query",
            ["operation"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )
        self.batch_size = Histogram(
            "omnimemory_batch_size",
            "Number of items in batch operation",
            ["operation"],
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
        )
        self.batch_success_rate = Histogram(
            "omnimemory_batch_success_rate",
            "Success rate of batch operation",
            ["operation"],
            buckets=[0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
        )

        self.health_status = Gauge(
            "omnimemory_health_status",
            "System health status (1=healthy, 0=unhealthy)",
        )
        self.active_queries = Gauge(
            "omnimemory_active_queries",
            "Currently active query operations",
        )
        self.active_writes = Gauge(
            "omnimemory_active_writes",
            "Currently active write operations",
        )

    def record_query(
        self,
        operation: str,
        duration: float,
        success: bool,
        results_count: Optional[int] = None,
        error_code: Optional[str] = None,
    ):
        """Record a query operation."""
        if not self.enabled:
            return

        status = "success" if success else "error"
        self.query_operations_total.labels(operation=operation, status=status).inc()
        self.query_duration_seconds.labels(operation=operation).observe(duration)

        if results_count is not None:
            self.query_results_count.labels(operation=operation).observe(
                float(results_count)
            )

        if not success and error_code:
            self.errors_total.labels(operation=operation, error_code=error_code).inc()

    def record_write(
        self,
        operation: str,
        duration: float,
        success: bool,
        error_code: Optional[str] = None,
    ):
        """Record a write operation."""
        if not self.enabled:
            return

        status = "success" if success else "error"
        self.write_operations_total.labels(operation=operation, status=status).inc()
        self.write_duration_seconds.labels(operation=operation).observe(duration)

        if not success and error_code:
            self.errors_total.labels(operation=operation, error_code=error_code).inc()

    def record_update(
        self,
        operation: str,
        duration: float,
        success: bool,
        error_code: Optional[str] = None,
    ):
        """Record an update operation."""
        if not self.enabled:
            return

        status = "success" if success else "error"
        self.update_operations_total.labels(operation=operation, status=status).inc()
        self.update_duration_seconds.labels(operation=operation).observe(duration)

        if not success and error_code:
            self.errors_total.labels(operation=operation, error_code=error_code).inc()

    def record_batch(
        self,
        operation: str,
        duration: float,
        total_items: int,
        succeeded: int,
        failed: int,
        error_code: Optional[str] = None,
    ):
        """Record a batch operation."""
        if not self.enabled:
            return

        status = (
            "success"
            if failed == 0
            else ("error" if failed == total_items else "partial")
        )
        self.batch_operations_total.labels(operation=operation, status=status).inc()
        self.batch_duration_seconds.labels(operation=operation).observe(duration)
        self.batch_size.labels(operation=operation).observe(float(total_items))

        if total_items > 0:
            success_rate = succeeded / total_items
            self.batch_success_rate.labels(operation=operation).observe(success_rate)

        if error_code:
            self.errors_total.labels(operation=operation, error_code=error_code).inc()

    def set_health(self, healthy: bool):
        """Set overall health status."""
        if self.enabled:
            self.health_status.set(1.0 if healthy else 0.0)

    def start_server(self, port: int = 8000) -> bool:
        """Start Prometheus metrics HTTP server."""
        if not self.enabled:
            return False

        if self._server_started:
            return False

        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()

            if result == 0:
                self._server_started = True
                return False

            start_http_server(port, registry=REGISTRY)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 98:
                self._server_started = True
                return False
            logger.warning(f"Failed to start metrics server on port {port}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to start metrics server on port {port}: {e}")
            return False

    def operation_timer(self, operation_type: str, operation_name: str):
        """Context manager for timing operations."""

        class Timer:
            def __init__(self, collector, op_type, op_name):
                self.collector = collector
                self.op_type = op_type
                self.op_name = op_name
                self.start_time = None
                self.success = True
                self.error_code = None
                self.results_count = None

            def __enter__(self):
                self.start_time = time.time()
                if self.op_type == "query":
                    self.collector._inc_active_queries()
                elif self.op_type == "write":
                    self.collector._inc_active_writes()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                if exc_type is not None:
                    self.success = False
                    self.error_code = exc_type.__name__

                if self.op_type == "query":
                    self.collector._dec_active_queries()
                elif self.op_type == "write":
                    self.collector._dec_active_writes()

                if self.op_type == "query":
                    self.collector.record_query(
                        operation=self.op_name,
                        duration=duration,
                        success=self.success,
                        results_count=self.results_count,
                        error_code=self.error_code,
                    )
                elif self.op_type == "write":
                    self.collector.record_write(
                        operation=self.op_name,
                        duration=duration,
                        success=self.success,
                        error_code=self.error_code,
                    )
                elif self.op_type == "update":
                    self.collector.record_update(
                        operation=self.op_name,
                        duration=duration,
                        success=self.success,
                        error_code=self.error_code,
                    )
                return False

        return Timer(self, operation_type, operation_name)

    def _inc_active_queries(self):
        """Increment active queries counter."""
        if self.enabled:
            self.active_queries.inc()

    def _dec_active_queries(self):
        """Decrement active queries counter."""
        if self.enabled:
            self.active_queries.dec()

    def _inc_active_writes(self):
        """Increment active writes counter."""
        if self.enabled:
            self.active_writes.inc()

    def _dec_active_writes(self):
        """Decrement active writes counter."""
        if self.enabled:
            self.active_writes.dec()

    def export(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.enabled:
            return "# Metrics disabled\n"
        try:
            return generate_latest(REGISTRY).decode("utf-8")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return f"# Error exporting metrics: {e}\n"


_metrics_instance: Optional[MetricsCollector] = None


def get_metrics_collector(enable: bool = True) -> MetricsCollector:
    """Get or create the singleton MetricsCollector instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector(enable=enable)
    return _metrics_instance


_metrics_server_started = False
_metrics_server_lock = threading.Lock()


def _auto_start_metrics_server():
    """Auto-start metrics server when module is imported (only once, if enabled)."""
    global _metrics_server_started

    if not ENABLE_METRICS_SERVER:
        logger.debug("Metrics server disabled (OMNIMEMORY_ENABLE_METRICS_SERVER=False)")
        return

    with _metrics_server_lock:
        if _metrics_server_started:
            return

        try:
            metrics = get_metrics_collector()
            if not metrics._server_started:
                if metrics.start_server(port=METRICS_SERVER_PORT):
                    _metrics_server_started = True
                    metrics.set_health(True)
                    metrics.active_queries.set(0.0)
                    metrics.active_writes.set(0.0)
            else:
                _metrics_server_started = True
        except Exception as e:
            logger.debug(f"Metrics server auto-start check: {e}")


def _start_metrics_async():
    """Start metrics server in background thread."""
    try:
        _auto_start_metrics_server()
    except Exception as e:
        logger.debug(f"Background metrics server startup: {e}")


_metrics_thread = threading.Thread(
    target=_start_metrics_async, daemon=True, name="metrics-init"
)
_metrics_thread.start()
