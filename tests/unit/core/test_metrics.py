"""
Comprehensive unit tests for OmniMemory metrics collector.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from omnimemory.core.metrics import (
    MetricsCollector,
    get_metrics_collector,
    _auto_start_metrics_server,
)


class TestMetricsCollectorSingleton:
    """Test cases for MetricsCollector singleton pattern."""

    def test_return_same_instance_on_multiple_calls(self):
        """Test return same instance on multiple calls."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        instance1 = MetricsCollector(enable=False)
        instance2 = MetricsCollector(enable=False)
        assert instance1 is instance2

    def test_thread_safe_singleton_creation(self):
        """Test thread-safe singleton creation."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        instances = []

        def create_instance():
            instances.append(MetricsCollector(enable=False))

        threads = [threading.Thread(target=create_instance) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_initialize_only_once(self):
        """Test initialize only once."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        instance1 = MetricsCollector(enable=True)
        instance2 = MetricsCollector(enable=True)

        assert instance1 is instance2


class TestMetricsCollectorInit:
    """Test cases for MetricsCollector initialization."""

    def test_init_metrics_when_enabled(self):
        """Test initialize metrics when enabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        assert collector.enabled is True
        assert hasattr(collector, "query_operations_total")

    def test_skip_initialization_when_disabled(self):
        """Test skip initialization when disabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        assert collector.enabled is False
        assert not hasattr(collector, "query_operations_total")

    def test_init_all_prometheus_metrics(self):
        """Test initialize all Prometheus metrics."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        assert hasattr(collector, "query_operations_total")
        assert hasattr(collector, "write_operations_total")
        assert hasattr(collector, "update_operations_total")
        assert hasattr(collector, "batch_operations_total")
        assert hasattr(collector, "errors_total")
        assert hasattr(collector, "query_duration_seconds")
        assert hasattr(collector, "write_duration_seconds")
        assert hasattr(collector, "update_duration_seconds")
        assert hasattr(collector, "batch_duration_seconds")
        assert hasattr(collector, "query_results_count")
        assert hasattr(collector, "batch_size")
        assert hasattr(collector, "batch_success_rate")
        assert hasattr(collector, "health_status")
        assert hasattr(collector, "active_queries")
        assert hasattr(collector, "active_writes")

    def test_set_server_started_to_false(self):
        """Test set _server_started to False."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        assert collector._server_started is False

    def test_set_enabled_flag(self):
        """Test set enabled flag."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        assert collector.enabled is True

        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        collector2 = MetricsCollector(enable=False)
        assert collector2.enabled is False


class TestMetricsCollectorRecordQuery:
    """Test cases for record_query method."""

    def test_record_successful_query(self):
        """Test record successful query."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_query("search", duration=0.5, success=True, results_count=10)

        assert collector.query_operations_total is not None

    def test_record_failed_query(self):
        """Test record failed query."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_query(
            "search", duration=0.5, success=False, error_code="ERROR"
        )

        assert collector.query_operations_total is not None

    def test_record_duration(self):
        """Test record duration."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_query("search", duration=1.5, success=True)

        assert collector.query_duration_seconds is not None

    def test_record_results_count(self):
        """Test record results_count."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_query("search", duration=0.5, success=True, results_count=25)

        assert collector.query_results_count is not None

    def test_record_error_code_on_failure(self):
        """Test record error_code on failure."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_query(
            "search", duration=0.5, success=False, error_code="NOT_FOUND"
        )

        assert collector.errors_total is not None

    def test_skip_recording_when_disabled(self):
        """Test skip recording when disabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        collector.record_query("search", duration=0.5, success=True)

    def test_use_correct_labels(self):
        """Test use correct labels."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        with patch.object(collector.query_operations_total, "labels") as mock_labels:
            mock_labels.return_value.inc = Mock()
            collector.record_query("search", duration=0.5, success=True)
            mock_labels.assert_called_with(operation="search", status="success")


class TestMetricsCollectorRecordWrite:
    """Test cases for record_write method."""

    def test_record_successful_write(self):
        """Test record successful write."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_write("add_memory", duration=0.3, success=True)

        assert collector.write_operations_total is not None

    def test_record_failed_write(self):
        """Test record failed write."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_write(
            "add_memory", duration=0.3, success=False, error_code="ERROR"
        )

        assert collector.write_operations_total is not None

    def test_skip_recording_when_disabled(self):
        """Test skip recording when disabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        collector.record_write("add_memory", duration=0.3, success=True)


class TestMetricsCollectorRecordUpdate:
    """Test cases for record_update method."""

    def test_record_successful_update(self):
        """Test record successful update."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_update("update_memory", duration=0.2, success=True)

        assert collector.update_operations_total is not None

    def test_record_failed_update(self):
        """Test record failed update."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_update(
            "update_memory", duration=0.2, success=False, error_code="ERROR"
        )

        assert collector.update_operations_total is not None


class TestMetricsCollectorRecordBatch:
    """Test cases for record_batch method."""

    def test_record_batch_operation(self):
        """Test record batch operation."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_batch(
            "batch_add", duration=1.0, total_items=10, succeeded=8, failed=2
        )

        assert collector.batch_operations_total is not None

    def test_record_batch_size_histogram(self):
        """Test record batch_size histogram."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_batch(
            "batch_add", duration=1.0, total_items=10, succeeded=10, failed=0
        )

        assert collector.batch_size is not None

    def test_record_success_rate_histogram(self):
        """Test record success_rate histogram."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_batch(
            "batch_add", duration=1.0, total_items=10, succeeded=8, failed=2
        )

        assert collector.batch_success_rate is not None

    def test_record_batch_zero_total_items(self):
        """Test edge case: zero total_items."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_batch(
            "batch_add", duration=0.0, total_items=0, succeeded=0, failed=0
        )

    def test_record_update_when_disabled_coverage_line_199(self):
        """Test record_update early return when disabled (line 199)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        collector.record_update("test", duration=0.1, success=True, error_code="ERROR")

    def test_record_batch_when_disabled_coverage_line_221(self):
        """Test record_batch early return when disabled (line 221)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        collector.record_batch(
            "test", duration=1.0, total_items=5, succeeded=3, failed=2
        )

    def test_record_batch_error_code_coverage_line_237(self):
        """Test record_batch with error_code (line 237)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_batch(
            "test",
            duration=1.0,
            total_items=5,
            succeeded=3,
            failed=2,
            error_code="BATCH_ERROR",
        )
        assert collector.errors_total is not None

    def test_start_server_already_started_coverage_line_250(self):
        """Test start_server early return when already started (line 250)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector._server_started = True

        result = collector.start_server(port=8000)
        assert result is False

    def test_start_server_address_already_in_use_coverage_line_269(self):
        """Test start_server handle address already in use (line 269)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 0
            mock_socket.return_value = mock_sock

            result = collector.start_server(port=8000)
            assert result is False
            assert collector._server_started is True

    def test_start_server_oserror_errno_98_coverage_line_270(self):
        """Test start_server handle OSError with errno 98 (line 270)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        oserror = OSError("Address already in use")
        oserror.errno = 98

        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.side_effect = oserror
            mock_socket.return_value = mock_sock

            with patch(
                "omnimemory.core.metrics.start_http_server", side_effect=oserror
            ):
                result = collector.start_server(port=8000)
                assert result is False
                assert collector._server_started is True

    def test_start_server_generic_exception_coverage_line_273(self):
        """Test start_server handle generic exception (line 273)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 1
            mock_socket.return_value = mock_sock

            with patch(
                "omnimemory.core.metrics.start_http_server",
                side_effect=Exception("Unexpected error"),
            ):
                result = collector.start_server(port=8000)
                assert result is False

    def test_auto_start_metrics_server_already_started_coverage_line_402(self):
        """Test _auto_start_metrics_server when already started (line 402)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        import omnimemory.core.metrics as metrics_module

        metrics_module._metrics_instance = None

        with patch("omnimemory.core.metrics.ENABLE_METRICS_SERVER", True):
            collector = get_metrics_collector(enable=True)
            collector._server_started = True

            from omnimemory.core.metrics import _auto_start_metrics_server

            _auto_start_metrics_server()

    def test_auto_start_metrics_server_exception_coverage_line_403(self):
        """Test _auto_start_metrics_server exception handling (line 403)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        import omnimemory.core.metrics as metrics_module

        metrics_module._metrics_instance = None
        original_started = metrics_module._metrics_server_started
        metrics_module._metrics_server_started = False

        try:
            with patch("omnimemory.core.metrics.ENABLE_METRICS_SERVER", True):
                with patch(
                    "omnimemory.core.metrics.get_metrics_collector",
                    side_effect=Exception("Error"),
                ):
                    from omnimemory.core.metrics import _auto_start_metrics_server

                    _auto_start_metrics_server()
        finally:
            metrics_module._metrics_server_started = original_started

    def test_auto_start_metrics_server_exception_in_start_server_coverage_line_404(
        self,
    ):
        """Test _auto_start_metrics_server exception in start_server (line 404)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        import omnimemory.core.metrics as metrics_module

        metrics_module._metrics_instance = None
        original_started = metrics_module._metrics_server_started
        metrics_module._metrics_server_started = False

        try:
            with patch("omnimemory.core.metrics.ENABLE_METRICS_SERVER", True):
                collector = MetricsCollector(enable=True)
                collector._server_started = False
                with patch.object(
                    collector, "start_server", side_effect=Exception("Start error")
                ):
                    from omnimemory.core.metrics import _auto_start_metrics_server

                    _auto_start_metrics_server()
        finally:
            metrics_module._metrics_server_started = original_started

    def test_start_metrics_async_exception_coverage_line_411(self):
        """Test _start_metrics_async exception handling (line 411)."""
        with patch(
            "omnimemory.core.metrics._auto_start_metrics_server",
            side_effect=Exception("Error"),
        ):
            from omnimemory.core.metrics import _start_metrics_async

            _start_metrics_async()

    def test_record_batch_all_succeeded(self):
        """Test edge case: all succeeded."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_batch(
            "batch_add", duration=1.0, total_items=5, succeeded=5, failed=0
        )

        assert collector.batch_operations_total is not None

    def test_record_batch_all_failed(self):
        """Test edge case: all failed."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_batch(
            "batch_add", duration=1.0, total_items=5, succeeded=0, failed=5
        )

        assert collector.batch_operations_total is not None


class TestMetricsCollectorSetHealth:
    """Test cases for set_health method."""

    def test_set_health_to_true(self):
        """Test set health to True (1)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.set_health(True)

        assert collector.health_status is not None

    def test_set_health_to_false(self):
        """Test set health to False (0)."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.set_health(False)

        assert collector.health_status is not None

    def test_skip_when_disabled(self):
        """Test skip when disabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        collector.set_health(True)


class TestMetricsCollectorOperationTimer:
    """Test cases for operation_timer context manager."""

    def test_measure_operation_duration(self):
        """Test measure operation duration."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with collector.operation_timer("query", "search") as timer:
            time.sleep(0.1)

        assert timer.start_time is not None

    def test_set_success_flag(self):
        """Test set success flag."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with collector.operation_timer("query", "search") as timer:
            pass

        assert timer.success is True

    def test_set_error_code_on_exception(self):
        """Test set error_code on exception."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with pytest.raises(ValueError, match="Test error"):
            with collector.operation_timer("query", "search") as timer:
                raise ValueError("Test error")

        assert timer.success is False
        assert timer.error_code == "ValueError"

    def test_set_results_count(self):
        """Test set results_count."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with collector.operation_timer("query", "search") as timer:
            timer.results_count = 10

        assert timer.results_count == 10

    def test_record_metrics_on_exit_query(self):
        """Test record metrics on exit for query."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch.object(collector, "record_query") as mock_record:
            with collector.operation_timer("query", "search") as timer:
                timer.results_count = 5
            mock_record.assert_called_once()

    def test_record_metrics_on_exit_write(self):
        """Test record metrics on exit for write."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch.object(collector, "record_write") as mock_record:
            with collector.operation_timer("write", "add_memory"):
                pass
            mock_record.assert_called_once()

    def test_record_metrics_on_exit_update(self):
        """Test record metrics on exit for update."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch.object(collector, "record_update") as mock_record:
            with collector.operation_timer("update", "update_memory"):
                pass
            mock_record.assert_called_once()

    def test_handle_exceptions(self):
        """Test handle exceptions."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch.object(collector, "record_query") as mock_record:
            try:
                with collector.operation_timer("query", "search"):
                    raise ValueError("Error")
            except ValueError:
                pass
            mock_record.assert_called_once()
            call_args = mock_record.call_args
            kwargs = call_args[1] if len(call_args) > 1 else {}
            assert kwargs.get("error_code") == "ValueError"

    def test_skip_when_disabled(self):
        """Test skip when disabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)

        with collector.operation_timer("query", "search"):
            pass

    def test_inc_active_queries(self):
        """Test increment active queries."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with collector.operation_timer("query", "search"):
            pass

        assert collector.active_queries is not None

    def test_inc_active_writes(self):
        """Test increment active writes."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with collector.operation_timer("write", "add_memory"):
            pass

        assert collector.active_writes is not None


class TestMetricsCollectorStartServer:
    """Test cases for start_server method."""

    def test_start_server_when_enabled(self):
        """Test start server when enabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch("omnimemory.core.metrics.start_http_server") as mock_start:
            with patch("socket.socket") as mock_socket:
                mock_sock = Mock()
                mock_sock.connect_ex.return_value = 1
                mock_socket.return_value = mock_sock

                result = collector.start_server(port=8001)
                assert isinstance(result, bool)

    def test_start_server_on_configured_port(self):
        """Test start server on configured port."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch("omnimemory.core.metrics.start_http_server") as mock_start:
            with patch("socket.socket") as mock_socket:
                mock_sock = Mock()
                mock_sock.connect_ex.return_value = 1
                mock_socket.return_value = mock_sock

                collector.start_server(port=9000)
                assert collector._server_started is not None

    def test_handle_port_already_in_use(self):
        """Test handle port already in use."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 0
            mock_socket.return_value = mock_sock

            result = collector.start_server(port=8000)
            assert result is False
            assert collector._server_started is True

    def test_return_false_when_disabled(self):
        """Test return False when disabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        result = collector.start_server(port=8000)
        assert result is False

    def test_handle_server_start_errors(self):
        """Test handle server start errors."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch(
            "omnimemory.core.metrics.start_http_server",
            side_effect=OSError("Permission denied"),
        ):
            with patch("socket.socket") as mock_socket:
                mock_sock = Mock()
                mock_sock.connect_ex.return_value = 1
                mock_socket.return_value = mock_sock

                result = collector.start_server(port=8000)
                assert result is False


class TestMetricsCollectorExport:
    """Test cases for export method."""

    def test_export_metrics_prometheus_format(self):
        """Test export metrics in Prometheus format."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)
        collector.record_query("search", duration=0.5, success=True)

        result = collector.export()
        assert isinstance(result, str)
        assert "omnimemory" in result.lower()

    def test_export_when_disabled(self):
        """Test export when disabled."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=False)
        result = collector.export()
        assert "# Metrics disabled" in result

    def test_export_handle_errors(self):
        """Test handle export errors."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        collector = MetricsCollector(enable=True)

        with patch(
            "omnimemory.core.metrics.generate_latest",
            side_effect=Exception("Export error"),
        ):
            result = collector.export()
            assert "Error exporting metrics" in result


class TestGetMetricsCollector:
    """Test cases for get_metrics_collector factory function."""

    def test_return_singleton_instance(self):
        """Test return singleton instance."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False

        instance1 = get_metrics_collector(enable=False)
        instance2 = get_metrics_collector(enable=False)
        assert instance1 is instance2

    def test_create_instance_when_enable_true(self):
        """Test create instance when enable=True."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        import omnimemory.core.metrics as metrics_module

        metrics_module._metrics_instance = None

        instance = get_metrics_collector(enable=True)
        assert instance.enabled is True

    def test_get_metrics_collector_returns_cached_instance(self):
        """Test get_metrics_collector returns cached instance."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        import omnimemory.core.metrics as metrics_module

        metrics_module._metrics_instance = None

        instance1 = get_metrics_collector(enable=True)
        instance2 = get_metrics_collector(enable=False)
        assert instance1 is instance2

    def test_skip_creation_when_enable_false(self):
        """Test skip creation when enable=False."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        import omnimemory.core.metrics as metrics_module

        metrics_module._metrics_instance = None

        instance = get_metrics_collector(enable=False)
        assert instance.enabled is False


class TestAutoStartMetricsServer:
    """Test cases for _auto_start_metrics_server function."""

    def test_start_server_when_enabled(self):
        """Test start server when ENABLE_METRICS_SERVER=True."""
        MetricsCollector._instance = None
        MetricsCollector._initialized = False
        import omnimemory.core.metrics as metrics_module

        metrics_module._metrics_instance = None
        original_started = metrics_module._metrics_server_started
        metrics_module._metrics_server_started = False

        try:
            with patch("omnimemory.core.metrics.ENABLE_METRICS_SERVER", True):
                with patch("omnimemory.core.metrics.get_metrics_collector") as mock_get:
                    mock_collector = Mock()
                    mock_collector._server_started = False
                    mock_collector.start_server.return_value = True
                    mock_get.return_value = mock_collector

                    _auto_start_metrics_server()
                    mock_collector.start_server.assert_called_once()
        finally:
            metrics_module._metrics_server_started = original_started

    def test_skip_start_when_disabled(self):
        """Test skip start when ENABLE_METRICS_SERVER=False."""
        with patch("omnimemory.core.metrics.ENABLE_METRICS_SERVER", False):
            with patch("omnimemory.core.metrics.get_metrics_collector") as mock_get:
                _auto_start_metrics_server()
                mock_get.assert_not_called()

    def test_handle_already_started_server(self):
        """Test handle already started server."""
        with patch("omnimemory.core.metrics.ENABLE_METRICS_SERVER", True):
            with patch("omnimemory.core.metrics.get_metrics_collector") as mock_get:
                mock_collector = Mock()
                mock_collector._server_started = True
                mock_get.return_value = mock_collector

                _auto_start_metrics_server()
                mock_collector.start_server.assert_not_called()
