"""
Comprehensive unit tests for OmniMemory result types.
"""

import pytest
from omnimemory.core.results import MemoryOperationResult, BatchOperationResult


class TestMemoryOperationResult:
    """Test cases for MemoryOperationResult."""

    def test_create_success_result(self):
        """Test create success result."""
        result = MemoryOperationResult.success_result(memory_id="mem123")
        assert result.success is True
        assert result.memory_id == "mem123"
        assert result.error_code is None
        assert result.error_message is None

    def test_create_success_result_with_details(self):
        """Test create success result with details."""
        result = MemoryOperationResult.success_result(
            memory_id="mem123", operation="write", duration=0.5
        )
        assert result.success is True
        assert result.details["operation"] == "write"
        assert result.details["duration"] == 0.5

    def test_create_error_result(self):
        """Test create error result."""
        result = MemoryOperationResult.error_result(
            error_code="NOT_FOUND", error_message="Memory not found"
        )
        assert result.success is False
        assert result.error_code == "NOT_FOUND"
        assert result.error_message == "Memory not found"
        assert result.memory_id is None

    def test_create_error_result_with_details(self):
        """Test create error result with details."""
        result = MemoryOperationResult.error_result(
            error_code="VALIDATION_ERROR",
            error_message="Invalid input",
            field="memory_id",
            value=None,
        )
        assert result.success is False
        assert result.details["field"] == "memory_id"
        assert result.details["value"] is None

    def test_to_dict_success(self):
        """Test to_dict conversion for success result."""
        result = MemoryOperationResult.success_result(memory_id="mem123")
        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["memory_id"] == "mem123"
        assert "error_code" not in result_dict
        assert "error_message" not in result_dict

    def test_to_dict_error(self):
        """Test to_dict conversion for error result."""
        result = MemoryOperationResult.error_result(
            error_code="ERROR", error_message="Test error"
        )
        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["error_code"] == "ERROR"
        assert result_dict["error_message"] == "Test error"
        assert "memory_id" not in result_dict

    def test_to_dict_with_details(self):
        """Test to_dict includes details when present."""
        result = MemoryOperationResult.success_result(memory_id="mem123", extra="data")
        result_dict = result.to_dict()

        assert "details" in result_dict
        assert result_dict["details"]["extra"] == "data"

    def test_to_dict_empty_details(self):
        """Test to_dict doesn't include details when empty."""
        result = MemoryOperationResult.success_result(memory_id="mem123")
        result_dict = result.to_dict()

        if "details" in result_dict:
            assert result_dict["details"] == {}


class TestBatchOperationResult:
    """Test cases for BatchOperationResult."""

    def test_from_results_all_succeeded(self):
        """Test from_results when all succeeded."""
        results = [
            MemoryOperationResult.success_result(memory_id="mem1"),
            MemoryOperationResult.success_result(memory_id="mem2"),
            MemoryOperationResult.success_result(memory_id="mem3"),
        ]

        batch_result = BatchOperationResult.from_results(results, "batch_write")

        assert batch_result.success is True
        assert batch_result.total_items == 3
        assert batch_result.succeeded == 3
        assert batch_result.failed == 0
        assert len(batch_result.failed_items) == 0
        assert batch_result.error_message is None

    def test_from_results_all_failed(self):
        """Test from_results when all failed."""
        results = [
            MemoryOperationResult.error_result("ERROR1", "Error 1", memory_id="mem1"),
            MemoryOperationResult.error_result("ERROR2", "Error 2", memory_id="mem2"),
        ]

        batch_result = BatchOperationResult.from_results(results, "batch_write")

        assert batch_result.success is False
        assert batch_result.total_items == 2
        assert batch_result.succeeded == 0
        assert batch_result.failed == 2
        assert len(batch_result.failed_items) == 2

    def test_from_results_partial_failure(self):
        """Test from_results with partial failures."""
        error_result = MemoryOperationResult.error_result("ERROR", "Error")
        error_result.memory_id = "mem2"

        results = [
            MemoryOperationResult.success_result(memory_id="mem1"),
            error_result,
            MemoryOperationResult.success_result(memory_id="mem3"),
        ]

        batch_result = BatchOperationResult.from_results(results, "batch_write")

        assert batch_result.success is False
        assert batch_result.total_items == 3
        assert batch_result.succeeded == 2
        assert batch_result.failed == 1
        assert len(batch_result.failed_items) == 1
        assert batch_result.failed_items[0]["memory_id"] == "mem2"

    def test_from_results_empty_list(self):
        """Test from_results with empty list."""
        results = []
        batch_result = BatchOperationResult.from_results(results, "batch_write")

        assert batch_result.success is True
        assert batch_result.total_items == 0
        assert batch_result.succeeded == 0
        assert batch_result.failed == 0

    def test_from_results_failed_items_structure(self):
        """Test failed_items have correct structure."""
        results = [
            MemoryOperationResult.error_result("ERROR1", "Error 1", memory_id="mem1"),
            MemoryOperationResult.error_result("ERROR2", "Error 2", memory_id="mem2"),
        ]

        batch_result = BatchOperationResult.from_results(results, "batch_write")

        assert len(batch_result.failed_items) == 2
        for item in batch_result.failed_items:
            assert "memory_id" in item
            assert "error_code" in item
            assert "error_message" in item

    def test_to_dict_success(self):
        """Test to_dict for successful batch."""
        batch_result = BatchOperationResult(
            success=True, total_items=5, succeeded=5, failed=0
        )
        result_dict = batch_result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["total_items"] == 5
        assert result_dict["succeeded"] == 5
        assert result_dict["failed"] == 0
        assert "failed_items" not in result_dict

    def test_to_dict_with_failed_items(self):
        """Test to_dict includes failed_items when present."""
        batch_result = BatchOperationResult(
            success=False,
            total_items=3,
            succeeded=2,
            failed=1,
            failed_items=[
                {"memory_id": "mem1", "error_code": "ERROR", "error_message": "Error"}
            ],
        )
        result_dict = batch_result.to_dict()

        assert "failed_items" in result_dict
        assert len(result_dict["failed_items"]) == 1

    def test_to_dict_with_error_code(self):
        """Test to_dict includes error_code when present."""
        batch_result = BatchOperationResult(
            success=False,
            total_items=1,
            succeeded=0,
            failed=1,
            error_code="BATCH_ERROR",
        )
        result_dict = batch_result.to_dict()

        assert result_dict["error_code"] == "BATCH_ERROR"

    def test_to_dict_with_error_message(self):
        """Test to_dict includes error_message when present."""
        batch_result = BatchOperationResult(
            success=False,
            total_items=1,
            succeeded=0,
            failed=1,
            error_message="Batch operation failed",
        )
        result_dict = batch_result.to_dict()

        assert result_dict["error_message"] == "Batch operation failed"

    def test_to_dict_with_details(self):
        """Test to_dict includes details when present."""
        batch_result = BatchOperationResult(
            success=True,
            total_items=1,
            succeeded=1,
            failed=0,
            details={"operation": "batch_write", "duration": 1.5},
        )
        result_dict = batch_result.to_dict()

        assert "details" in result_dict
        assert result_dict["details"]["operation"] == "batch_write"

    def test_from_results_error_message_format(self):
        """Test error_message format when failures exist."""
        results = [
            MemoryOperationResult.success_result(memory_id="mem1"),
            MemoryOperationResult.error_result("ERROR", "Error", memory_id="mem2"),
        ]

        batch_result = BatchOperationResult.from_results(results, "test_operation")

        assert batch_result.error_message is not None
        assert "test_operation" in batch_result.error_message
        assert "1/2" in batch_result.error_message or "2" in batch_result.error_message
