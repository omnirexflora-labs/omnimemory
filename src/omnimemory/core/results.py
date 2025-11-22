"""
Structured result types for memory operations.
Defines MemoryOperationResult and BatchOperationResult for consistent API responses.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class MemoryOperationResult:
    """Structured result for memory operations with detailed error context."""

    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    memory_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls, memory_id: str = None, **details
    ) -> "MemoryOperationResult":
        """Create a successful result."""
        return cls(success=True, memory_id=memory_id, details=details)

    @classmethod
    def error_result(
        cls, error_code: str, error_message: str, **details
    ) -> "MemoryOperationResult":
        """Create an error result."""
        return cls(
            success=False,
            error_code=error_code,
            error_message=error_message,
            details=details,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "success": self.success,
        }
        if self.memory_id:
            result["memory_id"] = self.memory_id
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_message:
            result["error_message"] = self.error_message
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class BatchOperationResult:
    """Result for batch operations with per-item success/failure tracking."""

    success: bool
    total_items: int
    succeeded: int
    failed: int
    failed_items: List[Dict[str, Any]] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: List[MemoryOperationResult],
        operation_name: str = "batch_operation",
    ) -> "BatchOperationResult":
        """Create batch result from individual operation results."""
        total = len(results)
        succeeded = sum(1 for r in results if r.success)
        failed = total - succeeded

        failed_items = [
            {
                "memory_id": r.memory_id,
                "error_code": r.error_code,
                "error_message": r.error_message,
            }
            for r in results
            if not r.success
        ]

        return cls(
            success=failed == 0,
            total_items=total,
            succeeded=succeeded,
            failed=failed,
            failed_items=failed_items,
            error_message=f"{operation_name}: {succeeded}/{total} succeeded"
            if failed > 0
            else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "success": self.success,
            "total_items": self.total_items,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }
        if self.failed_items:
            result["failed_items"] = self.failed_items
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_message:
            result["error_message"] = self.error_message
        if self.details:
            result["details"] = self.details
        return result
