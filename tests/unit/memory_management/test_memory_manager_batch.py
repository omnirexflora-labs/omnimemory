import pytest
from unittest.mock import AsyncMock, Mock

from omnimemory.core.results import BatchOperationResult, MemoryOperationResult
from omnimemory.memory_management.memory_manager import MemoryManager


def _make_manager(monkeypatch, mock_llm_connection, handler=None):
    if handler is None:
        handler = Mock(enabled=True)

    fake_ctx = AsyncMock()
    fake_ctx.__aenter__.return_value = handler
    fake_ctx.__aexit__.return_value = False

    fake_pool = Mock()
    fake_pool.get_handler.return_value = fake_ctx

    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.VectorDBHandlerPool.get_instance",
        lambda max_connections=None: fake_pool,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.ConflictResolutionAgent",
        lambda llm: Mock(),
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.SynthesisAgent",
        lambda llm: Mock(),
    )

    class Timer:
        def __init__(self):
            self.success = True
            self.error_code = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                self.success = False
                self.error_code = exc_type.__name__
            return False

    metrics = Mock()
    metrics.operation_timer.return_value = Timer()
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.get_metrics_collector",
        lambda *args, **kwargs: metrics,
    )

    return MemoryManager(mock_llm_connection), handler


@pytest.mark.asyncio
async def test_execute_batch_update_empty_decisions(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager._execute_batch_update([], {}, [])

    assert result.success is True
    assert result.total_items == 0


@pytest.mark.asyncio
async def test_execute_batch_update_no_existing_memories(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    decisions = [{"memory_id": "m1", "operation": "UPDATE"}]
    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [{"memory_id": "m2", "document": "doc2", "composite_score": 0.9}]

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert result.success is False
    assert result.error_code == "NO_EXISTING_MEMORIES"


@pytest.mark.asyncio
async def test_execute_batch_update_success(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    manager.synthesis_agent = Mock()
    manager.synthesis_agent.consolidate_memories = AsyncMock(
        return_value={
            "consolidated_memory": {"natural_memory_note": "consolidated"},
            "synthesis_summary": "summary",
        }
    )
    handler.update_memory = AsyncMock(return_value=True)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="new1")
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2, 0.3])

    decisions = [
        {"memory_id": "m1", "operation": "UPDATE", "reason": "test"},
        {"memory_id": "m2", "operation": "UPDATE", "reason": "test"},
    ]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "status": "active",
    }
    meaningful_links = [
        {"memory_id": "m1", "document": "old1", "composite_score": 0.9},
        {"memory_id": "m2", "document": "old2", "composite_score": 0.8},
    ]

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert isinstance(result, BatchOperationResult)
    assert result.total_items >= 2


@pytest.mark.asyncio
async def test_execute_batch_update_archives_with_failures(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    manager.synthesis_agent = Mock()
    manager.synthesis_agent.consolidate_memories = AsyncMock(
        return_value={
            "consolidated_memory": {"natural_memory_note": "consolidated"},
            "synthesis_summary": "summary",
        }
    )
    handler.update_memory = AsyncMock(return_value=True)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="new1")
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2, 0.3])
    manager._run_status_updates_chunked = AsyncMock(
        return_value=[
            MemoryOperationResult.success_result(memory_id="m1"),
            MemoryOperationResult.error_result(
                error_code="STATUS_FAIL",
                error_message="fail",
                memory_id="m2",
            ),
        ]
    )

    decisions = [
        {"memory_id": "m1", "operation": "UPDATE"},
        {"memory_id": "m2", "operation": "UPDATE"},
    ]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "status": "active",
    }
    meaningful_links = [
        {"memory_id": "m1", "document": "old1", "composite_score": 0.9},
        {"memory_id": "m2", "document": "old2", "composite_score": 0.8},
    ]

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert isinstance(result, BatchOperationResult)
    assert result.failed == 1
    assert result.details["new_memory_id"] == "new1"
    manager._run_status_updates_chunked.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_batch_update_handles_invalid_complexity(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    manager.synthesis_agent = Mock()
    manager.synthesis_agent.consolidate_memories = AsyncMock(
        return_value={
            "consolidated_memory": {"natural_memory_note": "combined"},
            "synthesis_summary": "summary",
        }
    )
    handler.update_memory = AsyncMock(return_value=True)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="new1")
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2])
    manager._run_status_updates_chunked = AsyncMock(
        return_value=[MemoryOperationResult.success_result(memory_id="m1")]
    )

    decisions = [{"memory_id": "m1", "operation": "UPDATE"}]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "status": "active",
    }
    meaningful_links = [
        {
            "memory_id": "m1",
            "document": "old1",
            "composite_score": 0.9,
            "conversation_complexity": None,
            "interaction_quality": None,
        }
    ]

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert result.success is True
    kwargs = manager.store_memory_note.await_args.kwargs
    assert kwargs["conversation_complexity"] == 1
    assert kwargs["interaction_quality"] == "low"


@pytest.mark.asyncio
async def test_execute_batch_update_limits_meaningful_links(
    monkeypatch, mock_llm_connection
):
    from omnimemory.memory_management.memory_manager import MAX_LINKS_FOR_SYNTHESIS

    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    manager.synthesis_agent = Mock()
    manager.synthesis_agent.consolidate_memories = AsyncMock(
        return_value={
            "consolidated_memory": {"natural_memory_note": "summary note"},
            "synthesis_summary": "summary",
        }
    )
    handler.update_memory = AsyncMock(return_value=True)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="new1")
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2, 0.3])
    manager._run_status_updates_chunked = AsyncMock(
        return_value=[
            MemoryOperationResult.success_result(memory_id=f"m{i}")
            for i in range(MAX_LINKS_FOR_SYNTHESIS)
        ]
    )

    decisions = [{"memory_id": f"m{i}", "operation": "UPDATE"} for i in range(6)]
    meaningful_links = [
        {
            "memory_id": f"m{i}",
            "document": f"doc{i}",
            "composite_score": 1 - (i * 0.1),
        }
        for i in range(6)
    ]

    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": 2,
        "interaction_quality": "medium",
        "status": "active",
    }

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert result.success is True
    existing = manager.synthesis_agent.consolidate_memories.await_args.kwargs[
        "existing_memories"
    ]
    assert len(existing) == MAX_LINKS_FOR_SYNTHESIS


@pytest.mark.asyncio
async def test_execute_batch_update_embedding_fails(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.synthesis_agent = Mock()
    manager.synthesis_agent.consolidate_memories = AsyncMock(
        return_value={
            "consolidated_memory": {"natural_memory_note": "consolidated"},
            "synthesis_summary": "summary",
        }
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    decisions = [{"memory_id": "m1", "operation": "UPDATE"}]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "status": "active",
    }
    meaningful_links = [{"memory_id": "m1", "document": "old1", "composite_score": 0.9}]

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_execute_batch_update_store_fails(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.synthesis_agent = Mock()
    manager.synthesis_agent.consolidate_memories = AsyncMock(
        return_value={
            "consolidated_memory": {"natural_memory_note": "consolidated"},
            "synthesis_summary": "summary",
        }
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2])
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.error_result(
            error_code="STORE_FAILED", error_message="Store failed", memory_id="new1"
        )
    )

    decisions = [{"memory_id": "m1", "operation": "UPDATE"}]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "status": "active",
    }
    meaningful_links = [{"memory_id": "m1", "document": "old1", "composite_score": 0.9}]

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert result.success is False
    assert result.error_code == "STORE_FAILED"


@pytest.mark.asyncio
async def test_execute_batch_update_exception(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.synthesis_agent = Mock()
    manager.synthesis_agent.consolidate_memories = AsyncMock(
        side_effect=RuntimeError("synthesis error")
    )

    decisions = [{"memory_id": "m1", "operation": "UPDATE"}]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "status": "active",
    }
    meaningful_links = [{"memory_id": "m1", "document": "old1", "composite_score": 0.9}]

    result = await manager._execute_batch_update(
        decisions, memory_data, meaningful_links
    )

    assert result.success is False
    assert result.error_code == "BATCH_UPDATE_EXCEPTION"


@pytest.mark.asyncio
async def test_execute_batch_delete_success(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.delete_from_collection = AsyncMock(return_value=True)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="new1")
    )

    decisions = [
        {"memory_id": "m1", "operation": "DELETE"},
        {"memory_id": "m2", "operation": "DELETE"},
    ]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "embedding": [0.1, 0.2, 0.3],
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "status": "active",
    }

    result = await manager._execute_batch_delete(decisions, memory_data)

    assert isinstance(result, BatchOperationResult)
    assert result.total_items >= 2


@pytest.mark.asyncio
async def test_execute_batch_delete_empty_decisions(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager._execute_batch_delete([], {})

    assert result.success is True
    assert result.total_items == 0


@pytest.mark.asyncio
async def test_execute_batch_delete_store_fails(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.error_result(
            error_code="STORE_FAILED", error_message="Store failed", memory_id="new1"
        )
    )

    decisions = [{"memory_id": "m1", "operation": "DELETE"}]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "embedding": [0.1, 0.2, 0.3],
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "status": "active",
    }

    result = await manager._execute_batch_delete(decisions, memory_data)

    assert result.success is False
    assert result.error_code == "STORE_FAILED"


@pytest.mark.asyncio
async def test_execute_batch_delete_exception(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.store_memory_note = AsyncMock(side_effect=RuntimeError("store error"))

    decisions = [{"memory_id": "m1", "operation": "DELETE"}]
    memory_data = {
        "natural_memory_note": "new note",
        "doc_id": "new1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "embedding": [0.1, 0.2],
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "status": "active",
    }

    result = await manager._execute_batch_delete(decisions, memory_data)

    assert result.success is False
    assert result.error_code == "BATCH_DELETE_EXCEPTION"


@pytest.mark.asyncio
async def test_execute_batch_skip_success(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.update_memory_timestamp = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="m1")
    )

    decisions = [
        {"memory_id": "m1", "operation": "SKIP"},
        {"memory_id": "m2", "operation": "SKIP"},
    ]

    result = await manager._execute_batch_skip(decisions, "app1")

    assert isinstance(result, BatchOperationResult)
    assert result.total_items == 2


@pytest.mark.asyncio
async def test_execute_batch_skip_empty_decisions(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager._execute_batch_skip([], "app1")

    assert result.success is True
    assert result.total_items == 0


@pytest.mark.asyncio
async def test_execute_batch_skip_exception(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._run_timestamp_updates_chunked = AsyncMock(
        side_effect=RuntimeError("timestamp error")
    )

    decisions = [{"memory_id": "m1", "operation": "SKIP"}]

    result = await manager._execute_batch_skip(decisions, "app1")

    assert result.success is False
    assert result.error_code == "BATCH_SKIP_EXCEPTION"


@pytest.mark.asyncio
async def test_run_status_updates_chunked_success(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.update_memory_status = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="m1")
    )

    update_specs = [
        {"app_id": "app1", "doc_id": "m1", "new_status": "updated"},
        {"app_id": "app1", "doc_id": "m2", "new_status": "updated"},
        {"app_id": "app1", "doc_id": "m3", "new_status": "updated"},
    ]

    result = await manager._run_status_updates_chunked(update_specs)

    assert len(result) == 3
    assert all(r.success for r in result)
    assert manager.update_memory_status.await_count == 3


@pytest.mark.asyncio
async def test_run_timestamp_updates_chunked_success(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.update_memory_timestamp = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="m1")
    )

    update_specs = [
        {"app_id": "app1", "doc_id": "m1"},
        {"app_id": "app1", "doc_id": "m2"},
        {"app_id": "app1", "doc_id": "m3"},
    ]

    result = await manager._run_timestamp_updates_chunked(update_specs)

    assert len(result) == 3
    assert all(r.success for r in result)
    assert manager.update_memory_timestamp.await_count == 3


@pytest.mark.asyncio
async def test_run_status_updates_chunked_partial_failure(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    def mock_update(app_id, doc_id, **kwargs):
        if doc_id == "m2":
            return MemoryOperationResult.error_result(
                error_code="ERROR", memory_id=doc_id
            )
        return MemoryOperationResult.success_result(memory_id=doc_id)

    manager.update_memory_status = AsyncMock(side_effect=mock_update)

    update_specs = [
        {"app_id": "app1", "doc_id": "m1", "new_status": "updated"},
        {"app_id": "app1", "doc_id": "m2", "new_status": "updated"},
        {"app_id": "app1", "doc_id": "m3", "new_status": "updated"},
    ]

    result = await manager._run_status_updates_chunked(update_specs)

    assert len(result) == 3
    assert sum(1 for r in result if r.success) == 2
    assert sum(1 for r in result if not r.success) == 1


@pytest.mark.asyncio
async def test_run_timestamp_updates_chunked_with_exception(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    def mock_update(**kwargs):
        if kwargs.get("doc_id") == "m2":
            raise RuntimeError("update error")
        return MemoryOperationResult.success_result(memory_id=kwargs.get("doc_id"))

    manager.update_memory_timestamp = AsyncMock(side_effect=mock_update)

    update_specs = [
        {"app_id": "app1", "doc_id": "m1"},
        {"app_id": "app1", "doc_id": "m2"},
        {"app_id": "app1", "doc_id": "m3"},
    ]

    result = await manager._run_timestamp_updates_chunked(update_specs)

    assert len(result) == 3
    assert sum(1 for r in result if r.success) == 2
    assert sum(1 for r in result if not r.success) == 1
    assert any(
        r.error_code == "UPDATE_TIMESTAMP_EXCEPTION" for r in result if not r.success
    )
