import pytest
from unittest.mock import AsyncMock, Mock

from omnimemory.core.results import MemoryOperationResult
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
            self.results_count = 0

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

    return MemoryManager(mock_llm_connection), handler, metrics


@pytest.mark.asyncio
async def test_embed_memory_note_success(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)
    handler._embed_text_with_chunking_async = AsyncMock(return_value=[0.1, 0.2, 0.3])

    result = await manager.embed_memory_note("test note")

    assert result == [0.1, 0.2, 0.3]
    handler._embed_text_with_chunking_async.assert_awaited_once_with("test note")


@pytest.mark.asyncio
async def test_embed_memory_note_raises_on_empty_string(
    monkeypatch, mock_llm_connection
):
    manager, _, _ = _make_manager(monkeypatch, mock_llm_connection)

    with pytest.raises(ValueError, match="Memory note text must be a non-empty string"):
        await manager.embed_memory_note("")

    with pytest.raises(
        ValueError, match="Memory note text cannot be empty or whitespace only"
    ):
        await manager.embed_memory_note("   ")


@pytest.mark.asyncio
async def test_embed_memory_note_raises_on_none(monkeypatch, mock_llm_connection):
    manager, _, _ = _make_manager(monkeypatch, mock_llm_connection)

    with pytest.raises(ValueError, match="Memory note text must be a non-empty string"):
        await manager.embed_memory_note(None)


@pytest.mark.asyncio
async def test_embed_memory_note_raises_on_handler_disabled(
    monkeypatch, mock_llm_connection
):
    manager, handler, _ = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    with pytest.raises(RuntimeError, match="Vector database handler is not available"):
        await manager.embed_memory_note("test")


@pytest.mark.asyncio
async def test_embed_memory_note_propagates_exception(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)
    handler._embed_text_with_chunking_async = AsyncMock(
        side_effect=RuntimeError("embedding failed")
    )

    with pytest.raises(RuntimeError, match="Memory note embedding failed"):
        await manager.embed_memory_note("test note")


@pytest.mark.asyncio
async def test_store_memory_note_success(monkeypatch, mock_llm_connection):
    manager, handler, metrics = _make_manager(monkeypatch, mock_llm_connection)
    handler.add_to_collection = AsyncMock(return_value=True)

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="user1",
        memory_note_text="test note",
        embedding=[0.1, 0.2, 0.3],
    )

    assert result.success is True
    assert result.memory_id == "doc1"
    handler.add_to_collection.assert_awaited_once()
    assert metrics.operation_timer.called


@pytest.mark.asyncio
async def test_store_memory_note_with_optional_fields(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)
    handler.add_to_collection = AsyncMock(return_value=True)

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="user1",
        memory_note_text="test note",
        embedding=[0.1, 0.2],
        session_id="session1",
        tags=["tag1"],
        keywords=["kw1"],
        semantic_queries=["query1"],
        conversation_complexity=3,
        interaction_quality="high",
        follow_up_potential=["follow1"],
    )

    assert result.success is True
    call_args = handler.add_to_collection.await_args
    metadata = call_args.kwargs["metadata"]
    assert metadata["tags"] == ["tag1"]
    assert metadata["keywords"] == ["kw1"]
    assert metadata["semantic_queries"] == ["query1"]
    assert metadata["conversation_complexity"] == 3
    assert metadata["interaction_quality"] == "high"
    assert metadata["follow_up_potential"] == ["follow1"]


@pytest.mark.asyncio
async def test_store_memory_note_invalid_app_id(monkeypatch, mock_llm_connection):
    manager, _, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="",
        user_id="user1",
        memory_note_text="test",
        embedding=[0.1],
    )

    assert result.success is False
    assert result.error_code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_store_memory_note_invalid_user_id(monkeypatch, mock_llm_connection):
    manager, _, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="",
        memory_note_text="test",
        embedding=[0.1],
    )

    assert result.success is False
    assert result.error_code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_store_memory_note_invalid_memory_note(monkeypatch, mock_llm_connection):
    manager, _, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="user1",
        memory_note_text="",
        embedding=[0.1],
    )

    assert result.success is False
    assert result.error_code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_store_memory_note_invalid_embedding(monkeypatch, mock_llm_connection):
    manager, _, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="user1",
        memory_note_text="test",
        embedding=[],
    )

    assert result.success is False
    assert result.error_code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_store_memory_note_handler_disabled(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="user1",
        memory_note_text="test",
        embedding=[0.1],
    )

    assert result.success is False
    assert result.error_code == "VECTOR_DB_DISABLED"


@pytest.mark.asyncio
async def test_store_memory_note_add_fails(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)
    handler.add_to_collection = AsyncMock(return_value=False)

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="user1",
        memory_note_text="test",
        embedding=[0.1],
    )

    assert result.success is False
    assert result.error_code == "STORE_FAILED"


@pytest.mark.asyncio
async def test_store_memory_note_exception(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)
    handler.add_to_collection = AsyncMock(side_effect=RuntimeError("db error"))

    result = await manager.store_memory_note(
        doc_id="doc1",
        app_id="app1",
        user_id="user1",
        memory_note_text="test",
        embedding=[0.1],
    )

    assert result.success is False
    assert result.error_code == "STORE_EXCEPTION"


@pytest.mark.asyncio
async def test_store_memory_directly_success(monkeypatch, mock_llm_connection):
    manager, handler, metrics = _make_manager(monkeypatch, mock_llm_connection)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="doc1")
    )

    memory_data = {
        "doc_id": "doc1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "natural_memory_note": "test doc",
        "embedding": [0.1, 0.2],
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "follow_up_potential": [],
        "status": "active",
    }

    result = await manager._store_memory_directly(memory_data)

    assert result.success is True
    manager.store_memory_note.assert_awaited_once()


@pytest.mark.asyncio
async def test_store_memory_directly_missing_handler(monkeypatch, mock_llm_connection):
    manager, _, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.error_result(
            error_code="VECTOR_DB_DISABLED",
            error_message="Vector DB disabled",
            memory_id="doc1",
        )
    )

    memory_data = {
        "doc_id": "doc1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "natural_memory_note": "test",
        "embedding": [0.1],
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "follow_up_potential": [],
        "status": "active",
    }

    result = await manager._store_memory_directly(memory_data)

    assert result.success is False


@pytest.mark.asyncio
async def test_store_memory_directly_add_fails(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.error_result(
            error_code="STORE_FAILED", error_message="Store failed", memory_id="doc1"
        )
    )

    memory_data = {
        "doc_id": "doc1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "natural_memory_note": "test",
        "embedding": [0.1],
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "follow_up_potential": [],
        "status": "active",
    }

    result = await manager._store_memory_directly(memory_data)

    assert result.success is False


@pytest.mark.asyncio
async def test_store_memory_directly_exception(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.store_memory_note = AsyncMock(side_effect=RuntimeError("error"))

    memory_data = {
        "doc_id": "doc1",
        "app_id": "app1",
        "user_id": "user1",
        "session_id": None,
        "natural_memory_note": "test",
        "embedding": [0.1],
        "retrieval_tags": [],
        "retrieval_keywords": [],
        "semantic_queries": [],
        "conversation_complexity": None,
        "interaction_quality": None,
        "follow_up_potential": [],
        "status": "active",
    }

    with pytest.raises(RuntimeError):
        await manager._store_memory_directly(memory_data)
