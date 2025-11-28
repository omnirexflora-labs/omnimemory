import pytest
from unittest.mock import AsyncMock, Mock

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

    return MemoryManager(mock_llm_connection), handler


@pytest.mark.asyncio
async def test_add_memory_success(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.add_to_collection = AsyncMock(return_value=True)

    result = await manager.add_memory(
        collection_name="app1",
        doc_id="doc1",
        document="test doc",
        embedding=[0.1, 0.2],
        metadata={"app_id": "app1"},
    )

    assert result.success is True
    assert result.memory_id == "doc1"
    handler.add_to_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_memory_fails(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.add_to_collection = AsyncMock(return_value=False)

    result = await manager.add_memory(
        collection_name="app1",
        doc_id="doc1",
        document="test",
        embedding=[0.1],
        metadata={},
    )

    assert result.success is False
    assert result.error_code == "ADD_FAILED"


@pytest.mark.asyncio
async def test_add_memory_exception(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.add_to_collection = AsyncMock(side_effect=RuntimeError("db error"))

    result = await manager.add_memory(
        collection_name="app1",
        doc_id="doc1",
        document="test",
        embedding=[0.1],
        metadata={},
    )

    assert result.success is False
    assert result.error_code == "ADD_EXCEPTION"


@pytest.mark.asyncio
async def test_get_memory_success(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"document": "test", "metadata": {}})

    result = await manager.get_memory(memory_id="doc1", app_id="app1")

    assert result == {"document": "test", "metadata": {}}
    handler.query_by_id.assert_awaited_once_with(collection_name="app1", doc_id="doc1")


@pytest.mark.asyncio
async def test_get_memory_handler_disabled(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    result = await manager.get_memory(memory_id="doc1", app_id="app1")

    assert result is None


@pytest.mark.asyncio
async def test_get_memory_exception(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(side_effect=RuntimeError("error"))

    result = await manager.get_memory(memory_id="doc1", app_id="app1")

    assert result is None


@pytest.mark.asyncio
async def test_get_memory_not_found(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value=None)

    result = await manager.get_memory(memory_id="doc1", app_id="app1")

    assert result is None


@pytest.mark.asyncio
async def test_delete_memory_success(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.delete_from_collection = AsyncMock(return_value=True)

    result = await manager.delete_memory(doc_id="doc1", collection_name="app1")

    assert result.success is True
    assert result.memory_id == "doc1"
    handler.delete_from_collection.assert_awaited_once_with(
        collection_name="app1", doc_id="doc1"
    )


@pytest.mark.asyncio
async def test_delete_memory_fails(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.delete_from_collection = AsyncMock(return_value=False)

    result = await manager.delete_memory(doc_id="doc1", collection_name="app1")

    assert result.success is False
    assert result.error_code == "DELETE_FAILED"


@pytest.mark.asyncio
async def test_delete_memory_exception(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.delete_from_collection = AsyncMock(side_effect=RuntimeError("db error"))

    result = await manager.delete_memory(doc_id="doc1", collection_name="app1")

    assert result.success is False
    assert result.error_code == "DELETE_EXCEPTION"


@pytest.mark.asyncio
async def test_delete_memory_invalid_doc_id(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager.delete_memory(doc_id="", collection_name="app1")

    assert result.success is False
    assert result.error_code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_delete_memory_invalid_collection_name(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = await manager.delete_memory(doc_id="doc1", collection_name="")

    assert result.success is False
    assert result.error_code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_delete_memory_handler_disabled(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    result = await manager.delete_memory(doc_id="doc1", collection_name="app1")

    assert result.success is False
    assert result.error_code == "VECTOR_DB_DISABLED"
