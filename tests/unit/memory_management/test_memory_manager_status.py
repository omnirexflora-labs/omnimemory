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
async def test_update_memory_status_success(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=True)

    result = await manager.update_memory_status(
        app_id="app1", doc_id="doc1", new_status="updated"
    )

    assert result.success is True
    handler.update_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_memory_status_with_reason(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=True)

    result = await manager.update_memory_status(
        app_id="app1",
        doc_id="doc1",
        new_status="updated",
        archive_reason="consolidated",
        new_memory_id="doc2",
    )

    assert result.success is True
    call_kwargs = handler.update_memory.await_args.kwargs
    assert call_kwargs["update_payload"]["status_reason"] == "consolidated"
    assert call_kwargs["update_payload"]["next_id"] == "doc2"


@pytest.mark.asyncio
async def test_update_memory_status_handler_disabled(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    result = await manager.update_memory_status(
        app_id="app1", doc_id="doc1", new_status="updated"
    )

    assert result.success is False
    assert result.error_code == "VECTOR_DB_DISABLED"


@pytest.mark.asyncio
async def test_update_memory_status_not_found(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value=None)

    result = await manager.update_memory_status(
        app_id="app1", doc_id="doc1", new_status="updated"
    )

    assert result.success is False
    assert result.error_code == "MEMORY_NOT_FOUND"


@pytest.mark.asyncio
async def test_update_memory_status_update_fails(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=False)

    result = await manager.update_memory_status(
        app_id="app1", doc_id="doc1", new_status="updated"
    )

    assert result.success is False
    assert result.error_code == "UPDATE_STATUS_FAILED"


@pytest.mark.asyncio
async def test_update_memory_status_exception(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(side_effect=RuntimeError("db error"))

    result = await manager.update_memory_status(
        app_id="app1", doc_id="doc1", new_status="updated"
    )

    assert result.success is False
    assert result.error_code == "UPDATE_STATUS_EXCEPTION"


@pytest.mark.asyncio
async def test_update_memory_timestamp_success(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=True)

    result = await manager.update_memory_timestamp(app_id="app1", doc_id="doc1")

    assert result.success is True
    handler.update_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_memory_timestamp_handler_disabled(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    result = await manager.update_memory_timestamp(app_id="app1", doc_id="doc1")

    assert result.success is False
    assert result.error_code == "VECTOR_DB_DISABLED"


@pytest.mark.asyncio
async def test_update_memory_timestamp_not_found(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value=None)

    result = await manager.update_memory_timestamp(app_id="app1", doc_id="doc1")

    assert result.success is False
    assert result.error_code == "MEMORY_NOT_FOUND"


@pytest.mark.asyncio
async def test_update_memory_timestamp_update_fails(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=False)

    result = await manager.update_memory_timestamp(app_id="app1", doc_id="doc1")

    assert result.success is False
    assert result.error_code == "UPDATE_TIMESTAMP_FAILED"


@pytest.mark.asyncio
async def test_update_memory_timestamp_exception(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(side_effect=RuntimeError("db error"))

    result = await manager.update_memory_timestamp(app_id="app1", doc_id="doc1")

    assert result.success is False
    assert result.error_code == "UPDATE_TIMESTAMP_EXCEPTION"


@pytest.mark.asyncio
async def test_update_memory_status_with_contradicted_reason(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=True)

    result = await manager.update_memory_status(
        app_id="app1",
        doc_id="doc1",
        new_status="deleted",
        caused_by_memory="doc2",
    )

    assert result.success is True
    call_kwargs = handler.update_memory.await_args.kwargs
    assert call_kwargs["update_payload"]["status_reason"] == "contradicted"


@pytest.mark.asyncio
async def test_update_memory_status_with_consolidated_reason(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=True)

    result = await manager.update_memory_status(
        app_id="app1",
        doc_id="doc1",
        new_status="updated",
    )

    assert result.success is True
    call_kwargs = handler.update_memory.await_args.kwargs
    assert call_kwargs["update_payload"]["status_reason"] == "consolidated"


@pytest.mark.asyncio
async def test_update_memory_status_with_manual_reason(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_id = AsyncMock(return_value={"memory_id": "doc1", "metadata": {}})
    handler.update_memory = AsyncMock(return_value=True)

    result = await manager.update_memory_status(
        app_id="app1",
        doc_id="doc1",
        new_status="active",
    )

    assert result.success is True
    call_kwargs = handler.update_memory.await_args.kwargs
    assert call_kwargs["update_payload"]["status_reason"] == "manual_update"
