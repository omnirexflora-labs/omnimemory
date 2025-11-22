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

    conflict_agent = Mock()
    synthesis_agent = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.ConflictResolutionAgent",
        lambda llm: conflict_agent,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.SynthesisAgent",
        lambda llm: synthesis_agent,
    )

    manager = MemoryManager(mock_llm_connection)
    manager.conflict_resolution_agent = conflict_agent
    manager.synthesis_agent = synthesis_agent

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

    return manager, handler


@pytest.mark.asyncio
async def test_create_and_store_memory_direct_path(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        return_value=(
            {"natural_memory_note": "test note", "retrieval_tags": []},
            None,
        )
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2, 0.3])
    manager._find_meaningful_links = AsyncMock(return_value=[])
    manager._store_memory_directly = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="doc1")
    )

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test messages",
        llm_connection=mock_llm_connection,
    )

    assert result.success is True
    manager._store_memory_directly.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_and_store_memory_with_conflict_resolution(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        return_value=(
            {"natural_memory_note": "test note", "retrieval_tags": []},
            None,
        )
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2, 0.3])
    manager._find_meaningful_links = AsyncMock(
        return_value=[{"memory_id": "m1", "composite_score": 0.9, "document": "old1"}]
    )
    manager._execute_conflict_resolution = AsyncMock(return_value=(True, []))

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test messages",
        llm_connection=mock_llm_connection,
    )

    assert result.success is True
    manager._execute_conflict_resolution.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_and_store_memory_creation_fails(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        return_value=(None, "Creation failed")
    )

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "MEMORY_NOTE_CREATION_FAILED"


@pytest.mark.asyncio
async def test_create_and_store_memory_embedding_fails(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        return_value=(
            {"natural_memory_note": "test note", "retrieval_tags": []},
            None,
        )
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_and_store_memory_embedding_exception(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        return_value=(
            {"natural_memory_note": "test note", "retrieval_tags": []},
            None,
        )
    )
    manager.embed_memory_note = AsyncMock(side_effect=RuntimeError("embedding error"))

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_EXCEPTION"


@pytest.mark.asyncio
async def test_create_and_store_memory_conflict_resolution_fails(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        return_value=(
            {"natural_memory_note": "test note", "retrieval_tags": []},
            None,
        )
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2])
    manager._find_meaningful_links = AsyncMock(
        return_value=[{"memory_id": "m1", "composite_score": 0.9}]
    )
    manager._execute_conflict_resolution = AsyncMock(return_value=(False, []))

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "CONFLICT_RESOLUTION_FAILED"


@pytest.mark.asyncio
async def test_create_and_store_memory_pipeline_exception(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        side_effect=RuntimeError("pipeline error")
    )

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "PIPELINE_EXCEPTION"


@pytest.mark.asyncio
async def test_create_and_store_memory_limits_links(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager._create_and_parse_memory_note = AsyncMock(
        return_value=(
            {"natural_memory_note": "test note", "retrieval_tags": []},
            None,
        )
    )
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2])
    manager._find_meaningful_links = AsyncMock(
        return_value=[
            {
                "memory_id": f"m{i}",
                "composite_score": 0.9 - i * 0.1,
                "document": f"doc{i}",
            }
            for i in range(10)
        ]
    )
    manager._execute_conflict_resolution = AsyncMock(return_value=(True, []))

    result = await manager.create_and_store_memory(
        app_id="app1",
        user_id="user1",
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is True
    call_args = manager._execute_conflict_resolution.await_args
    meaningful_links_arg = call_args.kwargs.get(
        "meaningful_links", call_args[0][1] if len(call_args[0]) > 1 else []
    )
    assert len(meaningful_links_arg) == 4
