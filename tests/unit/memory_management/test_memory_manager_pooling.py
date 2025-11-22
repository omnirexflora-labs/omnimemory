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

    return MemoryManager(mock_llm_connection), handler, fake_pool


@pytest.mark.asyncio
async def test_warm_up_connection_pool_success(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=True)
    )

    result = await manager.warm_up_connection_pool()

    assert result is True


@pytest.mark.asyncio
async def test_warm_up_connection_pool_handler_disabled(
    monkeypatch, mock_llm_connection
):
    manager, handler, _ = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    result = await manager.warm_up_connection_pool()

    assert result is False


@pytest.mark.asyncio
async def test_warm_up_connection_pool_handler_none(monkeypatch, mock_llm_connection):
    fake_ctx = AsyncMock()
    fake_ctx.__aenter__.return_value = None
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

    manager = MemoryManager(mock_llm_connection)

    result = await manager.warm_up_connection_pool()

    assert result is False


@pytest.mark.asyncio
async def test_warm_up_connection_pool_exception(monkeypatch, mock_llm_connection):
    manager, _, pool = _make_manager(monkeypatch, mock_llm_connection)
    pool.get_handler = Mock(side_effect=RuntimeError("pool error"))

    result = await manager.warm_up_connection_pool()

    assert result is False


@pytest.mark.asyncio
async def test_get_pooled_handler_success(monkeypatch, mock_llm_connection):
    manager, handler, _ = _make_manager(monkeypatch, mock_llm_connection)

    async with manager._get_pooled_handler() as h:
        assert h == handler


@pytest.mark.asyncio
async def test_get_pooled_handler_raises_without_llm(monkeypatch):
    fake_pool = Mock()
    fake_ctx = AsyncMock()
    fake_ctx.__aenter__.return_value = Mock(enabled=True)
    fake_ctx.__aexit__.return_value = False
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

    manager = MemoryManager(None)

    async with manager._get_pooled_handler() as h:
        assert h is not None
