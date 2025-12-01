import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from asyncio import QueueEmpty

from omnimemory.memory_management.connection_pool import VectorDBHandlerPool


@pytest.fixture(autouse=True)
def reset_pool_state():
    """Ensure each test starts with a clean singleton state."""
    VectorDBHandlerPool._instance = None
    VectorDBHandlerPool._lock = None
    VectorDBHandlerPool._class_initialized = False
    yield
    VectorDBHandlerPool._instance = None
    VectorDBHandlerPool._lock = None
    VectorDBHandlerPool._class_initialized = False


def _patch_handler_factory(monkeypatch, handler):
    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.VectorDBFactoryRegistry.create_from_env",
        AsyncMock(return_value=handler),
    )


@pytest.mark.asyncio
async def test_create_handler_disabled_returns_none(monkeypatch):
    _patch_handler_factory(monkeypatch, Mock(enabled=False))
    pool = VectorDBHandlerPool()
    result = await pool._create_handler(Mock())
    assert result is None


@pytest.mark.asyncio
async def test_create_handler_exception_returns_none(monkeypatch):
    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.VectorDBFactoryRegistry.create_from_env",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    pool = VectorDBHandlerPool()
    result = await pool._create_handler(Mock())
    assert result is None


@pytest.mark.asyncio
async def test_initialize_pool_creates_half_of_max_connections(monkeypatch):
    """Test that pool initializes with 50% of max_connections."""
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=10)
    await pool.initialize_pool(Mock())
    # Should create 5 handlers (50% of 10)
    assert pool._pool.qsize() == 5
    assert pool._created_handlers == 5
    assert pool._initialized is True
    assert pool._pool_lock is not None


@pytest.mark.asyncio
async def test_initialize_pool_with_small_max_connections(monkeypatch):
    """Test that pool creates at least 1 handler even with max_connections=1."""
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=1)
    await pool.initialize_pool(Mock())
    # Should create 1 handler (max(1, 1//2) = 1)
    assert pool._pool.qsize() == 1
    assert pool._created_handlers == 1
    assert pool._initialized is True


@pytest.mark.asyncio
async def test_initialize_pool_with_max_connections_2(monkeypatch):
    """Test that pool creates 1 handler when max_connections=2."""
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=2)
    await pool.initialize_pool(Mock())
    # Should create 1 handler (max(1, 2//2) = 1)
    assert pool._pool.qsize() == 1
    assert pool._created_handlers == 1
    assert pool._initialized is True


@pytest.mark.asyncio
async def test_initialize_pool_with_large_max_connections(monkeypatch):
    """Test that pool scales initialization with large max_connections."""
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=30)
    await pool.initialize_pool(Mock())
    # Should create 15 handlers (50% of 30)
    assert pool._pool.qsize() == 15
    assert pool._created_handlers == 15
    assert pool._initialized is True


@pytest.mark.asyncio
async def test_initialize_pool_reuses_singleton(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)

    pool1 = VectorDBHandlerPool(max_connections=2)
    await pool1.initialize_pool(Mock())

    VectorDBHandlerPool._instance = pool1
    VectorDBHandlerPool._class_initialized = True

    pool2 = VectorDBHandlerPool(max_connections=2)
    await pool2.initialize_pool(Mock())

    assert pool2._initialized is True
    assert pool2._pool is pool1._pool


@pytest.mark.asyncio
async def test_get_handler_requires_llm_when_uninitialized():
    pool = VectorDBHandlerPool(max_connections=1)
    with pytest.raises(ValueError):
        async with pool.get_handler():
            pass


@pytest.mark.asyncio
async def test_get_handler_initializes_pool_when_llm_provided(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=1)

    async with pool.get_handler(llm_connection=Mock()) as acquired:
        assert acquired == handler
        assert pool._initialized is True
        assert pool._pool_lock is not None


@pytest.mark.asyncio
async def test_get_handler_reuses_existing_pool_when_class_initialized(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool1 = VectorDBHandlerPool(max_connections=1)
    await pool1.initialize_pool(Mock())

    VectorDBHandlerPool._instance = pool1
    VectorDBHandlerPool._class_initialized = True

    pool2 = VectorDBHandlerPool(max_connections=1)
    async with pool2.get_handler(llm_connection=Mock()) as acquired:
        assert acquired == handler
        assert pool2._initialized is True


@pytest.mark.asyncio
async def test_get_handler_returns_handler_and_requeues(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=1)

    async with pool.get_handler(llm_connection=Mock()) as acquired:
        assert acquired == handler

    assert pool._active_handlers == 0
    assert pool._pool.qsize() == 1


@pytest.mark.asyncio
async def test_get_handler_logs_error_when_usage_fails(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=1)

    with pytest.raises(RuntimeError):
        async with pool.get_handler(llm_connection=Mock()):
            raise RuntimeError("handler failure")


@pytest.mark.asyncio
async def test_get_handler_timeout_when_returning_handler(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=1)
    await pool.initialize_pool(Mock())

    async def failing_put(_):
        raise asyncio.TimeoutError()

    pool._pool.put = failing_put

    async with pool.get_handler(llm_connection=Mock()) as acquired:
        assert acquired == handler

    assert pool._active_handlers == 0


@pytest.mark.asyncio
async def test_get_handler_exception_returning_handler_creates_replacement(monkeypatch):
    handler = Mock(enabled=True)
    replacement = Mock(enabled=True)
    factory = AsyncMock(side_effect=[handler, replacement])
    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.VectorDBFactoryRegistry.create_from_env",
        factory,
    )
    pool = VectorDBHandlerPool(max_connections=1)
    await pool.initialize_pool(Mock())

    call_tracker = {"count": 0}
    original_wait_for = asyncio.wait_for

    async def flaky_wait_for(coro, timeout):
        if (
            hasattr(coro, "__self__")
            and coro.__self__ is pool._pool
            and call_tracker["count"] == 0
        ):
            call_tracker["count"] += 1
            raise asyncio.TimeoutError()
        return await original_wait_for(coro, timeout)

    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.asyncio.wait_for",
        flaky_wait_for,
    )

    async with pool.get_handler(llm_connection=Mock()) as acquired:
        assert acquired == handler

    assert pool._pool.qsize() == 1


@pytest.mark.asyncio
async def test_get_handler_exception_replacement_creation_fails(monkeypatch):
    handler = Mock(enabled=True)
    factory = AsyncMock(side_effect=[handler, RuntimeError("factory boom")])
    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.VectorDBFactoryRegistry.create_from_env",
        factory,
    )

    pool = VectorDBHandlerPool(max_connections=1)
    await pool.initialize_pool(Mock())

    original_wait_for = asyncio.wait_for

    async def always_timeout(coro, timeout):
        if hasattr(coro, "__self__") and coro.__self__ is pool._pool:
            raise asyncio.TimeoutError()
        return await original_wait_for(coro, timeout)

    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.asyncio.wait_for",
        always_timeout,
    )

    async with pool.get_handler(llm_connection=Mock()) as acquired:
        assert acquired == handler

    assert pool._active_handlers == 0


@pytest.mark.asyncio
async def test_acquire_handler_creates_new_when_capacity_available(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=2)
    pool._pool = asyncio.Queue(maxsize=2)
    pool._pool_lock = asyncio.Lock()
    pool._llm_connection = Mock()
    pool._initialized = True

    async def fake_wait_for(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.asyncio.wait_for",
        fake_wait_for,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.asyncio.sleep",
        AsyncMock(return_value=None),
    )

    result = await pool._acquire_handler_with_retry()
    assert result == handler


@pytest.mark.asyncio
async def test_acquire_handler_times_out_when_pool_exhausted(monkeypatch):
    pool = VectorDBHandlerPool(max_connections=1)
    pool._pool = asyncio.Queue(maxsize=1)
    pool._pool_lock = asyncio.Lock()
    pool._created_handlers = 1
    pool._llm_connection = Mock()
    pool._initialized = True

    async def fake_wait_for(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.asyncio.wait_for",
        fake_wait_for,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.connection_pool.asyncio.sleep",
        AsyncMock(return_value=None),
    )

    with pytest.raises(TimeoutError):
        await pool._acquire_handler_with_retry()


@pytest.mark.asyncio
async def test_get_pool_stats_before_initialization():
    pool = VectorDBHandlerPool(max_connections=3)
    stats = await pool.get_pool_stats()
    assert stats["initialized"] is False
    assert stats["available_handlers"] == 0


@pytest.mark.asyncio
async def test_get_pool_stats_after_initialization(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool(max_connections=10)
    await pool.initialize_pool(Mock())

    stats = await pool.get_pool_stats()
    assert stats["initialized"] is True
    assert stats["created_handlers"] == 5  # 50% of 10
    assert stats["available_handlers"] == 5


@pytest.mark.asyncio
async def test_get_dedicated_handler_uses_factory(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool = VectorDBHandlerPool()
    result = await pool.get_dedicated_handler(Mock())
    assert result == handler


@pytest.mark.asyncio
async def test_get_instance_returns_singleton(monkeypatch):
    handler = Mock(enabled=True)
    _patch_handler_factory(monkeypatch, handler)
    pool1 = VectorDBHandlerPool.get_instance(max_connections=1)
    pool2 = VectorDBHandlerPool.get_instance(max_connections=5)
    assert pool1 is pool2


@pytest.mark.asyncio
async def test_close_all_closes_async_and_sync_clients(monkeypatch):
    class Client:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    async def async_close():
        async_close.called = True

    async_close.called = False

    handler1 = Mock()
    handler1.close = async_close

    handler2 = Mock()
    handler2.client = Client()
    handler2.close = None

    pool = VectorDBHandlerPool()
    pool._pool = asyncio.Queue()
    await pool._pool.put(handler1)
    await pool._pool.put(handler2)

    await pool.close_all()

    assert async_close.called is True
    assert handler2.client.closed is True


@pytest.mark.asyncio
async def test_close_all_handles_empty_queue():
    pool = VectorDBHandlerPool()
    pool._pool = asyncio.Queue()
    await pool.close_all()
    assert pool._created_handlers == 0


@pytest.mark.asyncio
async def test_close_all_returns_when_pool_none():
    pool = VectorDBHandlerPool()
    await pool.close_all()


@pytest.mark.asyncio
async def test_close_all_handles_queueempty(monkeypatch):
    class BrokenQueue:
        def __init__(self):
            self.calls = 0

        def empty(self):
            return False if self.calls == 0 else True

        def get_nowait(self):
            self.calls += 1
            raise QueueEmpty()

    pool = VectorDBHandlerPool()
    pool._pool = BrokenQueue()
    await pool.close_all()


@pytest.mark.asyncio
async def test_close_all_handles_async_client_close(monkeypatch):
    client_close = AsyncMock()

    class Handler:
        def __init__(self):
            self.close = None

            class Client:
                async def close(inner_self):
                    await client_close()

            self.client = Client()

    handler = Handler()
    pool = VectorDBHandlerPool()
    pool._pool = asyncio.Queue()
    await pool._pool.put(handler)

    await pool.close_all()

    client_close.assert_awaited()
