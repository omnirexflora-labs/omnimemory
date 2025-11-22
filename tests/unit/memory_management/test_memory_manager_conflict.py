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

    return manager, handler, conflict_agent, synthesis_agent


@pytest.mark.asyncio
async def test_execute_conflict_resolution_all_skip(monkeypatch, mock_llm_connection):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "SKIP"},
            {"memory_id": "m2", "operation": "SKIP"},
        ]
    )
    manager._execute_batch_skip = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is True
    manager._execute_batch_skip.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_conflict_resolution_all_delete(monkeypatch, mock_llm_connection):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "DELETE"},
            {"memory_id": "m2", "operation": "DELETE"},
        ]
    )
    manager._execute_batch_delete = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is True
    manager._execute_batch_delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_conflict_resolution_all_update(monkeypatch, mock_llm_connection):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "UPDATE"},
            {"memory_id": "m2", "operation": "UPDATE"},
        ]
    )
    manager._execute_batch_update = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is True
    manager._execute_batch_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_conflict_resolution_mixed_operations(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "UPDATE"},
            {"memory_id": "m2", "operation": "DELETE"},
            {"memory_id": "m3", "operation": "SKIP"},
        ]
    )
    manager._execute_batch_update = AsyncMock(return_value=Mock(success=True))
    manager._execute_batch_delete = AsyncMock(return_value=Mock(success=True))
    manager._execute_batch_skip = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
        {"memory_id": "m3", "memory_note": "old3"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is True
    manager._execute_batch_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_conflict_resolution_delete_and_skip(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "DELETE"},
            {"memory_id": "m2", "operation": "SKIP"},
            {"operation": "DELETE"},
        ]
    )
    manager._execute_batch_delete = AsyncMock(return_value=Mock(success=True))
    manager._execute_batch_skip = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is True
    manager._execute_batch_delete.assert_awaited_once()
    manager._execute_batch_skip.assert_awaited_once()
    assert invalid


@pytest.mark.asyncio
async def test_execute_conflict_resolution_delete_failure_in_mixed(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "DELETE"},
            {"memory_id": "m2", "operation": "SKIP"},
        ]
    )
    manager._execute_batch_delete = AsyncMock(return_value=Mock(success=False))
    manager._execute_batch_skip = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is False
    manager._execute_batch_delete.assert_awaited_once()
    assert invalid == []


@pytest.mark.asyncio
async def test_execute_conflict_resolution_no_decisions(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(return_value=[])
    manager.update_memory_timestamp = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is True
    assert manager.update_memory_timestamp.await_count == 1


@pytest.mark.asyncio
async def test_execute_conflict_resolution_invalid_decisions(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "INVALID"},
            {},
            {"memory_id": "m2"},
        ]
    )
    manager.update_memory_timestamp = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert len(invalid) > 0


@pytest.mark.asyncio
async def test_execute_conflict_resolution_respects_max_links(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[{"memory_id": f"m{i}", "operation": "UPDATE"} for i in range(10)]
    )
    manager._execute_batch_update = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": f"m{i}", "memory_note": f"old{i}"} for i in range(10)
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links[:5], "app1"
    )

    assert success is True
    call_args = manager._execute_batch_update.await_args
    assert call_args is not None


@pytest.mark.asyncio
async def test_execute_conflict_resolution_update_fails(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[{"memory_id": "m1", "operation": "UPDATE"}]
    )
    manager._execute_batch_update = AsyncMock(return_value=Mock(success=False))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [{"memory_id": "m1", "memory_note": "old1"}]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is False


@pytest.mark.asyncio
async def test_execute_conflict_resolution_delete_fails(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[{"memory_id": "m1", "operation": "DELETE"}]
    )
    manager._execute_batch_delete = AsyncMock(return_value=Mock(success=False))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [{"memory_id": "m1", "memory_note": "old1"}]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is False


@pytest.mark.asyncio
async def test_execute_conflict_resolution_mixed_with_update_failure(
    monkeypatch, mock_llm_connection
):
    manager, _, conflict_agent, _ = _make_manager(monkeypatch, mock_llm_connection)
    conflict_agent.decide = AsyncMock(
        return_value=[
            {"memory_id": "m1", "operation": "UPDATE"},
            {"memory_id": "m2", "operation": "DELETE"},
        ]
    )
    manager._execute_batch_update = AsyncMock(return_value=Mock(success=False))
    manager._execute_batch_delete = AsyncMock(return_value=Mock(success=True))

    memory_data = {"natural_memory_note": "test"}
    meaningful_links = [
        {"memory_id": "m1", "memory_note": "old1"},
        {"memory_id": "m2", "memory_note": "old2"},
    ]

    success, invalid = await manager._execute_conflict_resolution(
        memory_data, meaningful_links, "app1"
    )

    assert success is False
