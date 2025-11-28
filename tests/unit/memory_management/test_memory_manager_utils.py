import pytest
from unittest.mock import AsyncMock, Mock

from omnimemory.memory_management.memory_manager import MemoryManager


def _make_manager(monkeypatch, mock_llm_connection):
    fake_ctx = AsyncMock()
    fake_ctx.__aenter__.return_value = Mock(enabled=True)
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

    return MemoryManager(mock_llm_connection)


@pytest.mark.asyncio
async def test_create_and_parse_memory_note_success(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test note", "metadata": {}}'
    )

    result, error = await manager._create_and_parse_memory_note(
        "messages", mock_llm_connection
    )

    assert result is not None
    assert result["natural_memory_note"] == "test note"
    assert error is None


@pytest.mark.asyncio
async def test_create_and_parse_memory_note_failure(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(return_value=None)

    result, error = await manager._create_and_parse_memory_note(
        "messages", mock_llm_connection
    )

    assert result is None
    assert error is not None


@pytest.mark.asyncio
async def test_create_and_parse_memory_note_invalid_json(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(return_value="{invalid json")

    result, error = await manager._create_and_parse_memory_note(
        "messages", mock_llm_connection
    )

    assert result is None
    assert "Failed to parse" in error


@pytest.mark.asyncio
async def test_create_and_parse_memory_note_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(return_value=None)

    result, error = await manager._create_and_parse_memory_note(
        "messages", mock_llm_connection
    )

    assert result is None
    assert error is not None


def test_normalize_prepared_memory_note(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)

    result = manager._normalize_prepared_memory_note(
        {"text": "test note", "metadata": {}}
    )

    assert result["natural_memory_note"] == "test note"


def test_normalize_prepared_memory_note_not_dict(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)

    with pytest.raises(ValueError, match="Prepared memory note must be an object"):
        manager._normalize_prepared_memory_note("not a dict")


def test_normalize_prepared_memory_note_missing_text(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)

    with pytest.raises(ValueError, match="Prepared memory note missing 'text'"):
        manager._normalize_prepared_memory_note({"metadata": {}})


def test_depth_to_complexity(monkeypatch, mock_llm_connection):
    assert MemoryManager._depth_to_complexity("low") == 1
    assert MemoryManager._depth_to_complexity("medium") == 2
    assert MemoryManager._depth_to_complexity("high") == 3
    assert MemoryManager._depth_to_complexity(None) is None
    assert MemoryManager._depth_to_complexity("unknown") is None


@pytest.mark.asyncio
async def test_extract_memory_metadata(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)

    memory_note_data = {
        "natural_memory_note": "test",
        "retrieval_tags": ["tag1"],
        "retrieval_keywords": ["kw1"],
        "semantic_queries": ["q1"],
        "conversation_complexity": 2,
        "interaction_quality": "high",
        "follow_up_potential": ["f1"],
    }

    result = manager._extract_memory_metadata(memory_note_data)

    assert result["retrieval_tags"] == ["tag1"]
    assert result["retrieval_keywords"] == ["kw1"]
    assert result["conversation_complexity"] == 2


def test_build_memory_data_dict(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)

    result = manager._build_memory_data_dict(
        app_id="app1",
        user_id="user1",
        session_id=None,
        embedding=[0.1, 0.2, 0.3],
        natural_memory_note="test note",
        metadata={
            "retrieval_tags": ["tag1"],
            "retrieval_keywords": [],
            "semantic_queries": [],
            "conversation_complexity": None,
            "interaction_quality": None,
            "follow_up_potential": [],
        },
    )

    assert result["natural_memory_note"] == "test note"
    assert result["embedding"] == [0.1, 0.2, 0.3]
    assert result["retrieval_tags"] == ["tag1"]
