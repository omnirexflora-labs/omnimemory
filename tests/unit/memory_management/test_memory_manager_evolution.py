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

    return MemoryManager(mock_llm_connection), handler


@pytest.mark.asyncio
async def test_traverse_memory_evolution_chain_success(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    def mock_get_memory(memory_id, app_id):
        memories = {
            "m1": {
                "memory_id": "m1",
                "document": "doc1",
                "metadata": {"next_id": "m2", "status": "active"},
            },
            "m2": {
                "memory_id": "m2",
                "document": "doc2",
                "metadata": {"next_id": None, "status": "active"},
            },
        }
        return memories.get(memory_id)

    manager.get_memory = AsyncMock(side_effect=mock_get_memory)

    result = await manager.traverse_memory_evolution_chain(
        app_id="app1", memory_id="m1"
    )

    assert len(result) == 2
    assert result[0]["memory_id"] == "m1"
    assert result[1]["memory_id"] == "m2"


@pytest.mark.asyncio
async def test_traverse_memory_evolution_chain_detects_cycle(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    def mock_get_memory(memory_id, app_id):
        memories = {
            "m1": {
                "memory_id": "m1",
                "document": "doc1",
                "metadata": {"next_id": "m2", "status": "active"},
            },
            "m2": {
                "memory_id": "m2",
                "document": "doc2",
                "metadata": {"next_id": "m1", "status": "active"},
            },
        }
        return memories.get(memory_id)

    manager.get_memory = AsyncMock(side_effect=mock_get_memory)

    result = await manager.traverse_memory_evolution_chain(
        app_id="app1", memory_id="m1"
    )

    assert len(result) == 2


@pytest.mark.asyncio
async def test_traverse_memory_evolution_chain_stops_on_missing(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    def mock_get_memory(memory_id, app_id):
        if memory_id == "m1":
            return {
                "memory_id": "m1",
                "document": "doc1",
                "metadata": {"next_id": "m2", "status": "active"},
            }
        return None

    manager.get_memory = AsyncMock(side_effect=mock_get_memory)

    result = await manager.traverse_memory_evolution_chain(
        app_id="app1", memory_id="m1"
    )

    assert len(result) == 1
    assert result[0]["memory_id"] == "m1"


@pytest.mark.asyncio
async def test_traverse_memory_evolution_chain_empty_on_not_found(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.get_memory = AsyncMock(return_value=None)

    result = await manager.traverse_memory_evolution_chain(
        app_id="app1", memory_id="m1"
    )

    assert result == []


@pytest.mark.asyncio
async def test_traverse_memory_evolution_chain_single_memory(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    manager.get_memory = AsyncMock(
        return_value={
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {"next_id": None, "status": "active"},
        }
    )

    result = await manager.traverse_memory_evolution_chain(
        app_id="app1", memory_id="m1"
    )

    assert len(result) == 1
    assert result[0]["memory_id"] == "m1"


def test_generate_evolution_graph_mermaid(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "next_id": "m2",
            },
        },
        {
            "memory_id": "m2",
            "document": "doc2",
            "metadata": {
                "status": "updated",
                "created_at": "2024-01-02T00:00:00Z",
                "next_id": None,
            },
        },
    ]

    result = manager.generate_evolution_graph(chain, format="mermaid")

    assert "graph LR" in result
    assert "M0" in result
    assert "M1" in result
    assert "evolves to" in result


def test_generate_evolution_graph_dot(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {"status": "active", "created_at": "2024-01-01T00:00:00Z"},
        },
    ]

    result = manager.generate_evolution_graph(chain, format="dot")

    assert "digraph" in result
    assert "MemoryEvolution" in result


def test_generate_evolution_graph_html(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {"status": "active", "created_at": "2024-01-01T00:00:00Z"},
        },
    ]

    result = manager.generate_evolution_graph(chain, format="html")

    assert "<!DOCTYPE html>" in result
    assert "mermaid" in result.lower()


def test_generate_evolution_graph_invalid_format(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [{"memory_id": "m1", "metadata": {}}]

    with pytest.raises(ValueError, match="Unsupported format"):
        manager.generate_evolution_graph(chain, format="invalid")


def test_generate_evolution_graph_empty_chain(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = manager.generate_evolution_graph([], format="mermaid")

    assert result == ""


def test_generate_dot_graph_handles_digit_ids(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    class ExplodingDate:
        def __str__(self):
            raise ValueError("boom")

    chain = [
        {
            "memory_id": "12345678901234567890",
            "document": "old",
            "metadata": {"status": "deleted", "created_at": ExplodingDate()},
        },
        {
            "memory_id": "abc123",
            "document": "doc2",
            "metadata": {"status": "updated", "created_at": "2024-01-02"},
        },
        {
            "memory_id": "9xyz",
            "document": "doc3",
            "metadata": {"status": "active", "created_at": "2024-01-03"},
        },
    ]

    result = manager._generate_dot_graph(chain)

    assert "M_" in result
    assert 'fillcolor="#cce5ff"' in result
    assert "->" in result


def test_generate_evolution_graph_dot_with_multiple_memories(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "next_id": "m2",
            },
        },
        {
            "memory_id": "m2",
            "document": "doc2",
            "metadata": {
                "status": "updated",
                "created_at": "2024-01-02T00:00:00Z",
                "next_id": None,
            },
        },
    ]

    result = manager.generate_evolution_graph(chain, format="dot")

    assert "digraph" in result
    assert "m1" in result
    assert "m2" in result


def test_generate_text_report_handles_bad_date_and_next(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    class ExplodingDate:
        def __str__(self):
            raise ValueError("boom")

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {
                "status": "active",
                "created_at": ExplodingDate(),
                "next_id": "next1234567890",
            },
        },
        {
            "memory_id": "m2",
            "document": "doc2",
            "metadata": {
                "status": "updated",
                "created_at": "2024-01-02",
                "next_id": None,
            },
        },
    ]

    report = manager._generate_text_report(chain)

    assert "Evolves to: next1234567890" in report
    assert "(End of chain)" in report


def test_generate_evolution_graph_html_with_multiple_memories(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {"status": "active", "created_at": "2024-01-01T00:00:00Z"},
        },
        {
            "memory_id": "m2",
            "document": "doc2",
            "metadata": {"status": "updated", "created_at": "2024-01-02T00:00:00Z"},
        },
    ]

    result = manager.generate_evolution_graph(chain, format="html")

    assert "<!DOCTYPE html>" in result
    assert "mermaid" in result.lower()
    assert "m1" in result
    assert "m2" in result


def test_generate_mermaid_graph_handles_bad_date(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    class ExplodingDate:
        def __str__(self):
            raise ValueError("boom")

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {
                "status": "active",
                "created_at": ExplodingDate(),
                "next_id": None,
            },
        }
    ]

    result = manager.generate_evolution_graph(chain, format="mermaid")

    assert "graph LR" in result
    assert "m1" in result


def test_generate_evolution_graph_handles_long_ids(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m12345678901234567890",
            "document": "doc1",
            "metadata": {
                "status": "active",
                "created_at": "2024-01-01",
                "next_id": "m2",
            },
        },
        {
            "memory_id": "m2",
            "document": "doc2",
            "metadata": {
                "status": "updated",
                "created_at": "not-a-date",
            },
        },
    ]

    result = manager.generate_evolution_graph(chain, format="mermaid")

    assert "m1234567" in result
    assert "activeStyle" in result


def test_generate_evolution_report_text_with_multiple_memories(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["tag1"],
                "keywords": ["kw1"],
            },
        },
        {
            "memory_id": "m2",
            "document": "doc2",
            "metadata": {
                "status": "updated",
                "created_at": "2024-01-02T00:00:00Z",
                "tags": ["tag2"],
            },
        },
    ]

    result = manager.generate_evolution_report(chain, format="text")

    assert "MEMORY EVOLUTION CHAIN REPORT" in result
    assert "m1" in result
    assert "m2" in result


def test_generate_evolution_report_markdown(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {
                "status": "active",
                "created_at": "2024-01-01T00:00:00Z",
                "next_id": "m2",
                "tags": ["tag1"],
            },
        },
        {
            "memory_id": "m2",
            "document": "doc2",
            "metadata": {
                "status": "updated",
                "created_at": "2024-01-02T00:00:00Z",
                "next_id": None,
            },
        },
    ]

    result = manager.generate_evolution_report(chain, format="markdown")

    assert "# Memory Evolution Chain Report" in result
    assert "Total Memories in Chain" in result
    assert "tag1" in result


def test_generate_markdown_report_single_memory_standalone(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    class ExplodingDate:
        def __str__(self):
            raise ValueError("boom")

    chain = [
        {
            "memory_id": "m1",
            "document": "",
            "metadata": {
                "status": "active",
                "created_at": ExplodingDate(),
                "updated_at": ExplodingDate(),
                "next_id": None,
            },
        }
    ]

    report = manager._generate_markdown_report(chain)

    assert "standalone memory" in report


def test_generate_evolution_report_markdown_covers_dates(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    class BadDate:
        def __str__(self):
            raise ValueError("boom")

    chain = [
        {
            "memory_id": "m1234567890123456",
            "document": "doc",
            "metadata": {
                "status": "active",
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-02T11:00:00Z",
                "next_id": "next1234567890",
                "tags": ["t1"],
                "keywords": ["k1"],
            },
        },
        {
            "memory_id": "m2",
            "document": "",
            "metadata": {
                "status": "updated",
                "created_at": BadDate(),
                "updated_at": "2024-01-04",
            },
        },
    ]

    report = manager._generate_markdown_report(chain)

    assert "Time Span" in report
    assert "Evolves to" in report
    assert "t1" in report


def test_generate_evolution_report_text(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {"status": "active", "created_at": "2024-01-01T00:00:00Z"},
        },
    ]

    result = manager.generate_evolution_report(chain, format="text")

    assert "MEMORY EVOLUTION CHAIN REPORT" in result
    assert "SUMMARY STATISTICS" in result


def test_generate_evolution_report_json(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [
        {
            "memory_id": "m1",
            "document": "doc1",
            "metadata": {"status": "active", "created_at": "2024-01-01T00:00:00Z"},
        },
    ]

    result = manager.generate_evolution_report(chain, format="json")

    import json

    parsed = json.loads(result)
    assert "summary" in parsed
    assert "timeline" in parsed


def test_generate_evolution_report_invalid_format(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    chain = [{"memory_id": "m1", "metadata": {}}]

    with pytest.raises(ValueError, match="Unsupported format"):
        manager.generate_evolution_report(chain, format="invalid")


def test_generate_evolution_report_empty_chain(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    result = manager.generate_evolution_report([], format="markdown")

    assert result == ""
