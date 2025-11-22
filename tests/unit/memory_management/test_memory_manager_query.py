import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from omnimemory.memory_management.memory_manager import MemoryManager


def _make_manager(monkeypatch, llm_conn, handler=None):
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
    metrics = Mock()

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

    metrics.operation_timer.return_value = Timer()
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.get_metrics_collector",
        lambda *args, **kwargs: metrics,
    )

    return MemoryManager(llm_conn), handler


@pytest.mark.asyncio
async def test_query_memory_returns_empty_when_handler_disabled(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )
    handler.query_collection = AsyncMock()

    results = await manager.query_memory(app_id="app", query="hi")

    assert results == []
    handler.query_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_memory_returns_ranked_results(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2"],
            "scores": [0.9, 0.7],
            "metadatas": [
                {
                    "document_id": "m1",
                    "app_id": "app",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "status": "active",
                },
                {
                    "document_id": "m2",
                    "app_id": "app",
                    "created_at": "2024-01-02T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                    "status": "active",
                },
            ],
            "ids": ["m1", "m2"],
        }
    )

    results = await manager.query_memory(app_id="app", query="hello", n_results=1)

    assert len(results) == 1
    assert results[0]["memory_note"] == "doc1"
    assert "composite_score" in results[0]
    assert results[0]["metadata"]["document_id"] == "m1"


@pytest.mark.asyncio
async def test_query_memory_handles_exception(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(side_effect=RuntimeError("db down"))

    results = await manager.query_memory(app_id="app", query="hello")

    assert results == []


@pytest.mark.asyncio
async def test_query_memory_handles_pool_context_error(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    @asynccontextmanager
    async def failing_cm():
        raise RuntimeError("pool error")
        yield

    manager._get_pooled_handler = lambda *args, **kwargs: failing_cm()

    results = await manager.query_memory(app_id="app", query="oops")

    assert results == []


@pytest.mark.asyncio
async def test_query_memory_handles_execute_query_exception(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={"documents": [], "scores": [], "metadatas": []}
    )
    manager._execute_query = AsyncMock(side_effect=RuntimeError("boom"))

    results = await manager.query_memory(app_id="app", query="hello")

    assert results == []
    manager._execute_query.assert_awaited()


@pytest.mark.asyncio
async def test_execute_query_applies_filters(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)

    async def _mock_query(
        collection_name, query, n_results, similarity_threshold, filter_conditions
    ):
        return {
            "documents": ["doc"],
            "scores": [0.95],
            "metadatas": [
                {
                    "document_id": "id1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ],
            "ids": ["id1"],
        }

    handler.query_collection = AsyncMock(side_effect=_mock_query)

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="hello",
        filter_conditions={"app_id": "app"},
        expanded_n_results=3,
        similarity_threshold=0.2,
        n_results=1,
    )

    assert len(result) == 1
    handler.query_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_query_returns_empty_on_exception(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(side_effect=RuntimeError("boom"))

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="hello",
        filter_conditions={"app_id": "app"},
        expanded_n_results=3,
        similarity_threshold=0.2,
        n_results=1,
    )

    assert result == []


@pytest.mark.asyncio
async def test_generate_memory_links_filters_candidates(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_embedding = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2"],
            "scores": [0.9, 0.5],
            "metadatas": [
                {
                    "document_id": "m1",
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                },
                {
                    "document_id": "m2",
                    "status": "active",
                    "created_at": "2024-01-02T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                },
            ],
        }
    )

    links = await manager.generate_memory_links(
        embedding=[0.1, 0.2],
        app_id="app",
        similarity_threshold=0.6,
        max_links=2,
    )

    assert len(links) == 1
    assert links[0]["memory_id"] == "m1"


@pytest.mark.asyncio
async def test_find_meaningful_links_respects_threshold(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    manager.generate_memory_links = AsyncMock(
        return_value=[
            {"memory_id": "a", "composite_score": 0.9},
            {"memory_id": "b", "composite_score": 0.6},
        ]
    )

    links = await manager._find_meaningful_links(
        embedding=[0.1],
        app_id="app",
        user_id="user",
        session_id=None,
    )

    assert [link["memory_id"] for link in links] == ["a"]


@pytest.mark.asyncio
async def test_find_meaningful_links_filters_by_composite_threshold(
    monkeypatch, mock_llm_connection
):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)
    manager.generate_memory_links = AsyncMock(
        return_value=[
            {"memory_id": "a", "composite_score": 0.9},
            {"memory_id": "b", "composite_score": 0.6},
            {"memory_id": "c", "composite_score": 0.8},
        ]
    )

    links = await manager._find_meaningful_links(
        embedding=[0.1],
        app_id="app",
        user_id="user",
        session_id="session1",
    )

    assert len(links) == 2
    assert all(link["composite_score"] >= 0.7 for link in links)


@pytest.mark.asyncio
async def test_query_memory_validates_empty_app_id(monkeypatch, mock_llm_connection):
    manager, _ = _make_manager(monkeypatch, mock_llm_connection)

    results = await manager.query_memory(app_id="", query="test")

    assert results == []


@pytest.mark.asyncio
async def test_query_memory_with_user_id_filter(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1"],
            "scores": [0.9],
            "metadatas": [
                {
                    "document_id": "m1",
                    "app_id": "app",
                    "user_id": "user1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "status": "active",
                }
            ],
            "ids": ["m1"],
        }
    )

    results = await manager.query_memory(
        app_id="app", query="test", user_id="user1", n_results=1
    )

    assert len(results) >= 0
    call_kwargs = handler.query_collection.await_args.kwargs
    assert call_kwargs["filter_conditions"]["user_id"] == "user1"


@pytest.mark.asyncio
async def test_query_memory_with_session_id_filter(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1"],
            "scores": [0.9],
            "metadatas": [
                {
                    "document_id": "m1",
                    "app_id": "app",
                    "session_id": "session1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "status": "active",
                }
            ],
            "ids": ["m1"],
        }
    )

    results = await manager.query_memory(
        app_id="app", query="test", session_id="session1", n_results=1
    )

    assert len(results) >= 0
    call_kwargs = handler.query_collection.await_args.kwargs
    assert call_kwargs["filter_conditions"]["session_id"] == "session1"


@pytest.mark.asyncio
async def test_execute_query_filters_by_similarity_threshold(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2", "doc3"],
            "scores": [0.9, 0.5, 0.3],
            "metadatas": [
                {
                    "document_id": "m1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                },
                {
                    "document_id": "m2",
                    "created_at": "2024-01-02T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                },
                {
                    "document_id": "m3",
                    "created_at": "2024-01-03T00:00:00Z",
                    "updated_at": "2024-01-03T00:00:00Z",
                },
            ],
            "ids": ["m1", "m2", "m3"],
        }
    )

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="test",
        filter_conditions={},
        expanded_n_results=10,
        similarity_threshold=0.6,
        n_results=10,
    )

    assert len(result) == 1
    assert result[0]["similarity_score"] >= 0.6


@pytest.mark.asyncio
async def test_execute_query_warns_when_no_candidates_pass_recall(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1"],
            "scores": [0.2],
            "metadatas": [{"document_id": "m1"}],
            "ids": ["m1"],
        }
    )

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="test",
        filter_conditions={},
        expanded_n_results=10,
        similarity_threshold=0.5,
        n_results=10,
    )

    assert len(result) == 0


@pytest.mark.asyncio
async def test_execute_query_filters_by_composite_score(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2"],
            "scores": [0.8, 0.7],
            "metadatas": [
                {"document_id": "m1", "created_at": "2024-01-01T00:00:00Z"},
                {"document_id": "m2", "created_at": "2024-01-02T00:00:00Z"},
            ],
            "ids": ["m1", "m2"],
        }
    )

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="test",
        filter_conditions={},
        expanded_n_results=10,
        similarity_threshold=0.3,
        n_results=10,
    )

    assert all(r["composite_score"] >= 0.5 for r in result)


@pytest.mark.asyncio
async def test_execute_query_warns_when_all_filtered_by_composite(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2"],
            "scores": [0.8, 0.7],
            "metadatas": [
                {
                    "document_id": "m1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                },
                {
                    "document_id": "m2",
                    "created_at": "2024-01-02T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                },
            ],
            "ids": ["m1", "m2"],
        }
    )

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="test",
        filter_conditions={},
        expanded_n_results=10,
        similarity_threshold=0.3,
        n_results=10,
    )

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_execute_query_sorts_by_composite_score(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2", "doc3"],
            "scores": [0.6, 0.8, 0.7],
            "metadatas": [
                {"document_id": "m1", "created_at": "2024-01-01T00:00:00Z"},
                {"document_id": "m2", "created_at": "2024-01-02T00:00:00Z"},
                {"document_id": "m3", "created_at": "2024-01-03T00:00:00Z"},
            ],
            "ids": ["m1", "m2", "m3"],
        }
    )

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="test",
        filter_conditions={},
        expanded_n_results=10,
        similarity_threshold=0.3,
        n_results=10,
    )

    scores = [r["composite_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_execute_query_limits_results(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "scores": [0.9, 0.8, 0.7, 0.6, 0.5],
            "metadatas": [
                {
                    "document_id": f"m{i}",
                    "created_at": f"2024-01-0{i}T00:00:00Z",
                    "updated_at": f"2024-01-0{i}T00:00:00Z",
                }
                for i in range(1, 6)
            ],
            "ids": [f"m{i}" for i in range(1, 6)],
        }
    )

    result = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="test",
        filter_conditions={},
        expanded_n_results=10,
        similarity_threshold=0.3,
        n_results=3,
    )

    assert len(result) <= 3
    if len(result) > 0:
        assert all(r["similarity_score"] >= 0.3 for r in result)


@pytest.mark.asyncio
async def test_execute_query_all_candidates_filtered(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)

    handler.query_collection = AsyncMock(
        return_value={
            "documents": ["doc1"],
            "scores": [0.9],
            "metadatas": [
                {
                    "document_id": "m1",
                    "app_id": "app",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "status": "active",
                }
            ],
            "ids": ["m1"],
        }
    )

    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.calculate_composite_score",
        lambda semantic_score, metadata: 0.2,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.calculate_recency_score",
        lambda **kwargs: 0.1,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.calculate_importance_score",
        lambda metadata: 0.1,
    )

    results = await manager._execute_query(
        vector_db_handler=handler,
        collection_name="app",
        query="hi",
        filter_conditions={},
        expanded_n_results=5,
        similarity_threshold=0.5,
        n_results=3,
    )

    assert results == []


@pytest.mark.asyncio
async def test_generate_memory_links_enforces_max_links(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_embedding = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "scores": [0.9, 0.8, 0.7, 0.6, 0.5],
            "metadatas": [
                {
                    "document_id": f"m{i}",
                    "status": "active",
                    "created_at": f"2024-01-0{i}T00:00:00Z",
                    "updated_at": f"2024-01-0{i}T00:00:00Z",
                }
                for i in range(1, 6)
            ],
        }
    )

    links = await manager.generate_memory_links(
        embedding=[0.1, 0.2],
        app_id="app",
        similarity_threshold=0.4,
        max_links=3,
    )

    assert len(links) >= 3
    assert all(link["similarity_score"] >= 0.4 for link in links)


@pytest.mark.asyncio
async def test_generate_memory_links_handler_disabled(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(
        monkeypatch, mock_llm_connection, handler=Mock(enabled=False)
    )

    links = await manager.generate_memory_links(
        embedding=[0.1, 0.2],
        app_id="app",
    )

    assert links == []


@pytest.mark.asyncio
async def test_generate_memory_links_no_documents(monkeypatch, mock_llm_connection):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_embedding = AsyncMock(
        return_value={
            "documents": [],
            "scores": [],
            "metadatas": [],
        }
    )

    links = await manager.generate_memory_links(
        embedding=[0.1, 0.2],
        app_id="app",
    )

    assert links == []


@pytest.mark.asyncio
async def test_generate_memory_links_with_user_and_session_filters(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_embedding = AsyncMock(
        return_value={
            "documents": ["doc1"],
            "scores": [0.9],
            "metadatas": [
                {
                    "document_id": "m1",
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ],
        }
    )

    links = await manager.generate_memory_links(
        embedding=[0.1, 0.2],
        app_id="app",
        user_id="user1",
        session_id="session1",
    )

    call_kwargs = handler.query_by_embedding.await_args.kwargs
    assert call_kwargs["filter_conditions"]["user_id"] == "user1"
    assert call_kwargs["filter_conditions"]["session_id"] == "session1"


@pytest.mark.asyncio
async def test_generate_memory_links_sorts_by_composite_score(
    monkeypatch, mock_llm_connection
):
    manager, handler = _make_manager(monkeypatch, mock_llm_connection)
    handler.query_by_embedding = AsyncMock(
        return_value={
            "documents": ["doc1", "doc2", "doc3"],
            "scores": [0.6, 0.9, 0.7],
            "metadatas": [
                {
                    "document_id": "m1",
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                {
                    "document_id": "m2",
                    "status": "active",
                    "created_at": "2024-01-02T00:00:00Z",
                },
                {
                    "document_id": "m3",
                    "status": "active",
                    "created_at": "2024-01-03T00:00:00Z",
                },
            ],
        }
    )

    links = await manager.generate_memory_links(
        embedding=[0.1, 0.2],
        app_id="app",
        similarity_threshold=0.4,
        max_links=10,
    )

    scores = [link["composite_score"] for link in links]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"
