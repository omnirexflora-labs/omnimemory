import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from omnimemory.memory_management.qdrant_vector_db import QdrantVectorDB


def _fake_config_factory(monkeypatch, overrides=None):
    """Patch decouple.config used inside qdrant_vector_db with test overrides."""
    values = {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "QDRANT_TIMEOUT": 30,
    }
    if overrides:
        values.update(overrides)

    def fake_config(key, default=None, cast=None):
        value = values.get(key, default)
        if cast and value is not None:
            return cast(value)
        return value

    monkeypatch.setattr(
        "omnimemory.memory_management.qdrant_vector_db.config", fake_config
    )


@pytest.fixture
def mock_qdrant_client(monkeypatch):
    """Patch AsyncQdrantClient with a controllable mock instance."""
    client = AsyncMock()
    client.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[]))
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.query_points = AsyncMock()
    client.retrieve = AsyncMock()
    client.delete = AsyncMock()
    client.close = AsyncMock()
    mock_ctor = Mock(return_value=client)
    monkeypatch.setattr(
        "omnimemory.memory_management.qdrant_vector_db.AsyncQdrantClient", mock_ctor
    )
    return mock_ctor, client


def _make_db(monkeypatch, mock_llm_connection, client_fixture, overrides=None):
    _fake_config_factory(monkeypatch, overrides)
    ctor_mock, client = client_fixture
    db = QdrantVectorDB(llm_connection=mock_llm_connection)
    return db, ctor_mock, client


class TestInitialization:
    def test_init_with_valid_config_enables_client(
        self, monkeypatch, mock_llm_connection, mock_qdrant_client
    ):
        db, ctor_mock, _ = _make_db(
            monkeypatch, mock_llm_connection, mock_qdrant_client
        )

        ctor_mock.assert_called_once_with(host="localhost", port=6333, timeout=30)
        assert db.enabled is True

    def test_init_missing_host_disables(
        self, monkeypatch, mock_llm_connection, mock_qdrant_client
    ):
        db, ctor_mock, _ = _make_db(
            monkeypatch,
            mock_llm_connection,
            mock_qdrant_client,
            overrides={"QDRANT_HOST": None},
        )

        ctor_mock.assert_not_called()
        assert db.enabled is False
        assert db.client is None

    def test_init_without_llm_connection_warns_and_disables(
        self, monkeypatch, mock_qdrant_client
    ):
        _fake_config_factory(monkeypatch)
        db = QdrantVectorDB()

        mock_qdrant_client[0].assert_called_once()
        assert db.enabled is False

    def test_init_invalid_port_value_disables_client(
        self, monkeypatch, mock_llm_connection, mock_qdrant_client
    ):
        db, ctor_mock, _ = _make_db(
            monkeypatch,
            mock_llm_connection,
            mock_qdrant_client,
            overrides={"QDRANT_PORT": "not-an-int"},
        )

        ctor_mock.assert_not_called()
        assert db.enabled is False
        assert db.client is None

    def test_init_ctor_exception_disables(
        self, monkeypatch, mock_llm_connection, mock_qdrant_client
    ):
        ctor_mock, _ = mock_qdrant_client
        ctor_mock.side_effect = RuntimeError("boom")
        _fake_config_factory(monkeypatch)

        db = QdrantVectorDB(llm_connection=mock_llm_connection)

        assert db.enabled is False
        assert db.client is None


@pytest.mark.asyncio
async def test_close_calls_client_close(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)

    await db.close()

    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_no_client_is_noop(mock_llm_connection):
    db = QdrantVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    await db.close()


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db._get_embedding_dimensions = Mock(return_value=1536)
    client.get_collections.return_value = SimpleNamespace(collections=[])

    await db._ensure_collection("memories")

    client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_collection_skips_when_exists(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db._get_embedding_dimensions = Mock(return_value=1536)
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="memories")]
    )

    await db._ensure_collection("memories")

    client.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_collection_no_client_logs_warning(
    mock_llm_connection,
):
    db = QdrantVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    await db._ensure_collection("missing")


@pytest.mark.asyncio
async def test_ensure_collection_invalid_vector_size_raises(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, _ = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db._get_embedding_dimensions = Mock(return_value=None)

    with pytest.raises(ValueError):
        await db._ensure_collection("memories")


@pytest.mark.asyncio
async def test_add_to_collection_success(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db._ensure_collection = AsyncMock()

    result = await db.add_to_collection(
        "memories", "doc-1", "hello world", [0.1, 0.2], {"app_id": "a1"}
    )

    assert result is True
    db._ensure_collection.assert_awaited_once()
    client.upsert.assert_awaited_once()
    upsert_kwargs = client.upsert.await_args.kwargs
    assert upsert_kwargs["wait"] is True


@pytest.mark.asyncio
async def test_add_to_collection_invalid_embedding_returns_false(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)

    result = await db.add_to_collection("memories", "doc-1", "text", None, {})

    assert result is False
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_add_to_collection_handles_exception(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db._ensure_collection = AsyncMock(side_effect=RuntimeError("fail"))

    result = await db.add_to_collection(
        "memories", "doc-1", "text", [0.1, 0.2], {"app_id": "a1"}
    )

    assert result is False
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_query_collection_builds_filters_and_returns_results(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    point = SimpleNamespace(
        payload={"text": "doc text", "app_id": "a1"},
        score=0.95,
        id="doc-1",
    )
    client.query_points.return_value = SimpleNamespace(points=[point])

    results = await db.query_collection(
        "memories",
        query="hello",
        n_results=2,
        similarity_threshold=0.9,
        filter_conditions={"app_id": "a1", "user_id": None},
    )

    client.query_points.assert_awaited_once()
    assert results["documents"] == ["doc text"]
    assert results["ids"] == ["doc-1"]


@pytest.mark.asyncio
async def test_query_collection_typeerror_fallback_filters_scores(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    client.query_points.side_effect = [
        TypeError("score_threshold not supported"),
        SimpleNamespace(
            points=[
                SimpleNamespace(payload={"text": "keep"}, score=0.8, id="1"),
                SimpleNamespace(payload={"text": "drop"}, score=0.4, id="2"),
            ]
        ),
    ]

    results = await db.query_collection(
        "memories",
        query="hello",
        n_results=5,
        similarity_threshold=0.5,
    )

    assert results["documents"] == ["keep"]
    assert client.query_points.await_count == 2


@pytest.mark.asyncio
async def test_query_collection_missing_client_returns_empty():
    db = QdrantVectorDB()
    db.client = None
    db.enabled = False

    results = await db.query_collection("memories", "q", 3, 0.1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_collection_handles_not_found_exception(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    client.query_points.side_effect = RuntimeError("404 collection missing")

    results = await db.query_collection("memories", "hello", 1, 0.5)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_collection_general_exception_returns_empty(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    client.query_points.side_effect = RuntimeError("boom")

    results = await db.query_collection("memories", "hello", 1, 0.5)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_by_embedding_success(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    point = SimpleNamespace(payload={"text": "doc"}, score=0.7, id="1")
    client.query_points.return_value = SimpleNamespace(points=[point])

    results = await db.query_by_embedding(
        "memories",
        embedding=[0.1, 0.2],
        n_results=1,
        filter_conditions={"app_id": "a1"},
        similarity_threshold=0.5,
    )

    assert results["documents"] == ["doc"]
    client.query_points.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_by_embedding_invalid_embedding_returns_empty(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)

    results = await db.query_by_embedding("memories", None, 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}
    client.query_points.assert_not_called()


@pytest.mark.asyncio
async def test_query_by_embedding_typeerror_fallback(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    client.query_points.side_effect = [
        TypeError("unsupported score_threshold"),
        SimpleNamespace(
            points=[
                SimpleNamespace(payload={"text": "doc"}, score=0.7, id="1"),
                SimpleNamespace(payload={"text": "low"}, score=0.2, id="2"),
            ]
        ),
    ]

    results = await db.query_by_embedding(
        "memories", embedding=[0.1, 0.2], n_results=1, similarity_threshold=0.5
    )

    assert results["documents"] == ["doc"]
    assert client.query_points.await_count == 2


@pytest.mark.asyncio
async def test_query_by_embedding_handles_not_found_exception(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    client.query_points.side_effect = RuntimeError("doesn't exist")

    results = await db.query_by_embedding("memories", [0.1], 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_by_embedding_general_exception_returns_empty(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    client.query_points.side_effect = RuntimeError("boom")

    results = await db.query_by_embedding("memories", [0.1], 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_update_memory_success(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    point = SimpleNamespace(
        payload={"text": "old", "app_id": "a1"},
        vector=[0.1, 0.2],
    )
    client.retrieve.return_value = [point]

    result = await db.update_memory(
        "memories",
        "doc-1",
        {"status": "updated"},
    )

    assert result is True
    client.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_memory_handles_timeout(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    client.retrieve.side_effect = asyncio.TimeoutError()

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_invalid_vector_returns_false(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    point = SimpleNamespace(payload={"text": "old"}, vector={"bad": "format"})
    client.retrieve.return_value = [point]

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_missing_vector_returns_false(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    point = SimpleNamespace(payload={"text": "old"}, vector=None)
    client.retrieve.return_value = [point]

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_disabled_returns_false(mock_llm_connection):
    db = QdrantVectorDB(llm_connection=mock_llm_connection)
    db.enabled = False

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False


@pytest.mark.asyncio
async def test_update_memory_retrieve_timeout_string(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    client.retrieve.side_effect = RuntimeError("Request Timeout while retrieving")

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False


@pytest.mark.asyncio
async def test_update_memory_upsert_exception_returns_false(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    point = SimpleNamespace(
        payload={"text": "old"},
        vector=[0.1, 0.2],
    )
    client.retrieve.return_value = [point]
    client.upsert.side_effect = RuntimeError("upsert failed")

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False


@pytest.mark.asyncio
async def test_query_by_id_success(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    point = SimpleNamespace(
        id="doc-1",
        payload={"text": "hello", "app_id": "a1"},
    )
    client.retrieve.return_value = [point]

    result = await db.query_by_id("memories", "doc-1")

    assert result == {
        "memory_id": "doc-1",
        "document": "hello",
        "metadata": {"app_id": "a1"},
    }


@pytest.mark.asyncio
async def test_query_by_id_disabled_returns_none(mock_llm_connection):
    db = QdrantVectorDB(llm_connection=mock_llm_connection)
    db.enabled = False

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_query_by_id_timeout(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    client.retrieve.side_effect = asyncio.TimeoutError()

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_query_by_id_request_timeout_string(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    client.retrieve.side_effect = RuntimeError("Request Timeout while retrieving")

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_query_by_id_general_exception_returns_none(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    db.enabled = True
    db._ensure_collection = AsyncMock()
    client.retrieve.side_effect = RuntimeError("boom")

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_delete_from_collection_success(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="memories")]
    )

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is True
    client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_from_collection_missing_collection_returns_false(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    client.get_collections.return_value = SimpleNamespace(collections=[])

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is False
    client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_from_collection_no_client_returns_false(mock_llm_connection):
    db = QdrantVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is False


@pytest.mark.asyncio
async def test_delete_from_collection_exception_returns_false(
    monkeypatch, mock_llm_connection, mock_qdrant_client
):
    db, _, client = _make_db(monkeypatch, mock_llm_connection, mock_qdrant_client)
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="memories")]
    )
    client.delete.side_effect = RuntimeError("delete failed")

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is False
