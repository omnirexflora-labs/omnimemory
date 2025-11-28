import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from omnimemory.memory_management.chromadb_vector_db import (
    ChromaDBVectorDB,
)


def _fake_config_factory(monkeypatch, overrides=None):
    """Patch decouple.config used inside chromadb_vector_db with test overrides."""
    values = {
        "CHROMA_CLIENT_TYPE": "remote",
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
        "CHROMA_TENANT": None,
        "CHROMA_DATABASE": None,
        "CHROMA_API_KEY": None,
        "CHROMA_AUTH_TOKEN": None,
    }
    if overrides:
        values.update(overrides)

    def fake_config(key, default=None, cast=None):
        value = values.get(key, default)
        if cast and value is not None:
            return cast(value)
        return value

    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.config", fake_config
    )


@pytest.fixture
def mock_chromadb_client(monkeypatch):
    """Patch AsyncHttpClient and CloudClient with controllable mock instances."""
    client = AsyncMock()
    collection = AsyncMock()

    collection.add = AsyncMock()
    collection.query = AsyncMock()
    collection.get = AsyncMock()
    collection.upsert = AsyncMock()
    collection.delete = AsyncMock()

    client.get_or_create_collection = AsyncMock(return_value=collection)
    client.close = AsyncMock()

    mock_http_ctor = Mock(return_value=client)
    mock_cloud_ctor = Mock(return_value=client)
    mock_settings_ctor = Mock()

    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        mock_http_ctor,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.CloudClient", mock_cloud_ctor
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.Settings", mock_settings_ctor
    )

    return mock_http_ctor, mock_cloud_ctor, client, collection


def _make_db(monkeypatch, mock_llm_connection, client_fixture, overrides=None):
    _fake_config_factory(monkeypatch, overrides)
    http_ctor_mock, cloud_ctor_mock, client, collection = client_fixture
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    client_type = (
        overrides.get("CHROMA_CLIENT_TYPE", "remote") if overrides else "remote"
    )
    ctor_mock = cloud_ctor_mock if client_type == "cloud" else http_ctor_mock
    return db, ctor_mock, client, collection


class TestInitialization:
    def test_init_with_valid_remote_config_enables_client(
        self, monkeypatch, mock_llm_connection, mock_chromadb_client
    ):
        db, ctor_mock, _, _ = _make_db(
            monkeypatch, mock_llm_connection, mock_chromadb_client
        )

        ctor_mock.assert_called_once()
        assert db.enabled is True
        assert db.client is not None

    def test_init_with_valid_cloud_config_enables_client(
        self, monkeypatch, mock_llm_connection, mock_chromadb_client
    ):
        db, ctor_mock, _, _ = _make_db(
            monkeypatch,
            mock_llm_connection,
            mock_chromadb_client,
            overrides={
                "CHROMA_CLIENT_TYPE": "cloud",
                "CHROMA_TENANT": "test-tenant",
                "CHROMA_DATABASE": "test-db",
                "CHROMA_API_KEY": "test-key",
            },
        )

        ctor_mock.assert_called_once()
        assert db.enabled is True
        assert db.client is not None

    def test_init_cloud_missing_credentials_disables(
        self, monkeypatch, mock_llm_connection, mock_chromadb_client
    ):
        db, ctor_mock, _, _ = _make_db(
            monkeypatch,
            mock_llm_connection,
            mock_chromadb_client,
            overrides={
                "CHROMA_CLIENT_TYPE": "cloud",
                "CHROMA_TENANT": None,
            },
        )

        assert db.enabled is False
        assert db.client is None

    def test_init_without_llm_connection_warns_and_disables(
        self, monkeypatch, mock_chromadb_client
    ):
        _fake_config_factory(monkeypatch)
        db = ChromaDBVectorDB()

        mock_chromadb_client[0].assert_called_once()
        assert db.enabled is False

    def test_init_invalid_client_type_defaults_to_remote(
        self, monkeypatch, mock_llm_connection, mock_chromadb_client
    ):
        db, ctor_mock, _, _ = _make_db(
            monkeypatch,
            mock_llm_connection,
            mock_chromadb_client,
            overrides={"CHROMA_CLIENT_TYPE": "invalid"},
        )

        ctor_mock.assert_called_once()
        assert db.enabled is True

    def test_init_ctor_exception_disables(
        self, monkeypatch, mock_llm_connection, mock_chromadb_client
    ):
        http_ctor_mock, _, _, _ = mock_chromadb_client
        http_ctor_mock.side_effect = RuntimeError("boom")
        _fake_config_factory(monkeypatch)

        db = ChromaDBVectorDB(llm_connection=mock_llm_connection)

        assert db.enabled is False
        assert db.client is None


@pytest.mark.asyncio
async def test_close_calls_client_close(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, client, _ = _make_db(monkeypatch, mock_llm_connection, mock_chromadb_client)

    await db.close()

    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_no_client_is_noop(monkeypatch, mock_llm_connection):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    await db.close()


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, client, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )

    result = await db._ensure_collection("memories")

    client.get_or_create_collection.assert_awaited_once()
    assert result == collection


@pytest.mark.asyncio
async def test_ensure_collection_returns_existing_collection(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, client, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )

    result = await db._ensure_collection("memories")

    client.get_or_create_collection.assert_awaited_once()
    assert result == collection


@pytest.mark.asyncio
async def test_ensure_collection_no_client_logs_warning(
    monkeypatch, mock_llm_connection
):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    result = await db._ensure_collection("missing")

    assert result is None


@pytest.mark.asyncio
async def test_ensure_collection_handles_exception(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, client, _ = _make_db(monkeypatch, mock_llm_connection, mock_chromadb_client)
    client.get_or_create_collection.side_effect = RuntimeError("Collection error")

    with pytest.raises(RuntimeError):
        await db._ensure_collection("memories")


@pytest.mark.asyncio
async def test_add_to_collection_success(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db._ensure_collection = AsyncMock(return_value=collection)

    result = await db.add_to_collection(
        "memories", "doc-1", "hello world", [0.1, 0.2], {"app_id": "a1"}
    )

    assert result is True
    db._ensure_collection.assert_awaited_once()
    collection.add.assert_awaited_once()
    add_kwargs = collection.add.await_args.kwargs
    assert add_kwargs["ids"] == ["doc-1"]
    assert add_kwargs["documents"] == ["hello world"]
    assert add_kwargs["metadatas"][0]["app_id"] == "a1"
    assert add_kwargs["metadatas"][0]["text"] == "hello world"


@pytest.mark.asyncio
async def test_add_to_collection_invalid_embedding_returns_false(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )

    result = await db.add_to_collection("memories", "doc-1", "text", None, {})

    assert result is False
    collection.add.assert_not_called()


@pytest.mark.asyncio
async def test_add_to_collection_handles_exception(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db._ensure_collection = AsyncMock(side_effect=RuntimeError("fail"))

    result = await db.add_to_collection(
        "memories", "doc-1", "text", [0.1, 0.2], {"app_id": "a1"}
    )

    assert result is False
    collection.add.assert_not_called()


@pytest.mark.asyncio
async def test_add_to_collection_no_client_returns_false(
    monkeypatch, mock_llm_connection
):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    result = await db.add_to_collection("memories", "doc-1", "text", [0.1, 0.2], {})

    assert result is False


@pytest.mark.asyncio
async def test_query_collection_builds_filters_and_returns_results(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    collection.query.return_value = {
        "documents": [["doc text"]],
        "metadatas": [[{"app_id": "a1"}]],
        "ids": [["doc-1"]],
        "distances": [[0.1]],
    }

    results = await db.query_collection(
        "memories",
        query="hello",
        n_results=2,
        similarity_threshold=0.9,
        filter_conditions={"app_id": "a1", "user_id": None},
    )

    collection.query.assert_awaited_once()
    assert results["documents"] == ["doc text"]
    assert results["ids"] == ["doc-1"]
    assert results["scores"][0] == pytest.approx(0.95, rel=0.01)


@pytest.mark.asyncio
async def test_query_collection_filters_by_similarity_threshold(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    collection.query.return_value = {
        "documents": [["keep", "drop"]],
        "metadatas": [[{"app_id": "a1"}, {"app_id": "a1"}]],
        "ids": [["doc-1", "doc-2"]],
        "distances": [[0.1, 1.5]],
    }

    results = await db.query_collection(
        "memories",
        query="hello",
        n_results=5,
        similarity_threshold=0.5,
    )

    assert len(results["documents"]) == 1
    assert results["documents"] == ["keep"]
    assert results["ids"] == ["doc-1"]


@pytest.mark.asyncio
async def test_query_collection_missing_client_returns_empty(
    monkeypatch, mock_llm_connection
):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    results = await db.query_collection("memories", "q", 3, 0.1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_collection_handles_not_found_exception(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    collection.query.side_effect = RuntimeError("404 collection missing")

    results = await db.query_collection("memories", "hello", 1, 0.5)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_collection_general_exception_returns_empty(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    collection.query.side_effect = RuntimeError("boom")

    results = await db.query_collection("memories", "hello", 1, 0.5)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_by_embedding_success(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    collection.query.return_value = {
        "documents": [["doc"]],
        "metadatas": [[{"app_id": "a1"}]],
        "ids": [["1"]],
        "distances": [[0.6]],
    }

    results = await db.query_by_embedding(
        "memories",
        embedding=[0.1, 0.2],
        n_results=1,
        filter_conditions={"app_id": "a1"},
        similarity_threshold=0.5,
    )

    assert results["documents"] == ["doc"]
    collection.query.assert_awaited_once()
    query_kwargs = collection.query.await_args.kwargs
    assert query_kwargs["query_embeddings"] == [[0.1, 0.2]]
    assert query_kwargs["where"]["app_id"] == {"$eq": "a1"}


@pytest.mark.asyncio
async def test_query_by_embedding_invalid_embedding_returns_empty(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )

    results = await db.query_by_embedding("memories", None, 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}
    collection.query.assert_not_called()


@pytest.mark.asyncio
async def test_query_by_embedding_filters_by_threshold(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    collection.query.return_value = {
        "documents": [["doc", "low"]],
        "metadatas": [[{"app_id": "a1"}, {"app_id": "a1"}]],
        "ids": [["1", "2"]],
        "distances": [[0.6, 1.8]],
    }

    results = await db.query_by_embedding(
        "memories", embedding=[0.1, 0.2], n_results=10, similarity_threshold=0.5
    )

    assert len(results["documents"]) == 1
    assert results["documents"] == ["doc"]


@pytest.mark.asyncio
async def test_query_by_embedding_handles_not_found_exception(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    collection.query.side_effect = RuntimeError("doesn't exist")

    results = await db.query_by_embedding("memories", [0.1], 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_by_embedding_general_exception_returns_empty(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    collection.query.side_effect = RuntimeError("boom")

    results = await db.query_by_embedding("memories", [0.1], 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_by_embedding_no_client_returns_empty(
    monkeypatch, mock_llm_connection
):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    results = await db.query_by_embedding("memories", [0.1, 0.2], 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_update_memory_success(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["old text"],
        "metadatas": [{"text": "old text", "app_id": "a1"}],
        "embeddings": [[0.1, 0.2]],
    }

    result = await db.update_memory(
        "memories",
        "doc-1",
        {"status": "updated"},
    )

    assert result is True
    collection.upsert.assert_awaited_once()
    upsert_kwargs = collection.upsert.await_args.kwargs
    assert upsert_kwargs["ids"] == ["doc-1"]
    assert upsert_kwargs["metadatas"][0]["status"] == "updated"
    assert upsert_kwargs["metadatas"][0]["app_id"] == "a1"
    assert upsert_kwargs["embeddings"] == [[0.1, 0.2]]
    assert upsert_kwargs["documents"] == ["old text"]


@pytest.mark.asyncio
async def test_update_memory_handles_numpy_array_embedding(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    """Test that update_memory properly converts numpy arrays to lists."""
    import numpy as np

    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    numpy_embedding = np.array([0.1, 0.2, 0.3])
    collection.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["old text"],
        "metadatas": [{"text": "old text", "app_id": "a1"}],
        "embeddings": [numpy_embedding],
    }

    result = await db.update_memory(
        "memories",
        "doc-1",
        {"status": "updated"},
    )

    assert result is True
    collection.upsert.assert_awaited_once()
    upsert_kwargs = collection.upsert.await_args.kwargs
    assert isinstance(upsert_kwargs["embeddings"][0], list)
    assert upsert_kwargs["embeddings"][0] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_update_memory_handles_timeout(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.side_effect = asyncio.TimeoutError()

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False
    collection.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_missing_embedding_returns_false(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["old text"],
        "metadatas": [{"text": "old text"}],
        "embeddings": [],
    }

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False
    collection.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_disabled_returns_false(monkeypatch, mock_llm_connection):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.enabled = False

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False


@pytest.mark.asyncio
async def test_update_memory_retrieve_timeout_string(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.side_effect = RuntimeError("Request timeout while retrieving")

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False


@pytest.mark.asyncio
async def test_update_memory_not_found_returns_false(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
        "embeddings": [],
    }

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False
    collection.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_update_memory_upsert_exception_returns_false(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["old text"],
        "metadatas": [{"text": "old text"}],
        "embeddings": [[0.1, 0.2]],
    }
    collection.upsert.side_effect = RuntimeError("upsert failed")

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False


@pytest.mark.asyncio
async def test_query_by_id_success(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.return_value = {
        "ids": ["doc-1"],
        "documents": ["hello"],
        "metadatas": [{"text": "hello", "app_id": "a1"}],
    }

    result = await db.query_by_id("memories", "doc-1")

    assert result == {
        "memory_id": "doc-1",
        "document": "hello",
        "metadata": {"app_id": "a1"},
    }
    collection.get.assert_awaited_once()
    get_kwargs = collection.get.await_args.kwargs
    assert get_kwargs["ids"] == ["doc-1"]
    assert "include" in get_kwargs


@pytest.mark.asyncio
async def test_query_by_id_disabled_returns_none(monkeypatch, mock_llm_connection):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.enabled = False

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_query_by_id_timeout(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.side_effect = asyncio.TimeoutError()

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_query_by_id_request_timeout_string(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.side_effect = RuntimeError("Request timeout while retrieving")

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_query_by_id_not_found_returns_none(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_query_by_id_general_exception_returns_none(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.enabled = True
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.get.side_effect = RuntimeError("boom")

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_delete_from_collection_success(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db._ensure_collection = AsyncMock(return_value=collection)

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is True
    collection.delete.assert_awaited_once()
    delete_kwargs = collection.delete.await_args.kwargs
    assert delete_kwargs["ids"] == ["doc-1"]


@pytest.mark.asyncio
async def test_delete_from_collection_no_client_returns_false(
    monkeypatch, mock_llm_connection
):
    mock_client = Mock()
    monkeypatch.setattr(
        "omnimemory.memory_management.chromadb_vector_db.AsyncHttpClient",
        Mock(return_value=mock_client),
    )
    db = ChromaDBVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is False


@pytest.mark.asyncio
async def test_delete_from_collection_exception_returns_false(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db._ensure_collection = AsyncMock(return_value=collection)
    collection.delete.side_effect = RuntimeError("delete failed")

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is False


@pytest.mark.asyncio
async def test_query_collection_empty_results_handles_gracefully(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "ids": [[]],
        "distances": [[]],
    }

    results = await db.query_collection("memories", "hello", 5, 0.5)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_by_embedding_none_limit_uses_default(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    collection.query.return_value = {
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"app_id": "a1"}, {"app_id": "a1"}]],
        "ids": [["1", "2"]],
        "distances": [[0.1, 0.2]],
    }

    results = await db.query_by_embedding(
        "memories", embedding=[0.1, 0.2], n_results=None, similarity_threshold=0.0
    )

    assert len(results["documents"]) == 2
    query_kwargs = collection.query.await_args.kwargs
    assert query_kwargs["n_results"] == 10000


@pytest.mark.asyncio
async def test_query_collection_expands_limit_for_threshold_filtering(
    monkeypatch, mock_llm_connection, mock_chromadb_client
):
    db, _, _, collection = _make_db(
        monkeypatch, mock_llm_connection, mock_chromadb_client
    )
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    collection.query.return_value = {
        "documents": [["doc1", "doc2", "doc3"]],
        "metadatas": [[{"app_id": "a1"}] * 3],
        "ids": [["1", "2", "3"]],
        "distances": [[0.1, 0.8, 1.5]],
    }

    results = await db.query_collection(
        "memories", "hello", n_results=2, similarity_threshold=0.9
    )

    query_kwargs = collection.query.await_args.kwargs
    assert query_kwargs["n_results"] == 10
    assert len(results["documents"]) == 1
