from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from omnimemory.memory_management.mongodb_vector_db import MongoDBVectorDB


def _fake_config_factory(monkeypatch, overrides=None):
    """Patch decouple.config for MongoDB settings."""
    values = {
        "MONGODB_URI": "mongodb://localhost:27017",
        "MONGODB_DB_NAME": "omnimemory",
    }
    if overrides:
        values.update(overrides)

    def fake_config(key, default=None, cast=None):
        value = values.get(key, default)
        if cast and value is not None:
            return cast(value)
        return value

    monkeypatch.setattr(
        "omnimemory.memory_management.mongodb_vector_db.config", fake_config
    )


@pytest.fixture
def mock_mongodb_client(monkeypatch):
    """Patch AsyncIOMotorClient and MongoDB collection primitives."""
    client = MagicMock()
    database = MagicMock()
    collection = MagicMock()

    collection.create_search_index = AsyncMock()
    collection.replace_one = AsyncMock()
    collection.update_one = AsyncMock()
    collection.delete_one = AsyncMock()
    collection.find_one = AsyncMock()

    aggregate_cursor = MagicMock()
    aggregate_cursor.to_list = AsyncMock(return_value=[])
    collection.aggregate = MagicMock(return_value=aggregate_cursor)

    database.__getitem__.return_value = collection
    database.list_collection_names = AsyncMock(return_value=["existing"])
    database.create_collection = AsyncMock()

    client.__getitem__.return_value = database
    client.close = Mock()

    mock_client_ctor = Mock(return_value=client)
    mock_index_model = Mock(return_value="index-model")

    monkeypatch.setattr(
        "omnimemory.memory_management.mongodb_vector_db.AsyncIOMotorClient",
        mock_client_ctor,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.mongodb_vector_db.SearchIndexModel",
        mock_index_model,
    )

    return mock_client_ctor, client, database, collection, aggregate_cursor


def _make_db(monkeypatch, mock_llm_connection, mongo_fixture, overrides=None):
    _fake_config_factory(monkeypatch, overrides)
    ctor, client, database, collection, cursor = mongo_fixture
    db_instance = MongoDBVectorDB(llm_connection=mock_llm_connection)
    return db_instance, ctor, client, database, collection, cursor


class TestInitialization:
    def test_init_with_valid_config_enables_client(
        self, monkeypatch, mock_llm_connection, mock_mongodb_client
    ):
        db, ctor, client, database, _, _ = _make_db(
            monkeypatch, mock_llm_connection, mock_mongodb_client
        )

        ctor.assert_called_once_with("mongodb://localhost:27017")
        assert db.enabled is True
        assert db.client is client
        assert db.db is database

    def test_init_missing_uri_disables(
        self, monkeypatch, mock_llm_connection, mock_mongodb_client
    ):
        db, ctor, _, _, _, _ = _make_db(
            monkeypatch,
            mock_llm_connection,
            mock_mongodb_client,
            overrides={"MONGODB_URI": None},
        )

        ctor.assert_not_called()
        assert db.enabled is False
        assert db.client is None

    def test_init_without_llm_connection_warns_and_disables(
        self, monkeypatch, mock_mongodb_client
    ):
        _fake_config_factory(monkeypatch)
        db = MongoDBVectorDB()

        mock_mongodb_client[0].assert_called_once()
        assert db.enabled is False

    def test_init_ctor_exception_disables(
        self, monkeypatch, mock_llm_connection, mock_mongodb_client
    ):
        ctor, _, _, _, _ = mock_mongodb_client
        ctor.side_effect = RuntimeError("boom")
        _fake_config_factory(monkeypatch)

        db = MongoDBVectorDB(llm_connection=mock_llm_connection)

        assert db.enabled is False
        assert db.client is None


@pytest.mark.asyncio
async def test_close_calls_client_close(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, client, _, _, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )

    await db.close()

    client.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_no_client_is_noop(monkeypatch, mock_llm_connection):
    _fake_config_factory(monkeypatch)
    db = MongoDBVectorDB(llm_connection=mock_llm_connection)
    db.client = None

    await db.close()


@pytest.mark.asyncio
async def test_ensure_collection_creates_new_collection_and_index(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, database, _, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    db._get_embedding_dimensions = Mock(return_value=1536)
    db._create_vector_search_index = AsyncMock()
    database.list_collection_names = AsyncMock(return_value=[])
    database.create_collection = AsyncMock()

    await db._ensure_collection("memories")

    database.create_collection.assert_awaited_once_with("memories")
    db._create_vector_search_index.assert_awaited_once_with("memories", 1536)


@pytest.mark.asyncio
async def test_ensure_collection_existing_skips_creation(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, database, _, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    db._get_embedding_dimensions = Mock(return_value=1536)
    db._create_vector_search_index = AsyncMock()
    database.list_collection_names = AsyncMock(return_value=["memories"])
    database.create_collection = AsyncMock()

    await db._ensure_collection("memories")

    database.create_collection.assert_not_called()
    db._create_vector_search_index.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_collection_invalid_vector_size_raises(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, _, _ = _make_db(monkeypatch, mock_llm_connection, mock_mongodb_client)
    db._get_embedding_dimensions = Mock(return_value=None)

    with pytest.raises(ValueError, match="positive integer"):
        await db._ensure_collection("memories")


@pytest.mark.asyncio
async def test_add_to_collection_success(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    db._ensure_collection = AsyncMock()

    result = await db.add_to_collection(
        "memories", "doc-1", "text", [0.1, 0.2], {"app_id": "a1"}
    )

    assert result is True
    db._ensure_collection.assert_awaited_once_with("memories")
    collection.replace_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_to_collection_invalid_embedding_returns_false(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )

    result = await db.add_to_collection("memories", "doc-1", "text", None, {})

    assert result is False
    collection.replace_one.assert_not_called()


@pytest.mark.asyncio
async def test_vector_search_filters_results(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, cursor = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    db._ensure_collection = AsyncMock()
    cursor.to_list = AsyncMock(
        return_value=[
            {"_id": "1", "text": "keep", "score": 0.9, "app_id": "a1"},
            {"_id": "2", "text": "drop", "score": 0.1, "app_id": "a1"},
        ]
    )

    results = await db._vector_search(
        "memories",
        query_vector=[0.1, 0.2],
        n_results=1,
        similarity_threshold=0.5,
        filter_conditions={"app_id": "a1"},
    )

    assert results["documents"] == ["keep"]
    assert results["ids"] == ["1"]
    collection.aggregate.assert_called_once()


@pytest.mark.asyncio
async def test_query_collection_requires_llm_connection(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, _, _ = _make_db(monkeypatch, mock_llm_connection, mock_mongodb_client)
    db.llm_connection = None

    results = await db.query_collection("memories", "query", 1, 0.5)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_collection_success(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, _, _ = _make_db(monkeypatch, mock_llm_connection, mock_mongodb_client)
    db.embed_text = AsyncMock(return_value=[0.1, 0.2])
    db._vector_search = AsyncMock(return_value={"ids": ["1"]})

    results = await db.query_collection("memories", "query", 2, 0.5)

    db.embed_text.assert_awaited_once_with("query")
    db._vector_search.assert_awaited_once()
    assert results == {"ids": ["1"]}


@pytest.mark.asyncio
async def test_query_by_embedding_delegates_to_vector_search(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, _, _ = _make_db(monkeypatch, mock_llm_connection, mock_mongodb_client)
    db._vector_search = AsyncMock(return_value={"ids": ["1"]})

    results = await db.query_by_embedding(
        "memories", [0.1, 0.2], n_results=3, similarity_threshold=0.4
    )

    db._vector_search.assert_awaited_once()
    assert results == {"ids": ["1"]}


@pytest.mark.asyncio
async def test_query_by_embedding_invalid_embedding_returns_empty(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    db.enabled = True

    results = await db.query_by_embedding("memories", None, 1)

    assert results == {"documents": [], "scores": [], "metadatas": [], "ids": []}
    collection.aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_query_by_embedding_none_limit_uses_default(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    db._ensure_collection = AsyncMock()
    cursor = MagicMock()
    cursor.to_list = AsyncMock(
        return_value=[
            {"_id": "doc-1", "text": "doc1", "score": 0.95},
            {"_id": "doc-2", "text": "doc2", "score": 0.9},
        ]
    )
    collection.aggregate = MagicMock(return_value=cursor)

    results = await db.query_by_embedding(
        "memories", embedding=[0.1, 0.2], n_results=None, similarity_threshold=0.0
    )

    assert len(results["documents"]) == 2
    collection.aggregate.assert_called_once()


@pytest.mark.asyncio
async def test_update_memory_success(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    collection.find_one = AsyncMock(return_value={"_id": "doc-1"})
    collection.update_one = AsyncMock(return_value=SimpleNamespace(modified_count=1))

    result = await db.update_memory(
        "memories", "doc-1", {"status": "updated", "_id": "ignore"}
    )

    assert result is True
    collection.update_one.assert_awaited_once_with(
        {"_id": "doc-1"}, {"$set": {"status": "updated"}}
    )


@pytest.mark.asyncio
async def test_update_memory_missing_document_returns_false(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    collection.find_one = AsyncMock(return_value=None)

    result = await db.update_memory("memories", "doc-1", {"status": "updated"})

    assert result is False
    collection.update_one.assert_not_called()


@pytest.mark.asyncio
async def test_query_by_id_success(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    collection.find_one = AsyncMock(
        return_value={"_id": "doc-1", "text": "hello", "status": "active"}
    )

    result = await db.query_by_id("memories", "doc-1")

    assert result == {
        "memory_id": "doc-1",
        "document": "hello",
        "metadata": {"status": "active", "text": "hello"},
    }


@pytest.mark.asyncio
async def test_query_by_id_not_found_returns_none(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    collection.find_one = AsyncMock(return_value=None)

    result = await db.query_by_id("memories", "doc-1")

    assert result is None


@pytest.mark.asyncio
async def test_delete_from_collection_success(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=1))

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is True
    collection.delete_one.assert_awaited_once_with({"_id": "doc-1"})


@pytest.mark.asyncio
async def test_delete_from_collection_not_found_returns_false(
    monkeypatch, mock_llm_connection, mock_mongodb_client
):
    db, _, _, _, collection, _ = _make_db(
        monkeypatch, mock_llm_connection, mock_mongodb_client
    )
    collection.delete_one = AsyncMock(return_value=SimpleNamespace(deleted_count=0))

    result = await db.delete_from_collection("memories", "doc-1")

    assert result is False
