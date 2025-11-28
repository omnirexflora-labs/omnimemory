import json
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from omnimemory.memory_management.postgresql_vector_db import PostgreSQLVectorDB


def _fake_config_factory(monkeypatch, overrides=None):
    values = {
        "POSTGRES_URI": "postgresql://user:pass@localhost:5432/omnimemory",
        "POSTGRES_DB_NAME": "omnimemory",
    }
    if overrides:
        values.update(overrides)

    def fake_config(key, default=None, cast=None):
        value = values.get(key, default)
        if cast and value is not None:
            return cast(value)
        return value if value is not None else default

    monkeypatch.setattr(
        "omnimemory.memory_management.postgresql_vector_db.config", fake_config
    )


class _AsyncAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def mock_postgres(monkeypatch):
    connection = MagicMock()
    connection.execute = AsyncMock(return_value="OK")
    connection.fetch = AsyncMock(return_value=[])
    connection.fetchrow = AsyncMock(return_value=None)
    connection.fetchval = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncAcquire(connection))
    pool.close = AsyncMock()

    create_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(
        "omnimemory.memory_management.postgresql_vector_db.asyncpg.create_pool",
        create_pool,
    )

    return {"pool_ctor": create_pool, "pool": pool, "connection": connection}


def _make_db(monkeypatch, mock_llm_connection, pg_fixture, overrides=None):
    _fake_config_factory(monkeypatch, overrides)
    db = PostgreSQLVectorDB(llm_connection=mock_llm_connection)
    return {"db": db, **pg_fixture}


class TestInitialization:
    def test_init_with_uri_enables_client(
        self, monkeypatch, mock_llm_connection, mock_postgres
    ):
        deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
        db = deps["db"]

        deps["pool_ctor"].assert_not_called()
        assert db.enabled is True
        assert "postgresql://" in db.connection_string

    def test_init_missing_uri_disables(
        self, monkeypatch, mock_llm_connection, mock_postgres
    ):
        deps = _make_db(
            monkeypatch,
            mock_llm_connection,
            mock_postgres,
            overrides={"POSTGRES_URI": None},
        )
        db = deps["db"]

        assert db.enabled is False

    def test_init_without_llm_connection_disables(self, monkeypatch, mock_postgres):
        _fake_config_factory(monkeypatch)
        db = PostgreSQLVectorDB()

        assert db.enabled is False


@pytest.mark.asyncio
async def test_close_calls_pool_close(monkeypatch, mock_llm_connection, mock_postgres):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.pool = deps["pool"]

    await db.close()

    deps["pool"].close.assert_awaited_once()
    assert db.pool is None


@pytest.mark.asyncio
async def test_add_to_collection_success(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True
    db._ensure_collection = AsyncMock()
    db._get_pool = AsyncMock(return_value=deps["pool"])

    result = await db.add_to_collection(
        "memories", "doc-1", "hello", [0.1, 0.2], {"app_id": "a1"}
    )

    assert result is True
    insert_call = deps["connection"].execute.await_args_list[-1].args
    assert insert_call[3] == "[0.1,0.2]"
    assert json.loads(insert_call[4])["text"] == "hello"


@pytest.mark.asyncio
async def test_add_to_collection_invalid_embedding_returns_false(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True

    assert (await db.add_to_collection("memories", "doc-1", "text", None, {})) is False


@pytest.mark.asyncio
async def test_vector_search_returns_filtered_rows(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True
    db._ensure_collection = AsyncMock()
    db._get_pool = AsyncMock(return_value=deps["pool"])

    deps["connection"].fetch = AsyncMock(
        return_value=[
            {
                "id": "1",
                "text": "keep",
                "metadata": {"app_id": "a1"},
                "similarity": 0.9,
            },
            {
                "id": "2",
                "text": "drop",
                "metadata": {"app_id": "a1"},
                "similarity": 0.1,
            },
        ]
    )

    results = await db._vector_search(
        "memories",
        query_embedding=[0.1, 0.2],
        n_results=1,
        similarity_threshold=0.5,
        filter_conditions={"app_id": "a1"},
    )

    assert results["documents"] == ["keep"]
    fetch_args = deps["connection"].fetch.await_args_list[-1].args
    assert fetch_args[1] == "[0.1,0.2]"


@pytest.mark.asyncio
async def test_query_collection_requires_llm_connection(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.llm_connection = None

    result = await db.query_collection("memories", "query", 5, 0.5)

    assert result == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_query_by_embedding_delegates_to_vector_search(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True
    db._vector_search = AsyncMock(return_value={"ids": ["1"]})

    result = await db.query_by_embedding("memories", [0.1], 3)

    assert result == {"ids": ["1"]}
    db._vector_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_memory_success(monkeypatch, mock_llm_connection, mock_postgres):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True
    db._ensure_collection = AsyncMock()
    db._get_pool = AsyncMock(return_value=deps["pool"])

    deps["connection"].fetchrow = AsyncMock(
        return_value={"metadata": json.dumps({"status": "old"})}
    )
    deps["connection"].execute = AsyncMock(return_value="UPDATE 1")

    result = await db.update_memory("memories", "doc-1", {"status": "new"})

    assert result is True
    args = deps["connection"].execute.await_args_list[-1].args
    assert json.loads(args[1])["status"] == "new"


@pytest.mark.asyncio
async def test_query_by_id_success(monkeypatch, mock_llm_connection, mock_postgres):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True
    db._ensure_collection = AsyncMock()
    db._get_pool = AsyncMock(return_value=deps["pool"])

    deps["connection"].fetchrow = AsyncMock(
        return_value={
            "id": "doc-1",
            "text": "hello",
            "metadata": json.dumps({"status": "active"}),
        }
    )

    result = await db.query_by_id("memories", "doc-1")

    assert result == {
        "memory_id": "doc-1",
        "document": "hello",
        "metadata": {"status": "active"},
    }


@pytest.mark.asyncio
async def test_delete_from_collection_success(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True
    db._ensure_collection = AsyncMock()
    db._get_pool = AsyncMock(return_value=deps["pool"])
    deps["connection"].execute = AsyncMock(return_value="DELETE 1")

    assert (await db.delete_from_collection("memories", "doc-1")) is True


@pytest.mark.asyncio
async def test_ensure_collection_disabled_returns_early(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = False

    await db._ensure_collection("memories")


@pytest.mark.asyncio
async def test_ensure_collection_invalid_vector_size(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True
    db._get_embedding_dimensions = Mock(return_value=None)

    with pytest.raises(ValueError):
        await db._ensure_collection("memories")


@pytest.mark.asyncio
async def test_query_by_embedding_invalid_embedding_returns_empty(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = True

    result = await db.query_by_embedding("memories", None, 5)

    assert result == {"documents": [], "scores": [], "metadatas": [], "ids": []}


@pytest.mark.asyncio
async def test_delete_from_collection_disabled_returns_false(
    monkeypatch, mock_llm_connection, mock_postgres
):
    deps = _make_db(monkeypatch, mock_llm_connection, mock_postgres)
    db = deps["db"]
    db.enabled = False

    assert (await db.delete_from_collection("memories", "doc-1")) is False
