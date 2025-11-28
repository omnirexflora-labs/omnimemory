import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Set
from decouple import config
from omnimemory.core.logger_utils import get_logger
from omnimemory.memory_management.vector_db_base import VectorDBBase
import asyncpg


logger = get_logger(name="omnimemory.memory_management.postgresql_vector_db")

DB_TIMEOUT_SECONDS = 30.0
ALLOWED_FILTER_FIELDS = {"app_id", "user_id", "session_id", "status"}


def _sanitize_identifier(identifier: str) -> str:
    """Sanitize SQL identifier to prevent SQL injection.

    Only allows alphanumeric characters, underscores, and hyphens.
    Normalizes to lowercase for consistency.
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")
    if not re.match(r"^[a-zA-Z0-9_-]+$", identifier):
        raise ValueError(
            f"Invalid identifier '{identifier}': only alphanumeric, underscore, and hyphen allowed"
        )
    return identifier.lower()


class PostgreSQLVectorDB(VectorDBBase):
    """
    PostgreSQL with pgvector extension implementation.

    Provides vector database operations using PostgreSQL's pgvector extension
    with HNSW indexing for efficient similarity search.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize PostgreSQL vector database connection.

        Args:
            **kwargs: Additional parameters including llm_connection
        """
        super().__init__(**kwargs)
        self.enabled: bool = False
        self.pool: Optional[asyncpg.Pool] = None
        self._created_collections: Set[str] = set()
        self.connection_string: Optional[str] = None

        postgres_uri = config("POSTGRES_URI", default=None)
        self.db_name: str = config("POSTGRES_DB_NAME", default="omnimemory")

        if not postgres_uri:
            logger.warning("POSTGRES_URI not set. PostgreSQL will be disabled.")
            self.enabled = False
            return

        try:
            self.connection_string = postgres_uri

            if self.llm_connection:
                self.enabled = True
                logger.info("PostgreSQLVectorDB initialized successfully")
            else:
                logger.warning(
                    "PostgreSQLVectorDB initialized but LLM connection missing"
                )
                self.enabled = False

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}", exc_info=True)
            self.enabled = False

    async def _get_pool(self) -> asyncpg.Pool:
        """
        Get or create the PostgreSQL connection pool.

        Returns:
            asyncpg.Pool instance.

        Raises:
            Exception: If pool creation fails.
        """
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=1,
                    max_size=10,
                )
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL connection pool: {e}")
                raise
        return self.pool

    async def close(self) -> None:
        """
        Close the PostgreSQL connection pool.

        Releases all connections and cleans up resources.
        """
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def _ensure_collection(self, collection_name: str) -> None:
        """
        Ensure the collection (table) exists, create if it doesn't.

        Creates table with pgvector extension, HNSW index, and GIN metadata index.

        Args:
            collection_name: Name of the collection to ensure.

        Raises:
            ValueError: If vector size is invalid.
            asyncio.TimeoutError: If operation times out.
            Exception: If collection setup fails.
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] PostgreSQL is disabled")
            return

        if collection_name in self._created_collections:
            return

        safe_collection_name = _sanitize_identifier(collection_name)
        actual_vector_size = self._get_embedding_dimensions()
        if not isinstance(actual_vector_size, int) or actual_vector_size <= 0:
            raise ValueError(
                f"Vector size must be positive integer. Got: {actual_vector_size}"
            )

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await asyncio.wait_for(
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector"),
                    timeout=DB_TIMEOUT_SECONDS,
                )

                table_exists = await asyncio.wait_for(
                    conn.fetchval(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                        safe_collection_name,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )

                if not table_exists:
                    await asyncio.wait_for(
                        conn.execute(
                            f'CREATE TABLE IF NOT EXISTS "{safe_collection_name}" ('
                            f"    id TEXT PRIMARY KEY,"
                            f"    text TEXT NOT NULL,"
                            f"    embedding vector({actual_vector_size}),"
                            f"    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb"
                            f")"
                        ),
                        timeout=DB_TIMEOUT_SECONDS,
                    )
                    logger.debug(f"[{collection_name}] Created table")

                safe_index_name = _sanitize_identifier(
                    f"{collection_name}_embedding_idx"
                )
                index_exists = await asyncio.wait_for(
                    conn.fetchval(
                        "SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname = $1)",
                        safe_index_name,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )

                if not index_exists:
                    await asyncio.wait_for(
                        conn.execute(
                            f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" '
                            f'ON "{safe_collection_name}" '
                            f"USING hnsw (embedding vector_cosine_ops) "
                            f"WITH (m = 16, ef_construction = 200)"
                        ),
                        timeout=DB_TIMEOUT_SECONDS * 2,
                    )
                    logger.info(
                        f"[{collection_name}] Created HNSW index ({actual_vector_size}D)"
                    )

                safe_metadata_index = _sanitize_identifier(
                    f"{collection_name}_metadata_idx"
                )
                metadata_index_exists = await asyncio.wait_for(
                    conn.fetchval(
                        "SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname = $1)",
                        safe_metadata_index,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )

                if not metadata_index_exists:
                    await asyncio.wait_for(
                        conn.execute(
                            f'CREATE INDEX IF NOT EXISTS "{safe_metadata_index}" '
                            f'ON "{safe_collection_name}" '
                            f"USING gin (metadata)"
                        ),
                        timeout=DB_TIMEOUT_SECONDS,
                    )
                    logger.debug(f"[{collection_name}] Created GIN metadata index")

                self._created_collections.add(collection_name)

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout during collection setup")
            raise
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to initialize collection: {e}")
            raise

    async def add_to_collection(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Add document with embedding to collection .

        Args:
            collection_name: Name of the collection.
            doc_id: Document ID.
            document: Document content.
            embedding: Embedding vector for the document.
            metadata: Document metadata.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] PostgreSQL is disabled")
            return False
        if not embedding or not isinstance(embedding, list):
            logger.error(f"[{collection_name}] Valid embedding required for {doc_id}")
            return False

        try:
            await self._ensure_collection(collection_name)
            pool = await self._get_pool()
            safe_collection_name = _sanitize_identifier(collection_name)

            metadata_copy = {**metadata, "text": document}
            formatted_embedding = self._format_embedding(embedding)

            async with pool.acquire() as conn:
                await asyncio.wait_for(
                    conn.execute(
                        f'INSERT INTO "{safe_collection_name}" (id, text, embedding, metadata) '
                        f"VALUES ($1, $2, $3::vector, $4::jsonb) "
                        f"ON CONFLICT (id) DO UPDATE SET "
                        f"    text = EXCLUDED.text, "
                        f"    embedding = EXCLUDED.embedding, "
                        f"    metadata = EXCLUDED.metadata",
                        doc_id,
                        document,
                        formatted_embedding,
                        json.dumps(metadata_copy),
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
            return True

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout adding document {doc_id}")
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to add document {doc_id}: {e}")
        return False

    def _format_embedding(self, embedding: List[float]) -> str:
        """
        Format embedding vector as PostgreSQL vector literal string.

        Args:
            embedding: List of float values representing the embedding.

        Returns:
            PostgreSQL vector literal string (e.g., "[0.1,0.2,0.3]").

        Raises:
            ValueError: If embedding is empty or contains invalid values.
        """
        if not embedding:
            raise ValueError("Embedding must not be empty")
        try:
            return "[" + ",".join(f"{float(val):.15g}" for val in embedding) + "]"
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid embedding value: {e}")

    async def _vector_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int,
        similarity_threshold: float,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform vector similarity search using pgvector.

        Args:
            collection_name: Name of the collection to search.
            query_embedding: Query embedding vector.
            n_results: Number of results to return.
            similarity_threshold: Minimum similarity score (0.0-1.0).
            filter_conditions: Optional metadata filters (whitelisted fields only).

        Returns:
            Dictionary with documents, scores, metadatas, and ids.
        """
        if not self.enabled:
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        try:
            await self._ensure_collection(collection_name)
            pool = await self._get_pool()
            safe_collection_name = _sanitize_identifier(collection_name)

            where_clauses = []
            formatted_query = self._format_embedding(query_embedding)
            params = [formatted_query]
            param_idx = 2

            if filter_conditions:
                for field, value in filter_conditions.items():
                    if field not in ALLOWED_FILTER_FIELDS:
                        logger.warning(
                            f"[{collection_name}] Ignoring disallowed filter: {field}"
                        )
                        continue
                    if value is not None:
                        where_clauses.append(f"metadata->>'{field}' = ${param_idx}")
                        params.append(str(value))
                        param_idx += 1

            where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
            expanded_limit = min(n_results * 5, 1000)
            params.append(str(expanded_limit))

            query_sql = f"""
                SELECT 
                    id,
                    text,
                    metadata,
                    1 - (embedding <=> $1::vector) / 2.0 AS similarity
                FROM "{safe_collection_name}"
                WHERE {where_sql}
                ORDER BY embedding <=> $1::vector
                LIMIT ${param_idx}
            """

            async with pool.acquire() as conn:
                rows = await asyncio.wait_for(
                    conn.fetch(query_sql, *params), timeout=DB_TIMEOUT_SECONDS
                )

            if not rows:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            filtered = [r for r in rows if r["similarity"] >= similarity_threshold]
            limited = filtered[:n_results]

            metadatas = []
            for row in limited:
                meta = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
                metadatas.append(meta)

            return {
                "documents": [r["text"] for r in limited],
                "scores": [float(r["similarity"]) for r in limited],
                "metadatas": metadatas,
                "ids": [r["id"] for r in limited],
            }

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Query timeout")
        except Exception as e:
            msg = str(e).lower()
            if "does not exist" in msg or "relation" in msg:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            logger.error(f"[{collection_name}] Query failed: {e}")
        return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def query_collection(
        self,
        collection_name: str,
        query: str,
        n_results: int,
        similarity_threshold: float,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query collection with flexible filtering .

        Args:
            collection_name: Name of the collection to query.
            query: Query string for semantic search.
            n_results: Number of results to return.
            similarity_threshold: Minimum similarity score (0.0-1.0).
            filter_conditions: Dictionary of metadata field filters.

        Returns:
            Dictionary with documents, scores, metadatas, and ids.
        """
        if not self.llm_connection:
            logger.error(f"[{collection_name}] LLM connection required for embedding")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}
        try:
            query_embedding = await self.embed_text(query)
            return await self._vector_search(
                collection_name,
                query_embedding,
                n_results,
                similarity_threshold,
                filter_conditions,
            )
        except Exception as e:
            logger.error(f"[{collection_name}] Embedding failed: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def query_by_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        n_results: int,
        filter_conditions: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Query collection using an embedding vector directly.

        Args:
            collection_name: Name of the collection to query.
            embedding: Embedding vector to search with.
            n_results: Number of results to return.
            filter_conditions: Dictionary of metadata field filters.
            similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.0).

        Returns:
            Dictionary with documents, scores, metadatas, and ids.
        """
        return await self._vector_search(
            collection_name,
            embedding,
            n_results,
            similarity_threshold,
            filter_conditions,
        )

    async def update_memory(
        self,
        collection_name: str,
        doc_id: str,
        update_payload: Dict[str, Any],
    ) -> bool:
        """
        Update a memory document's metadata.

        Args:
            collection_name: Name of the collection.
            doc_id: Document ID to update.
            update_payload: Dictionary of fields to update.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] PostgreSQL is disabled")
            return False

        try:
            await self._ensure_collection(collection_name)
            pool = await self._get_pool()
            safe_collection_name = _sanitize_identifier(collection_name)

            async with pool.acquire() as conn:
                row = await asyncio.wait_for(
                    conn.fetchrow(
                        f'SELECT metadata FROM "{safe_collection_name}" WHERE id = $1',
                        doc_id,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
                if not row:
                    logger.warning(f"[{collection_name}] Document {doc_id} not found")
                    return False

                existing_meta = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
                updated_meta = {**existing_meta, **update_payload}

                result = await asyncio.wait_for(
                    conn.execute(
                        f'UPDATE "{safe_collection_name}" SET metadata = $1::jsonb WHERE id = $2',
                        json.dumps(updated_meta),
                        doc_id,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
                return str(result) == "UPDATE 1"

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout updating {doc_id}")
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to update {doc_id}: {e}")
        return False

    async def query_by_id(
        self,
        collection_name: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query a specific document by ID.

        Args:
            collection_name: Name of the collection.
            doc_id: Document ID to query.

        Returns:
            Dictionary with document data, or None if not found.
        """
        if not self.enabled:
            return None

        try:
            await self._ensure_collection(collection_name)
            pool = await self._get_pool()
            safe_collection_name = _sanitize_identifier(collection_name)

            async with pool.acquire() as conn:
                row = await asyncio.wait_for(
                    conn.fetchrow(
                        f'SELECT id, text, metadata FROM "{safe_collection_name}" WHERE id = $1',
                        doc_id,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
                if not row:
                    return None

                metadata = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )
                return {
                    "memory_id": row["id"],
                    "document": row["text"],
                    "metadata": metadata,
                }

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout querying {doc_id}")
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to query {doc_id}: {e}")
        return None

    async def delete_from_collection(
        self,
        collection_name: str,
        doc_id: str,
    ) -> bool:
        """
        Delete a document from collection by ID.

        Args:
            collection_name: Name of the collection.
            doc_id: Document ID to delete.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] PostgreSQL is disabled")
            return False

        try:
            await self._ensure_collection(collection_name)
            pool = await self._get_pool()
            safe_collection_name = _sanitize_identifier(collection_name)

            async with pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.execute(
                        f'DELETE FROM "{safe_collection_name}" WHERE id = $1',
                        doc_id,
                    ),
                    timeout=DB_TIMEOUT_SECONDS,
                )
                return str(result) == "DELETE 1"

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout deleting {doc_id}")
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to delete {doc_id}: {e}")
        return False
