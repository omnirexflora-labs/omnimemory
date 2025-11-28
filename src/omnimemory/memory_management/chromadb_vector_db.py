import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast
from decouple import config
from omnimemory.core.logger_utils import get_logger
from omnimemory.memory_management.vector_db_base import VectorDBBase
from chromadb import AsyncHttpClient, CloudClient
from chromadb.config import Settings
import numpy as np

if TYPE_CHECKING:
    from typing import Any as _Any

    Collection = _Any
else:
    Collection = Any

logger = get_logger(name="omnimemory.memory_management.chromadb_vector_db")


class ChromaClientType(Enum):
    """Enumeration for ChromaDB client types"""

    REMOTE = "remote"
    CLOUD = "cloud"


class ChromaDBVectorDB(VectorDBBase):
    """
    ChromaDB vector database implementation.

    Supports both remote (AsyncHttpClient) and cloud (CloudClient) modes.
    """

    _LIST_PREFIX: str = "__list__:"
    _DICT_PREFIX: str = "__dict__:"
    _NONE_SENTINEL: str = "__none__"

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize ChromaDB vector database.

        Args:
            **kwargs: Additional parameters including llm_connection and client_type
        """
        super().__init__(**kwargs)

        self.enabled: bool = False
        self.client: Optional[Any] = None

        client_type_str = config("CHROMA_CLIENT_TYPE", default="remote").lower()
        try:
            client_type = ChromaClientType(client_type_str)
        except ValueError:
            logger.warning(
                f"Invalid CHROMA_CLIENT_TYPE '{client_type_str}', defaulting to REMOTE"
            )
            client_type = ChromaClientType.REMOTE

        try:
            if client_type == ChromaClientType.CLOUD:
                cloud_tenant = config("CHROMA_TENANT", default=None)
                cloud_database = config("CHROMA_DATABASE", default=None)
                cloud_api_key = config("CHROMA_API_KEY", default=None)

                if not all([cloud_tenant, cloud_database, cloud_api_key]):
                    logger.error(
                        "ChromaDB Cloud requires CHROMA_TENANT, CHROMA_DATABASE, and CHROMA_API_KEY"
                    )
                    self.enabled = False
                    return

                self.client = CloudClient(
                    tenant=cloud_tenant,
                    database=cloud_database,
                    api_key=cloud_api_key,
                )
                logger.info("ChromaDB Cloud client initialized")

            else:
                chroma_host = config("CHROMA_HOST", default="localhost")
                chroma_port = config("CHROMA_PORT", default=8000, cast=int)

                settings = Settings()
                chroma_auth_token = config("CHROMA_AUTH_TOKEN", default=None)
                if chroma_auth_token:
                    settings.chroma_client_auth_provider = (
                        "chromadb.auth.token_authn.TokenAuthClientProvider"
                    )
                    settings.chroma_client_auth_credentials = chroma_auth_token

                self.client = AsyncHttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=settings,
                )
                logger.info(
                    f"ChromaDB Remote client initialized: {chroma_host}:{chroma_port}"
                )

            if self.llm_connection:
                self.enabled = True
                logger.info("ChromaDBVectorDB initialized successfully")
            else:
                logger.warning(
                    "ChromaDBVectorDB initialized but LLM connection missing"
                )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            self.enabled = False

    async def close(self) -> None:
        """
        Cleanup method to release async connection.

        Closes the ChromaDB client connection and releases resources.
        """
        client = await self._get_client()
        if not client:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            maybe_coro = close_fn()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro

    async def _get_client(self) -> Optional[Any]:
        """
        Return an initialized Chroma client, awaiting coroutine if needed.

        Returns:
            ChromaDB client instance (AsyncHttpClient or CloudClient), or None if unavailable.
        """
        if self.client is None:
            return None
        if asyncio.iscoroutine(self.client):
            try:
                self.client = await self.client
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}")
                self.client = None
        return self.client

    def _serialize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert metadata lists/dicts into strings for ChromaDB storage."""
        if metadata is None:
            return {}
        serialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                serialized[key] = f"{self._LIST_PREFIX}{json.dumps(value)}"
            elif isinstance(value, dict):
                serialized[key] = f"{self._DICT_PREFIX}{json.dumps(value)}"
            elif value is None:
                serialized[key] = self._NONE_SENTINEL
            elif isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    def _deserialize_metadata(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Decode serialized metadata back to Python objects."""
        if metadata is None:
            return {}
        deserialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str) and value.startswith(self._LIST_PREFIX):
                payload = value[len(self._LIST_PREFIX) :]
                try:
                    deserialized[key] = json.loads(payload)
                except json.JSONDecodeError:
                    deserialized[key] = value
            elif isinstance(value, str) and value.startswith(self._DICT_PREFIX):
                payload = value[len(self._DICT_PREFIX) :]
                try:
                    deserialized[key] = json.loads(payload)
                except json.JSONDecodeError:
                    deserialized[key] = value
            elif value == self._NONE_SENTINEL:
                deserialized[key] = None
            else:
                deserialized[key] = value
        return deserialized

    def _ensure_scalar_metadata(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ensure all metadata values are scalars supported by Chroma."""
        if metadata is None:
            return {}
        safe: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            else:
                safe[key] = str(value)
        return safe

    async def _ensure_collection(self, collection_name: str) -> Optional["Collection"]:
        """
        Ensure the collection exists, create if it doesn't.

        Args:
            collection_name: Name of the collection

        Returns:
            ChromaDB Collection instance, or None if client is unavailable.

        Raises:
            Exception: If collection creation fails.
        """
        client = await self._get_client()
        if not client:
            logger.warning("ChromaDB is not enabled. Cannot ensure collection.")
            return None
        try:
            collection = await client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"initialized ChromaDB collection: {collection_name}")
            return collection

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise

    async def add_to_collection(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> bool:
        """Add document with embedding to collection.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID
            document: Document content
            embedding: Pre-computed embedding vector
            metadata: Document metadata

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()
        if not client:
            logger.warning("ChromaDB is not enabled. Cannot add to collection.")
            return False

        if not embedding or not isinstance(embedding, list):
            logger.error("Valid embedding vector is required")
            return False

        try:
            collection = await self._ensure_collection(collection_name)
            if collection is None:
                return False
            metadata = metadata or {}
            metadata_copy = self._serialize_metadata(metadata.copy())
            metadata_copy["text"] = document
            metadata_copy = self._ensure_scalar_metadata(metadata_copy)

            await collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata_copy],
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to add document to collection: {e}. metadata={metadata}",
                exc_info=True,
            )
            return False

    async def query_collection(
        self,
        collection_name: str,
        query: str,
        n_results: int,
        similarity_threshold: float,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query collection with flexible filtering .

        Args:
            collection_name: Name of the collection to query
            query: Query string for semantic search
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            filter_conditions: Dictionary of metadata field filters

        Returns:
            Dictionary with documents, scores, metadatas, and ids
        """
        client = await self._get_client()
        if not client:
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        try:
            collection = await self._ensure_collection(collection_name)
            if collection is None:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            query_embedding = await self.embed_text(query)

            where_clause = None
            if filter_conditions:
                clauses = []
                for field_name, field_value in filter_conditions.items():
                    if field_value is None:
                        continue
                    clauses.append({field_name: {"$eq": field_value}})
                if clauses:
                    where_clause = {"$and": clauses} if len(clauses) > 1 else clauses[0]

            expanded_limit = min(n_results * 5, 1000)

            results = await collection.query(
                query_embeddings=[query_embedding],
                n_results=expanded_limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0] or []
            metadatas_raw = results.get("metadatas", [[]])[0] or []
            metadatas = []
            for meta in metadatas_raw:
                deserialized = self._deserialize_metadata(meta)
                deserialized.pop("text", None)
                metadatas.append(deserialized)
            ids = results.get("ids", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []

            scores = (
                [max(0.0, min(1.0, 1.0 - (d / 2.0))) for d in distances]
                if distances
                else []
            )

            filtered_results = [
                (doc, score, meta, doc_id)
                for doc, score, meta, doc_id in zip(documents, scores, metadatas, ids)
                if score >= similarity_threshold
            ]

            limited_results = filtered_results[:n_results]

            return {
                "documents": [r[0] for r in limited_results],
                "scores": [r[1] for r in limited_results],
                "metadatas": [r[2] for r in limited_results],
                "ids": [r[3] for r in limited_results],
            }

        except Exception as e:
            if (
                "404" in str(e)
                or "doesn't exist" in str(e)
                or "not found" in str(e).lower()
            ):
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            logger.error(f"Failed to query ChromaDB: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def query_by_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        n_results: int,
        filter_conditions: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Query collection using an embedding vector directly.

        Args:
            collection_name: Name of the collection to query
            embedding: Embedding vector to search with
            n_results: Number of results to return
            filter_conditions: Dictionary of metadata field filters
            similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.0 for no threshold)

        Returns:
            Dictionary with documents, scores, metadatas, and ids
        """
        client = await self._get_client()
        if not client:
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        if not embedding or not isinstance(embedding, list):
            logger.error("Valid embedding vector is required")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        try:
            collection = await self._ensure_collection(collection_name)
            if collection is None:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            where_clause = None
            if filter_conditions:
                clauses = []
                for field_name, field_value in filter_conditions.items():
                    if field_value is None:
                        continue
                    clauses.append({field_name: {"$eq": field_value}})
                if clauses:
                    where_clause = {"$and": clauses} if len(clauses) > 1 else clauses[0]

            limit = n_results if n_results is not None else 10000
            expanded_limit = min(limit * 3, 10000) if limit else 10000

            results = await collection.query(
                query_embeddings=[embedding],
                n_results=expanded_limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0] or []
            metadatas_raw = results.get("metadatas", [[]])[0] or []
            metadatas = []
            for meta in metadatas_raw:
                deserialized = self._deserialize_metadata(meta)
                deserialized.pop("text", None)
                metadatas.append(deserialized)
            ids = results.get("ids", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []

            scores = (
                [max(0.0, min(1.0, 1.0 - (d / 2.0))) for d in distances]
                if distances
                else []
            )

            filtered_results = [
                (doc, score, meta, doc_id)
                for doc, score, meta, doc_id in zip(documents, scores, metadatas, ids)
                if score >= similarity_threshold
            ]

            limited_results = filtered_results[:limit] if limit else filtered_results

            return {
                "documents": [r[0] for r in limited_results],
                "scores": [r[1] for r in limited_results],
                "metadatas": [r[2] for r in limited_results],
                "ids": [r[3] for r in limited_results],
            }

        except Exception as e:
            if (
                "404" in str(e)
                or "doesn't exist" in str(e)
                or "not found" in str(e).lower()
            ):
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            logger.error(f"Failed to query by embedding: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def update_memory(
        self,
        collection_name: str,
        doc_id: str,
        update_payload: Dict[str, Any],
    ) -> bool:
        """Update a memory document's metadata in ChromaDB.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to update
            update_payload: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()
        if not self.enabled or not client:
            logger.warning("ChromaDBVectorDB is disabled")
            return False

        try:
            collection = await self._ensure_collection(collection_name)
            if collection is None:
                return False

            try:
                existing = await asyncio.wait_for(
                    collection.get(
                        ids=[doc_id], include=["embeddings", "documents", "metadatas"]
                    ),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout retrieving document {doc_id} for update in collection {collection_name}"
                )
                return False
            except Exception as retrieve_error:
                error_str = str(retrieve_error).lower()
                if "not found" in error_str or "404" in error_str:
                    logger.warning(
                        f"Document {doc_id} not found in collection {collection_name}"
                    )
                    return False
                if "timeout" in error_str:
                    logger.warning(
                        f"Request timeout retrieving document {doc_id} for update: {retrieve_error}"
                    )
                    return False
                raise

            def _to_list(val: Any) -> List[Any]:
                if val is None:
                    return []
                if isinstance(val, np.ndarray):
                    return cast(List[Any], val.tolist())
                if isinstance(val, list):
                    return val
                if hasattr(val, "tolist"):
                    tolist_fn = getattr(val, "tolist")
                    if callable(tolist_fn):
                        converted = tolist_fn()
                        if isinstance(converted, list):
                            return converted
                        return list(converted)
                return [val]

            ids = _to_list(existing.get("ids"))
            documents = _to_list(existing.get("documents"))
            metadatas = _to_list(existing.get("metadatas"))
            embeddings = _to_list(existing.get("embeddings"))

            if not ids or len(ids) == 0:
                logger.warning(
                    f"Document {doc_id} not found in collection {collection_name}"
                )
                return False

            existing_document = documents[0] if documents else ""
            existing_embedding_raw = embeddings[0] if embeddings else None
            existing_metadata_raw = metadatas[0] if metadatas else {}

            existing_embedding: Optional[List[float]]
            if existing_embedding_raw is None:
                existing_embedding = None
            elif isinstance(existing_embedding_raw, list):
                existing_embedding = existing_embedding_raw
            elif isinstance(existing_embedding_raw, np.ndarray):
                existing_embedding = existing_embedding_raw.tolist()
            elif hasattr(existing_embedding_raw, "tolist"):
                tolist_fn = getattr(existing_embedding_raw, "tolist")
                if callable(tolist_fn):
                    converted = tolist_fn()
                    if isinstance(converted, list):
                        existing_embedding = converted
                    else:
                        existing_embedding = list(converted)
                else:
                    existing_embedding = list(existing_embedding_raw)
            else:
                existing_embedding = list(existing_embedding_raw)

            if existing_embedding is None:
                logger.error(f"No embedding found for document {doc_id}")
                return False

            existing_metadata = (
                self._deserialize_metadata(existing_metadata_raw)
                if existing_metadata_raw
                else {}
            )
            update_payload = update_payload or {}
            updated_metadata = {**existing_metadata, **update_payload}
            serialized_metadata = self._serialize_metadata(updated_metadata)
            serialized_metadata["text"] = existing_document
            serialized_metadata = self._ensure_scalar_metadata(serialized_metadata)

            await collection.upsert(
                ids=[doc_id],
                embeddings=[existing_embedding],
                documents=[existing_document],
                metadatas=[serialized_metadata],
            )

            return True

        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str:
                logger.warning(
                    f"Timeout updating memory {doc_id} in collection {collection_name}"
                )
            else:
                logger.error(
                    f"Failed to update memory {doc_id} in collection {collection_name}: {e}"
                )
            return False

    async def query_by_id(
        self,
        collection_name: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Query a specific document by ID in ChromaDB.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to query

        Returns:
            Dictionary with document data, or None if not found
        """
        client = await self._get_client()
        if not self.enabled or not client:
            logger.warning("ChromaDBVectorDB is disabled")
            return None

        try:
            collection = await self._ensure_collection(collection_name)
            if collection is None:
                return None

            try:
                results = await asyncio.wait_for(
                    collection.get(ids=[doc_id], include=["documents", "metadatas"]),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout querying document {doc_id} in collection {collection_name}"
                )
                return None
            except Exception as retrieve_error:
                error_str = str(retrieve_error)
                if "timeout" in error_str.lower():
                    logger.warning(
                        f"Request timeout querying document {doc_id} in collection {collection_name}"
                    )
                    return None
                raise

            if not results.get("ids") or len(results["ids"]) == 0:
                return None

            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            document = documents[0] if documents else ""
            metadata_raw = metadatas[0] if metadatas else {}

            metadata_copy = self._deserialize_metadata(metadata_raw)
            metadata_copy.pop("text", None)

            return {
                "memory_id": doc_id,
                "document": document,
                "metadata": metadata_copy,
            }

        except Exception as e:
            error_str = str(e)
            if "timeout" in error_str.lower():
                logger.warning(
                    f"Request timeout querying document {doc_id} in collection {collection_name}"
                )
                return None
            logger.error(
                f"Failed to query document {doc_id} in collection {collection_name}: {e}"
            )
            return None

    async def delete_from_collection(
        self,
        collection_name: str,
        doc_id: str,
    ) -> bool:
        """Delete a document from collection by ID.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()
        if not client:
            logger.warning("ChromaDB is not enabled. Cannot delete from collection.")
            return False

        try:
            collection = await self._ensure_collection(collection_name)
            if collection is None:
                return False

            await collection.delete(ids=[doc_id])

            return True

        except Exception as e:
            logger.error(
                f"Failed to delete document {doc_id} from collection {collection_name}: {e}"
            )
            return False
