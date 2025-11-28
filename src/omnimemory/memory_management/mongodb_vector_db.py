import asyncio
import struct
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from bson import Binary
from decouple import config
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from omnimemory.core.logger_utils import get_logger
from omnimemory.memory_management.vector_db_base import VectorDBBase

if TYPE_CHECKING:
    pass

logger = get_logger(name="omnimemory.memory_management.mongodb_vector_db")

DB_TIMEOUT_SECONDS = 30.0
ALLOWED_FILTER_FIELDS = {"app_id", "user_id", "session_id", "status"}
PROTECTED_UPDATE_FIELDS = {"_id", "embedding", "text"}

USE_BINARY_VECTORS = False


class MongoDBVectorDB(VectorDBBase):
    """
    MongoDB Atlas Vector Search implementation.

    Provides vector database operations using MongoDB Atlas with vector search indexes.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize MongoDB vector database.

        Args:
            **kwargs: Additional parameters including llm_connection
        """
        super().__init__(**kwargs)

        self.enabled: bool = False
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

        mongodb_uri = config("MONGODB_URI", default=None)
        self.db_name: str = config("MONGODB_DB_NAME", default="omnimemory")

        if mongodb_uri:
            try:
                self.client = AsyncIOMotorClient(mongodb_uri)
                self.db = self.client[self.db_name]
                if self.llm_connection:
                    self.enabled = True
                    logger.info("MongoDBVectorDB initialized successfully")
                else:
                    logger.warning(
                        "MongoDBVectorDB initialized but LLM connection missing"
                    )
                    self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB client: {e}", exc_info=True)
                self.enabled = False
        else:
            logger.warning("MONGODB_URI not set. MongoDB will be disabled.")
            self.enabled = False

    async def close(self) -> None:
        """
        Cleanup method to release async connection.

        Closes the MongoDB client connection and releases resources.
        """
        if self.client:
            self.client.close()

    async def _ensure_collection(self, collection_name: str) -> None:
        """
        Ensure the collection and vector search index exist.

        Args:
            collection_name: Name of the collection to ensure.

        Raises:
            ValueError: If vector size is invalid.
            asyncio.TimeoutError: If operation times out.
            Exception: If collection/index setup fails.
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] MongoDB is not enabled")
            return

        actual_vector_size = self._get_embedding_dimensions()
        if not isinstance(actual_vector_size, int) or actual_vector_size <= 0:
            raise ValueError(
                f"Vector size must be a positive integer. Got: {actual_vector_size}"
            )

        if self.db is None:
            raise RuntimeError("MongoDB database is not initialized")
        try:
            collections = await asyncio.wait_for(
                self.db.list_collection_names(), timeout=DB_TIMEOUT_SECONDS
            )
            if collection_name not in collections:
                await asyncio.wait_for(
                    self.db.create_collection(collection_name),
                    timeout=DB_TIMEOUT_SECONDS,
                )
                logger.debug(f"[{collection_name}] Created new collection")

            await self._create_vector_search_index(collection_name, actual_vector_size)

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout ensuring collection")
            raise
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to initialize collection: {e}")
            raise

    async def _create_vector_search_index(
        self, collection_name: str, dimensions: int
    ) -> None:
        """
        Create vector search index idempotently.

        Args:
            collection_name: Name of the collection.
            dimensions: Vector embedding dimensions.

        Raises:
            OperationFailure: If index creation fails (non-idempotent errors).
            asyncio.TimeoutError: If operation times out.
            Exception: If unexpected error occurs.
        """
        if self.db is None:
            raise RuntimeError("MongoDB database is not initialized")
        collection = self.db[collection_name]
        index_name = f"vector_index_{collection_name}"

        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": dimensions,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "app_id"},
                    {"type": "filter", "path": "user_id"},
                    {"type": "filter", "path": "session_id"},
                    {"type": "filter", "path": "status"},
                ]
            },
            name=index_name,
            type="vectorSearch",
        )

        try:
            await asyncio.wait_for(
                collection.create_search_index(model=search_index_model),
                timeout=DB_TIMEOUT_SECONDS * 2,
            )
            logger.info(
                f"[{collection_name}] Created vector index '{index_name}' ({dimensions}D)"
            )
        except OperationFailure as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.debug(
                    f"[{collection_name}] Vector index '{index_name}' already exists"
                )
            else:
                logger.error(f"[{collection_name}] Index creation failed: {e}")
                raise
        except asyncio.TimeoutError:
            logger.warning(
                f"[{collection_name}] Timeout during index creation (may be in progress)"
            )
        except Exception as e:
            logger.error(f"[{collection_name}] Unexpected error creating index: {e}")
            raise

    def _prepare_embedding(self, embedding: List[float]) -> Union[List[float], Binary]:
        """
        Optionally convert float list to binary for efficiency.

        Args:
            embedding: List of float values representing the embedding vector.

        Returns:
            Either the original list or a Binary object (if USE_BINARY_VECTORS is enabled).
        """
        if USE_BINARY_VECTORS:
            return Binary(struct.pack(f"{len(embedding)}f", *embedding), subtype=9)
        return embedding

    def _extract_embedding(self, stored: Union[List[float], Binary]) -> List[float]:
        """
        Reverse of _prepare_embedding (not currently used in read path).

        Args:
            stored: Either a list of floats or a Binary object.

        Returns:
            List of float values representing the embedding vector.
        """
        if isinstance(stored, Binary) and stored.subtype == 9:
            return list(struct.unpack(f"{len(stored) // 4}f", stored))
        return stored  # type: ignore[return-value]

    async def add_to_collection(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Add document with embedding to collection.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID
            document: Document content
            embedding: Pre-computed embedding vector
            metadata: Document metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] MongoDB is disabled")
            return False

        if not embedding or not isinstance(embedding, list):
            logger.error(
                f"[{collection_name}] Valid embedding required for doc {doc_id}"
            )
            return False

        if self.db is None:
            raise RuntimeError("MongoDB database is not initialized")
        try:
            await self._ensure_collection(collection_name)
            collection = self.db[collection_name]

            doc = {
                "_id": doc_id,
                "text": document,
                "embedding": self._prepare_embedding(embedding),
                **(metadata or {}),
            }

            await asyncio.wait_for(
                collection.replace_one({"_id": doc_id}, doc, upsert=True),
                timeout=DB_TIMEOUT_SECONDS,
            )
            return True

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout adding document {doc_id}")
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to add document {doc_id}: {e}")
        return False

    async def _vector_search(
        self,
        collection_name: str,
        query_vector: List[float],
        n_results: int,
        similarity_threshold: float,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        if self.db is None:
            raise RuntimeError("MongoDB database is not initialized")
        try:
            await self._ensure_collection(collection_name)
            collection = self.db[collection_name]

            filter_dict = {}
            if filter_conditions:
                for field, value in filter_conditions.items():
                    if field not in ALLOWED_FILTER_FIELDS:
                        logger.warning(
                            f"[{collection_name}] Ignoring disallowed filter field: {field}"
                        )
                        continue
                    if value is not None:
                        filter_dict[field] = value

            index_name = f"vector_index_{collection_name}"
            base_limit = n_results if n_results is not None else 1000
            expanded_limit = min(max(base_limit * 5, 1), 1000)

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "queryVector": query_vector,
                        "path": "embedding",
                        "limit": expanded_limit,
                        "numCandidates": expanded_limit * 10,
                        "filter": filter_dict or None,
                    }
                },
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$project": {"embedding": 0}},
                {"$sort": {"score": -1}},
            ]

            results = await asyncio.wait_for(
                collection.aggregate(pipeline).to_list(length=expanded_limit),  # type: ignore[arg-type]
                timeout=DB_TIMEOUT_SECONDS,
            )

            if not results:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            filtered = [
                r for r in results if r.get("score", 0.0) >= similarity_threshold
            ]
            limited = filtered[:n_results]

            metadatas = []
            for r in limited:
                metadata = {
                    k: v
                    for k, v in r.items()
                    if k not in {"_id", "embedding", "score", "text"}
                }
                metadata["text"] = r.get("text", "")
                metadatas.append(metadata)

            return {
                "documents": [r.get("text", "") for r in limited],
                "scores": [r.get("score", 0.0) for r in limited],
                "metadatas": metadatas,
                "ids": [str(r.get("_id", "")) for r in limited],
            }

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Query timeout")
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "doesn't exist" in msg or "not found" in msg:
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
            collection_name: Name of the collection to query
            query: Query string for semantic search
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            filter_conditions: Dictionary of metadata field filters

        Returns:
            Dictionary with documents, scores, metadatas, and ids
        """
        if not self.llm_connection:
            logger.error(
                f"[{collection_name}] LLM connection required for embedding generation"
            )
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
            logger.error(f"[{collection_name}] Embedding generation failed: {e}")
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
            collection_name: Name of the collection to query
            embedding: Embedding vector to search with
            n_results: Number of results to return
            filter_conditions: Dictionary of metadata field filters
            similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.0 for no threshold)

        Returns:
            Dictionary with documents, scores, metadatas, and ids
        """
        if not embedding or not isinstance(embedding, list):
            logger.error(f"[{collection_name}] Valid embedding required for search")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

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
        Update a memory document's metadata in MongoDB.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to update
            update_payload: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] MongoDB is disabled")
            return False

        if not update_payload:
            logger.debug(f"[{collection_name}] Empty update payload for {doc_id}")
            return True

        safe_update = {
            k: v for k, v in update_payload.items() if k not in PROTECTED_UPDATE_FIELDS
        }

        if not safe_update:
            logger.warning(f"[{collection_name}] No updatable fields for {doc_id}")
            return True

        try:
            if self.db is None:
                raise RuntimeError("MongoDB database is not initialized")
            collection = self.db[collection_name]

            exists = await asyncio.wait_for(
                collection.find_one({"_id": doc_id}, projection={"_id": 1}),
                timeout=DB_TIMEOUT_SECONDS,
            )
            if not exists:
                logger.warning(
                    f"[{collection_name}] Document {doc_id} not found for update"
                )
                return False

            result = await asyncio.wait_for(
                collection.update_one({"_id": doc_id}, {"$set": safe_update}),
                timeout=DB_TIMEOUT_SECONDS,
            )
            return result.modified_count > 0

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
        Query a specific document by ID in MongoDB.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to query

        Returns:
            Dictionary with document data, or None if not found
        """
        if not self.enabled:
            return None

        try:
            if self.db is None:
                raise RuntimeError("MongoDB database is not initialized")
            collection = self.db[collection_name]
            doc = await asyncio.wait_for(
                collection.find_one({"_id": doc_id}), timeout=DB_TIMEOUT_SECONDS
            )
            if not doc:
                return None

            metadata = {
                k: v for k, v in doc.items() if k not in {"_id", "text", "embedding"}
            }
            metadata["text"] = doc.get("text", "")

            return {
                "memory_id": str(doc["_id"]),
                "document": doc.get("text", ""),
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
            collection_name: Name of the collection
            doc_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning(f"[{collection_name}] MongoDB is disabled")
            return False

        try:
            if self.db is None:
                raise RuntimeError("MongoDB database is not initialized")
            collection = self.db[collection_name]
            result = await asyncio.wait_for(
                collection.delete_one({"_id": doc_id}), timeout=DB_TIMEOUT_SECONDS
            )
            return result.deleted_count > 0

        except asyncio.TimeoutError:
            logger.error(f"[{collection_name}] Timeout deleting {doc_id}")
        except Exception as e:
            logger.error(f"[{collection_name}] Failed to delete {doc_id}: {e}")
        return False
