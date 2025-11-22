import asyncio
from qdrant_client.http import models as rest
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff
from omnimemory.core.logger_utils import get_logger
from typing import Dict, Any, List, Optional
from qdrant_client import models, AsyncQdrantClient
from decouple import config
from omnimemory.memory_management.vector_db_base import (
    VectorDBBase,
)


logger = get_logger(name="omnimemory.memory_management.qdrant_vector_db")


class QdrantVectorDB(VectorDBBase):
    """Qdrant vector database implementation."""

    def __init__(self, **kwargs):
        """Initialize Qdrant vector database.

        Args:
            **kwargs: Additional parameters including llm_connection
        """
        super().__init__(**kwargs)

        self.enabled = False
        self.client = None

        self.qdrant_host = config("QDRANT_HOST", default=None)
        qdrant_port_str = config("QDRANT_PORT", default=None)

        self.qdrant_port = None
        if qdrant_port_str:
            try:
                self.qdrant_port = int(qdrant_port_str)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid QDRANT_PORT value: {qdrant_port_str}, must be an integer"
                )

        if self.qdrant_host and self.qdrant_port:
            try:
                timeout = config("QDRANT_TIMEOUT", default=30, cast=int)
                self.client = AsyncQdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    timeout=timeout,
                )
                if self.llm_connection:
                    self.enabled = True
                    logger.info("QdrantVectorDB initialized successfully")
                else:
                    logger.warning(
                        "QdrantVectorDB initialized but LLM connection missing"
                    )
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {e}", exc_info=True)
                self.enabled = False
        else:
            logger.warning(
                "QDRANT_HOST or QDRANT_PORT not set. Qdrant will be disabled"
            )
            self.enabled = False

    async def close(self):
        """Cleanup method to release async connection."""
        if self.client:
            await self.client.close()

    async def _ensure_collection(self, collection_name: str):
        """Ensure the collection exists, create if it doesn't.

        Args:
            collection_name: Name of the collection
        """
        if not self.client:
            logger.warning("Qdrant is not enabled. Cannot ensure collection.")
            return
        actual_vector_size = self._get_embedding_dimensions()
        if not isinstance(actual_vector_size, int):
            raise ValueError(
                f"Vector size must be set before creating a collection. Got: {actual_vector_size}"
            )

        try:
            collections = await self.client.get_collections()
            collection_names = [
                collection.name for collection in collections.collections
            ]

            if collection_name not in collection_names:
                hnsw_config = HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                    full_scan_threshold=10000,
                )

                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=actual_vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=hnsw_config,
                    ),
                )
                logger.info(
                    f"Created Qdrant collection: {collection_name} with vector size: {actual_vector_size}"
                )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
            raise

    async def add_to_collection(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: List[float],
        metadata: Dict,
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
        if not self.client:
            logger.warning("Qdrant is not enabled. Cannot add to collection.")
            return False

        if not embedding or not isinstance(embedding, list):
            logger.error("Valid embedding vector is required")
            return False

        try:
            await self._ensure_collection(collection_name)

            metadata_copy = metadata.copy()
            metadata_copy["text"] = document

            point = models.PointStruct(
                id=doc_id, vector=embedding, payload=metadata_copy
            )

            await self.client.upsert(
                collection_name=collection_name,
                points=[point],
                wait=True,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add document to collection: {e}")
            return False

    async def query_collection(
        self,
        collection_name: str,
        query: str,
        n_results: int,
        similarity_threshold: float,
        filter_conditions: Dict[str, Any] = None,
    ):
        """Query collection with flexible filtering (async version).

        Args:
            collection_name: Name of the collection to query
            query: Query string for semantic search
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            filter_conditions: Dictionary of metadata field filters (e.g., {"app_id": "123", "user_id": "456"})

        Returns:
            Dictionary with documents, scores, metadatas, and ids
        """
        if not self.client:
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        try:
            must_conditions = []

            if filter_conditions:
                for field_name, field_value in filter_conditions.items():
                    if field_value is not None:
                        must_conditions.append(
                            rest.FieldCondition(
                                key=field_name, match=rest.MatchValue(value=field_value)
                            )
                        )

            query_filter = (
                rest.Filter(must=must_conditions) if must_conditions else None
            )

            expanded_limit = min(n_results * 5, 1000)

            query_embedding = await self.embed_text(query)

            try:
                search_result = await self.client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=expanded_limit,
                    score_threshold=similarity_threshold,
                    with_payload=True,
                    query_filter=query_filter,
                )
                filtered_results = search_result.points
            except TypeError:
                search_result = await self.client.query_points(
                    collection_name=collection_name,
                    query=query_embedding,
                    limit=expanded_limit,
                    with_payload=True,
                    query_filter=query_filter,
                )
                filtered_results = [
                    hit
                    for hit in search_result.points
                    if hit.score >= similarity_threshold
                ]

            limited_results = filtered_results[:n_results]

            return {
                "documents": [hit.payload.get("text", "") for hit in limited_results],
                "scores": [hit.score for hit in limited_results],
                "metadatas": [hit.payload for hit in limited_results],
                "ids": [hit.id for hit in limited_results],
            }

        except Exception as e:
            if "404" in str(e) or "doesn't exist" in str(e):
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            logger.error(f"Failed to query Qdrant: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def query_by_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        n_results: int,
        filter_conditions: Dict[str, Any] = None,
        similarity_threshold: float = 0.0,
    ):
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
        if not self.client:
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        if not embedding or not isinstance(embedding, list):
            logger.error("Valid embedding vector is required")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        try:
            must_conditions = []

            if filter_conditions:
                for field_name, field_value in filter_conditions.items():
                    if field_value is not None:
                        must_conditions.append(
                            rest.FieldCondition(
                                key=field_name, match=rest.MatchValue(value=field_value)
                            )
                        )

            query_filter = (
                rest.Filter(must=must_conditions) if must_conditions else None
            )

            limit = n_results if n_results is not None else 10000
            expanded_limit = min(limit * 3, 10000) if limit else 10000

            try:
                search_result = await self.client.query_points(
                    collection_name=collection_name,
                    query=embedding,
                    limit=expanded_limit,
                    score_threshold=similarity_threshold,
                    with_payload=True,
                    query_filter=query_filter,
                )
                filtered_results = search_result.points
            except TypeError:
                search_result = await self.client.query_points(
                    collection_name=collection_name,
                    query=embedding,
                    limit=expanded_limit,
                    with_payload=True,
                    query_filter=query_filter,
                )
                filtered_results = [
                    hit
                    for hit in search_result.points
                    if hit.score >= similarity_threshold
                ]

            limited_results = filtered_results[:limit] if limit else filtered_results

            return {
                "documents": [hit.payload.get("text", "") for hit in limited_results],
                "scores": [hit.score for hit in limited_results],
                "metadatas": [hit.payload for hit in limited_results],
                "ids": [hit.id for hit in limited_results],
            }

        except Exception as e:
            if "404" in str(e) or "doesn't exist" in str(e):
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            logger.error(f"Failed to query by embedding: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def update_memory(
        self,
        collection_name: str,
        doc_id: str,
        update_payload: Dict[str, Any],
    ) -> bool:
        """
        Update a memory document's metadata in Qdrant.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to update
            update_payload: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("QdrantVectorDB is disabled")
            return False

        try:
            await self._ensure_collection(collection_name)

            try:
                existing_points = await asyncio.wait_for(
                    self.client.retrieve(
                        collection_name=collection_name,
                        ids=[doc_id],
                        with_vectors=True,
                    ),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout retrieving point {doc_id} for update in collection {collection_name}"
                )
                return False
            except Exception as retrieve_error:
                error_str = str(retrieve_error)
                if (
                    "408" in error_str
                    or "Request Timeout" in error_str
                    or "timeout" in error_str.lower()
                ):
                    logger.warning(
                        f"Request timeout retrieving point {doc_id} for update: {retrieve_error}"
                    )
                    return False
                raise

            if not existing_points:
                logger.warning(
                    f"Point {doc_id} not found in collection {collection_name}"
                )
                return False

            existing_point = existing_points[0]
            existing_payload = existing_point.payload

            if hasattr(existing_point, "vector") and existing_point.vector is not None:
                existing_vector = existing_point.vector
                if not isinstance(existing_vector, list):
                    if isinstance(existing_vector, dict):
                        existing_vector = (
                            list(existing_vector.values())[0] if existing_vector else []
                        )
                    else:
                        existing_vector = []
                if not all(isinstance(x, (int, float)) for x in existing_vector):
                    logger.error(f"Invalid vector format for point {doc_id}")
                    return False
            else:
                logger.error(f"No vector found for point {doc_id}")
                return False

            updated_payload = {**existing_payload, **update_payload}
            await self.client.upsert(
                collection_name=collection_name,
                points=[
                    rest.PointStruct(
                        id=doc_id, vector=existing_vector, payload=updated_payload
                    )
                ],
            )

            return True

        except Exception as e:
            error_str = str(e)
            if (
                "408" in error_str
                or "Request Timeout" in error_str
                or "timeout" in error_str.lower()
            ):
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
        """
        Query a specific document by ID in Qdrant.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to query

        Returns:
            Dictionary with document data, or None if not found
        """
        if not self.enabled:
            logger.warning("QdrantVectorDB is disabled")
            return None

        try:
            await self._ensure_collection(collection_name)

            try:
                points = await asyncio.wait_for(
                    self.client.retrieve(collection_name=collection_name, ids=[doc_id]),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout querying document {doc_id} in collection {collection_name}"
                )
                return None
            except Exception as retrieve_error:
                error_str = str(retrieve_error)
                if (
                    "408" in error_str
                    or "Request Timeout" in error_str
                    or "timeout" in error_str.lower()
                ):
                    logger.warning(
                        f"Request timeout querying document {doc_id} in collection {collection_name}"
                    )
                    return None
                raise

            if not points or len(points) == 0:
                return None

            point = points[0]
            metadata = point.payload.copy()
            del metadata["text"]
            return {
                "memory_id": point.id,
                "document": point.payload.get("text", ""),
                "metadata": metadata,
            }

        except Exception as e:
            error_str = str(e)
            if (
                "408" in error_str
                or "Request Timeout" in error_str
                or "timeout" in error_str.lower()
            ):
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
        if not self.client:
            logger.warning("Qdrant is not enabled. Cannot delete from collection.")
            return False

        try:
            collections = await self.client.get_collections()
            collection_names = [
                collection.name for collection in collections.collections
            ]

            if collection_name not in collection_names:
                logger.warning(
                    f"Collection {collection_name} does not exist. Cannot delete document."
                )
                return False

            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[doc_id]),
                wait=True,
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to delete document {doc_id} from collection {collection_name}: {e}"
            )
            return False
