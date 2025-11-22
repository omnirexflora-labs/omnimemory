from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from numbers import Number
from omnimemory.core.logger_utils import get_logger
from omnimemory.core.utils import (
    chunk_text_by_tokens,
    estimate_chunking_stats,
    count_tokens,
)
import base64
import numpy as np
from omnimemory.core.utils import get_cached_embedding, cache_embedding


logger = get_logger(name="omnimemory.core.memory_management.vector_db_base")


_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50
_EDGE_BOOST_PERCENT = 15


def _get_edge_boost_multiplier() -> float:
    """Get edge boost multiplier for first/last chunks (internal)."""
    return 1.0 + (_EDGE_BOOST_PERCENT / 100.0)


class VectorDBBase(ABC):
    """Base class for vector database operations - CORE OPERATIONS ONLY."""

    def __init__(self, **kwargs):
        """Initialize vector database connection.

        Args:
            **kwargs: Additional parameters including llm_connection
        """
        self.llm_connection = kwargs.pop("llm_connection", None)

        self._embed_model = None
        self._vector_size = None
        self.enabled = False

        if not self.llm_connection:
            logger.error("LLM connection is required for vector database operations")

        for key, value in kwargs.items():
            setattr(self, key, value)

    async def embed_text(self, text: str) -> List[float]:
        """Embed text using LLM connection via LiteLLM with smart chunking for long texts (async version).

        Uses Redis caching to avoid redundant API calls for identical text.

        For single strings (queries, memory notes), pass the string directly.
        The embedding API accepts Union[str, List[str]] and always returns a list response.
        """
        if not self.llm_connection:
            raise RuntimeError("Vector database is disabled by configuration")

        if not isinstance(text, str):
            raise ValueError("Text input must be a non-empty string")

        text = text.strip()
        if not text:
            raise ValueError("Text input must be a non-empty string")
        cached_embedding = get_cached_embedding(text)
        if cached_embedding is not None:
            logger.debug(f"Using cached embedding for query: {len(text)} characters")
            return cached_embedding

        try:
            logger.info(f"Attempting to embed text of length: {len(text)} characters")
            response = await self.llm_connection.embedding_call(text)

            if response is None:
                logger.warning(
                    "Embedding call returned None, this might be due to rate limits or temporary failure"
                )
                raise RuntimeError("Embedding service temporarily unavailable")

            embedding = self._process_embedding_response(response)
            cache_embedding(text, embedding)

            return embedding

        except Exception as e:
            error_msg = str(e).lower()

            if any(
                keyword in error_msg
                for keyword in ["token", "length", "too long", "exceed", "limit"]
            ):
                logger.info(
                    f"Text too long for single embedding, implementing smart chunking: {e}"
                )
                return await self._embed_text_with_chunking_async(text)
            else:
                logger.error(f"LLM embedding failed: {e}")
                raise RuntimeError(f"Failed to generate embedding: {e}")

    def _embed_text_with_chunking(self, text: str) -> List[float]:
        """Embed long text by splitting into token-based chunks and processing."""
        try:
            model_name = "gpt-4"
            if (
                hasattr(self.llm_connection, "embedding_config")
                and self.llm_connection.embedding_config
            ):
                model_name = self.llm_connection.embedding_config.get("model", "gpt-4")

            chunk_stats = estimate_chunking_stats(
                text,
                chunk_size=_CHUNK_SIZE,
                overlap=_CHUNK_OVERLAP,
                model_name=model_name,
            )

            logger.info(
                f"Token-based chunking: {chunk_stats['total_tokens']} tokens → "
                f"{chunk_stats['chunks_needed']} chunks "
                f"(efficiency: {chunk_stats['token_efficiency']:.2%})"
            )

            chunks = chunk_text_by_tokens(
                text=text,
                chunk_size=_CHUNK_SIZE,
                overlap=_CHUNK_OVERLAP,
                model_name=model_name,
            )

            if not chunks:
                raise RuntimeError("No chunks generated from text")

            logger.info(f"Processing {len(chunks)} token-based chunks")

            all_embeddings = []

            for i, chunk in enumerate(chunks):
                chunk_tokens = count_tokens(chunk, model_name)
                logger.debug(
                    f"Processing chunk {i + 1}/{len(chunks)}: {chunk_tokens} tokens, {len(chunk)} characters"
                )

                try:
                    response = self.llm_connection.embedding_call_sync(chunk)

                    if response is None:
                        logger.warning(
                            f"Chunk {i + 1} embedding call returned None, this might be due to rate limits or temporary failure"
                        )
                        raise RuntimeError("Embedding service temporarily unavailable")

                    chunk_embedding = self._process_embedding_response(response)
                    all_embeddings.append(chunk_embedding)

                except Exception as chunk_error:
                    logger.error(f"Failed to embed chunk {i + 1}: {chunk_error}")
                    raise RuntimeError(f"Chunk {i + 1} embedding failed: {chunk_error}")

            if not all_embeddings:
                raise RuntimeError("All chunk embeddings failed")

            if self._vector_size is None:
                self._vector_size = len(all_embeddings[0])

            combined_embedding = []
            weights = self._calculate_chunk_weights(len(all_embeddings))

            for i in range(self._vector_size):
                values = [emb[i] for emb in all_embeddings]
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                combined_embedding.append(weighted_sum / sum(weights))

            logger.debug(
                f"Successfully combined {len(all_embeddings)} token-based chunk embeddings "
                f"into single {self._vector_size}-dimensional vector"
            )
            return combined_embedding

        except Exception as e:
            logger.error(f"Token-based chunking embedding failed: {e}")
            raise RuntimeError(f"Failed to embed text with token-based chunking: {e}")

    async def _embed_text_with_chunking_async(self, text: str) -> List[float]:
        """Embed long text by splitting into token-based chunks and processing (async version)."""
        try:
            model_name = "gpt-4"
            if (
                hasattr(self.llm_connection, "embedding_config")
                and self.llm_connection.embedding_config
            ):
                model_name = self.llm_connection.embedding_config.get("model", "gpt-4")

            chunk_stats = estimate_chunking_stats(
                text,
                chunk_size=_CHUNK_SIZE,
                overlap=_CHUNK_OVERLAP,
                model_name=model_name,
            )

            logger.info(
                f"Token-based chunking: {chunk_stats['total_tokens']} tokens → "
                f"{chunk_stats['chunks_needed']} chunks "
                f"(efficiency: {chunk_stats['token_efficiency']:.2%})"
            )

            chunks = chunk_text_by_tokens(
                text=text,
                chunk_size=_CHUNK_SIZE,
                overlap=_CHUNK_OVERLAP,
                model_name=model_name,
            )

            if not chunks:
                raise RuntimeError("No chunks generated from text")

            logger.info(f"Processing {len(chunks)} token-based chunks")

            all_embeddings = []

            for i, chunk in enumerate(chunks):
                chunk_tokens = count_tokens(chunk, model_name)
                logger.debug(
                    f"Processing chunk {i + 1}/{len(chunks)}: {chunk_tokens} tokens, {len(chunk)} characters"
                )

                try:
                    response = await self.llm_connection.embedding_call(chunk)

                    if response is None:
                        logger.warning(
                            f"Chunk {i + 1} embedding call returned None, this might be due to rate limits or temporary failure"
                        )
                        raise RuntimeError("Embedding service temporarily unavailable")

                    chunk_embedding = self._process_embedding_response(response)
                    all_embeddings.append(chunk_embedding)

                except Exception as chunk_error:
                    logger.error(f"Failed to embed chunk {i + 1}: {chunk_error}")
                    raise RuntimeError(f"Chunk {i + 1} embedding failed: {chunk_error}")

            if not all_embeddings:
                raise RuntimeError("All chunk embeddings failed")

            if self._vector_size is None:
                self._vector_size = len(all_embeddings[0])

            combined_embedding = []
            weights = self._calculate_chunk_weights(len(all_embeddings))

            for i in range(self._vector_size):
                values = [emb[i] for emb in all_embeddings]
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                combined_embedding.append(weighted_sum / sum(weights))

            logger.debug(
                f"Successfully combined {len(all_embeddings)} token-based chunk embeddings "
                f"into single {self._vector_size}-dimensional vector"
            )
            return combined_embedding

        except Exception as e:
            logger.error(f"Token-based chunking embedding failed: {e}")
            raise RuntimeError(f"Failed to embed text with token-based chunking: {e}")

    def _calculate_chunk_weights(self, num_chunks: int) -> List[float]:
        """
        Calculate weights for combining chunk embeddings.
        Gives slightly more weight to first and last chunks for context preservation.

        Args:
            num_chunks: Number of chunks

        Returns:
            List of weights that sum to num_chunks
        """
        if num_chunks <= 1:
            return [1.0]
        edge_boost = _get_edge_boost_multiplier()
        weights = [1.0] * num_chunks
        if num_chunks >= 2:
            weights[0] = edge_boost
            weights[-1] = edge_boost

        total_weight = sum(weights)
        return [w * num_chunks / total_weight for w in weights]

    def _process_embedding_response(self, response) -> List[float]:
        """Process the embedding response and extract the embedding vector."""
        if not response:
            raise RuntimeError("LLM embedding returned None response")

        if not hasattr(response, "data"):
            raise RuntimeError("LLM embedding response missing data field")

        response_data = response.data
        if response_data is None:
            raise RuntimeError("LLM embedding response missing data field")

        if not isinstance(response_data, list) or len(response_data) == 0:
            raise RuntimeError("LLM embedding response data is empty or invalid")

        embedding_data = response_data[0]

        if isinstance(embedding_data, dict) and "embedding" in embedding_data:
            embedding = embedding_data["embedding"]
        elif hasattr(embedding_data, "embedding"):
            embedding = getattr(embedding_data, "embedding", None)
        else:
            raise RuntimeError("LLM embedding response missing embedding field")

        if embedding is None:
            raise RuntimeError("LLM embedding response missing embedding field")

        if isinstance(embedding, str):
            try:
                decoded = base64.b64decode(embedding)
                embedding = np.frombuffer(decoded, dtype=np.float32).tolist()
                logger.debug(
                    f"Converted base64 embedding to {len(embedding)} float values"
                )
            except Exception as e:
                logger.error(f"Failed to decode base64 embedding: {e}")
                raise RuntimeError(f"Failed to decode base64 embedding: {e}")
        elif isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        elif isinstance(embedding, tuple):
            embedding = list(embedding)
        elif not isinstance(embedding, list):
            raise RuntimeError("LLM embedding response missing embedding field")

        if not embedding:
            raise RuntimeError("LLM embedding is not a valid numeric array")

        if not all(isinstance(value, Number) for value in embedding):
            raise RuntimeError("LLM embedding is not a valid numeric array")

        if self._vector_size is None:
            self._vector_size = len(embedding)
        elif self._vector_size != len(embedding):
            logger.warning(
                f"Embedding dimension mismatch: expected {self._vector_size}, got {len(embedding)}"
            )
            self._vector_size = len(embedding)

        return list(embedding)

    def _get_embedding_dimensions(self) -> int:
        """Get embedding dimensions from configuration."""
        if not self.llm_connection:
            raise ValueError("LLM connection is required to get embedding dimensions")

        if not hasattr(self.llm_connection, "embedding_config"):
            raise ValueError("LLM connection does not have embedding configuration")

        embedding_config = self.llm_connection.embedding_config
        if embedding_config is None:
            raise ValueError("Embedding configuration is not available")

        if "dimensions" not in embedding_config:
            raise ValueError("Embedding configuration is missing 'dimensions' field")

        dimensions = embedding_config["dimensions"]
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError(
                f"Invalid dimensions value: {dimensions}. Must be a positive integer."
            )

        return dimensions

    @abstractmethod
    async def _ensure_collection(self, collection_name: str):
        """Ensure the collection exists, create if it doesn't.

        Args:
            collection_name: Name of the collection
        """
        raise NotImplementedError

    @abstractmethod
    async def add_to_collection(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: List[float],
        metadata: Dict,
    ) -> bool:
        """Add document with embedding to collection (async version).

        Args:
            collection_name: Name of the collection
            doc_id: Document ID
            document: Document content
            embedding: Embedding vector for the document
            metadata: Document metadata

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    async def query_collection(
        self,
        collection_name: str,
        query: str,
        n_results: int,
        similarity_threshold: float,
        filter_conditions: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
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
        raise NotImplementedError

    @abstractmethod
    async def query_by_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        n_results: int,
        filter_conditions: Dict[str, Any] = None,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Query collection using an embedding vector directly.

        Used for finding semantically similar memories to a new memory before storing it.
        This enables creating memory links/relationships in the memory network.

        Args:
            collection_name: Name of the collection to query
            embedding: Embedding vector to search with
            n_results: Number of results to return
            filter_conditions: Dictionary of metadata field filters
            similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.0 for no threshold)

        Returns:
            Dictionary with documents, scores, metadatas, and ids
        """
        raise NotImplementedError

    @abstractmethod
    async def update_memory(
        self,
        collection_name: str,
        doc_id: str,
        update_payload: Dict[str, Any],
    ) -> bool:
        """Update a memory document's metadata.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to update
            update_payload: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    async def query_by_id(
        self,
        collection_name: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Query a specific document by ID.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to query

        Returns:
            Dictionary with document data, or None if not found
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
