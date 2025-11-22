"""
Unit tests for VectorDBBase class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from types import SimpleNamespace
from typing import List, Dict, Any

from omnimemory.memory_management.vector_db_base import VectorDBBase


class _TestableVectorDB(VectorDBBase):
    """Concrete implementation of VectorDBBase for testing."""

    async def _ensure_collection(self, collection_name: str):
        """Mock implementation."""
        pass

    async def add_to_collection(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: List[float],
        metadata: Dict,
    ) -> bool:
        """Mock implementation."""
        return True


class _IterableWithNoItems:
    """Truth-y iterable that yields no items (for coverage edge cases)."""

    def __len__(self):
        return 1

    def __bool__(self):
        return True

    def __iter__(self):
        return iter([])

    async def query_collection(
        self,
        collection_name: str,
        query: str,
        n_results: int,
        similarity_threshold: float,
        filter_conditions: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Mock implementation."""
        return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def query_by_embedding(
        self,
        collection_name: str,
        embedding: List[float],
        n_results: int,
        filter_conditions: Dict[str, Any] = None,
        similarity_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Mock implementation."""
        return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    async def update_memory(
        self,
        collection_name: str,
        doc_id: str,
        update_payload: Dict[str, Any],
    ) -> bool:
        """Mock implementation."""
        return True

    async def query_by_id(
        self,
        collection_name: str,
        doc_id: str,
    ) -> Dict[str, Any] | None:
        """Mock implementation."""
        return None

    async def delete_from_collection(
        self,
        collection_name: str,
        doc_id: str,
    ) -> bool:
        """Mock implementation."""
        return True


_TestableVectorDB.__abstractmethods__ = set()


class TestVectorDBBaseInitialization:
    """Test cases for VectorDBBase.__init__"""

    def test_init_with_valid_llm_connection(self, mock_llm_connection):
        """Test initialization with valid llm_connection."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        assert db.llm_connection == mock_llm_connection
        assert db.enabled == False
        assert db._embed_model is None
        assert db._vector_size is None

    def test_init_without_llm_connection(self):
        """Test initialization without llm_connection (should log error but not raise)."""
        db = _TestableVectorDB()

        assert db.llm_connection is None
        assert db.enabled == False

    def test_init_with_additional_kwargs(self, mock_llm_connection):
        """Test initialization with additional kwargs (should set as attributes)."""
        db = _TestableVectorDB(
            llm_connection=mock_llm_connection,
            custom_param="test_value",
            another_param=123,
        )

        assert db.custom_param == "test_value"
        assert db.another_param == 123

    def test_init_with_none_llm_connection(self):
        """Test initialization with None as llm_connection."""
        db = _TestableVectorDB(llm_connection=None)

        assert db.llm_connection is None
        assert db.enabled == False

    def test_init_with_empty_kwargs(self, mock_llm_connection):
        """Test initialization with empty kwargs."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        assert db.llm_connection == mock_llm_connection
        assert db.enabled == False


class TestEmbedText:
    """Test cases for VectorDBBase.embed_text (async embedding)"""

    @pytest.mark.asyncio
    async def test_embed_short_text_successfully(
        self, mock_llm_connection, sample_text
    ):
        """Test embedding short text successfully."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        mock_response = Mock(data=[Mock(embedding=[0.1] * 1536)])
        mock_llm_connection.embedding_call = AsyncMock(return_value=mock_response)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with patch("omnimemory.memory_management.vector_db_base.cache_embedding"):
                result = await db.embed_text(sample_text)

        assert result == [0.1] * 1536
        assert len(result) == 1536
        mock_llm_connection.embedding_call.assert_called_once_with(sample_text)

    @pytest.mark.asyncio
    async def test_return_cached_embedding_when_available(
        self, mock_llm_connection, sample_text
    ):
        """Test that cached embedding is returned when available."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        cached_embedding = [0.2] * 1536

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=cached_embedding,
        ):
            result = await db.embed_text(sample_text)

        assert result == cached_embedding
        mock_llm_connection.embedding_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_embedding_after_successful_call(
        self, mock_llm_connection, sample_text
    ):
        """Test that embedding is cached after successful call."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        mock_response = Mock(data=[Mock(embedding=[0.1] * 1536)])
        mock_llm_connection.embedding_call = AsyncMock(return_value=mock_response)

        cache_embedding_mock = Mock()
        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with patch(
                "omnimemory.memory_management.vector_db_base.cache_embedding",
                cache_embedding_mock,
            ):
                await db.embed_text(sample_text)

        cache_embedding_mock.assert_called_once()
        args = cache_embedding_mock.call_args
        assert args[0][0] == sample_text
        assert args[0][1] == [0.1] * 1536

    @pytest.mark.asyncio
    async def test_handle_empty_string(self, mock_llm_connection):
        """Test that empty string raises ValueError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            await db.embed_text("")

    @pytest.mark.asyncio
    async def test_handle_none_input(self, mock_llm_connection):
        """Test that None input raises ValueError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            await db.embed_text(None)

    @pytest.mark.asyncio
    async def test_handle_non_string_input(self, mock_llm_connection):
        """Test that non-string input raises ValueError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with pytest.raises(ValueError, match="Text input must be a non-empty string"):
            await db.embed_text(123)

    @pytest.mark.asyncio
    async def test_handle_missing_llm_connection(self):
        """Test that missing llm_connection raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=None)

        with pytest.raises(RuntimeError, match="Vector database is disabled"):
            await db.embed_text("test text")

    @pytest.mark.asyncio
    async def test_handle_embedding_api_returning_none(
        self, mock_llm_connection, sample_text
    ):
        """Test that embedding API returning None raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)
        mock_llm_connection.embedding_call = AsyncMock(return_value=None)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with pytest.raises(
                RuntimeError, match="Embedding service temporarily unavailable"
            ):
                await db.embed_text(sample_text)

    @pytest.mark.asyncio
    async def test_handle_embedding_api_rate_limit_errors(
        self, mock_llm_connection, long_text
    ):
        """Test that rate limit errors trigger chunking."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        rate_limit_error = RuntimeError("rate limit exceeded")
        mock_llm_connection.embedding_call = AsyncMock(side_effect=rate_limit_error)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with patch.object(
                db, "_embed_text_with_chunking_async", new_callable=AsyncMock
            ) as mock_chunking:
                mock_chunking.return_value = [0.1] * 1536

                result = await db.embed_text(long_text)

        mock_chunking.assert_called_once()
        assert result == [0.1] * 1536

    @pytest.mark.asyncio
    async def test_handle_embedding_api_token_length_errors(
        self, mock_llm_connection, long_text
    ):
        """Test that token length errors trigger chunking."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        token_error = RuntimeError("token length exceeded")
        mock_llm_connection.embedding_call = AsyncMock(side_effect=token_error)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with patch.object(
                db, "_embed_text_with_chunking_async", new_callable=AsyncMock
            ) as mock_chunking:
                mock_chunking.return_value = [0.1] * 1536

                result = await db.embed_text(long_text)

        mock_chunking.assert_called_once()
        assert result == [0.1] * 1536

    @pytest.mark.asyncio
    async def test_handle_other_embedding_api_errors(
        self, mock_llm_connection, sample_text
    ):
        """Test that other embedding API errors raise RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        other_error = RuntimeError("Network error")
        mock_llm_connection.embedding_call = AsyncMock(side_effect=other_error)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="Failed to generate embedding"):
                await db.embed_text(sample_text)

    @pytest.mark.asyncio
    async def test_text_with_special_characters(self, mock_llm_connection):
        """Test embedding text with special characters/unicode."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        special_text = "Hello 世界! 🌍 Test with émojis and spéciál chárs"
        mock_response = Mock(data=[Mock(embedding=[0.1] * 1536)])
        mock_llm_connection.embedding_call = AsyncMock(return_value=mock_response)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with patch("omnimemory.memory_management.vector_db_base.cache_embedding"):
                result = await db.embed_text(special_text)

        assert result == [0.1] * 1536
        mock_llm_connection.embedding_call.assert_called_once_with(special_text)

    @pytest.mark.asyncio
    async def test_text_with_only_whitespace(self, mock_llm_connection):
        """Test that text with only whitespace raises ValueError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with pytest.raises(ValueError):
            await db.embed_text("   ")


class TestCalculateChunkWeights:
    """Test cases for VectorDBBase._calculate_chunk_weights"""

    def test_calculate_weights_for_1_chunk(self, mock_llm_connection):
        """Test that 1 chunk returns [1.0]."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(1)

        assert weights == [1.0]

    def test_calculate_weights_for_2_chunks(self, mock_llm_connection):
        """Test that 2 chunks both get edge boost."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(2)

        assert len(weights) == 2
        assert weights[0] >= 1.0
        assert weights[1] >= 1.0
        assert abs(sum(weights) - 2.0) < 0.001

    def test_calculate_weights_for_3_plus_chunks(self, mock_llm_connection):
        """Test that 3+ chunks: first and last get edge boost, middle chunks have weight 1.0."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(5)

        assert len(weights) == 5
        assert weights[0] > 1.0
        assert weights[-1] > 1.0
        assert abs(weights[1] - 1.0) < 0.1
        assert abs(weights[2] - 1.0) < 0.1
        assert abs(weights[3] - 1.0) < 0.1
        assert abs(sum(weights) - 5.0) < 0.001

    def test_weights_sum_to_num_chunks(self, mock_llm_connection):
        """Test that weights always sum to num_chunks."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        for num_chunks in [1, 2, 3, 5, 10, 20]:
            weights = db._calculate_chunk_weights(num_chunks)
            assert abs(sum(weights) - num_chunks) < 0.001

    def test_edge_boost_multiplier_applied(self, mock_llm_connection):
        """Test that edge boost multiplier (15%) is applied correctly."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(3)

        assert weights[0] > 1.0
        assert weights[-1] > 1.0

    def test_middle_chunks_have_weight_1(self, mock_llm_connection):
        """Test that middle chunks have weight 1.0."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(10)

        for i in range(1, 9):
            assert abs(weights[i] - 1.0) < 0.1


class TestProcessEmbeddingResponse:
    """Test cases for VectorDBBase._process_embedding_response"""

    def test_process_response_with_dict_embedding(self, mock_llm_connection):
        """Test processing response with data[0].embedding as dict format."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {"embedding": [0.1, 0.2, 0.3]}
        response = Mock(data=[embedding_data])

        result = db._process_embedding_response(response)

        assert result == [0.1, 0.2, 0.3]
        assert db._vector_size == 3

    def test_process_response_with_object_embedding(self, mock_llm_connection):
        """Test processing response with data[0].embedding as object attribute."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_obj = Mock(embedding=[0.1, 0.2, 0.3])
        response = Mock(data=[embedding_obj])

        result = db._process_embedding_response(response)

        assert result == [0.1, 0.2, 0.3]
        assert db._vector_size == 3

    def test_process_base64_encoded_embedding(self, mock_llm_connection):
        """Test processing base64 encoded embedding string."""
        import base64
        import numpy as np

        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_array = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        encoded = base64.b64encode(embedding_array.tobytes()).decode("utf-8")

        embedding_data = {"embedding": encoded}
        response = Mock(data=[embedding_data])

        result = db._process_embedding_response(response)

        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)
        assert db._vector_size == 3

    def test_handle_none_response(self, mock_llm_connection):
        """Test that None response raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with pytest.raises(RuntimeError, match="LLM embedding returned None response"):
            db._process_embedding_response(None)

    def test_handle_response_without_data_field(self, mock_llm_connection):
        """Test that response without data field raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        response = Mock(spec=[])

        with pytest.raises(
            RuntimeError, match="LLM embedding response missing data field"
        ):
            db._process_embedding_response(response)

    def test_handle_response_with_empty_data_list(self, mock_llm_connection):
        """Test that response with empty data list raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        response = Mock(data=[])

        with pytest.raises(
            RuntimeError, match="LLM embedding response data is empty or invalid"
        ):
            db._process_embedding_response(response)

    def test_handle_response_without_embedding_field(self, mock_llm_connection):
        """Test that response without embedding field raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {}
        response = Mock(data=[embedding_data])

        with pytest.raises(
            RuntimeError, match="LLM embedding response missing embedding field"
        ):
            db._process_embedding_response(response)

    def test_handle_invalid_base64_string(self, mock_llm_connection):
        """Test that invalid base64 string raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {"embedding": "invalid_base64!!!"}
        response = Mock(data=[embedding_data])

        with pytest.raises(RuntimeError, match="Failed to decode base64 embedding"):
            db._process_embedding_response(response)

    def test_handle_non_numeric_embedding_array(self, mock_llm_connection):
        """Test that non-numeric embedding array raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {"embedding": ["not", "numbers"]}
        response = Mock(data=[embedding_data])

        with pytest.raises(
            RuntimeError, match="LLM embedding is not a valid numeric array"
        ):
            db._process_embedding_response(response)

    def test_handle_empty_embedding_array(self, mock_llm_connection):
        """Test that empty embedding array raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {"embedding": []}
        response = Mock(data=[embedding_data])

        with pytest.raises(
            RuntimeError, match="LLM embedding is not a valid numeric array"
        ):
            db._process_embedding_response(response)

    def test_set_vector_size_on_first_embedding(self, mock_llm_connection):
        """Test that _vector_size is set on first embedding."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        assert db._vector_size is None

        embedding_data = {"embedding": [0.1] * 1536}
        response = Mock(data=[embedding_data])

        db._process_embedding_response(response)

        assert db._vector_size == 1536

    def test_update_vector_size_on_dimension_mismatch(self, mock_llm_connection):
        """Test that _vector_size is updated on dimension mismatch (should log warning)."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data_1 = {"embedding": [0.1] * 1536}
        response_1 = Mock(data=[embedding_data_1])
        db._process_embedding_response(response_1)
        assert db._vector_size == 1536

        embedding_data_2 = {"embedding": [0.1] * 768}
        response_2 = Mock(data=[embedding_data_2])

        with patch("omnimemory.memory_management.vector_db_base.logger") as mock_logger:
            db._process_embedding_response(response_2)

            mock_logger.warning.assert_called()
            assert db._vector_size == 768

    def test_return_list_of_floats(self, mock_llm_connection):
        """Test that result is always a list of floats."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {"embedding": [0.1, 0.2, 0.3]}
        response = Mock(data=[embedding_data])

        result = db._process_embedding_response(response)

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)


class TestGetEmbeddingDimensions:
    """Test cases for VectorDBBase._get_embedding_dimensions"""

    def test_get_dimensions_from_valid_config(self, mock_llm_connection):
        """Test getting dimensions from valid embedding_config."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        dimensions = db._get_embedding_dimensions()

        assert dimensions == 1536
        assert isinstance(dimensions, int)

    def test_handle_missing_llm_connection(self):
        """Test that missing llm_connection raises ValueError."""
        db = _TestableVectorDB(llm_connection=None)

        with pytest.raises(ValueError, match="LLM connection is required"):
            db._get_embedding_dimensions()

    def test_handle_llm_connection_without_embedding_config(self):
        """Test that llm_connection without embedding_config attribute raises ValueError."""
        mock_llm = Mock(spec=[])
        db = _TestableVectorDB(llm_connection=mock_llm)

        with pytest.raises(
            ValueError, match="LLM connection does not have embedding configuration"
        ):
            db._get_embedding_dimensions()

    def test_handle_none_embedding_config(self):
        """Test that None embedding_config raises ValueError."""
        mock_llm = Mock()
        mock_llm.embedding_config = None
        db = _TestableVectorDB(llm_connection=mock_llm)

        with pytest.raises(
            ValueError, match="Embedding configuration is not available"
        ):
            db._get_embedding_dimensions()

    def test_handle_missing_dimensions_field(self):
        """Test that missing dimensions field raises ValueError."""
        mock_llm = Mock()
        mock_llm.embedding_config = {}
        db = _TestableVectorDB(llm_connection=mock_llm)

        with pytest.raises(
            ValueError, match="Embedding configuration is missing 'dimensions' field"
        ):
            db._get_embedding_dimensions()

    def test_handle_dimensions_as_non_integer(self):
        """Test that dimensions as non-integer raises ValueError."""
        mock_llm = Mock()
        mock_llm.embedding_config = {"dimensions": "1536"}
        db = _TestableVectorDB(llm_connection=mock_llm)

        with pytest.raises(ValueError, match="Invalid dimensions value"):
            db._get_embedding_dimensions()

    def test_handle_dimensions_as_zero(self):
        """Test that dimensions as zero raises ValueError."""
        mock_llm = Mock()
        mock_llm.embedding_config = {"dimensions": 0}
        db = _TestableVectorDB(llm_connection=mock_llm)

        with pytest.raises(ValueError, match="Invalid dimensions value"):
            db._get_embedding_dimensions()

    def test_handle_dimensions_as_negative(self):
        """Test that dimensions as negative raises ValueError."""
        mock_llm = Mock()
        mock_llm.embedding_config = {"dimensions": -1}
        db = _TestableVectorDB(llm_connection=mock_llm)

        with pytest.raises(ValueError, match="Invalid dimensions value"):
            db._get_embedding_dimensions()

    def test_return_positive_integer_dimensions(self, mock_llm_connection):
        """Test that valid dimensions are returned as positive integer."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        dimensions = db._get_embedding_dimensions()

        assert isinstance(dimensions, int)
        assert dimensions > 0
        assert dimensions == 1536


class TestEmbedTextWithChunking:
    """Test cases for VectorDBBase._embed_text_with_chunking (sync) and _embed_text_with_chunking_async"""

    @pytest.mark.asyncio
    async def test_chunk_long_text_correctly(self, mock_llm_connection, long_text):
        """Test that long text is chunked correctly."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                with patch(
                    "omnimemory.memory_management.vector_db_base.count_tokens"
                ) as mock_count:
                    mock_stats.return_value = {
                        "total_tokens": 5000,
                        "chunks_needed": 10,
                        "token_efficiency": 0.95,
                    }
                    mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]
                    mock_count.return_value = 500

                    mock_llm_connection.embedding_call_sync = Mock(
                        side_effect=[
                            Mock(data=[Mock(embedding=[0.1] * 1536)]),
                            Mock(data=[Mock(embedding=[0.2] * 1536)]),
                            Mock(data=[Mock(embedding=[0.3] * 1536)]),
                        ]
                    )

                    result = db._embed_text_with_chunking(long_text)

        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_handle_single_chunk(self, mock_llm_connection):
        """Test that single chunk (no chunking needed) works correctly."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        short_text = "Short text"

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_stats.return_value = {
                    "total_tokens": 10,
                    "chunks_needed": 1,
                    "token_efficiency": 1.0,
                }
                mock_chunk.return_value = [short_text]

                mock_llm_connection.embedding_call_sync = Mock(
                    return_value=Mock(data=[Mock(embedding=[0.1] * 1536)])
                )

                result = db._embed_text_with_chunking(short_text)

        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_handle_no_chunks_generated(self, mock_llm_connection, long_text):
        """Test that no chunks generated raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
            return_value=[],
        ):
            with pytest.raises(RuntimeError, match="No chunks generated from text"):
                db._embed_text_with_chunking(long_text)

    @pytest.mark.asyncio
    async def test_handle_chunk_embedding_failure(self, mock_llm_connection, long_text):
        """Test that chunk embedding failure raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_chunk.return_value = ["chunk1", "chunk2"]
                mock_stats.return_value = {
                    "total_tokens": 2000,
                    "chunks_needed": 2,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call_sync = Mock(
                    side_effect=RuntimeError("Embedding failed")
                )

                with pytest.raises(RuntimeError, match="Chunk.*embedding failed"):
                    db._embed_text_with_chunking(long_text)

    @pytest.mark.asyncio
    async def test_handle_all_chunks_failing(self, mock_llm_connection, long_text):
        """Test that all chunks failing raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_chunk.return_value = ["chunk1"]
                mock_stats.return_value = {
                    "total_tokens": 1000,
                    "chunks_needed": 1,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call_sync = Mock(
                    side_effect=RuntimeError("All failed")
                )

                with pytest.raises(RuntimeError):
                    db._embed_text_with_chunking(long_text)

    @pytest.mark.asyncio
    async def test_handle_chunk_api_returning_none(
        self, mock_llm_connection, long_text
    ):
        """Test that chunk API returning None raises RuntimeError."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_chunk.return_value = ["chunk1"]
                mock_stats.return_value = {
                    "total_tokens": 1000,
                    "chunks_needed": 1,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call_sync = Mock(return_value=None)

                with pytest.raises(
                    RuntimeError, match="Embedding service temporarily unavailable"
                ):
                    db._embed_text_with_chunking(long_text)

    @pytest.mark.asyncio
    async def test_verify_combined_embedding_dimensions_match(
        self, mock_llm_connection, long_text
    ):
        """Test that combined embedding has correct dimensions."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_chunk.return_value = ["chunk1", "chunk2"]
                mock_stats.return_value = {
                    "total_tokens": 2000,
                    "chunks_needed": 2,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call_sync = Mock(
                    side_effect=[
                        Mock(data=[Mock(embedding=[0.1] * 1536)]),
                        Mock(data=[Mock(embedding=[0.2] * 1536)]),
                    ]
                )

                result = db._embed_text_with_chunking(long_text)

        assert len(result) == 1536
        assert db._vector_size == 1536

    @pytest.mark.asyncio
    async def test_combined_embedding_sets_vector_size_if_not_set(
        self, mock_llm_connection, long_text
    ):
        """Ensure _embed_text_with_chunking sets _vector_size when process method does not."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with (
            patch(
                "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
            ) as mock_chunk,
            patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats,
            patch.object(
                _TestableVectorDB,
                "_process_embedding_response",
                return_value=[0.1, 0.2],
            ),
        ):
            mock_chunk.return_value = ["chunk1", "chunk2"]
            mock_stats.return_value = {
                "total_tokens": 100,
                "chunks_needed": 2,
                "token_efficiency": 0.9,
            }

            mock_llm_connection.embedding_call_sync = Mock()

            result = db._embed_text_with_chunking(long_text)

        assert db._vector_size == 2
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_all_embeddings_empty_raises_runtime(
        self, mock_llm_connection, long_text
    ):
        """Force path where no embeddings are collected even though chunks are truthy."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        fake_chunks = _IterableWithNoItems()

        with (
            patch(
                "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
                return_value=fake_chunks,
            ),
            patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats",
                return_value={
                    "total_tokens": 100,
                    "chunks_needed": 1,
                    "token_efficiency": 1.0,
                },
            ),
        ):
            with pytest.raises(RuntimeError, match="All chunk embeddings failed"):
                db._embed_text_with_chunking(long_text)

    @pytest.mark.asyncio
    async def test_verify_weights_applied_correctly(
        self, mock_llm_connection, long_text
    ):
        """Test that weights are applied correctly in weighted average."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]
                mock_stats.return_value = {
                    "total_tokens": 3000,
                    "chunks_needed": 3,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call_sync = Mock(
                    side_effect=[
                        Mock(data=[Mock(embedding=[1.0] * 1536)]),
                        Mock(data=[Mock(embedding=[2.0] * 1536)]),
                        Mock(data=[Mock(embedding=[3.0] * 1536)]),
                    ]
                )

                result = db._embed_text_with_chunking(long_text)

        assert all(1.0 <= x <= 3.0 for x in result)
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_async_chunking_handles_concurrent_processing(
        self, mock_llm_connection, long_text
    ):
        """Test that async chunking handles concurrent chunk processing."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]
                mock_stats.return_value = {
                    "total_tokens": 3000,
                    "chunks_needed": 3,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call = AsyncMock(
                    side_effect=[
                        Mock(data=[Mock(embedding=[0.1] * 1536)]),
                        Mock(data=[Mock(embedding=[0.2] * 1536)]),
                        Mock(data=[Mock(embedding=[0.3] * 1536)]),
                    ]
                )

                result = await db._embed_text_with_chunking_async(long_text)

        assert len(result) == 1536
        assert mock_llm_connection.embedding_call.call_count == 3

    @pytest.mark.asyncio
    async def test_async_chunking_no_chunks_generated(
        self, mock_llm_connection, long_text
    ):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
            return_value=[],
        ):
            with pytest.raises(RuntimeError, match="No chunks generated from text"):
                await db._embed_text_with_chunking_async(long_text)

    @pytest.mark.asyncio
    async def test_async_chunking_chunk_returns_none(
        self, mock_llm_connection, long_text
    ):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with (
            patch(
                "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
                return_value=["chunk1"],
            ),
            patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats",
                return_value={
                    "total_tokens": 100,
                    "chunks_needed": 1,
                    "token_efficiency": 1.0,
                },
            ),
        ):
            mock_llm_connection.embedding_call = AsyncMock(return_value=None)
            with pytest.raises(
                RuntimeError, match="Embedding service temporarily unavailable"
            ):
                await db._embed_text_with_chunking_async(long_text)

    @pytest.mark.asyncio
    async def test_async_chunking_chunk_failure_raises(
        self, mock_llm_connection, long_text
    ):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with (
            patch(
                "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
                return_value=["chunk1"],
            ),
            patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats",
                return_value={
                    "total_tokens": 100,
                    "chunks_needed": 1,
                    "token_efficiency": 1.0,
                },
            ),
        ):
            mock_llm_connection.embedding_call = AsyncMock(
                side_effect=RuntimeError("fail")
            )
            with pytest.raises(RuntimeError, match="Chunk 1 embedding failed"):
                await db._embed_text_with_chunking_async(long_text)

    @pytest.mark.asyncio
    async def test_async_all_embeddings_empty_raises(
        self, mock_llm_connection, long_text
    ):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)
        fake_chunks = _IterableWithNoItems()

        with (
            patch(
                "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
                return_value=fake_chunks,
            ),
            patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats",
                return_value={
                    "total_tokens": 1,
                    "chunks_needed": 1,
                    "token_efficiency": 1.0,
                },
            ),
        ):
            with pytest.raises(RuntimeError, match="All chunk embeddings failed"):
                await db._embed_text_with_chunking_async(long_text)

    @pytest.mark.asyncio
    async def test_async_sets_vector_size_when_not_set(
        self, mock_llm_connection, long_text
    ):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with (
            patch(
                "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
                return_value=["chunk1"],
            ),
            patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats",
                return_value={
                    "total_tokens": 100,
                    "chunks_needed": 1,
                    "token_efficiency": 1.0,
                },
            ),
            patch.object(
                _TestableVectorDB,
                "_process_embedding_response",
                return_value=[0.1, 0.2],
            ),
        ):
            mock_llm_connection.embedding_call = AsyncMock()

            result = await db._embed_text_with_chunking_async(long_text)

        assert db._vector_size == 2
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_async_chunking_outer_exception(self, mock_llm_connection, long_text):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(
                RuntimeError, match="Failed to embed text with token-based chunking"
            ):
                await db._embed_text_with_chunking_async(long_text)

    @pytest.mark.asyncio
    async def test_text_with_exactly_token_limit(self, mock_llm_connection):
        """Test text with exactly token limit (boundary case)."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        mock_response = Mock(data=[Mock(embedding=[0.1] * 1536)])
        mock_llm_connection.embedding_call = AsyncMock(return_value=mock_response)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with patch("omnimemory.memory_management.vector_db_base.cache_embedding"):
                result = await db.embed_text("Short text")

        assert result == [0.1] * 1536
        assert (
            not hasattr(db, "_embed_text_with_chunking_async")
            or not hasattr(mock_llm_connection.embedding_call, "call_count")
            or mock_llm_connection.embedding_call.call_count == 1
        )

    @pytest.mark.asyncio
    async def test_text_just_over_token_limit(self, mock_llm_connection, long_text):
        """Test text just over token limit (should chunk)."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        token_error = RuntimeError("token length exceeded")
        mock_llm_connection.embedding_call = AsyncMock(side_effect=token_error)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            return_value=None,
        ):
            with patch.object(
                db, "_embed_text_with_chunking_async", new_callable=AsyncMock
            ) as mock_chunking:
                mock_chunking.return_value = [0.1] * 1536

                result = await db.embed_text(long_text)

        mock_chunking.assert_called_once()
        assert result == [0.1] * 1536

    @pytest.mark.asyncio
    async def test_very_long_text_multiple_chunks(self, mock_llm_connection):
        """Test very long text that generates multiple chunks."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        very_long_text = " ".join(["This is a very long text."] * 5000)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                chunks = [f"chunk{i}" for i in range(20)]
                mock_chunk.return_value = chunks
                mock_stats.return_value = {
                    "total_tokens": 20000,
                    "chunks_needed": 20,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call_sync = Mock(
                    side_effect=[
                        Mock(data=[Mock(embedding=[0.1] * 1536)]) for _ in chunks
                    ]
                )

                result = db._embed_text_with_chunking(very_long_text)

        assert len(result) == 1536
        assert mock_llm_connection.embedding_call_sync.call_count == 20

    @pytest.mark.asyncio
    async def test_concurrent_calls_with_same_text_cache_behavior(
        self, mock_llm_connection, sample_text
    ):
        """Test concurrent calls with same text (cache behavior)."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        mock_response = Mock(data=[Mock(embedding=[0.1] * 1536)])
        mock_llm_connection.embedding_call = AsyncMock(return_value=mock_response)

        with patch(
            "omnimemory.memory_management.vector_db_base.get_cached_embedding",
            side_effect=[None, [0.2] * 1536],
        ):
            with patch("omnimemory.memory_management.vector_db_base.cache_embedding"):
                result1 = await db.embed_text(sample_text)
                result2 = await db.embed_text(sample_text)

        assert mock_llm_connection.embedding_call.call_count == 1
        assert result2 == [0.2] * 1536

    def test_text_generates_exactly_2_chunks_both_get_edge_boost(
        self, mock_llm_connection
    ):
        """Test that text generating exactly 2 chunks gives both edge boost."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(2)

        assert len(weights) == 2
        assert weights[0] >= 1.0
        assert weights[1] >= 1.0
        assert abs(sum(weights) - 2.0) < 0.001

    def test_text_generates_many_chunks_only_first_last_get_edge_boost(
        self, mock_llm_connection
    ):
        """Test that many chunks only give edge boost to first and last."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(10)

        assert len(weights) == 10
        assert weights[0] > 1.0
        assert weights[-1] > 1.0
        for i in range(1, 9):
            assert abs(weights[i] - 1.0) < 0.1

    def test_weights_are_positive_floats(self, mock_llm_connection):
        """Test that all weights are positive floats."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        for num_chunks in [1, 2, 3, 5, 10, 20]:
            weights = db._calculate_chunk_weights(num_chunks)
            assert all(isinstance(w, float) for w in weights)
            assert all(w > 0 for w in weights)

    @pytest.mark.asyncio
    async def test_embedding_response_with_unexpected_format(self, mock_llm_connection):
        """Test handling of embedding response with unexpected format."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        response = Mock(data=[SimpleNamespace()])

        with pytest.raises(
            RuntimeError, match="LLM embedding response missing embedding field"
        ):
            db._process_embedding_response(response)

    @pytest.mark.asyncio
    async def test_dimension_mismatch_between_calls(self, mock_llm_connection):
        """Test dimension mismatch between embedding calls."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data_1 = {"embedding": [0.1] * 1536}
        response_1 = Mock(data=[embedding_data_1])
        db._process_embedding_response(response_1)
        assert db._vector_size == 1536

        embedding_data_2 = {"embedding": [0.1] * 768}
        response_2 = Mock(data=[embedding_data_2])

        with patch("omnimemory.memory_management.vector_db_base.logger") as mock_logger:
            db._process_embedding_response(response_2)

            mock_logger.warning.assert_called()
            assert db._vector_size == 768

    @pytest.mark.asyncio
    async def test_base64_decoding_with_invalid_characters(self, mock_llm_connection):
        """Test base64 decoding with invalid characters."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {"embedding": "!!!invalid_base64!!!"}
        response = Mock(data=[embedding_data])

        with pytest.raises(RuntimeError, match="Failed to decode base64 embedding"):
            db._process_embedding_response(response)

    @pytest.mark.asyncio
    async def test_embedding_with_nan_or_inf_values(self, mock_llm_connection):
        """Test handling of embedding with NaN or Inf values."""
        import math

        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        embedding_data = {"embedding": [0.1, float("nan"), 0.3]}
        response = Mock(data=[embedding_data])

        result = db._process_embedding_response(response)
        assert len(result) == 3

    def test_handle_num_chunks_zero_edge_case(self, mock_llm_connection):
        """Test handling of num_chunks = 0 (edge case)."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(0)
        assert isinstance(weights, list)

    def test_handle_num_chunks_one_edge_case(self, mock_llm_connection):
        """Test handling of num_chunks = 1 (edge case)."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        weights = db._calculate_chunk_weights(1)
        assert weights == [1.0]

    @pytest.mark.asyncio
    async def test_overlapping_chunks_have_correct_overlap(
        self, mock_llm_connection, long_text
    ):
        """Test that overlapping chunks have correct overlap."""
        db = _TestableVectorDB(llm_connection=mock_llm_connection)

        with patch(
            "omnimemory.memory_management.vector_db_base.chunk_text_by_tokens"
        ) as mock_chunk:
            with patch(
                "omnimemory.memory_management.vector_db_base.estimate_chunking_stats"
            ) as mock_stats:
                mock_chunk.return_value = ["chunk1", "chunk2"]
                mock_stats.return_value = {
                    "total_tokens": 2000,
                    "chunks_needed": 2,
                    "token_efficiency": 0.9,
                }

                mock_llm_connection.embedding_call_sync = Mock(
                    side_effect=[
                        Mock(data=[Mock(embedding=[0.1] * 1536)]),
                        Mock(data=[Mock(embedding=[0.2] * 1536)]),
                    ]
                )

                db._embed_text_with_chunking(long_text)

        mock_chunk.assert_called()
        call_kwargs = mock_chunk.call_args[1]
        assert "overlap" in call_kwargs
        assert call_kwargs["overlap"] == 50


class TestProcessEmbeddingResponseEdgeCases:
    def test_response_missing_data_field(self, mock_llm_connection):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)
        response = Mock(data=None)

        with pytest.raises(RuntimeError, match="missing data field"):
            db._process_embedding_response(response)

    def test_response_embedding_none(self, mock_llm_connection):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)
        response = Mock(data=[Mock(embedding=None)])

        with pytest.raises(RuntimeError, match="missing embedding field"):
            db._process_embedding_response(response)

    def test_response_numpy_array_converted(self, mock_llm_connection):
        import numpy as np

        db = _TestableVectorDB(llm_connection=mock_llm_connection)
        response = Mock(data=[Mock(embedding=np.array([0.1, 0.2]))])

        result = db._process_embedding_response(response)
        assert result == [0.1, 0.2]

    def test_response_tuple_converted(self, mock_llm_connection):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)
        response = Mock(data=[Mock(embedding=(0.1, 0.3))])

        result = db._process_embedding_response(response)
        assert result == [0.1, 0.3]

    def test_response_invalid_type_raises(self, mock_llm_connection):
        db = _TestableVectorDB(llm_connection=mock_llm_connection)
        response = Mock(data=[Mock(embedding={"unexpected": "value"})])

        with pytest.raises(RuntimeError, match="missing embedding field"):
            db._process_embedding_response(response)
