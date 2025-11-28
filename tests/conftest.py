"""
Pytest configuration and shared fixtures for OmniMemory tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from omnimemory.core.llm import LLMConnection


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_connection():
    """Create a mock LLM connection for testing."""
    mock = Mock(spec=LLMConnection)
    mock.llm_config = {"provider": "openai", "model": "gpt-4o-mini"}
    mock.embedding_config = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }

    mock.embedding_call = AsyncMock(
        return_value=Mock(data=[Mock(embedding=[0.1] * 1536)])
    )
    mock.embedding_call_sync = Mock(
        return_value=Mock(data=[Mock(embedding=[0.1] * 1536)])
    )
    mock.llm_call = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(content="test response"))])
    )

    return mock


@pytest.fixture
def mock_embedding_response():
    """Create a mock embedding response."""
    return Mock(data=[Mock(embedding=[0.1] * 1536)])


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    return [0.1] * 1536


@pytest.fixture
def sample_text():
    """Sample text for embedding tests."""
    return "This is a sample text for testing embedding functionality."


@pytest.fixture
def long_text():
    """Long text that requires chunking."""
    return " ".join(["This is a long text that will require chunking."] * 1000)


@pytest.fixture
def mock_vector_db_handler():
    """Create a mock vector database handler."""
    mock = Mock()
    mock.enabled = True
    mock.add_to_collection = AsyncMock(return_value=True)
    mock.query_collection = AsyncMock(
        return_value={
            "documents": [],
            "scores": [],
            "metadatas": [],
            "ids": [],
        }
    )
    mock.query_by_embedding = AsyncMock(
        return_value={
            "documents": [],
            "scores": [],
            "metadatas": [],
            "ids": [],
        }
    )
    mock.query_by_id = AsyncMock(return_value=None)
    mock.update_memory = AsyncMock(return_value=True)
    mock.delete_from_collection = AsyncMock(return_value=True)
    mock.embed_text = AsyncMock(return_value=[0.1] * 1536)
    return mock


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "doc_id": "test-memory-123",
        "app_id": "test-app-123",
        "user_id": "test-user-123",
        "session_id": "test-session-123",
        "document": "This is a test memory document.",
        "embedding": [0.1] * 1536,
        "metadata": {
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "status": "active",
        },
    }


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {
            "role": "user",
            "content": "Hello, how are you?",
            "timestamp": "2025-01-01T00:00:00Z",
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you!",
            "timestamp": "2025-01-01T00:00:01Z",
        },
    ]


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables before each test."""
    pass


@pytest.fixture
def mock_metrics_collector():
    """Create a mock metrics collector."""
    mock = Mock()
    mock.record_query = Mock()
    mock.record_write = Mock()
    mock.record_update = Mock()
    mock.record_batch = Mock()
    mock.set_health = Mock()
    mock.operation_timer = MagicMock()
    mock.operation_timer.__enter__ = Mock(return_value=Mock(success=True))
    mock.operation_timer.__exit__ = Mock(return_value=None)
    return mock
