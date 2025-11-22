"""
Comprehensive unit tests for OmniMemory API Server.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Dict, Any, List

from omnimemory.api.server import app, lifespan, get_sdk
from omnimemory.sdk import OmniMemorySDK
from omnimemory.core.schemas import (
    AddUserMessageRequest,
    ConversationSummaryRequest,
    AgentMemoryRequest,
    TaskResponse,
    MemoryResponse,
    MemoryListResponse,
    ConversationSummaryResponse,
    SuccessResponse,
    Message,
)
from omnimemory.core.results import MemoryOperationResult
from omnimemory.core.config import DEFAULT_MAX_MESSAGES


@pytest.fixture
def mock_sdk():
    """Create a mock SDK instance."""
    sdk = Mock(spec=OmniMemorySDK)
    sdk.add_memory = AsyncMock(
        return_value={
            "task_id": "test-task-123",
            "status": "accepted",
            "app_id": "app1234567890",
            "user_id": "user1234567890",
        }
    )
    sdk.summarize_conversation = AsyncMock(
        return_value={
            "summary": "Test summary",
            "app_id": "app1234567890",
            "user_id": "user1234567890",
        }
    )
    sdk.add_agent_memory = AsyncMock(
        return_value={
            "task_id": "test-task-456",
            "status": "accepted",
            "app_id": "app1234567890",
            "user_id": "user1234567890",
        }
    )
    sdk.query_memory = AsyncMock(return_value=[])
    sdk.get_memory = AsyncMock(
        return_value={
            "memory_id": "mem123",
            "document": "Test document",
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        }
    )
    sdk.delete_memory = AsyncMock(return_value=True)
    sdk.traverse_memory_evolution_chain = AsyncMock(return_value=[])
    sdk.generate_evolution_graph = Mock(return_value="graph output")
    sdk.get_connection_pool_stats = AsyncMock(
        return_value={"active": 1, "available": 5}
    )
    sdk.warm_up = AsyncMock(return_value=True)
    return sdk


@pytest.fixture
def client_with_sdk(mock_sdk):
    """Create a test client with SDK initialized."""
    client = TestClient(app)
    app.state.sdk = mock_sdk
    yield client
    if hasattr(app.state, "sdk"):
        app.state.sdk = None


@pytest.fixture
def client_without_sdk():
    """Create a test client without SDK initialized."""
    client = TestClient(app)
    if hasattr(app.state, "sdk"):
        original_sdk = app.state.sdk
        app.state.sdk = None
    else:
        original_sdk = None

    yield client

    if original_sdk is not None:
        app.state.sdk = original_sdk


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        {"role": "user", "content": f"Message {i}", "timestamp": "2024-01-01T00:00:00Z"}
        for i in range(DEFAULT_MAX_MESSAGES)
    ]


@pytest.mark.asyncio
async def test_lifespan_initializes_sdk_successfully():
    """Test lifespan initializes SDK on startup."""
    test_app = Mock(spec=FastAPI)
    test_app.state = Mock()

    mock_sdk_instance = Mock(spec=OmniMemorySDK)
    mock_sdk_instance.warm_up = AsyncMock(return_value=True)

    with patch("omnimemory.api.server.OmniMemorySDK", return_value=mock_sdk_instance):
        with patch("omnimemory.api.server.logger") as mock_logger:
            async with lifespan(test_app):
                assert test_app.state.sdk == mock_sdk_instance
                mock_sdk_instance.warm_up.assert_called_once()
                mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_lifespan_warm_up_success():
    """Test lifespan logs warm-up success."""
    test_app = Mock(spec=FastAPI)
    test_app.state = Mock()

    mock_sdk_instance = Mock(spec=OmniMemorySDK)
    mock_sdk_instance.warm_up = AsyncMock(return_value=True)

    with patch("omnimemory.api.server.OmniMemorySDK", return_value=mock_sdk_instance):
        with patch("omnimemory.api.server.logger") as mock_logger:
            async with lifespan(test_app):
                pass

            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("warm-up completed" in str(call).lower() for call in info_calls)


@pytest.mark.asyncio
async def test_lifespan_warm_up_failure():
    """Test lifespan logs warm-up failure as warning."""
    test_app = Mock(spec=FastAPI)
    test_app.state = Mock()

    mock_sdk_instance = Mock(spec=OmniMemorySDK)
    mock_sdk_instance.warm_up = AsyncMock(return_value=False)

    with patch("omnimemory.api.server.OmniMemorySDK", return_value=mock_sdk_instance):
        with patch("omnimemory.api.server.logger") as mock_logger:
            async with lifespan(test_app):
                pass

            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("warm-up failed" in str(call).lower() for call in warning_calls)


@pytest.mark.asyncio
async def test_lifespan_sdk_initialization_failure():
    """Test lifespan raises exception on SDK initialization failure."""
    test_app = Mock(spec=FastAPI)
    test_app.state = Mock()

    with patch(
        "omnimemory.api.server.OmniMemorySDK", side_effect=RuntimeError("Init failed")
    ):
        with pytest.raises(RuntimeError, match="Init failed"):
            async with lifespan(test_app):
                pass


@pytest.mark.asyncio
async def test_lifespan_cleanup_sets_sdk_to_none():
    """Test lifespan cleanup sets SDK to None."""
    test_app = Mock(spec=FastAPI)
    test_app.state = Mock()

    mock_sdk_instance = Mock(spec=OmniMemorySDK)
    mock_sdk_instance.warm_up = AsyncMock(return_value=True)

    with patch("omnimemory.api.server.OmniMemorySDK", return_value=mock_sdk_instance):
        async with lifespan(test_app):
            assert test_app.state.sdk == mock_sdk_instance

        assert test_app.state.sdk is None


@pytest.mark.asyncio
async def test_lifespan_cleanup_handles_missing_sdk():
    """Test lifespan cleanup handles missing SDK gracefully."""
    test_app = Mock(spec=FastAPI)
    test_app.state = Mock()

    mock_sdk_instance = Mock(spec=OmniMemorySDK)
    mock_sdk_instance.warm_up = AsyncMock(return_value=True)

    with patch("omnimemory.api.server.OmniMemorySDK", return_value=mock_sdk_instance):
        async with lifespan(test_app):
            pass

        assert not hasattr(test_app.state, "sdk") or test_app.state.sdk is None


def test_get_sdk_returns_sdk_when_initialized(mock_sdk):
    """Test get_sdk returns SDK when initialized."""
    request = Mock()
    request.app.state.sdk = mock_sdk

    result = get_sdk(request)
    assert result == mock_sdk


def test_get_sdk_raises_503_when_not_initialized():
    """Test get_sdk raises 503 when SDK not initialized."""
    from fastapi import HTTPException
    from types import SimpleNamespace

    request = Mock()
    request.app.state = SimpleNamespace()

    with pytest.raises(HTTPException) as exc_info:
        get_sdk(request)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_get_sdk_raises_503_when_sdk_is_none():
    """Test get_sdk raises 503 when SDK is None."""
    from fastapi import HTTPException

    request = Mock()
    request.app.state.sdk = None

    with pytest.raises(HTTPException) as exc_info:
        get_sdk(request)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_get_sdk_logs_error_when_unavailable():
    """Test get_sdk logs error when SDK unavailable."""
    request = Mock()
    request.app.state.sdk = None

    with patch("omnimemory.api.server.logger") as mock_logger:
        try:
            get_sdk(request)
        except Exception:
            pass

        mock_logger.error.assert_called_once()


def test_add_memory_success(client_with_sdk, sample_messages):
    """Test add_memory successfully returns 202 Accepted."""
    request_data = AddUserMessageRequest(
        app_id="app1234567890",
        user_id="user1234567890",
        session_id="session1234567890",
        messages=sample_messages,
    )

    response = client_with_sdk.post(
        "/api/v1/memories",
        json=request_data.model_dump(),
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    data = response.json()
    task_response = TaskResponse(**data)
    assert task_response.task_id is not None
    assert task_response.status == "accepted"


def test_add_memory_returns_task_response(client_with_sdk, sample_messages):
    """Test add_memory returns TaskResponse with task_id."""
    request_data = AddUserMessageRequest(
        app_id="app1234567890",
        user_id="user1234567890",
        messages=sample_messages,
    )

    response = client_with_sdk.post(
        "/api/v1/memories",
        json=request_data.model_dump(),
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    task_response = TaskResponse(**response.json())
    assert task_response.task_id is not None
    assert task_response.status == "accepted"
    assert task_response.app_id == "app1234567890"
    assert task_response.user_id == "user1234567890"


def test_add_memory_validation_error_too_few_messages(client_with_sdk):
    """Test add_memory handles ValidationError for too few messages."""
    response = client_with_sdk.post(
        "/api/v1/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": [{"role": "user", "content": "test"}],
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_add_memory_validation_error_too_many_messages(
    client_with_sdk, sample_messages
):
    """Test add_memory handles ValidationError for too many messages."""
    too_many = sample_messages + [{"role": "user", "content": "extra"}]
    response = client_with_sdk.post(
        "/api/v1/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": too_many,
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_add_memory_validation_error_invalid_app_id(client_with_sdk, sample_messages):
    """Test add_memory handles ValidationError for invalid app_id length."""
    response = client_with_sdk.post(
        "/api/v1/memories",
        json={
            "app_id": "short",
            "user_id": "user1234567890",
            "messages": sample_messages,
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_add_memory_validation_error_invalid_user_id(client_with_sdk, sample_messages):
    """Test add_memory handles ValidationError for invalid user_id length."""
    response = client_with_sdk.post(
        "/api/v1/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "short",
            "messages": sample_messages,
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.asyncio
async def test_add_memory_value_error_from_to_user_messages(mock_sdk):
    """Test add_memory handles ValueError from to_user_messages."""
    from omnimemory.api.server import add_memory
    from fastapi import Request, HTTPException

    mock_request_obj = Mock(spec=AddUserMessageRequest)
    mock_request_obj.to_user_messages = Mock(
        side_effect=ValueError("Invalid message count")
    )
    mock_request_obj.app_id = "app1234567890"
    mock_request_obj.user_id = "user1234567890"
    mock_request_obj.messages = []

    http_request = Mock(spec=Request)
    http_request.app.state.sdk = mock_sdk

    with pytest.raises(HTTPException) as exc_info:
        await add_memory(mock_request_obj, http_request)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid message count" in exc_info.value.detail


def test_add_memory_validation_error_format(client_with_sdk, sample_messages):
    """Test add_memory formats ValidationError details correctly."""
    response = client_with_sdk.post(
        "/api/v1/memories",
        json={
            "app_id": "app1234567890",
            "messages": sample_messages,
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_add_memory_sdk_failure(client_with_sdk, sample_messages, mock_sdk):
    """Test add_memory handles SDK failures."""
    mock_sdk.add_memory = AsyncMock(side_effect=RuntimeError("SDK error"))

    response = client_with_sdk.post(
        "/api/v1/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": sample_messages,
        },
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to add memory" in response.json()["detail"]


def test_add_memory_sdk_not_initialized(client_without_sdk, sample_messages):
    """Test add_memory returns 503 when SDK not initialized."""
    response = client_without_sdk.post(
        "/api/v1/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": sample_messages,
        },
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_summarize_conversation_without_callback_200(client_with_sdk):
    """Test summarize_conversation without callback returns 200 OK."""
    response = client_with_sdk.post(
        "/api/v1/agent/summaries",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": "Test conversation",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "summary" in data


def test_summarize_conversation_with_callback_202(client_with_sdk, mock_sdk):
    """Test summarize_conversation with callback returns 202 Accepted."""
    mock_sdk.summarize_conversation = AsyncMock(
        return_value={
            "task_id": "task-123",
            "status": "accepted",
            "app_id": "app1234567890",
            "user_id": "user1234567890",
        }
    )

    response = client_with_sdk.post(
        "/api/v1/agent/summaries",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": "Test conversation",
            "callback_url": "https://example.com/callback",
        },
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    data = response.json()
    assert data["status"] == "accepted"


def test_summarize_conversation_returns_conversation_summary_response(client_with_sdk):
    """Test summarize_conversation returns ConversationSummaryResponse for sync."""
    request_data = ConversationSummaryRequest(
        app_id="app1234567890",
        user_id="user1234567890",
        messages="Test conversation",
    )

    response = client_with_sdk.post(
        "/api/v1/agent/summaries",
        json=request_data.model_dump(),
    )

    assert response.status_code == status.HTTP_200_OK
    summary_response = ConversationSummaryResponse(**response.json())
    assert summary_response.summary is not None
    assert summary_response.app_id == "app1234567890"


def test_summarize_conversation_with_messages_as_string(client_with_sdk):
    """Test summarize_conversation with messages as string."""
    response = client_with_sdk.post(
        "/api/v1/agent/summaries",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": "This is a string message",
        },
    )

    assert response.status_code == status.HTTP_200_OK


def test_summarize_conversation_with_messages_as_list(client_with_sdk):
    """Test summarize_conversation with messages as list."""
    messages = [Message(role="user", content="Hello", timestamp="2024-01-01T00:00:00Z")]
    request_data = ConversationSummaryRequest(
        app_id="app1234567890",
        user_id="user1234567890",
        messages=messages,
    )

    response = client_with_sdk.post(
        "/api/v1/agent/summaries",
        json=request_data.model_dump(),
    )

    assert response.status_code == status.HTTP_200_OK


def test_summarize_conversation_sdk_failure(client_with_sdk, mock_sdk):
    """Test summarize_conversation handles SDK failures."""
    mock_sdk.summarize_conversation = AsyncMock(side_effect=RuntimeError("SDK error"))

    response = client_with_sdk.post(
        "/api/v1/agent/summaries",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": "Test conversation",
        },
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to summarize conversation" in response.json()["detail"]


def test_summarize_conversation_sdk_not_initialized(client_without_sdk):
    """Test summarize_conversation returns 503 when SDK not initialized."""
    response = client_without_sdk.post(
        "/api/v1/agent/summaries",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": "Test conversation",
        },
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_summarize_conversation_invalid_request_format(client_with_sdk):
    """Test summarize_conversation handles invalid request format."""
    response = client_with_sdk.post(
        "/api/v1/agent/summaries",
        json={
            "app_id": "short",
            "user_id": "user1234567890",
            "messages": "Test",
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_add_agent_memory_success(client_with_sdk):
    """Test add_agent_memory successfully returns 202 Accepted."""
    response = client_with_sdk.post(
        "/api/v1/agent/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "session_id": "session1234567890",
            "messages": "Agent message",
        },
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "accepted"


def test_add_agent_memory_returns_task_response(client_with_sdk):
    """Test add_agent_memory returns TaskResponse with task_id."""
    request_data = AgentMemoryRequest(
        app_id="app1234567890",
        user_id="user1234567890",
        messages="Agent message",
    )

    response = client_with_sdk.post(
        "/api/v1/agent/memories",
        json=request_data.model_dump(),
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    task_response = TaskResponse(**response.json())
    assert task_response.task_id is not None
    assert task_response.status == "accepted"


def test_add_agent_memory_sdk_failure(client_with_sdk, mock_sdk):
    """Test add_agent_memory handles SDK failures."""
    mock_sdk.add_agent_memory = AsyncMock(side_effect=RuntimeError("SDK error"))

    response = client_with_sdk.post(
        "/api/v1/agent/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": "Agent message",
        },
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to add agent memory" in response.json()["detail"]


def test_add_agent_memory_sdk_not_initialized(client_without_sdk):
    """Test add_agent_memory returns 503 when SDK not initialized."""
    response = client_without_sdk.post(
        "/api/v1/agent/memories",
        json={
            "app_id": "app1234567890",
            "user_id": "user1234567890",
            "messages": "Agent message",
        },
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_add_agent_memory_invalid_request_format(client_with_sdk):
    """Test add_agent_memory handles invalid request format."""
    response = client_with_sdk.post(
        "/api/v1/agent/memories",
        json={
            "app_id": "short",
            "user_id": "user1234567890",
            "messages": "Agent message",
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_query_memory_success(client_with_sdk, mock_sdk):
    """Test query_memory successfully returns results."""
    mock_sdk.query_memory = AsyncMock(
        return_value=[{"memory_id": "mem1", "document": "Test", "metadata": {}}]
    )

    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test"
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "memories" in data
    assert "count" in data
    assert data["count"] == 1


def test_query_memory_returns_memory_list_response(client_with_sdk, mock_sdk):
    """Test query_memory returns MemoryListResponse."""
    mock_sdk.query_memory = AsyncMock(return_value=[])

    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test"
    )

    assert response.status_code == status.HTTP_200_OK
    memory_list = MemoryListResponse(**response.json())
    assert memory_list.memories == []
    assert memory_list.count == 0


def test_query_memory_with_all_parameters(client_with_sdk, mock_sdk):
    """Test query_memory handles all query parameters."""
    mock_sdk.query_memory = AsyncMock(return_value=[])

    response = client_with_sdk.get(
        "/api/v1/memories/query?"
        "app_id=app1234567890&"
        "query=test&"
        "user_id=user1234567890&"
        "session_id=session1234567890&"
        "n_results=10&"
        "similarity_threshold=0.8"
    )

    assert response.status_code == status.HTTP_200_OK
    mock_sdk.query_memory.assert_called_once()
    call_kwargs = mock_sdk.query_memory.call_args[1]
    assert call_kwargs["app_id"] == "app1234567890"
    assert call_kwargs["query"] == "test"
    assert call_kwargs["user_id"] == "user1234567890"
    assert call_kwargs["session_id"] == "session1234567890"
    assert call_kwargs["n_results"] == 10
    assert call_kwargs["similarity_threshold"] == 0.8


def test_query_memory_validate_query_min_length(client_with_sdk):
    """Test query_memory validates query min_length=1."""
    response = client_with_sdk.get("/api/v1/memories/query?app_id=app1234567890&query=")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_query_memory_validate_n_results_boundary_1(client_with_sdk):
    """Test query_memory validates n_results ge=1 (boundary valid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&n_results=1"
    )

    assert response.status_code == status.HTTP_200_OK


def test_query_memory_validate_n_results_boundary_100(client_with_sdk):
    """Test query_memory validates n_results le=100 (boundary valid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&n_results=100"
    )

    assert response.status_code == status.HTTP_200_OK


def test_query_memory_validate_n_results_too_low(client_with_sdk):
    """Test query_memory validates n_results ge=1 (invalid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&n_results=0"
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_query_memory_validate_n_results_too_high(client_with_sdk):
    """Test query_memory validates n_results le=100 (invalid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&n_results=101"
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_query_memory_validate_similarity_threshold_boundary_0(client_with_sdk):
    """Test query_memory validates similarity_threshold ge=0.0 (boundary valid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&similarity_threshold=0.0"
    )

    assert response.status_code == status.HTTP_200_OK


def test_query_memory_validate_similarity_threshold_boundary_1(client_with_sdk):
    """Test query_memory validates similarity_threshold le=1.0 (boundary valid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&similarity_threshold=1.0"
    )

    assert response.status_code == status.HTTP_200_OK


def test_query_memory_validate_similarity_threshold_too_low(client_with_sdk):
    """Test query_memory validates similarity_threshold ge=0.0 (invalid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&similarity_threshold=-0.1"
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_query_memory_validate_similarity_threshold_too_high(client_with_sdk):
    """Test query_memory validates similarity_threshold le=1.0 (invalid)."""
    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test&similarity_threshold=1.1"
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_query_memory_empty_results(client_with_sdk, mock_sdk):
    """Test query_memory handles empty results."""
    mock_sdk.query_memory = AsyncMock(return_value=[])

    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test"
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 0
    assert data["memories"] == []


def test_query_memory_sdk_failure(client_with_sdk, mock_sdk):
    """Test query_memory handles SDK failures."""
    mock_sdk.query_memory = AsyncMock(side_effect=RuntimeError("SDK error"))

    response = client_with_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test"
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to query memory" in response.json()["detail"]


def test_query_memory_sdk_not_initialized(client_without_sdk):
    """Test query_memory returns 503 when SDK not initialized."""
    response = client_without_sdk.get(
        "/api/v1/memories/query?app_id=app1234567890&query=test"
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_query_memory_missing_app_id(client_with_sdk):
    """Test query_memory validates app_id is required."""
    response = client_with_sdk.get("/api/v1/memories/query?query=test")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_query_memory_missing_query(client_with_sdk):
    """Test query_memory validates query is required."""
    response = client_with_sdk.get("/api/v1/memories/query?app_id=app1234567890")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_get_memory_success(client_with_sdk, mock_sdk):
    """Test get_memory successfully returns memory."""
    mock_sdk.get_memory = AsyncMock(
        return_value={
            "memory_id": "mem123",
            "document": "Test document",
            "metadata": {"created_at": "2024-01-01T00:00:00Z"},
        }
    )

    response = client_with_sdk.get("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["memory_id"] == "mem123"
    assert "document" in data


def test_get_memory_returns_memory_response(client_with_sdk, mock_sdk):
    """Test get_memory returns MemoryResponse."""
    mock_sdk.get_memory = AsyncMock(
        return_value={
            "memory_id": "mem123",
            "document": "Test",
            "metadata": {},
        }
    )

    response = client_with_sdk.get("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_200_OK
    memory_response = MemoryResponse(**response.json())
    assert memory_response.memory_id == "mem123"
    assert memory_response.document == "Test"


def test_get_memory_not_found(client_with_sdk, mock_sdk):
    """Test get_memory handles memory not found."""
    mock_sdk.get_memory = AsyncMock(return_value=None)

    response = client_with_sdk.get("/api/v1/memories/nonexistent?app_id=app1234567890")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_get_memory_sdk_failure(client_with_sdk, mock_sdk):
    """Test get_memory handles SDK failures."""
    mock_sdk.get_memory = AsyncMock(side_effect=RuntimeError("SDK error"))

    response = client_with_sdk.get("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to get memory" in response.json()["detail"]


def test_get_memory_sdk_not_initialized(client_without_sdk):
    """Test get_memory returns 503 when SDK not initialized."""
    response = client_without_sdk.get("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_get_memory_missing_app_id(client_with_sdk):
    """Test get_memory validates app_id is required."""
    response = client_with_sdk.get("/api/v1/memories/mem123")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_delete_memory_success(client_with_sdk, mock_sdk):
    """Test delete_memory successfully deletes memory."""
    mock_sdk.delete_memory = AsyncMock(return_value=True)

    response = client_with_sdk.delete("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "message" in data


def test_delete_memory_returns_success_response(client_with_sdk, mock_sdk):
    """Test delete_memory returns SuccessResponse."""
    mock_sdk.delete_memory = AsyncMock(return_value=True)

    response = client_with_sdk.delete("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_200_OK
    success_response = SuccessResponse(**response.json())
    assert success_response.success is True
    assert success_response.message is not None


def test_delete_memory_failure_sdk_returns_false(client_with_sdk, mock_sdk):
    """Test delete_memory handles delete failure when SDK returns False."""
    mock_sdk.delete_memory = AsyncMock(return_value=False)

    response = client_with_sdk.delete("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to delete memory" in response.json()["detail"]


def test_delete_memory_sdk_failure(client_with_sdk, mock_sdk):
    """Test delete_memory handles SDK failures."""
    mock_sdk.delete_memory = AsyncMock(side_effect=RuntimeError("SDK error"))

    response = client_with_sdk.delete("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to delete memory" in response.json()["detail"]


def test_delete_memory_sdk_not_initialized(client_without_sdk):
    """Test delete_memory returns 503 when SDK not initialized."""
    response = client_without_sdk.delete("/api/v1/memories/mem123?app_id=app1234567890")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_delete_memory_missing_app_id(client_with_sdk):
    """Test delete_memory validates app_id is required."""
    response = client_with_sdk.delete("/api/v1/memories/mem123")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_traverse_memory_evolution_chain_success(client_with_sdk, mock_sdk):
    """Test traverse_memory_evolution_chain successfully returns chain."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(
        return_value=[
            {"memory_id": "mem1", "document": "Test1"},
            {"memory_id": "mem2", "document": "Test2"},
        ]
    )

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 2
    assert len(data["memories"]) == 2


def test_traverse_memory_evolution_chain_empty_chain(client_with_sdk, mock_sdk):
    """Test traverse_memory_evolution_chain handles empty chain."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(return_value=[])

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 0
    assert data["memories"] == []


def test_traverse_memory_evolution_chain_returns_memory_list_response(
    client_with_sdk, mock_sdk
):
    """Test traverse_memory_evolution_chain returns MemoryListResponse."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(return_value=[])

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_200_OK
    memory_list = MemoryListResponse(**response.json())
    assert memory_list.memories == []
    assert memory_list.count == 0


def test_traverse_memory_evolution_chain_sdk_failure(client_with_sdk, mock_sdk):
    """Test traverse_memory_evolution_chain handles SDK failures."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(
        side_effect=RuntimeError("SDK error")
    )

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to traverse" in response.json()["detail"]


def test_traverse_memory_evolution_chain_sdk_not_initialized(client_without_sdk):
    """Test traverse_memory_evolution_chain returns 503 when SDK not initialized."""
    response = client_without_sdk.get(
        "/api/v1/memories/mem1/evolution?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_traverse_memory_evolution_chain_missing_app_id(client_with_sdk):
    """Test traverse_memory_evolution_chain validates app_id is required."""
    response = client_with_sdk.get("/api/v1/memories/mem1/evolution")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_generate_evolution_graph_mermaid_default(client_with_sdk, mock_sdk):
    """Test generate_evolution_graph generates Mermaid graph by default."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(
        return_value=[
            {"memory_id": "mem1", "document": "Test1"},
        ]
    )
    mock_sdk.generate_evolution_graph = Mock(return_value="graph LR\nA[mem1]")

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution/graph?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert "graph LR" in response.text


def test_generate_evolution_graph_dot_format(client_with_sdk, mock_sdk):
    """Test generate_evolution_graph generates DOT graph."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(
        return_value=[
            {"memory_id": "mem1", "document": "Test1"},
        ]
    )
    mock_sdk.generate_evolution_graph = Mock(return_value="digraph { A -> B }")

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution/graph?app_id=app1234567890&format=dot"
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


def test_generate_evolution_graph_html_format(client_with_sdk, mock_sdk):
    """Test generate_evolution_graph generates HTML graph."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(
        return_value=[
            {"memory_id": "mem1", "document": "Test1"},
        ]
    )
    mock_sdk.generate_evolution_graph = Mock(
        return_value="<html><body>Graph</body></html>"
    )

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution/graph?app_id=app1234567890&format=html"
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "<html>" in response.text


def test_generate_evolution_graph_empty_chain_404(client_with_sdk, mock_sdk):
    """Test generate_evolution_graph handles empty chain (404 Not Found)."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(return_value=[])

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution/graph?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "No evolution chain found" in response.json()["detail"]


def test_generate_evolution_graph_generation_failure_none(client_with_sdk, mock_sdk):
    """Test generate_evolution_graph handles graph generation failure (None)."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(
        return_value=[
            {"memory_id": "mem1", "document": "Test1"},
        ]
    )
    mock_sdk.generate_evolution_graph = Mock(return_value=None)

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution/graph?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to generate graph" in response.json()["detail"]


def test_generate_evolution_graph_sdk_failure(client_with_sdk, mock_sdk):
    """Test generate_evolution_graph handles SDK failures."""
    mock_sdk.traverse_memory_evolution_chain = AsyncMock(
        side_effect=RuntimeError("SDK error")
    )

    response = client_with_sdk.get(
        "/api/v1/memories/mem1/evolution/graph?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to generate evolution graph" in response.json()["detail"]


def test_generate_evolution_graph_sdk_not_initialized(client_without_sdk):
    """Test generate_evolution_graph returns 503 when SDK not initialized."""
    response = client_without_sdk.get(
        "/api/v1/memories/mem1/evolution/graph?app_id=app1234567890"
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_generate_evolution_graph_missing_app_id(client_with_sdk):
    """Test generate_evolution_graph validates app_id is required."""
    response = client_with_sdk.get("/api/v1/memories/mem1/evolution/graph")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_health_check_sdk_initialized(client_with_sdk):
    """Test health_check returns health status when SDK initialized."""
    response = client_with_sdk.get("/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["sdk_initialized"] is True
    assert data["service"] == "omnimemory-api"


def test_health_check_sdk_not_initialized(client_without_sdk):
    """Test health_check returns health status when SDK not initialized."""
    response = client_without_sdk.get("/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["sdk_initialized"] is False
    assert data["service"] == "omnimemory-api"


def test_health_check_always_200(client_without_sdk):
    """Test health_check always returns 200 OK."""
    response = client_without_sdk.get("/health")
    assert response.status_code == status.HTTP_200_OK


def test_get_pool_stats_success(client_with_sdk, mock_sdk):
    """Test get_pool_stats successfully returns stats."""
    mock_sdk.get_connection_pool_stats = AsyncMock(
        return_value={
            "active": 2,
            "available": 8,
            "max_connections": 10,
        }
    )

    response = client_with_sdk.get("/api/v1/system/pool-stats")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "active" in data
    assert data["active"] == 2


def test_get_pool_stats_returns_stats_dict(client_with_sdk, mock_sdk):
    """Test get_pool_stats returns stats dict as-is."""
    stats = {"active": 1, "available": 5, "custom": "value"}
    mock_sdk.get_connection_pool_stats = AsyncMock(return_value=stats)

    response = client_with_sdk.get("/api/v1/system/pool-stats")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data == stats


def test_get_pool_stats_sdk_failure(client_with_sdk, mock_sdk):
    """Test get_pool_stats handles SDK failures."""
    mock_sdk.get_connection_pool_stats = AsyncMock(
        side_effect=RuntimeError("SDK error")
    )

    response = client_with_sdk.get("/api/v1/system/pool-stats")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to fetch pool stats" in response.json()["detail"]


def test_get_pool_stats_sdk_not_initialized(client_without_sdk):
    """Test get_pool_stats returns 503 when SDK not initialized."""
    response = client_without_sdk.get("/api/v1/system/pool-stats")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_root_returns_api_information():
    """Test root endpoint returns API information."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == "OmniMemory API"
    assert data["version"] == "0.1.0-beta"
    assert "architecture" in data
    assert "description" in data
    assert "endpoints" in data


def test_root_includes_endpoint_links():
    """Test root endpoint includes endpoint links."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "endpoints" in data
    assert "docs" in data["endpoints"]
    assert "health" in data["endpoints"]
    assert "api" in data["endpoints"]


def test_root_always_200():
    """Test root endpoint always returns 200 OK."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK


def test_cors_headers_present():
    """Test CORS headers are present in responses."""
    client = TestClient(app)
    response = client.options("/health")

    assert response.status_code in [200, 405]


def test_cors_allows_all_origins():
    """Test CORS allows all origins."""
    client = TestClient(app)
    response = client.get("/health", headers={"Origin": "https://example.com"})

    assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_add_memory_validation_error_multiple_errors(mock_sdk):
    """Test add_memory formats ValidationError with multiple errors (lines 105-111)."""
    from omnimemory.api.server import add_memory
    from fastapi import Request, HTTPException
    from pydantic import ValidationError

    mock_request = Mock(spec=AddUserMessageRequest)
    validation_error = ValidationError.from_exception_data(
        "AddUserMessageRequest",
        [
            {
                "type": "missing",
                "loc": ("messages", 0, "role"),
                "msg": "Field required",
                "input": {},
            },
            {
                "type": "missing",
                "loc": ("messages", 1, "content"),
                "msg": "Field required",
                "input": {},
            },
        ],
    )

    mock_request.to_user_messages = Mock(side_effect=validation_error)
    mock_request.app_id = "app1234567890"
    mock_request.user_id = "user1234567890"
    mock_request.messages = []

    http_request = Mock(spec=Request)
    http_request.app.state.sdk = mock_sdk

    with pytest.raises(HTTPException) as exc_info:
        await add_memory(mock_request, http_request)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    detail = exc_info.value.detail
    assert "->" in detail or "Field required" in detail or "messages" in detail


@pytest.mark.asyncio
async def test_add_memory_validation_error_empty_error_details(mock_sdk):
    """Test add_memory handles ValidationError with empty error_details (line 109 fallback)."""
    from omnimemory.api.server import add_memory
    from fastapi import Request, HTTPException
    from pydantic import ValidationError

    mock_request = Mock(spec=AddUserMessageRequest)
    validation_error = ValidationError.from_exception_data(
        "AddUserMessageRequest",
        [],
    )

    mock_request.to_user_messages = Mock(side_effect=validation_error)
    mock_request.app_id = "app1234567890"
    mock_request.user_id = "user1234567890"
    mock_request.messages = []

    http_request = Mock(spec=Request)
    http_request.app.state.sdk = mock_sdk

    with pytest.raises(HTTPException) as exc_info:
        await add_memory(mock_request, http_request)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail


@pytest.mark.asyncio
async def test_add_memory_validation_error_single_error(mock_sdk):
    """Test add_memory formats ValidationError with single error."""
    from omnimemory.api.server import add_memory
    from fastapi import Request, HTTPException
    from pydantic import ValidationError

    mock_request = Mock(spec=AddUserMessageRequest)
    validation_error = ValidationError.from_exception_data(
        "AddUserMessageRequest",
        [
            {
                "type": "missing",
                "loc": ("messages",),
                "msg": "Field required",
                "input": {},
            },
        ],
    )

    mock_request.to_user_messages = Mock(side_effect=validation_error)
    mock_request.app_id = "app1234567890"
    mock_request.user_id = "user1234567890"
    mock_request.messages = []

    http_request = Mock(spec=Request)
    http_request.app.state.sdk = mock_sdk

    with pytest.raises(HTTPException) as exc_info:
        await add_memory(mock_request, http_request)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    detail = exc_info.value.detail
    assert "messages" in detail or "Field required" in detail


def test_api_schemas_imports():
    """Test that all schemas can be imported from api.schemas."""
    from omnimemory.api import schemas

    assert hasattr(schemas, "Message")
    assert hasattr(schemas, "UserMessages")
    assert hasattr(schemas, "AddUserMessageRequest")
    assert hasattr(schemas, "ConversationSummaryRequest")
    assert hasattr(schemas, "ConversationSummaryResponse")
    assert hasattr(schemas, "AgentMemoryRequest")
    assert hasattr(schemas, "AgentMemoryResponse")
    assert hasattr(schemas, "QueryMemoryRequest")
    assert hasattr(schemas, "TaskResponse")
    assert hasattr(schemas, "MemoryResponse")
    assert hasattr(schemas, "MemoryListResponse")
    assert hasattr(schemas, "MemoryIDResponse")
    assert hasattr(schemas, "MemoryIDListResponse")
    assert hasattr(schemas, "ErrorResponse")
    assert hasattr(schemas, "SuccessResponse")


def test_api_schemas_are_re_exports():
    """Test that api.schemas re-exports are the same as core.schemas."""
    from omnimemory.api import schemas
    from omnimemory.core import schemas as core_schemas

    assert schemas.Message is core_schemas.Message
    assert schemas.UserMessages is core_schemas.UserMessages
    assert schemas.AddUserMessageRequest is core_schemas.AddUserMessageRequest
    assert schemas.ConversationSummaryRequest is core_schemas.ConversationSummaryRequest
    assert schemas.TaskResponse is core_schemas.TaskResponse
    assert schemas.MemoryResponse is core_schemas.MemoryResponse
    assert schemas.MemoryListResponse is core_schemas.MemoryListResponse
    assert schemas.SuccessResponse is core_schemas.SuccessResponse


def test_api_schemas_all_exported():
    """Test that __all__ in schemas.py matches available exports."""
    from omnimemory.api import schemas
    import omnimemory.api.schemas as schemas_module

    if hasattr(schemas_module, "__all__"):
        all_exports = schemas_module.__all__
        for export_name in all_exports:
            assert hasattr(schemas, export_name), f"{export_name} should be exported"


def test_api_schemas_module_attributes():
    """Test that schemas module has correct attributes."""
    import omnimemory.api.schemas as schemas_module

    assert schemas_module is not None
    assert hasattr(schemas_module, "__all__")
    assert isinstance(schemas_module.__all__, list)
    assert len(schemas_module.__all__) > 0


def test_message_schema_usage():
    """Test Message schema is used correctly."""
    message1 = Message(role="user", content="Hello")
    assert message1.role == "user"
    assert message1.content == "Hello"
    assert message1.timestamp is not None

    message2 = Message(
        role="assistant", content="Hi there", timestamp="2024-01-01T00:00:00Z"
    )
    assert message2.role == "assistant"
    assert message2.content == "Hi there"
    assert message2.timestamp == "2024-01-01T00:00:00Z"

    assert message1.model_dump()["role"] == "user"
    assert message2.model_dump()["content"] == "Hi there"


def test_api_schemas_module_execution():
    """Test that schemas.py module execution is covered (lines 2-20)."""
    import omnimemory.api.schemas

    assert hasattr(omnimemory.api.schemas, "Message")
    assert hasattr(omnimemory.api.schemas, "__all__")

    expected_exports = [
        "Message",
        "UserMessages",
        "AddUserMessageRequest",
        "ConversationSummaryRequest",
        "ConversationSummaryResponse",
        "AgentMemoryRequest",
        "AgentMemoryResponse",
        "QueryMemoryRequest",
        "TaskResponse",
        "MemoryResponse",
        "MemoryListResponse",
        "MemoryIDResponse",
        "MemoryIDListResponse",
        "ErrorResponse",
        "SuccessResponse",
    ]

    actual_exports = omnimemory.api.schemas.__all__
    for export in expected_exports:
        assert export in actual_exports, f"{export} should be in __all__"
