"""
Comprehensive unit tests for OmniMemory core schemas.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from omnimemory.core.schemas import (
    Message,
    UserMessages,
    AddUserMessageRequest,
    ConversationSummaryRequest,
    QueryMemoryRequest,
    AgentMemoryRequest,
    TaskResponse,
    MemoryResponse,
    MemoryListResponse,
    MemoryIDResponse,
    MemoryIDListResponse,
    ErrorResponse,
    SuccessResponse,
    ConversationSummaryResponse,
    AgentMemoryResponse,
)
from omnimemory.core.config import DEFAULT_MAX_MESSAGES


class TestMessage:
    """Test cases for Message schema."""

    def test_create_message_with_all_fields(self):
        """Test create message with role, content, timestamp."""
        msg = Message(role="user", content="Hello", timestamp="2024-01-01T00:00:00Z")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == "2024-01-01T00:00:00Z"

    def test_create_message_auto_generate_timestamp(self):
        """Test auto-generate timestamp when not provided."""
        msg = Message(role="user", content="Hello")
        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, str)

    def test_create_message_validate_required_fields(self):
        """Test validate required fields."""
        with pytest.raises(ValidationError):
            Message(role="user")

        with pytest.raises(ValidationError):
            Message(content="Hello")

    def test_create_message_handle_optional_timestamp(self):
        """Test handle optional timestamp."""
        msg1 = Message(role="user", content="Hello", timestamp="2024-01-01T00:00:00Z")
        msg2 = Message(role="user", content="Hello")

        assert msg1.timestamp == "2024-01-01T00:00:00Z"
        assert msg2.timestamp is not None


class TestUserMessages:
    """Test cases for UserMessages schema."""

    def test_create_with_valid_data(self):
        """Test create with valid data."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES)
        ]
        user_msg = UserMessages(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id="session1234567890",
            messages=messages,
        )
        assert user_msg.app_id == "app1234567890"
        assert len(user_msg.messages) == DEFAULT_MAX_MESSAGES

    def test_validate_app_id_length(self):
        """Test validate app_id length (10-36)."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES)
        ]

        with pytest.raises(ValidationError):
            UserMessages(app_id="short", user_id="user1234567890", messages=messages)

        with pytest.raises(ValidationError):
            UserMessages(app_id="a" * 37, user_id="user1234567890", messages=messages)

    def test_validate_user_id_length(self):
        """Test validate user_id length (10-36)."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES)
        ]

        with pytest.raises(ValidationError):
            UserMessages(app_id="app1234567890", user_id="short", messages=messages)

    def test_validate_session_id_length(self):
        """Test validate session_id length (10-36, optional)."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES)
        ]

        user_msg = UserMessages(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id=None,
            messages=messages,
        )
        assert user_msg.session_id is None

        with pytest.raises(ValidationError):
            UserMessages(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id="short",
                messages=messages,
            )

    def test_validate_messages_count_exactly_default_max(self):
        """Test validate messages count (exactly DEFAULT_MAX_MESSAGES)."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES - 1)
        ]
        with pytest.raises(ValidationError):
            UserMessages(
                app_id="app1234567890", user_id="user1234567890", messages=messages
            )

        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES + 1)
        ]
        with pytest.raises(ValidationError):
            UserMessages(
                app_id="app1234567890", user_id="user1234567890", messages=messages
            )

    def test_validate_messages_field_validator_coverage_line_60(self):
        """Test field_validator coverage for line 60."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES - 1)
        ]
        with pytest.raises(ValidationError) as exc_info:
            UserMessages(
                app_id="app1234567890", user_id="user1234567890", messages=messages
            )
        error_str = str(exc_info.value)
        assert "at least" in error_str.lower() or "exactly" in error_str.lower()

    def test_validate_messages_model_validator_coverage_line_70(self):
        """Test model_validator coverage for line 70."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES + 1)
        ]
        with pytest.raises(ValidationError) as exc_info:
            UserMessages(
                app_id="app1234567890", user_id="user1234567890", messages=messages
            )
        error_str = str(exc_info.value)
        assert "at most" in error_str.lower() or "exactly" in error_str.lower()

    def test_conversation_summary_response_generated_at_coverage_line_130(self):
        """Test ConversationSummaryResponse generated_at field (line 130)."""
        response = ConversationSummaryResponse(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id="session1234567890",
            summary="Summary text",
            key_points="Key points",
            tags=["tag1"],
            keywords=["kw1"],
            semantic_queries=["query1"],
        )
        assert response.generated_at is not None
        from datetime import datetime

        assert isinstance(response.generated_at, datetime)

    def test_validate_messages_field_validator(self):
        """Test validate messages in field_validator."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES - 1)
        ]
        with pytest.raises(ValidationError):
            UserMessages(
                app_id="app1234567890", user_id="user1234567890", messages=messages
            )

    def test_validate_messages_model_validator(self):
        """Test validate messages in model_validator."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES + 1)
        ]
        with pytest.raises(ValidationError):
            UserMessages(
                app_id="app1234567890", user_id="user1234567890", messages=messages
            )

    def test_handle_none_session_id(self):
        """Test handle None session_id."""
        messages = [
            Message(role="user", content=f"Message {i}")
            for i in range(DEFAULT_MAX_MESSAGES)
        ]
        user_msg = UserMessages(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id=None,
            messages=messages,
        )
        assert user_msg.session_id is None


class TestAddUserMessageRequest:
    """Test cases for AddUserMessageRequest schema."""

    def test_create_with_valid_data(self):
        """Test create with valid data."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(DEFAULT_MAX_MESSAGES)
        ]
        request = AddUserMessageRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id="session1234567890",
            messages=messages,
        )
        assert request.app_id == "app1234567890"
        assert len(request.messages) == DEFAULT_MAX_MESSAGES

    def test_validate_app_id_min_length(self):
        """Test validate app_id (min_length=10)."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(DEFAULT_MAX_MESSAGES)
        ]
        with pytest.raises(ValidationError):
            AddUserMessageRequest(
                app_id="short", user_id="user1234567890", messages=messages
            )

    def test_validate_messages_count(self):
        """Test validate messages count (exactly DEFAULT_MAX_MESSAGES)."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(DEFAULT_MAX_MESSAGES - 1)
        ]
        with pytest.raises(ValidationError):
            AddUserMessageRequest(
                app_id="app1234567890", user_id="user1234567890", messages=messages
            )

    def test_to_user_messages(self):
        """Test convert to UserMessages via to_user_messages()."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(DEFAULT_MAX_MESSAGES)
        ]
        request = AddUserMessageRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id="session1234567890",
            messages=messages,
        )

        user_messages = request.to_user_messages()
        assert isinstance(user_messages, UserMessages)
        assert user_messages.app_id == "app1234567890"
        assert len(user_messages.messages) == DEFAULT_MAX_MESSAGES

    def test_to_user_messages_create_message_objects(self):
        """Test create Message objects from dicts."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(DEFAULT_MAX_MESSAGES)
        ]
        request = AddUserMessageRequest(
            app_id="app1234567890", user_id="user1234567890", messages=messages
        )

        user_messages = request.to_user_messages()
        assert all(isinstance(msg, Message) for msg in user_messages.messages)

    def test_handle_none_session_id(self):
        """Test handle None session_id."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(DEFAULT_MAX_MESSAGES)
        ]
        request = AddUserMessageRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id=None,
            messages=messages,
        )
        assert request.session_id is None


class TestConversationSummaryRequest:
    """Test cases for ConversationSummaryRequest schema."""

    def test_create_with_string_messages(self):
        """Test create with string messages."""
        request = ConversationSummaryRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            messages="This is a conversation",
        )
        assert isinstance(request.messages, str)
        assert request.messages == "This is a conversation"

    def test_create_with_list_messages(self):
        """Test create with list of Message objects."""
        messages = [Message(role="user", content="Hello")]
        request = ConversationSummaryRequest(
            app_id="app1234567890", user_id="user1234567890", messages=messages
        )
        assert isinstance(request.messages, list)
        assert len(request.messages) == 1

    def test_validate_non_empty_string(self):
        """Test validate non-empty string."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ConversationSummaryRequest(
                app_id="app1234567890", user_id="user1234567890", messages=""
            )

    def test_validate_non_empty_list(self):
        """Test validate non-empty list."""
        with pytest.raises(ValidationError, match="At least one message"):
            ConversationSummaryRequest(
                app_id="app1234567890", user_id="user1234567890", messages=[]
            )

    def test_validate_app_id_min_length(self):
        """Test validate app_id (min_length=10)."""
        with pytest.raises(ValidationError):
            ConversationSummaryRequest(
                app_id="short", user_id="user1234567890", messages="test"
            )

    def test_handle_optional_callback_url(self):
        """Test handle optional callback_url."""
        request = ConversationSummaryRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            messages="test",
            callback_url="https://example.com/webhook",
        )
        assert request.callback_url is not None

    def test_handle_optional_callback_headers(self):
        """Test handle optional callback_headers."""
        request = ConversationSummaryRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            messages="test",
            callback_headers={"Authorization": "Bearer token"},
        )
        assert request.callback_headers is not None

    def test_validate_callback_url_format(self):
        """Test validate callback_url format (AnyHttpUrl)."""
        with pytest.raises(ValidationError):
            ConversationSummaryRequest(
                app_id="app1234567890",
                user_id="user1234567890",
                messages="test",
                callback_url="not-a-url",
            )

    def test_raise_value_error_on_empty_string(self):
        """Test raise ValueError on empty string."""
        with pytest.raises(ValidationError):
            ConversationSummaryRequest(
                app_id="app1234567890", user_id="user1234567890", messages=""
            )

    def test_raise_value_error_on_empty_list(self):
        """Test raise ValueError on empty list."""
        with pytest.raises(ValidationError):
            ConversationSummaryRequest(
                app_id="app1234567890", user_id="user1234567890", messages=[]
            )

    def test_raise_value_error_on_invalid_type(self):
        """Test raise ValueError on invalid type."""
        with pytest.raises(ValidationError):
            ConversationSummaryRequest(
                app_id="app1234567890", user_id="user1234567890", messages=123
            )


class TestQueryMemoryRequest:
    """Test cases for QueryMemoryRequest schema."""

    def test_create_with_valid_data(self):
        """Test create with valid data."""
        request = QueryMemoryRequest(
            app_id="app1234567890", query="What did we discuss about Python?"
        )
        assert request.app_id == "app1234567890"
        assert request.query == "What did we discuss about Python?"

    def test_validate_app_id_min_length(self):
        """Test validate app_id (min_length=10)."""
        with pytest.raises(ValidationError):
            QueryMemoryRequest(app_id="short", query="test query here")

    def test_validate_query_min_length(self):
        """Test validate query (min_length=10)."""
        with pytest.raises(ValidationError):
            QueryMemoryRequest(app_id="app1234567890", query="short")

    def test_validate_n_results_range(self):
        """Test validate n_results (ge=1, le=100)."""
        with pytest.raises(ValidationError):
            QueryMemoryRequest(
                app_id="app1234567890", query="test query here", n_results=0
            )

        with pytest.raises(ValidationError):
            QueryMemoryRequest(
                app_id="app1234567890", query="test query here", n_results=101
            )

    def test_validate_similarity_threshold_range(self):
        """Test validate similarity_threshold (ge=0.0, le=1.0)."""
        with pytest.raises(ValidationError):
            QueryMemoryRequest(
                app_id="app1234567890",
                query="test query here",
                similarity_threshold=-0.1,
            )

        with pytest.raises(ValidationError):
            QueryMemoryRequest(
                app_id="app1234567890",
                query="test query here",
                similarity_threshold=1.1,
            )

    def test_handle_optional_user_id(self):
        """Test handle optional user_id."""
        request = QueryMemoryRequest(
            app_id="app1234567890", query="test query here", user_id="user1234567890"
        )
        assert request.user_id == "user1234567890"

    def test_handle_optional_session_id(self):
        """Test handle optional session_id."""
        request = QueryMemoryRequest(
            app_id="app1234567890",
            query="test query here",
            session_id="session1234567890",
        )
        assert request.session_id == "session1234567890"


class TestAgentMemoryRequest:
    """Test cases for AgentMemoryRequest schema."""

    def test_create_with_string_messages(self):
        """Test create with string messages."""
        request = AgentMemoryRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            messages="Agent conversation text",
        )
        assert isinstance(request.messages, str)

    def test_create_with_list_messages(self):
        """Test create with list of message dicts."""
        messages = [{"role": "user", "content": "Hello"}]
        request = AgentMemoryRequest(
            app_id="app1234567890", user_id="user1234567890", messages=messages
        )
        assert isinstance(request.messages, list)

    def test_validate_app_id_min_length(self):
        """Test validate app_id (min_length=10)."""
        with pytest.raises(ValidationError):
            AgentMemoryRequest(
                app_id="short", user_id="user1234567890", messages="test"
            )

    def test_handle_optional_session_id(self):
        """Test handle optional session_id."""
        request = AgentMemoryRequest(
            app_id="app1234567890",
            user_id="user1234567890",
            messages="test",
            session_id="session1234567890",
        )
        assert request.session_id == "session1234567890"


class TestResponseSchemas:
    """Test cases for response schemas."""

    def test_task_response(self):
        """Test TaskResponse - validate task_id, status, message."""
        response = TaskResponse(
            task_id="task123", status="completed", message="Task completed successfully"
        )
        assert response.task_id == "task123"
        assert response.status == "completed"
        assert response.message == "Task completed successfully"

    def test_memory_response(self):
        """Test MemoryResponse - validate memory_id, document, metadata."""
        response = MemoryResponse(
            memory_id="mem123", document="Memory content", metadata={"key": "value"}
        )
        assert response.memory_id == "mem123"
        assert response.document == "Memory content"
        assert response.metadata == {"key": "value"}

    def test_memory_list_response(self):
        """Test MemoryListResponse - validate memories list, count."""
        response = MemoryListResponse(
            memories=[{"memory_id": "mem1"}, {"memory_id": "mem2"}], count=2
        )
        assert len(response.memories) == 2
        assert response.count == 2

    def test_conversation_summary_response(self):
        """Test ConversationSummaryResponse - validate all fields."""
        response = ConversationSummaryResponse(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id="session1234567890",
            summary="Summary text",
            key_points="Key points",
            tags=["tag1", "tag2"],
            keywords=["keyword1"],
            semantic_queries=["query1"],
            metadata={"key": "value"},
        )
        assert response.summary == "Summary text"
        assert len(response.tags) == 2
        assert response.generated_at is not None

    def test_agent_memory_response(self):
        """Test AgentMemoryResponse - validate all fields."""
        response = AgentMemoryResponse(
            memory_id="mem123",
            app_id="app1234567890",
            user_id="user1234567890",
            summary="Summary",
        )
        assert response.memory_id == "mem123"
        assert response.summary == "Summary"
        assert response.created_at is not None

    def test_success_response(self):
        """Test SuccessResponse - validate success, message."""
        response = SuccessResponse(success=True, message="Operation successful")
        assert response.success is True
        assert response.message == "Operation successful"

    def test_error_response(self):
        """Test ErrorResponse - validate error_code, error_message."""
        response = ErrorResponse(
            error="Error occurred", error_code="ERROR_CODE", details={"field": "value"}
        )
        assert response.error == "Error occurred"
        assert response.error_code == "ERROR_CODE"
        assert response.details == {"field": "value"}

    def test_memory_id_response(self):
        """Test MemoryIDResponse - validate all fields."""
        response = MemoryIDResponse(
            memory_id="mem123",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            status="active",
            preview="Preview text",
        )
        assert response.memory_id == "mem123"
        assert response.status == "active"

    def test_memory_id_list_response(self):
        """Test MemoryIDListResponse - validate all fields."""
        memory_ids = [
            MemoryIDResponse(
                memory_id="mem1",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                status="active",
                preview="Preview 1",
            )
        ]
        response = MemoryIDListResponse(memory_ids=memory_ids, count=1)
        assert len(response.memory_ids) == 1
        assert response.count == 1
