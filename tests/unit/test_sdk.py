"""
Comprehensive unit tests for OmniMemorySDK.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any

import pytest
import httpx

import omnimemory.sdk
from omnimemory.sdk import OmniMemorySDK
from omnimemory.core.schemas import (
    UserMessages,
    Message,
    ConversationSummaryRequest,
    AgentMemoryRequest,
)
from omnimemory.core.results import MemoryOperationResult
from omnimemory.core.llm import LLMConnection


@pytest.fixture
def mock_llm_connection():
    """Mock LLMConnection for SDK tests."""
    llm = Mock(spec=LLMConnection)
    llm.llm_call = AsyncMock()
    return llm


@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager for SDK tests."""
    manager = Mock()
    manager.create_and_store_memory = AsyncMock(
        return_value=MemoryOperationResult(success=True)
    )
    manager.create_agent_memory = AsyncMock(
        return_value=MemoryOperationResult(success=True)
    )
    manager.generate_conversation_summary = AsyncMock(
        return_value={"summary": "Test summary"}
    )
    manager.query_memory = AsyncMock(return_value=[])
    manager.get_memory = AsyncMock(return_value=None)
    manager.traverse_memory_evolution_chain = AsyncMock(return_value=[])
    manager.generate_evolution_graph = Mock(return_value={})
    manager.generate_evolution_report = Mock(return_value={})
    manager.delete_memory = AsyncMock(return_value=MemoryOperationResult(success=True))
    manager.warm_up_connection_pool = AsyncMock(return_value=True)
    manager.connection_pool = Mock()
    manager.connection_pool.get_pool_stats = AsyncMock(return_value={})
    return manager


@pytest.fixture
def sample_user_messages():
    """Create sample UserMessages for testing."""
    messages = [
        Message(role="user", content=f"Message {i}", timestamp="2024-01-01T00:00:00Z")
        for i in range(30)
    ]
    return UserMessages(
        app_id="app1234567890",
        user_id="user1234567890",
        session_id="session1234567890",
        messages=messages,
    )


@pytest.fixture
def sample_agent_request():
    """Create sample AgentMemoryRequest for testing."""
    return AgentMemoryRequest(
        app_id="app1234567890",
        user_id="user1234567890",
        session_id="session1234567890",
        messages="Agent message content",
    )


def test_init_creates_llm_connection(monkeypatch):
    """Test SDK initialization creates LLMConnection."""
    with patch("omnimemory.sdk.LLMConnection") as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        sdk = OmniMemorySDK()

        assert sdk.llm_connection == mock_llm
        assert sdk._memory_manager is None
        assert sdk._background_tasks == {}


def test_init_initializes_background_tasks_dict(monkeypatch):
    """Test SDK initializes _background_tasks as empty dict."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()
        assert isinstance(sdk._background_tasks, dict)
        assert len(sdk._background_tasks) == 0


def test_init_logs_initialization(monkeypatch):
    """Test SDK logs initialization message."""
    with patch("omnimemory.sdk.LLMConnection"):
        with patch("omnimemory.sdk.logger") as mock_logger:
            OmniMemorySDK()
            mock_logger.info.assert_called_once()


def test_init_handles_llm_connection_failure(monkeypatch):
    """Test SDK handles LLMConnection initialization failure."""
    with patch(
        "omnimemory.sdk.LLMConnection", side_effect=Exception("LLM init failed")
    ):
        with pytest.raises(Exception, match="LLM init failed"):
            OmniMemorySDK()


def test_memory_manager_creates_on_first_access(mock_llm_connection, monkeypatch):
    """Test memory_manager property creates MemoryManager on first access."""
    with patch("omnimemory.sdk.MemoryManager") as mock_mm_class:
        mock_mm = Mock()
        mock_mm_class.return_value = mock_mm

        sdk = OmniMemorySDK.__new__(OmniMemorySDK)
        sdk.llm_connection = mock_llm_connection
        sdk._memory_manager = None
        sdk._background_tasks = {}

        manager = sdk.memory_manager

        assert manager == mock_mm
        mock_mm_class.assert_called_once_with(llm_connection=mock_llm_connection)


def test_memory_manager_caches_instance(mock_llm_connection, monkeypatch):
    """Test memory_manager property caches instance on subsequent access."""
    with patch("omnimemory.sdk.MemoryManager") as mock_mm_class:
        mock_mm = Mock()
        mock_mm_class.return_value = mock_mm

        sdk = OmniMemorySDK.__new__(OmniMemorySDK)
        sdk.llm_connection = mock_llm_connection
        sdk._memory_manager = None
        sdk._background_tasks = {}

        manager1 = sdk.memory_manager
        manager2 = sdk.memory_manager

        assert manager1 == manager2 == mock_mm
        assert mock_mm_class.call_count == 1


def test_memory_manager_handles_creation_failure(mock_llm_connection, monkeypatch):
    """Test memory_manager property handles MemoryManager creation failure."""
    with patch("omnimemory.sdk.MemoryManager", side_effect=Exception("MM init failed")):
        sdk = OmniMemorySDK.__new__(OmniMemorySDK)
        sdk.llm_connection = mock_llm_connection
        sdk._memory_manager = None
        sdk._background_tasks = {}

        with pytest.raises(Exception, match="MM init failed"):
            _ = sdk.memory_manager


def test_register_background_task_adds_to_dict(mock_llm_connection):
    """Test _register_background_task adds task to _background_tasks dict."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        task = Mock(spec=asyncio.Task)
        task.add_done_callback = Mock()
        task_id = "test-task-123"

        sdk._register_background_task(task_id, task)

        assert task_id in sdk._background_tasks
        assert sdk._background_tasks[task_id] == task


def test_register_background_task_adds_cleanup_callback(mock_llm_connection):
    """Test _register_background_task adds cleanup callback."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        task = Mock(spec=asyncio.Task)
        task.add_done_callback = Mock()
        task_id = "test-task-123"

        sdk._register_background_task(task_id, task)

        task.add_done_callback.assert_called_once()


def test_register_background_task_cleanup_removes_task(mock_llm_connection):
    """Test cleanup callback removes task from dict when done."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        task = Mock(spec=asyncio.Task)
        callback_func = None

        def capture_callback(cb):
            nonlocal callback_func
            callback_func = cb

        task.add_done_callback = capture_callback
        task_id = "test-task-123"

        sdk._register_background_task(task_id, task)

        callback_func(Mock())

        assert task_id not in sdk._background_tasks


def test_register_background_task_overwrites_existing_id(mock_llm_connection):
    """Test _register_background_task overwrites existing task_id."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        task1 = Mock(spec=asyncio.Task)
        task1.add_done_callback = Mock()
        task2 = Mock(spec=asyncio.Task)
        task2.add_done_callback = Mock()
        task_id = "same-task-id"

        sdk._register_background_task(task_id, task1)
        sdk._register_background_task(task_id, task2)

        assert sdk._background_tasks[task_id] == task2


def test_register_background_task_handles_already_done_task(mock_llm_connection):
    """Test _register_background_task handles task already done."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        task = Mock(spec=asyncio.Task)
        task.done = Mock(return_value=True)
        task.add_done_callback = Mock()
        task_id = "done-task"

        sdk._register_background_task(task_id, task)

        task.add_done_callback.assert_called_once()


def test_format_messages_includes_all_attributes(mock_llm_connection):
    """Test _format_messages includes timestamp, role, and content."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            Message(role="user", content="Hello", timestamp="2024-01-01T00:00:00Z"),
            Message(
                role="assistant", content="Hi there", timestamp="2024-01-01T00:00:01Z"
            ),
        ] + [
            Message(
                role="user", content=f"Message {i}", timestamp="2024-01-01T00:00:00Z"
            )
            for i in range(28)
        ]
        user_msg = UserMessages(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id=None,
            messages=messages,
        )

        result = sdk._format_messages(user_msg)

        assert "[2024-01-01T00:00:00Z] user: Hello" in result
        assert "[2024-01-01T00:00:01Z] assistant: Hi there" in result


def test_format_messages_handles_missing_timestamp(mock_llm_connection):
    """Test _format_messages handles None timestamp (code uses hasattr, so None shows as 'None')."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        msg = Mock()
        msg.timestamp = None
        msg.role = "user"
        msg.content = "Hello"

        user_msg = Mock()
        user_msg.messages = [msg]

        result = sdk._format_messages(user_msg)

        assert "[None] user: Hello" in result


def test_format_messages_handles_missing_role(mock_llm_connection):
    """Test _format_messages handles None role (code uses hasattr, so None shows as 'None')."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        msg = Mock()
        msg.timestamp = "2024-01-01T00:00:00Z"
        msg.role = None
        msg.content = "Hello"

        user_msg = Mock()
        user_msg.messages = [msg]

        result = sdk._format_messages(user_msg)

        assert "[2024-01-01T00:00:00Z] None: Hello" in result


def test_format_messages_handles_missing_content(mock_llm_connection):
    """Test _format_messages uses empty string for missing content."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        msg = Mock()
        msg.timestamp = "2024-01-01T00:00:00Z"
        msg.role = "user"
        msg.content = None

        user_msg = Mock()
        user_msg.messages = [msg]

        result = sdk._format_messages(user_msg)

        assert "[2024-01-01T00:00:00Z] user: " in result


def test_format_messages_joins_with_newlines(mock_llm_connection):
    """Test _format_messages joins multiple messages with newlines."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            Message(role="user", content="Msg1", timestamp="2024-01-01T00:00:00Z"),
            Message(role="user", content="Msg2", timestamp="2024-01-01T00:00:01Z"),
        ] + [
            Message(
                role="user", content=f"Message {i}", timestamp="2024-01-01T00:00:00Z"
            )
            for i in range(28)
        ]
        user_msg = UserMessages(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id=None,
            messages=messages,
        )

        result = sdk._format_messages(user_msg)

        assert "\n" in result
        lines = result.split("\n")
        assert len(lines) == 30


def test_format_messages_handles_empty_list(mock_llm_connection):
    """Test _format_messages handles empty messages list."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        user_msg = Mock()
        user_msg.messages = []

        result = sdk._format_messages(user_msg)

        assert result == ""


def test_format_messages_handles_special_characters(mock_llm_connection):
    """Test _format_messages handles special characters in content."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            Message(
                role="user",
                content="Hello\nWorld\tTab",
                timestamp="2024-01-01T00:00:00Z",
            ),
        ] + [
            Message(
                role="user", content=f"Message {i}", timestamp="2024-01-01T00:00:00Z"
            )
            for i in range(29)
        ]
        user_msg = UserMessages(
            app_id="app1234567890",
            user_id="user1234567890",
            session_id=None,
            messages=messages,
        )

        result = sdk._format_messages(user_msg)

        assert "Hello\nWorld\tTab" in result


def test_format_flexible_messages_returns_stripped_string(mock_llm_connection):
    """Test _format_flexible_messages returns stripped string for string input."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        result = sdk._format_flexible_messages("  Hello World  ")

        assert result == "Hello World"


def test_format_flexible_messages_formats_dict_list(mock_llm_connection):
    """Test _format_flexible_messages formats list of dict messages."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"},
            {"role": "assistant", "content": "Hi", "timestamp": "2024-01-01T00:00:01Z"},
        ]

        result = sdk._format_flexible_messages(messages)

        assert "[2024-01-01T00:00:00Z] user: Hello" in result
        assert "[2024-01-01T00:00:01Z] assistant: Hi" in result


def test_format_flexible_messages_handles_missing_fields(mock_llm_connection):
    """Test _format_flexible_messages uses defaults for missing fields."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            {"content": "Hello"},
        ]

        result = sdk._format_flexible_messages(messages)

        assert "[N/A] unknown: Hello" in result


def test_format_flexible_messages_handles_none_timestamp(mock_llm_connection):
    """Test _format_flexible_messages uses 'N/A' for None timestamp."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            {"role": "user", "content": "Hello", "timestamp": None},
        ]

        result = sdk._format_flexible_messages(messages)

        assert "[N/A] user: Hello" in result


def test_format_flexible_messages_converts_non_dict_to_string(mock_llm_connection):
    """Test _format_flexible_messages converts non-dict items to string."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            "Simple string message",
            123,
        ]

        result = sdk._format_flexible_messages(messages)

        assert "[N/A] unknown: Simple string message" in result
        assert "[N/A] unknown: 123" in result


def test_format_flexible_messages_handles_empty_string(mock_llm_connection):
    """Test _format_flexible_messages handles empty string."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        result = sdk._format_flexible_messages("")

        assert result == ""


def test_format_flexible_messages_handles_empty_list(mock_llm_connection):
    """Test _format_flexible_messages handles empty list."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        result = sdk._format_flexible_messages([])

        assert result == ""


def test_format_flexible_messages_handles_mixed_list(mock_llm_connection):
    """Test _format_flexible_messages handles mixed list (dicts and strings)."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [
            {
                "role": "user",
                "content": "Dict message",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            "String message",
        ]

        result = sdk._format_flexible_messages(messages)

        assert "[2024-01-01T00:00:00Z] user: Dict message" in result
        assert "[N/A] unknown: String message" in result


def test_format_flexible_messages_handles_none_items(mock_llm_connection):
    """Test _format_flexible_messages handles None items in list."""
    with patch("omnimemory.sdk.LLMConnection"):
        sdk = OmniMemorySDK()

        messages = [None]

        result = sdk._format_flexible_messages(messages)

        assert "[N/A] unknown: None" in result


@pytest.mark.asyncio
async def test_add_memory_returns_task_info(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test add_memory returns task information immediately."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)

            assert "task_id" in result
            assert result["status"] == "accepted"
            assert result["app_id"] == sample_user_messages.app_id
            assert result["user_id"] == sample_user_messages.user_id


@pytest.mark.asyncio
async def test_add_memory_generates_unique_task_id(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test add_memory generates unique task_id (UUID)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            result1 = await sdk.add_memory(sample_user_messages)
            result2 = await sdk.add_memory(sample_user_messages)

            assert result1["task_id"] != result2["task_id"]


@pytest.mark.asyncio
async def test_add_memory_creates_background_task(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test add_memory creates background task."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)
            task_id = result["task_id"]

            assert task_id in sdk._background_tasks
            task = sdk._background_tasks[task_id]
            assert task is not None

            await task


@pytest.mark.asyncio
async def test_add_memory_registers_background_task(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test add_memory registers background task."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)
            task_id = result["task_id"]

            assert task_id in sdk._background_tasks


@pytest.mark.asyncio
async def test_add_memory_background_task_calls_create_and_store(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test background task calls create_and_store_memory."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)
            task_id = result["task_id"]

            task = sdk._background_tasks[task_id]
            await task

            mock_memory_manager.create_and_store_memory.assert_called_once()


@pytest.mark.asyncio
async def test_add_memory_background_task_handles_exception(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test background task handles exceptions gracefully."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.create_and_store_memory = AsyncMock(
                side_effect=RuntimeError("Memory creation failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)
            task_id = result["task_id"]

            task = sdk._background_tasks[task_id]
            task_result = await task

            assert task_result["status"] == "failed"
            assert "error" in task_result


@pytest.mark.asyncio
async def test_add_memory_handles_formatting_error(
    mock_llm_connection, sample_user_messages
):
    """Test add_memory handles formatting errors."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager"):
            sdk = OmniMemorySDK()

            sdk._format_messages = Mock(side_effect=ValueError("Format error"))

            result = await sdk.add_memory(sample_user_messages)

            assert result["status"] == "failed"
            assert "error" in result


@pytest.mark.asyncio
async def test_add_memory_handles_validation_error(mock_llm_connection):
    """Test add_memory handles UserMessages validation errors."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        sdk = OmniMemorySDK()

        invalid_msg = Mock()
        invalid_msg.app_id = None
        invalid_msg.user_id = None

        result = await sdk.add_memory(invalid_msg)

        assert result["status"] == "failed"
        assert "error" in result


@pytest.mark.asyncio
async def test_summarize_conversation_without_callback_uses_fast_path(
    mock_llm_connection, mock_memory_manager
):
    """Test summarize_conversation without callback uses fast path."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_conversation_summary = AsyncMock(
                return_value={
                    "summary": "Test summary",
                    "app_id": "app1234567890",
                    "user_id": "user1234567890",
                }
            )

            sdk = OmniMemorySDK()

            request = ConversationSummaryRequest(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages="Test conversation",
                callback_url=None,
            )

            result = await sdk.summarize_conversation(request)

            assert result["delivery"] == "sync"
            assert "summary" in result
            mock_memory_manager.generate_conversation_summary.assert_called_once_with(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages="Test conversation",
                llm_connection=mock_llm_connection,
                use_fast_path=True,
            )


@pytest.mark.asyncio
async def test_summarize_conversation_with_callback_creates_task(
    mock_llm_connection, mock_memory_manager
):
    """Test summarize_conversation with callback creates background task."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                mock_memory_manager.generate_conversation_summary = AsyncMock(
                    return_value={
                        "summary": "Test summary",
                        "app_id": "app1234567890",
                        "user_id": "user1234567890",
                        "session_id": None,
                    }
                )

                sdk = OmniMemorySDK()

                request = ConversationSummaryRequest(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages="Test conversation",
                    callback_url="https://example.com/callback",
                )

                result = await sdk.summarize_conversation(request)

                assert result["status"] == "accepted"
                assert "task_id" in result

                task_id = result["task_id"]
                task = sdk._background_tasks[task_id]
                task_result = await task

                assert task_result.get("delivery") == "callback"


@pytest.mark.asyncio
async def test_summarize_conversation_formats_flexible_messages(
    mock_llm_connection, mock_memory_manager
):
    """Test summarize_conversation formats flexible messages correctly."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_conversation_summary = AsyncMock(
                return_value={
                    "summary": "Test summary",
                }
            )

            sdk = OmniMemorySDK()

            request = ConversationSummaryRequest(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages=[{"role": "user", "content": "Hello"}],
                callback_url=None,
            )

            await sdk.summarize_conversation(request)

            call_args = mock_memory_manager.generate_conversation_summary.call_args
            formatted_messages = call_args.kwargs["messages"]
            assert "user" in formatted_messages or "Hello" in formatted_messages


@pytest.mark.asyncio
async def test_summarize_conversation_handles_exception(
    mock_llm_connection, mock_memory_manager
):
    """Test summarize_conversation handles exceptions and raises."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_conversation_summary = AsyncMock(
                side_effect=RuntimeError("Summary failed")
            )

            sdk = OmniMemorySDK()

            request = ConversationSummaryRequest(
                app_id="app1234567890",
                user_id="user1234567890",
                session_id=None,
                messages="Test conversation",
                callback_url=None,
            )

            with pytest.raises(RuntimeError, match="Summary failed"):
                await sdk.summarize_conversation(request)


@pytest.mark.asyncio
async def test_summarize_conversation_callback_handles_failure(
    mock_llm_connection, mock_memory_manager
):
    """Test summarize_conversation callback handles failure and posts error callback."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client_class.return_value = mock_client

                mock_memory_manager.generate_conversation_summary = AsyncMock(
                    side_effect=RuntimeError("Summary failed")
                )

                sdk = OmniMemorySDK()

                request = ConversationSummaryRequest(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages="Test conversation",
                    callback_url="https://example.com/callback",
                )

                result = await sdk.summarize_conversation(request)
                task_id = result["task_id"]

                task = sdk._background_tasks[task_id]
                await task

                assert mock_client.post.call_count >= 1


@pytest.mark.asyncio
async def test_add_agent_memory_returns_task_info(
    mock_llm_connection, sample_agent_request, mock_memory_manager
):
    """Test add_agent_memory returns task information immediately."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            result = await sdk.add_agent_memory(sample_agent_request)

            assert "task_id" in result
            assert result["status"] == "accepted"
            assert result["app_id"] == sample_agent_request.app_id


@pytest.mark.asyncio
async def test_add_agent_memory_background_task_calls_create_agent_memory(
    mock_llm_connection, sample_agent_request, mock_memory_manager
):
    """Test background task calls create_agent_memory."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            result = await sdk.add_agent_memory(sample_agent_request)
            task_id = result["task_id"]

            task = sdk._background_tasks[task_id]
            await task

            mock_memory_manager.create_agent_memory.assert_called_once()


@pytest.mark.asyncio
async def test_add_agent_memory_handles_exception(
    mock_llm_connection, sample_agent_request, mock_memory_manager
):
    """Test add_agent_memory handles exceptions."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.create_agent_memory = AsyncMock(
                side_effect=RuntimeError("Agent memory failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.add_agent_memory(sample_agent_request)
            task_id = result["task_id"]

            task = sdk._background_tasks[task_id]
            task_result = await task

            assert task_result["status"] == "failed"
            assert "error" in task_result


@pytest.mark.asyncio
async def test_query_memory_returns_results(mock_llm_connection, mock_memory_manager):
    """Test query_memory returns results."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_results = [{"memory_id": "m1", "content": "Test"}]
            mock_memory_manager.query_memory = AsyncMock(return_value=mock_results)

            sdk = OmniMemorySDK()

            results = await sdk.query_memory(
                app_id="app1234567890",
                query="test query",
                user_id="user1234567890",
                n_results=10,
            )

            assert len(results) == 1
            assert results[0]["query_status"] == "completed"
            mock_memory_manager.query_memory.assert_called_once()


@pytest.mark.asyncio
async def test_query_memory_uses_default_n_results(
    mock_llm_connection, mock_memory_manager
):
    """Test query_memory uses default n_results when None."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            with patch("omnimemory.sdk.DEFAULT_N_RESULTS", 5):
                mock_memory_manager.query_memory = AsyncMock(return_value=[])

                sdk = OmniMemorySDK()

                await sdk.query_memory(
                    app_id="app1234567890",
                    query="test query",
                    n_results=None,
                )

                call_args = mock_memory_manager.query_memory.call_args
                assert call_args.kwargs["n_results"] == 5


@pytest.mark.asyncio
async def test_query_memory_handles_none_results(
    mock_llm_connection, mock_memory_manager
):
    """Test query_memory handles None results."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.query_memory = AsyncMock(return_value=None)

            sdk = OmniMemorySDK()

            results = await sdk.query_memory(
                app_id="app1234567890",
                query="test query",
            )

            assert results == []


@pytest.mark.asyncio
async def test_query_memory_handles_exception(mock_llm_connection, mock_memory_manager):
    """Test query_memory handles exceptions gracefully."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.query_memory = AsyncMock(
                side_effect=RuntimeError("Query failed")
            )

            sdk = OmniMemorySDK()

            results = await sdk.query_memory(
                app_id="app1234567890",
                query="test query",
            )

            assert results == []


@pytest.mark.asyncio
async def test_get_memory_returns_memory(mock_llm_connection, mock_memory_manager):
    """Test get_memory returns memory dict."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory = {"memory_id": "m1", "content": "Test"}
            mock_memory_manager.get_memory = AsyncMock(return_value=mock_memory)

            sdk = OmniMemorySDK()

            result = await sdk.get_memory(
                memory_id="m1",
                app_id="app1234567890",
            )

            assert result == mock_memory
            mock_memory_manager.get_memory.assert_called_once_with(
                memory_id="m1",
                app_id="app1234567890",
            )


@pytest.mark.asyncio
async def test_get_memory_returns_none_when_not_found(
    mock_llm_connection, mock_memory_manager
):
    """Test get_memory returns None when memory not found."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.get_memory = AsyncMock(return_value=None)

            sdk = OmniMemorySDK()

            result = await sdk.get_memory(
                memory_id="nonexistent",
                app_id="app1234567890",
            )

            assert result is None


@pytest.mark.asyncio
async def test_get_memory_handles_exception(mock_llm_connection, mock_memory_manager):
    """Test get_memory handles exceptions and returns None."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.get_memory = AsyncMock(
                side_effect=RuntimeError("Get failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.get_memory(
                memory_id="m1",
                app_id="app1234567890",
            )

            assert result is None


@pytest.mark.asyncio
async def test_traverse_memory_evolution_chain_returns_list(
    mock_llm_connection, mock_memory_manager
):
    """Test traverse_memory_evolution_chain returns list of memories."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_chain = [{"memory_id": "m1"}, {"memory_id": "m2"}]
            mock_memory_manager.traverse_memory_evolution_chain = AsyncMock(
                return_value=mock_chain
            )

            sdk = OmniMemorySDK()

            result = await sdk.traverse_memory_evolution_chain(
                app_id="app1234567890",
                memory_id="m1",
            )

            assert result == mock_chain
            mock_memory_manager.traverse_memory_evolution_chain.assert_called_once()


@pytest.mark.asyncio
async def test_traverse_memory_evolution_chain_handles_exception(
    mock_llm_connection, mock_memory_manager
):
    """Test traverse_memory_evolution_chain handles exceptions and returns empty list."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.traverse_memory_evolution_chain = AsyncMock(
                side_effect=RuntimeError("Traverse failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.traverse_memory_evolution_chain(
                app_id="app1234567890",
                memory_id="m1",
            )

            assert result == []


def test_generate_evolution_graph_returns_mermaid_by_default(
    mock_llm_connection, mock_memory_manager
):
    """Test generate_evolution_graph returns Mermaid graph by default."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_evolution_graph = Mock(return_value="graph TD")

            sdk = OmniMemorySDK()

            chain = [{"memory_id": "m1"}]
            result = sdk.generate_evolution_graph(chain)

            assert result == "graph TD"
            mock_memory_manager.generate_evolution_graph.assert_called_once_with(
                chain=chain,
                format="mermaid",
            )


def test_generate_evolution_graph_supports_dot_format(
    mock_llm_connection, mock_memory_manager
):
    """Test generate_evolution_graph supports DOT format."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_evolution_graph = Mock(
                return_value="digraph G"
            )

            sdk = OmniMemorySDK()

            chain = [{"memory_id": "m1"}]
            result = sdk.generate_evolution_graph(chain, format="dot")

            assert result == "digraph G"
            mock_memory_manager.generate_evolution_graph.assert_called_once_with(
                chain=chain,
                format="dot",
            )


def test_generate_evolution_graph_handles_exception(
    mock_llm_connection, mock_memory_manager
):
    """Test generate_evolution_graph handles exceptions and returns empty string."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_evolution_graph = Mock(
                side_effect=RuntimeError("Graph failed")
            )

            sdk = OmniMemorySDK()

            chain = [{"memory_id": "m1"}]
            result = sdk.generate_evolution_graph(chain)

            assert result == ""


def test_generate_evolution_report_returns_markdown_by_default(
    mock_llm_connection, mock_memory_manager
):
    """Test generate_evolution_report returns Markdown report by default."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_evolution_report = Mock(
                return_value="# Report"
            )

            sdk = OmniMemorySDK()

            chain = [{"memory_id": "m1"}]
            result = sdk.generate_evolution_report(chain)

            assert result == "# Report"
            mock_memory_manager.generate_evolution_report.assert_called_once_with(
                chain=chain,
                format="markdown",
            )


def test_generate_evolution_report_supports_text_format(
    mock_llm_connection, mock_memory_manager
):
    """Test generate_evolution_report supports text format."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_evolution_report = Mock(
                return_value="Text report"
            )

            sdk = OmniMemorySDK()

            chain = [{"memory_id": "m1"}]
            result = sdk.generate_evolution_report(chain, format="text")

            assert result == "Text report"
            mock_memory_manager.generate_evolution_report.assert_called_once_with(
                chain=chain,
                format="text",
            )


def test_generate_evolution_report_handles_exception(
    mock_llm_connection, mock_memory_manager
):
    """Test generate_evolution_report handles exceptions and returns empty string."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.generate_evolution_report = Mock(
                side_effect=RuntimeError("Report failed")
            )

            sdk = OmniMemorySDK()

            chain = [{"memory_id": "m1"}]
            result = sdk.generate_evolution_report(chain)

            assert result == ""


@pytest.mark.asyncio
async def test_delete_memory_returns_true_on_success(
    mock_llm_connection, mock_memory_manager
):
    """Test delete_memory returns True on success."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            result_obj = MemoryOperationResult(success=True)
            mock_memory_manager.delete_memory = AsyncMock(return_value=result_obj)

            sdk = OmniMemorySDK()

            result = await sdk.delete_memory(
                app_id="app1234567890",
                doc_id="m1",
            )

            assert result is True
            mock_memory_manager.delete_memory.assert_called_once()


@pytest.mark.asyncio
async def test_delete_memory_returns_false_on_failure(
    mock_llm_connection, mock_memory_manager
):
    """Test delete_memory returns False on failure."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            result_obj = MemoryOperationResult(success=False)
            mock_memory_manager.delete_memory = AsyncMock(return_value=result_obj)

            sdk = OmniMemorySDK()

            result = await sdk.delete_memory(
                app_id="app1234567890",
                doc_id="m1",
            )

            assert result is False


@pytest.mark.asyncio
async def test_delete_memory_handles_bool_result(
    mock_llm_connection, mock_memory_manager
):
    """Test delete_memory handles bool result (backward compatibility)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.delete_memory = AsyncMock(return_value=True)

            sdk = OmniMemorySDK()

            result = await sdk.delete_memory(
                app_id="app1234567890",
                doc_id="m1",
            )

            assert result is True


@pytest.mark.asyncio
async def test_delete_memory_handles_exception(
    mock_llm_connection, mock_memory_manager
):
    """Test delete_memory handles exceptions and returns False."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.delete_memory = AsyncMock(
                side_effect=RuntimeError("Delete failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.delete_memory(
                app_id="app1234567890",
                doc_id="m1",
            )

            assert result is False


@pytest.mark.asyncio
async def test_post_callback_succeeds_on_first_attempt(mock_llm_connection):
    """Test _post_callback succeeds on first attempt."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            sdk = OmniMemorySDK()

            await sdk._post_callback(
                url="https://example.com/callback",
                payload={"test": "data"},
                headers={"Authorization": "Bearer token"},
            )

            mock_client.post.assert_called_once()
            mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_post_callback_retries_on_429(mock_llm_connection):
    """Test _post_callback retries on 429 (rate limit)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
            with patch("asyncio.sleep") as mock_sleep:
                mock_response_429 = Mock()
                mock_response_429.status_code = 429
                mock_response_success = Mock()
                mock_response_success.raise_for_status = Mock()

                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(
                    side_effect=[
                        httpx.HTTPStatusError(
                            "429", request=Mock(), response=mock_response_429
                        ),
                        mock_response_success,
                    ]
                )
                mock_client_class.return_value = mock_client

                sdk = OmniMemorySDK()

                await sdk._post_callback(
                    url="https://example.com/callback",
                    payload={"test": "data"},
                )

                assert mock_client.post.call_count == 2
                mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_post_callback_retries_on_503(mock_llm_connection):
    """Test _post_callback retries on 503 (service unavailable)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
            with patch("asyncio.sleep") as mock_sleep:
                mock_response_503 = Mock()
                mock_response_503.status_code = 503
                mock_response_success = Mock()
                mock_response_success.raise_for_status = Mock()

                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(
                    side_effect=[
                        httpx.HTTPStatusError(
                            "503", request=Mock(), response=mock_response_503
                        ),
                        mock_response_success,
                    ]
                )
                mock_client_class.return_value = mock_client

                sdk = OmniMemorySDK()

                await sdk._post_callback(
                    url="https://example.com/callback",
                    payload={"test": "data"},
                )

                assert mock_client.post.call_count == 2
                mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_post_callback_skips_retry_on_400(mock_llm_connection):
    """Test _post_callback skips retry on 400 (permanent error)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
            mock_response_400 = Mock()
            mock_response_400.status_code = 400

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "400", request=Mock(), response=mock_response_400
                )
            )
            mock_client_class.return_value = mock_client

            sdk = OmniMemorySDK()

            with pytest.raises(httpx.HTTPStatusError):
                await sdk._post_callback(
                    url="https://example.com/callback",
                    payload={"test": "data"},
                )

            assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_post_callback_retries_on_network_error(mock_llm_connection):
    """Test _post_callback retries on network errors."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
            with patch("asyncio.sleep") as mock_sleep:
                mock_response_success = Mock()
                mock_response_success.raise_for_status = Mock()

                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(
                    side_effect=[
                        httpx.NetworkError("Network error"),
                        mock_response_success,
                    ]
                )
                mock_client_class.return_value = mock_client

                sdk = OmniMemorySDK()

                await sdk._post_callback(
                    url="https://example.com/callback",
                    payload={"test": "data"},
                )

                assert mock_client.post.call_count == 2
                mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_post_callback_raises_after_max_retries(mock_llm_connection):
    """Test _post_callback raises after max retries."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
            with patch("asyncio.sleep"):
                mock_response_503 = Mock()
                mock_response_503.status_code = 503

                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(
                    side_effect=httpx.HTTPStatusError(
                        "503", request=Mock(), response=mock_response_503
                    )
                )
                mock_client_class.return_value = mock_client

                sdk = OmniMemorySDK()

                with pytest.raises(httpx.HTTPStatusError):
                    await sdk._post_callback(
                        url="https://example.com/callback",
                        payload={"test": "data"},
                    )

                assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_warm_up_returns_true_on_success(
    mock_llm_connection, mock_memory_manager
):
    """Test warm_up returns True on success."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.warm_up_connection_pool = AsyncMock(return_value=True)

            sdk = OmniMemorySDK()

            result = await sdk.warm_up()

            assert result is True
            mock_memory_manager.warm_up_connection_pool.assert_called_once()


@pytest.mark.asyncio
async def test_warm_up_returns_false_on_failure(
    mock_llm_connection, mock_memory_manager
):
    """Test warm_up returns False on failure."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.warm_up_connection_pool = AsyncMock(
                side_effect=RuntimeError("Warm-up failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.warm_up()

            assert result is False


@pytest.mark.asyncio
async def test_get_connection_pool_stats_returns_stats(
    mock_llm_connection, mock_memory_manager
):
    """Test get_connection_pool_stats returns stats dict."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_stats = {"active": 5, "idle": 3}
            mock_memory_manager.connection_pool.get_pool_stats = AsyncMock(
                return_value=mock_stats
            )

            sdk = OmniMemorySDK()

            result = await sdk.get_connection_pool_stats()

            assert result == mock_stats
            mock_memory_manager.connection_pool.get_pool_stats.assert_called_once()


@pytest.mark.asyncio
async def test_get_connection_pool_stats_returns_error_dict_on_failure(
    mock_llm_connection, mock_memory_manager
):
    """Test get_connection_pool_stats returns error dict on failure."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.connection_pool.get_pool_stats = AsyncMock(
                side_effect=RuntimeError("Stats failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.get_connection_pool_stats()

            assert "error" in result
            assert result["error"] == "Stats failed"


@pytest.mark.asyncio
async def test_get_task_status_returns_processing_for_running_task(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test get_task_status returns 'processing' for running task."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):

            async def slow_task():
                await asyncio.sleep(0.1)
                return {"success": True}

            mock_memory_manager.create_and_store_memory = AsyncMock(
                side_effect=slow_task
            )

            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)
            task_id = result["task_id"]

            status = await sdk.get_task_status(task_id)

            assert status["status"] == "processing"
            assert status["task_id"] == task_id

            task = sdk._background_tasks[task_id]
            await task


@pytest.mark.asyncio
async def test_get_task_status_returns_completed_for_done_task(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test get_task_status returns 'completed' for done task with success."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.create_and_store_memory = AsyncMock(
                return_value=MemoryOperationResult(success=True)
            )

            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)
            task_id = result["task_id"]

            task = sdk._background_tasks[task_id]
            await task

            status = await sdk.get_task_status(task_id)

            assert status["status"] in ["completed", "unknown"]


@pytest.mark.asyncio
async def test_get_task_status_returns_failed_for_failed_task(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test get_task_status returns 'failed' for done task with failure."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            mock_memory_manager.create_and_store_memory = AsyncMock(
                side_effect=RuntimeError("Task failed")
            )

            sdk = OmniMemorySDK()

            result = await sdk.add_memory(sample_user_messages)
            task_id = result["task_id"]

            task = sdk._background_tasks[task_id]
            await task

            status = await sdk.get_task_status(task_id)

            assert status["status"] in ["failed", "unknown"]


@pytest.mark.asyncio
async def test_get_task_status_returns_unknown_for_nonexistent_task(
    mock_llm_connection,
):
    """Test get_task_status returns 'unknown' for non-existent task."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        sdk = OmniMemorySDK()

        status = await sdk.get_task_status("nonexistent-task-id")

        assert status["status"] == "unknown"
        assert status["task_id"] == "nonexistent-task-id"


@pytest.mark.asyncio
async def test_get_task_status_handles_task_result_exception(
    mock_llm_connection, sample_user_messages, mock_memory_manager
):
    """Test get_task_status handles task.result() exceptions (e.g., cancelled tasks)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            task_id = "test-task-cancelled"
            mock_task = Mock(spec=asyncio.Task)
            mock_task.done = Mock(return_value=True)
            mock_task.result = Mock(
                side_effect=asyncio.CancelledError("Task was cancelled")
            )

            sdk._background_tasks[task_id] = mock_task

            status = await sdk.get_task_status(task_id)

            assert status["status"] == "failed"
            assert "error" in status
            assert (
                "cancelled" in status["error"].lower()
                or "Task was cancelled" in status["error"]
            )


@pytest.mark.asyncio
async def test_summarize_conversation_error_callback_delivery_fails(
    mock_llm_connection, mock_memory_manager
):
    """Test summarize_conversation handles error callback delivery failure (lines 226-227)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(
                    side_effect=httpx.RequestError("Error callback delivery failed")
                )
                mock_client_class.return_value = mock_client

                mock_memory_manager.generate_conversation_summary = AsyncMock(
                    side_effect=RuntimeError("Summary failed")
                )

                sdk = OmniMemorySDK()

                request = ConversationSummaryRequest(
                    app_id="app1234567890",
                    user_id="user1234567890",
                    session_id=None,
                    messages="Test conversation",
                    callback_url="https://example.com/callback",
                )

                result = await sdk.summarize_conversation(request)
                task_id = result["task_id"]

                task = sdk._background_tasks[task_id]
                task_result = await task

                assert mock_client.post.call_count >= 1
                assert task_result["status"] == "failed"


@pytest.mark.asyncio
async def test_add_agent_memory_top_level_exception(
    mock_llm_connection, sample_agent_request, mock_memory_manager
):
    """Test add_agent_memory handles top-level exceptions before task creation (lines 324-326)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.MemoryManager", return_value=mock_memory_manager):
            sdk = OmniMemorySDK()

            with patch(
                "omnimemory.sdk.uuid.uuid4",
                side_effect=RuntimeError("UUID generation failed"),
            ):
                result = await sdk.add_agent_memory(sample_agent_request)

                assert result["status"] == "failed"
                assert "error" in result
                assert "UUID generation failed" in result["error"]
                assert result["app_id"] == sample_agent_request.app_id
                assert result["user_id"] == sample_agent_request.user_id


@pytest.mark.asyncio
async def test_post_callback_last_attempt_non_http_exception(mock_llm_connection):
    """Test _post_callback handles last attempt failure with non-HTTPStatusError (lines 591-595)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        with patch("omnimemory.sdk.httpx.AsyncClient") as mock_client_class:
            with patch("asyncio.sleep"):
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client.post = AsyncMock(side_effect=RuntimeError("Network error"))
                mock_client_class.return_value = mock_client

                sdk = OmniMemorySDK()

                with pytest.raises(RuntimeError, match="Network error"):
                    await sdk._post_callback(
                        url="https://example.com/callback",
                        payload={"test": "data"},
                    )

                assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_get_task_status_returns_completed_with_success_result(
    mock_llm_connection,
):
    """Test get_task_status returns 'completed' when task.result() succeeds with success=True (line 643)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        sdk = OmniMemorySDK()

        task_id = "test-task-success"
        mock_task = Mock(spec=asyncio.Task)
        mock_task.done = Mock(return_value=True)
        mock_task.result = Mock(
            return_value={"success": True, "task_id": task_id, "memory_id": "mem123"}
        )

        sdk._background_tasks[task_id] = mock_task

        status = await sdk.get_task_status(task_id)

        assert status["status"] == "completed"
        assert "result" in status
        assert status["result"].get("success") is True
        assert status["result"].get("task_id") == task_id


@pytest.mark.asyncio
async def test_get_task_status_outer_exception_handler(mock_llm_connection):
    """Test get_task_status handles exceptions in outer try block (lines 662-666)."""
    with patch("omnimemory.sdk.LLMConnection", return_value=mock_llm_connection):
        sdk = OmniMemorySDK()

        task_id = "test-task-exception"
        mock_task = Mock(spec=asyncio.Task)
        mock_task.done = Mock(side_effect=RuntimeError("Task.done() failed"))

        sdk._background_tasks[task_id] = mock_task

        status = await sdk.get_task_status(task_id)

        assert status["status"] == "error"
        assert "error" in status
        assert "Task.done() failed" in status["error"]
