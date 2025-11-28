import json
import uuid
from unittest.mock import AsyncMock, Mock

import pytest

# ruff: noqa: F811

from omnimemory.core.results import MemoryOperationResult
from omnimemory.memory_management.memory_manager import MemoryManager


def _stub_pool(monkeypatch, handler=None):
    """Patch the handler pool + agents so MemoryManager can be constructed safely."""
    if handler is None:
        handler = Mock(enabled=True)

    fake_ctx = AsyncMock()
    fake_ctx.__aenter__.return_value = handler
    fake_ctx.__aexit__.return_value = False

    fake_pool = Mock()
    fake_pool.get_handler.return_value = fake_ctx

    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.VectorDBHandlerPool.get_instance",
        lambda max_connections=None: fake_pool,
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.ConflictResolutionAgent",
        lambda llm: Mock(),
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.SynthesisAgent",
        lambda llm: Mock(),
    )


def _make_manager(monkeypatch, mock_llm_connection):
    _stub_pool(monkeypatch)
    return MemoryManager(mock_llm_connection)


def _stub_metrics(monkeypatch):
    class FakeMetrics:
        def __init__(self):
            self.record_write = Mock()

        def operation_timer(self, *_args, **_kwargs):
            class Timer:
                def __init__(self):
                    self.success = True
                    self.error_code = None
                    self.results_count = None

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type:
                        self.success = False
                        self.error_code = exc_type.__name__
                    return False

            return Timer()

    metrics = FakeMetrics()
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.get_metrics_collector",
        lambda *args, **kwargs: metrics,
    )
    return metrics


@pytest.mark.asyncio
async def test_create_episodic_memory_returns_content(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    mock_llm_connection.llm_call = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(content="episodic-note"))])
    )

    result = await manager.create_episodic_memory("hello", mock_llm_connection)

    assert result == "episodic-note"
    mock_llm_connection.llm_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_episodic_memory_returns_none_for_empty_choices(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    mock_llm_connection.llm_call = AsyncMock(return_value=Mock(choices=[]))

    result = await manager.create_episodic_memory("hello", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_parse_error(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value="invalid json")
    manager.create_summarizer_memory = AsyncMock(return_value="also invalid")

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_combined_memory_returns_none_when_summarizer_missing(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value="{}")
    manager.create_summarizer_memory = AsyncMock(return_value=None)

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_episodic_memory_handles_exception(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    mock_llm_connection.llm_call = AsyncMock(side_effect=RuntimeError("llm down"))

    result = await manager.create_episodic_memory("hello", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_parse_error(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value="invalid json")
    manager.create_summarizer_memory = AsyncMock(return_value="also invalid")

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(side_effect=RuntimeError("error"))

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_summarizer_memory_returns_content(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    mock_llm_connection.llm_call = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(content="summary-note"))])
    )

    result = await manager.create_summarizer_memory("hello", mock_llm_connection)

    assert result == "summary-note"


@pytest.mark.asyncio
async def test_create_summarizer_memory_handles_missing_choices(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    mock_llm_connection.llm_call = AsyncMock(return_value=Mock(choices=None))

    result = await manager.create_summarizer_memory("hello", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_parse_error(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value="invalid json")
    manager.create_summarizer_memory = AsyncMock(return_value="also invalid")

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(side_effect=RuntimeError("error"))

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_generate_conversation_summary_uses_fast_path(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager._generate_fast_summary = AsyncMock(return_value={"fast": True})
    manager._generate_full_summary = AsyncMock()
    _stub_metrics(monkeypatch)

    result = await manager.generate_conversation_summary(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages="hi",
        llm_connection=mock_llm_connection,
        use_fast_path=True,
    )

    assert result == {"fast": True}
    manager._generate_fast_summary.assert_awaited_once()
    manager._generate_full_summary.assert_not_called()


@pytest.mark.asyncio
async def test_generate_conversation_summary_uses_full_path(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager._generate_fast_summary = AsyncMock(return_value={"fast": True})
    manager._generate_full_summary = AsyncMock(return_value={"full": True})
    _stub_metrics(monkeypatch)

    result = await manager.generate_conversation_summary(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages="hi",
        llm_connection=mock_llm_connection,
        use_fast_path=False,
    )

    assert result == {"full": True}
    manager._generate_full_summary.assert_awaited_once()
    manager._generate_fast_summary.assert_not_called()


@pytest.mark.asyncio
async def test_generate_conversation_summary_records_failure(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager._generate_fast_summary = AsyncMock(side_effect=RuntimeError("boom"))
    metrics = _stub_metrics(monkeypatch)

    with pytest.raises(RuntimeError):
        await manager.generate_conversation_summary(
            app_id="app",
            user_id="user",
            session_id="sess",
            messages="hi",
            llm_connection=mock_llm_connection,
            use_fast_path=True,
        )

    metrics.record_write.assert_called()
    kwargs = metrics.record_write.call_args.kwargs
    assert kwargs["success"] is False
    assert kwargs["error_code"] == "RuntimeError"


@pytest.mark.asyncio
async def test_generate_fast_summary_returns_structured_result(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    mock_llm_connection.llm_call = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(content=" summary text "))])
    )

    result = await manager._generate_fast_summary(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages="hello",
        llm_connection=mock_llm_connection,
    )

    assert result["summary"] == "summary text"
    assert result["app_id"] == "app"
    assert result["user_id"] == "user"
    assert result["session_id"] == "sess"


@pytest.mark.asyncio
async def test_generate_fast_summary_raises_when_no_response(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    mock_llm_connection.llm_call = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="Fast summary agent returned no response"):
        await manager._generate_fast_summary(
            app_id="app",
            user_id="user",
            session_id=None,
            messages="hello",
            llm_connection=mock_llm_connection,
        )


@pytest.mark.asyncio
async def test_generate_full_summary_normalizes_data(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_summarizer_memory = AsyncMock(return_value="{}")

    summary_json = {
        "narrative": "Primary summary",
    }

    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.clean_and_parse_json",
        lambda raw: summary_json,
    )

    result = await manager._generate_full_summary(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages="hi",
        llm_connection=mock_llm_connection,
    )

    assert result["summary"] == "Primary summary"
    assert result["app_id"] == "app"
    assert result["user_id"] == "user"
    assert result["session_id"] == "sess"
    assert "generated_at" in result


@pytest.mark.asyncio
async def test_generate_full_summary_raises_on_parse_error(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_summarizer_memory = AsyncMock(return_value="raw")

    def _raise(_raw):
        raise ValueError("bad json")

    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.clean_and_parse_json",
        _raise,
    )

    with pytest.raises(ValueError, match="bad json"):
        await manager._generate_full_summary(
            app_id="app",
            user_id="user",
            session_id="sess",
            messages="hi",
            llm_connection=mock_llm_connection,
        )


@pytest.mark.asyncio
async def test_generate_full_summary_handles_non_dict_response(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_summarizer_memory = AsyncMock(return_value="[]")
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.clean_and_parse_json",
        lambda raw: "not-a-dict",
    )

    with pytest.raises(ValueError, match="Summarizer response is not a dict structure"):
        await manager._generate_full_summary(
            app_id="app",
            user_id="user",
            session_id="sess",
            messages="hi",
            llm_connection=mock_llm_connection,
        )


@pytest.mark.asyncio
async def test_generate_full_summary_handles_list_response(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_summarizer_memory = AsyncMock(return_value="raw")

    payload = [
        {
            "narrative": "primary",
        }
    ]
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.clean_and_parse_json",
        lambda raw: payload,
    )

    result = await manager._generate_full_summary(
        app_id="app",
        user_id="user",
        session_id=None,
        messages="hello",
        llm_connection=mock_llm_connection,
    )

    assert result["summary"] == "primary"


@pytest.mark.asyncio
async def test_generate_full_summary_raises_when_no_content(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_summarizer_memory = AsyncMock(return_value="")

    with pytest.raises(ValueError, match="Summarizer agent returned no content"):
        await manager._generate_full_summary(
            app_id="app",
            user_id="user",
            session_id=None,
            messages="hello",
            llm_connection=mock_llm_connection,
        )


@pytest.mark.asyncio
async def test_create_agent_memory_successful_flow(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    _stub_metrics(monkeypatch)

    # Mock the new agent memory generation method with valid JSON
    agent_memory_json = json.dumps(
        {
            "narrative": "Test narrative",
            "retrieval": {
                "tags": ["test"],
                "keywords": ["test"],
                "queries": ["test query"],
            },
            "metadata": {"depth": "medium", "follow_ups": ["N/A"]},
        }
    )
    manager._generate_add_agent_memory = AsyncMock(return_value=agent_memory_json)
    manager.embed_memory_note = AsyncMock(return_value=[0.1, 0.2])

    success_result = MemoryOperationResult.success_result(memory_id="stored-id")
    manager.store_memory_note = AsyncMock(return_value=success_result)

    fixed_uuid = uuid.UUID(int=1)
    monkeypatch.setattr("uuid.uuid4", lambda: fixed_uuid)

    result = await manager.create_agent_memory(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages="hello",
        llm_connection=mock_llm_connection,
    )

    assert result.success is True
    manager.embed_memory_note.assert_awaited_once()
    manager.store_memory_note.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_agent_memory_handles_embedding_exception(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    metrics = _stub_metrics(monkeypatch)

    agent_memory_json = json.dumps(
        {
            "narrative": "Test narrative",
            "retrieval": {"tags": [], "keywords": [], "queries": []},
            "metadata": {"depth": "low", "follow_ups": []},
        }
    )
    manager._generate_add_agent_memory = AsyncMock(return_value=agent_memory_json)
    manager.embed_memory_note = AsyncMock(side_effect=RuntimeError("embed fail"))

    result = await manager.create_agent_memory(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages="hello",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_EXCEPTION"
    metrics.record_write.assert_called()


@pytest.mark.asyncio
async def test_create_agent_memory_handles_store_failure(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    metrics = _stub_metrics(monkeypatch)

    agent_memory_json = json.dumps(
        {
            "narrative": "Test narrative",
            "retrieval": {"tags": [], "keywords": [], "queries": []},
            "metadata": {"depth": "medium", "follow_ups": []},
        }
    )
    manager._generate_add_agent_memory = AsyncMock(return_value=agent_memory_json)
    manager.embed_memory_note = AsyncMock(return_value=[0.1])

    failure_result = MemoryOperationResult.error_result(
        error_code="STORE_FAILED", error_message="fail"
    )
    manager.store_memory_note = AsyncMock(return_value=failure_result)

    result = await manager.create_agent_memory(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages="hello",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "STORE_FAILED"
    metrics.record_write.assert_called()


@pytest.mark.asyncio
async def test_create_agent_memory_formats_message_list(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    _stub_metrics(monkeypatch)

    agent_memory_json = json.dumps(
        {
            "narrative": "Test note",
            "retrieval": {"tags": [], "keywords": [], "queries": []},
            "metadata": {"depth": "medium", "follow_ups": []},
        }
    )
    agent_memory_mock = AsyncMock(return_value=agent_memory_json)
    manager._generate_add_agent_memory = agent_memory_mock
    manager.embed_memory_note = AsyncMock(return_value=[0.1])
    manager.store_memory_note = AsyncMock(
        return_value=MemoryOperationResult.success_result(memory_id="m1")
    )

    from omnimemory.core.utils import format_conversation

    raw_messages = [
        {"role": "user", "content": "Hi"},
        "Raw follow-up",
    ]
    formatted_messages = format_conversation(raw_messages)

    await manager.create_agent_memory(
        app_id="app",
        user_id="user",
        session_id="sess",
        messages=formatted_messages,
        llm_connection=mock_llm_connection,
    )

    formatted = agent_memory_mock.await_args.kwargs["message"]
    assert formatted == formatted_messages


@pytest.mark.asyncio
async def test_create_agent_memory_records_metrics_on_exception(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    metrics = _stub_metrics(monkeypatch)

    agent_memory_json = json.dumps(
        {
            "narrative": "Test note",
            "retrieval": {"tags": [], "keywords": [], "queries": []},
            "metadata": {"depth": "medium", "follow_ups": []},
        }
    )
    manager._generate_add_agent_memory = AsyncMock(return_value=agent_memory_json)
    manager.embed_memory_note = AsyncMock(return_value=[0.1])
    manager.store_memory_note = AsyncMock(side_effect=RuntimeError("store boom"))

    with pytest.raises(RuntimeError):
        await manager.create_agent_memory(
            app_id="app",
            user_id="user",
            session_id=None,
            messages="hello",
            llm_connection=mock_llm_connection,
        )

    last_call = metrics.record_write.call_args_list[-1]
    assert last_call.kwargs["success"] is False
    assert last_call.kwargs["error_code"] == "RuntimeError"


@pytest.mark.asyncio
async def test_create_combined_memory_returns_merged_note(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)

    manager.create_episodic_memory = AsyncMock(return_value='{"episodic": true}')
    manager.create_summarizer_memory = AsyncMock(return_value='{"summary": true}')

    episodic_data = {"episodic": True}
    summarizer_data = {"summary": True}
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.clean_and_parse_json",
        Mock(side_effect=[episodic_data, summarizer_data]),
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.create_zettelkasten_memory_note",
        lambda episodic_data, summary_data: "zettelkasten-note",
    )
    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.prepare_memory_for_storage",
        lambda **kwargs: {"prepared": True, **kwargs},
    )

    result = await manager.create_combined_memory("hello", mock_llm_connection)

    assert json.loads(result)["prepared"] is True


@pytest.mark.asyncio
async def test_create_combined_memory_returns_none_when_episode_missing(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value=None)
    manager.create_summarizer_memory = AsyncMock(return_value="{}")

    result = await manager.create_combined_memory("hello", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_parse_error(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value="invalid json")
    manager.create_summarizer_memory = AsyncMock(return_value="also invalid")

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(side_effect=RuntimeError("error"))

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_returns_none_on_parse_failure(
    monkeypatch, mock_llm_connection
):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value="{}")
    manager.create_summarizer_memory = AsyncMock(return_value="{}")

    def _raise(_raw):
        raise ValueError("bad json")

    monkeypatch.setattr(
        "omnimemory.memory_management.memory_manager.clean_and_parse_json",
        Mock(side_effect=_raise),
    )

    result = await manager.create_combined_memory("hello", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_combined_memory = AsyncMock(
        return_value='{"text": "test", "metadata": {}}'
    )
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_parse_error(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(return_value="invalid json")
    manager.create_summarizer_memory = AsyncMock(return_value="also invalid")

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None


@pytest.mark.asyncio
async def test_create_agent_memory_embedding_fails(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    agent_memory_json = json.dumps(
        {
            "narrative": "Test",
            "retrieval": {"tags": [], "keywords": [], "queries": []},
            "metadata": {"depth": "medium", "follow_ups": []},
        }
    )
    manager._generate_add_agent_memory = AsyncMock(return_value=agent_memory_json)
    manager.embed_memory_note = AsyncMock(return_value=None)

    result = await manager.create_agent_memory(
        app_id="app1",
        user_id="user1",
        session_id=None,
        messages="test",
        llm_connection=mock_llm_connection,
    )

    assert result.success is False
    assert result.error_code == "EMBEDDING_FAILED"


@pytest.mark.asyncio
async def test_create_combined_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    manager.create_episodic_memory = AsyncMock(side_effect=RuntimeError("error"))

    result = await manager.create_combined_memory("messages", mock_llm_connection)

    assert result is None


@pytest.mark.asyncio
async def test_create_summarizer_memory_exception(monkeypatch, mock_llm_connection):
    manager = _make_manager(monkeypatch, mock_llm_connection)
    failing_llm = Mock()
    failing_llm.llm_call = AsyncMock(side_effect=RuntimeError("llm error"))

    result = await manager.create_summarizer_memory("message", failing_llm)

    assert result is None
