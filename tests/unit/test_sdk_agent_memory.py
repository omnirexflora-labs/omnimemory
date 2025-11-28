import pytest
from unittest.mock import AsyncMock, MagicMock

from omnimemory.sdk_agent_memory import AgentMemorySDK


@pytest.mark.asyncio
async def test_answer_query_returns_llm_answer(monkeypatch):
    mock_memory_manager = MagicMock()
    mock_memory_manager.query_memory = AsyncMock(
        return_value=[{"memory_note": "User met Alice yesterday."}]
    )

    mock_llm = MagicMock()
    mock_llm.llm_call = AsyncMock(
        return_value={"choices": [{"message": {"content": "Alice said hi."}}]}
    )

    monkeypatch.setattr(
        "omnimemory.sdk_agent_memory.MemoryManager",
        lambda llm_connection: mock_memory_manager,
    )

    sdk = AgentMemorySDK(llm_connection=mock_llm)

    result = await sdk.answer_query(app_id="app", query="Who did I meet?")

    assert result["answer"] == "Alice said hi."
    mock_memory_manager.query_memory.assert_awaited_once()
    mock_llm.llm_call.assert_awaited_once()
    system_message = mock_llm.llm_call.await_args.kwargs["messages"][0]["content"]
    assert "User met Alice yesterday." in system_message


@pytest.mark.asyncio
async def test_answer_query_handles_empty_results_and_llm(monkeypatch):
    mock_memory_manager = MagicMock()
    mock_memory_manager.query_memory = AsyncMock(return_value=[])

    mock_llm = MagicMock()
    mock_llm.llm_call = AsyncMock(return_value=None)

    monkeypatch.setattr(
        "omnimemory.sdk_agent_memory.MemoryManager",
        lambda llm_connection: mock_memory_manager,
    )

    sdk = AgentMemorySDK(llm_connection=mock_llm)

    result = await sdk.answer_query(app_id="app", query="What's new?")

    assert result["answer"] == "LLM response unavailable."
    mock_llm.llm_call.assert_awaited_once()
    system_message = mock_llm.llm_call.await_args.kwargs["messages"][0]["content"]
    assert "No relevant memories were found" in system_message
