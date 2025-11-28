"""
Comprehensive unit tests for OmniMemory core agents.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from omnimemory.core.agents import ConflictResolutionAgent, SynthesisAgent


class TestConflictResolutionAgent:
    """Test cases for ConflictResolutionAgent."""

    def test_init_with_valid_llm_connection(self):
        """Test initialize with valid llm_connection."""
        mock_llm = Mock()
        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        assert agent.llm_connection == mock_llm

    def test_init_stores_llm_connection(self):
        """Test that llm_connection is stored correctly."""
        mock_llm = Mock()
        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        assert hasattr(agent, "llm_connection")
        assert agent.llm_connection is mock_llm

    @pytest.mark.asyncio
    async def test_decide_success(self):
        """Test make decisions successfully."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "UPDATE",
                    "confidence_score": 0.9,
                    "reasoning": "Test reasoning",
                }
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test memory"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Existing", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"
        assert result[0]["operation"] == "UPDATE"
        assert result[0]["confidence_score"] == 0.9

    @pytest.mark.asyncio
    async def test_decide_returns_list(self):
        """Test that decide returns a list of decisions."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "UPDATE",
                    "confidence_score": 0.8,
                    "reasoning": "Reason 1",
                },
                {
                    "memory_id": "mem2",
                    "operation": "DELETE",
                    "confidence_score": 0.7,
                    "reasoning": "Reason 2",
                },
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc1", "composite_score": 0.8},
            {"memory_id": "mem2", "document": "Doc2", "composite_score": 0.7},
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_decide_required_fields(self):
        """Test each decision has required fields."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "SKIP",
                    "confidence_score": 0.5,
                    "reasoning": "Test",
                }
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.5}
        ]

        result = await agent.decide(new_memory, linked_memories)
        decision = result[0]
        assert "memory_id" in decision
        assert "operation" in decision
        assert "confidence_score" in decision
        assert "reasoning" in decision

    @pytest.mark.asyncio
    async def test_decide_valid_operations(self):
        """Test operations are valid (UPDATE, DELETE, SKIP)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "UPDATE",
                    "confidence_score": 0.8,
                    "reasoning": "R1",
                },
                {
                    "memory_id": "mem2",
                    "operation": "DELETE",
                    "confidence_score": 0.7,
                    "reasoning": "R2",
                },
                {
                    "memory_id": "mem3",
                    "operation": "SKIP",
                    "confidence_score": 0.6,
                    "reasoning": "R3",
                },
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.8},
            {"memory_id": "mem2", "document": "D2", "composite_score": 0.7},
            {"memory_id": "mem3", "document": "D3", "composite_score": 0.6},
        ]

        result = await agent.decide(new_memory, linked_memories)
        operations = [d["operation"] for d in result]
        assert "UPDATE" in operations
        assert "DELETE" in operations
        assert "SKIP" in operations

    @pytest.mark.asyncio
    async def test_decide_confidence_scores_range(self):
        """Test confidence scores are 0.0-1.0."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "UPDATE",
                    "confidence_score": 0.0,
                    "reasoning": "R1",
                },
                {
                    "memory_id": "mem2",
                    "operation": "DELETE",
                    "confidence_score": 1.0,
                    "reasoning": "R2",
                },
                {
                    "memory_id": "mem3",
                    "operation": "SKIP",
                    "confidence_score": 0.5,
                    "reasoning": "R3",
                },
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.0},
            {"memory_id": "mem2", "document": "D2", "composite_score": 1.0},
            {"memory_id": "mem3", "document": "D3", "composite_score": 0.5},
        ]

        result = await agent.decide(new_memory, linked_memories)
        for decision in result:
            assert 0.0 <= decision["confidence_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_decide_no_choices_returns_skip(self):
        """Test handle LLM response with no choices (return SKIP for all)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.choices = []
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.8},
            {"memory_id": "mem2", "document": "D2", "composite_score": 0.7},
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 2
        assert all(d["operation"] == "SKIP" for d in result)
        assert all(d["confidence_score"] == 0.0 for d in result)

    @pytest.mark.asyncio
    async def test_decide_none_response_returns_skip(self):
        """Test handle LLM response with None (return SKIP for all)."""
        mock_llm = AsyncMock()
        mock_llm.llm_call.return_value = None

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert result[0]["confidence_score"] == 0.0

    @pytest.mark.asyncio
    async def test_decide_parse_json_array(self):
        """Test parse JSON array response."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = '[{"memory_id": "mem1", "operation": "UPDATE", "confidence_score": 0.9, "reasoning": "Test"}]'
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_decide_parse_json_object_with_decisions(self):
        """Test parse JSON object with 'decisions' field."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "decisions": [
                    {
                        "memory_id": "mem1",
                        "operation": "UPDATE",
                        "confidence_score": 0.9,
                        "reasoning": "R1",
                    }
                ]
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"

    @pytest.mark.asyncio
    async def test_decide_extract_json_from_markdown(self):
        """Test extract JSON from markdown code blocks."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = """Here's the analysis:

```json
[{"memory_id": "mem1", "operation": "UPDATE", "confidence_score": 0.9, "reasoning": "Test"}]
```

That's the decision."""
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_decide_normalize_invalid_confidence_score(self):
        """Test normalize invalid confidence scores to 0.5."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "UPDATE",
                    "confidence_score": 2.0,
                    "reasoning": "R1",
                },
                {
                    "memory_id": "mem2",
                    "operation": "DELETE",
                    "confidence_score": -1.0,
                    "reasoning": "R2",
                },
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.8},
            {"memory_id": "mem2", "document": "D2", "composite_score": 0.7},
        ]

        with patch("omnimemory.core.agents.logger"):
            result = await agent.decide(new_memory, linked_memories)
            assert result[0]["confidence_score"] == 0.5
            assert result[1]["confidence_score"] == 0.5

    @pytest.mark.asyncio
    async def test_decide_json_parsing_error_returns_skip(self):
        """Test handle JSON parsing errors (return SKIP for all)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "{invalid json"
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert (
            "Failed to parse" in result[0]["reasoning"]
            or "defaulting to SKIP" in result[0]["reasoning"]
        )

    @pytest.mark.asyncio
    async def test_decide_no_json_in_response_returns_skip(self):
        """Test handle no JSON in response (return SKIP for all)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "This is just plain text with no JSON at all"
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert "No valid JSON" in result[0]["reasoning"]

    @pytest.mark.asyncio
    async def test_decide_exception_returns_skip(self):
        """Test handle exceptions (return SKIP for all)."""
        mock_llm = AsyncMock()
        mock_llm.llm_call.side_effect = Exception("Network error")

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "D1", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert "Agent error" in result[0]["reasoning"]

    @pytest.mark.asyncio
    async def test_decide_format_agent_input_correctly(self):
        """Test format agent input correctly."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "UPDATE",
                    "confidence_score": 0.9,
                    "reasoning": "R1",
                }
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New memory content"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Existing doc", "composite_score": 0.85}
        ]

        await agent.decide(new_memory, linked_memories)

        call_args = mock_llm.llm_call.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]

        assert "new_memory" in user_message
        assert "linked_memories" in user_message
        assert "New memory content" in user_message
        assert "Existing doc" in user_message

    @pytest.mark.asyncio
    async def test_decide_include_relationship_strength(self):
        """Test include relationship_strength in input."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "UPDATE",
                    "confidence_score": 0.9,
                    "reasoning": "R1",
                }
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.856}
        ]

        await agent.decide(new_memory, linked_memories)

        call_args = mock_llm.llm_call.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        parsed = json.loads(user_message.split("\n\n")[1])

        assert "relationship_strength" in parsed["linked_memories"][0]
        assert parsed["linked_memories"][0]["relationship_strength"] == 0.856

    @pytest.mark.asyncio
    async def test_decide_empty_linked_memories(self):
        """Test edge case: empty linked_memories."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps([])
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = []

        result = await agent.decide(new_memory, linked_memories)
        assert result == []

    @pytest.mark.asyncio
    async def test_decide_missing_memory_id(self):
        """Test edge case: missing memory_id in linked memories."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "",
                    "operation": "SKIP",
                    "confidence_score": 0.0,
                    "reasoning": "Missing ID",
                }
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [{"document": "Doc", "composite_score": 0.8}]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["memory_id"] == ""

    @pytest.mark.asyncio
    async def test_decide_invalid_operation_value(self):
        """Test edge case: invalid operation values."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [
                {
                    "memory_id": "mem1",
                    "operation": "INVALID",
                    "confidence_score": 0.9,
                    "reasoning": "R1",
                }
            ]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert "Failed to parse" in result[0]["reasoning"]

    @pytest.mark.asyncio
    async def test_decide_missing_required_fields(self):
        """Test edge case: missing required fields."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            [{"memory_id": "mem1", "operation": "UPDATE"}]
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert "Failed to parse" in result[0]["reasoning"]

    @pytest.mark.asyncio
    async def test_decide_json_with_trailing_commas(self):
        """Test edge case: JSON with trailing commas."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = '[{"memory_id": "mem1", "operation": "UPDATE", "confidence_score": 0.9, "reasoning": "R1",}]'
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"

    @pytest.mark.asyncio
    async def test_decide_response_with_extra_text(self):
        """Test edge case: response with extra text before/after JSON."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = """Here's my analysis:
[{"memory_id": "mem1", "operation": "UPDATE", "confidence_score": 0.9, "reasoning": "R1"}]
That's the decision."""
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"

    @pytest.mark.asyncio
    async def test_decide_dict_with_decisions_key(self):
        """Test parse JSON object with 'decisions' field (line 97)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "decisions": [
                    {
                        "memory_id": "mem1",
                        "operation": "UPDATE",
                        "confidence_score": 0.9,
                        "reasoning": "R1",
                    }
                ]
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem1"

    @pytest.mark.asyncio
    async def test_decide_not_list_not_dict_raises_error(self):
        """Test handle when parsed JSON is not list or dict with decisions (line 99)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = '"just a string"'
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert (
            "Failed to parse" in result[0]["reasoning"]
            or "No valid JSON" in result[0]["reasoning"]
        )

    @pytest.mark.asyncio
    async def test_decide_dict_with_decisions_key_coverage_line_97(self):
        """Test coverage for line 97: dict with 'decisions' key extraction."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "decisions": [
                    {
                        "memory_id": "mem1",
                        "operation": "UPDATE",
                        "confidence_score": 0.9,
                        "reasoning": "R1",
                    },
                    {
                        "memory_id": "mem2",
                        "operation": "SKIP",
                        "confidence_score": 0.8,
                        "reasoning": "R2",
                    },
                ]
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc1", "composite_score": 0.8},
            {"memory_id": "mem2", "document": "Doc2", "composite_score": 0.7},
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 2
        assert result[0]["memory_id"] == "mem1"
        assert result[1]["memory_id"] == "mem2"

    @pytest.mark.asyncio
    async def test_decide_value_error_not_list_coverage_line_99(self):
        """Test coverage for line 99: ValueError when decisions is not list."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "123"
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = ConflictResolutionAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "Test"}
        linked_memories = [
            {"memory_id": "mem1", "document": "Doc", "composite_score": 0.8}
        ]

        result = await agent.decide(new_memory, linked_memories)
        assert len(result) == 1
        assert result[0]["operation"] == "SKIP"
        assert (
            "No valid JSON" in result[0]["reasoning"]
            or "Failed to parse" in result[0]["reasoning"]
        )

    @pytest.mark.asyncio
    async def test_consolidate_memories_markdown_code_block_else(self):
        """Test handle markdown code block with else branch (line 237)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = """```
{
  "consolidated_memory": {
    "natural_memory_note": "Note"
  },
  "synthesis_metadata": {
    "notes": "Summary"
  }
}
```"""
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        result = await agent.consolidate_memories(new_memory, existing_memories)
        assert result["consolidated_memory"]["natural_memory_note"] == "Note"


class TestSynthesisAgent:
    """Test cases for SynthesisAgent."""

    def test_init_with_valid_llm_connection(self):
        """Test initialize with valid llm_connection."""
        mock_llm = Mock()
        agent = SynthesisAgent(llm_connection=mock_llm)
        assert agent.llm_connection == mock_llm

    def test_init_stores_llm_connection(self):
        """Test that llm_connection is stored correctly."""
        mock_llm = Mock()
        agent = SynthesisAgent(llm_connection=mock_llm)
        assert hasattr(agent, "llm_connection")
        assert agent.llm_connection is mock_llm

    @pytest.mark.asyncio
    async def test_consolidate_memories_success(self):
        """Test consolidate memories successfully."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Consolidated note"},
                "synthesis_metadata": {"notes": "Synthesis summary"},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New memory"}
        existing_memories = [{"document": "Existing memory 1"}]

        result = await agent.consolidate_memories(new_memory, existing_memories)

        assert "consolidated_memory" in result
        assert "synthesis_summary" in result
        assert (
            result["consolidated_memory"]["natural_memory_note"] == "Consolidated note"
        )
        assert result["synthesis_summary"] == "Synthesis summary"

    @pytest.mark.asyncio
    async def test_consolidate_memories_extract_natural_memory_note(self):
        """Test extract natural_memory_note from consolidated_memory."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Extracted note"},
                "synthesis_metadata": {},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        result = await agent.consolidate_memories(new_memory, existing_memories)
        assert result["consolidated_memory"]["natural_memory_note"] == "Extracted note"

    @pytest.mark.asyncio
    async def test_consolidate_memories_use_synthesis_metadata_notes(self):
        """Test use synthesis_metadata.notes as synthesis_summary."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Note"},
                "synthesis_metadata": {"notes": "Custom summary from metadata"},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        result = await agent.consolidate_memories(new_memory, existing_memories)
        assert result["synthesis_summary"] == "Custom summary from metadata"

    @pytest.mark.asyncio
    async def test_consolidate_memories_fallback_default_summary(self):
        """Test fall back to default summary if extraction fails."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Note"},
                "synthesis_metadata": {},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing 1"}, {"document": "Existing 2"}]

        result = await agent.consolidate_memories(new_memory, existing_memories)
        assert "Consolidated" in result["synthesis_summary"]
        assert "3" in result["synthesis_summary"]

    @pytest.mark.asyncio
    async def test_consolidate_memories_parse_json_markdown(self):
        """Test handle JSON in markdown code blocks."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = """```json
{
  "consolidated_memory": {
    "natural_memory_note": "Note"
  },
  "synthesis_metadata": {
    "notes": "Summary"
  }
}
```"""
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        result = await agent.consolidate_memories(new_memory, existing_memories)
        assert result["consolidated_memory"]["natural_memory_note"] == "Note"

    @pytest.mark.asyncio
    async def test_consolidate_memories_no_choices_raises_error(self):
        """Test handle LLM response with no choices (raise RuntimeError)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_response.choices = []
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        with pytest.raises(RuntimeError, match="Synthesis agent failed to respond"):
            await agent.consolidate_memories(new_memory, existing_memories)

    @pytest.mark.asyncio
    async def test_consolidate_memories_none_response_raises_error(self):
        """Test handle LLM response with None (raise RuntimeError)."""
        mock_llm = AsyncMock()
        mock_llm.llm_call.return_value = None

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        with pytest.raises(RuntimeError, match="Synthesis agent failed to respond"):
            await agent.consolidate_memories(new_memory, existing_memories)

    @pytest.mark.asyncio
    async def test_consolidate_memories_json_parsing_error_raises_error(self):
        """Test handle JSON parsing errors (raise RuntimeError)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Invalid JSON {"
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        with pytest.raises(RuntimeError, match="not valid JSON"):
            await agent.consolidate_memories(new_memory, existing_memories)

    @pytest.mark.asyncio
    async def test_consolidate_memories_not_dict_raises_error(self):
        """Test handle response that's not a dict (raise RuntimeError)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(["not", "a", "dict"])
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        with pytest.raises(RuntimeError, match="must be a JSON object"):
            await agent.consolidate_memories(new_memory, existing_memories)

    @pytest.mark.asyncio
    async def test_consolidate_memories_missing_natural_memory_note_raises_error(self):
        """Test handle missing natural_memory_note (raise RuntimeError)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {"consolidated_memory": {}, "synthesis_metadata": {}}
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        with pytest.raises(RuntimeError, match="missing 'natural_memory_note' field"):
            await agent.consolidate_memories(new_memory, existing_memories)

    @pytest.mark.asyncio
    async def test_consolidate_memories_format_input_correctly(self):
        """Test format synthesis input correctly (only natural_memory_note)."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Note"},
                "synthesis_metadata": {},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {
            "natural_memory_note": "New memory note",
            "other_field": "should not be included",
        }
        existing_memories = [
            {"document": "Existing doc", "other_field": "should not be included"}
        ]

        await agent.consolidate_memories(new_memory, existing_memories)

        call_args = mock_llm.llm_call.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        parsed = json.loads(user_message.split("\n\n")[1])

        assert "natural_memory_note" in parsed["new_memory"]
        assert "other_field" not in parsed["new_memory"]
        assert "natural_memory_note" in parsed["existing_memories"][0]
        assert "other_field" not in parsed["existing_memories"][0]

    @pytest.mark.asyncio
    async def test_consolidate_memories_include_all_existing(self):
        """Test include all existing memories."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Note"},
                "synthesis_metadata": {},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [
            {"document": "Existing 1"},
            {"document": "Existing 2"},
            {"document": "Existing 3"},
        ]

        await agent.consolidate_memories(new_memory, existing_memories)

        call_args = mock_llm.llm_call.call_args
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]
        parsed = json.loads(user_message.split("\n\n")[1])

        assert len(parsed["existing_memories"]) == 3

    @pytest.mark.asyncio
    async def test_consolidate_memories_log_synthesis_input(self):
        """Test log synthesis input."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Note"},
                "synthesis_metadata": {},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New memory"}
        existing_memories = [{"document": "Existing"}]

        with patch("omnimemory.core.agents.logger") as mock_logger:
            await agent.consolidate_memories(new_memory, existing_memories)
            assert mock_logger.info.called

    @pytest.mark.asyncio
    async def test_consolidate_memories_empty_existing_memories(self):
        """Test edge case: empty existing_memories."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Note"},
                "synthesis_metadata": {},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = []

        result = await agent.consolidate_memories(new_memory, existing_memories)
        assert result["consolidated_memory"]["natural_memory_note"] == "Note"

    @pytest.mark.asyncio
    async def test_consolidate_memories_missing_document(self):
        """Test edge case: missing document in existing memories."""
        mock_llm = AsyncMock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(
            {
                "consolidated_memory": {"natural_memory_note": "Note"},
                "synthesis_metadata": {},
            }
        )
        mock_response.choices = [mock_choice]
        mock_llm.llm_call.return_value = mock_response

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"other_field": "value"}]

        await agent.consolidate_memories(new_memory, existing_memories)

    @pytest.mark.asyncio
    async def test_consolidate_memories_exception_raises_runtime_error(self):
        """Test handle exceptions (raise RuntimeError)."""
        mock_llm = AsyncMock()
        mock_llm.llm_call.side_effect = Exception("Network error")

        agent = SynthesisAgent(llm_connection=mock_llm)
        new_memory = {"natural_memory_note": "New"}
        existing_memories = [{"document": "Existing"}]

        with pytest.raises(RuntimeError, match="Synthesis failed"):
            await agent.consolidate_memories(new_memory, existing_memories)
