import json
from typing import Any, Dict, List, Protocol, cast

from omnimemory.core.logger_utils import get_logger
from omnimemory.core.system_prompts import (
    conflict_resolution_agent_prompt,
    synthesis_agent_prompt,
)
from omnimemory.core.utils import clean_and_parse_json

logger = get_logger(name="omnimemory.core.agents")


class SupportsLLMCall(Protocol):
    """Protocol describing dependencies that expose an async llm_call method."""

    async def llm_call(self, messages: List[Dict[str, str]]) -> Any:
        """Submit chat messages to the backing LLM client."""
        ...


class ConflictResolutionAgent:
    """AI agent that decides how to handle memory conflicts and relationships."""

    def __init__(self, llm_connection: SupportsLLMCall) -> None:
        """Initialize the conflict resolution agent.

        Args:
            llm_connection: LLM connection for decision making
        """
        self.llm_connection = llm_connection

    async def decide(
        self, new_memory: Dict[str, Any], linked_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze new memory and linked memories to decide on conflict resolution strategy.

        Args:
            new_memory: The new memory being considered for storage.
            linked_memories: List of semantically linked existing memories.

        Returns:
            List of decision dictionaries (memory_id, operation, confidence, reasoning).
        """
        try:
            agent_input = {
                "new_memory": {
                    "content": new_memory.get("natural_memory_note", ""),
                },
                "linked_memories": [
                    {
                        "memory_id": linked.get("memory_id", ""),
                        "relationship_strength": round(
                            linked.get("composite_score", 0.0), 3
                        ),
                        "content": linked.get("document", ""),
                    }
                    for linked in linked_memories
                ],
            }

            messages = [
                {"role": "system", "content": conflict_resolution_agent_prompt},
                {
                    "role": "user",
                    "content": f"Analyze this memory conflict scenario and decide on the appropriate action:\n\n{json.dumps(agent_input, indent=2)}",
                },
            ]

            response = await self.llm_connection.llm_call(messages=messages)

            if not response or not response.choices:
                logger.warning(
                    "Conflict resolution agent returned no response, defaulting to SKIP for all"
                )
                default_decisions = []
                for linked in linked_memories:
                    default_decisions.append(
                        {
                            "memory_id": linked.get("memory_id", ""),
                            "operation": "SKIP",
                            "confidence_score": 0.0,
                            "reasoning": "Agent failed to respond, defaulting to SKIP for safety",
                        }
                    )
                return default_decisions

            content = response.choices[0].message.content.strip()

            json_start = content.find("[")
            if json_start == -1:
                json_start = content.find("{")
            json_end = (
                content.rfind("]") + 1
                if content.find("[") != -1
                else content.rfind("}") + 1
            )

            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                try:
                    parsed = clean_and_parse_json(json_content)

                    if isinstance(parsed, dict) and "decisions" in parsed:
                        decisions = cast(List[Dict[str, Any]], parsed["decisions"])
                    elif isinstance(parsed, list):
                        decisions = cast(List[Dict[str, Any]], parsed)
                    else:
                        raise ValueError("Expected JSON array of decisions")

                    for i, dec in enumerate(decisions):
                        required_fields = [
                            "memory_id",
                            "operation",
                            "confidence_score",
                            "reasoning",
                        ]
                        if not all(key in dec for key in required_fields):
                            raise ValueError(
                                f"Decision {i} missing required fields: {required_fields}"
                            )

                        if dec["operation"] not in ["UPDATE", "DELETE", "SKIP"]:
                            raise ValueError(
                                f"Invalid operation in decision {i}: {dec['operation']}"
                            )

                        confidence = dec.get("confidence_score", 0.0)
                        if not isinstance(confidence, (int, float)) or not (
                            0.0 <= confidence <= 1.0
                        ):
                            logger.warning(
                                f"Invalid confidence score {confidence} in decision {i}, setting to 0.5"
                            )
                            dec["confidence_score"] = 0.5

                    logger.info(
                        f"Conflict resolution agent returned {len(decisions)} granular decisions"
                    )
                    return decisions

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse agent response: {e}, content: {content}"
                    )
                    fallback_decisions = []
                    for linked in linked_memories:
                        fallback_decisions.append(
                            {
                                "memory_id": linked.get("memory_id", ""),
                                "operation": "SKIP",
                                "confidence_score": 0.0,
                                "reasoning": f"Failed to parse agent response: {e}, defaulting to SKIP",
                            }
                        )
                    return fallback_decisions
            else:
                logger.error(f"No JSON found in agent response: {content}")
                fallback_decisions = []
                for linked in linked_memories:
                    fallback_decisions.append(
                        {
                            "memory_id": linked.get("memory_id", ""),
                            "operation": "SKIP",
                            "confidence_score": 0.0,
                            "reasoning": "No valid JSON in agent response, defaulting to SKIP",
                        }
                    )
                return fallback_decisions

        except Exception as e:
            logger.error(f"Error in conflict resolution agent: {e}")
            fallback_decisions = []
            for linked in linked_memories:
                fallback_decisions.append(
                    {
                        "memory_id": linked.get("memory_id", ""),
                        "operation": "SKIP",
                        "confidence_score": 0.0,
                        "reasoning": f"Agent error: {e}, defaulting to SKIP for safety",
                    }
                )
            return fallback_decisions


class SynthesisAgent:
    """AI agent that consolidates multiple memories into a single comprehensive memory."""

    def __init__(self, llm_connection: SupportsLLMCall) -> None:
        """Initialize the synthesis agent.

        Args:
            llm_connection: LLM connection for memory consolidation
        """
        self.llm_connection = llm_connection

    async def consolidate_memories(
        self, new_memory: Dict[str, Any], existing_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Consolidate a new memory with existing related memories into a single comprehensive memory.

        Args:
            new_memory: The new memory to consolidate.
            existing_memories: List of existing memories to consolidate with.

        Returns:
            Dict containing consolidated_memory and synthesis_summary strings.
        """
        try:
            logger.debug(
                f"Synthesis agent input: {new_memory.get('natural_memory_note', '')[:150]}..."
            )
            logger.debug(f"Existing memories: {existing_memories[:150]}...")
            synthesis_input = {
                "new_memory": {
                    "natural_memory_note": new_memory.get("natural_memory_note", ""),
                },
                "existing_memories": [
                    {
                        "natural_memory_note": mem.get("document", ""),
                    }
                    for mem in existing_memories
                ],
            }

            messages = [
                {"role": "system", "content": synthesis_agent_prompt},
                {
                    "role": "user",
                    "content": f"Consolidate these memories into a single, comprehensive memory:\n\n{json.dumps(synthesis_input, indent=2)}",
                },
            ]

            response = await self.llm_connection.llm_call(messages=messages)

            if not response or not response.choices:
                logger.error("Synthesis agent returned no response")
                raise RuntimeError("Synthesis agent failed to respond")

            content = response.choices[0].message.content.strip()

            cleaned_content = content
            if "```json" in cleaned_content:
                cleaned_content = (
                    cleaned_content.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in cleaned_content:
                cleaned_content = (
                    cleaned_content.split("```")[1].split("```")[0].strip()
                )

            try:
                parsed_json = clean_and_parse_json(cleaned_content)
            except Exception as exc:
                logger.error(
                    "Failed to parse synthesis agent response: %s | Content preview: %s",
                    exc,
                    content[:200],
                )
                raise RuntimeError(
                    "Synthesis agent response is not valid JSON"
                ) from exc

            if not isinstance(parsed_json, dict):
                logger.error(
                    "Synthesis agent response is not a JSON object: %s", content[:200]
                )
                raise RuntimeError("Synthesis agent response must be a JSON object")

            consolidated_memory = parsed_json.get("consolidated_memory", {}) or {}
            natural_memory_note = consolidated_memory.get("natural_memory_note")
            synthesis_metadata = parsed_json.get("synthesis_metadata", {}) or {}
            synthesis_notes = synthesis_metadata.get("notes")

            if not natural_memory_note:
                logger.error(
                    f"Could not extract natural_memory_note from synthesis response: {content[:500]}"
                )
                raise RuntimeError(
                    "Synthesis agent response missing 'natural_memory_note' field"
                )

            if synthesis_notes:
                synthesis_summary = synthesis_notes
            else:
                synthesis_summary = f"Consolidated {len(existing_memories) + 1} memories into comprehensive memory"

            result = {
                "consolidated_memory": {"natural_memory_note": natural_memory_note},
                "synthesis_summary": synthesis_summary,
            }

            logger.info(f"Synthesis completed: {synthesis_summary[:100]}...")
            return result

        except Exception as e:
            logger.error(f"Error in synthesis agent: {e}")
            raise RuntimeError(f"Synthesis failed: {e}")
