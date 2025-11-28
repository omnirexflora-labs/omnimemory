from typing import List, Dict, Any, Optional

from omnimemory.core.logger_utils import get_logger
from omnimemory.core.llm import LLMConnection
from omnimemory.memory_management.memory_manager import MemoryManager


logger = get_logger(name="omnimemory.agent_sdk")

DEFAULT_AGENT_PROMPT = """
You are OmniMemory Agent, an assistant that answers user questions using the
provided memory context. If the memories do not contain the information needed,
state that clearly instead of hallucinating.

Memories:
{memory_context}
""".strip()


class AgentMemorySDK:
    """
    Simple SDK for answering queries with OmniMemory context + LLM.

    Provides a high-level interface for querying memories and generating
    LLM responses with memory context.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        llm_connection: Optional[LLMConnection] = None,
    ) -> None:
        """
        Initialize AgentMemorySDK.

        Args:
            system_prompt: Optional custom system prompt. If not provided, uses DEFAULT_AGENT_PROMPT.
            llm_connection: Optional LLM connection instance. If not provided, creates a new one.
        """
        self.llm_connection: LLMConnection = llm_connection or LLMConnection()
        self.memory_manager: MemoryManager = MemoryManager(
            llm_connection=self.llm_connection
        )
        self.default_prompt: str = system_prompt or DEFAULT_AGENT_PROMPT

    async def answer_query(
        self,
        app_id: str,
        query: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        n_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve memories for the query and call the LLM with the combined context.

        Args:
            app_id: Application identifier (required).
            query: User query string.
            user_id: Optional user identifier for filtering memories.
            session_id: Optional session identifier for filtering memories.
            n_results: Optional number of memory results to retrieve.
            similarity_threshold: Optional minimum similarity threshold for memory retrieval.
            system_prompt: Optional custom system prompt (overrides default).

        Returns:
            Dictionary containing:
                - answer: LLM-generated answer string
                - memories: List of retrieved memory dictionaries
                - llm_response: Raw LLM response object
        """
        memories = await self._fetch_memories(
            app_id=app_id,
            query=query,
            user_id=user_id,
            session_id=session_id,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
        )

        memory_context = self._build_memory_context(memories)
        prompt = (system_prompt or self.default_prompt).format(
            memory_context=memory_context
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        llm_response = await self.llm_connection.llm_call(messages=messages)
        answer = self._extract_answer(llm_response)

        return {
            "answer": answer,
            "memories": memories,
            "llm_response": llm_response,
        }

    async def _fetch_memories(
        self,
        app_id: str,
        query: str,
        user_id: Optional[str],
        session_id: Optional[str],
        n_results: Optional[int],
        similarity_threshold: Optional[float],
    ) -> List[Dict[str, Any]]:
        """
        Fetch relevant memories for a query.

        Args:
            app_id: Application identifier.
            query: Query string for semantic search.
            user_id: Optional user identifier filter.
            session_id: Optional session identifier filter.
            n_results: Optional number of results to return.
            similarity_threshold: Optional minimum similarity threshold.

        Returns:
            List of memory dictionaries, or empty list if query fails.
        """
        try:
            results = await self.memory_manager.query_memory(
                app_id=app_id,
                query=query,
                user_id=user_id,
                session_id=session_id,
                n_results=n_results,
                similarity_threshold=similarity_threshold,
            )
            return results or []
        except Exception as exc:
            logger.error(f"Failed to query memories: {exc}", exc_info=True)
            return []

    def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        Build a formatted memory context string from memory dictionaries.

        Args:
            memories: List of memory dictionaries.

        Returns:
            Formatted string containing all memory documents, or a message
            indicating no memories were found.
        """
        if not memories:
            return "No relevant memories were found."

        chunks = []
        for idx, memory in enumerate(memories, start=1):
            doc = memory.get("memory_note") or ""
            chunks.append(f"[Memory {idx}] {doc}".strip())
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _extract_answer(self, response: Optional[Dict[str, Any]]) -> str:
        """
        Extract the answer text from an LLM response.

        Handles various response formats including:
        - Standard OpenAI-style responses with choices[0].message.content
        - LiteLLM list-based content format
        - Empty or malformed responses

        Args:
            response: LLM response dictionary, or None.

        Returns:
            Extracted answer string, or an error message if extraction fails.
        """
        if not response:
            return "LLM response unavailable."

        try:
            choices = response.get("choices") or []
            if not choices:
                return "LLM returned no content."
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                return (
                    "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ).strip()
                    or "LLM returned empty content."
                )
            if isinstance(content, str):
                return content.strip() or "LLM returned empty content."
        except Exception as exc:
            logger.error(f"Failed to parse LLM response: {exc}", exc_info=True)
        return "Unable to parse LLM response."
