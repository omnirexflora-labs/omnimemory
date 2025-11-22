import asyncio
import uuid
from typing import List, Dict, Any, Optional
import httpx
from omnimemory.core.schemas import (
    UserMessages,
    ConversationSummaryRequest,
    AgentMemoryRequest,
)
from omnimemory.core.llm import LLMConnection
from omnimemory.memory_management.memory_manager import MemoryManager
from omnimemory.core.logger_utils import get_logger
from omnimemory.core.config import DEFAULT_N_RESULTS
from omnimemory.core.metrics import get_metrics_collector


logger = get_logger(name="omnimemory.memory_layer")


class OmniMemorySDK:
    """Main SDK interface for interacting with the OmniMemory system."""

    def __init__(self):
        """Initialize OmniMemorySDK with LLM connection (async version)."""
        self.llm_connection = LLMConnection()
        self._memory_manager = None
        self._background_tasks: Dict[str, asyncio.Task] = {}

        logger.info("OmniMemorySDK initialized with LLMConnection (async)")

    @property
    def memory_manager(self):
        """Get or create cached MemoryManager instance with connection pool for queries."""
        if self._memory_manager is None:
            self._memory_manager = MemoryManager(
                llm_connection=self.llm_connection,
            )
        return self._memory_manager

    def _register_background_task(self, task_id: str, task: asyncio.Task):
        """Track lifecycle of background asyncio tasks."""
        self._background_tasks[task_id] = task

        def _cleanup(_future):
            self._background_tasks.pop(task_id, None)

        task.add_done_callback(_cleanup)

    def _format_messages(self, user_message: UserMessages) -> str:
        """
        Format UserMessages into a single string for memory processing.

        Only includes the actual conversation messages (with timestamps and roles).
        app_id, user_id, and session_id are excluded as they are metadata passed separately
        and used for storage/retrieval, not needed by the LLM for understanding conversations.

        Args:
            user_message: UserMessages object containing app_id, user_id, session_id, and messages

        Returns:
            Formatted string containing only the conversation messages
        """
        formatted_parts = []

        for msg in user_message.messages:
            timestamp = msg.timestamp if hasattr(msg, "timestamp") else "N/A"
            role = msg.role if hasattr(msg, "role") else "unknown"
            content = msg.content if hasattr(msg, "content") else ""
            formatted_parts.append(f"[{timestamp}] {role}: {content}")

        return "\n".join(formatted_parts)

    def _format_flexible_messages(self, messages: List[Dict[str, Any]] | str) -> str:
        """Normalize flexible conversation payloads into a single string."""
        if isinstance(messages, str):
            return messages.strip()

        formatted_parts = []
        for msg in messages:
            timestamp = msg.get("timestamp") if isinstance(msg, dict) else "N/A"
            role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            formatted_parts.append(f"[{timestamp or 'N/A'}] {role}: {content}")

        return "\n".join(formatted_parts)

    async def add_memory(self, user_message: UserMessages) -> dict:
        """
        Add memory from user messages and trigger asynchronous memory processing (async version).

        This method:
        1. Validates the UserMessages object
        2. Formats the messages for processing
        3. Creates a background task for the complete memory pipeline (create → embed → store)
        4. Returns task information immediately without blocking

        The pipeline creates a memory note from the conversation, embeds it,
        and stores the embedding with comprehensive metadata in the vector database.

        Args:
            user_message: UserMessages object containing app_id, user_id, session_id, and messages

        Returns:
            dict with task_id and status="accepted" if successful, or error information
        """
        try:
            formatted_message = self._format_messages(user_message)

            logger.info(
                f"Processing memory for app_id={user_message.app_id}, "
                f"user_id={user_message.user_id}, session_id={user_message.session_id}, "
                f"message_count={len(user_message.messages)}"
            )
            task_id = str(uuid.uuid4())

            async def _process_memory_background():
                """Background task to process memory asynchronously."""
                try:
                    result = await self.memory_manager.create_and_store_memory(
                        app_id=user_message.app_id,
                        user_id=user_message.user_id,
                        session_id=user_message.session_id,
                        messages=formatted_message,
                        llm_connection=self.llm_connection,
                    )
                    logger.info(
                        f"Background memory processing completed for task_id={task_id}, "
                        f"success={result.success}"
                    )
                    return result.to_dict()
                except Exception as e:
                    logger.error(
                        f"Background memory processing failed for task_id={task_id}: {e}",
                        exc_info=True,
                    )
                    return {
                        "status": "failed",
                        "error": str(e),
                        "task_id": task_id,
                    }

            task = asyncio.create_task(_process_memory_background())
            self._register_background_task(task_id, task)

            result = {
                "task_id": task_id,
                "status": "accepted",
                "message": "Memory creation task submitted successfully",
                "app_id": user_message.app_id,
                "user_id": user_message.user_id,
                "session_id": user_message.session_id,
            }

            logger.info(
                f"Memory creation task submitted for app_id={user_message.app_id}, "
                f"user_id={user_message.user_id}, task_id={task_id}"
            )

            return result

        except Exception as e:
            logger.error(f"Error adding memory: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "app_id": getattr(user_message, "app_id", None),
                "user_id": getattr(user_message, "user_id", None),
            }

    async def summarize_conversation(
        self, summary_request: ConversationSummaryRequest
    ) -> Dict[str, Any]:
        """
        Generate a conversation summary via a single agent with optional webhook delivery.

        If a callback URL is supplied, the work is delegated to a background task and this
        method returns immediately with task metadata. Otherwise the summary result is
        returned synchronously.
        """

        async def _run_summary(use_fast: bool = False) -> Dict[str, Any]:
            return await self.memory_manager.generate_conversation_summary(
                app_id=summary_request.app_id,
                user_id=summary_request.user_id,
                session_id=summary_request.session_id,
                messages=self._format_flexible_messages(summary_request.messages),
                llm_connection=self.llm_connection,
                use_fast_path=use_fast,
            )

        if summary_request.callback_url:
            task_id = str(uuid.uuid4())

            async def _process_and_callback():
                """Background task to compute summary and deliver to webhook."""
                try:
                    summary_payload = await _run_summary(use_fast=False)
                    summary_payload["delivery"] = "callback"
                    await self._post_callback(
                        str(summary_request.callback_url),
                        summary_payload,
                        summary_request.callback_headers,
                    )
                    return summary_payload
                except Exception as exc:
                    logger.error(
                        f"Conversation summary callback task failed: {exc}",
                        exc_info=True,
                    )
                    failure_payload = {
                        "status": "failed",
                        "error": str(exc),
                        "app_id": summary_request.app_id,
                        "user_id": summary_request.user_id,
                        "session_id": summary_request.session_id,
                    }
                    if summary_request.callback_url:
                        try:
                            await self._post_callback(
                                str(summary_request.callback_url),
                                failure_payload,
                                summary_request.callback_headers,
                            )
                        except Exception as callback_exc:
                            logger.error(
                                f"Failed to deliver error callback: {callback_exc}",
                                exc_info=True,
                            )
                    return failure_payload

            task = asyncio.create_task(_process_and_callback())
            self._register_background_task(task_id, task)

            return {
                "task_id": task_id,
                "status": "accepted",
                "message": "Conversation summary scheduled for callback delivery",
                "app_id": summary_request.app_id,
                "user_id": summary_request.user_id,
                "session_id": summary_request.session_id,
            }

        try:
            summary_result = await _run_summary(use_fast=True)
            summary_result["delivery"] = "sync"
            return summary_result
        except Exception as exc:
            logger.error(f"Conversation summary failed: {exc}", exc_info=True)
            raise

    async def add_agent_memory(self, agent_request: AgentMemoryRequest) -> dict:
        """
        Add memory from agent message and trigger asynchronous memory processing (async version).

        This method:
        1. Validates the AgentMemoryRequest
        2. Creates a background task for agent memory processing (summary → embed → store)
        3. Returns task information immediately without blocking

        Simple flow: agent sends messages (string or list), we generate a summary
        using the fast prompt, embed it, and store it directly. No conflict resolution,
        no linking, no metadata extraction - just store.

        Args:
            agent_request: AgentMemoryRequest with app_id, user_id, session_id, and messages

        Returns:
            dict with task_id and status="accepted" if successful, or error information
        """
        try:
            logger.info(
                f"Processing agent memory for app_id={agent_request.app_id}, "
                f"user_id={agent_request.user_id}, session_id={agent_request.session_id}"
            )

            task_id = str(uuid.uuid4())

            async def _process_agent_memory_background():
                """Background task to process agent memory asynchronously."""
                try:
                    result = await self.memory_manager.create_agent_memory(
                        app_id=agent_request.app_id,
                        user_id=agent_request.user_id,
                        session_id=agent_request.session_id,
                        messages=agent_request.messages,
                        llm_connection=self.llm_connection,
                    )
                    logger.info(
                        f"Background agent memory processing completed for task_id={task_id}, "
                        f"success={result.success}"
                    )
                    return result.to_dict()
                except Exception as e:
                    logger.error(
                        f"Background agent memory processing failed for task_id={task_id}: {e}",
                        exc_info=True,
                    )
                    return {
                        "status": "failed",
                        "error": str(e),
                        "task_id": task_id,
                    }

            task = asyncio.create_task(_process_agent_memory_background())
            self._register_background_task(task_id, task)

            result = {
                "task_id": task_id,
                "status": "accepted",
                "message": "Agent memory creation task submitted successfully",
                "app_id": agent_request.app_id,
                "user_id": agent_request.user_id,
                "session_id": agent_request.session_id,
            }

            logger.info(
                f"Agent memory creation task submitted for app_id={agent_request.app_id}, "
                f"user_id={agent_request.user_id}, task_id={task_id}"
            )

            return result

        except Exception as exc:
            logger.error(f"Error adding agent memory: {exc}", exc_info=True)
            return {
                "status": "failed",
                "error": str(exc),
                "app_id": getattr(agent_request, "app_id", None),
                "user_id": getattr(agent_request, "user_id", None),
            }

    async def query_memory(
        self,
        app_id: str,
        query: str,
        user_id: str = None,
        session_id: str = None,
        n_results: int = None,
        similarity_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Query memory with intelligent multi-dimensional ranking (async version).

        Performs semantic search with composite scoring combining:
        - Relevance (semantic similarity)
        - Recency (time-based freshness)
        - Importance (content significance)

        Uses multiplicative approach: composite = relevance × (1 + recency_boost + importance_boost)

        Features flexible hierarchical filtering:
        - Always filters by app_id (collection)
        - Optionally filters by user_id
        - Optionally filters by session_id

        Args:
            app_id: Application ID (required) - determines which collection to search
            query: Natural language query for semantic search
            user_id: Optional user ID filter (searches only this user's memories)
            session_id: Optional session ID filter (searches only this session's memories)
            n_results: Maximum number of results to return (default: from config)
            similarity_threshold: Minimum similarity score (0.0-1.0, default: auto-adjusted from RECALL_THRESHOLD)

        Returns:
            List of memory objects with composite scores and metadata, ranked by relevance
        """

        if n_results is None:
            n_results = DEFAULT_N_RESULTS

        try:
            results = await self.memory_manager.query_memory(
                app_id=app_id,
                query=query,
                user_id=user_id,
                session_id=session_id,
                n_results=n_results,
                similarity_threshold=similarity_threshold,
            )

            if results is None:
                logger.warning("Query returned None, treating as empty results")
                results = []

            for result in results:
                result["query_status"] = "completed"

            logger.info(
                f"Memory query completed for app_id={app_id}, "
                f"user_id={user_id}, session_id={session_id}, "
                f"query='{query[:50]}...', results={len(results)}, status=completed"
            )

            return results

        except Exception as e:
            logger.error(f"Error querying memory: {e}", exc_info=True)
            return []

    async def get_memory(
        self,
        memory_id: str,
        app_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single memory by ID (async version).

        Args:
            memory_id: Document ID of the memory to retrieve
            app_id: Application ID (collection name)

        Returns:
            Dictionary with memory data (document, metadata, etc.) or None if not found
        """
        try:
            return await self.memory_manager.get_memory(
                memory_id=memory_id,
                app_id=app_id,
            )
        except Exception as e:
            logger.error(f"Error getting memory: {e}", exc_info=True)
            return None

    async def traverse_memory_evolution_chain(
        self,
        app_id: str,
        memory_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Traverse the memory evolution chain using singly linked list algorithm.

        Starting from the given memory_id, follows the next_id links forward until
        reaching None, collecting all memories in the evolution chain.

        Args:
            app_id: Application ID (collection name)
            memory_id: Starting memory ID to begin traversal

        Returns:
            List of memory dictionaries in evolution order (oldest to newest)
            Empty list if starting memory not found
        """
        try:
            return await self.memory_manager.traverse_memory_evolution_chain(
                app_id=app_id,
                memory_id=memory_id,
            )
        except Exception as e:
            logger.error(f"Error traversing memory evolution chain: {e}", exc_info=True)
            return []

    def generate_evolution_graph(
        self,
        chain: List[Dict[str, Any]],
        format: str = "mermaid",
    ) -> str:
        """
        Generate a graph visualization of the memory evolution chain.

        Supports multiple formats:
        - "mermaid": Mermaid diagram syntax (text-based, widely supported)
        - "dot": Graphviz DOT format (can be rendered to PNG/SVG)
        - "html": HTML file with embedded Mermaid.js visualization

        Args:
            chain: List of memory dictionaries from traverse_memory_evolution_chain
            format: Output format ("mermaid", "dot", or "html")

        Returns:
            String representation of the graph in the requested format
        """
        try:
            return self.memory_manager.generate_evolution_graph(
                chain=chain, format=format
            )
        except Exception as e:
            logger.error(f"Error generating evolution graph: {e}", exc_info=True)
            return ""

    def generate_evolution_report(
        self,
        chain: List[Dict[str, Any]],
        format: str = "markdown",
    ) -> str:
        """
        Generate a comprehensive evolution report with statistics and insights.

        Args:
            chain: List of memory dictionaries from traverse_memory_evolution_chain
            format: Output format ("markdown", "text", or "json")

        Returns:
            String representation of the evolution report
        """
        try:
            return self.memory_manager.generate_evolution_report(
                chain=chain, format=format
            )
        except Exception as e:
            logger.error(f"Error generating evolution report: {e}", exc_info=True)
            return ""

    async def delete_memory(
        self,
        app_id: str,
        doc_id: str,
    ) -> bool:
        """
        Delete a memory from collection with flexible filtering (async version).

        Args:
            doc_id: Document ID to delete
            app_id: Application ID (collection name)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            memory_manager = self.memory_manager

            result = await memory_manager.delete_memory(
                doc_id=doc_id,
                collection_name=app_id,
            )

            success = result.success if hasattr(result, "success") else result

            if success:
                logger.info(
                    f"Memory deleted: doc_id={doc_id}, collection_name={app_id}"
                )
            else:
                logger.warning(
                    f"Failed to delete memory: doc_id={doc_id}, collection_name={app_id}"
                )

            return success

        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            return False

    async def _post_callback(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Send summary payload to a callback URL with retry logic.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s).
        Skips retries for permanent errors (4xx except 429, 5xx except 503/504).
        """
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                logger.info(
                    f"Delivered conversation summary callback to {url} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                return
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                is_permanent_error = (
                    400 <= status_code < 500 and status_code != 429
                ) or (status_code >= 500 and status_code not in [503, 504])

                is_last_attempt = attempt == max_retries - 1

                if is_permanent_error or is_last_attempt:
                    logger.error(
                        f"Callback delivery to {url} failed after {attempt + 1} attempt(s): {exc}. "
                        f"{'Permanent error, not retrying' if is_permanent_error else 'Max retries reached'}",
                        exc_info=True,
                    )
                    raise
                else:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Callback delivery to {url} failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
            except Exception as exc:
                is_last_attempt = attempt == max_retries - 1
                if is_last_attempt:
                    logger.error(
                        f"Callback delivery to {url} failed after {max_retries} attempts: {exc}",
                        exc_info=True,
                    )
                    raise
                else:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Callback delivery to {url} failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

    async def warm_up(self) -> bool:
        """Warm up underlying connection pools before serving requests."""
        try:
            return await self.memory_manager.warm_up_connection_pool()
        except Exception as exc:
            logger.warning(f"SDK warm-up failed: {exc}")
            return False

    async def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Expose connection pool stats for health checks or debugging."""
        try:
            return await self.memory_manager.connection_pool.get_pool_stats()
        except Exception as exc:
            logger.warning(f"Failed to fetch pool stats: {exc}")
            return {"error": str(exc)}

    async def get_task_status(self, task_id: str) -> dict:
        """
        Get the status and result of a memory creation task (async version).

        Args:
            task_id: The task ID from add_memory

        Returns:
            dict with task status and result (if completed)
        """
        try:
            task = self._background_tasks.get(task_id)

            if task is None:
                return {
                    "task_id": task_id,
                    "status": "unknown",
                    "message": "Task not found (may have completed)",
                }

            if task.done():
                try:
                    result = task.result()
                    return {
                        "task_id": task_id,
                        "status": "completed" if result.get("success") else "failed",
                        "result": result,
                    }
                except BaseException as e:
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": str(e),
                    }
            else:
                return {
                    "task_id": task_id,
                    "status": "processing",
                    "message": "Task is still running",
                }

        except Exception as e:
            logger.error(
                f"Error getting task status for task_id={task_id}: {e}", exc_info=True
            )
            return {"task_id": task_id, "status": "error", "error": str(e)}
