import json
import re
import asyncio
import time
from datetime import datetime, timezone
import uuid
from typing import List, Any, Callable, Optional, Dict, Tuple
from contextlib import asynccontextmanager
from omnimemory.core.logger_utils import get_logger
from omnimemory.core.system_prompts import (
    episodic_memory_constructor_system_prompt,
    summarizer_memory_constructor_system_prompt,
    fast_conversation_summary_prompt,
)
from omnimemory.core.utils import (
    calculate_composite_score,
    calculate_recency_score,
    calculate_importance_score,
    determine_relationship_type,
    create_zettelkasten_memory_note,
    prepare_memory_for_storage,
    fuzzy_dedup,
    clean_and_parse_json,
)
from omnimemory.core.config import (
    DEFAULT_N_RESULTS,
    RECALL_THRESHOLD,
    LINK_THRESHOLD,
    COMPOSITE_SCORE_THRESHOLD,
    VECTOR_DB_MAX_CONNECTIONS,
)
from omnimemory.core.agents import ConflictResolutionAgent, SynthesisAgent
from omnimemory.core.results import MemoryOperationResult, BatchOperationResult
from omnimemory.core.metrics import get_metrics_collector
from .connection_pool import VectorDBHandlerPool
from omnimemory.core.types import MemoryNoteData, MemoryDataDict

logger = get_logger(name="omnimemory.memory_management.memory_manager")

_CANDIDATE_MULTIPLIER = 3
_MAX_EXPANDED_RESULTS = 100
MAX_LINKS_FOR_SYNTHESIS = 4
STATUS_REASON_CONSOLIDATED = "consolidated"
STATUS_REASON_CONTRADICTED = "contradicted"
STATUS_REASON_MANUAL = "manual_update"
_DEFAULT_STATUS_CHUNK_SIZE = 10


class MemoryManager:
    """Memory management operations."""

    def __init__(self, llm_connection: Callable):
        """Initialize memory manager with automatic vector database setup.

        Args:
            llm_connection: LLM connection for embeddings (required)
        """
        self.llm_connection = llm_connection
        self.connection_pool = VectorDBHandlerPool.get_instance(
            max_connections=VECTOR_DB_MAX_CONNECTIONS
        )

        self.conflict_resolution_agent = ConflictResolutionAgent(llm_connection)
        self.synthesis_agent = SynthesisAgent(llm_connection)

    @asynccontextmanager
    async def _get_pooled_handler(self):
        """Get a handler from the connection pool."""
        async with self.connection_pool.get_handler(
            self.llm_connection
        ) as vector_db_handler:
            yield vector_db_handler

    async def warm_up_connection_pool(self) -> bool:
        """Eagerly initialize the vector DB handler pool."""
        try:
            async with self._get_pooled_handler() as handler:
                if handler and handler.enabled:
                    logger.info("VectorDB connection pool warm-up completed")
                    return True
                logger.warning("Warm-up handler unavailable or disabled")
                return False
        except Exception as exc:
            logger.warning(f"VectorDB connection pool warm-up failed: {exc}")
            return False

    async def create_episodic_memory(
        self, message: str, llm_connection: Callable
    ) -> Optional[str]:
        """Create an episodic memory from a conversation."""
        try:
            llm_messages = [
                {
                    "role": "system",
                    "content": episodic_memory_constructor_system_prompt,
                },
                {"role": "user", "content": message},
            ]

            response = await llm_connection.llm_call(messages=llm_messages)
            if response and response.choices:
                content = response.choices[0].message.content
                return content
            return None
        except Exception as e:
            logger.error(f"Error creating episodic memory: {e}")
            return None

    async def create_summarizer_memory(
        self, message: str, llm_connection: Callable
    ) -> Optional[str]:
        """Create a summarizer memory from a conversation."""
        try:
            llm_messages = [
                {
                    "role": "system",
                    "content": summarizer_memory_constructor_system_prompt,
                },
                {"role": "user", "content": message},
            ]

            response = await llm_connection.llm_call(messages=llm_messages)
            if response and response.choices:
                content = response.choices[0].message.content
                return content
            return None
        except Exception as e:
            logger.error(f"Error creating summarizer memory: {e}")
            return None

    async def generate_conversation_summary(
        self,
        app_id: str,
        user_id: str,
        session_id: Optional[str],
        messages: str,
        llm_connection: Callable,
        use_fast_path: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate conversation summary.

        Args:
            use_fast_path: If True, use simple text-only summary (faster, no metadata).
                          If False, use full structured summary with metadata (slower).
        """
        start_time = time.time()
        metrics = get_metrics_collector()

        logger.info(
            f"Generating conversation summary (fast_path={use_fast_path})",
            extra={
                "app_id": app_id,
                "user_id": user_id,
                "session_id": session_id,
            },
        )

        try:
            if use_fast_path:
                result = await self._generate_fast_summary(
                    app_id, user_id, session_id, messages, llm_connection
                )
            else:
                result = await self._generate_full_summary(
                    app_id, user_id, session_id, messages, llm_connection
                )

            duration = time.time() - start_time
            metrics.record_write(
                operation="generate_conversation_summary",
                duration=duration,
                success=True,
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            error_code = type(e).__name__
            metrics.record_write(
                operation="generate_conversation_summary",
                duration=duration,
                success=False,
                error_code=error_code,
            )
            raise

    async def _generate_fast_summary(
        self,
        app_id: str,
        user_id: str,
        session_id: Optional[str],
        messages: str,
        llm_connection: Callable,
    ) -> Dict[str, Any]:
        """Generate a simple text-only summary (fast path)."""
        try:
            llm_messages = [
                {"role": "system", "content": fast_conversation_summary_prompt},
                {"role": "user", "content": messages},
            ]

            response = await llm_connection.llm_call(messages=llm_messages)
            if not response or not response.choices:
                raise ValueError("Fast summary agent returned no response")

            summary_text = response.choices[0].message.content.strip()

            return {
                "app_id": app_id,
                "user_id": user_id,
                "session_id": session_id,
                "summary": summary_text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error generating fast summary: {e}", exc_info=True)
            raise

    async def create_agent_memory(
        self,
        app_id: str,
        user_id: str,
        session_id: Optional[str],
        messages: str | List[Dict[str, Any]],
        llm_connection: Callable,
    ) -> MemoryOperationResult:
        """
        Create and store a memory from agent messages.

        Simple flow: generate summary using fast prompt, embed, and store.
        No conflict resolution, no linking, no metadata extraction - just store.

        Args:
            app_id: Application ID
            user_id: User ID
            session_id: Optional session ID
            messages: Messages content (string or list of message objects)
            llm_connection: LLM connection for summary generation

        Returns:
            MemoryOperationResult with success status and memory_id
        """
        start_time = time.time()
        metrics = get_metrics_collector()

        logger.info(
            f"Creating agent memory: app_id={app_id}, user_id={user_id}, session_id={session_id}"
        )

        try:
            if isinstance(messages, list):
                formatted_parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        timestamp = msg.get("timestamp", "")
                        formatted_parts.append(f"[{timestamp}] {role}: {content}")
                    else:
                        formatted_parts.append(str(msg))
                message_text = "\n".join(formatted_parts)
            else:
                message_text = str(messages)

            summary_result = await self._generate_fast_summary(
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                messages=message_text,
                llm_connection=llm_connection,
            )
            summary_text = summary_result["summary"]

            try:
                embedding = await self.embed_memory_note(summary_text)
                if not embedding:
                    duration = time.time() - start_time
                    metrics.record_write(
                        operation="create_agent_memory",
                        duration=duration,
                        success=False,
                        error_code="EMBEDDING_FAILED",
                    )
                    return MemoryOperationResult.error_result(
                        error_code="EMBEDDING_FAILED",
                        error_message="Failed to embed agent memory summary",
                        collection_name=app_id,
                    )
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_write(
                    operation="create_agent_memory",
                    duration=duration,
                    success=False,
                    error_code="EMBEDDING_EXCEPTION",
                )
                return MemoryOperationResult.error_result(
                    error_code="EMBEDDING_EXCEPTION",
                    error_message=f"Embedding exception: {str(e)}",
                    collection_name=app_id,
                )

            memory_id = str(uuid.uuid4())

            store_result = await self.store_memory_note(
                doc_id=memory_id,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                memory_note_text=summary_text,
                embedding=embedding,
                tags=None,
                keywords=None,
                semantic_queries=None,
                conversation_complexity=None,
                interaction_quality=None,
                follow_up_potential=None,
                status="active",
            )

            duration = time.time() - start_time
            if not store_result.success:
                metrics.record_write(
                    operation="create_agent_memory",
                    duration=duration,
                    success=False,
                    error_code=store_result.error_code or "STORE_FAILED",
                )
                return store_result

            metrics.record_write(
                operation="create_agent_memory",
                duration=duration,
                success=True,
            )
            logger.info(f"Agent memory created successfully: memory_id={memory_id}")
            return store_result
        except Exception as e:
            duration = time.time() - start_time
            error_code = type(e).__name__
            metrics.record_write(
                operation="create_agent_memory",
                duration=duration,
                success=False,
                error_code=error_code,
            )
            raise

    async def _generate_full_summary(
        self,
        app_id: str,
        user_id: str,
        session_id: Optional[str],
        messages: str,
        llm_connection: Callable,
    ) -> Dict[str, Any]:
        """Generate full structured summary with metadata (slower path)."""
        raw_summary = await self.create_summarizer_memory(messages, llm_connection)
        if not raw_summary:
            raise ValueError("Summarizer agent returned no content")

        try:
            summary_data = clean_and_parse_json(raw_summary)
        except Exception as exc:
            logger.error(f"Failed to parse summarizer response: {exc}")
            raise

        if isinstance(summary_data, list) and summary_data:
            summary_data = summary_data[0]

        if not isinstance(summary_data, dict):
            raise ValueError("Summarizer response is not a dict structure")

        narrative = summary_data.get("narrative", "")

        return {
            "app_id": app_id,
            "user_id": user_id,
            "session_id": session_id,
            "summary": narrative or "No summary generated.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def create_combined_memory(
        self, message: str, llm_connection: Callable
    ) -> Optional[str]:
        """Create a combined memory note by running both episodic and summarizer constructors in parallel and merging results."""
        try:
            logger.info("Starting parallel memory creation using asyncio.gather")

            episodic_result, summarizer_result = await asyncio.gather(
                self.create_episodic_memory(message, llm_connection),
                self.create_summarizer_memory(message, llm_connection),
            )

            if not episodic_result:
                logger.error("Failed to create episodic memory")
                return None

            if not summarizer_result:
                logger.error("Failed to create summarizer memory")
                return None
            try:
                episodic_raw = clean_and_parse_json(episodic_result)
                summarizer_raw = clean_and_parse_json(summarizer_result)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse memory JSON: {e}")
                return None

            episodic_data = episodic_raw
            summarizer_data = summarizer_raw
            memory_note_text = create_zettelkasten_memory_note(
                episodic_data=episodic_data, summary_data=summarizer_data
            )
            memory_note = prepare_memory_for_storage(
                note_text=memory_note_text,
                summary_data=summarizer_data,
                episodic_data=episodic_data,
                message_count=len(memory_note_text),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("Memory note normalization and union completed successfully")
            return json.dumps(memory_note, indent=2)

        except Exception as e:
            logger.error(f"Error creating combined memory: {e}")
            return None

    async def query_memory(
        self,
        app_id: str,
        query: str,
        n_results: int = None,
        user_id: str = None,
        session_id: str = None,
        similarity_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Query memory with intelligent multi-dimensional ranking (async version).

        Performs semantic search with composite scoring combining relevance, recency, and importance.
        Uses multiplicative approach: composite = relevance × (1 + recency_boost + importance_boost)
        Uses user-provided similarity_threshold if given, otherwise uses default RECALL_THRESHOLD from config.
        Uses the app_id as the collection name, and the status as active.

        Args:
            app_id: Application ID (collection name) - required
            query: Natural language query for semantic search
            n_results: Maximum number of results to return (default: from config)
            user_id: Optional user ID filter
            session_id: Optional session ID filter
            similarity_threshold: Minimum similarity score (0.0-1.0, default: RECALL_THRESHOLD from config)

        Returns:
            List of memory objects with composite scores and metadata, ranked by relevance
        """
        if n_results is None:
            n_results = DEFAULT_N_RESULTS

        if similarity_threshold is None:
            similarity_threshold = RECALL_THRESHOLD

        if not app_id:
            logger.error("app_id is required for memory queries")
            return []

        metrics = get_metrics_collector()
        with metrics.operation_timer("query", "query_memory") as timer:
            try:
                filter_conditions = {
                    "app_id": app_id,
                    "status": "active",
                }

                if user_id:
                    filter_conditions["user_id"] = user_id

                if session_id:
                    filter_conditions["session_id"] = session_id

                collection_name = app_id

                async with self._get_pooled_handler() as vector_db_handler:
                    if not vector_db_handler or not vector_db_handler.enabled:
                        logger.info("Vector database handler not available")
                        timer.success = False
                        return []

                    results = await self._execute_query(
                        vector_db_handler=vector_db_handler,
                        collection_name=collection_name,
                        query=query,
                        filter_conditions=filter_conditions,
                        expanded_n_results=min(
                            n_results * _CANDIDATE_MULTIPLIER, _MAX_EXPANDED_RESULTS
                        ),
                        similarity_threshold=similarity_threshold,
                        n_results=n_results,
                    )
                    timer.success = len(results) > 0
                    timer.results_count = len(results)
                    return results
            except Exception as e:
                timer.success = False
                timer.error_code = type(e).__name__
                logger.error(f"Error in query_memory: {e}", exc_info=True)
                return []

    async def _execute_query(
        self,
        vector_db_handler,
        collection_name: str,
        query: str,
        filter_conditions: Dict[str, Any],
        expanded_n_results: int,
        similarity_threshold: float,
        n_results: int,
    ) -> List[Dict[str, Any]]:
        """Execute the actual query (extracted for reuse with pool) - async version."""
        try:
            logger.info(
                f"Querying with similarity_threshold={similarity_threshold}, "
                f"expanded_n_results={expanded_n_results}, filter_conditions={filter_conditions}"
            )

            results = await vector_db_handler.query_collection(
                collection_name=collection_name,
                query=query,
                n_results=expanded_n_results,
                similarity_threshold=similarity_threshold,
                filter_conditions=filter_conditions,
            )

            raw_candidates_count = len(results.get("documents", []))
            logger.info(
                f"Stage 1 (DB Query): Retrieved {raw_candidates_count} candidates from Qdrant "
                f"with threshold={similarity_threshold}"
            )

            all_candidates = []
            for i, (doc, score, metadata) in enumerate(
                zip(
                    results.get("documents", []),
                    results.get("scores", []),
                    results.get("metadatas", []),
                )
            ):
                if score >= similarity_threshold:
                    composite_score = calculate_composite_score(
                        semantic_score=score, metadata=metadata
                    )

                    memory_result = {
                        "rank": i + 1,
                        "similarity_score": round(score, 3),
                        "composite_score": round(composite_score, 3),
                        "recency_score": round(
                            calculate_recency_score(
                                created_at_str=metadata.get("created_at"),
                                updated_at_str=metadata.get("updated_at"),
                            ),
                            3,
                        ),
                        "importance_score": round(
                            calculate_importance_score(metadata), 3
                        ),
                        "memory_note": doc,
                        "metadata": {
                            "document_id": metadata.get("document_id"),
                            "app_id": metadata.get("app_id"),
                            "user_id": metadata.get("user_id"),
                            "session_id": metadata.get("session_id"),
                            "created_at": metadata.get("created_at"),
                            "updated_at": metadata.get("updated_at"),
                            "status": metadata.get("status"),
                            "next_id": metadata.get("next_id"),
                            "tags": metadata.get("tags", []),
                            "keywords": metadata.get("keywords", []),
                            "semantic_queries": metadata.get("semantic_queries", []),
                            "conversation_complexity": metadata.get(
                                "conversation_complexity"
                            ),
                            "interaction_quality": metadata.get("interaction_quality"),
                            "follow_up_potential": metadata.get(
                                "follow_up_potential", []
                            ),
                        },
                    }
                    all_candidates.append(memory_result)

            logger.info(
                f"Stage 1 (Recall Filter): {len(all_candidates)} candidates passed "
                f"similarity threshold ({similarity_threshold})"
            )

            if len(all_candidates) == 0:
                logger.warning(
                    f"No candidates passed recall threshold! Raw scores from DB: "
                    f"{[round(s, 3) for s in results.get('scores', [])[:5]]}"
                )

            all_candidates.sort(key=lambda x: x["composite_score"], reverse=True)

            precision_filtered = [
                candidate
                for candidate in all_candidates
                if candidate["composite_score"] >= COMPOSITE_SCORE_THRESHOLD
            ]

            logger.info(
                f"Stage 2 (Precision Filter): {len(all_candidates)} candidates → "
                f"{len(precision_filtered)} after composite score filter (composite>={COMPOSITE_SCORE_THRESHOLD})"
            )

            if len(all_candidates) > 0 and len(precision_filtered) == 0:
                top_composite = (
                    all_candidates[0]["composite_score"] if all_candidates else 0.0
                )
                logger.warning(
                    f"All candidates filtered out by composite score threshold! "
                    f"Top composite score: {top_composite:.3f}, threshold: {COMPOSITE_SCORE_THRESHOLD}"
                )

            final_results = precision_filtered[:n_results]

            if len(precision_filtered) < len(all_candidates):
                logger.info(
                    f"Precision filter removed {len(all_candidates) - len(precision_filtered)} low-quality candidates. "
                    f"Returning {len(final_results)} high-quality results (requested {n_results})."
                )
            for i, result in enumerate(final_results, 1):
                result["rank"] = i

            logger.info(
                f"Memory query returned {len(final_results)} top results (from {len(all_candidates)} candidates) for query: '{query[:50]}...'"
            )
            return final_results

        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            return []

    async def add_memory(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        embedding: List[float],
        metadata: dict,
    ) -> MemoryOperationResult:
        """Add memory to collection.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID
            document: Document content
            embedding: Embedding vector for the document
            metadata: Document metadata

        Returns:
            MemoryOperationResult with success status and error details
        """
        metrics = get_metrics_collector()
        with metrics.operation_timer("write", "add_memory") as timer:
            async with self._get_pooled_handler() as vector_db_handler:
                try:
                    success = await vector_db_handler.add_to_collection(
                        collection_name=collection_name,
                        doc_id=doc_id,
                        document=document,
                        embedding=embedding,
                        metadata=metadata,
                    )
                except Exception as e:
                    logger.error(f"Error adding memory {doc_id}: {e}", exc_info=True)
                    result = MemoryOperationResult.error_result(
                        error_code="ADD_EXCEPTION",
                        error_message=f"Exception while adding memory: {str(e)}",
                        memory_id=doc_id,
                        collection_name=collection_name,
                        exception_type=type(e).__name__,
                    )
                    timer.success = False
                    timer.error_code = result.error_code
                    return result

            if success:
                logger.info(f"Successfully added memory {doc_id} to {collection_name}")
                result = MemoryOperationResult.success_result(
                    memory_id=doc_id,
                    collection_name=collection_name,
                    operation="add_memory",
                )
                timer.success = True
                return result

            result = MemoryOperationResult.error_result(
                error_code="ADD_FAILED",
                error_message=f"Failed to add memory {doc_id} to collection",
                memory_id=doc_id,
                collection_name=collection_name,
            )
            timer.success = False
            timer.error_code = result.error_code
            return result

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

        async with self._get_pooled_handler() as vector_db_handler:
            if not vector_db_handler or not vector_db_handler.enabled:
                logger.warning("Vector database handler not available")
                return None

            try:
                memory = await vector_db_handler.query_by_id(
                    collection_name=app_id, doc_id=memory_id
                )
                return memory
            except Exception as e:
                logger.error(f"Error getting memory {memory_id}: {e}", exc_info=True)
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

        Algorithm:
        1. Start from the given memory_id
        2. Fetch the memory and add it to the result list
        3. Extract next_id from metadata
        4. If next_id is not None, repeat from step 2 with next_id
        5. Continue until next_id is None (end of chain)
        6. Return all memories in the chain

        Includes cycle detection to prevent infinite loops.

        Args:
            app_id: Application ID (collection name)
            memory_id: Starting memory ID to begin traversal

        Returns:
            List of memory dictionaries in evolution order (oldest to newest)
            Empty list if starting memory not found
        """
        result = []
        current_id = memory_id
        visited = set()

        logger.info(
            f"Starting memory evolution chain traversal from {memory_id} in {app_id}"
        )

        while current_id is not None:
            if current_id in visited:
                logger.warning(
                    f"Cycle detected in memory evolution chain at {current_id}. "
                    f"Stopping traversal to prevent infinite loop."
                )
                break

            memory = await self.get_memory(
                memory_id=current_id,
                app_id=app_id,
            )

            if memory is None:
                logger.warning(
                    f"Memory {current_id} not found in {app_id}. "
                    f"End of chain or memory deleted."
                )
                break

            result.append(memory)
            visited.add(current_id)

            metadata = memory.get("metadata", {})
            next_id = metadata.get("next_id")

            logger.debug(f"Memory {current_id} added to chain. next_id: {next_id}")

            current_id = next_id

        logger.info(
            f"Memory evolution chain traversal completed. "
            f"Found {len(result)} memories in chain."
        )

        return result

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
        if not chain:
            return ""

        if format == "mermaid":
            return self._generate_mermaid_graph(chain)
        elif format == "dot":
            return self._generate_dot_graph(chain)
        elif format == "html":
            return self._generate_html_graph(chain)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'mermaid', 'dot', or 'html'"
            )

    def _generate_mermaid_graph(self, chain: List[Dict[str, Any]]) -> str:
        """Generate Mermaid diagram syntax showing memory evolution chain."""
        lines = ["graph LR"]
        lines.append("    %% Memory Evolution Chain - Forward Linked List")
        lines.append("    %% Direction: Oldest (left) → Newest (right)")
        lines.append("")

        lines.append(
            "    classDef activeStyle fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000"
        )
        lines.append(
            "    classDef updatedStyle fill:#cce5ff,stroke:#007bff,stroke-width:2px,color:#000"
        )
        lines.append("")

        active_nodes = []
        updated_nodes = []

        for i, memory in enumerate(chain):
            mem_id = memory.get("memory_id", "unknown")
            metadata = memory.get("metadata", {})
            status = metadata.get("status", "active")
            created_at = metadata.get("created_at", "")
            next_id = metadata.get("next_id")

            if len(mem_id) > 16:
                display_id = f"{mem_id[:8]}...{mem_id[-4:]}"
            else:
                display_id = mem_id

            date_str = ""
            if created_at:
                try:
                    if "T" in str(created_at):
                        date_str = str(created_at).split("T")[0]
                    else:
                        date_str = str(created_at)[:10]
                except:
                    date_str = ""

            status_emoji = {"active": "✅", "updated": "🔄", "deleted": "❌"}.get(
                status, "📝"
            )

            order_num = i + 1
            label_parts = [f"{order_num}. {status_emoji} {status}"]
            label_parts.append(display_id)
            if date_str:
                label_parts.append(date_str)
            label = "\\n".join(label_parts)

            node_id = f"M{i}"
            lines.append(f'    {node_id}["{label}"]')

            if status == "active":
                active_nodes.append(node_id)
            elif status == "updated":
                updated_nodes.append(node_id)

            if i < len(chain) - 1:
                next_node_id = f"M{i + 1}"
                lines.append(f"    {node_id} -->|evolves to| {next_node_id}")

        lines.append("")
        if active_nodes:
            lines.append(f"    class {','.join(active_nodes)} activeStyle")
        if updated_nodes:
            lines.append(f"    class {','.join(updated_nodes)} updatedStyle")

        return "\n".join(lines)

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
        if not chain:
            return ""

        if format == "markdown":
            return self._generate_markdown_report(chain)
        elif format == "text":
            return self._generate_text_report(chain)
        elif format == "json":
            return self._generate_json_report(chain)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Use 'markdown', 'text', or 'json'"
            )

    def _generate_markdown_report(self, chain: List[Dict[str, Any]]) -> str:
        """Generate Markdown evolution report."""
        lines = ["# Memory Evolution Chain Report", ""]

        total_memories = len(chain)
        active_count = sum(
            1 for m in chain if m.get("metadata", {}).get("status") == "active"
        )
        updated_count = sum(
            1 for m in chain if m.get("metadata", {}).get("status") == "updated"
        )

        dates = []
        for memory in chain:
            created_at = memory.get("metadata", {}).get("created_at", "")
            if created_at:
                try:
                    if "T" in str(created_at):
                        dates.append(str(created_at).split("T")[0])
                except:
                    pass

        time_span = ""
        if len(dates) >= 2:
            time_span = f"{dates[0]} to {dates[-1]}"
        elif len(dates) == 1:
            time_span = dates[0]

        lines.append("## Summary Statistics")
        lines.append("")
        lines.append(f"- **Total Memories in Chain:** {total_memories}")
        lines.append(f"- **Active Memories:** {active_count}")
        lines.append(f"- **Consolidated Memories:** {updated_count}")
        if time_span:
            lines.append(f"- **Time Span:** {time_span}")
        lines.append(f"- **Evolution Depth:** {total_memories - 1} consolidations")
        lines.append("")

        lines.append("## Evolution Timeline")
        lines.append("")
        for i, memory in enumerate(chain):
            mem_id = memory.get("memory_id", "unknown")
            metadata = memory.get("metadata", {})
            status = metadata.get("status", "active")
            created_at = metadata.get("created_at", "")
            updated_at = metadata.get("updated_at", "")
            next_id = metadata.get("next_id")

            display_id = f"{mem_id[:8]}...{mem_id[-4:]}" if len(mem_id) > 16 else mem_id

            created_str = ""
            updated_str = ""
            if created_at:
                try:
                    created_str = (
                        str(created_at).split("T")[0]
                        if "T" in str(created_at)
                        else str(created_at)[:10]
                    )
                except:
                    pass
            if updated_at:
                try:
                    updated_str = (
                        str(updated_at).split("T")[0]
                        if "T" in str(updated_at)
                        else str(updated_at)[:10]
                    )
                except:
                    pass

            status_emoji = {"active": "✅", "updated": "🔄", "deleted": "❌"}.get(
                status, "📝"
            )

            lines.append(f"### {i + 1}. Memory {display_id} ({status_emoji} {status})")
            lines.append("")
            if created_str:
                lines.append(f"- **Created:** {created_str}")
            if updated_str and updated_str != created_str:
                lines.append(f"- **Updated:** {updated_str}")
            if next_id:
                next_display = (
                    f"{next_id[:8]}...{next_id[-4:]}" if len(next_id) > 16 else next_id
                )
                lines.append(f"- **Evolves to:** {next_display}")
            else:
                lines.append("- **Evolves to:** *(End of chain)*")

            document = memory.get("document", "")
            if document:
                preview = document[:200] + "..." if len(document) > 200 else document
                lines.append(f"- **Preview:** {preview}")

            lines.append("")

        lines.append("## Content Evolution")
        lines.append("")
        all_tags = set()
        all_keywords = set()
        for memory in chain:
            metadata = memory.get("metadata", {})
            tags = metadata.get("tags", [])
            keywords = metadata.get("keywords", [])
            all_tags.update(tags)
            all_keywords.update(keywords)

        if all_tags:
            lines.append(f"**All Tags:** {', '.join(sorted(all_tags))}")
            lines.append("")
        if all_keywords:
            lines.append(f"**All Keywords:** {', '.join(sorted(all_keywords))}")
            lines.append("")

        lines.append("## Insights")
        lines.append("")
        if total_memories > 1:
            lines.append(
                f"- This memory has evolved through **{total_memories - 1} consolidation(s)**"
            )
            lines.append(
                f"- The chain represents the progressive refinement and enrichment of the original memory"
            )
            if updated_count > 0:
                lines.append(
                    f"- **{updated_count} memory(ies)** were consolidated into newer versions"
                )
            if active_count > 0:
                lines.append(
                    f"- **{active_count} memory(ies)** are currently active (most recent)"
                )
        else:
            lines.append("- This is a standalone memory with no evolution history")
        lines.append("")

        return "\n".join(lines)

    def _generate_text_report(self, chain: List[Dict[str, Any]]) -> str:
        """Generate plain text evolution report."""
        lines = ["=" * 60]
        lines.append("MEMORY EVOLUTION CHAIN REPORT")
        lines.append("=" * 60)
        lines.append("")

        total_memories = len(chain)
        active_count = sum(
            1 for m in chain if m.get("metadata", {}).get("status") == "active"
        )
        updated_count = sum(
            1 for m in chain if m.get("metadata", {}).get("status") == "updated"
        )

        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 60)
        lines.append(f"Total Memories in Chain: {total_memories}")
        lines.append(f"Active Memories: {active_count}")
        lines.append(f"Consolidated Memories: {updated_count}")
        lines.append(f"Evolution Depth: {total_memories - 1} consolidations")
        lines.append("")

        lines.append("EVOLUTION TIMELINE")
        lines.append("-" * 60)
        for i, memory in enumerate(chain):
            mem_id = memory.get("memory_id", "unknown")
            metadata = memory.get("metadata", {})
            status = metadata.get("status", "active")
            created_at = metadata.get("created_at", "")
            next_id = metadata.get("next_id", "")

            display_id = f"{mem_id[:8]}...{mem_id[-4:]}" if len(mem_id) > 16 else mem_id
            created_str = ""
            if created_at:
                try:
                    created_str = (
                        str(created_at).split("T")[0]
                        if "T" in str(created_at)
                        else str(created_at)[:10]
                    )
                except:
                    pass

            lines.append(f"{i + 1}. {display_id} ({status})")
            if created_str:
                lines.append(f"   Created: {created_str}")
            if next_id:
                next_display = (
                    f"{next_id[:8]}...{next_id[-4:]}" if len(next_id) > 16 else next_id
                )
                lines.append(f"   Evolves to: {next_display}")
            else:
                lines.append("   Evolves to: (End of chain)")
            lines.append("")

        return "\n".join(lines)

    def _generate_json_report(self, chain: List[Dict[str, Any]]) -> str:
        """Generate JSON evolution report."""
        import json

        report = {
            "summary": {
                "total_memories": len(chain),
                "active_count": sum(
                    1 for m in chain if m.get("metadata", {}).get("status") == "active"
                ),
                "updated_count": sum(
                    1 for m in chain if m.get("metadata", {}).get("status") == "updated"
                ),
                "evolution_depth": len(chain) - 1,
            },
            "timeline": [],
            "content_evolution": {
                "all_tags": [],
                "all_keywords": [],
            },
        }

        all_tags = set()
        all_keywords = set()

        for i, memory in enumerate(chain):
            mem_id = memory.get("memory_id", "unknown")
            metadata = memory.get("metadata", {})

            timeline_entry = {
                "order": i + 1,
                "memory_id": mem_id,
                "status": metadata.get("status", "active"),
                "created_at": metadata.get("created_at", ""),
                "updated_at": metadata.get("updated_at", ""),
                "next_id": metadata.get("next_id"),
            }
            report["timeline"].append(timeline_entry)

            tags = metadata.get("tags", [])
            keywords = metadata.get("keywords", [])
            all_tags.update(tags)
            all_keywords.update(keywords)

        report["content_evolution"]["all_tags"] = sorted(list(all_tags))
        report["content_evolution"]["all_keywords"] = sorted(list(all_keywords))

        return json.dumps(report, indent=2, default=str)

    def _generate_dot_graph(self, chain: List[Dict[str, Any]]) -> str:
        """Generate Graphviz DOT format showing memory evolution chain."""
        lines = [
            "digraph MemoryEvolution {",
            "    rankdir=LR;",
            "    node [shape=box, style=rounded];",
            "    // Memory Evolution Chain - Forward Linked List",
            "    // Direction: Oldest (left) → Newest (right)",
            "",
        ]

        for i, memory in enumerate(chain):
            mem_id = memory.get("memory_id", "unknown")
            metadata = memory.get("metadata", {})
            status = metadata.get("status", "active")
            created_at = metadata.get("created_at", "")

            if len(mem_id) > 16:
                display_id = f"{mem_id[:8]}...{mem_id[-4:]}"
            else:
                display_id = mem_id

            date_str = ""
            if created_at:
                try:
                    if "T" in str(created_at):
                        date_str = str(created_at).split("T")[0]
                    else:
                        date_str = str(created_at)[:10]
                except:
                    date_str = ""

            order_num = i + 1
            label_parts = [f"{order_num}. {status}", display_id]
            if date_str:
                label_parts.append(date_str)
            label = "\\n".join(label_parts)

            sanitized_id = mem_id.replace("-", "")
            if sanitized_id[0].isdigit():
                node_id = f"M_{sanitized_id[:12]}"
            else:
                node_id = f"M{sanitized_id[:12]}"

            if status == "active":
                lines.append(
                    f'    {node_id} [label="{label}", fillcolor="#d4edda", style="filled,rounded"];'
                )
            elif status == "updated":
                lines.append(
                    f'    {node_id} [label="{label}", fillcolor="#cce5ff", style="filled,rounded"];'
                )
            else:
                lines.append(f'    {node_id} [label="{label}"];')

            if i < len(chain) - 1:
                next_mem_id = chain[i + 1].get("memory_id", "unknown")
                next_sanitized = next_mem_id.replace("-", "")
                if next_sanitized[0].isdigit():
                    next_node_id = f"M_{next_sanitized[:12]}"
                else:
                    next_node_id = f"M{next_sanitized[:12]}"
                lines.append(f'    {node_id} -> {next_node_id} [label="evolves to"];')

        lines.append("}")
        return "\n".join(lines)

    def _generate_html_graph(self, chain: List[Dict[str, Any]]) -> str:
        """Generate HTML file with embedded Mermaid.js visualization."""
        mermaid_code = self._generate_mermaid_graph(chain)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Evolution Chain</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        // Instructions for viewing other formats
        console.log("📊 Graph Viewing Instructions:");
        console.log("Mermaid: Copy the code below and paste at https://mermaid.live");
        console.log("DOT: Copy the DOT code and paste at https://dreampuf.github.io/GraphvizOnline/");
    </script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background:
            color:
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background:
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color:
            margin-top: 0;
            border-bottom: 2px solid
            padding-bottom: 10px;
        }}
        .info {{
            background:
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            border-left: 4px solid
        }}
        .mermaid {{
            background: white;
            padding: 20px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔗 Memory Evolution Chain</h1>
        <div class="info">
            <strong>Chain Length:</strong> {len(chain)} memories<br>
            <strong>Format:</strong> Forward-linked list (oldest → newest)
        </div>
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>"""
        return html

    async def delete_memory(
        self,
        doc_id: str,
        collection_name: str,
    ) -> MemoryOperationResult:
        """Delete a memory from collection with flexible filtering (async version).

        Args:
            doc_id: Document ID to delete
            collection_name: Name of the collection
        Returns:
            MemoryOperationResult with success status and error details
        """
        metrics = get_metrics_collector()
        with metrics.operation_timer("write", "delete_memory") as timer:
            if not doc_id:
                logger.error("doc_id is required for memory deletion")
                result = MemoryOperationResult.error_result(
                    error_code="INVALID_INPUT",
                    error_message="doc_id is required for memory deletion",
                    collection_name=collection_name,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            if not collection_name:
                logger.error("collection_name is required for memory deletion")
                result = MemoryOperationResult.error_result(
                    error_code="INVALID_INPUT",
                    error_message="collection_name is required for memory deletion",
                    memory_id=doc_id,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            try:
                async with self._get_pooled_handler() as vector_db_handler:
                    if not vector_db_handler or not vector_db_handler.enabled:
                        logger.warning(
                            "Vector DB is not enabled. Cannot delete memory."
                        )
                        result = MemoryOperationResult.error_result(
                            error_code="VECTOR_DB_DISABLED",
                            error_message="Vector database handler not available or disabled",
                            memory_id=doc_id,
                            collection_name=collection_name,
                        )
                        timer.success = False
                        timer.error_code = result.error_code
                        return result

                    success = await vector_db_handler.delete_from_collection(
                        collection_name=collection_name,
                        doc_id=doc_id,
                    )

                if success:
                    logger.info(
                        f"Successfully deleted memory {doc_id} from {collection_name}"
                    )
                    result = MemoryOperationResult.success_result(
                        memory_id=doc_id,
                        collection_name=collection_name,
                        operation="delete_memory",
                    )
                    timer.success = True
                    return result
                else:
                    result = MemoryOperationResult.error_result(
                        error_code="DELETE_FAILED",
                        error_message=f"Failed to delete memory {doc_id} from collection",
                        memory_id=doc_id,
                        collection_name=collection_name,
                    )
                    timer.success = False
                    timer.error_code = result.error_code
                    return result
            except Exception as e:
                logger.error(f"Error deleting memory {doc_id}: {e}", exc_info=True)
                result = MemoryOperationResult.error_result(
                    error_code="DELETE_EXCEPTION",
                    error_message=f"Exception while deleting memory: {str(e)}",
                    memory_id=doc_id,
                    collection_name=collection_name,
                    exception_type=type(e).__name__,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

    async def generate_memory_links(
        self,
        embedding: List[float],
        app_id: str,
        user_id: str = None,
        session_id: str = None,
        max_links: int = None,
        similarity_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate semantic links to related existing memories (async version).

        Before storing a new memory, this method finds semantically similar existing
        memories and creates "links" for memory networking. Used by conflict resolution
        system to determine if agent intervention is needed.

        Uses multiplicative approach: composite = relevance × (1 + recency_boost + importance_boost)

        Args:
            embedding: The embedding vector of the new memory
            app_id: Application ID (collection name)
            user_id: Optional user ID filter for personalized links
            session_id: Optional session ID filter for personalized links
            max_links: Maximum number of links to generate (default: 10)
            similarity_threshold: similarity threshold for linking (default: LINK_THRESHOLD = 0.7)
        Returns:
            List of memory link objects with composite scores for threshold filtering
        """
        metrics = get_metrics_collector()
        with metrics.operation_timer("query", "generate_memory_links") as timer:
            filter_conditions = {"app_id": app_id, "status": "active"}
            if user_id:
                filter_conditions["user_id"] = user_id
            if session_id:
                filter_conditions["session_id"] = session_id
            collection_name = app_id

            try:
                if not max_links:
                    max_links = DEFAULT_N_RESULTS
                n_results = max_links * 2

                async with self._get_pooled_handler() as vector_db_handler:
                    if not vector_db_handler or not vector_db_handler.enabled:
                        logger.warning(
                            "Vector database handler not available for link generation"
                        )
                        timer.success = False
                        return []

                    candidates = await vector_db_handler.query_by_embedding(
                        collection_name=collection_name,
                        embedding=embedding,
                        n_results=n_results,
                        filter_conditions=filter_conditions,
                        similarity_threshold=similarity_threshold,
                    )

                if not candidates.get("documents"):
                    timer.success = True
                    timer.results_count = 0
                    return []

                memory_links = []
                for i, (doc, score, metadata) in enumerate(
                    zip(
                        candidates.get("documents", []),
                        candidates.get("scores", []),
                        candidates.get("metadatas", []),
                    )
                ):
                    logger.info(
                        f"Processing candidate {i} with score {score} and metadata {metadata}"
                    )
                    if score >= similarity_threshold:
                        composite_score = calculate_composite_score(
                            semantic_score=score, metadata=metadata
                        )
                        memory_link = {
                            "memory_id": metadata.get("document_id"),
                            "similarity_score": round(score, 3),
                            "composite_score": round(composite_score, 3),
                            "link_strength": round(composite_score, 3),
                            "relationship_type": determine_relationship_type(
                                composite_score, metadata
                            ),
                            "document": doc,
                            "retrieval_tags": metadata.get("retrieval_tags", []),
                            "retrieval_keywords": metadata.get(
                                "retrieval_keywords", []
                            ),
                            "semantic_queries": metadata.get("semantic_queries", []),
                            "conversation_complexity": metadata.get(
                                "conversation_complexity", 1
                            ),
                            "interaction_quality": metadata.get(
                                "interaction_quality", "low"
                            ),
                            "follow_up_potential": metadata.get(
                                "follow_up_potential", []
                            ),
                            "created_at": metadata.get("created_at"),
                            "updated_at": metadata.get("updated_at"),
                            "status": metadata.get("status", "active"),
                            "next_id": metadata.get("next_id"),
                        }
                        memory_links.append(memory_link)

                memory_links.sort(
                    key=lambda x: x.get("composite_score", 0.0), reverse=True
                )

                timer.success = True
                timer.results_count = len(memory_links)
                return memory_links
            except Exception as e:
                logger.error(f"Error generating memory links: {e}", exc_info=True)
                timer.success = False
                timer.error_code = type(e).__name__
                return []

    async def embed_memory_note(self, memory_note_text: str) -> List[float]:
        """
        Embed a memory note text and return the embedding vector (async version).

        This method handles the embedding of combined memory notes that have been
        created by merging episodic and summarizer memories.

        Args:
            memory_note_text: The natural memory note text to embed

        Returns:
            List[float]: The embedding vector

        Raises:
            RuntimeError: If embedding fails after retries and chunking
        """
        if not memory_note_text or not isinstance(memory_note_text, str):
            raise ValueError("Memory note text must be a non-empty string")

        if not memory_note_text.strip():
            raise ValueError("Memory note text cannot be empty or whitespace only")

        logger.info(f"Embedding memory note: {len(memory_note_text)} characters")

        async with self._get_pooled_handler() as handler:
            if not handler or not handler.enabled:
                raise RuntimeError(
                    "Vector database handler is not available or not enabled"
                )

            try:
                embedding = await handler._embed_text_with_chunking_async(
                    memory_note_text
                )
                logger.info(
                    f"Successfully embedded memory note into {len(embedding)}-dimensional vector using token-based chunking"
                )
                return embedding
            except Exception as e:
                logger.error(f"Failed to embed memory note with chunking: {e}")
                raise RuntimeError(f"Memory note embedding failed: {e}")

    async def store_memory_note(
        self,
        doc_id: str,
        app_id: str,
        user_id: str,
        memory_note_text: str,
        embedding: List[float],
        session_id: str = None,
        tags: List[str] = None,
        keywords: List[str] = None,
        semantic_queries: List[str] = None,
        conversation_complexity: int = None,
        interaction_quality: str = None,
        follow_up_potential: List[str] = None,
        status: str = "active",
    ) -> MemoryOperationResult:
        """
        Store a complete memory note with embedding and comprehensive metadata (async version).

        The memory note text is stored as the document content and will be available
        as 'text' field in the vector database payload for retrieval.

        Args:
            doc_id: Document ID
            app_id: Application ID (used as collection name)
            user_id: User ID
            session_id: Session ID (optional)
            memory_note_text: The full memory note text (stored as document)
            embedding: The embedding vector
            tags: Optional list of behavioral/content tags
            keywords: Optional list of keywords
            semantic_queries: Optional list of semantic query suggestions
            conversation_complexity: Optional complexity score (1-5)
            interaction_quality: Optional quality rating
            follow_up_potential: Optional list of follow-up topics
            status: Status of the memory (default: "active")

        Returns:
            MemoryOperationResult with success status and error details
        """
        metrics = get_metrics_collector()
        with metrics.operation_timer("write", "store_memory_note") as timer:
            if not app_id or not isinstance(app_id, str):
                logger.error("Valid app_id is required")
                result = MemoryOperationResult.error_result(
                    error_code="INVALID_INPUT",
                    error_message="Valid app_id is required",
                    memory_id=doc_id,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            if not user_id or not isinstance(user_id, str):
                logger.error("Valid user_id is required")
                result = MemoryOperationResult.error_result(
                    error_code="INVALID_INPUT",
                    error_message="Valid user_id is required",
                    memory_id=doc_id,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            if not memory_note_text or not isinstance(memory_note_text, str):
                logger.error("Valid memory_note_text is required")
                result = MemoryOperationResult.error_result(
                    error_code="INVALID_INPUT",
                    error_message="Valid memory_note_text is required",
                    memory_id=doc_id,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
                logger.error("Valid embedding vector is required")
                result = MemoryOperationResult.error_result(
                    error_code="INVALID_INPUT",
                    error_message="Valid embedding vector is required",
                    memory_id=doc_id,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            collection_name = app_id

            created_at = datetime.now(timezone.utc).isoformat()

            metadata = {
                "document_id": doc_id,
                "app_id": app_id,
                "user_id": user_id,
                "session_id": session_id if session_id else "none",
                "created_at": created_at,
                "updated_at": created_at,
                "embedding_dimensions": len(embedding),
                "status": status,
                "status_reason": "created",
                "next_id": None,
            }

            if tags:
                metadata["tags"] = tags
            if keywords:
                metadata["keywords"] = keywords
            if semantic_queries:
                metadata["semantic_queries"] = semantic_queries
            if conversation_complexity is not None:
                metadata["conversation_complexity"] = conversation_complexity
            if interaction_quality:
                metadata["interaction_quality"] = interaction_quality
            if follow_up_potential:
                metadata["follow_up_potential"] = follow_up_potential

            logger.info(
                f"Storing memory note for app_id={app_id}, user_id={user_id}, "
                f"session_id={session_id if session_id else 'none'} in collection '{collection_name}'"
            )

            try:
                async with self._get_pooled_handler() as vector_db_handler:
                    if not vector_db_handler or not vector_db_handler.enabled:
                        logger.error(
                            "Vector database handler is not available or not enabled"
                        )
                        result = MemoryOperationResult.error_result(
                            error_code="VECTOR_DB_DISABLED",
                            error_message="Vector database handler not available or disabled",
                            memory_id=doc_id,
                            collection_name=app_id,
                        )
                        timer.success = False
                        timer.error_code = result.error_code
                        return result

                    success = await vector_db_handler.add_to_collection(
                        collection_name=collection_name,
                        doc_id=doc_id,
                        document=memory_note_text,
                        embedding=embedding,
                        metadata=metadata,
                    )

                if success:
                    logger.info(
                        f"Successfully stored memory note with {len(embedding)}-dimensional "
                        f"embedding and {len(metadata)} metadata fields"
                    )
                    result = MemoryOperationResult.success_result(
                        memory_id=doc_id,
                        collection_name=collection_name,
                        operation="store_memory_note",
                        embedding_dim=len(embedding),
                        metadata_fields=len(metadata),
                    )
                    timer.success = True
                    return result
                else:
                    logger.error("Failed to store memory note in vector database")
                    result = MemoryOperationResult.error_result(
                        error_code="STORE_FAILED",
                        error_message="Failed to store memory note in vector database",
                        memory_id=doc_id,
                        collection_name=collection_name,
                    )
                    timer.success = False
                    timer.error_code = result.error_code
                    return result

            except Exception as e:
                logger.error(f"Error storing memory note: {e}", exc_info=True)
                result = MemoryOperationResult.error_result(
                    error_code="STORE_EXCEPTION",
                    error_message=f"Exception while storing memory note: {str(e)}",
                    memory_id=doc_id,
                    collection_name=collection_name,
                    exception_type=type(e).__name__,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

    async def update_memory_status(
        self,
        app_id: str,
        doc_id: str,
        new_status: str,
        archive_reason: str = None,
        caused_by_memory: str = None,
        new_memory_id: str = None,
    ) -> MemoryOperationResult:
        """
        Update the status of an existing memory using the simple status+reason approach.

        Args:
            app_id: Application/collection ID
            doc_id: Document ID of the memory to update
            new_status: New status ("active", "updated", "deleted")
            archive_reason: Reason for archiving ("consolidated", "contradicted")
            caused_by_memory: ID of the memory that caused this status change
            new_memory_id: ID of the consolidated memory (for "updated" status)

        Returns:
            MemoryOperationResult with success status and error details
        """
        metrics = get_metrics_collector()
        with metrics.operation_timer("update", "update_memory_status") as timer:
            try:
                async with self._get_pooled_handler() as vector_db_handler:
                    if not vector_db_handler or not vector_db_handler.enabled:
                        logger.warning(
                            "Vector database handler not available for status update"
                        )
                        result = MemoryOperationResult.error_result(
                            error_code="VECTOR_DB_DISABLED",
                            error_message="Vector database handler not available or disabled",
                            memory_id=doc_id,
                            collection_name=app_id,
                        )
                        timer.success = False
                        timer.error_code = result.error_code
                        return result

                    current_memory = await vector_db_handler.query_by_id(
                        collection_name=app_id, doc_id=doc_id
                    )
                    if not current_memory:
                        result = MemoryOperationResult.error_result(
                            error_code="MEMORY_NOT_FOUND",
                            error_message=f"Memory {doc_id} not found",
                            memory_id=doc_id,
                            collection_name=app_id,
                        )
                        timer.success = False
                        timer.error_code = result.error_code
                        return result

                    reason = archive_reason
                    if not reason:
                        if caused_by_memory:
                            reason = STATUS_REASON_CONTRADICTED
                        elif new_status == "updated":
                            reason = STATUS_REASON_CONSOLIDATED
                        else:
                            reason = STATUS_REASON_MANUAL

                    update_payload = {
                        "status": new_status,
                        "status_reason": reason,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if new_memory_id:
                        update_payload["next_id"] = new_memory_id

                    success = await vector_db_handler.update_memory(
                        collection_name=app_id,
                        doc_id=doc_id,
                        update_payload=update_payload,
                    )
            except Exception as e:
                logger.error(
                    f"Error updating memory status for {doc_id}: {e}", exc_info=True
                )
                result = MemoryOperationResult.error_result(
                    error_code="UPDATE_STATUS_EXCEPTION",
                    error_message=f"Exception while updating memory status: {str(e)}",
                    memory_id=doc_id,
                    collection_name=app_id,
                    exception_type=type(e).__name__,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            if success:
                logger.info(
                    f"Updated memory {doc_id} status to {new_status} in {app_id}"
                )
                result = MemoryOperationResult.success_result(
                    memory_id=doc_id,
                    collection_name=app_id,
                    operation="update_memory_status",
                    new_status=new_status,
                    archive_reason=archive_reason,
                    new_memory_id=new_memory_id,
                )
                timer.success = True
                return result

            logger.error(f"Failed to update memory {doc_id} status in {app_id}")
            result = MemoryOperationResult.error_result(
                error_code="UPDATE_STATUS_FAILED",
                error_message=f"Failed to update memory {doc_id} status to {new_status}",
                memory_id=doc_id,
                collection_name=app_id,
                new_status=new_status,
            )
            timer.success = False
            timer.error_code = result.error_code
            return result

    async def update_memory_timestamp(
        self,
        app_id: str,
        doc_id: str,
        timestamp: datetime = None,
    ) -> MemoryOperationResult:
        """
        Update the timestamp of an existing memory (for SKIP operations) - simple approach.

        Args:
            app_id: Application/collection ID
            doc_id: Document ID of the memory to update
            timestamp: New timestamp (defaults to current time)

        Returns:
            MemoryOperationResult with success status and error details
        """
        metrics = get_metrics_collector()
        with metrics.operation_timer("update", "update_memory_timestamp") as timer:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            try:
                async with self._get_pooled_handler() as vector_db_handler:
                    if not vector_db_handler or not vector_db_handler.enabled:
                        logger.warning(
                            "Vector database handler not available for timestamp update"
                        )
                        result = MemoryOperationResult.error_result(
                            error_code="VECTOR_DB_DISABLED",
                            error_message="Vector database handler not available or disabled",
                            memory_id=doc_id,
                            collection_name=app_id,
                        )
                        timer.success = False
                        timer.error_code = result.error_code
                        return result

                    current_memory = await vector_db_handler.query_by_id(
                        collection_name=app_id, doc_id=doc_id
                    )
                    if not current_memory:
                        result = MemoryOperationResult.error_result(
                            error_code="MEMORY_NOT_FOUND",
                            error_message=f"Memory {doc_id} not found",
                            memory_id=doc_id,
                            collection_name=app_id,
                        )
                        timer.success = False
                        timer.error_code = result.error_code
                        return result

                    update_payload = {
                        "updated_at": timestamp.isoformat(),
                    }

                    success = await vector_db_handler.update_memory(
                        collection_name=app_id,
                        doc_id=doc_id,
                        update_payload=update_payload,
                    )
            except Exception as e:
                logger.error(
                    f"Error updating timestamp for {doc_id}: {e}", exc_info=True
                )
                result = MemoryOperationResult.error_result(
                    error_code="UPDATE_TIMESTAMP_EXCEPTION",
                    error_message=f"Exception while updating timestamp: {str(e)}",
                    memory_id=doc_id,
                    collection_name=app_id,
                    exception_type=type(e).__name__,
                )
                timer.success = False
                timer.error_code = result.error_code
                return result

            if success:
                result = MemoryOperationResult.success_result(
                    memory_id=doc_id,
                    collection_name=app_id,
                    operation="update_memory_timestamp",
                    timestamp=timestamp.isoformat(),
                )
                timer.success = True
                return result

            logger.error(f"Failed to update timestamp for memory {doc_id} in {app_id}")
            result = MemoryOperationResult.error_result(
                error_code="UPDATE_TIMESTAMP_FAILED",
                error_message=f"Failed to update timestamp for memory {doc_id}",
                memory_id=doc_id,
                collection_name=app_id,
            )
            timer.success = False
            timer.error_code = result.error_code
            return result

    async def _create_and_parse_memory_note(
        self, messages: str, llm_connection: Callable
    ) -> Tuple[Optional[MemoryNoteData], Optional[str]]:
        """
        Create memory note from messages and parse JSON structure.

        Args:
            messages: Formatted conversation messages to process
            llm_connection: LLM connection for memory creation

        Returns:
            Tuple of (parsed_memory_note_data, error_message)
            If successful: (MemoryNoteData, None)
            If failed: (None, error_message)
        """
        memory_note_json_str = await self.create_combined_memory(
            messages, llm_connection
        )
        if not memory_note_json_str:
            error_msg = "Failed to create memory note - LLM returned empty response"
            logger.error(error_msg)
            return None, error_msg

        try:
            raw_note = clean_and_parse_json(memory_note_json_str)
            memory_note_data = self._normalize_prepared_memory_note(raw_note)
            return memory_note_data, None
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"Failed to parse memory note JSON: {e}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg

    def _normalize_prepared_memory_note(self, memory_note: Any) -> MemoryNoteData:
        if not isinstance(memory_note, dict):
            raise ValueError("Prepared memory note must be an object")

        note_text = memory_note.get("text")
        if not note_text:
            raise ValueError("Prepared memory note missing 'text'")

        metadata = memory_note.get("metadata", {}) or {}

        structured_metadata = {
            "conversation_complexity": self._depth_to_complexity(
                metadata.get("content_depth")
            ),
            "interaction_quality": "high"
            if metadata.get("has_behavioral_data")
            else None,
            "follow_up_potential": metadata.get("follow_up_areas", []),
        }

        return MemoryNoteData(
            natural_memory_note=note_text,
            retrieval_tags=metadata.get("tags", []),
            retrieval_keywords=metadata.get("keywords", []),
            semantic_queries=metadata.get("query_hooks", []),
            conversation_complexity=structured_metadata["conversation_complexity"],
            interaction_quality=structured_metadata["interaction_quality"],
            follow_up_potential=structured_metadata["follow_up_potential"],
        )

    @staticmethod
    def _depth_to_complexity(depth: Optional[str]) -> Optional[int]:
        if not depth:
            return None
        mapping = {"low": 1, "medium": 2, "high": 3}
        return mapping.get(depth.lower())

    async def _find_meaningful_links(
        self,
        embedding: List[float],
        app_id: str,
        user_id: str,
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Find semantic links for a memory embedding (async version).

        Args:
            embedding: The embedding vector
            app_id: Application ID
            user_id: User ID
            session_id: Optional session ID

        Returns:
            List of meaningful links (above threshold)
        """
        memory_links = await self.generate_memory_links(
            embedding=embedding,
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            max_links=_MAX_EXPANDED_RESULTS,
            similarity_threshold=LINK_THRESHOLD,
        )

        meaningful_links = [
            link
            for link in memory_links
            if link.get("composite_score", 0.0) >= LINK_THRESHOLD
        ]

        logger.info(
            f"Found {len(memory_links)} total links, {len(meaningful_links)} above {LINK_THRESHOLD} threshold"
        )
        return meaningful_links

    def _extract_memory_metadata(
        self, memory_note_data: MemoryNoteData
    ) -> Dict[str, Any]:
        """
        Extract metadata from parsed memory note data.

        Args:
            memory_note_data: Parsed memory note data

        Returns:
            Dictionary with extracted metadata fields
        """
        retrieval_tags = memory_note_data.get("retrieval_tags", [])
        retrieval_keywords = memory_note_data.get("retrieval_keywords", [])
        semantic_queries = memory_note_data.get("semantic_queries", [])
        conversation_complexity = memory_note_data.get("conversation_complexity")
        interaction_quality = memory_note_data.get("interaction_quality")
        follow_up_potential = memory_note_data.get("follow_up_potential", [])

        return {
            "retrieval_tags": retrieval_tags,
            "retrieval_keywords": retrieval_keywords,
            "semantic_queries": semantic_queries,
            "conversation_complexity": conversation_complexity,
            "interaction_quality": interaction_quality,
            "follow_up_potential": follow_up_potential,
        }

    def _build_memory_data_dict(
        self,
        app_id: str,
        user_id: str,
        session_id: Optional[str],
        embedding: List[float],
        natural_memory_note: str,
        metadata: Dict[str, Any],
    ) -> MemoryDataDict:
        """
        Build memory data dictionary from components.

        Args:
            app_id: Application ID
            user_id: User ID
            session_id: Optional session ID
            embedding: Embedding vector
            natural_memory_note: Natural memory note text
            metadata: Extracted metadata dictionary

        Returns:
            MemoryDataDict with all fields
        """
        new_memory_doc_id = str(uuid.uuid4())
        return {
            "doc_id": new_memory_doc_id,
            "app_id": app_id,
            "user_id": user_id,
            "session_id": session_id,
            "embedding": embedding,
            "natural_memory_note": natural_memory_note,
            "retrieval_tags": metadata["retrieval_tags"],
            "retrieval_keywords": metadata["retrieval_keywords"],
            "semantic_queries": metadata["semantic_queries"],
            "conversation_complexity": metadata["conversation_complexity"],
            "interaction_quality": metadata["interaction_quality"],
            "follow_up_potential": metadata["follow_up_potential"],
            "status": "active",
        }

    async def _run_status_updates_chunked(
        self,
        update_specs: List[Dict[str, Any]],
        chunk_size: int = _DEFAULT_STATUS_CHUNK_SIZE,
    ) -> List[MemoryOperationResult]:
        """
        Execute status updates in bounded async batches to reduce latency.
        """
        results: List[MemoryOperationResult] = []
        for start in range(0, len(update_specs), chunk_size):
            chunk = update_specs[start : start + chunk_size]
            chunk_results = await asyncio.gather(
                *[self.update_memory_status(**spec) for spec in chunk],
                return_exceptions=True,
            )
            for spec, chunk_result in zip(chunk, chunk_results):
                if isinstance(chunk_result, Exception):
                    logger.error(
                        f"Status update chunk failed for {spec.get('doc_id')}: {chunk_result}",
                        exc_info=True,
                    )
                    results.append(
                        MemoryOperationResult.error_result(
                            error_code="UPDATE_STATUS_EXCEPTION",
                            error_message=str(chunk_result),
                            memory_id=spec.get("doc_id"),
                            collection_name=spec.get("app_id"),
                        )
                    )
                else:
                    results.append(chunk_result)
        return results

    async def _run_timestamp_updates_chunked(
        self,
        update_specs: List[Dict[str, Any]],
        chunk_size: int = _DEFAULT_STATUS_CHUNK_SIZE,
    ) -> List[MemoryOperationResult]:
        """
        Execute timestamp refreshes in bounded async batches.
        """
        results: List[MemoryOperationResult] = []
        for start in range(0, len(update_specs), chunk_size):
            chunk = update_specs[start : start + chunk_size]
            chunk_results = await asyncio.gather(
                *[self.update_memory_timestamp(**spec) for spec in chunk],
                return_exceptions=True,
            )
            for spec, chunk_result in zip(chunk, chunk_results):
                if isinstance(chunk_result, Exception):
                    logger.error(
                        f"Timestamp refresh failed for {spec.get('doc_id')}: {chunk_result}",
                        exc_info=True,
                    )
                    results.append(
                        MemoryOperationResult.error_result(
                            error_code="UPDATE_TIMESTAMP_EXCEPTION",
                            error_message=str(chunk_result),
                            memory_id=spec.get("doc_id"),
                            collection_name=spec.get("app_id"),
                        )
                    )
                else:
                    results.append(chunk_result)
        return results

    async def _store_memory_directly(
        self, memory_data: MemoryDataDict
    ) -> MemoryOperationResult:
        """
        Store memory directly without conflict resolution (fast path) - async version.

        Args:
            memory_data: Memory data dictionary

        Returns:
            MemoryOperationResult
        """
        logger.info("No meaningful links found - performing direct storage")
        return await self.store_memory_note(
            doc_id=memory_data["doc_id"],
            app_id=memory_data["app_id"],
            user_id=memory_data["user_id"],
            session_id=memory_data["session_id"],
            memory_note_text=memory_data["natural_memory_note"],
            embedding=memory_data["embedding"],
            tags=memory_data["retrieval_tags"],
            keywords=memory_data["retrieval_keywords"],
            semantic_queries=memory_data["semantic_queries"],
            conversation_complexity=memory_data["conversation_complexity"],
            interaction_quality=memory_data["interaction_quality"],
            follow_up_potential=memory_data["follow_up_potential"],
            status=memory_data["status"],
        )

    async def _execute_conflict_resolution(
        self,
        memory_data: MemoryDataDict,
        meaningful_links: List[Dict[str, Any]],
        app_id: str,
    ) -> bool:
        """
        Execute conflict resolution logic for memories with meaningful links (async version).

        Args:
            memory_data: New memory data dictionary
            meaningful_links: List of meaningful linked memories
            app_id: Application ID

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(
            f"Found {len(meaningful_links)} meaningful links - running conflict resolution"
        )

        new_memory_for_agent = {
            "natural_memory_note": memory_data["natural_memory_note"],
        }

        decisions = await self.conflict_resolution_agent.decide(
            new_memory=new_memory_for_agent, linked_memories=meaningful_links
        )

        logger.info(f"Agent made {len(decisions)} granular decisions")
        invalid_decisions: List[Dict[str, Any]] = []

        if not decisions:
            logger.warning("Agent returned no decisions, defaulting to SKIP all")
            for linked in meaningful_links:
                await self.update_memory_timestamp(
                    app_id=app_id, doc_id=linked["memory_id"]
                )
            return True, invalid_decisions

        valid_operations = {"UPDATE", "DELETE", "SKIP"}
        normalized_decisions: List[Dict[str, Any]] = []
        for decision in decisions:
            memory_id = decision.get("memory_id")
            op_raw = decision.get("operation")
            if not memory_id:
                invalid_decisions.append(
                    {
                        "operation": op_raw or "UNKNOWN",
                        "error": "missing_memory_id",
                        "decision": decision,
                    }
                )
                continue
            if not op_raw:
                invalid_decisions.append(
                    {
                        "memory_id": memory_id,
                        "error": "missing_operation",
                        "decision": decision,
                    }
                )
                continue
            op = op_raw.upper()
            if op not in valid_operations:
                invalid_decisions.append(
                    {
                        "memory_id": memory_id,
                        "operation": op,
                        "error": "unknown_operation",
                        "decision": decision,
                    }
                )
                continue
            normalized = dict(decision)
            normalized["operation"] = op
            normalized_decisions.append(normalized)

        if not normalized_decisions:
            logger.warning(
                "Conflict agent decisions were invalid, defaulting to SKIP timestamps only"
            )
            for linked in meaningful_links:
                await self.update_memory_timestamp(
                    app_id=app_id, doc_id=linked["memory_id"]
                )
            return True, invalid_decisions

        update_decisions = [
            d for d in normalized_decisions if d["operation"] == "UPDATE"
        ]
        delete_decisions = [
            d for d in normalized_decisions if d["operation"] == "DELETE"
        ]
        skip_decisions = [d for d in normalized_decisions if d["operation"] == "SKIP"]

        logger.info(
            f"Grouped decisions: UPDATE={len(update_decisions)}, DELETE={len(delete_decisions)}, SKIP={len(skip_decisions)}"
        )

        has_update = len(update_decisions) > 0
        has_delete = len(delete_decisions) > 0
        has_skip = len(skip_decisions) > 0

        if has_skip and not has_update and not has_delete:
            logger.info("All SKIP operations - refreshing timestamps only")
            batch_result = await self._execute_batch_skip(skip_decisions, app_id)
            return batch_result.success, invalid_decisions

        elif has_delete and not has_update and not has_skip:
            logger.info(
                "All DELETE operations - storing new memory and archiving contradicted ones"
            )
            batch_result = await self._execute_batch_delete(
                delete_decisions, memory_data
            )
            return batch_result.success, invalid_decisions

        elif has_update and not has_delete and not has_skip:
            logger.info("All UPDATE operations - synthesizing consolidated memory")
            batch_result = await self._execute_batch_update(
                update_decisions, memory_data, meaningful_links
            )
            return batch_result.success, invalid_decisions

        else:
            logger.info("Mixed operations detected - applying smart execution logic")

            if has_update:
                logger.info("Processing UPDATE operations (highest priority)")
                batch_result = await self._execute_batch_update(
                    update_decisions, memory_data, meaningful_links
                )
                if not batch_result.success:
                    logger.error("UPDATE operations failed, aborting")
                    return False, invalid_decisions

            if has_delete and not has_update:
                logger.info(
                    "Processing DELETE operations (UPDATE not present, so we add new memory)"
                )
                batch_result = await self._execute_batch_delete(
                    delete_decisions, memory_data
                )
                if not batch_result.success:
                    logger.error("DELETE operations failed")
                    return False, invalid_decisions

            if has_skip:
                logger.info("Processing SKIP operations (timestamp refresh)")
                batch_result = await self._execute_batch_skip(skip_decisions, app_id)
            success = batch_result.success

        if invalid_decisions:
            logger.warning(
                f"Conflict agent returned {len(invalid_decisions)} invalid decisions: {invalid_decisions}"
            )
        return success, invalid_decisions

    async def create_and_store_memory(
        self,
        app_id: str,
        user_id: str,
        messages: str,
        llm_connection: Callable,
        session_id: Optional[str] = None,
    ) -> MemoryOperationResult:
        """
        Complete memory processing pipeline: create → embed → store.

        This method orchestrates the full memory lifecycle using smaller, testable helper methods:
        1. Creates and parses memory note from conversation messages
        2. Embeds the natural_memory_note text into a vector
        3. Finds semantic links to existing memories
        4. Extracts metadata and builds memory data structure
        5. Stores memory (directly or via conflict resolution)

        Args:
            app_id: Application ID (used as collection name)
            user_id: User ID
            session_id: Session ID (optional)
            messages: Formatted conversation messages to process
            llm_connection: LLM connection for memory creation

        Returns:
            MemoryOperationResult with success status, memory_id, and error details
        """
        logger.info(
            f"Starting complete memory processing for app_id={app_id}, "
            f"user_id={user_id}, session_id={session_id if session_id else 'none'}"
        )

        try:
            memory_note_data, error = await self._create_and_parse_memory_note(
                messages, llm_connection
            )
            if error or not memory_note_data:
                return MemoryOperationResult.error_result(
                    error_code="MEMORY_NOTE_CREATION_FAILED",
                    error_message=error or "Failed to create memory note",
                    collection_name=app_id,
                )

            natural_memory_note = memory_note_data["natural_memory_note"]
            try:
                embedding = await self.embed_memory_note(natural_memory_note)
                if not embedding:
                    return MemoryOperationResult.error_result(
                        error_code="EMBEDDING_FAILED",
                        error_message="Failed to embed memory note",
                        collection_name=app_id,
                    )
            except Exception as e:
                return MemoryOperationResult.error_result(
                    error_code="EMBEDDING_EXCEPTION",
                    error_message=f"Embedding exception: {str(e)}",
                    collection_name=app_id,
                )

            meaningful_links = await self._find_meaningful_links(
                embedding=embedding,
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
            )

            metadata = self._extract_memory_metadata(memory_note_data)
            memory_data = self._build_memory_data_dict(
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                embedding=embedding,
                natural_memory_note=natural_memory_note,
                metadata=metadata,
            )

            if len(meaningful_links) == 0:
                store_result = await self._store_memory_directly(memory_data)
                if store_result.success:
                    logger.info(
                        f"Direct storage completed for doc_id={memory_data['doc_id']}"
                    )
                return store_result
            else:
                meaningful_links_sorted = sorted(
                    meaningful_links,
                    key=lambda x: x.get("composite_score", 0.0),
                    reverse=True,
                )

                limited_links = meaningful_links_sorted[:MAX_LINKS_FOR_SYNTHESIS]
                if len(meaningful_links_sorted) > MAX_LINKS_FOR_SYNTHESIS:
                    logger.info(
                        f"Limiting meaningful_links from {len(meaningful_links_sorted)} to {MAX_LINKS_FOR_SYNTHESIS} "
                        f"to avoid bloating LLM context window (keeping top {MAX_LINKS_FOR_SYNTHESIS} by composite_score)"
                    )

                success, invalid_decisions = await self._execute_conflict_resolution(
                    memory_data=memory_data,
                    meaningful_links=limited_links,
                    app_id=app_id,
                )
                if success:
                    return MemoryOperationResult.success_result(
                        memory_id=memory_data["doc_id"],
                        collection_name=app_id,
                        links_processed=len(meaningful_links),
                        invalid_conflict_decisions=invalid_decisions,
                    )
                else:
                    return MemoryOperationResult.error_result(
                        error_code="CONFLICT_RESOLUTION_FAILED",
                        error_message="Conflict resolution processing failed",
                        collection_name=app_id,
                        memory_id=memory_data["doc_id"],
                        invalid_conflict_decisions=invalid_decisions,
                    )

        except Exception as e:
            logger.error(
                f"Error in complete memory processing pipeline: {e}", exc_info=True
            )
            return MemoryOperationResult.error_result(
                error_code="PIPELINE_EXCEPTION",
                error_message=f"Unexpected error: {str(e)}",
                collection_name=app_id,
                exception_type=type(e).__name__,
            )

    async def _execute_batch_update(
        self,
        update_decisions: List[Dict[str, Any]],
        new_memory_data: Dict[str, Any],
        meaningful_links: List[Dict[str, Any]],
    ) -> BatchOperationResult:
        """Execute batch UPDATE operations: consolidate memories into new synthesized memory.

        NOTE: Does NOT store the original new_memory_data directly.
        Instead, synthesizes all UPDATE targets + new memory into one consolidated memory.
        """
        import time

        metrics = get_metrics_collector()
        start_time = time.time()
        logger.info(f"Executing batch UPDATE for {len(update_decisions)} decisions")

        if not update_decisions:
            return BatchOperationResult(
                success=True,
                total_items=0,
                succeeded=0,
                failed=0,
                failed_items=[],
            )

        try:
            memory_ids_to_consolidate = [d["memory_id"] for d in update_decisions]

            meaningful_links_sorted = sorted(
                meaningful_links,
                key=lambda x: x.get("composite_score", 0.0),
                reverse=True,
            )

            limited_links = meaningful_links_sorted[:MAX_LINKS_FOR_SYNTHESIS]
            if len(meaningful_links_sorted) > MAX_LINKS_FOR_SYNTHESIS:
                logger.info(
                    f"Limiting meaningful_links from {len(meaningful_links_sorted)} to {MAX_LINKS_FOR_SYNTHESIS} "
                    f"for batch UPDATE to avoid bloating LLM context window (keeping top {MAX_LINKS_FOR_SYNTHESIS} by composite_score)"
                )
                limited_memory_ids = set(link["memory_id"] for link in limited_links)
                memory_ids_to_consolidate = [
                    mid
                    for mid in memory_ids_to_consolidate
                    if mid in limited_memory_ids
                ]

            existing_memories_for_synthesis = [
                {"natural_memory_note": link["document"]}
                for link in limited_links
                if link["memory_id"] in memory_ids_to_consolidate
            ]

            if not existing_memories_for_synthesis:
                logger.error(
                    "No existing memories found in meaningful_links for batch UPDATE"
                )
                return BatchOperationResult(
                    success=False,
                    total_items=len(update_decisions),
                    succeeded=0,
                    failed=len(update_decisions),
                    failed_items=[
                        {
                            "memory_id": d["memory_id"],
                            "error_code": "NO_EXISTING_MEMORIES",
                            "error_message": "No existing memories found in meaningful_links",
                        }
                        for d in update_decisions
                    ],
                    error_code="NO_EXISTING_MEMORIES",
                    error_message="No existing memories found in meaningful_links for batch UPDATE",
                )

            synthesis_result = await self.synthesis_agent.consolidate_memories(
                new_memory={
                    "natural_memory_note": new_memory_data["natural_memory_note"]
                },
                existing_memories=existing_memories_for_synthesis,
            )

            consolidated_memory = synthesis_result["consolidated_memory"]
            consolidated_note_text = consolidated_memory["natural_memory_note"]
            logger.info(f"Consolidated note text: {consolidated_note_text}")
            logger.info(f"Synthesis completed: {synthesis_result['synthesis_summary']}")

            all_tags = [new_memory_data.get("retrieval_tags", [])]
            all_keywords = [new_memory_data.get("retrieval_keywords", [])]
            all_semantic_queries = [new_memory_data.get("semantic_queries", [])]
            all_follow_ups = [new_memory_data.get("follow_up_potential", [])]

            for link in meaningful_links:
                if link["memory_id"] in memory_ids_to_consolidate:
                    all_tags.append(link.get("retrieval_tags", []))
                    all_keywords.append(link.get("retrieval_keywords", []))
                    all_semantic_queries.append(link.get("semantic_queries", []))
                    all_follow_ups.append(link.get("follow_up_potential", []))

            strict_unioned = {
                "tags": list(set(item for sublist in all_tags for item in sublist)),
                "keywords": list(
                    set(item for sublist in all_keywords for item in sublist)
                ),
                "semantic_queries": list(
                    set(item for sublist in all_semantic_queries for item in sublist)
                ),
                "follow_up_potential": list(
                    set(item for sublist in all_follow_ups for item in sublist)
                ),
            }

            fuzzy_unioned = {
                "tags": fuzzy_dedup(strict_unioned["tags"]),
                "keywords": fuzzy_dedup(strict_unioned["keywords"]),
                "semantic_queries": fuzzy_dedup(strict_unioned["semantic_queries"]),
                "follow_up_potential": fuzzy_dedup(
                    strict_unioned["follow_up_potential"]
                ),
            }

            def _safe_complexity(value: Optional[Any]) -> int:
                if isinstance(value, (int, float)):
                    return int(value)
                return 1

            def _safe_quality(value: Optional[str]) -> str:
                if isinstance(value, str) and value.lower() in {
                    "low",
                    "medium",
                    "high",
                }:
                    return value.lower()
                return "low"

            all_complexities = [
                _safe_complexity(new_memory_data.get("conversation_complexity"))
            ]
            all_qualities = [_safe_quality(new_memory_data.get("interaction_quality"))]
            for link in meaningful_links:
                if link["memory_id"] in memory_ids_to_consolidate:
                    all_complexities.append(
                        _safe_complexity(link.get("conversation_complexity"))
                    )
                    all_qualities.append(_safe_quality(link.get("interaction_quality")))

            consolidated_complexity = max(all_complexities)
            quality_hierarchy = {"high": 3, "medium": 2, "low": 1}
            consolidated_quality = max(
                all_qualities, key=lambda q: quality_hierarchy.get(q, 0)
            )

            consolidated_embedding = await self.embed_memory_note(
                consolidated_note_text
            )
            if not consolidated_embedding:
                logger.error("Failed to embed consolidated memory")
                duration = time.time() - start_time
                metrics.record_batch(
                    operation="batch_update",
                    duration=duration,
                    total_items=len(update_decisions),
                    succeeded=0,
                    failed=len(update_decisions),
                    error_code="EMBEDDING_FAILED",
                )
                return BatchOperationResult(
                    success=False,
                    total_items=len(update_decisions),
                    succeeded=0,
                    failed=len(update_decisions),
                    failed_items=[
                        {
                            "memory_id": d["memory_id"],
                            "error_code": "EMBEDDING_FAILED",
                            "error_message": "Failed to embed consolidated memory",
                        }
                        for d in update_decisions
                    ],
                    error_code="EMBEDDING_FAILED",
                    error_message="Failed to embed consolidated memory in batch UPDATE operation",
                )

            store_result = await self.store_memory_note(
                doc_id=new_memory_data["doc_id"],
                app_id=new_memory_data["app_id"],
                user_id=new_memory_data["user_id"],
                session_id=new_memory_data["session_id"],
                memory_note_text=consolidated_note_text,
                embedding=consolidated_embedding,
                tags=fuzzy_unioned["tags"],
                keywords=fuzzy_unioned["keywords"],
                semantic_queries=fuzzy_unioned["semantic_queries"],
                conversation_complexity=consolidated_complexity,
                interaction_quality=consolidated_quality,
                follow_up_potential=fuzzy_unioned["follow_up_potential"],
                status="active",
            )

            if not store_result.success:
                logger.error("Failed to store consolidated memory")
                duration = time.time() - start_time
                metrics.record_batch(
                    operation="batch_update",
                    duration=duration,
                    total_items=len(update_decisions),
                    succeeded=0,
                    failed=len(update_decisions),
                    error_code="STORE_FAILED",
                )
                return BatchOperationResult(
                    success=False,
                    total_items=len(update_decisions),
                    succeeded=0,
                    failed=len(update_decisions),
                    failed_items=[
                        {
                            "memory_id": d["memory_id"],
                            "error_code": "STORE_FAILED",
                            "error_message": "Consolidated memory storage failed",
                        }
                        for d in update_decisions
                    ],
                    error_code="STORE_FAILED",
                    error_message="Failed to store consolidated memory in batch UPDATE operation",
                )

            update_specs = [
                {
                    "app_id": new_memory_data["app_id"],
                    "doc_id": memory_id,
                    "new_status": "updated",
                    "archive_reason": STATUS_REASON_CONSOLIDATED,
                    "new_memory_id": new_memory_data["doc_id"],
                }
                for memory_id in sorted(memory_ids_to_consolidate)
            ]
            archive_results = await self._run_status_updates_chunked(update_specs)

            batch_result = BatchOperationResult.from_results(
                archive_results, operation_name="batch_update_archive"
            )

            if batch_result.failed > 0:
                logger.error(
                    f"CRITICAL: Batch UPDATE stored new memory {new_memory_data['doc_id']} but "
                    f"failed to archive {batch_result.failed}/{batch_result.total_items} old memories. "
                    f"Failed memory IDs: {[item['memory_id'] for item in batch_result.failed_items]}. "
                    f"Manual intervention may be required to archive these memories."
                )

            logger.info(
                f"Batch UPDATE completed: consolidated {len(memory_ids_to_consolidate)} memories into {new_memory_data['doc_id']} "
                f"(archived: {batch_result.succeeded}, failed: {batch_result.failed})"
            )

            batch_result.details.update(
                {
                    "new_memory_id": new_memory_data["doc_id"],
                    "consolidated_count": len(memory_ids_to_consolidate),
                }
            )

            duration = time.time() - start_time
            metrics.record_batch(
                operation="batch_update",
                duration=duration,
                total_items=batch_result.total_items,
                succeeded=batch_result.succeeded,
                failed=batch_result.failed,
                error_code=batch_result.error_code,
            )

            return batch_result

        except Exception as e:
            logger.error(f"Error in batch UPDATE operation: {e}", exc_info=True)
            result = BatchOperationResult(
                success=False,
                total_items=len(update_decisions),
                succeeded=0,
                failed=len(update_decisions),
                failed_items=[
                    {
                        "memory_id": d["memory_id"],
                        "error_code": "BATCH_UPDATE_EXCEPTION",
                        "error_message": str(e),
                    }
                    for d in update_decisions
                ],
                error_code="BATCH_UPDATE_EXCEPTION",
                error_message=f"Exception in batch UPDATE operation: {str(e)}",
                details={"exception_type": type(e).__name__},
            )
            duration = time.time() - start_time
            metrics.record_batch(
                operation="batch_update",
                duration=duration,
                total_items=result.total_items,
                succeeded=result.succeeded,
                failed=result.failed,
                error_code=result.error_code,
            )
            return result

    async def _execute_batch_delete(
        self,
        delete_decisions: List[Dict[str, Any]],
        new_memory_data: Dict[str, Any],
    ) -> BatchOperationResult:
        """Execute batch DELETE operations: store new memory, archive contradicted ones (async version).

        NOTE: DOES store the new_memory_data directly as an "active" memory,
        then archives the contradicted memories as "deleted".
        """
        import time

        metrics = get_metrics_collector()
        start_time = time.time()
        logger.info(f"Executing batch DELETE for {len(delete_decisions)} decisions")

        if not delete_decisions:
            return BatchOperationResult(
                success=True,
                total_items=0,
                succeeded=0,
                failed=0,
                failed_items=[],
            )

        try:
            store_result = await self.store_memory_note(
                doc_id=new_memory_data["doc_id"],
                app_id=new_memory_data["app_id"],
                user_id=new_memory_data["user_id"],
                session_id=new_memory_data["session_id"],
                memory_note_text=new_memory_data["natural_memory_note"],
                embedding=new_memory_data["embedding"],
                tags=new_memory_data["retrieval_tags"],
                keywords=new_memory_data["retrieval_keywords"],
                semantic_queries=new_memory_data["semantic_queries"],
                conversation_complexity=new_memory_data["conversation_complexity"],
                interaction_quality=new_memory_data["interaction_quality"],
                status=new_memory_data["status"],
            )

            if not store_result.success:
                logger.error("Failed to store new memory in batch DELETE operation")
                result = BatchOperationResult(
                    success=False,
                    total_items=len(delete_decisions) + 1,
                    succeeded=0,
                    failed=len(delete_decisions) + 1,
                    failed_items=[
                        {
                            "memory_id": new_memory_data["doc_id"],
                            "error_code": store_result.error_code,
                            "error_message": store_result.error_message,
                        }
                    ]
                    + [
                        {
                            "memory_id": d["memory_id"],
                            "error_code": "STORE_FAILED",
                            "error_message": "Store operation failed, archiving skipped",
                        }
                        for d in delete_decisions
                    ],
                    error_code="STORE_FAILED",
                    error_message="Failed to store new memory in batch DELETE operation",
                )
                duration = time.time() - start_time
                metrics.record_batch(
                    operation="batch_delete",
                    duration=duration,
                    total_items=result.total_items,
                    succeeded=result.succeeded,
                    failed=result.failed,
                    error_code=result.error_code,
                )
                return result

            contradicted_ids = [d["memory_id"] for d in delete_decisions]

            archive_specs = [
                {
                    "app_id": new_memory_data["app_id"],
                    "doc_id": memory_id,
                    "new_status": "deleted",
                    "archive_reason": STATUS_REASON_CONTRADICTED,
                    "caused_by_memory": new_memory_data["doc_id"],
                    "new_memory_id": new_memory_data["doc_id"],
                }
                for memory_id in sorted(contradicted_ids)
            ]
            archive_results = await self._run_status_updates_chunked(archive_specs)

            batch_result = BatchOperationResult.from_results(
                archive_results, operation_name="batch_delete_archive"
            )

            batch_result.total_items += 1
            batch_result.succeeded += 1

            if batch_result.failed > 0:
                logger.error(
                    f"CRITICAL: Batch DELETE stored new memory {new_memory_data['doc_id']} but "
                    f"failed to archive {batch_result.failed}/{len(contradicted_ids)} contradicted memories. "
                    f"Failed memory IDs: {[item['memory_id'] for item in batch_result.failed_items]}. "
                    f"Manual intervention may be required."
                )

            logger.info(
                f"Batch DELETE completed: stored new memory {new_memory_data['doc_id']}, "
                f"archived {batch_result.succeeded - 1}/{len(contradicted_ids)} contradicted memories"
            )

            batch_result.details.update(
                {
                    "new_memory_id": new_memory_data["doc_id"],
                    "contradicted_count": len(contradicted_ids),
                }
            )

            duration = time.time() - start_time
            metrics.record_batch(
                operation="batch_delete",
                duration=duration,
                total_items=batch_result.total_items,
                succeeded=batch_result.succeeded,
                failed=batch_result.failed,
                error_code=batch_result.error_code,
            )

            return batch_result

        except Exception as e:
            logger.error(f"Error in batch DELETE operation: {e}", exc_info=True)
            result = BatchOperationResult(
                success=False,
                total_items=len(delete_decisions) + 1,
                succeeded=0,
                failed=len(delete_decisions) + 1,
                failed_items=[
                    {
                        "memory_id": d["memory_id"],
                        "error_code": "BATCH_DELETE_EXCEPTION",
                        "error_message": str(e),
                    }
                    for d in delete_decisions
                ],
                error_code="BATCH_DELETE_EXCEPTION",
                error_message=f"Exception in batch DELETE operation: {str(e)}",
                details={"exception_type": type(e).__name__},
            )
            duration = time.time() - start_time
            metrics.record_batch(
                operation="batch_delete",
                duration=duration,
                total_items=result.total_items,
                succeeded=result.succeeded,
                failed=result.failed,
                error_code=result.error_code,
            )
            return result

    async def _execute_batch_skip(
        self,
        skip_decisions: List[Dict[str, Any]],
        app_id: str,
    ) -> BatchOperationResult:
        """Execute batch SKIP operations: refresh timestamps on specified memories (async version)."""
        import time

        metrics = get_metrics_collector()
        start_time = time.time()
        logger.info(f"Executing batch SKIP for {len(skip_decisions)} decisions")

        if not skip_decisions:
            return BatchOperationResult(
                success=True,
                total_items=0,
                succeeded=0,
                failed=0,
                failed_items=[],
            )

        try:
            timestamp_specs = [
                {
                    "app_id": app_id,
                    "doc_id": d["memory_id"],
                    "timestamp": datetime.now(timezone.utc),
                }
                for d in sorted(skip_decisions, key=lambda item: item["memory_id"])
            ]
            timestamp_results = await self._run_timestamp_updates_chunked(
                timestamp_specs
            )

            batch_result = BatchOperationResult.from_results(
                timestamp_results, operation_name="batch_skip_timestamp"
            )

            logger.info(
                f"Batch SKIP completed: refreshed timestamps for {batch_result.succeeded}/{batch_result.total_items} memories"
            )

            duration = time.time() - start_time
            metrics.record_batch(
                operation="batch_skip",
                duration=duration,
                total_items=batch_result.total_items,
                succeeded=batch_result.succeeded,
                failed=batch_result.failed,
                error_code=batch_result.error_code,
            )

            return batch_result

        except Exception as e:
            logger.error(f"Error in batch SKIP operation: {e}", exc_info=True)
            result = BatchOperationResult(
                success=False,
                total_items=len(skip_decisions),
                succeeded=0,
                failed=len(skip_decisions),
                failed_items=[
                    {
                        "memory_id": d["memory_id"],
                        "error_code": "BATCH_SKIP_EXCEPTION",
                        "error_message": str(e),
                    }
                    for d in skip_decisions
                ],
                error_code="BATCH_SKIP_EXCEPTION",
                error_message=f"Exception in batch SKIP operation: {str(e)}",
                details={"exception_type": type(e).__name__},
            )
            duration = time.time() - start_time
            metrics.record_batch(
                operation="batch_skip",
                duration=duration,
                total_items=result.total_items,
                succeeded=result.succeeded,
                failed=result.failed,
                error_code=result.error_code,
            )
            return result
