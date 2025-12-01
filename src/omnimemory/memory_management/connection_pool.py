"""
Connection pool manager for vector database handlers
"""

import asyncio
from asyncio import Queue, QueueEmpty
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, TYPE_CHECKING, cast

from omnimemory.core.llm import LLMConnection
from omnimemory.core.logger_utils import get_logger
from .vector_db_factory import VectorDBFactoryRegistry

if TYPE_CHECKING:
    from .vector_db_base import VectorDBBase

logger = get_logger(name="omnimemory.memory_management.connection_pool")


class VectorDBHandlerPool:
    """
    Connection pool for vector database handlers (MemoryManager level).

    Pools entire vector_db_handler instances (QdrantVectorDB, MongoDBVectorDB, etc.)
    which already include LLM connections. This avoids recreating handlers and
    LLM connections for each query.

    Features:
    - Thread-safe connection pooling
    - Reuses vector_db_handler instances (including LLM connections)
    - Separate handling for queries (pooled) and writes (dedicated handlers)
    - Tunable capacity: adjust via VectorDBHandlerPool.get_instance(max_connections=<value>)
      depending on available resources (e.g., increase to 30 for heavy embedding loads)
    """

    _instance: Optional["VectorDBHandlerPool"] = None
    _lock: Optional[Any] = None
    _class_initialized: bool = False

    def __init__(self, max_connections: int = 10) -> None:
        """
        Initialize connection pool .

        Args:
            max_connections: Maximum number of vector_db_handler instances in pool
        """
        self.max_connections: int = max_connections
        self._pool: Optional[Queue["VectorDBBase"]] = None
        self._created_handlers: int = 0
        self._active_handlers: int = 0
        self._total_checkouts: int = 0
        self._pending_waiters: int = 0
        self._llm_connection: Optional[LLMConnection] = None
        self._pool_lock: Optional[asyncio.Lock] = None
        self._initialized: bool = False

        logger.info(
            f"VectorDBHandlerPool initialized with max_connections={max_connections}"
        )

    def _require_pool(self) -> Queue["VectorDBBase"]:
        if self._pool is None:
            raise RuntimeError("VectorDB handler pool is not initialized")
        return self._pool

    def _require_pool_lock(self) -> asyncio.Lock:
        if self._pool_lock is None:
            raise RuntimeError("VectorDB handler pool lock is not initialized")
        return self._pool_lock

    def _require_llm_connection(self) -> LLMConnection:
        if self._llm_connection is None:
            raise RuntimeError("LLM connection is not set for the pool")
        return self._llm_connection

    async def _create_handler(
        self, llm_connection: LLMConnection
    ) -> Optional["VectorDBBase"]:
        """
        Create a new vector_db_handler instance .

        Args:
            llm_connection: LLM connection to use for the handler.

        Returns:
            VectorDBBase instance if successful, None otherwise.
        """
        try:
            handler = await VectorDBFactoryRegistry.create_from_env(llm_connection)
            if handler and handler.enabled:
                return handler
            else:
                logger.warning("Created vector_db_handler but it's not enabled")
                return None
        except Exception as e:
            logger.error(f"Failed to create vector_db_handler: {e}", exc_info=True)
            return None

    async def initialize_pool(self, llm_connection: LLMConnection) -> None:
        """
        Initialize the pool with shared LLM connection .

        NOTE: This method should be called while holding the pool_lock.
        It does NOT acquire the lock itself to avoid deadlocks.

        Args:
            llm_connection: Shared LLM connection for all pooled handlers
        """
        if VectorDBHandlerPool._class_initialized:
            instance = VectorDBHandlerPool._instance
            if instance and instance._pool:
                self._pool = instance._pool
                self._pool_lock = instance._pool_lock
                self._llm_connection = instance._llm_connection
                self._created_handlers = instance._created_handlers
                self._initialized = True
            return

        if self._pool is None:
            self._pool = Queue(maxsize=self.max_connections)

        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()

        if self._llm_connection is None:
            self._llm_connection = llm_connection

        initial_size = max(1, self.max_connections // 2)
        for _ in range(initial_size):
            handler = await self._create_handler(self._llm_connection)
            if handler:
                await self._pool.put(handler)
                self._created_handlers += 1

        self._initialized = True
        VectorDBHandlerPool._class_initialized = True
        VectorDBHandlerPool._instance = self
        logger.info(
            "Pool initialized with %s handlers (max=%s)",
            self._created_handlers,
            self.max_connections,
        )

    @asynccontextmanager
    async def get_handler(
        self, llm_connection: Optional[LLMConnection] = None
    ) -> AsyncGenerator["VectorDBBase", None]:
        """
        Get a vector_db_handler from the pool (async context manager for queries).

        Usage:
            async with pool.get_handler(llm_connection) as handler:
                results = await handler.query_collection(...)

        Args:
            llm_connection: LLM connection (optional, used to initialize pool if needed and not already initialized)

        Yields:
            vector_db_handler instance (QdrantVectorDB, etc.)

        Raises:
            ValueError: If llm_connection is required but not provided.
        """
        instance = VectorDBHandlerPool._instance
        if instance is not None and self._pool is None:
            self._pool = instance._pool
            self._pool_lock = instance._pool_lock
            self._llm_connection = instance._llm_connection
            self._created_handlers = instance._created_handlers
            self._initialized = instance._initialized

        if (
            VectorDBHandlerPool._class_initialized
            and not self._initialized
            and instance is not None
        ):
            self._pool = instance._pool
            self._pool_lock = instance._pool_lock
            self._llm_connection = instance._llm_connection
            self._created_handlers = instance._created_handlers
            self._initialized = instance._initialized

        if not self._initialized and not VectorDBHandlerPool._class_initialized:
            if llm_connection is None:
                raise ValueError("llm_connection is required for pool initialization")
            if self._pool_lock is None:
                self._pool_lock = asyncio.Lock()
            pool_lock = self._require_pool_lock()
            async with pool_lock:
                if not self._initialized:
                    await self.initialize_pool(llm_connection)

        handler = None
        try:
            handler = await self._acquire_handler_with_retry()

            yield handler

        except Exception as e:
            logger.error(f"Error using pooled vector_db_handler: {e}")
            handler = None
            raise
        finally:
            if handler is not None:
                try:
                    pool = self._require_pool()
                    await asyncio.wait_for(pool.put(handler), timeout=1.0)
                    self._active_handlers -= 1
                    logger.debug(
                        "Returned handler (active=%s, available=%s, max=%s, pending=%s)",
                        self._active_handlers,
                        pool.qsize(),
                        self.max_connections,
                        self._pending_waiters,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout returning handler to pool")
                    self._active_handlers -= 1
                except Exception as e:
                    logger.warning(f"Failed to return handler to pool: {e}")
                    try:
                        llm_conn = self._require_llm_connection()
                        new_handler = await self._create_handler(llm_conn)
                        if new_handler:
                            pool = self._require_pool()
                            await pool.put(new_handler)
                    except Exception as create_error:
                        logger.error(
                            f"Failed to create replacement handler: {create_error}"
                        )
                    self._active_handlers -= 1

    async def _acquire_handler_with_retry(self) -> "VectorDBBase":
        """
        Try to acquire handler with retries/backoff when pool is exhausted.

        Returns:
            VectorDBBase handler instance.

        Raises:
            TimeoutError: If handler cannot be acquired after max retries.
        """
        retries = 0
        max_retries = 5
        backoff = 0.5
        handler: Optional["VectorDBBase"] = None

        while True:
            try:
                self._pending_waiters += 1
                pool = self._require_pool()
                handler = cast(
                    "VectorDBBase", await asyncio.wait_for(pool.get(), timeout=5.0)
                )
                self._pending_waiters -= 1
                self._active_handlers += 1
                self._total_checkouts += 1
                logger.debug(
                    "Checked-out handler (active=%s, available=%s, max=%s, pending=%s)",
                    self._active_handlers,
                    pool.qsize(),
                    self.max_connections,
                    self._pending_waiters,
                )
                return handler
            except asyncio.TimeoutError:
                self._pending_waiters = max(0, self._pending_waiters - 1)
                pool_lock = self._require_pool_lock()
                async with pool_lock:
                    if self._created_handlers < self.max_connections:
                        llm_conn = self._require_llm_connection()
                        handler = await self._create_handler(llm_conn)
                        if handler:
                            self._created_handlers += 1
                            self._active_handlers += 1
                            self._total_checkouts += 1
                            logger.debug(
                                "Created handler (active=%s, available=%s, max=%s, pending=%s)",
                                self._active_handlers,
                                self._require_pool().qsize(),
                                self.max_connections,
                                self._pending_waiters,
                            )
                            return handler
                if retries >= max_retries:
                    logger.error(
                        "Failed to acquire handler after %s attempts; raising Timeout",
                        max_retries,
                    )
                    raise TimeoutError("Connection pool exhausted")
                retries += 1
                logger.warning(
                    "Pool exhausted (retry %s/%s). Backing off for %.1fs",
                    retries,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def get_dedicated_handler(
        self, llm_connection: LLMConnection
    ) -> Optional["VectorDBBase"]:
        """
        Get a dedicated vector_db_handler for write operations (not pooled) - async version.

        This should be used for write operations that need dedicated handlers.
        The handler should be kept for the lifetime of the operation.

        Args:
            llm_connection: LLM connection for the handler

        Returns:
            vector_db_handler instance (dedicated, not pooled), or None if creation fails
        """
        return await self._create_handler(llm_connection)

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics ."""
        if self._pool_lock is None:
            return {
                "max_connections": self.max_connections,
                "created_handlers": 0,
                "active_handlers": 0,
                "available_handlers": 0,
                "initialized": False,
            }

        async with self._pool_lock:
            pool_size = self._pool.qsize() if self._pool else 0
            return {
                "max_connections": self.max_connections,
                "created_handlers": self._created_handlers,
                "active_handlers": self._active_handlers,
                "available_handlers": pool_size,
                "pending_waiters": self._pending_waiters,
                "total_checkouts": self._total_checkouts,
                "initialized": self._initialized,
            }

    @classmethod
    def get_instance(cls, max_connections: int = 10) -> "VectorDBHandlerPool":
        """
        Get singleton instance of connection pool (thread-safe).

        Args:
            max_connections: Maximum connections (only used on first call)

        Returns:
            VectorDBHandlerPool instance
        """
        if cls._instance is None:
            if cls._lock is None:
                import threading

                cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_connections=max_connections)
        return cls._instance

    async def close_all(self) -> None:
        """
        Close all handlers in the pool (cleanup, async version).

        Gracefully closes all pooled handlers and resets pool statistics.
        """
        logger.info("Closing all handlers in pool...")
        closed_count = 0

        if self._pool is None:
            return

        while not self._pool.empty():
            try:
                handler = self._pool.get_nowait()
                if handler and hasattr(handler, "close"):
                    try:
                        if asyncio.iscoroutinefunction(handler.close):
                            await handler.close()
                        elif hasattr(handler, "client") and hasattr(
                            handler.client, "close"
                        ):
                            if asyncio.iscoroutinefunction(handler.client.close):
                                await handler.client.close()
                            else:
                                handler.client.close()
                        closed_count += 1
                    except Exception as e:
                        logger.warning(f"Error closing handler: {e}")
            except QueueEmpty:
                break

        self._created_handlers = 0
        self._active_handlers = 0
        logger.info(f"Closed {closed_count} handlers from pool")
