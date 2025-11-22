"""
Connection pool manager for vector database handlers (async version).

Manages connection pooling at the MemoryManager/vector_db_handler level,
not at the underlying client level. This ensures we reuse entire
vector_db_handler instances (including their LLM connections) for queries,
while providing dedicated handlers for write operations.
"""

import asyncio
from typing import Dict, Any, Optional, Callable
from asyncio import Queue, QueueEmpty
from contextlib import asynccontextmanager
from decouple import config
from omnimemory.core.logger_utils import get_logger
from . import create_vector_db_handler

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
    - Separate handling for queries (pooled) vs writes (dedicated)
    - Tunable capacity: adjust via VectorDBHandlerPool.get_instance(max_connections=<value>)
      depending on available resources (e.g., increase to 30 for heavy embedding loads)
    """

    _instance = None
    _lock = None
    _class_initialized = False

    def __init__(self, max_connections: int = 10):
        """
        Initialize connection pool (async version).

        Args:
            max_connections: Maximum number of vector_db_handler instances in pool
        """
        self.max_connections = max_connections
        self._pool: Optional[Queue] = None
        self._created_handlers = 0
        self._active_handlers = 0
        self._total_checkouts = 0
        self._pending_waiters = 0
        self._llm_connection = None
        self._pool_lock: Optional[asyncio.Lock] = None
        self._initialized = False

        logger.info(
            f"VectorDBHandlerPool initialized with max_connections={max_connections}"
        )

    async def _create_handler(self, llm_connection: Callable):
        """Create a new vector_db_handler instance (async version)."""
        try:
            handler = await create_vector_db_handler(llm_connection)
            if handler and handler.enabled:
                return handler
            else:
                logger.warning("Created vector_db_handler but it's not enabled")
                return None
        except Exception as e:
            logger.error(f"Failed to create vector_db_handler: {e}", exc_info=True)
            return None

    async def initialize_pool(self, llm_connection: Callable):
        """
        Initialize the pool with shared LLM connection (async version).

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

        initial_size = min(2, self.max_connections)
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
    async def get_handler(self, llm_connection: Callable = None):
        """
        Get a vector_db_handler from the pool (async context manager for queries).

        Usage:
            async with pool.get_handler(llm_connection) as handler:
                results = await handler.query_collection(...)

        Args:
            llm_connection: LLM connection (optional, used to initialize pool if needed and not already initialized)

        Yields:
            vector_db_handler instance (QdrantVectorDB, etc.)
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
            async with self._pool_lock:
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
                    await asyncio.wait_for(self._pool.put(handler), timeout=1.0)
                    self._active_handlers -= 1
                    logger.debug(
                        "Returned handler (active=%s, available=%s, max=%s, pending=%s)",
                        self._active_handlers,
                        self._pool.qsize(),
                        self.max_connections,
                        self._pending_waiters,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout returning handler to pool")
                    self._active_handlers -= 1
                except Exception as e:
                    logger.warning(f"Failed to return handler to pool: {e}")
                    try:
                        new_handler = await self._create_handler(self._llm_connection)
                        if new_handler:
                            await self._pool.put(new_handler)
                    except Exception as create_error:
                        logger.error(
                            f"Failed to create replacement handler: {create_error}"
                        )
                    self._active_handlers -= 1

    async def _acquire_handler_with_retry(self) -> Any:
        """Try to acquire handler with retries/backoff when pool is exhausted."""
        retries = 0
        max_retries = 5
        backoff = 0.5

        while True:
            try:
                self._pending_waiters += 1
                handler = await asyncio.wait_for(self._pool.get(), timeout=5.0)
                self._pending_waiters -= 1
                self._active_handlers += 1
                self._total_checkouts += 1
                logger.debug(
                    "Checked-out handler (active=%s, available=%s, max=%s, pending=%s)",
                    self._active_handlers,
                    self._pool.qsize(),
                    self.max_connections,
                    self._pending_waiters,
                )
                return handler
            except asyncio.TimeoutError:
                self._pending_waiters = max(0, self._pending_waiters - 1)
                async with self._pool_lock:
                    if self._created_handlers < self.max_connections:
                        handler = await self._create_handler(self._llm_connection)
                        if handler:
                            self._created_handlers += 1
                            self._active_handlers += 1
                            self._total_checkouts += 1
                            logger.debug(
                                "Created handler (active=%s, available=%s, max=%s, pending=%s)",
                                self._active_handlers,
                                self._pool.qsize(),
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

    async def get_dedicated_handler(self, llm_connection: Callable):
        """
        Get a dedicated vector_db_handler for write operations (not pooled) - async version.

        This should be used for write operations that need dedicated handlers.
        The handler should be kept for the lifetime of the operation.

        Args:
            llm_connection: LLM connection for the handler

        Returns:
            vector_db_handler instance (dedicated, not pooled)
        """
        return await self._create_handler(llm_connection)

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics (async version)."""
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

    async def close_all(self):
        """Close all handlers in the pool (cleanup, async version)."""
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
