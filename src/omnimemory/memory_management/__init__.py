"""
Memory Management Package

This package provides advanced memory management functionality:
- Vector Database Management (Qdrant, MongoDB, ChromaDB)
- Memory Manager
- Factory functions for dependency injection
"""

from typing import Callable, Any
from decouple import config
from omnimemory.core.logger_utils import get_logger

logger = get_logger(name="omnimemory.memory_management")


async def create_vector_db_handler(llm_connection: Callable) -> Any:
    """Factory function to create vector database handler based on OMNI_MEMORY_PROVIDER config (async version).

    This factory function implements dependency injection, allowing MemoryManager
    to remain lean by delegating vector database creation to this factory.

    Args:
        llm_connection: LLM connection instance for embedding generation

    Returns:
        Vector database handler instance (QdrantVectorDB, MongoDBVectorDB, etc.)
        or None if provider not configured or unsupported
    """
    provider = config("OMNI_MEMORY_PROVIDER", default=None)
    if not provider:
        logger.error(
            "OMNI_MEMORY_PROVIDER is not set - vector database operations will be disabled"
        )
        return None

    provider = provider.lower()

    try:
        if provider == "qdrant-remote":
            from omnimemory.memory_management.qdrant_vector_db import QdrantVectorDB

            return QdrantVectorDB(llm_connection=llm_connection)

        else:
            logger.error(f"Unsupported OMNI_MEMORY_PROVIDER: {provider}")
            return None

    except Exception as e:
        logger.error(f"Failed to create vector database handler for {provider}: {e}")
        return None


from .memory_manager import MemoryManager
from .vector_db_base import VectorDBBase
from .qdrant_vector_db import QdrantVectorDB

__all__ = [
    "MemoryManager",
    "VectorDBBase",
    "QdrantVectorDB",
    "create_vector_db_handler",
]
