"""
Vector database factory registry.

Provides a classic factory pattern implementation where each backend exposes its own
factory class and the registry picks the correct one based on configuration.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Protocol, Tuple, Type, TYPE_CHECKING

from decouple import config

from omnimemory.core.llm import LLMConnection
from omnimemory.core.logger_utils import get_logger

if TYPE_CHECKING:
    from .vector_db_base import VectorDBBase


logger = get_logger(name="omnimemory.memory_management.vector_db_factory")


class ProviderFactory(Protocol):
    """Protocol for provider-specific factories."""

    provider_key: ClassVar[str]

    @staticmethod
    async def create(llm_connection: LLMConnection) -> Optional["VectorDBBase"]: ...


class QdrantVectorDBFactory:
    provider_key: ClassVar[str] = "qdrant-remote"

    @staticmethod
    async def create(llm_connection: LLMConnection) -> Optional["VectorDBBase"]:
        from .qdrant_vector_db import QdrantVectorDB

        return QdrantVectorDB(llm_connection=llm_connection)


class ChromaRemoteVectorDBFactory:
    provider_key: ClassVar[str] = "chromadb-remote"

    @staticmethod
    async def create(llm_connection: LLMConnection) -> Optional["VectorDBBase"]:
        from .chromadb_vector_db import ChromaDBVectorDB, ChromaClientType

        return ChromaDBVectorDB(
            client_type=ChromaClientType.REMOTE, llm_connection=llm_connection
        )


class ChromaCloudVectorDBFactory:
    provider_key: ClassVar[str] = "chromadb-cloud"

    @staticmethod
    async def create(llm_connection: LLMConnection) -> Optional["VectorDBBase"]:
        from .chromadb_vector_db import ChromaDBVectorDB, ChromaClientType

        return ChromaDBVectorDB(
            client_type=ChromaClientType.CLOUD, llm_connection=llm_connection
        )


class MongoVectorDBFactory:
    provider_key: ClassVar[str] = "mongodb-remote"

    @staticmethod
    async def create(llm_connection: LLMConnection) -> Optional["VectorDBBase"]:
        from .mongodb_vector_db import MongoDBVectorDB

        return MongoDBVectorDB(llm_connection=llm_connection)


class PostgresVectorDBFactory:
    provider_key: ClassVar[str] = "postgresql"

    @staticmethod
    async def create(llm_connection: LLMConnection) -> Optional["VectorDBBase"]:
        from .postgresql_vector_db import PostgreSQLVectorDB

        return PostgreSQLVectorDB(llm_connection=llm_connection)


class VectorDBFactoryRegistry:
    """Registry coordinating provider-specific factories."""

    _factory_classes: Tuple[Type[ProviderFactory], ...] = (
        QdrantVectorDBFactory,
        ChromaRemoteVectorDBFactory,
        ChromaCloudVectorDBFactory,
        MongoVectorDBFactory,
        PostgresVectorDBFactory,
    )
    _factories: Dict[str, Type[ProviderFactory]] = {
        factory.provider_key: factory for factory in _factory_classes
    }

    @classmethod
    def register_factory(cls, factory: Type[ProviderFactory]) -> None:
        cls._factories[factory.provider_key] = factory

    @classmethod
    async def create_from_env(
        cls, llm_connection: LLMConnection
    ) -> Optional["VectorDBBase"]:
        provider = config("OMNI_MEMORY_PROVIDER", default=None)
        if not provider:
            logger.error(
                "OMNI_MEMORY_PROVIDER is not set - vector database operations will be disabled"
            )
            return None
        return await cls.create(provider.lower(), llm_connection)

    @classmethod
    async def create(
        cls, provider: str, llm_connection: LLMConnection
    ) -> Optional["VectorDBBase"]:
        factory = cls._factories.get(provider)
        if not factory:
            logger.error(f"Unsupported OMNI_MEMORY_PROVIDER: {provider}")
            return None
        try:
            return await factory.create(llm_connection)
        except Exception as exc:
            logger.error(
                f"Failed to create vector database handler for {provider}: {exc}",
                exc_info=True,
            )
            return None
