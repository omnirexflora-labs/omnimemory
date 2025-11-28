"""
Memory Management Package

This package provides advanced memory management functionality:
- Vector Database Management (Qdrant, MongoDB, ChromaDB, PostgreSQL)
- Memory Manager
- Factory functions for dependency injection
"""

from .memory_manager import MemoryManager
from .vector_db_base import VectorDBBase
from .chromadb_vector_db import ChromaDBVectorDB, ChromaClientType
from .qdrant_vector_db import QdrantVectorDB
from .mongodb_vector_db import MongoDBVectorDB
from .postgresql_vector_db import PostgreSQLVectorDB
from .vector_db_factory import (
    VectorDBFactoryRegistry,
    QdrantVectorDBFactory,
    ChromaRemoteVectorDBFactory,
    ChromaCloudVectorDBFactory,
    MongoVectorDBFactory,
    PostgresVectorDBFactory,
)

__all__ = [
    "MemoryManager",
    "VectorDBBase",
    "QdrantVectorDB",
    "ChromaDBVectorDB",
    "ChromaClientType",
    "MongoDBVectorDB",
    "PostgreSQLVectorDB",
    "VectorDBFactoryRegistry",
    "QdrantVectorDBFactory",
    "ChromaRemoteVectorDBFactory",
    "ChromaCloudVectorDBFactory",
    "MongoVectorDBFactory",
    "PostgresVectorDBFactory",
]
