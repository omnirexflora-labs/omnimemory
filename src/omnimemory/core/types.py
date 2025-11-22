from typing import TypedDict, List, Dict, Any, Optional


class MemoryNoteData(TypedDict, total=False):
    """Type definition for parsed memory note data."""

    natural_memory_note: str
    retrieval_tags: List[str]
    retrieval_keywords: List[str]
    semantic_queries: List[str]
    conversation_complexity: str
    interaction_quality: str
    follow_up_potential: List[str]


class MemoryDataDict(TypedDict, total=False):
    """Type definition for memory data dictionary."""

    doc_id: str
    app_id: str
    user_id: str
    session_id: Optional[str]
    embedding: List[float]
    natural_memory_note: str
    retrieval_tags: List[str]
    retrieval_keywords: List[str]
    semantic_queries: List[str]
    conversation_complexity: Optional[int]
    interaction_quality: Optional[str]
    follow_up_potential: List[str]
    status: str
