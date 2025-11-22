"""
OmniMemory Core Schemas
Defines Pydantic models for request and response schemas used in OmniMemory core components.
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    AnyHttpUrl,
)
from omnimemory.core.config import DEFAULT_MAX_MESSAGES
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


class Message(BaseModel):
    role: str = Field(..., description="The role of the message")
    content: str = Field(..., description="The content of the message")
    timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="The timestamp of the message",
    )


class UserMessages(BaseModel):
    app_id: str = Field(
        ...,
        description="The ID of the application that the user is interacting with",
        min_length=10,
        max_length=36,
    )
    user_id: str = Field(
        ...,
        description="The ID of the user that is interacting with the application",
        min_length=10,
        max_length=36,
    )
    session_id: str | None = Field(
        None,
        description="The ID of the session that the user is interacting with",
        min_length=10,
        max_length=36,
    )
    messages: List[Message] = Field(
        ...,
        description=f"The messages that the user is sending to the omnimemory. Must be between 1 and {DEFAULT_MAX_MESSAGES} messages.",
        min_length=DEFAULT_MAX_MESSAGES,
        max_length=DEFAULT_MAX_MESSAGES,
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_count(cls, v: List[Message]) -> List[Message]:
        """Validate that messages count is DEFAULT_MAX_MESSAGES"""

        if len(v) != DEFAULT_MAX_MESSAGES:
            raise ValueError(
                f"Exactly {DEFAULT_MAX_MESSAGES} messages are required. You provided {len(v)} messages. "
                f"Please provide {DEFAULT_MAX_MESSAGES} messages."
            )
        return v

    @model_validator(mode="after")
    def validate_messages_range(self):
        """Additional validation to ensure messages count is DEFAULT_MAX_MESSAGES"""
        if len(self.messages) != DEFAULT_MAX_MESSAGES:
            raise ValueError(
                f"Exactly {DEFAULT_MAX_MESSAGES} messages are required. You provided {len(self.messages)} messages. "
                f"Please provide {DEFAULT_MAX_MESSAGES} messages."
            )
        return self


class AddUserMessageRequest(BaseModel):
    """Request schema for adding user messages (API/CLI entry point)."""

    app_id: str = Field(..., description="Application ID", min_length=10)
    user_id: str = Field(..., description="User ID", min_length=10)
    session_id: Optional[str] = Field(None, description="Session ID")
    messages: List[Dict[str, str]] = Field(
        ...,
        description=f"List of messages with role, content, and timestamp. Must be {DEFAULT_MAX_MESSAGES} messages.",
        min_length=DEFAULT_MAX_MESSAGES,
        max_length=DEFAULT_MAX_MESSAGES,
    )

    def to_user_messages(self) -> "UserMessages":
        """Convert to UserMessages schema (validates message count)."""
        return UserMessages(
            app_id=self.app_id,
            user_id=self.user_id,
            session_id=self.session_id,
            messages=[Message(**msg) for msg in self.messages],
        )


class ConversationSummaryRequest(BaseModel):
    """Request schema for single-agent conversation summarization."""

    app_id: str = Field(..., description="Application ID", min_length=10)
    user_id: str = Field(..., description="User ID", min_length=10)
    session_id: Optional[str] = Field(None, description="Session ID")
    messages: List[Message] | str = Field(
        ...,
        description="Conversation payload. Accepts a raw string or list of message objects.",
    )
    callback_url: Optional[AnyHttpUrl] = Field(
        None, description="Optional webhook to receive the summary asynchronously"
    )
    callback_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional headers to include when invoking the callback",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value):
        """Ensure messages payload is non-empty."""
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("messages string cannot be empty")
            return value
        if isinstance(value, list):
            if not value:
                raise ValueError("At least one message is required")
            return value
        raise ValueError("messages must be a string or list of Message objects")


class QueryMemoryRequest(BaseModel):
    """Request schema for querying memories."""

    app_id: str = Field(..., description="Application ID", min_length=10)
    query: str = Field(..., description="Natural language query", min_length=10)
    user_id: Optional[str] = Field(None, description="User ID filter")
    session_id: Optional[str] = Field(None, description="Session ID filter")
    n_results: Optional[int] = Field(
        None, description="Maximum number of results", ge=1, le=100
    )
    similarity_threshold: Optional[float] = Field(
        None, description="Similarity threshold", ge=0.0, le=1.0
    )


class TaskResponse(BaseModel):
    """Response schema for task operations."""

    task_id: Optional[str] = Field(None, description="Task ID")
    status: str = Field(..., description="Task status")
    message: Optional[str] = Field(None, description="Status message")
    app_id: Optional[str] = Field(None, description="Application ID")
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    error: Optional[str] = Field(None, description="Error message")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")


class MemoryResponse(BaseModel):
    """Response schema for a single memory."""

    memory_id: Optional[str] = Field(None, description="Memory ID")
    document: Optional[str] = Field(None, description="Memory document text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Memory metadata")


class MemoryListResponse(BaseModel):
    """Response schema for memory list."""

    memories: List[Dict[str, Any]] = Field(..., description="List of memory objects")
    count: int = Field(..., description="Number of memories returned")


class MemoryIDResponse(BaseModel):
    """Response schema for memory ID listing."""

    memory_id: str = Field(..., description="Memory ID")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    status: str = Field(..., description="Memory status")
    preview: str = Field(..., description="Memory preview text")


class MemoryIDListResponse(BaseModel):
    """Response schema for memory ID list."""

    memory_ids: List[MemoryIDResponse] = Field(..., description="List of memory IDs")
    count: int = Field(..., description="Number of memory IDs returned")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


class SuccessResponse(BaseModel):
    """Response schema for success operations."""

    success: bool = Field(True, description="Operation success status")
    message: Optional[str] = Field(None, description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


class AgentMemoryRequest(BaseModel):
    """Request schema for agent-driven memory creation."""

    app_id: str = Field(..., description="Application ID", min_length=10)
    user_id: str = Field(..., description="User ID", min_length=10)
    session_id: Optional[str] = Field(None, description="Session ID")
    messages: List[Message] | str = Field(
        ...,
        description="Messages from agent. Accepts a raw string or list of message objects.",
    )


class AgentMemoryResponse(BaseModel):
    """Response schema for agent memory creation."""

    memory_id: str = Field(..., description="ID of the created memory")
    app_id: str = Field(..., description="Application ID")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    summary: str = Field(..., description="Generated summary text")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the memory was created",
    )


class ConversationSummaryResponse(BaseModel):
    """Response schema for single-agent conversation summaries."""

    app_id: str = Field(..., description="Application ID")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    summary: str = Field(..., description="Primary conversation summary text")
    key_points: Optional[str] = Field(
        None, description="Key insights or bullet summary, if available"
    )
    tags: List[str] = Field(default_factory=list, description="Suggested tags")
    keywords: List[str] = Field(default_factory=list, description="Suggested keywords")
    semantic_queries: List[str] = Field(
        default_factory=list, description="Suggested semantic queries"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata returned by the agent"
    )
    raw_sections: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw agent response sections (for debugging/auditing)"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the summary was generated",
    )
