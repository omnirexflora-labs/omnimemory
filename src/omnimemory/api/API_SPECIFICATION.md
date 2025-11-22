# OmniMemory API Specification

Complete API specification for OmniMemory REST API.

**Architecture:** Self-Evolving Composite Memory Synthesis Architecture (SECMSA)

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement appropriate authentication mechanisms.

## Content Type

All requests and responses use `application/json`.

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

HTTP Status Codes:
- `400 Bad Request` - Invalid request parameters or validation errors
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - SDK not initialized

---

## Memory Operations

### Add Memory

Create a new memory from user messages asynchronously.

**Endpoint:** `POST /api/v1/memories`

**Request Body:**
```json
{
  "app_id": "string (required, min 1 char)",
  "user_id": "string (required, min 1 char)",
  "session_id": "string (optional)",
  "messages": [
    {
      "role": "string (required)",
      "content": "string (required)",
      "timestamp": "string (required)"
    }
  ]
}
```

**Validation:**
- `messages` array must contain exactly the OMNIMEMORY_DEFAULT_MAX_MESSAGES
- Each message must have `role`, `content`, and `timestamp`

**Response:** `202 Accepted`
```json
{
  "task_id": "string",
  "status": "accepted",
  "message": "Memory creation task submitted successfully",
  "app_id": "string",
  "user_id": "string",
  "session_id": "string or null",
  "error": null,
  "result": null
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "session_id": "session-456",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?",
        "timestamp": "2024-01-01T00:00:00Z"
      },
      {
        "role": "assistant",
        "content": "I am doing well, thank you!",
        "timestamp": "2024-01-01T00:00:05Z"
      }
    ]
  }'
```

**Error Responses:**
- `400 Bad Request` - If messages count is < 1 or > 30, or validation fails
- `500 Internal Server Error` - If memory creation fails

---

### Query Memories

Query memories with intelligent multi-dimensional ranking.

**Endpoint:** `GET /api/v1/memories/query`

**Query Parameters:**
- `app_id` (required, string) - Application ID
- `query` (required, string, min 1 char) - Natural language query
- `user_id` (optional, string) - User ID filter
- `session_id` (optional, string) - Session ID filter
- `n_results` (optional, int, 1-100) - Maximum number of results
- `similarity_threshold` (optional, float, 0.0-1.0) - Similarity threshold

**Response:** `200 OK`
```json
{
  "memories": [
    {
      "document": "string",
      "metadata": {},
      "composite_score": 0.95,
      "similarity_score": 0.88,
      "recency_score": 0.05,
      "importance_score": 0.02,
      "query_status": "completed"
    }
  ],
  "count": 10
}
```

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/memories/query?app_id=my-app-id&query=hello&user_id=user-123&n_results=5"
```

**Scoring:**
The query uses composite scoring combining:
- **Relevance** (semantic similarity)
- **Recency** (time-based freshness)
- **Importance** (content significance)

Formula: `composite = relevance × (1 + recency_boost + importance_boost)`

---

### Get Memory

Get a single memory by its ID.

**Endpoint:** `GET /api/v1/memories/{memory_id}`

**Path Parameters:**
- `memory_id` (required, string) - Memory ID

**Query Parameters:**
- `app_id` (required, string) - Application ID

**Response:** `200 OK`
```json
{
  "memory_id": "string",
  "document": "string",
  "metadata": {}
}
```

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/memories/memory-123?app_id=my-app-id"
```

**Error Responses:**
- `404 Not Found` - If memory not found

---

### Delete Memory

Delete a memory from the collection.

**Endpoint:** `DELETE /api/v1/memories/{memory_id}`

**Path Parameters:**
- `memory_id` (required, string) - Memory ID

**Query Parameters:**
- `app_id` (required, string) - Application ID

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Memory {memory_id} deleted successfully",
  "data": null
}
```

**Example Request:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/memories/memory-123?app_id=my-app-id"
```

**Error Responses:**
- `500 Internal Server Error` - If deletion fails

---

### Traverse Memory Evolution Chain

Traverse the memory evolution chain using singly linked list algorithm. Starting from the given memory_id, follows the `next_id` links forward until reaching None, collecting all memories in the evolution chain.

**Endpoint:** `GET /api/v1/memories/{memory_id}/evolution`

**Path Parameters:**
- `memory_id` (required, string) - Starting memory ID to begin traversal

**Query Parameters:**
- `app_id` (required, string) - Application ID

**Response:** `200 OK`
```json
{
  "memories": [
    {
      "memory_id": "string",
      "document": "string",
      "metadata": {
        "next_id": "string or null",
        "created_at": "string",
        "updated_at": "string",
        "status": "string",
        ...
      }
    }
  ],
  "count": 3
}
```

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/memories/memory-123/evolution?app_id=my-app-id"
```

**Algorithm:**
1. Start from the given `memory_id`
2. Fetch the memory and add it to the result list
3. Extract `next_id` from memory metadata
4. If `next_id` is not None, repeat from step 2 with `next_id`
5. Continue until `next_id` is None (end of chain)
6. Return all memories in evolution order (oldest to newest)

**Features:**
- Forward traversal (oldest → newest)
- Cycle detection to prevent infinite loops
- Handles missing memories gracefully
- Returns empty list if starting memory not found

**Error Responses:**
- `500 Internal Server Error` - If traversal fails

---

## Agent Operations

### Add Agent Memory

Create a memory directly from agent messages. This endpoint uses a fast, single-agent pipeline optimized for agent-driven memory creation. Perfect for AI agents that need to store information quickly without the full dual-agent processing pipeline.

**Endpoint:** `POST /api/v1/agent/memories`

**Request Body:**
```json
{
  "app_id": "string (required, min 1 char)",
  "user_id": "string (required, min 1 char)",
  "session_id": "string (optional)",
  "messages": "string or array of message objects"
}
```

**Messages Format:**
- **String:** Raw text messages from the agent
- **Array:** List of message objects with `role`, `content`, and optional `timestamp`

**Response:** `202 Accepted`
```json
{
  "task_id": "string",
  "status": "accepted",
  "message": "Agent memory creation task submitted successfully",
  "app_id": "string",
  "user_id": "string",
  "session_id": "string or null"
}
```

**Example Request (String):**
```bash
curl -X POST "http://localhost:8000/api/v1/agent/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "session_id": "session-456",
    "messages": "User completed onboarding and selected premium plan"
  }'
```

**Example Request (Array):**
```bash
curl -X POST "http://localhost:8000/api/v1/agent/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "messages": [
      {"role": "agent", "content": "User asked about pricing", "timestamp": "2024-01-01T00:00:00Z"},
      {"role": "user", "content": "What are your plans?", "timestamp": "2024-01-01T00:00:05Z"}
    ]
  }'
```

**Features:**
- **Fast Processing:** Uses optimized single-agent summary generation (<10 seconds)
- **Flexible Input:** Accepts both string and structured message arrays
- **Simple Storage:** Direct storage without conflict resolution or metadata extraction
- **Async Processing:** Returns immediately with task_id for background processing

---

### Summarize Conversation

Generate a comprehensive conversation summary using a single-agent pipeline. Perfect for summarizing long conversations, chat logs, or any text-based interactions.

**Endpoint:** `POST /api/v1/agent/summaries`

**Request Body:**
```json
{
  "app_id": "string (required)",
  "user_id": "string (required)",
  "session_id": "string (optional)",
  "messages": "string or array of message objects",
  "callback_url": "string (optional)",
  "callback_headers": {
    "Authorization": "Bearer token"
  }
}
```

**Response Modes:**

**Synchronous (`200 OK`)** - When no callback URL is provided:
- Returns summary immediately (<10 seconds)
- Fast, simple text summary optimized for quick retrieval
- Perfect for real-time applications

**Asynchronous (`202 Accepted`)** - When callback URL is provided:
- Returns task_id immediately
- Delivers full structured summary with metadata to webhook
- Includes retry logic (3 attempts with exponential backoff)
- Perfect for batch processing or when you need rich metadata

**Example Request (Sync):**
```bash
curl -X POST "http://localhost:8000/api/v1/agent/summaries" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "messages": "User conversation text here..."
  }'
```

**Example Request (Async with Callback):**
```bash
curl -X POST "http://localhost:8000/api/v1/agent/summaries" \
  -H "Content-Type: application/json" \
  -H "X-Callback-URL: https://your-webhook.com/callback" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "messages": "Long conversation text..."
  }'
```

**Features:**
- **Dual Mode:** Fast sync for immediate results, async with callback for rich metadata
- **Flexible Input:** Accepts string or structured message arrays
- **Smart Retry:** Automatic retry with exponential backoff for webhook delivery
- **Comprehensive:** Sync mode provides clean summaries, async mode includes full metadata

---

## System Endpoints

### Health Check

Check API health and SDK initialization status.

**Endpoint:** `GET /health`

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "sdk_initialized": true,
  "service": "omnimemory-api"
}
```

**Example Request:**
```bash
curl "http://localhost:8000/health"
```

---

### API Information

Get basic API information.

**Endpoint:** `GET /`

**Response:** `200 OK`
```json
{
  "name": "OmniMemory API",
  "version": "1.0.0",
  "description": "REST API for OmniMemory - Advanced Memory Management System",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "api": "/api/v1"
  }
}
```

**Example Request:**
```bash
curl "http://localhost:8000/"
```

---

### Connection Pool Stats

Inspect the vector database connection pool state for observability/debugging.

**Endpoint:** `GET /api/v1/system/pool-stats`

**Response:** `200 OK`
```json
{
  "max_connections": 30,
  "created_handlers": 12,
  "active_handlers": 4,
  "available_handlers": 8,
  "initialized": true
}
```

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/system/pool-stats"
```

---

## Agent Operations

### Summarize Conversation

Generate a conversation summary with the single-agent pipeline. Supports synchronous replies or asynchronous webhook delivery.

**Endpoint:** `POST /api/v1/agent/summaries`

**Request Body:**
```json
{
  "app_id": "string",
  "user_id": "string",
  "session_id": "string or null",
  "messages": [
    {
      "role": "user",
      "content": "Need help with rate limits",
      "timestamp": "2025-01-01T12:00:00Z"
    }
  ],
  "callback_url": "https://example.com/webhook",
  "callback_headers": {
    "Authorization": "Bearer token"
  }
}
```
- `messages` may also be provided as a single raw string instead of an array.
- When `callback_url` is present the API returns `202 Accepted` immediately and posts the summary to the webhook when ready.

**Synchronous Response (`200 OK`):**
```json
{
  "app_id": "my-app",
  "user_id": "user-123",
  "session_id": "session-456",
  "summary": "Concise narrative...",
  "key_points": "Important highlights",
  "tags": ["api", "support"],
  "keywords": ["rate limits", "throttling"],
  "semantic_queries": ["api rate limits"],
  "metadata": {
    "conversation_complexity": 2
  },
  "generated_at": "2025-01-01T12:34:56.789Z"
}
```

**Async Response (`202 Accepted`):**
```json
{
  "task_id": "uuid",
  "status": "accepted",
  "message": "Conversation summary scheduled for callback delivery",
  "app_id": "my-app",
  "user_id": "user-123",
  "session_id": "session-456"
}
```

---

## Interactive API Documentation

The API provides interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI JSON:** http://localhost:8000/openapi.json

---

## Rate Limiting

Currently, the API does not implement rate limiting. For production deployments, implement appropriate rate limiting mechanisms.

## CORS

CORS is currently configured to allow all origins (`*`). For production, configure appropriate CORS settings in `server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Configure appropriately
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## SDK Methods Reference

The API endpoints map to the following SDK methods:

### Memory Operations

| API Endpoint | SDK Method | Description |
|-------------|------------|-------------|
| `POST /api/v1/memories` | `add_memory()` | Create memory from user messages (full pipeline) |
| `GET /api/v1/memories/query` | `query_memory()` | Query memories with intelligent ranking |
| `GET /api/v1/memories/{id}` | `get_memory()` | Get a single memory by ID |
| `DELETE /api/v1/memories/{id}` | `delete_memory()` | Delete a memory from collection |

### Agent Operations

| API Endpoint | SDK Method | Description |
|-------------|------------|-------------|
| `POST /api/v1/agent/memories` | `add_agent_memory()` | Create memory from agent message (fast path) |
| `POST /api/v1/agent/summaries` | `summarize_conversation()` | Generate conversation summary (sync or async) |

---

## Architecture Notes

**Self-Evolving Composite Memory Synthesis Architecture (SECMSA)**

1. **Dual-Agent Construction:** Memories are created through parallel execution of Episodic and Summarizer agents, then synthesized into canonical memory notes.

2. **Composite Scoring:** Multi-dimensional ranking using `composite = relevance × (1 + recency_boost + importance_boost)` ensures relevance-first retrieval.

3. **Self-Evolution:** AI-powered conflict resolution agents autonomously execute UPDATE/DELETE/SKIP/CREATE operations to maintain memory coherence.

4. **Status-Driven Lineage:** Conflict resolution keeps memories simple by updating `status`, `status_reason`, and `updated_at` (e.g., consolidated, contradicted), avoiding complex version metadata.

5. **Asynchronous Processing:** Memory creation endpoints return immediately with a `task_id`, and the SDK completes creation/embedding/storage via internal `asyncio` background tasks.

6. **Message Validation:** Messages are validated to ensure they contain between 1 and 30 messages (configurable via `DEFAULT_MAX_MESSAGES`).

7. **Connection Pooling:** The API uses connection pooling for efficient vector database access. The SDK is initialized once at startup and reused for all requests.

8. **Error Handling:** All endpoints include comprehensive error handling and logging.

9. **Agent Operations:** Fast, single-agent workflows for agent-driven memory creation and conversation summarization. Optimized for speed (<10 seconds) with flexible input formats (string or structured arrays).

10. **Smart Webhook Delivery:** Conversation summaries support webhook callbacks with automatic retry logic (3 attempts, exponential backoff) and intelligent error handling (skips retries for permanent errors).

