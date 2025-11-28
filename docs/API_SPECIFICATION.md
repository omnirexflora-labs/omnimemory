# OmniMemory API Specification

Complete REST API contract for OmniMemory. This document focuses purely on endpoints, payloads, and responses (see `README.md` for setup and `docs/ARCHITECTURE.md` for system design).

## Base URL

```
http://localhost:8001/api/v1
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement appropriate authentication mechanisms.

## Content Type

All requests and responses use `application/json`.

For deployment profiles and vector database setup instructions, see the Quick Start section of `README.md`.

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

> **Identifier Length:** All `app_id`, `user_id`, and `session_id` values must be at least 10 characters. Requests that provide shorter identifiers will be rejected with `400 Bad Request`.

---

## Memory Operations

### Add Memory

Create a new memory from user messages asynchronously.

**Endpoint:** `POST /api/v1/memories`

**Request Body:**
```json
{
  "app_id": "string (required, min 10 chars)",
  "user_id": "string (required, min 10 chars)",
  "session_id": "string (optional, min 10 chars if provided)",
  "messages": [
    {
      "role": "string (required)",
      "content": "string (required)",
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
curl -X POST "http://localhost:8001/api/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "session_id": "session-456",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?",
      },
      {
        "role": "assistant",
        "content": "I am doing well, thank you!",
      }
    ]
  }'
```

**Error Responses:**
- `400 Bad Request` - If messages count differs from `OMNIMEMORY_DEFAULT_MAX_MESSAGES` (default: 10) or validation fails
- `500 Internal Server Error` - If memory creation fails

---

### Stream Messages (Memory Batcher)

Append role/content messages and let the server-side batcher automatically call `add_memory()` once your configured `OMNIMEMORY_DEFAULT_MAX_MESSAGES` window is reached.

**Endpoint:** `POST /api/v1/memory-batcher/messages`

**Request Body:**
```json
{
  "app_id": "string (required, min 10 chars)",
  "user_id": "string (required, min 10 chars)",
  "session_id": "string (optional, min 10 chars if provided)",
  "messages": [
    {
      "role": "string (required, user|assistant|system)",
      "content": "string (required)"
    }
  ]
}
```

The batcher keeps buffering per `(app_id, user_id, session_id)` and automatically Flushes via the full dual-agent pipeline when the batch size is met. Partial batches are left pending—no manual flush endpoint is exposed.

**Response:** `200 OK`
```json
{
  "app_id": "my-app-id",
  "user_id": "user-123",
  "session_id": "session-456",
  "pending_messages": 4,
  "batch_size": 10,
  "status": "pending",          // or "flushed" once the window is reached
  "last_delivery": "add_memory" // null when nothing has been flushed yet
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8001/api/v1/memory-batcher/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "messages": [
      {"role": "user", "content": "Booked a demo for Thursday"},
      {"role": "assistant", "content": "Confirmed calendar invite"}
    ]
  }'
```

**Notes:**
- The server maintains in-memory buffers per `(app_id, user_id, session_id)`.
- Each request appends messages to the corresponding buffer.
- When the number of buffered messages reaches `OMNIMEMORY_DEFAULT_MAX_MESSAGES` (default: 10), the server automatically calls `add_memory()` with the full batch.
- The response indicates the current `pending_messages` count and the `batch_size`.
- Once the batch window is met, the response will report `status: "flushed"` and include the last delivery metadata for troubleshooting.

---

### Query Memories

Query memories with intelligent multi-dimensional ranking.

**Endpoint:** `GET /api/v1/memories/query`

**Query Parameters:**
- `app_id` (required, string, min 10 chars) - Application ID
- `query` (required, string, min 10 chars) - Natural language query
- `user_id` (optional, string, min 10 chars if provided) - User ID filter
- `session_id` (optional, string, min 10 chars if provided) - Session ID filter
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
curl "http://localhost:8001/api/v1/memories/query?app_id=my-app-id&query=hello&user_id=user-123&n_results=5"
```
**Error Responses:**
- `400 Bad Request` - If required parameters are missing or validation fails
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
curl "http://localhost:8001/api/v1/memories/memory-123?app_id=my-app-id"
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
curl -X DELETE "http://localhost:8001/api/v1/memories/memory-123?app_id=my-app-id"
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
curl "http://localhost:8001/api/v1/memories/memory-123/evolution?app_id=my-app-id"
```

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
  "app_id": "string (required, min 10 chars)",
  "user_id": "string (required, min 10 chars)",
  "session_id": "string (optional, min 10 chars if provided)",
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
curl -X POST "http://localhost:8001/api/v1/agent/memories" \
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
curl -X POST "http://localhost:8001/api/v1/agent/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "messages": [
      {"role": "agent", "content": "User asked about pricing"},
      {"role": "user", "content": "What are your plans?"}
    ]
  }'
```

**Features:**
- **Fast Processing:** Uses optimized single-agent summary generation (<5 seconds)
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
  "app_id": "string (required, min 10 chars)",
  "user_id": "string (required, min 10 chars)",
  "session_id": "string (optional, min 10 chars if provided)",
  "messages": "string or array of message objects",
  "callback_url": "string (optional)",
  "callback_headers": {
    "Authorization": "Bearer token"
  }
}
```

**Response Modes:**

**Synchronous (`200 OK`)** - When no callback URL is provided:
- Returns summary immediately (<5 seconds)
- Fast, simple text summary optimized for quick retrieval
- Perfect for real-time applications

**Asynchronous (`202 Accepted`)** - When callback URL is provided:
- Returns task_id immediately
- Delivers full structured summary with metadata to webhook
- Includes retry logic (3 attempts with exponential backoff)
- Perfect for batch processing or when you need rich metadata

**Example Request (Sync):**
```bash
curl -X POST "http://localhost:8001/api/v1/agent/summaries" \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "my-app-id",
    "user_id": "user-123",
    "messages": "User conversation text here..."
  }'
```

**Example Request (Async with Callback):**
```bash
curl -X POST "http://localhost:8001/api/v1/agent/summaries" \
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
curl "http://localhost:8001/health"
```

---

### API Information

Get basic API information.

**Endpoint:** `GET /`

**Response:** `200 OK`
```json
{
  "name": "OmniMemory API",
  "version": "0.0.1",
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
curl "http://localhost:8001/"
```

---

### Connection Pool Stats

Inspect the vector database connection pool state for observability/debugging.

**Endpoint:** `GET /api/v1/system/pool-stats`

**Response:** `200 OK`
```json
{
  "max_connections": 10,
  "created_handlers": 12,
  "active_handlers": 4,
  "available_handlers": 8,
  "initialized": true
}
```

**Example Request:**
```bash
curl "http://localhost:8001/api/v1/system/pool-stats"
```

---

## Interactive API Documentation

The API provides interactive documentation:

- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc
- **OpenAPI JSON:** http://localhost:8001/openapi.json

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


