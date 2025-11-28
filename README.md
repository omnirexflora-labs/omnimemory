# OmniMemory · The Living Brain for Autonomous Agents

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-89.58%25-green)](https://github.com/omnirexflora-labs/omnimemory)

**Don't just store data. Synthesize memories.**
OmniMemory transforms static embeddings into a self-evolving cognitive substrate.

[Quick Start](#quick-start) · [CLI](#cli-tool) · [SDK](#sdk-usage-guide) · [Production](#production-deployment) · [Environment Variables](#environment-variables) · [Architecture](#architecture) · [REST API](docs/API_SPECIFICATION.md)

</div>

---

## Why OmniMemory?

Traditional RAG is a filing cabinet: you put documents in, you take documents out.
**OmniMemory is a living brain.**

It doesn't just "store" messages. It employs a **Dual-Agent Synthesis** engine to interpret conversations, extract behavioral patterns, and resolve contradictions automatically. When a new memory conflicts with an old one, OmniMemory doesn't just append—it **updates**, **deletes**, or **consolidates** the knowledge graph, just like human memory.

| Feature | Traditional Vector RAG | OmniMemory (SECMSA) |
| :--- | :--- | :--- |
| **Input Handling** | Naive chunking & embedding | **Dual-Agent Synthesis** (Episodic + Summarizer) |
| **Conflict Resolution** | None (contradictions coexist) | **Self-Evolving** (Update/Delete/Skip operations) |
| **Retrieval Logic** | Cosine similarity only | **Composite Scoring** (Relevance × [1 + Recency + Importance]) |
| **Context Awareness** | Static text chunks | **Structured Memory Notes** (Behavior, Learnings, Guidance) |
| **Multi-Tenancy** | Often manual filtering | **Native Isolation** (App / User / Session tiers) |

## Core Features

### 1. Dual-Agent Synthesis
Two specialized agents work in parallel to process every interaction:
*   **Episodic Agent**: Analyzes *behavior*. "User prefers concise answers," "User struggles with async concepts."
*   **Summarizer Agent**: Analyzes *narrative*. "Project X is delayed," "Deployed v2.0 to prod."

### 2. Self-Evolving Memory
Memories aren't static. The system automatically detects conflicts between new and existing information.
*   **UPDATE**: Merges fragmented details into a single, comprehensive note.
*   **DELETE**: Removes outdated or contradicted information.
*   **SKIP**: Ignores redundant inputs to keep the index clean.

### 3. Composite Scoring
We don't just return the "nearest neighbor." We return the most *useful* memory.
```python
Score = Relevance * (1 + Recency_Boost + Importance_Boost)
```
This ensures high-relevance memories always win, but recent and critical memories get the nudge they need to surface.

### 4. Enterprise Multi-Tenancy
Built for SaaS from day one.
*   **App Level**: Physical isolation (separate collections).
*   **User Level**: Logical isolation (metadata filtering).
*   **Session Level**: Conversation grouping.

## Supported Backends

Switch providers by changing `OMNI_MEMORY_PROVIDER`. No code changes required.

| Provider | Env Value | Best For |
| :--- | :--- | :--- |
| **Qdrant** | `qdrant-remote` | Production default. High performance, rich filtering |
| **ChromaDB** | `chromadb-remote` | Simple deployments, local development |
| **PostgreSQL** | `postgresql` | Teams already using Postgres (via pgvector) |
| **MongoDB** | `mongodb` | Atlas users needing vector search + document store |

## When to Use: API vs SDK

### Use the REST API Server (Recommended for Production)
**Why**: Language-agnostic. Works with **any** programming language (Node.js, Go, Rust, Java, PHP, etc.)

**Best For**:
- ✅ Production deployments
- ✅ Multi-language teams
- ✅ Microservices architectures
- ✅ Need built-in metrics, health checks, connection pooling

### Use the Python SDK (Dev/Prototyping)
**Why**: Direct Python integration for rapid testing

**Best For**:
- ✅ Python-only agents
- ✅ Local development and testing
- ✅ Prototyping memory operations

---

## Quick Start

> **TL;DR**: Want to see it in action immediately?
> ```bash
> # Run the complete Customer Support Agent example
> python examples/complete_sdk_example.py
> ```

### 1. Install
```bash
uv add omnimemory
# or
pip install omnimemory
```

### 2. Configure
Create a `.env` file (templates in [`examples/env/`](examples/env)):
```bash
# LLM & Embeddings
LLM_API_KEY=sk-...
LLM_PROVIDER=openai
EMBEDDING_API_KEY=sk-...
EMBEDDING_PROVIDER=openai

# Vector DB (Choose one: qdrant, chromadb, postgresql, mongodb)
OMNI_MEMORY_PROVIDER=qdrant-remote
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Run (Choose Your Backend)

Start the vector DB and API server:

#### **Qdrant** (Production Default - High Performance)
```bash
docker compose -f docker-compose.local.yml --profile qdrant up -d
uv run uvicorn omnimemory.api.server:app --host 0.0.0.0 --port 8001 --reload
```

#### **ChromaDB** (Simple Deployments)
```bash
docker compose -f docker-compose.local.yml --profile chromadb up -d
uv run uvicorn omnimemory.api.server:app --host 0.0.0.0 --port 8001 --reload
```

#### **PostgreSQL** (Existing Postgres Users)
```bash
docker compose -f docker-compose.local.yml --profile pgvector up -d
uv run uvicorn omnimemory.api.server:app --host 0.0.0.0 --port 8001 --reload
```

#### **MongoDB** (Configure MongoDB Atlas separately)
```bash
# Set MONGO_URI in .env first
uv run uvicorn omnimemory.api.server:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Use (Python SDK)
```python
from omnimemory.sdk import OmniMemorySDK
from omnimemory.core.schemas import UserMessages, Message
import asyncio

async def main():
    sdk = OmniMemorySDK()
    
    # CRITICAL: Initialize connection pools
    if not await sdk.warm_up():
        print("Failed to warm up SDK")
        return

    # Add a memory (returns a background task ID)
    response = await sdk.add_memory(UserMessages(
        app_id="my-app",
        user_id="user-123",
        messages=[
            Message(role="user", content="I'm building a Python web scraper."),
            Message(role="assistant", content="I can help with libraries like BeautifulSoup.")
        ] * 5 # Need sufficient context (default 10 messages)
    ))
    
    task_id = response["task_id"]
    print(f"Memory processing started. Task ID: {task_id}")
    
    # Fire-and-Forget: Memory processes in background
    # No need to poll - check logs if debugging needed
    await asyncio.sleep(3)  # Give it time to process

    # Query memory (semantic search)
    results = await sdk.query_memory(
        app_id="my-app",
        user_id="user-123",
        session_id="session-123", # Optional
        n_results=1, # Optional
        similarity_threshold=0.7, # Optional
        query="What is the user working on?"
    )
    
    print(results[0]['memory_note']) 
    # Output: "User is developing a Python-based web scraper..."

asyncio.run(main())
```

### 5. Quick Test Examples

Want to see it in action? We provide complete, real-world examples for both SDK and API usage.

**Run the SDK Example:**
```bash
# Demonstrates full customer support workflow with memory batching
python examples/complete_sdk_example.py
```

**Run the API Example:**
```bash
# Requires running server (uv run uvicorn omnimemory.api.server:app --port 8001)
python examples/complete_api_example.py
```

---

## Production Features

**Fully Asynchronous**: O(1) latency from user perspective. Memory synthesis happens in **fire-and-forget** background tasks. No polling needed - check logs if debugging required.

**Connection Pooling**: Configurable pool size (default 10) for high concurrency workloads.

**Metrics & Observability**: Prometheus-compatible metrics at `http://localhost:9001/metrics` (enable with `OMNIMEMORY_ENABLE_METRICS_SERVER=true`).

**Multi-Tenancy**: 3-tier isolation (app/user/session) for SaaS deployments. Complete data separation.

**89.58% Test Coverage**: Production-grade reliability with comprehensive test suite.

**Language Agnostic**: REST API works with any language. Python SDK provided for convenience.

---

## Agent Memory SDK

**For Agents That Need to Answer Questions Using Stored Memories**

The `AgentMemorySDK` provides a complete "query memory + generate answer" loop. It retrieves relevant memories and calls your LLM with context to generate grounded responses.

```python
from omnimemory import AgentMemorySDK

agent_sdk = AgentMemorySDK()

response = await agent_sdk.answer_query(
    app_id="my-app-id-1234",
    query="What does the user prefer?",
    user_id="user-123456", # Optional
    session_id="session-123456", # Optional
    n_results=5, # Optional
    similarity_threshold=0.7 # Optional
)

print(response["answer"])  # LLM-generated answer
print(f"Based on {len(response['memories'])} memories")
```

**Use When**: Your agent needs to answer user questions using stored memories.

**Not For**: Storing new memories (use `OmniMemorySDK.add_memory` or `add_agent_memory` for that).

---

## CLI Tool

OmniMemory includes a powerful command-line interface for quick operations and testing:

```bash
# Install
uv add omnimemory

# Get help
omnimemory --help

# Start daemon for background operations
omnimemory daemon start

# Add memory
omnimemory memory add \
  --app-id "myapp-1234567890" \
  --user-id "user-1234567890" \
  --message "user:I prefer dark mode" \
  --message "assistant:Noted, I'll remember that"

# Query memory
omnimemory memory query \
  --app-id "myapp-1234567890" \
  --query "user preferences"

# Check system health
omnimemory health

# View comprehensive feature guide
omnimemory info

# Daemon management
omnimemory daemon status
omnimemory daemon stop
```

**Available Commands**:
- `omnimemory memory` - Memory operations (add, query, get, delete)
- `omnimemory memory batch` - Batch message operations
- `omnimemory daemon` - Background daemon management
- `omnimemory agent` - Agent-specific operations
- `omnimemory health` - System health diagnostics
- `omnimemory info` - Feature overview

For detailed CLI documentation, run `omnimemory --help`.

---

## SDK Usage Guide

> **Note**: This guide covers the Python SDK. For the HTTP REST API, see [API_SPECIFICATION.md](docs/API_SPECIFICATION.md).

### Initialization

**CRITICAL**: You must warm up the connection pools before making requests.

**Why**: Initializes vector DB connections for low latency on first request.

```python
from omnimemory.sdk import OmniMemorySDK

sdk = OmniMemorySDK()
success = await sdk.warm_up()
if not success:
    print("Failed to initialize connections")
```

---

## Core Memory Operations

### 1. Add Memory (`add_memory`)

**Use Case**: Primary engine for conversation analysis with **Dual-Agent Synthesis**.

**Why**: Needs a flow of conversation (default 10 messages) to understand context. The Episodic and Summarizer agents extract behavioral patterns and resolve conflicts. Single messages won't work.

**Parameters**:
- `user_message: UserMessages` - Contains `app_id`, `user_id`, `session_id` (optional), `messages` (list of Message objects)
- `messages` must have exactly `OMNIMEMORY_DEFAULT_MAX_MESSAGES` (default 10)

**Returns**: Task ID immediately (async processing)

```python
from omnimemory.core.schemas import UserMessages, Message

response = await sdk.add_memory(UserMessages(
    app_id="my-app-id-1234",
    user_id="user-123456",
    session_id="session-789",  # Optional
    messages=[
        Message(role="user", content="I prefer dark mode"),
        Message(role="assistant", content="Noted, I'll remember that")
        # ... total 10 messages required
    ]
))
task_id = response["task_id"]
print(f"Processing in background: {task_id}")
```

### 2. Add Agent Memory (`add_agent_memory`)

**Use Case**: **Agent Tool** for quick saves.

**Why**: When your agent learns new info or user says "save this," the agent calls this directly. Accepts **both structured and unstructured** messages. Bypasses conflict resolution for speed.

**Best Practice**: Add to agent system prompt as a tool.

**Parameters**:
- `agent_request: AgentMemoryRequest` - Contains `app_id`, `user_id`, `session_id` (optional), `messages` (string or list)

**Returns**: Task ID immediately

```python
from omnimemory.core.schemas import AgentMemoryRequest

# Unstructured (string)
response = await sdk.add_agent_memory(AgentMemoryRequest(
    app_id="my-app-id-1234",
    user_id="user-123456",
    messages="User completed premium signup and selected annual plan"
))

# Structured (list)
response = await sdk.add_agent_memory(AgentMemoryRequest(
    app_id="my-app-id-1234",
    user_id="user-123456",
    messages=[
        {"role": "user", "content": "What's my email?"},
        {"role": "assistant", "content": "It's user@example.com"}
    ]
))
```

### 3. Query Memory (`query_memory`)

**Use Case**: Retrieve memories using semantic search and composite scoring.

**How**: Uses Relevance × (1 + Recency + Importance) scoring.

**Parameters**:
- `app_id: str` (required)
- `query: str` (required) - Natural language query
- `user_id: str` (optional) - Filter by user
- `session_id: str` (optional) - Filter by session
- `n_results: int` (optional, default from env) - Max results to return
- `similarity_threshold: float` (optional, default from env) - Min similarity (0.0-1.0). **Overrides** `OMNIMEMORY_RECALL_THRESHOLD` env var.

**Returns**: List of memory dictionaries

```python
results = await sdk.query_memory(
    app_id="my-app-id-1234",
    query="What does the user like?",
    user_id="user-123456",          # Optional
    session_id="session-789",       # Optional
    n_results=10,                   # Optional (default 5)
    similarity_threshold=0.75       # Optional (overrides env default 0.3)
)

for memory in results:
    print(memory["document"])
    print(f"Score: {memory['composite_score']}")
```

### 4. Get Memory (`get_memory`)

**Use Case**: Retrieve a single memory by its ID.

**Why**: When you have a memory ID from a previous operation and need full content.

**Parameters**:
- `memory_id: str` (required)
- `app_id: str` (required)

**Returns**: Memory dict or None

```python
memory = await sdk.get_memory(
    memory_id="uuid-1234-5678",
    app_id="my-app-id-1234"
)
if memory:
    print(memory["document"])
```

### 5. Delete Memory (`delete_memory`)

**Use Case**: Manual memory deletion (GDPR, cleanup).

**Why**: User requests deletion or you need to remove test data.

**Parameters**:
- `app_id: str` (required)
- `doc_id: str` (required) - Document ID to delete

**Returns**: Boolean (success/failure)

```python
success = await sdk.delete_memory(
    app_id="my-app-id-1234",
    doc_id="uuid-1234-5678"
)
if success:
    print("Memory deleted")
```

---

## Summarization

### Summarize Conversation (`summarize_conversation`)

**Use Case**: **Context Window Management**.

**Why**: When working memory is full, generate a summary, save it, delete old messages to free tokens.

**Accepts**: Both structured and unstructured messages.

**Two Modes**:

#### Sync Mode (No `callback_url`)
- **Returns**: Summary immediately
- **Processing**: Fast (`use_fast_path=True`)
- **Use When**: Real-time responses needed, short contexts

```python
from omnimemory.core.schemas import ConversationSummaryRequest

summary = await sdk.summarize_conversation(ConversationSummaryRequest(
    app_id="my-app-id-1234",
    user_id="user-123456",
    messages=[...]  # Structured or unstructured
))
print(summary["content"])
print(summary["delivery"])  # "sync"
```

#### Webhook Mode (With `callback_url`)
- **Returns**: Task ID immediately
- **Processing**: Full structured summary (`use_fast_path=False`)
- **Delivery**: POSTs result to your webhook URL
- **Retry**: 3 attempts with exponential backoff
- **Use When**: Long conversations, background processing, need auto-replacement

**Parameters**:
- `summary_request: ConversationSummaryRequest`
  - `app_id: str` (required)
  - `user_id: str` (required)
  - `session_id: str` (optional)
  - `messages: str | list` (required)
  - `callback_url: str` (optional) - If provided, enables webhook mode
  - `callback_headers: dict` (optional) - Custom headers for webhook (e.g., auth)

```python
response = await sdk.summarize_conversation(ConversationSummaryRequest(
    app_id="my-app-id-1234",
    user_id="user-123456",
    messages=[...],  # Long conversation
    callback_url="https://api.myapp.com/webhooks/summary",
    callback_headers={"Authorization": "Bearer token123"}
))
print(response["task_id"])
print(response["status"])  # "accepted"
```

---

## Batching

### Memory Batcher (`memory_batcher_add_message`)

**Use Case**: Streaming chat loops.

**Why**: Automatically buffers messages and calls `add_memory` when limit is reached. No manual counting.

**How**: Non-blocking. Monitors message count per `(app_id, user_id, session_id)` tuple. When it hits `OMNIMEMORY_DEFAULT_MAX_MESSAGES` (default 10), auto-flushes.

**Parameters**:
- `app_id: str` (required)
- `user_id: str` (required)
- `session_id: str` (optional)
- `role: str` (required) - "user", "assistant", "system"
- `content: str` (required)

```python
# In your chat loop
for message in stream:
    await sdk.memory_batcher_add_message(
        app_id="my-app-id-1234",
        user_id="user-123456",
        role=message.role,
        content=message.content
    )
    # SDK handles auto-flush at 10 messages
```

---

## Evolution & Auditing

### 1. Traverse Evolution Chain (`traverse_memory_evolution_chain`)

**Use Case**: See how a memory evolved over time.

**Why**: Memories update, delete, merge. This traces the full history.

**How**: Follows `next_id` pointers from original to final memory.

**Parameters**:
- `app_id: str` (required)
- `memory_id: str` (required) - Starting memory ID

**Returns**: List of memories in chronological order

```python
chain = await sdk.traverse_memory_evolution_chain(
    app_id="my-app-id-1234",
    memory_id="original-uuid-1234"
)
print(f"Memory evolved {len(chain)} times")
for memory in chain:
    print(f"{memory['metadata']['status']} - {memory['document'][:50]}")
```

### 2. Generate Evolution Graph (`generate_evolution_graph`)

**Use Case**: Visualize evolution chain.

**Formats**: `mermaid`, `dot`, `html`

**Parameters**:
- `chain: List[Dict]` (required) - Output from `traverse_memory_evolution_chain`
- `format: str` (required) - "mermaid", "dot", or "html"

```python
chain = await sdk.traverse_memory_evolution_chain(...)

# Mermaid (for docs)
mermaid = sdk.generate_evolution_graph(chain, format="mermaid")
print(mermaid)

# HTML (for browser visualization)
html = sdk.generate_evolution_graph(chain, format="html")
with open("evolution.html", "w") as f:
    f.write(html)
```

### 3. Generate Evolution Report (`generate_evolution_report`)

**Use Case**: Detailed analysis of memory changes.

**Formats**: `markdown`, `text`, `json`

**Parameters**:
- `chain: List[Dict]` (required)
- `format: str` (required) - "markdown", "text", or "json"

```python
report = sdk.generate_evolution_report(chain, format="markdown")
print(report)  # Includes stats, timeline, insights
```

---

## System & Monitoring

### Connection Pool Stats (`get_connection_pool_stats`)

**Use Case**: Production monitoring and debugging.

**Why**: If queries are slow, check if you're hitting connection limits.

**Returns**: Dict with pool metrics

```python
stats = await sdk.get_connection_pool_stats()
print(f"Active: {stats['active_handlers']}/{stats['max_connections']}")
print(f"Available: {stats['available_handlers']}")
```

---

## Configuration & Tuning

Tune these hyperparameters in your `.env` file to optimize for your specific use case.

| Parameter | Default | Description | Tuning Guide |
| :--- | :--- | :--- | :--- |
| `OMNIMEMORY_RECALL_THRESHOLD` | `0.3` | Minimum cosine similarity for initial retrieval from Vector DB. | Lower to `0.2` for broader recall (more noise); raise to `0.5` for stricter relevance. |
| `OMNIMEMORY_COMPOSITE_SCORE_THRESHOLD` | `0.5` | Minimum *final* score (Relevance × Boosts) to return a memory. | Raise to `0.6+` if you only want high-confidence memories. Lowering it may return less relevant but "important" or "recent" memories. |
| `OMNIMEMORY_LINK_THRESHOLD` | `0.7` | Similarity required to "link" memories for conflict resolution. | Lower to `0.6` to trigger evolution/updates more often. Raise to `0.8` to reduce "noise" and only link very similar topics. |
| `OMNIMEMORY_DEFAULT_MAX_MESSAGES` | `10` | Number of messages required for `add_memory`. | Match this to your LLM's context window preference. Too low = poor synthesis; Too high = context bloat. |
| `OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS` | `10` | Max concurrent DB connections. | Reduce to `3-5` for low-resource environments (e.g., local dev). Increase for high-throughput production. |

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
| :--- | :--- | :--- |
| `LLM_API_KEY` | LLM provider API key | `sk-...` |
| `LLM_PROVIDER` | LLM provider name | `openai`, `anthropic`, `mistral` |
| `EMBEDDING_API_KEY` | Embedding provider API key | `sk-...` |
| `EMBEDDING_PROVIDER` | Embedding provider | `openai` |
| `OMNI_MEMORY_PROVIDER` | Vector DB backend | `qdrant-remote`, `chromadb-remote`, `postgresql`, `mongodb` |

### LLM Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_MODEL` | - | Model name (e.g., `gpt-4`, `claude-3-opus`) |
| `LLM_TEMPERATURE` | `0.4` | Creativity (0.0-2.0) |
| `LLM_MAX_TOKENS` | `3000` | Max response tokens |
| `LLM_TOP_P` | `0.9` | Nucleus sampling |

### Embedding Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL` | - | Embedding model name |
| `EMBEDDING_DIMENSIONS` | - | Vector dimensions |
| `EMBEDDING_ENCODING_FORMAT` | `base64` | Response encoding format |
| `EMBEDDING_TIMEOUT` | `600` | Request timeout (seconds) |

### Vector Database

**Qdrant**:
- `QDRANT_HOST` - Qdrant server host
- `QDRANT_PORT` - Qdrant port (default 6333)

**ChromaDB**:
- `CHROMA_HOST` - ChromaDB server host
- `CHROMA_PORT` - ChromaDB port (default 8000)
- `CHROMA_AUTH_TOKEN` - Authentication token
- `CHROMA_CLIENT_TYPE` - Client type (`remote` for server)

**PostgreSQL**:
- `POSTGRES_URI` - Full connection string (e.g., `postgresql://user:pass@host:5432/db`)

**MongoDB**:
- `MONGO_URI` - MongoDB Atlas connection string

### Observability

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OMNIMEMORY_ENABLE_METRICS_SERVER` | `false` | Enable Prometheus metrics endpoint |
| `OMNIMEMORY_METRICS_PORT` | `9001` | Metrics HTTP server port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_DIR` | `./logs` | Log file directory path |

---

## Production Deployment

### Step 1: Prepare Environment Variables

Create a `.env` file with all required configuration:

```bash
# LLM Configuration (Required)
LLM_API_KEY=your-api-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=3000

# Embedding Configuration (Required)
EMBEDDING_API_KEY=your-api-key-here
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=1536

# Vector Database (Choose one)
OMNI_MEMORY_PROVIDER=qdrant-remote
QDRANT_HOST=your-qdrant-host.com
QDRANT_PORT=6333

# OmniMemory Hyperparameters (Optional - tune for your use case)
OMNIMEMORY_DEFAULT_MAX_MESSAGES=10
OMNIMEMORY_RECALL_THRESHOLD=0.3
OMNIMEMORY_COMPOSITE_SCORE_THRESHOLD=0.4
OMNIMEMORY_LINK_THRESHOLD=0.8
OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS=10

# Metrics & Observability (Optional)
OMNIMEMORY_ENABLE_METRICS_SERVER=true
OMNIMEMORY_METRICS_PORT=9001
LOG_LEVEL=INFO
```

### Step 2: Deploy with Docker Compose

```bash
# Start with your chosen backend
docker compose -f docker-compose.local.yml --profile qdrant up -d
```

### Step 3: Production Hardening

> ⚠️ **CRITICAL SECURITY WARNING**: The provided `docker-compose.local.yml` is designed for **local development only**. For production deployments, you **MUST** implement the following security measures:

#### 1. Enable HTTPS

Add a reverse proxy (nginx, Traefik, or Caddy) with SSL certificates:

```yaml
# Add to your docker-compose
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - api-qdrant
```

#### 2. Implement Authentication

- Use API keys in request headers
- Implement OAuth 2.0 or JWT tokens
- Configure authentication middleware in nginx/Traefik

#### 3. Use Secrets Management

```bash
# Don't use .env files in production
# Use Docker secrets or cloud provider secrets management
docker secret create llm_api_key ./llm_key.txt
docker secret create embedding_api_key ./embedding_key.txt
```

#### 4. Network Security

```bash
# Configure firewall to only expose necessary ports:
# - 443 (HTTPS) - public facing
# - 6333 (Qdrant) - internal network only
# - 9001 (Metrics) - internal network only

# Example: UFW firewall rules
sudo ufw allow 443/tcp
sudo ufw enable
```

#### 5. Enable Monitoring

```bash
# Start with monitoring profile for Prometheus + Grafana
docker compose -f docker-compose.local.yml \
  --profile qdrant \
  --profile monitoring up -d

# Access Grafana at http://localhost:3000 (default: admin/admin)
# Access Prometheus at http://localhost:9090
# Configure alerts and dashboards for production monitoring
```

#### 6. Backup & Disaster Recovery

- Configure automated backups for your vector database
- Test recovery procedures regularly
- Use persistent volumes for data

---

## Development & Testing

### Running Tests

```bash
# Install development dependencies
uv sync

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/omnimemory --cov-report=html

# View coverage report
open htmlcov/index.html
```

**Current Test Coverage**: 89.58%

### Running Locally

```bash
# Start vector database
docker compose -f docker-compose.local.yml --profile qdrant up -d

# Run API server in development mode
uv run uvicorn omnimemory.api.server:app --host 0.0.0.0 --port 8001 --reload

# Or use the provided script
python run_api_server.py
```

---

## Architecture

OmniMemory implements the **Self-Evolving Composite Memory Synthesis Architecture (SECMSA)**.

For comprehensive architecture documentation:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Deep dive into SECMSA, mathematical foundations, scoring algorithms, conflict resolution, and design decisions
- **[C4_ARCHITECTURE.md](docs/C4_ARCHITECTURE.md)** - Visual system architecture with PlantUML diagrams:
  - Level 1: System Context
  - Level 2: Container Diagram
  - Level 3: Component Diagram
  - Level 4: Code Structure
  - Sequence diagrams for memory creation and retrieval flows

---



## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1.  Clone: `git clone https://github.com/omnirexflora-labs/omnimemory`
2.  Sync: `uv sync --group dev`
3.  Test: `uv run pytest`

## License

MIT © [OmniRexFlora Labs](https://github.com/omnirexflora-labs)
