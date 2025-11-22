# OmniMemory - Self-Evolving Composite Memory Synthesis Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready memory framework for autonomous AI agents. OmniMemory gives your agents human-like memory capabilities - they remember, learn, and adapt just like humans do.

## Table of Contents

- [Why OmniMemory?](#why-omnimemory)
- [Quick Start](#quick-start)
- [Memory & Multi-Tenant Architecture](#memory--multi-tenant-architecture)
  - [How Memory Works](#how-memory-works)
  - [Multi-Tenant Isolation](#multi-tenant-isolation)
  - [Composite Scoring with Exponential Decay](#composite-scoring-with-exponential-decay)
- [Integration Methods](#integration-methods)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Architecture Overview](#architecture-overview)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Why OmniMemory?

Traditional AI memory systems are like filing cabinets - they store information but don't understand, organize, or evolve. OmniMemory is different - it works like human cognitive memory.

**How Human Memory Works:**
- When you learn something new, your brain doesn't just store it - it connects it to related memories
- Over time, similar memories merge and consolidate (you don't remember every detail of every conversation)
- When you recall information, you don't search through everything - your brain prioritizes what's relevant, recent, and important
- When new information contradicts old memories, your brain resolves the conflict and updates your understanding

**How OmniMemory Works:**
- **Dual-Agent Synthesis**: Two specialized agents (Episodic and Summarizer) work together to understand conversations, just like your brain uses different regions for different types of memory
- **Self-Evolution**: Memories automatically merge, consolidate, and resolve conflicts - no manual intervention needed
- **Composite Scoring**: Retrieval prioritizes relevance first, then considers recency and importance - exactly like human recall
- **Memory Linking**: Related memories form connections, creating a knowledge graph that grows organically

**The Result:** Your AI agents remember context across conversations, learn from interactions, and adapt their understanding - just like humans do.

For detailed technical architecture, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick Start

### 1. Install

```bash
git clone https://github.com/omnirexflora-labs/omnimemory.git
cd omnimemory
uv sync  # or: pip install -e .
```

### 2. Configure

Create `.env` file:
```bash
# LLM Configuration
LLM_API_KEY=your_llm_api_key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# Embedding Configuration
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# Qdrant (auto-configured in docker-compose)
OMNI_MEMORY_PROVIDER=qdrant-remote
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

### 3. Start Services

```bash
docker-compose -f docker-compose.local.yml up -d
```

### 4. Verify

```bash
curl http://localhost:8001/health
```

## Memory & Multi-Tenant Architecture

### How Memory Works

OmniMemory creates memories through **dual-agent synthesis**: two specialized agents (Episodic and Summarizer) work in parallel to understand conversations. Episodic agent extracts behavioral patterns and interaction dynamics, while Summarizer agent creates narrative summaries and retrieval metadata. These outputs are combined into structured memory notes that are embedded and stored.

**Memory Creation Paths:**
1. **Standard Memory**: Full dual-agent synthesis with conflict resolution (rich conversational context)
2. **Agent Memory**: Fast single-agent path (quick agent message storage)
3. **Conversation Summary**: Standalone summarization with optional webhooks

**Self-Evolution**: When new memories conflict with existing ones, AI-powered agents automatically resolve conflicts through UPDATE (consolidate), DELETE (eliminate), SKIP (preserve), or CREATE (new memory) operations. This creates evolution chains where memories link forward, forming a knowledge graph.

### Multi-Tenant Isolation

**Core Selling Point:** OmniMemory is built for **multi-agent, multi-user, multi-app** scenarios from day zero. Deploy one instance, serve multiple applications, users, and agents - all with complete data isolation.

**Three-Tier Isolation Model:**

**`app_id` (Required) - Application-Level Isolation:**
- Each application gets its own memory collection in the vector database
- Complete physical separation - memories from different apps are stored in different collections
- Use case: Deploy one OmniMemory instance for "customer-support-app", "sales-assistant-app", "internal-tool-app" - all completely isolated
- Queries always require `app_id` - you can only access memories from the specified app

**`user_id` (Required) - User-Level Isolation:**
- Identifies which user within an application owns the memory
- Multiple users in the same app are completely isolated - user-123 can never see user-456's memories
- Stored as metadata, used for query-time filtering
- When you query with `user_id`, you only get that user's memories. Without it, you can search across all users in the app

**`session_id` (Optional) - Session-Level Grouping:**
- Groups related conversations together (like a support ticket or onboarding flow)
- Stored as metadata, can be used for filtering
- Use case: "support-ticket-789", "onboarding-session-456", "checkout-flow-123"
- Optional - if provided, you can query memories for a specific session, or omit it to search across all sessions

**Example:**
```python
# Multi-app: same user, different apps (completely isolated)
await sdk.add_memory(UserMessages(app_id="support-app", user_id="user-123", ...))
await sdk.add_memory(UserMessages(app_id="sales-app", user_id="user-123", ...))  # No data leakage

# Multi-user: different users, same app (isolated by user_id)
await sdk.add_memory(UserMessages(app_id="support-app", user_id="user-123", ...))
await sdk.add_memory(UserMessages(app_id="support-app", user_id="user-456", ...))  # Isolated

# Query with filters
results = await sdk.query_memory(
    app_id="support-app",
    user_id="user-123",  # Optional: filter to specific user
    session_id="ticket-789",  # Optional: filter to specific session
    query="..."
)
```

**Data Isolation Guarantees:**
- Different apps = different collections (physical separation)
- Different users = metadata filtering (logical separation)
- Different sessions = metadata filtering (logical separation)
- No data leakage possible - system enforces isolation at storage and query level

For detailed isolation guarantees, see [ARCHITECTURE.md](docs/ARCHITECTURE.md#multi-tenant-architecture-and-data-isolation).

### Composite Scoring with Exponential Decay

OmniMemory uses **composite scoring** to rank memories during retrieval, just like human recall prioritizes relevant, recent, and important information.

**Formula:** `composite_score = relevance × (1 + recency_boost + importance_boost)`

**How It Works:**
- **Relevance** (primary): Semantic similarity from vector search - always the base score
- **Recency Boost**: Uses exponential decay - recent memories get a small boost that decreases over time
- **Importance Boost**: Content significance signal - important memories get a small boost
- **Max boosts**: Each boost capped at 10% to ensure relevance remains primary

**Why Exponential Decay for Recency?**
- Recent memories are naturally more accessible (like human recall)
- Exponential decay ensures the boost decreases smoothly over time
- Very old memories get no recency boost, but can still rank high if highly relevant
- This prevents temporally recent but semantically irrelevant memories from dominating results

**Example:** A memory from yesterday with 0.7 relevance might score 0.77 (with 10% recency boost), while a memory from last month with 0.8 relevance scores 0.8 (no recency boost). The more relevant memory still wins, but recent memories get a helpful nudge.

For mathematical foundations and proofs, see [ARCHITECTURE.md](docs/ARCHITECTURE.md#composite-scoring-function).

## Integration Methods

### REST API (Recommended for Production)

Deploy once, call from any language. Best for production deployments.

```bash
# Deploy
docker-compose -f docker-compose.local.yml up -d

# Use from any language
curl -X POST http://localhost:8001/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"app_id": "my-app", "user_id": "user-123", "messages": [...]}'
```

**Interactive API Docs:** http://localhost:8001/docs

### Python SDK (Best for Python Codebases)

Direct integration into your Python application.

```python
from omnimemory.sdk import OmniMemorySDK
from omnimemory.core.schemas import UserMessages, Message

async def main():
    sdk = OmniMemorySDK()
    await sdk.warm_up()  # Always warm up first!
    
    # Create memory
    user_messages = UserMessages(
        app_id="my-app",
        user_id="user-123",
        session_id="session-456",
        messages=[
            Message(role="user", content="...", timestamp="2024-01-15T10:00:00Z"),
            Message(role="assistant", content="...", timestamp="2024-01-15T10:00:05Z")
            # ... must provide exactly OMNIMEMORY_DEFAULT_MAX_MESSAGES (default: 30) messages
        ]
    )
    result = await sdk.add_memory(user_messages)
    
    # Query memories
    results = await sdk.query_memory(
        app_id="my-app",
        query="Python async",
        user_id="user-123",
        n_results=5,
        similarity_threshold=0.35  # Optional: override default recall threshold
    )

asyncio.run(main())
```

### CLI (Testing Only)

For testing and development only - not for production.

```bash
omnimemory memory add --app-id my-app --user-id user-123 messages.json
omnimemory memory query --app-id my-app --query "Python async"
```

## Usage Examples

### Create Memory

**Standard Memory** (full dual-agent synthesis with conflict resolution):
```python
user_messages = UserMessages(
    app_id="my-app",
    user_id="user-123",
    session_id="session-456",
    messages=[
        Message(role="user", content="I need help with Python", timestamp="2024-01-15T10:00:00Z"),
        Message(role="assistant", content="I can help!", timestamp="2024-01-15T10:00:05Z")
        # ... must provide exactly OMNIMEMORY_DEFAULT_MAX_MESSAGES (default: 30) messages
    ]
)
result = await sdk.add_memory(user_messages)
```

**Important:** `add_memory()` requires exactly `OMNIMEMORY_DEFAULT_MAX_MESSAGES` messages (default: 30). This ensures sufficient conversation flow for meaningful memory synthesis.

**Agent Memory** (fast path, no conflict resolution):
```python
agent_request = AgentMemoryRequest(
    app_id="my-app",
    user_id="user-123",
    messages="Agent completed task: analyzed user preferences"  # String or list format
)
result = await sdk.add_agent_memory(agent_request)
```

### Query Memories

```python
results = await sdk.query_memory(
    app_id="my-app",
    query="Python async await",
    user_id="user-123",  # Optional: filter to user
    session_id="session-456",  # Optional: filter to session
    n_results=5,
    similarity_threshold=0.35  # Optional: override default recall threshold
)

for memory in results:
    print(f"Score: {memory['composite_score']}")
    print(f"Memory: {memory['memory_note']}")
```

**Query Parameters:**
- `similarity_threshold` (optional): Overrides `OMNIMEMORY_RECALL_THRESHOLD` from environment or default. Use this to adjust recall per-query without changing global settings.

### Conversation Summary

```python
summary_request = ConversationSummaryRequest(
    app_id="my-app",
    user_id="user-123",
    messages="User: Hello\nAssistant: Hi there!"  # String or list format
)
summary = await sdk.summarize_conversation(summary_request)
print(summary['summary'])
```

### Memory Evolution

```python
# Traverse evolution chain
chain = await sdk.traverse_memory_evolution_chain(
    app_id="my-app",
    memory_id="memory-uuid"
)

# Generate visualization
graph = sdk.generate_evolution_graph(chain, format="mermaid")
report = sdk.generate_evolution_report(chain, format="markdown")
```

**Message Format Notes:**
- `add_memory()` requires structured `Message` objects (role, content, timestamp)
- `add_agent_memory()` and `summarize_conversation()` accept flexible format (string or list)

## Configuration

### Essential Environment Variables

**LLM Configuration:**
- `LLM_API_KEY` (required): Your LLM API key
- `LLM_PROVIDER` (required): Provider (openai, anthropic, etc.)
- `LLM_MODEL` (required): Model name (gpt-4, claude-3-opus, etc.)
- `LLM_TEMPERATURE` (default: 0.4): Temperature for LLM calls
- `LLM_MAX_TOKENS` (default: 3000): Maximum tokens per response

**Embedding Configuration:**
- `EMBEDDING_API_KEY` (required): Your embedding API key
- `EMBEDDING_PROVIDER` (required): Provider (openai, cohere, etc.)
- `EMBEDDING_MODEL` (required): Model name
- `EMBEDDING_DIMENSIONS` (required): Dimensions (1536, 3072, etc.)

**Qdrant Connection:**
- `OMNI_MEMORY_PROVIDER` (default: qdrant-remote): Provider type
- `QDRANT_HOST` (default: localhost): Qdrant host
- `QDRANT_PORT` (default: 6333): Qdrant port

**OmniMemory Hyperparameters:**
- `OMNIMEMORY_DEFAULT_MAX_MESSAGES` (default: 30): Fixed limit - exactly this many messages required per memory creation
- `OMNIMEMORY_RECALL_THRESHOLD` (default: 0.3): Minimum similarity for recall (0.0-1.0), can be overridden per-query
- `OMNIMEMORY_COMPOSITE_SCORE_THRESHOLD` (default: 0.4): Minimum composite score for precision (0.0-1.0)
- `OMNIMEMORY_DEFAULT_N_RESULTS` (default: 10): Default number of query results
- `OMNIMEMORY_LINK_THRESHOLD` (default: 0.7): Minimum score for memory linking (0.0-1.0)
- `OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS` (default: 10): Connection pool size

### Hyperparameter Tuning

**`OMNIMEMORY_DEFAULT_MAX_MESSAGES`** - Fixed message limit per memory creation
- **Default: 30** - Ensures sufficient conversation flow for meaningful memory synthesis
- **Important:** This is a fixed limit, not a range. You must provide exactly this many messages
- **Why 30?** Provides enough context for dual-agent synthesis while keeping memory creation efficient
- **Warning:** Don't set too high (e.g., >50). Higher values consume more LLM context length, slow down processing, and may exceed token limits. Keep it balanced for your use case
- **Adjust if:** Need more context per memory (increase carefully) or want faster processing (decrease)

**`OMNIMEMORY_RECALL_THRESHOLD`** - Controls initial candidate retrieval
- Lower (0.1-0.2): More candidates, broader recall, may include less relevant results
- Higher (0.4-0.5): Fewer candidates, tighter recall, only highly similar memories
- **Adjust if:** Getting too many irrelevant results (increase) or missing relevant memories (decrease)
- **Note:** Can be overridden per-query using `similarity_threshold` parameter

**`OMNIMEMORY_COMPOSITE_SCORE_THRESHOLD`** - Controls final result filtering
- Lower (0.2-0.3): More results pass filter, includes lower-quality matches
- Higher (0.5-0.6): Stricter filtering, only high-quality results
- **Adjust if:** Final results contain low-quality matches (increase) or high-quality memories filtered out (decrease)

**`OMNIMEMORY_LINK_THRESHOLD`** - Controls memory linking and consolidation
- Lower (0.5-0.6): More memories linked, more consolidation
- Higher (0.8-0.9): Fewer links, more independent memories
- **Adjust if:** Memories incorrectly consolidated (increase) or related memories not linked (decrease)

**`OMNIMEMORY_DEFAULT_N_RESULTS`** - Controls number of results returned
- Lower (5-8): Faster queries, fewer results
- Higher (15-20): More results, slower queries
- **Adjust if:** Need more context (increase) or queries too slow (decrease)

For complete configuration details, see [ARCHITECTURE.md](docs/ARCHITECTURE.md#configuration).

## Architecture Overview

OmniMemory implements the **Self-Evolving Composite Memory Synthesis Architecture (SECMSA)**:

**Core Principles:**
- **Dual-Agent Construction**: Episodic and Summarizer agents work in parallel (like different brain regions)
- **Composite Scoring**: `relevance × (1 + recency_boost + importance_boost)` - relevance first, just like human recall
- **Self-Evolution**: Memories automatically merge, resolve conflicts, and adapt (like memory consolidation)
- **Memory Linking**: Related memories form connections (like associative memory)

**Memory Creation Paths:**
1. **Standard Memory**: Full dual-agent synthesis with conflict resolution (rich conversational context)
2. **Agent Memory**: Fast single-agent path (quick agent message storage)
3. **Conversation Summary**: Standalone summarization with optional webhooks

**Performance:**
- O(1) user-perceived latency (async background processing)
- O(log n) vector search + O(k) composite scoring for queries
- Connection pooling and embedding caching for efficiency

For comprehensive architecture documentation, mathematical foundations, and proofs, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Monitoring

### Health Check

```bash
curl http://localhost:8001/health
```

### Connection Pool Stats

```bash
curl http://localhost:8001/api/v1/system/pool-stats
```

### Prometheus Metrics

Enable with `OMNIMEMORY_ENABLE_METRICS_SERVER=true`, then access:
```
http://localhost:9001/metrics
```

Key metrics:
- `omnimemory_operations_total`: Operation counts
- `omnimemory_operations_duration_seconds`: Operation duration
- `omnimemory_operations_errors_total`: Error counts
- `omnimemory_connection_pool_size`: Pool size
- `omnimemory_connection_pool_active`: Active connections

## Troubleshooting

**Connection Pool Exhaustion**
- Increase `OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS`
- Check for connection leaks

**Slow Queries**
- Check Qdrant performance
- Adjust similarity thresholds
- Reduce `OMNIMEMORY_DEFAULT_N_RESULTS`

**LLM API Errors**
- Verify API keys
- Check rate limits
- Adjust `LLM_MAX_TOKENS` if responses too long

**SDK Warm-up**
- Always call `await sdk.warm_up()` after initialization
- Without warm-up, first request takes 2-5 seconds

For detailed troubleshooting, see [ARCHITECTURE.md](docs/ARCHITECTURE.md#critical-review-and-design-decisions).

## Development

```bash
# Run tests
pytest

# Format code
ruff format .

# Lint code
ruff check .

# Run API server locally
uvicorn omnimemory.api.server:app --host 0.0.0.0 --port 8001 --reload
```

## Support

- **Architecture Documentation**: [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **API Documentation**: http://localhost:8001/docs (when server is running)
- **Issues**: [GitHub Issues](https://github.com/omnirexflora-labs/omnimemory/issues)

## License

MIT License - see [LICENSE](LICENSE) for details.
