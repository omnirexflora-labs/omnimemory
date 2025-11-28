# OmniMemory Examples

This directory contains comprehensive real-world examples demonstrating OmniMemory usage.

## Examples

### 1. Complete SDK Example (`complete_sdk_example.py`)

**Scenario**: Customer Support Agent with Memory

Demonstrates **ALL 14 SDK methods** in a realistic customer support workflow:
- ✅ Initialize SDK and warm up connections
- ✅ Add structured conversations
- ✅ Quick agent memory saves
- ✅ Semantic memory queries with composite scoring
- ✅ Conversation summarization (sync mode)
- ✅ Memory batching for streaming chats
- ✅ Task status monitoring
- ✅ Memory evolution tracking
- ✅ GDPR-compliant memory deletion
- ✅ Connection pool monitoring
- ✅ Agent contextual answers (AgentMemorySDK)

**Run**:
```bash
# Ensure API server is running
uv run python examples/complete_sdk_example.py
```

### 2. Complete API Example (`complete_api_example.py`)

**Scenario**: Customer Support via REST API (Language-Agnostic)

Demonstrates **ALL REST endpoints** using HTTP calls:
- ✅ Health checks
- ✅ Memory creation (POST /api/v1/memories)
- ✅ Memory queries (GET /api/v1/memories/query)
- ✅ Agent memory (POST /api/v1/agent/memories)
- ✅ Memory batching (POST /api/v1/memory-batcher/messages)
- ✅ Conversation summarization (POST /api/v1/agent/summaries)
- ✅ Evolution chains (GET /api/v1/memories/{id}/evolution)
- ✅ Memory deletion (DELETE /api/v1/memories/{id})
- ✅ System monitoring (GET /api/v1/system/pool-stats)

**Run**:
```bash
# Ensure API server is running on port 8001
uv run python examples/complete_api_example.py
```

**Port to Other Languages**:
This example uses Python httpx but the HTTP patterns work in:
- **Node.js**: Use `axios` or `fetch`
- **Go**: Use `net/http`
- **Rust**: Use `reqwest`
- **Java**: Use `HttpClient`
- **PHP**: Use `Guzzle`

## Prerequisites

1. **Environment Setup**:
```bash
# Copy environment template
cp examples/env/qdrant.env.example .env

# Edit with your API keys
LLM_API_KEY=your-key-here
EMBEDDING_API_KEY=your-key-here
```

2. **Start Backend**:
```bash
# Start Qdrant
docker compose -f docker-compose.local.yml --profile qdrant up -d

# Start API Server (for REST API example)
uv run uvicorn omnimemory.api.server:app --host 0.0.0.0 --port 8001 --reload
```

## Real-World Use Cases Covered

Both examples demonstrate practical scenarios:

1. **Customer Support Agent**
   - Remembering customer preferences
   - Tracking conversation history
   - Handling upgrades and billing issues
   - GDPR compliance (deletion)

2. **Context Window Management**
   - Summarizing long conversations
   - Streaming chat integration
   - Memory batching

3. **Memory Evolution**
   - Tracking how memories change over time
   - Visualizing evolution chains
   - Generating evolution reports

4. **Production Monitoring**
   - Connection pool statistics
   - Task status tracking
   - System health checks

## Output

Both examples provide detailed console output showing:
- ✅ Success confirmations
- 📊 Data statistics
- 🔍 Query results with scores
- ⚠️ Error handling
- 📈 Progress tracking

Run them to see OmniMemory in action!
