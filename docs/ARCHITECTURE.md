# Self-Evolving Composite Memory Synthesis Architecture (SECMSA)

## Table of Contents

1. [Abstract](#abstract)
2. [System Specifications](#system-specifications)
3. [Multi-Tenant Architecture and Data Isolation](#multi-tenant-architecture-and-data-isolation)
4. [Architecture Overview](#architecture-diagram-overview)
5. [Core Components](#core-components)
   - [Memory Creation Paths](#1-memory-creation-paths)
   - [Dual-Agent Construction](#2-dual-agent-memory-construction)
   - [Conflict Resolution](#3-conflict-resolution)
   - [Composite Scoring](#4-composite-scoring)
   - [Query Execution](#5-query-execution)
   - [Asynchronous Architecture](#6-asynchronous-architecture)
   - [Embedding Generation](#7-embedding-generation)
   - [Metrics & Observability](#8-metrics--observability)
   - [Vector Database Integration](#9-vector-database-integration)
6. [Mathematical Foundations](#mathematical-foundations)
   - [Composite Scoring Function](#composite-scoring-function)
   - [Evolution Operations](#evolution-operations)
   - [Query Execution](#query-execution)
   - [Fuzzy Deduplication](#fuzzy-deduplication)
   - [Memory Link Generation and Tree Formation](#memory-link-generation-and-tree-formation)
   - [Memory Note Synthesis](#memory-note-synthesis)
7. [API & SDK Interface](#api--sdk-interface)
8. [Implementation Details](#implementation-details)
9. [Scalability & Performance](#scalability--performance)
10. [Production Features](#production-features)
11. [Configuration](#configuration)
12. [Future Enhancements](#future-enhancements)
13. [Critical Review and Design Decisions](#critical-review-and-design-decisions)

---

## Abstract

This document presents the **Self-Evolving Composite Memory Synthesis Architecture (SECMSA)**, a production-ready memory framework that enables autonomous AI agents to construct, evolve, and synthesize memory through intelligent multi-agent coordination. Unlike traditional memory systems that treat storage as static repositories, SECMSA implements memory as a continuously self-organizing cognitive substrate where semantic units autonomously merge, resolve conflicts, and adapt their relational structure based on emerging patterns and contradictions.

The architecture achieves **memory synthesis** through parallel dual-agent construction, where specialized Episodic and Summarizer agents independently analyze conversational context and generate complementary memory representations. These outputs undergo normalization, fuzzy deduplication, and intelligent merging to produce canonical memory notes embedded as single concatenated vectors for semantic retrieval.

SECMSA implements **composite multi-dimensional scoring** using a multiplicative relevance-first approach: `composite_score = relevance × (1 + recency_boost + importance_boost)`, ensuring that semantic relevance remains the primary ranking criterion while recency and importance provide adaptive modulation (max 10% boost each). This composite scoring mechanism prevents temporally recent but semantically irrelevant memories from dominating retrieval results.

The system achieves **self-evolution** through AI-powered conflict resolution agents that autonomously execute four primitive operations: UPDATE (consolidate redundant memories), DELETE (eliminate superseded information), SKIP (preserve independent memories), and CREATE (generate net-new memories). When conflicts are detected, a SynthesisAgent deterministically generates updated successor memories. The system uses a **simple status-based approach** where memories are marked with `status` ("active", "updated", "deleted") and `status_reason` ("consolidated", "contradicted", "manual_update") to track evolution without complex versioning metadata.

SECMSA provides formal guarantees: (1) **Semantic Coherence** - conflict resolution maintains logical consistency across merged memories, (2) **Retrieval Optimality** - composite scoring provably prioritizes relevance while incorporating temporal and importance signals, and (3) **Computational Tractability** - fully asynchronous processing via `asyncio` ensures O(1) memory creation latency from the user perspective.

The architecture supports multiple memory creation paths: (1) **Standard Memory Creation** - full dual-agent synthesis with conflict resolution, (2) **Agent Memory Creation** - simplified fast path for direct agent message storage, and (3) **Conversation Summary** - standalone summarization with optional webhook callbacks.

**Key Contributions:**
1. A formally specified dual-agent memory synthesis framework with parallel construction
2. Self-evolving conflict resolution using AI agents with deterministic update operations
3. Composite multi-dimensional scoring with provable relevance-first guarantees
4. Simple status-based memory evolution tracking (status + reason)
5. Multiple memory creation paths for different use cases
6. Fully asynchronous architecture with connection pooling for scalability
7. Production-ready implementation with comprehensive metrics and observability
8. Multi-tenant architecture with three-tier isolation (app_id, user_id, session_id) for multi-agent, multi-user, multi-app deployments

SECMSA represents a paradigm shift from passive memory storage to active cognitive synthesis, where memories autonomously organize, evolve, and adapt to maintain semantic coherence. The architecture is production-ready, with implementations deployed in autonomous agent systems requiring human-like memory capabilities including context retention, contradiction resolution, and temporal reasoning.

---

## Keywords

Self-evolving memory, composite scoring, memory synthesis, dual-agent construction, conflict resolution, semantic memory, autonomous agents, cognitive architecture, status-based evolution, asynchronous processing, vector database, Qdrant, embeddings, LLM integration

---

## System Specifications

**Memory Construction:** Parallel dual-agent (Episodic + Summarizer) with normalization and fuzzy deduplication  
**Scoring Function:** `composite = relevance × (1 + recency_boost + importance_boost)` where boosts are bounded by score ranges (max 10% each)  
**Evolution Operations:** UPDATE | DELETE | SKIP | CREATE  
**Status Tracking:** Simple status-based (`active`, `updated`, `deleted`) with `status_reason` (`consolidated`, `contradicted`, `manual_update`)  
**Consistency Model:** Status-based updates with conflict resolution  
**Processing Model:** Fully asynchronous (`asyncio`) with O(1) user-perceived latency  
**Storage Backend:** Vector database (Qdrant) with connection pooling  
**Metrics:** In-memory Prometheus metrics with HTTP endpoint  
**Embedding Strategy:** Token-based chunking with automatic fallback for large texts  
**Connection Pooling:** Configurable pool size (default: 10) with async context managers  
**Multi-Tenant Architecture:** Three-tier isolation (app_id, user_id, session_id) for multi-agent, multi-user, multi-app deployments

---

## Multi-Tenant Architecture and Data Isolation

SECMSA is designed for **multi-agent, multi-user, multi-app** scenarios from day zero. The system implements a three-tier isolation model to ensure complete data separation and prevent any data leakage between different applications, users, or sessions.

### Isolation Model

**`app_id` (Required):**
- **Purpose:** Application-level isolation - each application has its own memory collection
- **Storage:** Used as the collection name in the vector database (Qdrant)
- **Isolation:** Memories from different apps are stored in separate collections - complete physical separation
- **Use case:** Deploy one OmniMemory instance for multiple applications (e.g., "customer-support-app", "sales-assistant-app")

**`user_id` (Required):**
- **Purpose:** User-level isolation within an application
- **Storage:** Stored as metadata field, used for query-time filtering
- **Isolation:** Memories are tagged with user_id - queries can filter by user_id or search across all users
- **Use case:** Multiple users in the same application (e.g., user-123, user-456 in "customer-support-app")

**`session_id` (Optional):**
- **Purpose:** Session-level isolation for conversation grouping
- **Storage:** Stored as metadata field (or "none" if not provided), used for query-time filtering
- **Isolation:** Optional grouping - if provided, enables session-scoped queries
- **Use case:** Group related conversations (e.g., "support-ticket-789", "onboarding-session-456")

### Isolation Guarantees

**Storage Level:**
- `app_id` determines the vector database collection - different apps = different collections
- All memories include `app_id`, `user_id`, and `session_id` (or "none") in metadata
- No cross-collection queries possible - app_id is always required

**Query Level:**
- Queries always require `app_id` - determines which collection to search
- Optional `user_id` filter - restricts results to specific user's memories
- Optional `session_id` filter - restricts results to specific session's memories
- Filtering is enforced at the vector database level - no data leakage possible

**Mathematical Formulation:**
```
Memory Storage:
  M(app_id, user_id, session_id, content) → stored_in_collection(app_id)

Query Execution:
  Q(app_id, query, user_id?, session_id?) → results where:
    collection = app_id
    AND (user_id = user_id OR user_id is NULL)
    AND (session_id = session_id OR session_id is NULL)
```

**Data Leakage Prevention:**
- Collection-level isolation: Different `app_id` values use different collections
- Metadata filtering: `user_id` and `session_id` filters are applied at query time
- No cross-app access: Queries cannot access memories from different app_id
- No cross-user access: When `user_id` is specified, only that user's memories are returned
- No cross-session access: When `session_id` is specified, only that session's memories are returned

---

## Architecture Diagram Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECMSA Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Conversational Context / Agent Messages                │
│           ↓                                                      │
│  ┌──────────────────────────────────────────┐                 │
│  │   Memory Creation Paths                  │                 │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────┐│                 │
│  │  │  Standard   │  │   Agent     │  │Summary│                 │
│  │  │  Memory     │  │   Memory   │  │       │                 │
│  │  │  (Full)     │  │  (Fast)    │  │       │                 │
│  │  └──────┬──────┘  └──────┬──────┘  └──┬───┘│                 │
│  │         │                │            │                     │
│  │         └────────┬───────┴────────────┘                     │
│  └───────────┬───────┼─────────────────────────────────────────┘
│              │       │                                           │
│              ↓       ↓                                           │
│  ┌──────────────────────────────────────────┐                 │
│  │   Parallel Dual Construction             │                 │
│  │  ┌─────────────┐  ┌─────────────┐       │                 │
│  │  │  Episodic   │  │ Summarizer  │       │                 │
│  │  │   Agent     │  │   Agent     │       │                 │
│  │  │             │  │             │       │                 │
│  │  │ • Behavioral│  │ • Narrative │       │                 │
│  │  │ • Patterns  │  │ • Content   │       │                 │
│  │  │ • Insights   │  │ • Tags      │       │                 │
│  │  └──────┬──────┘  └──────┬──────┘       │                 │
│  └─────────┼─────────────────┼───────────────┘                 │
│            │                 │                                   │
│            └────────┬────────┘                                   │
│                     ↓                                            │
│  ┌──────────────────────────────────────────┐                   │
│  │  Synthesis & Normalization              │                   │
│  │  • Merge outputs                        │                   │
│  │  • Fuzzy deduplication                  │                   │
│  │  • Generate Zettelkasten note            │                   │
│  │  • Prepare metadata                     │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │   Embedding Generation                   │                   │
│  │  • Token-based chunking                  │                   │
│  │  • Embedding API call                     │                   │
│  │  • In-memory caching                     │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │   Semantic Link Discovery               │                   │
│  │  • Vector similarity search             │                   │
│  │  • Composite score calculation          │                   │
│  │  • Link threshold filtering             │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │   Conflict Resolution Agent              │                   │
│  │   Analyzes: Redundancy,                  │                   │
│  │   Contradictions, Superseding            │                   │
│  │   Outputs: UPDATE|DELETE|SKIP|CREATE    │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │   Synthesis Agent (if UPDATE)            │                   │
│  │   • Consolidate memories                 │                   │
│  │   • Merge content intelligently          │                   │
│  │   • Preserve all unique information     │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │   Status-Based Update                    │                   │
│  │   • Update status (active/updated/deleted)│                  │
│  │   • Set status_reason                    │                   │
│  │   • Update timestamp                     │                   │
│  │   • Set next_id (for evolution chain)    │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │    Vector Store (Qdrant)                │                   │
│  │   • Embeddings (retrieval)               │                   │
│  │   • Metadata (status, reason, next_id)  │                   │
│  │   • Connection Pooling                  │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │   Query & Retrieval                      │                   │
│  │  • Semantic search                       │                   │
│  │  • Composite scoring                     │                   │
│  │  • Multi-dimensional ranking             │                   │
│  └─────────────┬──────────────────────────┘                   │
│                ↓                                                 │
│  ┌──────────────────────────────────────────┐                   │
│  │   Metrics & Observability                │                   │
│  │   • Prometheus metrics                   │                   │
│  │   • Operation timing                     │                   │
│  │   • Error tracking                       │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                 │
│  OUTPUT: Self-Organizing Memory Store                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Memory Creation Paths

#### 1.1 Standard Memory Creation (`create_and_store_memory`)

**Flow:**
1. Format conversation messages
2. Parallel dual-agent construction (Episodic + Summarizer)
3. Parse and normalize JSON outputs
4. Create Zettelkasten-style memory note
5. Generate embedding with token-based chunking
6. Find semantic links to existing memories
7. Conflict resolution (if links found)
8. Store memory with comprehensive metadata

**Agents:** Episodic Agent + Summarizer Agent (parallel execution via `asyncio.gather`)

**Features:**
- Full metadata extraction (tags, keywords, semantic queries)
- Behavioral pattern analysis
- Conflict detection and resolution
- Semantic linking to related memories
- Status-based evolution tracking

**Use Case:** Rich conversational context requiring comprehensive memory synthesis

**Mathematical Complexity:**
- Time: O(1) user-perceived (async background processing)
- Space: O(n) where n = message length
- LLM Calls: 2 (parallel) + 1 (conflict resolution, if needed) + 1 (synthesis, if UPDATE)

#### 1.2 Agent Memory Creation (`create_agent_memory`)

**Flow:**
1. Format agent messages (string or list)
2. Fast summary generation (single agent)
3. Generate embedding
4. Direct storage (no conflict resolution)

**Agent:** Single fast summary agent

**Features:**
- Minimal processing overhead
- No conflict resolution
- Immediate storage
- Fast path for agent message storage

**Use Case:** Direct agent message storage without full pipeline overhead

**Mathematical Complexity:**
- Time: O(1) user-perceived (async background processing)
- Space: O(n) where n = message length
- LLM Calls: 1 (fast summary)

#### 1.3 Conversation Summary (`generate_conversation_summary`)

**Flow:**
1. Summary generation (fast or full path)
2. Optional callback/webhook delivery

**Modes:**
- **Fast Path:** Simple text-only summary (< 10 seconds)
- **Full Path:** Structured summary with metadata (slower)

**Delivery:**
- **Sync:** Return summary directly (no callback)
- **Async:** Return 202 Accepted, deliver via webhook with retry logic (max 3 attempts, exponential backoff)

**Use Case:** Standalone conversation summarization for external systems

**Mathematical Complexity:**
- Time: O(1) user-perceived (async if callback provided)
- Space: O(n) where n = message length
- LLM Calls: 1 (fast) or 1 (full structured)

### 2. Dual-Agent Memory Construction

#### 2.1 Episodic Agent

**Purpose:** Extract behavioral patterns and interaction dynamics from conversations

**Focus Areas:**
- Communication style (formality, detail preference, pace)
- Learning preferences (examples vs theory, hands-on vs conceptual)
- Problem-solving approach (systematic vs intuitive)
- Decision-making patterns (quick vs deliberate)
- Engagement dynamics (what energizes vs frustrates)
- Success/failure patterns (what worked vs what failed)

**Output Structure:**
```json
{
  "context": {
    "available_data": "1 sentence: what was observable",
    "user_intent": "1 sentence: what they were trying to accomplish",
    "analysis_limitation": "1 sentence: key constraint, or N/A"
  },
  "what_worked": {
    "strategies": ["max 3, 1 sentence each"],
    "pattern": "1 sentence: why these worked"
  },
  "what_failed": {
    "strategies": ["max 2, 1 sentence each"],
    "pattern": "1 sentence: why these failed"
  },
  "behavioral_profile": {
    "communication": "2 sentences: style, formality, detail level, pace",
    "learning": "1-2 sentences: preferred learning modes",
    "problem_solving": "1-2 sentences: approach style",
    "decision_making": "1-2 sentences: how they make choices"
  },
  "interaction_insights": {
    "engagement_triggers": "1 sentence: what energizes",
    "friction_points": "1 sentence: what creates confusion",
    "optimal_approach": "1 sentence: conditions for best interaction"
  },
  "future_guidance": {
    "recommended_approaches": ["max 3, 1 sentence each"],
    "avoid_approaches": ["max 2, 1 sentence each"],
    "adaptation_note": "1-2 sentences: key insight"
  }
}
```

#### 2.2 Summarizer Agent

**Purpose:** Create narrative summaries that capture conversation content, knowledge, and outcomes

**Focus Areas:**
- Narrative flow (situation → evolution → insights → outcomes)
- Content preservation (technical details, solutions, concepts)
- Retrieval optimization (tags, keywords, semantic queries)
- Metadata extraction (depth, follow-ups)

**Output Structure:**
```json
{
  "context": {
    "available_data": "1 sentence: what messages were available",
    "content_scope": "1-2 sentences: topics and knowledge covered"
  },
  "narrative": "150-200 words: Complete flowing story",
  "retrieval": {
    "tags": ["8 max: topic tags, domain tags, outcome tags"],
    "keywords": ["10 max: key terms, concepts, technologies"],
    "queries": ["4 max: natural search queries this note should match"]
  },
  "metadata": {
    "depth": "high/medium/low",
    "follow_ups": ["max 2, 1 sentence each, or N/A"]
  }
}
```

#### 2.3 Synthesis Process

**Steps:**
1. Both agents process input in parallel (`asyncio.gather`)
2. Parse JSON outputs with error handling
3. Normalize and validate data structures
4. Create Zettelkasten-style memory note:
   - Summary section (from narrative)
   - Behavioral Patterns section (from episodic)
   - Experience Learnings section (what worked/failed)
   - Guidance section (future recommendations)
   - Footer with tags, keywords, metadata
5. Prepare metadata for storage
6. Single embedding created from concatenated content

**Zettelkasten Note Structure:**
```
## Summary
[Narrative from summarizer]

## Behavioral Patterns
[Communication style, learning preferences, problem-solving approach]

## Experience Learnings
[What worked, what failed, patterns]

## Guidance
[Recommended approaches, avoid approaches, key insights]

---
Tags: [comma-separated tags] | Keywords: [comma-separated keywords] | [Metadata]
```

### 3. Conflict Resolution

#### 3.1 Conflict Detection

**Process:**
1. Generate embedding for new memory
2. Semantic similarity search in vector database
3. Filter by `status="active"` and optional `user_id`/`session_id`
4. Calculate composite scores for candidates
5. Filter by link threshold (default: 0.7)
6. Return top linked memories (max: 4 for synthesis)

**Link Threshold:** `LINK_THRESHOLD` (default: 0.7) - minimum composite score for linking

**Candidate Multiplier:** `_CANDIDATE_MULTIPLIER = 3` - retrieve 3× requested links for filtering

#### 3.2 Conflict Resolution Agent

**Purpose:** Analyze relationships between new memory and linked memories

**Decision Types:**
- **UPDATE:** Consolidate multiple memories into one
  - Status: `updated`
  - Reason: `consolidated`
  - New memory contains merged content
- **DELETE:** Mark memory as superseded
  - Status: `deleted`
  - Reason: `contradicted` or `consolidated`
- **SKIP:** No action needed
  - Status: `active` (unchanged)
  - Timestamp updated for recency boost

**Agent Output:**
```json
[
  {
    "memory_id": "uuid",
    "operation": "UPDATE|DELETE|SKIP",
    "confidence_score": 0.0-1.0,
    "reasoning": "Brief explanation"
  }
]
```

**Decision Criteria:**
- **UPDATE:** Multiple memories cover similar topics, new memory adds context
- **DELETE:** New memory contradicts or supersedes existing memories
- **SKIP:** New memory is redundant, adds minimal value

#### 3.3 Synthesis Agent

**Purpose:** Consolidate multiple related memories into a single comprehensive memory

**Consolidation Strategy:**
1. **Summary Section:** Merge narratives chronologically/thematically, add new information, remove only word-for-word duplicates
2. **Behavioral Patterns:** Combine all unique observations, preserve evolution ("Initially X, now also Y")
3. **Experience Learnings:** Combine all "what worked" and "what failed" examples, deduplicate identical strategies
4. **Guidance:** Merge all recommendations, combine follow-up areas
5. **Tags/Keywords:** Combine all tags and keywords, deduplicate, sort alphabetically

**Quality Checks:**
- All 8 sections present
- Output length ≥ longest input
- "Related Topics" appears once (at end)
- No vague generalizations replacing specific details
- All unique information preserved

**Agent Output:**
```json
{
  "consolidated_memory": {
    "natural_memory_note": "Complete memory note with all sections"
  },
  "synthesis_metadata": {
    "memories_merged": "integer",
    "new_information_added": "high/medium/low",
    "deduplication_count": "approximate number",
    "quality_check": "pass/warning",
    "notes": "1 sentence: key changes"
  }
}
```

#### 3.4 Status-Based Tracking

**Status Values:**
- `"active"`: Current, active memory
- `"updated"`: Consolidated into newer memory
- `"deleted"`: Superseded or contradicted
- `"archived"`: (Future) Archived to external storage due to age decay (see Future Enhancements)

**Status Reasons:**
- `"consolidated"`: Merged with other memories (UPDATE operation)
- `"contradicted"`: Replaced by contradictory information (DELETE operation)
- `"manual_update"`: Manually updated by user/admin
- `"created"`: Newly created memory (initial status)
- `"age_decay"`: (Future) Archived due to age threshold (see Future Enhancements)

**Evolution Chain:**
- `next_id`: Points to the memory that superseded this one
- Enables forward traversal of evolution chain
- Cycle detection prevents infinite loops

**Metadata Fields:**
```json
{
  "status": "active|updated|deleted|archived",
  "status_reason": "consolidated|contradicted|manual_update|created|age_decay",
  "updated_at": "ISO timestamp",
  "next_id": "uuid or null",
  "archived_at": "ISO timestamp (if archived)",
  "archival_location": "storage path/URL (if archived)"
}
```

Note: `archived` status and `age_decay` reason are future enhancements (see Future Enhancements section).

### 4. Composite Scoring

#### 4.1 Formula

```
S_composite(m, q, t) = S_relevance(m, q) × (1 + β_recency(t) + β_importance(m))

where:
  S_relevance(m, q) = cosine_similarity(embed(m), embed(q)) ∈ [0, 1]
  β_recency(t) = recency_score × RECENCY_BOOST_FACTOR
  β_importance(m) = importance_score × IMPORTANCE_BOOST_FACTOR
  RECENCY_BOOST_FACTOR = 0.1
  IMPORTANCE_BOOST_FACTOR = 0.1
  
Note: Since recency_score ∈ [0, 1] and importance_score ∈ [0, 1],
      the maximum boost from each is 0.1 (10%), achieved when score = 1.0.
      The formula ensures: 0 ≤ β_recency ≤ 0.1 and 0 ≤ β_importance ≤ 0.1.
```

#### 4.2 Recency Score

**Formula:**
```
recency_score = exp(-age_hours / half_life_hours)

where:
  age_hours = (current_time - max(created_at, updated_at)) / 3600
  half_life_hours = MAX_AGE_HOURS / HALF_LIFE_FACTOR
  MAX_AGE_HOURS = 43800 (5 years)
  HALF_LIFE_FACTOR = 4
  half_life_hours = 10950 hours (≈ 1.25 years)
```

**Properties:**
- Exponential decay over time
- Uses most recent timestamp (created_at or updated_at)
- Returns 1.0 for future timestamps
- Returns 0.01 minimum for very old memories

**Recency Boost:**
```
β_recency = recency_score × RECENCY_BOOST_FACTOR
         = recency_score × 0.1
```

Since `recency_score ∈ [0.01, 1.0]`, the boost ranges from 0.001 (0.1%) to 0.1 (10%).
Maximum 10% boost from recency (when recency_score = 1.0, i.e., very recent memory).

#### 4.3 Importance Score

**Formula:**
```
importance_score = (quality_score × w_quality + followup_score × w_followup + richness_score × w_richness) / total_weight

where:
  w_quality = 0.5
  w_followup = 0.3
  w_richness = 0.2
  total_weight = 1.0
```

**Quality Score:**
- `"high"`: 1.0
- `"medium"`: 0.6
- `"low"`: 0.2
- Default: 0.5

**Follow-up Score:**
- ≥ 3 follow-ups: 1.0
- 2 follow-ups: 0.8
- 1 follow-up: 0.6
- 0 follow-ups: 0.3

**Richness Score:**
- ≥ 10 tags+keywords: 1.0
- 5-9 tags+keywords: 0.8
- 2-4 tags+keywords: 0.6
- < 2 tags+keywords: 0.3

**Importance Boost:**
```
β_importance = importance_score × IMPORTANCE_BOOST_FACTOR
             = importance_score × 0.1
```

Since `importance_score ∈ [0, 1.0]`, the boost ranges from 0 (0%) to 0.1 (10%).
Maximum 10% boost from importance (when importance_score = 1.0, i.e., high-quality memory).

#### 4.4 Properties

**Relevance-First Guarantee:**
- Semantic similarity is the primary factor
- Low relevance (0.4) cannot be saved by recency/importance
- High relevance (0.7) gets small boost from recency/importance
- Relevance is always the base multiplier

**Bounded Boosts:**
- Recency: max +10%
- Importance: max +10%
- Combined: max +20% total boost

**Multiplicative Nature:**
- Ensures irrelevant memories never rank high
- `composite_score ≤ relevance_score × 1.2`
- Prevents temporal bias from dominating results

### 5. Query Execution

#### 5.1 Query Pipeline

**Steps:**
1. **Semantic Search:** Retrieve candidates with similarity > threshold
   - Uses vector similarity search in Qdrant
   - Filters by `status="active"`, `app_id`, optional `user_id`/`session_id`
   - Retrieves `n_results × CANDIDATE_MULTIPLIER` candidates (default: 3×)
2. **Composite Scoring:** Rank by `S_composite(m, query, t)`
   - Calculate relevance (cosine similarity)
   - Calculate recency score
   - Calculate importance score
   - Compute composite score
3. **Precision Filtering:** Apply composite score threshold
   - Filter by `COMPOSITE_SCORE_THRESHOLD` (default: 0.4)
   - Ensures quality results
4. **Top-K Selection:** Return top `n_results` by composite score
   - Sorted by composite score (descending)
   - Returns ranked list with scores and metadata

**Mathematical Complexity:**
- Time: O(log n) for vector search + O(k) for scoring where k = candidates
- Space: O(k) for candidate storage

#### 5.2 Query Result Structure

```json
{
  "rank": 1,
  "similarity_score": 0.85,
  "composite_score": 0.92,
  "recency_score": 0.95,
  "importance_score": 0.88,
  "memory_note": "Full memory note text",
  "metadata": {
    "document_id": "uuid",
    "app_id": "app_id",
    "user_id": "user_id",
    "session_id": "session_id",
    "created_at": "ISO timestamp",
    "updated_at": "ISO timestamp",
    "status": "active",
    "tags": ["tag1", "tag2"],
    "keywords": ["keyword1", "keyword2"],
    "semantic_queries": ["query1", "query2"]
  }
}
```

### 6. Asynchronous Architecture

#### 6.1 Processing Model

**Framework:** Python `asyncio` (not Celery)

**Non-Blocking Operations:**
- All I/O operations use `async`/`await`
- LLM API calls are async
- Vector database operations are async
- HTTP callbacks are async

**Background Tasks:**
- Memory creation runs as `asyncio.create_task`
- Webhook callbacks with retry logic
- Non-blocking API responses (202 Accepted for async operations)

**User-Perceived Latency:**
- O(1) - immediate response with task ID
- Actual processing happens in background
- Task status can be queried via `get_task_status`

#### 6.2 Connection Pooling

**Implementation:** `VectorDBHandlerPool` (singleton pattern)

**Features:**
- Configurable pool size (default: 10, via `OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS`)
- Async context managers for resource management
- Automatic connection lifecycle management
- Retry logic with exponential backoff
- Pool statistics via `get_connection_pool_stats()`

**Pool Management:**
```python
async with memory_manager._get_pooled_handler() as handler:
    # Use handler for operations
    results = await handler.query_collection(...)
```

**Benefits:**
- Prevents resource exhaustion
- Reuses connections efficiently
- Handles connection failures gracefully
- Tracks pool utilization

#### 6.3 Background Task Management

**Task Registration:**
- Tasks registered with unique task IDs
- Automatic cleanup on completion
- Task status tracking

**Webhook Callbacks:**
- Max 3 retry attempts
- Exponential backoff (1s, 2s, 4s)
- Permanent error detection (4xx except 429, 5xx except 503/504)
- Error payload delivery on failure

### 7. Embedding Generation

#### 7.1 Token-Based Chunking

**Strategy:**
- Uses `tiktoken` for token counting
- Chunks text by token count (not character count)
- Overlap between chunks for context preservation
- Automatic fallback to character-based chunking if tokenizer unavailable

**Parameters:**
- Default chunk size: 500 tokens
- Default overlap: 50 tokens
- Model-specific tokenizers

**Process:**
1. Count tokens in text
2. If exceeds chunk size, split into chunks with overlap
3. Embed each chunk
4. Average embeddings (or use first chunk for very long texts)

#### 7.2 Embedding Caching

**Implementation:** In-memory cache with TTL

**Cache Parameters:**
- TTL: 3600 seconds (1 hour)
- Max entries: 512
- Key: SHA256 hash of text
- Thread-safe with `RLock`

**Cache Benefits:**
- Reduces API calls for repeated texts
- Improves performance for common queries
- Automatic expiration and pruning

### 8. Metrics & Observability

#### 8.1 Prometheus Metrics

**Metric Types:**
- **Counters:** Operation totals, error counts
- **Histograms:** Operation durations, batch sizes, success rates
- **Gauges:** Active operations, health status

**In-Memory Storage:** No external dependencies (Redis removed)

**Operation Tracking:**
- Query operations: `query_memory`
- Write operations: `create_and_store_memory`, `create_agent_memory`, `store_memory_note`
- Update operations: `update_memory_status`, `update_memory_timestamp`
- Batch operations: Status updates, timestamp updates
- Summary operations: `generate_conversation_summary`

#### 8.2 Metrics Endpoint

**Configuration:**
- HTTP server on configurable port (default: 9001, via `OMNIMEMORY_METRICS_PORT`)
- Prometheus-compatible `/metrics` endpoint
- Auto-starts in background thread (non-blocking)
- Enabled via `OMNIMEMORY_ENABLE_METRICS_SERVER` (default: false)

**Metrics Exposed:**
- `omnimemory_operations_total`: Total operations by type
- `omnimemory_operations_duration_seconds`: Operation duration histogram
- `omnimemory_operations_errors_total`: Error counts by type
- `omnimemory_batch_operations_total`: Batch operation counts
- `omnimemory_connection_pool_size`: Connection pool size
- `omnimemory_connection_pool_active`: Active connections

### 9. Vector Database Integration

#### 9.1 Qdrant Backend

**Features:**
- Async Qdrant client
- Collection management (auto-create on first use)
- Metadata filtering
- Similarity search with configurable thresholds

**Collection Naming:**
- Uses `app_id` as collection name
- Automatic collection creation
- Per-app isolation

#### 9.2 Metadata Storage

**Stored Fields:**
- `document_id`: Memory ID (UUID)
- `app_id`: Application ID
- `user_id`: User ID
- `session_id`: Session ID (or "none")
- `created_at`: ISO timestamp
- `updated_at`: ISO timestamp
- `status`: "active" | "updated" | "deleted"
- `status_reason`: "consolidated" | "contradicted" | "manual_update" | "created"
- `next_id`: UUID of next memory in evolution chain (or null)
- `tags`: List of tags
- `keywords`: List of keywords
- `semantic_queries`: List of semantic queries
- `conversation_complexity`: Integer (1-5)
- `interaction_quality`: "high" | "medium" | "low"
- `follow_up_potential`: List of follow-up topics
- `embedding_dimensions`: Integer

**Filtering:**
- Query-time filtering by metadata fields
- Always filters by `status="active"` for queries
- Optional filtering by `user_id`, `session_id`

---

## Mathematical Foundations

### Composite Scoring Function

```
S_composite(m, q, t) = S_relevance(m, q) × (1 + β_recency(t) + β_importance(m))

where:
  S_relevance(m, q) = cosine_similarity(embed(m), embed(q)) ∈ [0, 1]
  β_recency(t) = recency_score × RECENCY_BOOST_FACTOR
  β_importance(m) = importance_score × IMPORTANCE_BOOST_FACTOR
  RECENCY_BOOST_FACTOR = 0.1
  IMPORTANCE_BOOST_FACTOR = 0.1
  
  recency_score = exp(-age_hours / half_life_hours)
  age_hours = (current_time - max(created_at, updated_at)) / 3600
  half_life_hours = MAX_AGE_HOURS / HALF_LIFE_FACTOR = 43800 / 4 = 10950 hours (≈ 1.25 years)
  MAX_AGE_HOURS = 43800 (5 years)
  HALF_LIFE_FACTOR = 4
  
  importance_score = (quality_score × 0.5 + followup_score × 0.3 + richness_score × 0.2)
  
  quality_score ∈ {1.0, 0.6, 0.2, 0.5} (high, medium, low, default)
  followup_score ∈ {1.0, 0.8, 0.6, 0.3} (≥3, 2, 1, 0 follow-ups)
  richness_score ∈ {1.0, 0.8, 0.6, 0.3} (≥10, 5-9, 2-4, <2 tags+keywords)
  
Note: Since recency_score ∈ [0.01, 1.0] and importance_score ∈ [0, 1.0],
      the maximum boost from each is 0.1 (10%), achieved when score = 1.0.
      The bounds are naturally enforced by the score ranges, not by explicit min().
```

### Evolution Operations

```
Ω_conflict(M) → {UPDATE, DELETE, SKIP, CREATE}

UPDATE(m₁, m₂, ..., mₙ) → m' where:
  content(m') = SynthesisAgent(content(m₁), ..., content(mₙ))
  status(m') = "active"
  status_reason(m') = "consolidated"
  created_at(m') = current_timestamp
  updated_at(m') = current_timestamp
  status(mᵢ) = "updated" for i ∈ {1, ..., n}
  status_reason(mᵢ) = "consolidated" for i ∈ {1, ..., n}
  updated_at(mᵢ) = current_timestamp for i ∈ {1, ..., n}
  next_id(mᵢ) = id(m') for i ∈ {1, ..., n}

DELETE(m) → ∅ where:
  status(m) = "deleted"
  status_reason(m) = "contradicted" | "consolidated"
  updated_at(m) = current_timestamp

SKIP(m) → m where:
  updated_at(m) = current_timestamp
  (no status change)

CREATE(context) → m_new where:
  m_new = DualConstruction(context)
  status(m_new) = "active"
  status_reason(m_new) = "created"
  created_at(m_new) = current_timestamp
  updated_at(m_new) = created_at(m_new)
```

### Query Execution

```
Q(query, app_id, filters) → [m₁, m₂, ..., mₙ] where:

  1. Semantic search:
     candidates = vector_search(query, app_id, filters, n_results × 3)
     candidates = {m | similarity(m, query) ≥ RECALL_THRESHOLD}
  
  2. Composite scoring:
     scored = {(m, S_composite(m, query, t)) | m ∈ candidates}
  
  3. Precision filtering:
     filtered = {m | S_composite(m, query, t) ≥ COMPOSITE_SCORE_THRESHOLD}
  
  4. Top-K selection:
     results = top_k(filtered, n_results, key=S_composite)
  
  5. Ranking:
     results = sorted(results, key=S_composite, reverse=True)
```

### Fuzzy Deduplication

```
fuzzy_dedup(items, threshold=75) → deduped_items where:

  normalized_map = {item: normalize_token(item) | item ∈ items}
  seen = set()
  deduped = []
  
  for (orig, norm) in normalized_map.items():
    if norm in seen:
      continue
    seen.add(norm)
    
    close_matches = process.extract(
      norm,
      normalized_map.values(),
      scorer=fuzz.token_sort_ratio,
      score_cutoff=threshold
    )
    
    for (match, score, _) in close_matches:
      seen.add(match)
    
    deduped.append(orig)
  
  return deduped

where normalize_token(token):
  token = token.lower()
  token = re.sub(r"[-_]", " ", token)
  token = re.sub(r"[^a-z0-9\s]", "", token)
  token = re.sub(r"\s+", " ", token)
  return token.strip()
```

### Memory Link Generation and Tree Formation

#### Link Generation Algorithm

```
L(new_memory, app_id, filters) → [link₁, link₂, ..., linkₙ] where:

  1. Embedding extraction:
     e_new = embed(new_memory.content)
  
  2. Candidate retrieval:
     candidates = vector_search(
       embedding=e_new,
       collection=app_id,
       n_results=min(max_links × CANDIDATE_MULTIPLIER, MAX_EXPANDED_RESULTS),
       filter_conditions={status="active", user_id?, session_id?},
       similarity_threshold=LINK_THRESHOLD
     )
     where CANDIDATE_MULTIPLIER = 3
     where MAX_EXPANDED_RESULTS = 100
     
     Note: The vector search filters by similarity_threshold at the database level,
           so only candidates with similarity ≥ LINK_THRESHOLD are returned.
  
  3. Composite scoring:
     links = []
     for each candidate m in candidates:
       S_similarity = score from vector_search (already ≥ LINK_THRESHOLD)
       S_composite = S_similarity × (1 + β_recency(m) + β_importance(m))
       link = {
         memory_id: id(m),
         similarity_score: S_similarity,
         composite_score: S_composite,
         link_strength: S_composite,
         relationship_type: determine_relationship_type(S_composite, metadata(m))
       }
       links.append(link)
  
  4. Threshold filtering:
     meaningful_links = {link | link.composite_score ≥ LINK_THRESHOLD}
     Note: Since S_composite = S_similarity × (1 + β_recency + β_importance)
           and S_similarity ≥ LINK_THRESHOLD (from vector search) and boosts ≥ 0,
           we have S_composite ≥ S_similarity ≥ LINK_THRESHOLD.
           This filter ensures consistency but is mathematically redundant.
  
  5. Ranking:
     meaningful_links = sorted(meaningful_links, key=composite_score, reverse=True)
     meaningful_links = meaningful_links[:MAX_LINKS_FOR_SYNTHESIS]
     where MAX_LINKS_FOR_SYNTHESIS = 4
  
  return meaningful_links
```

#### Tree Formation Theorem

**Definition:** A memory evolution tree T = (V, E) is a directed acyclic graph (DAG) where:
- V = {memory nodes}
- E = {(mᵢ, mⱼ) | mᵢ.next_id = id(mⱼ)}

**Theorem 1 (Tree Structure):** The memory evolution structure forms a forest of forward-linked chains (singly linked lists).

**Proof:**
1. Each memory m has at most one `next_id` pointer: `|{m' | m'.next_id = id(m)}| ≤ 1`
2. The `next_id` relationship is asymmetric: if `m₁.next_id = id(m₂)`, then `m₂.next_id ≠ id(m₁)` (prevents cycles)
3. Cycle detection ensures: `∀ path m₁ → m₂ → ... → mₖ, m₁ ≠ mₖ`
4. Therefore, the structure is a DAG with maximum out-degree 1, forming chains

**Theorem 2 (Link Strength Property):** For a link between memories m₁ (new) and m₂ (existing):
```
link_strength(m₁, m₂) = S_composite(m₁, m₂) ≥ LINK_THRESHOLD

where:
  S_composite(m₁, m₂) = S_relevance(m₁, m₂) × (1 + β_recency(m₂) + β_importance(m₂))
  S_relevance(m₁, m₂) = cosine_similarity(embed(m₁), embed(m₂))
  LINK_THRESHOLD = 0.7 (default)
```

**Proof:**
- Link generation filters by `S_similarity ≥ LINK_THRESHOLD` (default: 0.7)
- Composite score: `S_composite = S_similarity × (1 + β_recency + β_importance)`
- Since `β_recency ≥ 0` and `β_importance ≥ 0`, we have `S_composite ≥ S_similarity`
- Final filtering ensures `S_composite ≥ LINK_THRESHOLD`
- Therefore, all links satisfy the minimum strength requirement: `link_strength(m₁, m₂) ≥ 0.7`

**Theorem 3 (Evolution Chain Traversal):** The evolution chain traversal algorithm is correct and terminates.

**Proof:**
- **Correctness:** Starting from memory_id, we follow `next_id` pointers forward
- **Termination:** 
  - Base case: If `next_id = null`, traversal stops
  - Cycle detection: `visited` set prevents infinite loops
  - Maximum depth: Bounded by number of memories in collection
- **Complexity:** O(k) where k = chain length (linear in chain size)

**Algorithm:**
```
traverse_chain(start_id, app_id) → [m₁, m₂, ..., mₖ]:

  chain = []
  current_id = start_id
  visited = set()
  
  while current_id ≠ null:
    if current_id ∈ visited:
      break  // Cycle detected
    visited.add(current_id)
    
    memory = get_memory(current_id, app_id)
    if memory = null:
      break  // End of chain
    
    chain.append(memory)
    current_id = memory.metadata.next_id
  
  return chain
```

### Memory Note Synthesis

#### Synthesis Function

The memory note is generated by combining episodic and summarizer agent outputs:

```
M_note(episodic_data, summary_data) → note_text where:

  note_text = concat(
    S_summary(summary_data),
    S_behavioral(episodic_data),
    S_experience(episodic_data),
    S_guidance(episodic_data),
    S_footer(summary_data, episodic_data)
  )
```

#### Section Generation Functions

**1. Summary Section:**
```
S_summary(summary_data) = 
  if summary_data.narrative ≠ "N/A" and summary_data.narrative ≠ "":
    return "## Summary\n" + summary_data.narrative
  else:
    return ""
```

**2. Behavioral Patterns Section:**
```
S_behavioral(episodic_data) = 
  behavioral = episodic_data.behavioral_profile
  insights = episodic_data.interaction_insights
  
  parts = []
  
  if behavioral ≠ null:
    style_parts = filter_N/A([
      behavioral.communication,
      behavioral.learning,
      behavioral.problem_solving,
      behavioral.decision_making
    ])
    if style_parts ≠ []:
      parts.append("User's style: " + join(style_parts, " "))
  
  if insights ≠ null:
    insight_parts = filter_N/A([
      "Engages well with: " + insights.engagement_triggers,
      "Struggles with: " + insights.friction_points,
      "Works best when: " + insights.optimal_approach
    ])
    if insight_parts ≠ []:
      parts.append(join(insight_parts, " "))
  
  if parts ≠ []:
    return "## Behavioral Patterns\n" + join(parts, " ")
  else:
    return ""
```

**3. Experience Learnings Section:**
```
S_experience(episodic_data) = 
  worked = episodic_data.what_worked
  failed = episodic_data.what_failed
  parts = []
  
  if worked ≠ null and worked.strategies ≠ []:
    clean_strategies = filter_N/A(worked.strategies)
    if clean_strategies ≠ []:
      text = "Successful approaches: " + join(clean_strategies, "; ")
      if worked.pattern ≠ "N/A":
        text += " — " + worked.pattern
      parts.append(text)
  
  if failed ≠ null and failed.strategies ≠ []:
    clean_strategies = filter_N/A(failed.strategies)
    if clean_strategies ≠ []:
      text = "Approaches to avoid: " + join(clean_strategies, "; ")
      if failed.pattern ≠ "N/A":
        text += " — " + failed.pattern
      parts.append(text)
  
  if parts ≠ []:
    return "## Experience Learnings\n" + join(parts, " ")
  else:
    return ""
```

**4. Guidance Section:**
```
S_guidance(episodic_data) = 
  guidance = episodic_data.future_guidance
  if guidance = null:
    return ""
  
  parts = []
  
  if guidance.recommended_approaches ≠ []:
    clean_rec = filter_N/A(guidance.recommended_approaches)
    if clean_rec ≠ []:
      parts.append("Do: " + join(clean_rec, "; "))
  
  if guidance.avoid_approaches ≠ []:
    clean_avoid = filter_N/A(guidance.avoid_approaches)
    if clean_avoid ≠ []:
      parts.append("Don't: " + join(clean_avoid, "; "))
  
  if guidance.adaptation_note ≠ "N/A":
    parts.append("Key insight: " + guidance.adaptation_note)
  
  if parts ≠ []:
    return "## Guidance\n" + join(parts, " ")
  else:
    return ""
```

**5. Footer Section:**
```
S_footer(summary_data, episodic_data) = 
  retrieval = summary_data.retrieval
  metadata = summary_data.metadata
  parts = []
  
  if retrieval ≠ null:
    tags = filter_N/A(retrieval.tags)
    keywords = filter_N/A(retrieval.keywords)
    
    if tags ≠ []:
      parts.append("Tags: " + join(tags, ", "))
    if keywords ≠ []:
      parts.append("Keywords: " + join(keywords, ", "))
  
  if metadata ≠ null:
    meta_parts = []
    if metadata.depth ≠ "N/A":
      meta_parts.append("Content depth: " + metadata.depth)
    
    follow_ups = filter_N/A(metadata.follow_ups)
    if follow_ups ≠ []:
      meta_parts.append("Follow-up areas: " + join(follow_ups, "; "))
    
    if meta_parts ≠ []:
      parts.append(join(meta_parts, " | "))
  
  if parts ≠ []:
    return "---\n" + join(parts, " | ")
  else:
    return ""
```

#### Synthesis Properties

**Theorem 4 (Synthesis Completeness):** The synthesis function preserves all non-N/A information from both agents.

**Proof:**
- Each section function `S_i` filters out only "N/A" values and empty strings
- All valid data from episodic_data and summary_data is included in at least one section
- The `filter_N/A` function preserves all non-N/A values: `filter_N/A(x) = {x | x ≠ "N/A" and x ≠ ""}`
- Therefore, `∀ valid_data ∈ episodic_data ∪ summary_data, valid_data appears in M_note`

**Theorem 5 (Synthesis Structure):** The synthesized note maintains a consistent Zettelkasten structure.

**Proof:**
- Structure is deterministic: sections appear in fixed order
- Each section has a well-defined header: `## SectionName`
- Sections are separated by `\n\n` (double newline)
- Footer is separated by `---\n`
- Therefore, the structure is consistent and parseable

**Theorem 6 (Synthesis Normalization):** The final note is normalized (no excessive whitespace).

**Proof:**
- Multiple newlines are collapsed: `re.sub(r"\n{3,}", "\n\n", text)`
- Multiple spaces are collapsed: `re.sub(r" {2,}", " ", text)`
- Leading/trailing whitespace is removed: `text.strip()`
- Therefore, the output is normalized

#### Mathematical Formulation

The complete synthesis can be expressed as:

```
M_note(E, S) = normalize(concat(
  S_summary(S),
  S_behavioral(E),
  S_experience(E),
  S_guidance(E),
  S_footer(S, E)
))

where:
  E = episodic_data
  S = summary_data
  normalize(text) = re.sub(r"\n{3,}", "\n\n", re.sub(r" {2,}", " ", text.strip()))
  concat(sections) = join(filter_empty(sections), "\n\n")
  filter_empty(sections) = {s | s ≠ ""}
```

**Complexity Analysis:**
- Time: O(|E| + |S|) where |E|, |S| are sizes of episodic and summary data
- Space: O(|M_note|) where |M_note| is the length of the final note
- The synthesis is linear in input size

---

## API & SDK Interface

### SDK Methods

#### Memory Operations

**`add_memory(app_id, user_id, messages, session_id=None)`**
- Standard memory creation with full pipeline
- Returns: `{"task_id": "uuid", "status": "accepted", ...}`
- Async background processing

**`query_memory(app_id, query, n_results=None, user_id=None, session_id=None, similarity_threshold=None)`**
- Semantic search with composite scoring
- Returns: List of memory objects with scores
- Synchronous (fast query path)

**`get_memory(memory_id, app_id)`**
- Get single memory by ID
- Returns: Memory dictionary or None
- Synchronous

**`delete_memory(app_id, memory_id)`**
- Mark memory as deleted
- Returns: Boolean success status
- Synchronous

**`update_memory_status(app_id, memory_id, new_status, archive_reason)`**
- Update memory status
- Returns: MemoryOperationResult
- Synchronous

**`traverse_memory_evolution_chain(app_id, memory_id)`**
- Traverse evolution chain forward
- Returns: List of memories in evolution order
- Synchronous

**`generate_evolution_graph(chain, format="mermaid")`**
- Generate graph visualization
- Formats: "mermaid", "dot", "html"
- Returns: Graph string
- Synchronous

#### Agent Operations

**`add_agent_memory(agent_request)`**
- Fast agent memory storage
- Returns: `{"task_id": "uuid", "status": "accepted", ...}`
- Async background processing

**`summarize_conversation(summary_request)`**
- Conversation summarization
- Returns: Summary dict (sync) or task info (async with callback)
- Sync or async depending on callback URL

#### System Operations

**`warm_up()`**
- Pre-initialize connection pool
- Returns: Boolean success status
- Synchronous

**`get_connection_pool_stats()`**
- Pool metrics
- Returns: Dictionary with pool statistics
- Synchronous

**`get_task_status(task_id)`**
- Get background task status
- Returns: Task status and result (if completed)
- Synchronous

### API Endpoints

#### Memory Endpoints

**`POST /api/v1/memories`**
- Create memory (async)
- Request: `AddUserMessageRequest`
- Response: `TaskResponse` (202 Accepted)

**`GET /api/v1/memories/query`**
- Query memories
- Query params: `app_id`, `query`, `user_id?`, `session_id?`, `n_results?`, `similarity_threshold?`
- Response: `MemoryListResponse`

**`GET /api/v1/memories/{memory_id}`**
- Get memory by ID
- Query params: `app_id`
- Response: `MemoryResponse`

**`DELETE /api/v1/memories/{memory_id}`**
- Delete memory
- Query params: `app_id`
- Response: `SuccessResponse`

**`GET /api/v1/memories/{memory_id}/evolution`**
- Traverse evolution chain
- Query params: `app_id`
- Response: `MemoryListResponse`

**`GET /api/v1/memories/{memory_id}/evolution/graph`**
- Generate evolution graph
- Query params: `app_id`, `format?` (mermaid/dot/html)
- Response: Graph string or HTML

#### Agent Endpoints

**`POST /api/v1/agent/memories`**
- Create agent memory (async)
- Request: `AgentMemoryRequest`
- Response: `TaskResponse` (202 Accepted)

**`POST /api/v1/agent/summaries`**
- Generate summary (sync or async)
- Request: `ConversationSummaryRequest`
- Response: `ConversationSummaryResponse` (200) or `TaskResponse` (202)

#### System Endpoints

**`GET /api/v1/system/pool-stats`**
- Connection pool statistics
- Response: Pool stats dictionary

**`GET /health`**
- Health check
- Response: Health status dictionary

**`GET /metrics`**
- Prometheus metrics
- Response: Prometheus metrics format

**`GET /`**
- API information
- Response: API metadata

---

## Implementation Details

### Asynchronous Processing

**Framework:** Python `asyncio` (not Celery)

**Non-Blocking:**
- All I/O operations use `async`/`await`
- Background tasks via `asyncio.create_task`
- Connection pooling with async context managers

**Background Task Lifecycle:**
1. Task created with unique task ID
2. Registered in `_background_tasks` dictionary
3. Cleanup callback registered
4. Task executes asynchronously
5. Task removed from registry on completion

### Storage Backend

**Vector Database:** Qdrant (async client)

**Connection Pool:**
- Configurable pool size (default: 10)
- Retry logic with exponential backoff
- Automatic connection lifecycle management
- Pool statistics tracking

**Metadata:**
- Stored alongside embeddings in Qdrant
- Query-time filtering by metadata fields
- Status filtering always applied (`status="active"`)

### Conflict Resolution

**Agent-Based:**
- LLM agents analyze relationships
- Deterministic decisions based on content analysis
- Confidence scores for each decision

**Status Updates:**
- Simple status + reason tracking
- Batch processing for efficiency (chunk size: 10)
- Evolution chain linking via `next_id`

### Error Handling

**Retry Logic:**
- Exponential backoff for transient failures
- Max 3 retries for webhook callbacks
- Permanent error detection

**Graceful Degradation:**
- System continues operating on partial failures
- Fallback to SKIP on agent failures
- Comprehensive error logging

**Comprehensive Logging:**
- Structured logging with context
- Operation tracking with metrics
- Error details preserved in results

---

## Scalability & Performance

### Query Performance

**Vector Search:**
- O(log n) retrieval via approximate nearest neighbor search
- Configurable similarity thresholds for filtering
- Candidate multiplier for quality results

**Composite Scoring:**
- Computed in-memory after retrieval
- O(k) where k = number of candidates
- Efficient score calculation

**Filtering:**
- Query-time metadata filtering
- Status filtering always applied
- Optional user_id/session_id filtering

### Write Performance

**User-Perceived Latency:**
- O(1) - immediate response with task ID
- Actual processing in background
- No blocking on LLM or database operations

**Connection Pooling:**
- Prevents resource exhaustion
- Reuses connections efficiently
- Configurable pool size

**Batch Operations:**
- Chunked status/timestamp updates
- Efficient batch processing
- Configurable chunk size (default: 10)

### Resource Management

**Connection Pool:**
- Limits concurrent database operations
- Prevents resource exhaustion
- Tracks pool utilization

**Background Tasks:**
- Automatic cleanup on completion
- Task registry for status tracking
- No unbounded growth

**Embedding Cache:**
- In-memory cache with TTL
- Max 512 entries
- Automatic expiration and pruning

**Metrics:**
- In-memory storage (no external dependencies)
- Efficient metric collection
- Optional Prometheus endpoint

---

## Production Features

### Observability

**Prometheus Metrics:**
- Operation counts and durations
- Error tracking
- Batch operation statistics
- Connection pool metrics

**Structured Logging:**
- Context-aware logging
- Operation tracking
- Error details preserved

**Connection Pool Statistics:**
- Pool size and utilization
- Active connections
- Pool health status

**Health Status Tracking:**
- SDK initialization status
- Vector database availability
- Service health endpoint

### Reliability

**Retry Logic:**
- Exponential backoff for transient failures
- Max 3 retries for webhook callbacks
- Permanent error detection

**Graceful Error Handling:**
- System continues operating on partial failures
- Fallback to SKIP on agent failures
- Comprehensive error logging

**Connection Pool Resilience:**
- Automatic connection lifecycle management
- Retry logic with exponential backoff
- Graceful degradation on failures

**Webhook Callback Retry:**
- Max 3 attempts with exponential backoff
- Permanent error detection
- Error payload delivery on failure

### Simplicity

**Status-Based Evolution:**
- Simple status + reason tracking
- No complex versioning metadata
- Easy to understand and debug

**Clear Operation Paths:**
- Standard vs. agent vs. summary
- Clear use cases for each path
- Straightforward API design

**Minimal Dependencies:**
- In-memory metrics (no Redis)
- Simple status tracking
- Easy to deploy and maintain

---

## Configuration

### Environment Variables

**Essential LLM Configuration:**
- `LLM_API_KEY`: LLM API key (required)
- `LLM_PROVIDER`: LLM provider (required)
- `LLM_MODEL`: LLM model name (required)
- `LLM_TEMPERATURE`: Temperature (default: 0.4)
- `LLM_MAX_TOKENS`: Max tokens (default: 3000)
- `LLM_TOP_P`: Top-p sampling (default: 0.9)

**Essential Embedding Configuration:**
- `EMBEDDING_API_KEY`: Embedding API key (required)
- `EMBEDDING_PROVIDER`: Embedding provider (required)
- `EMBEDDING_MODEL`: Embedding model name (required)
- `EMBEDDING_DIMENSIONS`: Embedding dimensions (required)
- `EMBEDDING_ENCODING_FORMAT`: Encoding format (default: "base64")
- `EMBEDDING_TIMEOUT`: Timeout in seconds (default: 600)

**Qdrant Connection:**
- `OMNI_MEMORY_PROVIDER`: Provider type (default: "qdrant-remote")
- `QDRANT_HOST`: Qdrant host (default: "localhost")
- `QDRANT_PORT`: Qdrant port (default: 6333)

**OmniMemory Hyperparameters:**
- `OMNIMEMORY_DEFAULT_MAX_MESSAGES`: Max messages (default: 30)
- `OMNIMEMORY_RECALL_THRESHOLD`: Recall threshold (default: 0.3)
- `OMNIMEMORY_COMPOSITE_SCORE_THRESHOLD`: Composite threshold (default: 0.4)
- `OMNIMEMORY_DEFAULT_N_RESULTS`: Default results (default: 10)
- `OMNIMEMORY_LINK_THRESHOLD`: Link threshold (default: 0.7)
- `OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS`: Pool size (default: 10)

**Metrics & Observability:**
- `OMNIMEMORY_ENABLE_METRICS_SERVER`: Enable metrics (default: false)
- `OMNIMEMORY_METRICS_PORT`: Metrics port (default: 9001)

**Logging:**
- `LOG_LEVEL`: Log level (default: "INFO")
- `LOG_DIR`: Log directory (default: "./logs")

---

## Future Enhancements

### Memory Evolution Decay and Long-Term Storage

**Concept:** Inspired by human memory consolidation, where older memories (years old) are archived to long-term storage while remaining accessible when needed. This reduces vector database storage costs while preserving memory history.

**Proposed Architecture:**

**Decay Threshold:**
- Memories older than configurable threshold (e.g., 10 years) are candidates for archival
- Recency score naturally decays to near-zero for very old memories
- Threshold configurable via `OMNIMEMORY_ARCHIVAL_AGE_YEARS` (default: 10)

**Archival Process:**
1. **Identification:** Periodic scan for memories where:
   - `age_years ≥ ARCHIVAL_AGE_YEARS`
   - `status = "active"` or `status = "updated"`
   - Recency score < threshold (e.g., < 0.01)
2. **Streaming to External Storage:**
   - Export memory note, metadata, and evolution chain to external file storage (S3, GCS, local filesystem)
   - Format: JSON or compressed archive per memory
   - Preserve complete evolution chain in archival format
3. **Vector Database Cleanup:**
   - Remove embedding from vector database (reduces storage)
   - Mark as `status = "archived"` with `status_reason = "age_decay"`
   - Store archival location in metadata (if supported)
4. **Retrieval on Demand:**
   - When querying, if archived memory would have been relevant:
     - Retrieve from external storage
     - Re-embed if needed (optional, for very old memories)
     - Return in results with archival flag

**Mathematical Formulation:**
```
ARCHIVE(m, t) → (m_archived, location) where:
  age_hours(m, t) = (t - max(created_at(m), updated_at(m))) / 3600
  age_years(m, t) = age_hours(m, t) / (365.25 × 24)
  
  if age_years(m, t) ≥ ARCHIVAL_AGE_YEARS:
    location = stream_to_external_storage(m, evolution_chain(m))
    delete_embedding(m)  // Remove from vector DB
    status(m) = "archived"
    status_reason(m) = "age_decay"
    archival_location(m) = location
    archived_at(m) = t
    updated_at(m) = t
```

**Archival Criteria:**
- Age threshold: `age_years ≥ ARCHIVAL_AGE_YEARS` (default: 10 years)
- Recency decay: `recency_score < ARCHIVAL_RECENCY_THRESHOLD` (default: 0.01, corresponds to ~10 years with current half-life of 1.25 years)
- Status filter: Only `status = "active"` or `status = "updated"` (exclude "deleted")
- Evolution chain: Archive entire chain together for context preservation

**Benefits:**
- **Storage Efficiency:** Reduces vector database size for very old memories (10+ years)
- **Cost Optimization:** External storage (S3, GCS, Azure Blob, local filesystem) is cheaper than vector DB storage
- **Memory Preservation:** Complete history preserved with evolution chains, just not in active search
- **Human-Like Behavior:** Mimics how human brains archive old memories (10+ years) to long-term storage while keeping them accessible when needed
- **Scalability:** Enables indefinite memory retention without vector DB bloat

**Implementation Considerations:**
- Background archival job (periodic scan, e.g., daily or weekly)
- Configurable archival age threshold (`OMNIMEMORY_ARCHIVAL_AGE_YEARS`, default: 10)
- Support for multiple external storage backends (S3, GCS, Azure Blob, local filesystem)
- Evolution chain preservation: Archive entire chain together for context
- Optional re-embedding for archived memories on retrieval (if high relevance detected)
- Metadata preservation: All metadata (tags, keywords, timestamps) preserved in archival format
- Query integration: Archived memories can be retrieved on-demand if highly relevant to query

**Other Potential Additions:**
- Memory linking pipeline for agent memories
- Advanced graph queries (if needed)
- Multi-tenant isolation enhancements
- Advanced caching strategies (LRU, LFU)
- Redis-based metrics for multi-process scenarios (if needed)

---

## Critical Review and Design Decisions

### Correctness Verification

**Status Consistency:** The UPDATE operation correctly sets `status_reason="consolidated"` for both the new consolidated memory and all old memories being consolidated. This ensures consistent status tracking throughout the evolution chain.

**Link Generation:** The link generation algorithm correctly filters by similarity threshold first, then computes composite scores. This optimization reduces unnecessary composite score calculations for low-similarity candidates.

**Tree Structure:** The evolution chain forms a DAG (directed acyclic graph) with maximum out-degree 1, creating forward-linked chains. Cycle detection prevents infinite loops during traversal.

**Synthesis Completeness:** The memory note synthesis preserves all non-N/A information from both episodic and summarizer agents, ensuring no data loss during the combination process.

### Known Limitations

**Embedding Cache Size:** The embedding cache is hardcoded to 512 entries. For high-volume systems, this may need to be configurable.

**Connection Pool:** Default pool size of 10 may be insufficient for high-concurrency scenarios. Monitor pool statistics and adjust `OMNIMEMORY_VECTOR_DB_MAX_CONNECTIONS` accordingly.

**Batch Operations:** Status and timestamp updates are processed in chunks of 10. For large-scale consolidations, this may create many small batches. Consider making chunk size configurable.

**Agent Memory Path:** The fast agent memory path bypasses conflict resolution. This is intentional for speed but means agent memories may not be linked to related standard memories.

### Design Trade-offs

**Status-Based vs Versioning:** Chose simple status-based tracking over complex versioning to reduce metadata overhead and simplify queries. Trade-off: less detailed evolution history, but faster queries and simpler implementation.

**In-Memory Metrics vs Redis:** Chose in-memory metrics to reduce dependencies. Trade-off: metrics lost on restart, but no external service required. For multi-process deployments, Redis-based metrics can be re-enabled.

**Async Background Tasks vs Celery:** Chose `asyncio` tasks over Celery for simplicity. Trade-off: tasks lost on process restart, but no message broker required. For distributed systems, Celery integration can be added.

**Composite Score Bounds:** Recency and importance boosts are capped at 10% each. This ensures relevance remains primary while allowing temporal and importance signals to modulate results. Trade-off: very recent or important but irrelevant memories still won't rank highly, which is the desired behavior.

### Future Enhancements

**Potential Improvements:**
- Configurable embedding cache size
- Redis-based metrics for multi-process deployments
- Celery integration for distributed task processing
- Memory linking pipeline for agent memories
- Advanced graph queries for complex relationship traversal
- Multi-tenant isolation enhancements
- Advanced caching strategies (LRU, LFU)

## Conclusion

SECMSA provides a production-ready, scalable memory architecture for autonomous AI systems. By combining dual-agent synthesis, intelligent conflict resolution, composite scoring, and fully asynchronous processing, the system enables agents to maintain coherent, evolving memory stores that adapt to new information while preserving semantic relevance in retrieval.

The architecture prioritizes simplicity and reliability, using status-based tracking instead of complex versioning, and in-memory metrics instead of external dependencies. This design enables rapid deployment, easy debugging, and straightforward scaling as agent workloads grow.

The system's mathematical foundations ensure provable guarantees about retrieval optimality and semantic coherence, while the asynchronous architecture provides O(1) user-perceived latency and efficient resource utilization. The multiple memory creation paths accommodate different use cases, from rich conversational context to fast agent message storage.

SECMSA represents a paradigm shift from passive memory storage to active cognitive synthesis, where memories autonomously organize, evolve, and adapt to maintain semantic coherence in autonomous AI systems.

