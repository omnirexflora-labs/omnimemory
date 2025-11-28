"""
OmniMemory REST API - Complete Real-World Example
==================================================

Scenario: Customer Support Agent via REST API
- Language-agnostic HTTP calls
- Demonstrates ALL REST endpoints
- Shows how to integrate from any programming language

This example uses Python's httpx but can be adapted to:
- Node.js (axios, fetch)
- Go (net/http)
- Rust (reqwest)
- Java (HttpClient)
- PHP (Guzzle)
"""

import asyncio
import httpx
import json
from datetime import datetime


class OmniMemoryAPIClient:
    """REST API client for OmniMemory (can be ported to any language)."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def health_check(self):
        """Check API health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def add_memory(
        self, app_id: str, user_id: str, messages: list, session_id: str = None
    ):
        """Add structured memory via REST API."""
        payload = {"app_id": app_id, "user_id": user_id, "messages": messages}
        if session_id:
            payload["session_id"] = session_id

        response = await self.client.post(
            f"{self.base_url}/api/v1/memories", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def add_agent_memory(
        self, app_id: str, user_id: str, messages, session_id: str = None
    ):
        """Add agent memory via REST API."""
        payload = {"app_id": app_id, "user_id": user_id, "messages": messages}
        if session_id:
            payload["session_id"] = session_id

        response = await self.client.post(
            f"{self.base_url}/api/v1/agent/memories", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def query_memory(
        self,
        app_id: str,
        query: str,
        user_id: str = None,
        session_id: str = None,
        n_results: int = 5,
        similarity_threshold: float = None,
    ):
        """Query memories via REST API."""
        params = {"app_id": app_id, "query": query}
        if user_id:
            params["user_id"] = user_id
        if session_id:
            params["session_id"] = session_id
        if n_results:
            params["n_results"] = n_results
        if similarity_threshold:
            params["similarity_threshold"] = similarity_threshold

        response = await self.client.get(
            f"{self.base_url}/api/v1/memories/query", params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_memory(self, app_id: str, memory_id: str):
        """Get specific memory by ID via REST API."""
        response = await self.client.get(
            f"{self.base_url}/api/v1/memories/{memory_id}", params={"app_id": app_id}
        )
        response.raise_for_status()
        return response.json()

    async def delete_memory(self, app_id: str, memory_id: str):
        """Delete memory via REST API."""
        response = await self.client.delete(
            f"{self.base_url}/api/v1/memories/{memory_id}", params={"app_id": app_id}
        )
        response.raise_for_status()
        return response.json()

    async def traverse_evolution(self, app_id: str, memory_id: str):
        """Traverse memory evolution chain via REST API."""
        response = await self.client.get(
            f"{self.base_url}/api/v1/memories/{memory_id}/evolution",
            params={"app_id": app_id},
        )
        response.raise_for_status()
        return response.json()

    async def summarize_conversation(
        self,
        app_id: str,
        user_id: str,
        messages,
        callback_url: str = None,
        session_id: str = None,
    ):
        """Summarize conversation via REST API."""
        payload = {"app_id": app_id, "user_id": user_id, "messages": messages}
        if session_id:
            payload["session_id"] = session_id
        if callback_url:
            payload["callback_url"] = callback_url

        response = await self.client.post(
            f"{self.base_url}/api/v1/agent/summaries", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def batch_add_messages(
        self, app_id: str, user_id: str, messages: list, session_id: str = None
    ):
        """Add messages to batcher via REST API."""
        payload = {"app_id": app_id, "user_id": user_id, "messages": messages}
        if session_id:
            payload["session_id"] = session_id

        response = await self.client.post(
            f"{self.base_url}/api/v1/memory-batcher/messages", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def get_pool_stats(self):
        """Get connection pool statistics via REST API."""
        response = await self.client.get(f"{self.base_url}/api/v1/system/pool-stats")
        response.raise_for_status()
        return response.json()


async def main():
    """Complete customer support workflow via REST API."""

    print("🌐 OmniMemory REST API Example")
    print("=" * 60)

    # Initialize API client
    client = OmniMemoryAPIClient(base_url="http://localhost:8001")

    # Customer context
    app_id = "customer-support-prod"
    user_id = "customer-bob-2024"
    session_id = "session-dec-28-api"

    try:
        # ==========================================
        # STEP 1: Health Check
        # ==========================================
        print("\n🏥 Checking API health...")
        health = await client.health_check()
        print(f"✅ API Status: {health.get('status')}")
        print(f"   SDK Initialized: {health.get('sdk_initialized')}")

        # ==========================================
        # STEP 2: Add Memory (POST /api/v1/memories)
        # ==========================================
        print("\n💬 Adding customer conversation via REST API...")

        messages = [
            {"role": "user", "content": "Hi, I need help with billing"},
            {
                "role": "assistant",
                "content": "I'd be happy to help with your billing question",
            },
            {"role": "user", "content": "Why was I charged twice this month?"},
            {"role": "assistant", "content": "Let me look into your billing history"},
            {"role": "user", "content": "I'm on the annual plan, not monthly"},
            {
                "role": "assistant",
                "content": "I see the issue - there was a duplicate charge",
            },
            {"role": "user", "content": "Can you refund the extra charge?"},
            {
                "role": "assistant",
                "content": "Absolutely, I'm processing the refund now",
            },
            {
                "role": "user",
                "content": "Thanks! Also, please update my payment method to the new card ending in 4567",
            },
            {
                "role": "assistant",
                "content": "Done! Your payment method has been updated",
            },
        ]

        add_response = await client.add_memory(
            app_id=app_id, user_id=user_id, session_id=session_id, messages=messages
        )

        task_id = add_response["task_id"]
        print(f"✅ Memory task created: {task_id}")
        print(f"   Status: {add_response['status']}")

        # NOTE: Fire-and-forget async processing
        # Task status is NOT persisted - check logs for errors
        print("\n💡 Fire-and-Forget: Memory processes in background")
        print("   Check application logs if you need to debug")

        # Wait reasonable time for processing
        print("⏳ Waiting for background processing...")
        await asyncio.sleep(5)

        # ==========================================
        # STEP 3: Query Memory (GET /api/v1/memories/query)
        # ==========================================
        print("\n🔍 Querying memory about billing issues...")

        query_result = await client.query_memory(
            app_id=app_id,
            user_id=user_id,
            query="What billing issues did the customer have?",
            n_results=3,
            similarity_threshold=0.6,
        )

        memories = query_result.get("memories", [])
        print(f"✅ Found {len(memories)} relevant memories:")
        for i, memory in enumerate(memories, 1):
            print(f"\n  Memory {i}:")
            print(f"    Content: {memory.get('document', '')[:100]}...")
            print(f"    Score: {memory.get('composite_score', 0):.3f}")

        # ==========================================
        # STEP 4: Add Agent Memory (POST /api/v1/agent/memories)
        # ==========================================
        print("\n🤖 Agent saving new insight...")

        agent_response = await client.add_agent_memory(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            messages="Customer Bob prefers refunds to account credit. Payment method updated to card ending 4567.",
        )

        print(f"✅ Agent memory saved: {agent_response['task_id']}")

        # ==========================================
        # STEP 5: Memory Batcher (POST /api/v1/memory-batcher/messages)
        # ==========================================
        print("\n📨 Using memory batcher for streaming messages...")

        stream_messages = [
            {"role": "user", "content": "The refund appeared in my account!"},
            {"role": "assistant", "content": "Wonderful! I'm glad it was processed quickly"},
            {"role": "user", "content": "Yes, thank you for the excellent service"},
            {"role": "assistant", "content": "You're welcome! Is there anything else?"},
            {"role": "user", "content": "I also want to know when my subscription expires"},
            {"role": "assistant", "content": "Your subscription is valid until December 2025."},
            {"role": "user", "content": "That's great news, thank you."},
            {"role": "assistant", "content": "You're welcome! Do you have any other questions?"},
            {"role": "user", "content": "No, that's all for now."},
            {"role": "assistant", "content": "Have a wonderful day!"},
            {"role": "user", "content": "Goodbye!"},
        ]

        for msg in stream_messages:
            batch_response = await client.batch_add_messages(
                app_id=app_id,
                user_id=user_id,
                session_id=session_id,
                messages=[msg],
            )
            print(
                f"   Buffered: {msg['role'][:4]}... ({batch_response.get('pending_messages')}/{batch_response.get('batch_size')})"
            )
            if batch_response.get('status') == 'flushed':
                print("   🌊 Batch FLUSHED to storage!")

        # ==========================================
        # STEP 6: Summarize Conversation (POST /api/v1/agent/summaries)
        # ==========================================
        print("\n📝 Summarizing conversation (sync mode)...")

        summary_response = await client.summarize_conversation(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            messages=messages[:5],  # First 5 for demo
        )

        print(f"✅ Summary generated:")
        print(f"   {summary_response.get('summary', '')[:200]}...")
        print(f"   Delivery: {summary_response.get('delivery')}")

        # ==========================================
        # STEP 7: Get Specific Memory (GET /api/v1/memories/{id})
        # ==========================================
        if memories:
            print("\n📄 Retrieving specific memory by ID...")
            memory_id = memories[0].get("id") or memories[0].get("memory_id")

            if memory_id:
                specific = await client.get_memory(app_id=app_id, memory_id=memory_id)

                print(f"✅ Retrieved memory {memory_id[:16]}...")
                print(f"   Content: {specific.get('document', '')[:100]}...")

        # ==========================================
        # STEP 8: Traverse Evolution (GET /api/v1/memories/{id}/evolution)
        # ==========================================
        if memories:
            print("\n🔗 Traversing memory evolution chain...")
            memory_id = memories[0].get("id") or memories[0].get("memory_id")

            if memory_id:
                evolution = await client.traverse_evolution(
                    app_id=app_id, memory_id=memory_id
                )

                chain_memories = evolution.get("memories", [])
                print(f"✅ Evolution chain: {len(chain_memories)} memories")
                print(f"   Count: {evolution.get('count')}")

        # ==========================================
        # STEP 9: Connection Pool Stats (GET /api/v1/system/pool-stats)
        # ==========================================
        print("\n📊 Checking connection pool statistics...")

        pool_stats = await client.get_pool_stats()
        print(f"✅ Connection Pool:")
        print(f"   Max: {pool_stats.get('max_connections')}")
        print(f"   Active: {pool_stats.get('active_handlers')}")
        print(f"   Available: {pool_stats.get('available_handlers')}")

        # ==========================================
        # STEP 10: Delete Memory (DELETE /api/v1/memories/{id})
        # ==========================================
        print("\n🗑️  Demonstrating memory deletion...")

        # Create test memory
        test_response = await client.add_agent_memory(
            app_id=app_id, user_id=user_id, messages="TEST MEMORY FOR DELETION"
        )

        await asyncio.sleep(3)

        # Query to find it
        test_query = await client.query_memory(
            app_id=app_id,
            user_id=user_id,
            query="TEST MEMORY FOR DELETION",
            n_results=1,
        )

        if test_query.get("memories"):
            test_id = test_query["memories"][0].get("id") or test_query["memories"][
                0
            ].get("memory_id")
            if test_id:
                delete_response = await client.delete_memory(
                    app_id=app_id, memory_id=test_id
                )

                print(f"✅ Memory deleted:")
                print(f"   Success: {delete_response.get('success')}")
                print(f"   Message: {delete_response.get('message')}")

        # ==========================================
        # SUMMARY
        # ==========================================
        print("\n" + "=" * 60)
        print("🎉 COMPLETE REST API WORKFLOW DEMONSTRATED!")
        print("=" * 60)
        print("\nAll REST Endpoints Used:")
        print("  ✅ GET  /health - Health check")
        print("  ✅ POST /api/v1/memories - Add structured memory")
        print("  ✅ GET  /api/v1/memories/query - Query memories")
        print("  ✅ GET  /api/v1/memories/{id} - Get specific memory")
        print("  ✅ DELETE /api/v1/memories/{id} - Delete memory")
        print("  ✅ GET  /api/v1/memories/{id}/evolution - Evolution chain")
        print("  ✅ POST /api/v1/agent/memories - Add agent memory")
        print("  ✅ POST /api/v1/agent/summaries - Summarize conversation")
        print("  ✅ POST /api/v1/memory-batcher/messages - Batch messages")
        print("  ✅ GET  /api/v1/system/pool-stats - Pool statistics")
        print("\n💡 Language Agnostic - This pattern works in:")
        print("   - Node.js (axios, fetch)")
        print("   - Go (net/http)")
        print("   - Rust (reqwest)")
        print("   - Java (HttpClient)")
        print("   - PHP (Guzzle)")
        print("   - Any language with HTTP client!")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
