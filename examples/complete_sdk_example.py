"""
OmniMemory SDK - Complete Real-World Example
==============================================

Scenario: Customer Support Agent with Memory
- Agent remembers customer preferences
- Handles conversation history
- Summarizes long conversations
- Tracks customer journey evolution

This example demonstrates ALL SDK methods in a realistic use case.
"""

import asyncio
from datetime import datetime
from omnimemory.sdk import OmniMemorySDK
from omnimemory import AgentMemorySDK
from omnimemory.core.schemas import (
    UserMessages,
    Message,
    AgentMemoryRequest,
    ConversationSummaryRequest,
)


async def main():
    """Complete customer support agent workflow."""

    # Initialize SDKs
    print("🧠 Initializing OmniMemory SDK...")
    sdk = OmniMemorySDK()
    agent_sdk = AgentMemorySDK()

    # ==========================================
    # STEP 1: Warm Up (CRITICAL)
    # ==========================================
    print("\n📡 Warming up connection pools...")
    if not await sdk.warm_up():
        print("❌ Failed to warm up SDK")
        return
    print("✅ SDK ready!")

    # Customer context
    app_id = "customer-support-prod"
    user_id = "customer-alice-2024"
    session_id = "session-dec-28-2024"

    # ==========================================
    # STEP 2: Add Structured Memory (Full Conversation)
    # ==========================================
    print("\n💬 Adding customer conversation memory...")

    # Simulate 10 messages from a support conversation
    conversation_messages = [
        Message(
            role="user", content="Hi, I'm having trouble accessing my premium account"
        ),
        Message(
            role="assistant",
            content="I'd be happy to help. Can you describe the issue?",
        ),
        Message(
            role="user",
            content="I upgraded yesterday but still see the free tier features",
        ),
        Message(role="assistant", content="Let me check your account status"),
        Message(role="user", content="Also, I prefer dark mode but it keeps resetting"),
        Message(
            role="assistant",
            content="I see both issues. Your upgrade is being processed",
        ),
        Message(role="user", content="How long does processing take?"),
        Message(
            role="assistant",
            content="Usually 24-48 hours. I've escalated for faster processing",
        ),
        Message(
            role="user",
            content="Thanks! Also noting I'm in the EU for billing purposes",
        ),
        Message(
            role="assistant", content="Perfect, I've updated your billing region to EU"
        ),
    ]

    add_response = await sdk.add_memory(UserMessages(
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        messages=conversation_messages
    ))

    task_id = add_response["task_id"]
    print(f"✅ Memory creation task started: {task_id}")
    print(f"   Status: {add_response['status']}")

    # NOTE: OmniMemory uses fire-and-forget async processing
    # Task status is NOT persisted, so there's no way to poll for completion
    # Check your application logs for any errors during processing
    print("\n💡 Fire-and-Forget: Memory processes in background")
    print("   ✅ As long as task accepted, it will complete")
    print("   📋 Check logs if you need to debug errors")

    # Wait a reasonable time for processing to complete
    print("\n⏳ Waiting for background processing (reasonable delay)...")
    await asyncio.sleep(5)  # Give it time to process

    # ==========================================
    # STEP 4: Query Memory (Semantic Search)
    # ==========================================
    print("\n🔍 Querying memory about customer preferences...")

    results = await sdk.query_memory(
        app_id=app_id,
        user_id=user_id,
        query="What are the customer's preferences and issues?",
        n_results=5,
        similarity_threshold=0.5,
    )

    print(f"Found {len(results)} relevant memories:")
    for i, memory in enumerate(results, 1):
        # print(f"{memory}")
        print(f"\n  Memory {i}:")
        print(f"    Content: {memory['memory_note'][:100]}...")
        print(f"    rank: {memory.get('rank', 'N/A')}")
        print(f"    Composite Score: {memory.get('composite_score', 0):.3f}")
        print(f"    Relevance: {memory.get('similarity_score', 0):.3f}")
        print(f"    Recency: {memory.get('recency_score', 0):.3f}")
        print(f"    Importance: {memory.get('importance_score', 0):.3f}")
        print(f"    ")

    # ==========================================
    # STEP 5: Add Agent Memory (Quick Save)
    # ==========================================
    print("\n🤖 Agent learning new information (quick save)...")

    # Agent discovers something new during the conversation
    agent_response = await sdk.add_agent_memory(
        AgentMemoryRequest(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            messages="Customer Alice explicitly requested subscription renewal reminders 7 days before expiry. High priority preference.",
        )
    )

    print(f"✅ Agent memory task accepted: {agent_response['task_id']}")
    print("   Processing in background...")
    await asyncio.sleep(5)  # Reasonable delay

    # ==========================================
    # STEP 6: Use Memory Batcher (Streaming Chat)
    # ==========================================
    print("\n📨 Using Memory Batcher for streaming conversation...")

    # Simulate a streaming chat where messages arrive one by one
    # We add >10 messages to demonstrate flushing (default batch size is 10)
    stream_messages = [
        ("user", "I just checked, the premium features are now active!"),
        ("assistant", "Wonderful! I'm glad the upgrade came through."),
        ("user", "Yes, and I love the dark mode being persistent now"),
        ("assistant", "Great! Is there anything else I can help with?"),
        ("user", "I also want to know when my subscription expires"),
        ("assistant", "Your subscription is valid until December 2025."),
        ("user", "That's great news, thank you."),
        ("assistant", "You're welcome! Do you have any other questions?"),
        ("user", "No, that's all for now."),
        ("assistant", "Have a wonderful day!"),
        ("user", "Goodbye!"),
    ]

    for role, content in stream_messages:
        batcher_response = await sdk.memory_batcher_add_message(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            role=role,
            content=content,
        )
        print(
            f"  Buffered: {role[:4]}... ({batcher_response.get('pending_messages', 0)}/{batcher_response.get('batch_size', 10)})"
        )
        if batcher_response.get('status') == 'flushed':
            print("  🌊 Batch FLUSHED to storage!")

    # add little delay for memory stored
    await asyncio.sleep(5)
    
    # ==========================================
    # STEP 7: Summarize Conversation (Context Window Management)
    # ==========================================
    print("\n📝 Summarizing conversation for context window management...")

    # Get all messages so far
    all_messages = conversation_messages + [
        Message(role=role, content=content) for role, content in stream_messages
    ]

    # Sync mode (immediate return)
    summary_response = await sdk.summarize_conversation(
        ConversationSummaryRequest(
            app_id=app_id,
            user_id=user_id,
            session_id=session_id,
            messages=[
                {"role": m.role, "content": m.content} for m in all_messages[:5]
            ],  # First 5 for demo
        )
    )

    print(f"✅ Summary generated:")
    print(f"   {summary_response.get('summary', 'N/A')[:200]}...")
    print(f"   Delivery: {summary_response.get('delivery', 'N/A')}")

    results = await sdk.query_memory(
        app_id=app_id,
        user_id=user_id,
        session_id=session_id,
        query="What does Alice prefer and what issues did she have?",
        n_results=5,
        similarity_threshold=0.5,
    )

    print(f"\nFound {len(results)} relevant memories (post-batch):")
    for i, memory in enumerate(results, 1):
        # print(f"{memory}")
        print(f"\n  Memory {i}:")
        print(f"    Content: {memory['memory_note'][:100]}...")
        print(f"    rank: {memory.get('rank', 'N/A')}")
        print(f"    Composite Score: {memory.get('composite_score', 0):.3f}")
        print(f"    Relevance: {memory.get('similarity_score', 0):.3f}")
        print(f"    Recency: {memory.get('recency_score', 0):.3f}")
        print(f"    Importance: {memory.get('importance_score', 0):.3f}")
        print(f"    ")

    # ==========================================
    # STEP 8: Agent Memory SDK - Answer with Context
    # ==========================================
    print("\n🎯 Using AgentMemorySDK to answer query with memory context...")

    agent_response = await agent_sdk.answer_query(
        app_id=app_id,
        query="What does Alice prefer and what issues did she have?",
        user_id=user_id,
        session_id=session_id,
        n_results=3,
        similarity_threshold=0.5,
    )

    print(f"✅ Agent Answer:")
    print(f"   {agent_response['answer'][:300]}...")
    print(f"   Based on {len(agent_response['memories'])} memories")

    # ==========================================
    # STEP 9: Get Specific Memory by ID
    # ==========================================
    if results:
        print("\n📄 Retrieving specific memory by ID...")
        memory_id = results[0]["metadata"].get("document_id")

        if memory_id:
            specific_memory = await sdk.get_memory(memory_id=memory_id, app_id=app_id)

            if specific_memory:
                print(f"✅ Retrieved memory:")
                print(f"   ID: {memory_id}")
                print(f"   Content: {specific_memory.get('document', '')[:150]}...")

    # ==========================================
    # STEP 10: Traverse Memory Evolution Chain
    # ==========================================
    print("\n🔗 Checking memory evolution chain...")

    if results:
        memory_id = results[0]["metadata"].get("document_id")
        if memory_id:
            evolution_chain = await sdk.traverse_memory_evolution_chain(
                app_id=app_id, memory_id=memory_id
            )

            print(f"✅ Evolution chain has {len(evolution_chain)} memories")

            if len(evolution_chain) > 1:
                # Generate evolution graph
                graph = sdk.generate_evolution_graph(evolution_chain, format="mermaid")
                print(f"\n📊 Evolution Graph (Mermaid):")
                print(graph[:200] + "...")

                # Generate evolution report
                report = sdk.generate_evolution_report(
                    evolution_chain, format="markdown"
                )
                print(f"\n📋 Evolution Report:")
                print(report[:300] + "...")

    print("\n💡 Real-World Use Case: Customer Support Agent")
    print("   - Remembers customer preferences")
    print("   - Tracks conversation history")
    print("   - Handles context window limits")
    print("   - Provides GDPR compliance")
    print("   - Monitors system health")


if __name__ == "__main__":
    asyncio.run(main())
