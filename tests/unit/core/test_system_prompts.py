"""
Unit tests for OmniMemory system prompts.
"""

import pytest
from omnimemory.core.system_prompts import (
    episodic_memory_constructor_system_prompt,
    summarizer_memory_constructor_system_prompt,
    fast_conversation_summary_prompt,
    conflict_resolution_agent_prompt,
    synthesis_agent_prompt,
)


class TestSystemPrompts:
    """Test cases for system prompts."""

    def test_episodic_memory_constructor_prompt_importable(self):
        """Test episodic_memory_constructor_system_prompt is importable."""
        assert episodic_memory_constructor_system_prompt is not None
        assert isinstance(episodic_memory_constructor_system_prompt, str)
        assert len(episodic_memory_constructor_system_prompt) > 0

    def test_summarizer_memory_constructor_prompt_importable(self):
        """Test summarizer_memory_constructor_system_prompt is importable."""
        assert summarizer_memory_constructor_system_prompt is not None
        assert isinstance(summarizer_memory_constructor_system_prompt, str)
        assert len(summarizer_memory_constructor_system_prompt) > 0

    def test_fast_conversation_summary_prompt_importable(self):
        """Test fast_conversation_summary_prompt is importable."""
        assert fast_conversation_summary_prompt is not None
        assert isinstance(fast_conversation_summary_prompt, str)
        assert len(fast_conversation_summary_prompt) > 0

    def test_conflict_resolution_agent_prompt_importable(self):
        """Test conflict_resolution_agent_prompt is importable."""
        assert conflict_resolution_agent_prompt is not None
        assert isinstance(conflict_resolution_agent_prompt, str)
        assert len(conflict_resolution_agent_prompt) > 0

    def test_synthesis_agent_prompt_importable(self):
        """Test synthesis_agent_prompt is importable."""
        assert synthesis_agent_prompt is not None
        assert isinstance(synthesis_agent_prompt, str)
        assert len(synthesis_agent_prompt) > 0

    def test_episodic_prompt_contains_keywords(self):
        """Test episodic prompt contains expected keywords."""
        assert (
            "Episodic Memory Constructor" in episodic_memory_constructor_system_prompt
        )
        assert "behavioral" in episodic_memory_constructor_system_prompt.lower()

    def test_summarizer_prompt_contains_keywords(self):
        """Test summarizer prompt contains expected keywords."""
        assert (
            "Summary Memory Constructor" in summarizer_memory_constructor_system_prompt
        )
        assert "narrative" in summarizer_memory_constructor_system_prompt.lower()

    def test_conflict_resolution_prompt_contains_keywords(self):
        """Test conflict resolution prompt contains expected keywords."""
        assert "Conflict Resolution Agent" in conflict_resolution_agent_prompt
        assert "UPDATE" in conflict_resolution_agent_prompt
        assert "DELETE" in conflict_resolution_agent_prompt
        assert "SKIP" in conflict_resolution_agent_prompt

    def test_synthesis_prompt_contains_keywords(self):
        """Test synthesis prompt contains expected keywords."""
        assert "Memory Synthesis Agent" in synthesis_agent_prompt
        assert "consolidate" in synthesis_agent_prompt.lower()

    def test_fast_summary_prompt_contains_keywords(self):
        """Test fast summary prompt contains expected keywords."""
        assert "conversation summarizer" in fast_conversation_summary_prompt.lower()
        assert "summary" in fast_conversation_summary_prompt.lower()
