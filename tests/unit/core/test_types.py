"""
Unit tests for OmniMemory type definitions.
"""

from omnimemory.core.types import MemoryNoteData, MemoryDataDict


class TestTypeDefinitions:
    """Test cases for type definitions."""

    def test_memory_note_data_importable(self):
        """Test MemoryNoteData is importable."""
        assert MemoryNoteData is not None

    def test_memory_data_dict_importable(self):
        """Test MemoryDataDict is importable."""
        assert MemoryDataDict is not None

    def test_memory_note_data_has_expected_fields(self):
        """Test MemoryNoteData has expected fields."""
        annotations = MemoryNoteData.__annotations__
        assert "natural_memory_note" in annotations
        assert "retrieval_tags" in annotations
        assert "retrieval_keywords" in annotations
        assert "semantic_queries" in annotations
        assert "conversation_complexity" in annotations
        assert "interaction_quality" in annotations
        assert "follow_up_potential" in annotations

    def test_memory_data_dict_has_expected_fields(self):
        """Test MemoryDataDict has expected fields."""
        annotations = MemoryDataDict.__annotations__
        assert "doc_id" in annotations
        assert "app_id" in annotations
        assert "user_id" in annotations
        assert "session_id" in annotations
        assert "embedding" in annotations
        assert "natural_memory_note" in annotations

    def test_memory_note_data_total_false(self):
        """Test MemoryNoteData has total=False (all fields optional)."""
        data: MemoryNoteData = {}
        assert isinstance(data, dict)

    def test_memory_data_dict_total_false(self):
        """Test MemoryDataDict has total=False (all fields optional)."""
        data: MemoryDataDict = {}
        assert isinstance(data, dict)
