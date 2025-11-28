"""
Comprehensive unit tests for OmniMemory utility functions.
"""

import pytest
import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from omnimemory.core.utils import (
    get_tokenizer_for_model,
    count_tokens,
    chunk_text_by_tokens,
    estimate_chunking_stats,
    normalize_token,
    clean_and_parse_json,
    fuzzy_dedup,
    format_conversation,
    parse_iso_to_datetime,
    parse_timestamp,
    calculate_recency_score,
    calculate_importance_score,
    calculate_composite_score,
    determine_relationship_type,
    get_embedding_cache_key,
    get_cached_embedding,
    cache_embedding,
    _prune_embedding_cache,
    _get_importance_weights,
    create_zettelkasten_memory_note,
    prepare_memory_for_storage,
    create_agent_memory_note,
    prepare_agent_memory_for_storage,
)


class TestTokenizerFunctions:
    """Test cases for tokenizer-related functions."""

    def test_get_tokenizer_for_model_valid_model(self):
        """Test get tokenizer for valid model."""
        tokenizer = get_tokenizer_for_model("gpt-4")
        assert tokenizer is not None

    def test_get_tokenizer_for_model_invalid_model_fallback(self):
        """Test fallback to gpt-4 for invalid model."""
        with patch(
            "omnimemory.core.utils.tiktoken.encoding_for_model"
        ) as mock_encoding:
            mock_encoding.side_effect = [KeyError("Model not found"), Mock()]
            tokenizer = get_tokenizer_for_model("invalid-model")
            assert tokenizer is not None
            assert mock_encoding.call_count == 2

    def test_get_tokenizer_for_model_all_fail_return_none(self):
        """Test return None when all attempts fail."""
        with patch(
            "omnimemory.core.utils.tiktoken.encoding_for_model",
            side_effect=KeyError("Error"),
        ):
            tokenizer = get_tokenizer_for_model("invalid-model")
            assert tokenizer is None

    def test_count_tokens_with_valid_text(self):
        """Test count tokens with valid text."""
        result = count_tokens("Hello world", "gpt-4")
        assert result > 0

    def test_count_tokens_empty_string(self):
        """Test count tokens for empty string."""
        result = count_tokens("", "gpt-4")
        assert result == 0

    def test_count_tokens_fallback_estimation(self):
        """Test fallback to character-based estimation."""
        with patch("omnimemory.core.utils.get_tokenizer_for_model", return_value=None):
            result = count_tokens("Hello world", "gpt-4")
            assert result == len("Hello world") // 4

    def test_count_tokens_handle_exception(self):
        """Test handle exceptions in tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Error")

        with patch(
            "omnimemory.core.utils.get_tokenizer_for_model", return_value=mock_tokenizer
        ):
            result = count_tokens("Hello world", "gpt-4")
            assert result == len("Hello world") // 4

    def test_chunk_text_by_tokens_basic(self):
        """Test chunk text by tokens."""
        text = "This is a test sentence. " * 100
        chunks = chunk_text_by_tokens(
            text, chunk_size=50, overlap=10, model_name="gpt-4"
        )
        assert len(chunks) > 0
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_text_by_tokens_with_overlap(self):
        """Test chunk text with overlap."""
        text = "Word " * 200
        chunks = chunk_text_by_tokens(
            text, chunk_size=50, overlap=10, model_name="gpt-4"
        )
        assert len(chunks) > 1

    def test_chunk_text_by_tokens_single_chunk(self):
        """Test single chunk when text is small."""
        text = "Short text"
        chunks = chunk_text_by_tokens(
            text, chunk_size=100, overlap=10, model_name="gpt-4"
        )
        assert len(chunks) == 1

    def test_chunk_text_by_tokens_empty_text(self):
        """Test handle empty text."""
        chunks = chunk_text_by_tokens("", chunk_size=50, overlap=10, model_name="gpt-4")
        assert len(chunks) == 0

    def test_estimate_chunking_stats(self):
        """Test estimate chunking statistics."""
        text = "This is a test. " * 100
        stats = estimate_chunking_stats(
            text, chunk_size=50, overlap=10, model_name="gpt-4"
        )
        assert "chunks_needed" in stats
        assert "total_tokens" in stats
        assert stats["chunks_needed"] > 0

    def test_estimate_chunking_stats_empty_text(self):
        """Test estimate stats for empty text."""
        stats = estimate_chunking_stats(
            "", chunk_size=50, overlap=10, model_name="gpt-4"
        )
        assert stats["chunks_needed"] == 1
        assert stats["total_tokens"] == 0


class TestNormalizeToken:
    """Test cases for normalize_token function."""

    def test_normalize_token_lowercase(self):
        """Test convert to lowercase."""
        result = normalize_token("HELLO")
        assert result == "hello"

    def test_normalize_token_remove_punctuation(self):
        """Test remove punctuation."""
        result = normalize_token("hello!")
        assert result == "hello"

    def test_normalize_token_remove_whitespace(self):
        """Test remove whitespace."""
        result = normalize_token("  hello  ")
        assert result == "hello"

    def test_normalize_token_handle_empty(self):
        """Test handle empty string."""
        result = normalize_token("")
        assert result == ""

    def test_normalize_token_handle_special_chars(self):
        """Test handle special characters."""
        result = normalize_token("hello@world#123")
        assert result == "helloworld123"


class TestCleanAndParseJSON:
    """Test cases for clean_and_parse_json function."""

    def test_parse_valid_json(self):
        """Test parse valid JSON."""
        json_str = '{"key": "value"}'
        result = clean_and_parse_json(json_str)
        assert result == {"key": "value"}

    def test_parse_valid_json_array(self):
        """Test parse valid JSON array."""
        json_str = '[{"key": "value"}]'
        result = clean_and_parse_json(json_str)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_parse_json_with_trailing_commas(self):
        """Test parse JSON with trailing commas."""
        json_str = '{"key": "value",}'
        result = clean_and_parse_json(json_str)
        assert result == {"key": "value"}

    def test_parse_json_with_comments(self):
        """Test parse JSON with comments."""
        json_str = '{"key": "value"} // comment'
        result = clean_and_parse_json(json_str)
        assert result == {"key": "value"}

    def test_parse_json_with_multiline_comments(self):
        """Test parse JSON with multiline comments."""
        json_str = '{"key": /* comment */ "value"}'
        result = clean_and_parse_json(json_str)
        assert result == {"key": "value"}

    def test_parse_json_in_markdown_code_block(self):
        """Test parse JSON in markdown code block."""
        json_str = '```json\n{"key": "value"}\n```'
        result = clean_and_parse_json(json_str)
        assert result == {"key": "value"}

    def test_parse_json_in_code_block_no_lang(self):
        """Test parse JSON in code block without language."""
        json_str = '```\n{"key": "value"}\n```'
        result = clean_and_parse_json(json_str)
        assert result == {"key": "value"}

    def test_raise_value_error_on_empty_string(self):
        """Test raise ValueError on empty string."""
        with pytest.raises(ValueError, match="Empty JSON string"):
            clean_and_parse_json("")

    def test_raise_json_decode_error_on_invalid_json(self):
        """Test raise JSONDecodeError on invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            clean_and_parse_json("invalid json {")

    def test_handle_trailing_commas_in_array(self):
        """Test handle trailing commas in array."""
        json_str = '[{"key": "value"},]'
        result = clean_and_parse_json(json_str)
        assert isinstance(result, list)

    def test_handle_commas_inside_strings(self):
        """Test preserve commas inside strings."""
        json_str = '{"key": "value, with comma"}'
        result = clean_and_parse_json(json_str)
        assert result["key"] == "value, with comma"

    def test_handle_escaped_quotes(self):
        """Test handle escaped quotes."""
        json_str = '{"key": "value with \\"quotes\\""}'
        result = clean_and_parse_json(json_str)
        assert "quotes" in result["key"]


class TestFuzzyDedup:
    """Test cases for fuzzy_dedup function."""

    def test_dedup_exact_duplicates(self):
        """Test deduplicate exact duplicates."""
        items = ["hello", "world", "hello", "world"]
        result = fuzzy_dedup(items)
        assert len(result) == 2
        assert "hello" in result
        assert "world" in result

    def test_dedup_similar_strings(self):
        """Test deduplicate similar strings."""
        items = ["hello world", "hello  world", "hello world!"]
        result = fuzzy_dedup(items, threshold=75)
        assert len(result) <= len(items)

    def test_dedup_with_custom_threshold(self):
        """Test deduplicate with custom threshold."""
        items = ["hello", "helo", "world"]
        result = fuzzy_dedup(items, threshold=80)
        assert len(result) >= 1

    def test_dedup_empty_list(self):
        """Test handle empty list."""
        result = fuzzy_dedup([])
        assert result == []

    def test_dedup_single_item(self):
        """Test handle single item."""
        result = fuzzy_dedup(["hello"])
        assert result == ["hello"]

    def test_dedup_normalize_tokens(self):
        """Test normalize tokens before deduplication."""
        items = ["Hello World!", "hello world", "HELLO WORLD"]
        result = fuzzy_dedup(items)
        assert len(result) <= len(items)


class TestFormatConversation:
    """Test cases for format_conversation function."""

    def test_format_message_dicts(self):
        """Test format list of message dicts."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = format_conversation(messages)
        assert "Hello" in result
        assert "Hi there" in result

    def test_format_message_objects(self):
        """Test format list of message objects."""

        class Message:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [Message("user", "Hello"), Message("assistant", "Hi")]
        result = format_conversation(messages)
        assert "Hello" in result

    def test_format_empty_list(self):
        """Test handle empty list."""
        result = format_conversation([])
        assert result == ""

    def test_format_with_timestamps(self):
        """Test format messages with timestamps."""
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00Z"}
        ]
        result = format_conversation(messages)
        assert "Hello" in result


class TestTimestampParsing:
    """Test cases for timestamp parsing functions."""

    def test_parse_iso_to_datetime_valid(self):
        """Test parse valid ISO timestamp."""
        from omnimemory.core.utils import parse_iso_to_datetime

        result = parse_iso_to_datetime("2024-01-01T00:00:00Z")
        assert isinstance(result, datetime)

    def test_parse_iso_to_datetime_invalid(self):
        """Test return None for invalid timestamp."""
        from omnimemory.core.utils import parse_iso_to_datetime

        result = parse_iso_to_datetime("invalid")
        assert result is None

    def test_parse_timestamp_valid(self):
        """Test parse valid timestamp."""
        result = parse_timestamp("2024-01-01T00:00:00Z")
        assert isinstance(result, datetime)

    def test_parse_timestamp_with_timezone(self):
        """Test parse timestamp with timezone."""
        result = parse_timestamp("2024-01-01T00:00:00+00:00")
        assert isinstance(result, datetime)

    def test_parse_timestamp_raise_on_invalid(self):
        """Test raise ValueError on invalid timestamp."""
        with pytest.raises(ValueError):
            parse_timestamp("invalid")


class TestScoringFunctions:
    """Test cases for scoring functions."""

    def test_calculate_recency_score_recent(self):
        """Test calculate recency score for recent memory."""
        now = datetime.now(timezone.utc)
        created = now.isoformat()
        updated = now.isoformat()
        score = calculate_recency_score(created, updated)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_calculate_recency_score_old(self):
        """Test calculate recency score for old memory."""
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        created = old_time.isoformat()
        updated = old_time.isoformat()
        score = calculate_recency_score(created, updated)
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_calculate_recency_score_future_time(self):
        """Test handle future timestamps (return 1.0)."""
        future = datetime.now(timezone.utc).replace(year=2030)
        created = future.isoformat()
        updated = future.isoformat()
        score = calculate_recency_score(created, updated)
        assert score == 1.0

    def test_calculate_recency_score_handle_exception(self):
        """Test handle exceptions (return 0.1)."""
        score = calculate_recency_score("invalid", "invalid")
        assert score == 0.1

    def test_calculate_importance_score_high_quality(self):
        """Test calculate importance with high quality."""
        metadata = {"interaction_quality": "high"}
        score = calculate_importance_score(metadata)
        assert 0.0 <= score <= 1.0

    def test_calculate_importance_score_with_followups(self):
        """Test calculate importance with follow-ups."""
        metadata = {
            "interaction_quality": "medium",
            "follow_up_potential": ["action1", "action2", "action3"],
        }
        score = calculate_importance_score(metadata)
        assert 0.0 <= score <= 1.0

    def test_calculate_importance_score_with_tags(self):
        """Test calculate importance with tags and keywords."""
        metadata = {
            "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
            "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
        }
        score = calculate_importance_score(metadata)
        assert 0.0 <= score <= 1.0

    def test_calculate_importance_score_empty_metadata(self):
        """Test calculate importance with empty metadata."""
        score = calculate_importance_score({})
        assert 0.0 <= score <= 1.0

    def test_calculate_composite_score_basic(self):
        """Test calculate composite score."""
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        score = calculate_composite_score(0.7, metadata)
        assert 0.0 <= score <= 1.0

    def test_calculate_composite_score_low_relevance(self):
        """Test composite score with low relevance."""
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        score = calculate_composite_score(0.3, metadata)
        assert score < 0.5

    def test_calculate_composite_score_high_relevance(self):
        """Test composite score with high relevance."""
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "interaction_quality": "high",
        }
        score = calculate_composite_score(0.9, metadata)
        assert score > 0.5

    def test_determine_relationship_type_high_score(self):
        """Test determine relationship type for high score."""
        rel_type = determine_relationship_type(0.9, {})
        assert rel_type in [
            "highly_similar",
            "very_similar",
            "related_technical",
            "related_general",
            "loosely_related",
            "tangentially_related",
        ]

    def test_determine_relationship_type_low_score(self):
        """Test determine relationship type for low score."""
        rel_type = determine_relationship_type(0.2, {})
        assert rel_type in [
            "highly_similar",
            "very_similar",
            "related_technical",
            "related_general",
            "loosely_related",
            "tangentially_related",
        ]


class TestEmbeddingCache:
    """Test cases for embedding cache functions."""

    def test_get_embedding_cache_key(self):
        """Test generate cache key."""
        key1 = get_embedding_cache_key("test text")
        key2 = get_embedding_cache_key("test text")
        assert key1 == key2

    def test_get_embedding_cache_key_different_texts(self):
        """Test different texts generate different keys."""
        key1 = get_embedding_cache_key("text1")
        key2 = get_embedding_cache_key("text2")
        assert key1 != key2

    def test_get_cached_embedding_miss(self):
        """Test get cached embedding when not cached."""
        result = get_cached_embedding("new text")
        assert result is None

    def test_get_cached_embedding_hit(self):
        """Test get cached embedding when cached."""
        embedding = [0.1, 0.2, 0.3]
        cache_embedding("test text", embedding)
        result = get_cached_embedding("test text")
        assert result == embedding

    def test_cache_embedding_store(self):
        """Test store embedding in cache."""
        embedding = [0.1, 0.2, 0.3]
        cache_embedding("test text", embedding)
        result = get_cached_embedding("test text")
        assert result == embedding

    def test_cache_embedding_expire_old_entries(self):
        """Test expire old cache entries."""
        embeddings = [[0.1] * 100] * 600
        for i, emb in enumerate(embeddings):
            cache_embedding(f"text{i}", emb)

        result = get_cached_embedding("text0")
        assert result is None or result == embeddings[0]


class TestZettelkasten:
    """Test cases for create_zettelkasten_memory_note function."""

    def test_create_zettelkasten_note_basic(self):
        """Test create Zettelkasten note."""
        note = create_zettelkasten_memory_note(
            episodic_data={"context": {"user_intent": "test"}},
            summary_data={"narrative": "Test content"},
        )
        assert "Test content" in note or "test" in note

    def test_create_zettelkasten_note_with_tags(self):
        """Test create note with tags."""
        note = create_zettelkasten_memory_note(
            episodic_data={}, summary_data={"retrieval": {"tags": ["tag1", "tag2"]}}
        )
        assert "tag1" in note or "tag2" in note

    def test_create_zettelkasten_note_with_timestamp(self):
        """Test create note with timestamp."""
        note = create_zettelkasten_memory_note(
            episodic_data={}, summary_data={"narrative": "Test"}
        )
        assert isinstance(note, str)

    def test_create_zettelkasten_note_empty_content(self):
        """Test handle empty content."""
        note = create_zettelkasten_memory_note(episodic_data={}, summary_data={})
        assert isinstance(note, str)


class TestPrepareMemoryForStorage:
    """Test cases for prepare_memory_for_storage function."""

    def test_prepare_memory_basic(self):
        """Test prepare memory for storage."""
        result = prepare_memory_for_storage(
            note_text="Test document",
            summary_data={"retrieval": {"tags": []}},
            episodic_data={},
        )
        assert "text" in result or "metadata" in result

    def test_prepare_memory_with_timestamps(self):
        """Test prepare memory with timestamps."""
        result = prepare_memory_for_storage(
            note_text="Test",
            summary_data={},
            episodic_data={},
            timestamp="2024-01-01T00:00:00Z",
        )
        assert isinstance(result, dict)

    def test_prepare_memory_with_metadata(self):
        """Test prepare memory with metadata."""
        result = prepare_memory_for_storage(
            note_text="Test",
            summary_data={"retrieval": {"tags": ["tag1"], "keywords": ["kw1"]}},
            episodic_data={},
        )
        assert isinstance(result, dict)

    def test_prepare_memory_handle_missing_fields(self):
        """Test handle missing fields."""
        result = prepare_memory_for_storage(
            note_text="", summary_data={}, episodic_data={}
        )
        assert isinstance(result, dict)

    def test_normalize_importance_weights_total_not_one_coverage_line_44(self):
        """Test _get_importance_weights when total != 1.0 (line 44)."""
        with patch("omnimemory.core.utils._IMPORTANCE_RICHNESS_WEIGHT", 0.6):
            with patch("omnimemory.core.utils._IMPORTANCE_QUALITY_WEIGHT", 0.4):
                with patch("omnimemory.core.utils._IMPORTANCE_FOLLOWUP_WEIGHT", 0.3):
                    weights = _get_importance_weights()
                    total = sum(weights.values())
                    assert abs(total - 1.0) < 0.0001
                    assert weights["quality"] == pytest.approx(0.4 / 1.3, rel=1e-5)

    def test_chunk_text_no_tokenizer_coverage_line_113(self):
        """Test chunk_text_by_tokens when no tokenizer available (line 113)."""
        with patch("omnimemory.core.utils.get_tokenizer_for_model", return_value=None):
            text = "a" * 200
            chunks = chunk_text_by_tokens(text, chunk_size=50, overlap=10)
            assert len(chunks) > 0

    def test_chunk_text_exception_fallback_coverage_line_145(self):
        """Test chunk_text_by_tokens exception fallback (line 145)."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Error")
        with patch(
            "omnimemory.core.utils.get_tokenizer_for_model", return_value=mock_tokenizer
        ):
            text = "test text" * 10
            chunks = chunk_text_by_tokens(text, chunk_size=5, overlap=1)
            assert len(chunks) > 0

    def test_chunk_text_by_tokens_break_conditions_coverage_line_141(self):
        """Test chunk_text_by_tokens break conditions (line 141)."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "chunk"
        with patch(
            "omnimemory.core.utils.get_tokenizer_for_model", return_value=mock_tokenizer
        ):
            text = "test"
            chunks = chunk_text_by_tokens(text, chunk_size=2, overlap=3)
            assert len(chunks) > 0

    def test_clean_and_parse_json_escape_handling_coverage_line_259(self):
        """Test clean_and_parse_json escape character handling (line 259)."""
        json_str = '{"key": "value with \\"quotes\\""}'
        result = clean_and_parse_json(json_str)
        assert result == {"key": 'value with "quotes"'}

    def test_clean_and_parse_json_escape_next_coverage_line_261(self):
        """Test clean_and_parse_json escape_next flag (line 261)."""
        json_str = '{"key": "value\\\\with\\\\backslashes"}'
        result = clean_and_parse_json(json_str)
        assert "key" in result

    def test_clean_and_parse_json_backslash_coverage_line_265(self):
        """Test clean_and_parse_json backslash handling (line 265)."""
        json_str = '{"key": "value\\nnewline"}'
        result = clean_and_parse_json(json_str)
        assert "key" in result

    def test_clean_and_parse_json_escape_next_continue_coverage_line_268(self):
        """Test clean_and_parse_json escape_next continue (line 268)."""
        json_str = '{"key": "value\\"escaped"}'
        result = clean_and_parse_json(json_str)
        assert "key" in result

    def test_clean_and_parse_json_escape_next_coverage_line_265(self):
        """Test clean_and_parse_json escape_next flag (line 265)."""
        json_str = '{"key": "value\\\\with\\\\backslashes"}'
        result = clean_and_parse_json(json_str)
        assert result == {"key": "value\\with\\backslashes"}

    def test_clean_and_parse_json_json_decode_error_coverage_line_317(self):
        """Test clean_and_parse_json JSONDecodeError handling (line 317)."""
        json_str = 'Some text {"invalid": json} more text'
        with pytest.raises(ValueError):
            clean_and_parse_json(json_str)

    def test_format_conversation_with_metadata_coverage_line_380(self):
        """Test format_conversation ignores metadata (metadata no longer included)."""
        messages = [
            {
                "role": "user",
                "content": "test",
                "metadata": {"key": "value"},
            }
        ]
        result = format_conversation(messages)
        assert "user: test" in result
        assert "metadata" not in result

    def test_format_conversation_else_continue_coverage_line_380(self):
        """Test format_conversation else continue path (line 380)."""
        messages = [123, "string", {"role": "user", "content": "valid"}]
        result = format_conversation(messages)
        assert "valid" in result

    def test_parse_iso_to_datetime_empty_string_coverage_line_389(self):
        """Test parse_iso_to_datetime with empty string (line 389)."""
        result = parse_iso_to_datetime("")
        assert result is None

    def test_parse_timestamp_tzinfo_none_coverage_line_395(self):
        """Test parse_timestamp when tzinfo is None (line 395)."""
        result = parse_timestamp("2024-01-01T00:00:00")
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_parse_timestamp_astimezone_coverage_line_415(self):
        """Test parse_timestamp astimezone conversion (line 415)."""
        result = parse_timestamp("2024-01-01T00:00:00+05:00")
        assert result.tzinfo == timezone.utc

    def test_calculate_importance_score_quality_low_coverage_line_486(self):
        """Test calculate_importance_score with low quality (line 486)."""
        metadata = {"interaction_quality": "low"}
        score = calculate_importance_score(metadata)
        assert isinstance(score, float)

    def test_calculate_importance_score_followup_not_list_coverage_line_501(self):
        """Test calculate_importance_score with followup not a list (line 501)."""
        metadata = {"follow_up_potential": "not a list"}
        score = calculate_importance_score(metadata)
        assert isinstance(score, float)

    def test_calculate_importance_score_followup_scores_coverage_line_506(self):
        """Test calculate_importance_score followup score thresholds (line 506)."""
        metadata = {"follow_up_potential": ["item1", "item2"]}
        score = calculate_importance_score(metadata)
        assert isinstance(score, float)

    def test_calculate_importance_score_followup_one_item_coverage_line_508(self):
        """Test calculate_importance_score with one followup item (line 508)."""
        metadata = {"follow_up_potential": ["item1"]}
        score = calculate_importance_score(metadata)
        assert isinstance(score, float)

    def test_calculate_importance_score_followup_count_three_coverage_line_521(self):
        """Test calculate_importance_score with 3+ followup items (line 521)."""
        metadata = {"follow_up_potential": ["item1", "item2", "item3"]}
        score = calculate_importance_score(metadata)
        assert isinstance(score, float)

    def test_calculate_importance_score_followup_count_two_coverage_line_524(self):
        """Test calculate_importance_score with 2 followup items (line 524)."""
        metadata = {"follow_up_potential": ["item1", "item2"]}
        score = calculate_importance_score(metadata)
        assert isinstance(score, float)

    def test_determine_relationship_type_very_similar_coverage_line_597(self):
        """Test determine_relationship_type with very_similar threshold (line 597)."""
        result = determine_relationship_type(0.85, {})
        assert result == "very_similar"

    def test_determine_relationship_type_related_technical_coverage_line_601(self):
        """Test determine_relationship_type with related_technical (line 601)."""
        metadata = {"conversation_complexity": 5}
        result = determine_relationship_type(0.75, metadata)
        assert result == "related_technical"

    def test_determine_relationship_type_complexity_fallback_coverage_line_562(self):
        """Test determine_relationship_type with complexity default (line 562)."""
        metadata = {}
        result = determine_relationship_type(0.75, metadata)
        assert result == "related_general"


class TestAgentMemoryUtilities:
    """Test cases for agent memory utility functions."""

    def test_create_agent_memory_note_with_complete_data(self):
        """Test create_agent_memory_note with complete agent data."""

        agent_data = {
            "narrative": "User prefers dark mode for better visibility",
            "retrieval": {
                "tags": ["preferences", "ui", "display"],
                "keywords": ["dark mode", "theme", "interface"],
            },
            "metadata": {
                "depth": "low",
                "follow_ups": ["Check other UI preferences"],
            },
        }

        result = create_agent_memory_note(agent_data)

        assert "## Note" in result
        assert "User prefers dark mode" in result
        assert "Tags:" in result
        assert "preferences" in result
        assert "dark mode" in result
        assert "depth: low" in result.lower()

    def test_create_agent_memory_note_with_minimal_data(self):
        """Test create_agent_memory_note with only narrative."""

        agent_data = {"narrative": "Simple user note"}

        result = create_agent_memory_note(agent_data)

        assert "## Note" in result
        assert "Simple user note" in result
        assert isinstance(result, str)

    def test_create_agent_memory_note_filters_na_values(self):
        """Test create_agent_memory_note filters out N/A values."""

        agent_data = {
            "narrative": "User feedback",
            "retrieval": {
                "tags": ["valid", "N/A", "tag"],
                "keywords": ["word", "N/A"],
            },
            "metadata": {
                "depth": "medium",
                "follow_ups": ["N/A"],
            },
        }

        result = create_agent_memory_note(agent_data)

        assert "N/A" not in result
        assert "valid" in result
        assert "tag" in result
        assert "word" in result

    def test_create_agent_memory_note_empty_metadata(self):
        """Test create_agent_memory_note with empty metadata."""

        agent_data = {
            "narrative": "Test note",
            "retrieval": {},
            "metadata": {},
        }

        result = create_agent_memory_note(agent_data)

        assert "## Note" in result
        assert "Test note" in result

    def test_create_agent_memory_note_missing_narrative(self):
        """Test create_agent_memory_note with missing narrative."""

        agent_data = {
            "retrieval": {"tags": ["test"]},
            "metadata": {"depth": "low"},
        }

        result = create_agent_memory_note(agent_data)

        assert isinstance(result, str)
        # Should still include footer
        assert "Tags:" in result or "test" in result

    def test_create_agent_memory_note_formatting(self):
        """Test create_agent_memory_note output formatting."""

        agent_data = {
            "narrative": "Test content",
            "retrieval": {"tags": ["tag1"], "keywords": ["kw1"]},
            "metadata": {"depth": "high", "follow_ups": ["action"]},
        }

        result = create_agent_memory_note(agent_data)

        # Check no excessive whitespace
        assert "\n\n\n" not in result
        assert "  " not in result or result.count("  ") < 3

    def test_prepare_agent_memory_for_storage_complete_data(self):
        """Test prepare_agent_memory_for_storage with complete data."""

        note_text = "Formatted agent memory note"
        agent_data = {
            "retrieval": {
                "tags": ["tag1", "tag2"],
                "keywords": ["kw1", "kw2"],
                "queries": ["query1", "query2"],
            },
            "metadata": {
                "depth": "high",
                "follow_ups": ["action1", "action2"],
            },
        }

        result = prepare_agent_memory_for_storage(note_text, agent_data)

        assert result["text"] == note_text
        assert "metadata" in result
        assert result["metadata"]["tags"] == ["tag1", "tag2"]
        assert result["metadata"]["keywords"] == ["kw1", "kw2"]
        assert result["metadata"]["query_hooks"] == ["query1", "query2"]
        assert result["metadata"]["content_depth"] == "high"
        assert result["metadata"]["follow_up_areas"] == ["action1", "action2"]
        assert result["metadata"]["source"] == "agent_memory"
        assert "timestamp" in result["metadata"]

    def test_prepare_agent_memory_for_storage_filters_na(self):
        """Test prepare_agent_memory_for_storage filters N/A values."""

        note_text = "Test note"
        agent_data = {
            "retrieval": {
                "tags": ["valid", "N/A", "tag"],
                "keywords": ["word1", "N/A"],
                "queries": ["N/A", "query1"],
            },
            "metadata": {
                "depth": "medium",
                "follow_ups": ["N/A", "action1"],
            },
        }

        result = prepare_agent_memory_for_storage(note_text, agent_data)

        assert "N/A" not in result["metadata"]["tags"]
        assert "N/A" not in result["metadata"]["keywords"]
        assert "N/A" not in result["metadata"]["query_hooks"]
        assert "N/A" not in result["metadata"]["follow_up_areas"]
        assert result["metadata"]["tags"] == ["valid", "tag"]
        assert result["metadata"]["keywords"] == ["word1"]

    def test_prepare_agent_memory_for_storage_default_depth(self):
        """Test prepare_agent_memory_for_storage uses default depth."""

        note_text = "Test"
        agent_data = {"retrieval": {}, "metadata": {}}

        result = prepare_agent_memory_for_storage(note_text, agent_data)

        assert result["metadata"]["content_depth"] == "medium"

    def test_prepare_agent_memory_for_storage_custom_timestamp(self):
        """Test prepare_agent_memory_for_storage with custom timestamp."""

        note_text = "Test"
        agent_data = {"retrieval": {}, "metadata": {}}
        custom_time = "2024-01-01T00:00:00Z"

        result = prepare_agent_memory_for_storage(
            note_text, agent_data, timestamp=custom_time
        )

        assert result["metadata"]["timestamp"] == custom_time

    def test_prepare_agent_memory_for_storage_auto_timestamp(self):
        """Test prepare_agent_memory_for_storage generates timestamp if not provided."""

        note_text = "Test"
        agent_data = {"retrieval": {}, "metadata": {}}

        result = prepare_agent_memory_for_storage(note_text, agent_data)

        assert "timestamp" in result["metadata"]
        assert isinstance(result["metadata"]["timestamp"], str)
        # Should be valid ISO format
        assert "T" in result["metadata"]["timestamp"]

    def test_prepare_agent_memory_for_storage_empty_lists(self):
        """Test prepare_agent_memory_for_storage with empty arrays."""

        note_text = "Test"
        agent_data = {
            "retrieval": {
                "tags": [],
                "keywords": [],
                "queries": [],
            },
            "metadata": {
                "follow_ups": [],
            },
        }

        result = prepare_agent_memory_for_storage(note_text, agent_data)

        assert result["metadata"]["tags"] == []
        assert result["metadata"]["keywords"] == []
        assert result["metadata"]["query_hooks"] == []
        assert result["metadata"]["follow_up_areas"] == []

    def test_prepare_agent_memory_for_storage_missing_keys(self):
        """Test prepare_agent_memory_for_storage handles missing keys gracefully."""

        note_text = "Test"
        agent_data = {}

        result = prepare_agent_memory_for_storage(note_text, agent_data)

        assert result["text"] == note_text
        assert result["metadata"]["tags"] == []
        assert result["metadata"]["keywords"] == []
        assert result["metadata"]["query_hooks"] == []
        assert result["metadata"]["content_depth"] == "medium"
        assert result["metadata"]["source"] == "agent_memory"

    def test_determine_relationship_type_related_general_coverage_line_603(self):
        """Test determine_relationship_type with related_general (line 603)."""
        metadata = {"conversation_complexity": 2}
        result = determine_relationship_type(0.75, metadata)
        assert result == "related_general"

    def test_determine_relationship_type_loosely_related_coverage_line_605(self):
        """Test determine_relationship_type with loosely_related (line 605)."""
        result = determine_relationship_type(0.65, {})
        assert result == "loosely_related"

    def test_clean_expired_cache_entries_coverage_line_620(self):
        """Test _prune_embedding_cache removes expired entries (line 620)."""
        from omnimemory.core.utils import _EMBEDDING_CACHE

        _EMBEDDING_CACHE["expired"] = ([1.0, 2.0], time.time() - 1000)
        _EMBEDDING_CACHE["valid"] = ([1.0, 2.0], time.time() + 1000)
        now = time.time()
        _EMBEDDING_CACHE["expiring_now"] = ([1.0, 2.0], now)

        _prune_embedding_cache(now)

        assert "expired" not in _EMBEDDING_CACHE
        assert "expiring_now" not in _EMBEDDING_CACHE
        assert "valid" in _EMBEDDING_CACHE

    def test_calculate_recency_score_none_timestamps_coverage(self):
        """Test calculate_recency_score with None timestamps."""
        score = calculate_recency_score(None, None)
        assert score == 0.1

        score = calculate_recency_score(None, "2024-01-01T00:00:00Z")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        score = calculate_recency_score("2024-01-01T00:00:00Z", None)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_cache_embedding_prune_on_add_coverage_line_635(self):
        """Test cache_embedding prunes expired entries when adding (line 635)."""
        from omnimemory.core.utils import (
            cache_embedding,
            _EMBEDDING_CACHE,
            _EMBEDDING_CACHE_PREFIX,
        )

        _EMBEDDING_CACHE.clear()
        now = time.time()
        _EMBEDDING_CACHE["expired"] = ([1.0], now - 1000)

        cache_embedding("new_text", [1.0, 2.0])

        assert "expired" not in _EMBEDDING_CACHE
        cache_key = f"{_EMBEDDING_CACHE_PREFIX}{get_embedding_cache_key('new_text')}"
        assert cache_key in _EMBEDDING_CACHE

    def test_cache_embedding_max_entries_coverage_line_668(self):
        """Test cache_embedding max entries eviction (line 668)."""
        from omnimemory.core.utils import (
            cache_embedding,
            _EMBEDDING_CACHE,
            _EMBEDDING_CACHE_MAX_ENTRIES,
            _EMBEDDING_CACHE_PREFIX,
        )

        _EMBEDDING_CACHE.clear()
        now = time.time()
        for i in range(_EMBEDDING_CACHE_MAX_ENTRIES):
            _EMBEDDING_CACHE[f"key_{i}"] = ([1.0], now + 1000 + i)

        cache_embedding("new_text", [1.0, 2.0])

        assert len(_EMBEDDING_CACHE) == _EMBEDDING_CACHE_MAX_ENTRIES
        cache_key = f"{_EMBEDDING_CACHE_PREFIX}{get_embedding_cache_key('new_text')}"
        assert cache_key in _EMBEDDING_CACHE

    def test_cache_embedding_empty_embedding_coverage_line_643(self):
        """Test cache_embedding with empty embedding (line 643)."""
        from omnimemory.core.utils import cache_embedding, _EMBEDDING_CACHE

        _EMBEDDING_CACHE.clear()

        cache_embedding("test", [])
        assert "test" not in _EMBEDDING_CACHE

    def test_create_zettelkasten_memory_note_all_sections_coverage(self):
        """Test create_zettelkasten_memory_note with all sections populated."""
        episodic_data = {
            "behavioral_profile": {
                "communication": "direct",
                "learning": "visual",
                "problem_solving": "analytical",
                "decision_making": "data-driven",
            },
            "interaction_insights": {
                "engagement_triggers": "interactive content",
                "friction_points": "long forms",
                "optimal_approach": "step-by-step",
            },
            "what_worked": {
                "strategies": ["strategy1", "strategy2"],
                "pattern": "pattern1",
            },
            "what_failed": {"strategies": ["bad1", "bad2"], "pattern": "pattern2"},
            "future_guidance": {
                "recommended_approaches": ["rec1", "rec2"],
                "avoid_approaches": ["avoid1"],
                "adaptation_note": "key insight",
            },
        }

        summary_data = {
            "narrative": "Summary narrative",
            "retrieval": {"tags": ["tag1", "tag2"], "keywords": ["kw1", "kw2"]},
            "metadata": {"depth": "deep", "follow_ups": ["follow1", "follow2"]},
        }

        result = create_zettelkasten_memory_note(episodic_data, summary_data)
        assert "## Summary" in result
        assert "## Behavioral Patterns" in result
        assert "## Experience Learnings" in result
        assert "## Guidance" in result
        assert "Tags:" in result
        assert "Keywords:" in result

    def test_create_zettelkasten_memory_note_n_a_values_coverage(self):
        """Test create_zettelkasten_memory_note filters out N/A values."""
        episodic_data = {
            "behavioral_profile": {"communication": "N/A", "learning": "visual"}
        }
        summary_data = {}

        result = create_zettelkasten_memory_note(episodic_data, summary_data)
        assert "N/A" not in result
        assert "visual" in result

    def test_prepare_memory_for_storage_all_fields_coverage(self):
        """Test prepare_memory_for_storage with all fields populated."""
        summary_data = {
            "narrative": "test narrative",
            "retrieval": {"tags": ["tag1"], "keywords": ["kw1"]},
        }
        episodic_data = {"behavioral_profile": {"communication": "direct"}}

        result = prepare_memory_for_storage(
            note_text="test note",
            summary_data=summary_data,
            episodic_data=episodic_data,
        )
        assert result["text"] == "test note"
        assert "metadata" in result
        assert result["metadata"]["tags"] == ["tag1"]
