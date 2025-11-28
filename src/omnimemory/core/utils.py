import re
import json
import tiktoken
import hashlib
import math
import time
from threading import RLock
from rapidfuzz import fuzz, process
from typing import Any, List, Optional, Tuple, Dict, TYPE_CHECKING
from omnimemory.core.logger_utils import get_logger
from datetime import datetime, timezone
from omnimemory.core.schemas import UserMessages, Message, MemoryBatcherMessage

if TYPE_CHECKING:
    from omnimemory.sdk import OmniMemorySDK

logger = get_logger(name="omnimemory.core.utils")

_FUZZY_DEDUP_THRESHOLD = 75
_MAX_AGE_HOURS = 43800
_HALF_LIFE_FACTOR = 4
_IMPORTANCE_QUALITY_WEIGHT = 0.5
_IMPORTANCE_FOLLOWUP_WEIGHT = 0.3
_IMPORTANCE_RICHNESS_WEIGHT = 0.2
_RECENCY_BOOST_FACTOR = 0.1
_IMPORTANCE_BOOST_FACTOR = 0.1
_EMBEDDING_CACHE_PREFIX = "omnimemory:embedding_cache:"
_CACHE_TTL_SECONDS = 3600
_EMBEDDING_CACHE_MAX_ENTRIES = 512
_EMBEDDING_CACHE: Dict[str, Tuple[List[float], float]] = {}
_EMBEDDING_CACHE_LOCK = RLock()


def _get_half_life_hours() -> float:
    """Calculate half-life in hours for exponential decay"""
    return _MAX_AGE_HOURS / _HALF_LIFE_FACTOR


def _get_importance_weights() -> Dict[str, float]:
    """Get importance scoring weights."""
    weights = {
        "quality": _IMPORTANCE_QUALITY_WEIGHT,
        "followup": _IMPORTANCE_FOLLOWUP_WEIGHT,
        "richness": _IMPORTANCE_RICHNESS_WEIGHT,
    }
    total = sum(weights.values())
    if total != 1.0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def get_tokenizer_for_model(model_name: str) -> Optional[Any]:
    """
    Get the appropriate tokenizer for a given model name.

    Args:
        model_name: Name of the model (will be used directly, fallback to gpt-4)

    Returns:
        Tokenizer object or None if not available
    """

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.encoding_for_model("gpt-4")
        except Exception:
            return None


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string using the appropriate tokenizer.

    Args:
        text: Input text to count tokens for
        model_name: Model name to determine which tokenizer to use

    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0

    try:
        tokenizer = get_tokenizer_for_model(model_name)
        if tokenizer:
            return len(tokenizer.encode(text))
        else:
            return len(text) // 4
    except Exception:
        return len(text) // 4


def chunk_text_by_tokens(
    text: str, chunk_size: int = 500, overlap: int = 50, model_name: str = "gpt-4"
) -> List[str]:
    """
    Split text into chunks based on token count rather than character count.

    Args:
        text: Input text to chunk
        chunk_size: Maximum tokens per chunk (default: 500)
        overlap: Number of tokens to overlap between chunks (default: 50)
        model_name: Model name for token counting

    Returns:
        List of text chunks
    """
    if not text:
        return []

    try:
        tokenizer = get_tokenizer_for_model(model_name)
        if not tokenizer:
            char_chunk_size = chunk_size * 4
            char_overlap = overlap * 4
            chunks = []
            start = 0
            while start < len(text):
                end = start + char_chunk_size
                chunk = text[start:end]
                if chunk:
                    chunks.append(chunk)
                start = end - char_overlap if end < len(text) else end
            return chunks

        tokens = tokenizer.encode(text)
        chunks = []

        i = 0
        while i < len(tokens):
            chunk_end = min(i + chunk_size, len(tokens))

            chunk_tokens = tokens[i:chunk_end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if chunk_end == len(tokens):
                break

            next_i = chunk_end - overlap
            if next_i <= i:
                i = i + 1
            else:
                i = next_i

            if i >= len(tokens):
                break

        return chunks

    except Exception:
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = start + char_chunk_size
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            start = end - char_overlap if end < len(text) else end
        return chunks


def estimate_chunking_stats(
    text: str, chunk_size: int = 500, overlap: int = 50, model_name: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Get statistics about how text will be chunked without actually chunking it.

    Args:
        text: Input text
        chunk_size: Maximum tokens per chunk
        overlap: Token overlap between chunks
        model_name: Model name for token counting

    Returns:
        Dictionary with chunking statistics
    """
    total_tokens = count_tokens(text, model_name)
    total_chars = len(text)

    if total_tokens <= chunk_size:
        return {
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "chunks_needed": 1,
            "chunk_size_tokens": chunk_size,
            "overlap_tokens": 0,
            "effective_tokens": total_tokens,
            "token_efficiency": 1.0,
        }

    effective_chunk_size = chunk_size - overlap
    chunks_needed = (total_tokens + effective_chunk_size - 1) // effective_chunk_size

    total_overlap_tokens = overlap * (chunks_needed - 1) if chunks_needed > 1 else 0
    effective_tokens = total_tokens + total_overlap_tokens

    token_efficiency = total_tokens / effective_tokens if effective_tokens > 0 else 1.0

    return {
        "total_tokens": total_tokens,
        "total_characters": total_chars,
        "chunks_needed": chunks_needed,
        "chunk_size_tokens": chunk_size,
        "overlap_tokens": overlap,
        "effective_tokens": effective_tokens,
        "token_efficiency": token_efficiency,
    }


def normalize_token(token: str) -> str:
    """
    Normalize a token string for fuzzy matching.

    Converts to lowercase, replaces hyphens/underscores with spaces,
    removes non-alphanumeric characters, and normalizes whitespace.

    Args:
        token: Input token string to normalize.

    Returns:
        Normalized token string.
    """
    token = token.lower()
    token = re.sub(r"[-_]", " ", token)
    token = re.sub(r"[^a-z0-9\s]", "", token)
    token = re.sub(r"\s+", " ", token)
    token = token.strip()
    return token


def clean_and_parse_json(json_string: str) -> Any:
    """
    Clean and parse JSON string from LLM output.

    Handles common JSON issues from LLM generation:
    - Trailing commas in objects and arrays
    - Comments (removes them)
    - Unescaped quotes in strings
    - Whitespace issues

    Args:
        json_string: Raw JSON string from LLM

    Returns:
        Parsed JSON object (dict, list, or primitive)

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after cleaning
        ValueError: If JSON string is empty
    """
    if not json_string:
        raise ValueError("Empty JSON string")

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass

    json_string = re.sub(r"//.*?$", "", json_string, flags=re.MULTILINE)
    json_string = re.sub(r"/\*.*?\*/", "", json_string, flags=re.DOTALL)

    def remove_trailing_commas(text):
        """Remove trailing commas before } or ] while preserving commas inside strings."""
        result = []
        in_string = False
        escape_next = False
        i = 0

        while i < len(text):
            char = text[i]

            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == "\\":
                result.append(char)
                escape_next = True
                i += 1
                continue

            if char in ('"', "'"):
                in_string = not in_string
                result.append(char)
                i += 1
                continue

            if not in_string:
                if char == ",":
                    j = i + 1
                    while j < len(text) and text[j] in " \t\n\r":
                        j += 1
                    if j < len(text) and text[j] in "}]":
                        i += 1
                        continue

            result.append(char)
            i += 1

        return "".join(result)

    json_string = remove_trailing_commas(json_string)

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:

        def remove_trailing_commas_regex(text):
            """More aggressive regex-based trailing comma removal."""
            lines = text.split("\n")
            cleaned_lines = []
            for line in lines:
                line = re.sub(r",(\s*[}\]])", r"\1", line)
                cleaned_lines.append(line)
            result = "\n".join(cleaned_lines)
            result = re.sub(r",(\s*[}\]])", r"\1", result)
            return result

        json_string = remove_trailing_commas_regex(json_string)

        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", json_string, re.DOTALL
            )
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            logger.error(
                f"Failed to parse JSON even after cleaning. Original error: {e}"
            )
            logger.debug(f"Problematic JSON (first 1000 chars): {json_string[:1000]}")
            raise


def fuzzy_dedup(items: List[str], threshold: Optional[int] = None) -> List[str]:
    """
    Deduplicate list of strings by fuzzy similarity after normalization.

    Args:
        items: List of strings to deduplicate.
        threshold: Minimum similarity score (0-100, default: 75).

    Returns:
        List of deduplicated strings.
    """
    if threshold is None:
        threshold = _FUZZY_DEDUP_THRESHOLD

    normalized_map = {item: normalize_token(item) for item in items}
    deduped = []
    seen = set()

    for orig, norm in normalized_map.items():
        if norm in seen:
            continue
        seen.add(norm)

        close_matches = process.extract(
            norm,
            list(normalized_map.values()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )

        if close_matches:
            for match, score, _ in close_matches:
                seen.add(match)
        deduped.append(orig)

    return deduped


def format_conversation(messages: List[Any] | str | Any) -> str:
    """
    Format conversation messages into a single text string.

    This is the single source of truth for message formatting. Handles:
    - Structured messages: List of Message objects or dicts with role/content
    - Unstructured text: Plain string (packaged as user message)
    - UserMessages objects: Extracts messages list and formats

    Args:
        messages: Can be:
            - List[Message] or List[Dict]: Structured messages
            - str: Unstructured text (packaged as "user: {text}")
            - UserMessages: Object with messages attribute

    Returns:
        Formatted string in "role: content" format, one per line
    """
    if isinstance(messages, str):
        return f"user: {messages.strip()}"

    if hasattr(messages, "messages") and hasattr(messages, "app_id"):
        messages = messages.messages

    if isinstance(messages, list):
        formatted_parts = []
        for msg in messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role
                content = msg.content
                formatted_parts.append(f"{role}: {content}")
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_parts.append(f"{role}: {content}")
            else:
                formatted_parts.append(f"user: {str(msg)}")
        return "\n".join(formatted_parts)

    return f"user: {str(messages)}"


class _MemoryBatcher:
    """
    Utility helper that buffers messages until the configured threshold.

    Automatically flushes messages to memory when the buffer reaches max_messages.
    """

    def __init__(
        self,
        sdk: "OmniMemorySDK",
        app_id: str,
        user_id: str,
        session_id: Optional[str],
        max_messages: int,
    ) -> None:
        """
        Initialize MemoryBatcher.

        Args:
            sdk: OmniMemorySDK instance for adding memories.
            app_id: Application identifier.
            user_id: User identifier.
            session_id: Optional session identifier.
            max_messages: Maximum number of messages before auto-flush.
        """
        self.sdk = sdk
        self.app_id = app_id
        self.user_id = user_id
        self.session_id = session_id
        self.max_messages = max_messages
        self._buffer: List[Dict[str, str]] = []
        self._last_delivery: Optional[str] = None
        self._last_task_id: Optional[str] = None

    @property
    def pending_count(self) -> int:
        """
        Get the number of pending messages in the buffer.

        Returns:
            Number of messages currently buffered.
        """
        return len(self._buffer)

    @property
    def can_cleanup(self) -> bool:
        """
        Check if the batcher can be safely cleaned up.

        Returns:
            True if buffer is empty, False otherwise.
        """
        return not self._buffer

    def status_dict(self, status_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status dictionary for the batcher.

        Args:
            status_override: Optional status string to override default status.

        Returns:
            Dictionary containing batcher status information.
        """
        if status_override:
            status = status_override
        else:
            status = "pending" if self._buffer else "empty"
        return {
            "app_id": self.app_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "pending_messages": len(self._buffer),
            "batch_size": self.max_messages,
            "status": status,
            "last_delivery": self._last_delivery,
            "last_task_id": self._last_task_id,
        }

    async def add_messages(
        self, messages: List["MemoryBatcherMessage"]
    ) -> Dict[str, Any]:
        """
        Add messages to the buffer and flush if threshold is reached.

        Args:
            messages: List of MemoryBatcherMessage objects to add.

        Returns:
            Status dictionary with current batcher state.
        """
        status = None
        for msg in messages:
            payload = msg.model_dump()
            self._buffer.append(
                {"role": payload["role"], "content": payload["content"]}
            )
            if len(self._buffer) == self.max_messages:
                status = await self._flush_full_batch()
        return self.status_dict(status_override=status)

    async def _flush_full_batch(self) -> str:
        """
        Flush the current buffer to memory storage.

        Returns:
            Status string indicating flush completion.
        """
        user_message = UserMessages(
            app_id=self.app_id,
            user_id=self.user_id,
            session_id=self.session_id,
            messages=[Message(**msg) for msg in self._buffer],
        )
        result = await self.sdk.add_memory(user_message)
        self._buffer.clear()
        self._last_task_id = result.get("task_id")
        self._last_delivery = "add_memory"
        return "flushed"


def parse_iso_to_datetime(timestamp_str: str) -> Optional[datetime]:
    """
    Convert an ISO 8601 timestamp string to a UTC datetime object.

    Args:
        timestamp_str: ISO 8601 formatted timestamp string.

    Returns:
        UTC datetime object, or None if parsing fails or string is empty.
    """
    if not timestamp_str:
        return None
    try:
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(timestamp_str)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO timestamp string to datetime object.

    Args:
        timestamp_str: ISO format timestamp

    Returns:
        datetime: Parsed datetime object
    """
    if not timestamp_str:
        raise ValueError("timestamp_str cannot be None or empty")
    if isinstance(timestamp_str, str) and timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(timestamp_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def calculate_recency_score(
    created_at_str: Optional[str],
    updated_at_str: Optional[str],
    max_age_hours: Optional[int] = None,
) -> float:
    """
    Calculate recency score based on how recent the memory is.

    Uses exponential decay over time for more natural recency weighting.

    Args:
        created_at_str: ISO format timestamp string
        updated_at_str: ISO format timestamp string
        max_age_hours: Maximum age to consider (default: 43800 hours = 5 years)

    Returns:
        float: Recency score from 0.0 (old) to 1.0 (very recent)
    """
    if max_age_hours is None:
        max_age_hours = _MAX_AGE_HOURS

    try:
        if not created_at_str and not updated_at_str:
            return 0.1

        time_to_use = None
        if created_at_str:
            created_at = parse_timestamp(created_at_str)
            time_to_use = created_at
        if updated_at_str:
            updated_at = parse_timestamp(updated_at_str)
            if time_to_use:
                time_to_use = max(time_to_use, updated_at)
            else:
                time_to_use = updated_at

        if not time_to_use:
            return 0.1

        now = datetime.now(timezone.utc)

        age_hours = (now - time_to_use).total_seconds() / 3600

        if age_hours <= 0:
            return 1.0

        half_life = _get_half_life_hours()
        recency_score = math.exp(-age_hours / half_life)

        return max(0.01, min(1.0, recency_score))

    except Exception as e:
        logger.warning(f"Error calculating recency score: {e}")
        return 0.1


def calculate_importance_score(metadata: Dict[str, Any]) -> float:
    """
    Calculate importance score based on content significance.
    Importance IS about:
    1. Interaction Quality (50%): Did the conversation go well? High quality = valuable memory
    2. Follow-up Potential (30%): Are there actionable items or continuation points?
    3. Content Richness (20%): Good tags/keywords help retrieval find it

    Args:
        metadata: Memory metadata dictionary

    Returns:
        float: Importance score from 0.0 to 1.0
    """
    importance_weights = _get_importance_weights()
    importance_factors = []

    quality = metadata.get("interaction_quality", "").lower()
    if quality == "high":
        quality_score = 1.0
    elif quality == "medium":
        quality_score = 0.6
    elif quality == "low":
        quality_score = 0.2
    else:
        quality_score = 0.5
    importance_factors.append(("quality", quality_score, importance_weights["quality"]))

    followups = metadata.get("follow_up_potential", [])
    if isinstance(followups, list):
        followup_count = len(
            [
                f
                for f in followups
                if f and str(f).strip() and str(f).strip().upper() != "N/A"
            ]
        )
    else:
        followup_count = 0

    if followup_count >= 3:
        followup_score = 1.0
    elif followup_count >= 2:
        followup_score = 0.8
    elif followup_count >= 1:
        followup_score = 0.6
    else:
        followup_score = 0.3
    importance_factors.append(
        ("followup", followup_score, importance_weights["followup"])
    )

    tags_count = len(metadata.get("tags", []))
    keywords_count = len(metadata.get("keywords", []))
    total_content_items = tags_count + keywords_count

    if total_content_items >= 10:
        richness_score = 1.0
    elif total_content_items >= 5:
        richness_score = 0.8
    elif total_content_items >= 2:
        richness_score = 0.6
    else:
        richness_score = 0.3
    importance_factors.append(
        ("richness", richness_score, importance_weights["richness"])
    )

    total_weight: float = 0.0
    weighted_sum: float = 0.0

    for factor_name, score, weight in importance_factors:
        weighted_sum += score * weight
        total_weight += weight

    final_score = weighted_sum / total_weight if total_weight > 0 else 0.5

    return max(0.0, min(1.0, final_score))


def calculate_composite_score(
    semantic_score: float,
    metadata: Dict[str, Any],
) -> float:
    """
    Calculate composite score using MULTIPLICATIVE approach (relevance-first).

    REDESIGNED: Relevance is PRIMARY. Recency and importance are small boosts.

    Formula: composite = relevance × (1 + recency_boost + importance_boost)
    - recency_boost = recency × RECENCY_BOOST_FACTOR (max +10%)
    - importance_boost = importance × IMPORTANCE_BOOST_FACTOR (max +10%)

    This ensures:
    - Low relevance (0.4) can't be saved by recency/importance
    - High relevance (0.7) gets small boost from recency/importance
    - Relevance is always the base multiplier

    Args:
        semantic_score: Vector similarity score (0.0-1.0) - this is the relevance
        metadata: Memory metadata

    Returns:
        float: Composite score from 0.0 to 1.0
    """
    relevance_score = semantic_score
    recency_score = calculate_recency_score(
        created_at_str=metadata.get("created_at"),
        updated_at_str=metadata.get("updated_at"),
    )
    importance_score = calculate_importance_score(metadata)

    recency_boost = recency_score * _RECENCY_BOOST_FACTOR
    importance_boost = importance_score * _IMPORTANCE_BOOST_FACTOR
    composite_score = relevance_score * (1.0 + recency_boost + importance_boost)
    return max(0.0, min(1.0, composite_score))


def determine_relationship_type(
    composite_score: float, metadata: Dict[str, Any]
) -> str:
    """
    Determine the type of relationship based on composite score and metadata.

    Args:
        composite_score: Composite similarity score
        metadata: Memory metadata

    Returns:
        Relationship type string
    """
    if composite_score >= 0.9:
        return "highly_similar"
    elif composite_score >= 0.8:
        return "very_similar"
    elif composite_score >= 0.7:
        complexity = metadata.get("conversation_complexity", 1)
        if complexity >= 4:
            return "related_technical"
        else:
            return "related_general"
    elif composite_score >= 0.6:
        return "loosely_related"
    else:
        return "tangentially_related"


def get_embedding_cache_key(text: str) -> str:
    """
    Generate cache key for text embedding.

    Args:
        text: Input text to generate cache key for.

    Returns:
        SHA256 hash of the text as a hexadecimal string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prune_embedding_cache(now: float) -> None:
    """
    Remove expired entries from the in-memory embedding cache.

    Args:
        now: Current timestamp in seconds since epoch.
    """
    expired_keys = [
        key for key, (_, expiry) in _EMBEDDING_CACHE.items() if expiry <= now
    ]
    for key in expired_keys:
        _EMBEDDING_CACHE.pop(key, None)


def get_cached_embedding(text: str) -> Optional[List[float]]:
    """
    Get cached embedding from in-memory cache if available.

    Args:
        text: Input text to retrieve embedding for.

    Returns:
        Cached embedding vector if available and not expired, None otherwise.
    """
    cache_key = f"{_EMBEDDING_CACHE_PREFIX}{get_embedding_cache_key(text)}"
    now = time.time()

    with _EMBEDDING_CACHE_LOCK:
        entry = _EMBEDDING_CACHE.get(cache_key)
        if not entry:
            return None

        embedding, expiry = entry
        if expiry <= now:
            _EMBEDDING_CACHE.pop(cache_key, None)
            return None
        return list(embedding)


def cache_embedding(text: str, embedding: List[float]) -> None:
    """
    Cache embedding in in-memory cache with TTL.

    Automatically prunes expired entries and evicts oldest entries
    if cache exceeds maximum size.

    Args:
        text: Input text that was embedded.
        embedding: Embedding vector to cache.
    """
    if not embedding:
        return

    cache_key = f"{_EMBEDDING_CACHE_PREFIX}{get_embedding_cache_key(text)}"
    expires_at = time.time() + _CACHE_TTL_SECONDS

    with _EMBEDDING_CACHE_LOCK:
        _prune_embedding_cache(time.time())

        if len(_EMBEDDING_CACHE) >= _EMBEDDING_CACHE_MAX_ENTRIES:
            oldest_key = min(_EMBEDDING_CACHE.items(), key=lambda item: item[1][1])[0]
            _EMBEDDING_CACHE.pop(oldest_key, None)

        _EMBEDDING_CACHE[cache_key] = (list(embedding), expires_at)
        logger.debug("Cached embedding in memory (entries=%d)", len(_EMBEDDING_CACHE))


def create_zettelkasten_memory_note(
    episodic_data: Dict[str, Any],
    summary_data: Dict[str, Any],
) -> str:
    """
    Create a Zettelkasten-style memory note from episodic and summary data.
    This creates a flowing, natural note optimized for semantic search.

    Structure:
    1. Main narrative (from summary)
    2. Behavioral insights (from episodic)
    3. Guidance for future interactions (from episodic)
    4. Tags and metadata (from summary)

    Args:
        episodic_data: Episodic memory data dictionary.
        summary_data: Summary memory data dictionary.

    Returns:
        Formatted Zettelkasten-style memory note string.
    """
    sections = []

    narrative = summary_data.get("narrative", "")
    if narrative and narrative != "N/A":
        sections.append(f"## Summary\n{narrative}")

    behavioral_parts = []

    behavioral = episodic_data.get("behavioral_profile", {})
    if behavioral:
        comm = behavioral.get("communication", "")
        learning = behavioral.get("learning", "")
        problem = behavioral.get("problem_solving", "")
        decision = behavioral.get("decision_making", "")

        style_parts = []
        if comm and comm != "N/A":
            style_parts.append(comm)
        if learning and learning != "N/A":
            style_parts.append(learning)
        if problem and problem != "N/A":
            style_parts.append(problem)
        if decision and decision != "N/A":
            style_parts.append(decision)

        if style_parts:
            behavioral_parts.append(f"User's style: {' '.join(style_parts)}")

    insights = episodic_data.get("interaction_insights", {})
    if insights:
        triggers = insights.get("engagement_triggers", "")
        friction = insights.get("friction_points", "")
        optimal = insights.get("optimal_approach", "")

        insight_parts = []
        if triggers and triggers != "N/A":
            insight_parts.append(f"Engages well with: {triggers}")
        if friction and friction != "N/A":
            insight_parts.append(f"Struggles with: {friction}")
        if optimal and optimal != "N/A":
            insight_parts.append(f"Works best when: {optimal}")

        if insight_parts:
            behavioral_parts.append(" ".join(insight_parts))

    if behavioral_parts:
        sections.append(f"## Behavioral Patterns\n{' '.join(behavioral_parts)}")

    experience_parts = []

    worked = episodic_data.get("what_worked", {})
    if worked:
        strategies = worked.get("strategies", [])
        pattern = worked.get("pattern", "")

        if strategies and any(s and s != "N/A" for s in strategies):
            clean_strategies = [s for s in strategies if s and s != "N/A"]
            worked_text = f"Successful approaches: {'; '.join(clean_strategies)}"
            if pattern and pattern != "N/A":
                worked_text += f" — {pattern}"
            experience_parts.append(worked_text)

    failed = episodic_data.get("what_failed", {})
    if failed:
        strategies = failed.get("strategies", [])
        pattern = failed.get("pattern", "")

        if strategies and any(s and s != "N/A" for s in strategies):
            clean_strategies = [s for s in strategies if s and s != "N/A"]
            failed_text = f"Approaches to avoid: {'; '.join(clean_strategies)}"
            if pattern and pattern != "N/A":
                failed_text += f" — {pattern}"
            experience_parts.append(failed_text)

    if experience_parts:
        sections.append(f"## Experience Learnings\n{' '.join(experience_parts)}")

    guidance = episodic_data.get("future_guidance", {})
    if guidance:
        recommended = guidance.get("recommended_approaches", [])
        avoid = guidance.get("avoid_approaches", [])
        adaptation = guidance.get("adaptation_note", "")

        guidance_parts = []
        if recommended and any(r and r != "N/A" for r in recommended):
            clean_rec = [r for r in recommended if r and r != "N/A"]
            guidance_parts.append(f"Do: {'; '.join(clean_rec)}")

        if avoid and any(a and a != "N/A" for a in avoid):
            clean_avoid = [a for a in avoid if a and a != "N/A"]
            guidance_parts.append(f"Don't: {'; '.join(clean_avoid)}")

        if adaptation and adaptation != "N/A":
            guidance_parts.append(f"Key insight: {adaptation}")

        if guidance_parts:
            sections.append(f"## Guidance\n{' '.join(guidance_parts)}")

    footer_parts = []

    retrieval = summary_data.get("retrieval", {})
    if retrieval:
        tags = retrieval.get("tags", [])
        keywords = retrieval.get("keywords", [])

        if tags and any(t and t != "N/A" for t in tags):
            clean_tags = [str(t).strip() for t in tags if t and str(t).strip() != "N/A"]
            if clean_tags:
                footer_parts.append(f"Tags: {', '.join(clean_tags)}")

        if keywords and any(k and k != "N/A" for k in keywords):
            clean_kw = [
                str(k).strip() for k in keywords if k and str(k).strip() != "N/A"
            ]
            if clean_kw:
                footer_parts.append(f"Keywords: {', '.join(clean_kw)}")

    metadata = summary_data.get("metadata", {})
    if metadata:
        depth = metadata.get("depth", "")
        follow_ups = metadata.get("follow_ups", [])

        meta_text = []
        if depth and depth != "N/A":
            meta_text.append(f"Content depth: {depth}")

        if follow_ups and any(f and f != "N/A" for f in follow_ups):
            clean_follow = [f for f in follow_ups if f and f != "N/A"]
            if clean_follow:
                meta_text.append(f"Follow-up areas: {'; '.join(clean_follow)}")

        if meta_text:
            footer_parts.append(" | ".join(meta_text))

    if footer_parts:
        sections.append(f"---\n{' | '.join(footer_parts)}")

    full_note = "\n\n".join(sections)

    full_note = re.sub(r"\n{3,}", "\n\n", full_note)
    full_note = re.sub(r" {2,}", " ", full_note)
    full_note = full_note.strip()

    return full_note


def prepare_memory_for_storage(
    note_text: str,
    summary_data: Dict[str, Any],
    episodic_data: Dict[str, Any],
    message_count: int = 0,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare memory note for vector database storage.

    Returns a dict with:
    - text: The Zettelkasten note (for embedding and retrieval)
    - metadata: Structured data for filtering, ranking, and hybrid search

    Note: When retrieving, only return the 'text' field to the agent.
    The metadata is used internally for better search/ranking.

    Args:
        note_text: Zettelkasten-formatted memory note text.
        summary_data: Summary memory data dictionary.
        episodic_data: Episodic memory data dictionary.
        message_count: Number of messages in the conversation (default: 0).
        timestamp: ISO format timestamp string (default: current time).

    Returns:
        Dictionary containing 'text' and 'metadata' keys for storage.
    """
    from datetime import datetime

    retrieval = summary_data.get("retrieval", {})
    metadata_obj = summary_data.get("metadata", {})

    episodic_context = episodic_data.get("context", {})

    return {
        "text": note_text,
        "metadata": {
            "tags": [t for t in retrieval.get("tags", []) if t and t != "N/A"],
            "keywords": [k for k in retrieval.get("keywords", []) if k and k != "N/A"],
            "query_hooks": [
                q for q in retrieval.get("queries", []) if q and q != "N/A"
            ],
            "content_depth": metadata_obj.get("depth", "medium"),
            "has_behavioral_data": bool(episodic_data.get("behavioral_profile")),
            "has_guidance": bool(episodic_data.get("future_guidance")),
            "analysis_complete": episodic_context.get("analysis_limitation", "")
            == "N/A",
            "timestamp": timestamp or datetime.now().isoformat(),
            "message_count": message_count,
            "follow_up_areas": [
                f for f in metadata_obj.get("follow_ups", []) if f and f != "N/A"
            ],
        },
    }


def create_agent_memory_note(agent_data: Dict[str, Any]) -> str:
    """
    Create a simple memory note from agent add_memory data.
    This creates a concise note optimized for semantic search.

    Structure:
    1. Main narrative with header
    2. Tags and metadata

    Args:
        agent_data: Agent memory data dictionary.

    Returns:
        Formatted memory note string.
    """
    sections = []

    # Main narrative with header
    narrative = agent_data.get("narrative", "")
    if narrative and narrative != "N/A":
        sections.append(f"## Note\n{narrative}")

    # Footer with metadata
    footer_parts = []

    retrieval = agent_data.get("retrieval", {})
    if retrieval:
        tags = retrieval.get("tags", [])
        keywords = retrieval.get("keywords", [])

        if tags and any(t and t != "N/A" for t in tags):
            clean_tags = [str(t).strip() for t in tags if t and str(t).strip() != "N/A"]
            if clean_tags:
                footer_parts.append(f"Tags: {', '.join(clean_tags)}")

        if keywords and any(k and k != "N/A" for k in keywords):
            clean_kw = [
                str(k).strip() for k in keywords if k and str(k).strip() != "N/A"
            ]
            if clean_kw:
                footer_parts.append(f"Keywords: {', '.join(clean_kw)}")

    metadata = agent_data.get("metadata", {})
    if metadata:
        depth = metadata.get("depth", "")
        follow_ups = metadata.get("follow_ups", [])

        meta_text = []
        if depth and depth != "N/A":
            meta_text.append(f"Depth: {depth}")

        if follow_ups and any(f and f != "N/A" for f in follow_ups):
            clean_follow = [f for f in follow_ups if f and f != "N/A"]
            if clean_follow:
                meta_text.append(f"Follow-up: {'; '.join(clean_follow)}")

        if meta_text:
            footer_parts.append(" | ".join(meta_text))

    if footer_parts:
        sections.append(f"---\n{' | '.join(footer_parts)}")

    full_note = "\n\n".join(sections)

    # Clean up extra whitespace
    full_note = re.sub(r"\n{3,}", "\n\n", full_note)
    full_note = re.sub(r" {2,}", " ", full_note)
    full_note = full_note.strip()

    return full_note


def prepare_agent_memory_for_storage(
    note_text: str,
    agent_data: Dict[str, Any],
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare agent memory note for vector database storage.
    Returns a dict with:
    - text: The formatted note (for embedding and retrieval)
    - metadata: Structured data for filtering, ranking, and hybrid search

    Note: When retrieving, only return the 'text' field to the agent.
    The metadata is used internally for better search/ranking.

    Args:
        note_text: Formatted agent memory note text.
        agent_data: Agent memory data dictionary.
        timestamp: ISO format timestamp string (default: current time).

    Returns:
        Dictionary containing 'text' and 'metadata' keys for storage.
    """
    from datetime import datetime

    retrieval = agent_data.get("retrieval", {})
    metadata_obj = agent_data.get("metadata", {})

    return {
        "text": note_text,
        "metadata": {
            "tags": [t for t in retrieval.get("tags", []) if t and t != "N/A"],
            "keywords": [k for k in retrieval.get("keywords", []) if k and k != "N/A"],
            "query_hooks": [
                q for q in retrieval.get("queries", []) if q and q != "N/A"
            ],
            "content_depth": metadata_obj.get("depth", "medium"),
            "timestamp": timestamp or datetime.now().isoformat(),
            "follow_up_areas": [
                f for f in metadata_obj.get("follow_ups", []) if f and f != "N/A"
            ],
            "source": "agent_memory",
        },
    }
