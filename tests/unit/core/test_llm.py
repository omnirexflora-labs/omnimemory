"""
Comprehensive unit tests for OmniMemory LLM connection and retry logic.
"""

import pytest
import os
from unittest.mock import Mock, AsyncMock, patch

from omnimemory.core.llm import LLMConnection, retry_with_backoff


class TestRetryWithBackoff:
    """Test cases for retry_with_backoff decorator."""

    def test_execute_successfully_on_first_attempt(self):
        """Test execute function successfully on first attempt."""

        @retry_with_backoff(max_retries=3)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_retry_on_rate_limit_error(self):
        """Test retry on retryable errors (rate limit)."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("rate limit exceeded")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count[0] == 2

    def test_retry_on_timeout_error(self):
        """Test retry on timeout errors."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("timeout error")
            return "success"

        result = test_func()
        assert result == "success"

    def test_retry_on_connection_error(self):
        """Test retry on connection errors."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("connection error")
            return "success"

        result = test_func()
        assert result == "success"

    def test_use_exponential_backoff(self):
        """Test use exponential backoff."""
        delays = []

        def mock_sleep(delay):
            delays.append(delay)

        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.1, backoff_factor=2)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("rate limit")
            return "success"

        with patch("omnimemory.core.llm.time.sleep", side_effect=mock_sleep):
            test_func()

        assert len(delays) == 2
        assert delays[0] >= 0.1
        assert delays[1] >= 0.2

    def test_add_jitter_to_delay(self):
        """Test add jitter to delay."""
        delays = []

        def mock_sleep(delay):
            delays.append(delay)

        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=1.0, backoff_factor=2)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("rate limit")
            return "success"

        with patch("omnimemory.core.llm.time.sleep", side_effect=mock_sleep):
            with patch("omnimemory.core.llm.random.uniform", return_value=0.05):
                test_func()

        assert len(delays) == 1
        assert delays[0] >= 1.0

    def test_respect_max_delay(self):
        """Test respect max_delay."""
        delays = []

        def mock_sleep(delay):
            delays.append(delay)

        call_count = [0]

        @retry_with_backoff(
            max_retries=3, base_delay=0.5, max_delay=1.0, backoff_factor=2
        )
        def test_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("rate limit")
            return "success"

        with patch("omnimemory.core.llm.time.sleep", side_effect=mock_sleep):
            with patch("omnimemory.core.llm.random.uniform", return_value=0.0):
                test_func()

        assert len(delays) == 2
        assert delays[0] <= 1.0
        assert delays[1] <= 1.0

    def test_retry_up_to_max_retries(self):
        """Test retry up to max_retries."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            raise Exception("rate limit")

        with pytest.raises(Exception, match="rate limit"):
            test_func()

        assert call_count[0] == 4

    def test_raise_exception_after_max_retries(self):
        """Test raise exception after max retries."""

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def test_func():
            raise Exception("rate limit")

        with pytest.raises(Exception, match="rate limit"):
            test_func()

    def test_skip_retry_on_non_retryable_errors(self):
        """Test skip retry on non-retryable errors."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            raise Exception("permission denied")

        with pytest.raises(Exception, match="permission denied"):
            test_func()

        assert call_count[0] == 1

    def test_handle_all_retryable_error_keywords(self):
        """Test handle all retryable error keywords."""
        retryable_keywords = [
            "rate limit",
            "rate_limit",
            "rpm",
            "tpm",
            "quota",
            "throttle",
            "too many requests",
            "429",
            "temporary",
            "timeout",
            "connection",
        ]

        for keyword in retryable_keywords:
            call_count = [0]

            @retry_with_backoff(max_retries=2, base_delay=0.01)
            def test_func():
                call_count[0] += 1
                if call_count[0] < 2:
                    raise Exception(f"Error: {keyword}")
                return "success"

            result = test_func()
            assert result == "success"
            assert call_count[0] == 2

    def test_log_retry_attempts(self):
        """Test log retry attempts."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("rate limit")
            return "success"

        with patch("omnimemory.core.llm.logger") as mock_logger:
            test_func()
            assert mock_logger.warning.called
            assert mock_logger.info.called

    def test_log_final_failure(self):
        """Test log final failure."""

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def test_func():
            raise Exception("rate limit")

        with patch("omnimemory.core.llm.logger") as mock_logger:
            with pytest.raises(Exception):
                test_func()
            assert mock_logger.error.called


class TestLLMConnection:
    """Test cases for LLMConnection."""

    @patch.dict(os.environ, {}, clear=True)
    def test_init_with_valid_env_vars(self):
        """Test initialize with valid environment variables."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()
            assert conn.llm_config is not None
            assert conn.llm_config["provider"] == "openai"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_load_llm_configuration(self):
        """Test load LLM configuration."""
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "test-key",
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "gpt-4",
                "LLM_TEMPERATURE": "0.7",
                "LLM_MAX_TOKENS": "1000",
            },
        ):
            conn = LLMConnection()
            assert conn.llm_config is not None
            assert conn.llm_config["temperature"] == 0.7
            assert conn.llm_config["max_tokens"] == 1000

    @patch.dict(os.environ, {}, clear=True)
    def test_init_load_embedding_configuration(self):
        """Test load embedding configuration."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            assert conn.embedding_config is not None
            assert conn.embedding_config["dimensions"] == 1536

    @patch.dict(os.environ, {}, clear=True)
    def test_init_handle_missing_llm_api_key(self):
        """Test handle missing LLM_API_KEY (skip LLM config)."""
        with patch.dict(os.environ, {}):
            conn = LLMConnection()
            assert conn.llm_config is None

    @patch.dict(os.environ, {}, clear=True)
    def test_init_handle_missing_embedding_api_key(self):
        """Test handle missing EMBEDDING_API_KEY (skip embedding config)."""
        with patch.dict(os.environ, {}):
            conn = LLMConnection()
            assert conn.embedding_config is None

    @patch.dict(os.environ, {}, clear=True)
    def test_init_handle_missing_provider_model(self):
        """Test handle missing LLM_PROVIDER/MODEL (skip config)."""
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
            conn = LLMConnection()
            assert conn.llm_config is None

    @patch.dict(os.environ, {}, clear=True)
    def test_init_configure_azure_settings(self):
        """Test configure Azure-specific settings."""
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "test-key",
                "LLM_PROVIDER": "azure",
                "LLM_MODEL": "gpt-4",
                "LLM_AZURE_ENDPOINT": "https://test.openai.azure.com",
                "LLM_AZURE_API_VERSION": "2023-05-15",
                "LLM_AZURE_DEPLOYMENT": "gpt-4-deployment",
            },
        ):
            conn = LLMConnection()
            assert conn.llm_config is not None
            assert "azure" in conn.llm_config["model"].lower()
            assert os.environ.get("AZURE_API_BASE") == "https://test.openai.azure.com"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_configure_ollama_settings(self):
        """Test configure Ollama-specific settings."""
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "test-key",
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "llama2",
                "LLM_OLLAMA_HOST": "http://localhost:11434",
            },
        ):
            conn = LLMConnection()
            assert conn.llm_config is not None
            assert os.environ.get("OLLAMA_API_BASE") == "http://localhost:11434"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_set_litellm_env_vars(self):
        """Test set LiteLLM environment variables."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()
            assert isinstance(conn, LLMConnection)
            assert os.environ.get("LITELLM_LOGGING") == "False"
            assert os.environ.get("LITELLM_LOGGING_FAILSAFE") == "False"

    def test_get_env(self):
        """Test get environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            conn = LLMConnection()
            value = conn._get_env("TEST_VAR")
            assert value == "test_value"

    def test_get_env_return_default(self):
        """Test return default when not set."""
        conn = LLMConnection()
        value = conn._get_env("NONEXISTENT_VAR", default="default_value")
        assert value == "default_value"

    def test_get_env_int_parse_correctly(self):
        """Test parse integer correctly."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            conn = LLMConnection()
            value = conn._get_env_int("TEST_INT")
            assert value == 42

    def test_get_env_int_handle_invalid(self):
        """Test handle invalid integer (return default, log warning)."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            conn = LLMConnection()
            with patch("omnimemory.core.llm.logger") as mock_logger:
                value = conn._get_env_int("TEST_INT", default=10)
                assert value == 10
                assert mock_logger.warning.called

    def test_get_env_float_parse_correctly(self):
        """Test parse float correctly."""
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            conn = LLMConnection()
            value = conn._get_env_float("TEST_FLOAT")
            assert value == 3.14

    def test_get_env_float_handle_invalid(self):
        """Test handle invalid float (return default, log warning)."""
        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            conn = LLMConnection()
            with patch("omnimemory.core.llm.logger") as mock_logger:
                value = conn._get_env_float("TEST_FLOAT", default=0.5)
                assert value == 0.5
                assert mock_logger.warning.called

    @patch.dict(os.environ, {}, clear=True)
    def test_load_llm_configuration_all_providers(self):
        """Test handle all supported providers."""
        providers = [
            "openai",
            "anthropic",
            "groq",
            "gemini",
            "azure",
            "ollama",
            "mistral",
            "deepseek",
            "openrouter",
        ]

        for provider in providers:
            with patch.dict(
                os.environ,
                {
                    "LLM_API_KEY": "test-key",
                    "LLM_PROVIDER": provider,
                    "LLM_MODEL": "test-model",
                },
            ):
                conn = LLMConnection()
                if provider in ["azure", "azureopenai"]:
                    continue
                assert conn.llm_config is not None
                assert (
                    provider.lower() in conn.llm_config["model"].lower()
                    or conn.llm_config["model"] == "test-model"
                )

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_all_providers(self):
        """Test handle all supported embedding providers."""
        providers = [
            "openai",
            "cohere",
            "mistral",
            "gemini",
            "vertex_ai",
            "voyage",
            "nebius",
            "nvidia_nim",
            "bedrock",
            "huggingface",
            "azure",
        ]

        for provider in providers:
            with patch.dict(
                os.environ,
                {
                    "EMBEDDING_API_KEY": "test-key",
                    "EMBEDDING_PROVIDER": provider,
                    "EMBEDDING_MODEL": "test-model",
                    "EMBEDDING_DIMENSIONS": "1536",
                },
            ):
                conn = LLMConnection()
                if provider in ["bedrock", "vertex_ai"]:
                    continue
                if provider in ["azure", "azureopenai"]:
                    continue
                assert conn.embedding_config is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_set_llm_api_key_known_provider(self):
        """Test set API key for known provider."""
        with patch.dict(os.environ, {}, clear=True):
            conn = LLMConnection()
            conn._set_llm_api_key("openai", "test-api-key")
            assert os.environ.get("OPENAI_API_KEY") == "test-api-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_set_llm_api_key_unknown_provider(self):
        """Test handle unknown provider (log warning)."""
        conn = LLMConnection()
        with patch("omnimemory.core.llm.logger") as mock_logger:
            conn._set_llm_api_key("unknown_provider", "test-key")
            assert mock_logger.warning.called

    @patch.dict(os.environ, {}, clear=True)
    def test_set_embedding_api_key_known_provider(self):
        """Test set API key for known embedding provider."""
        with patch.dict(os.environ, {}, clear=True):
            conn = LLMConnection()
            conn._set_embedding_api_key("openai", "test-api-key")
            assert os.environ.get("OPENAI_API_KEY") == "test-api-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_set_embedding_api_key_providers_without_key(self):
        """Test handle providers without API keys (Bedrock, Vertex AI)."""
        conn = LLMConnection()
        with patch("omnimemory.core.llm.logger"):
            conn._set_embedding_api_key("bedrock", "test-key")
            conn._set_embedding_api_key("vertex_ai", "test-key")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_llm_call_success(self):
        """Test call LLM successfully."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion:
                mock_acompletion.return_value = mock_response
                result = await conn.llm_call([{"role": "user", "content": "test"}])
                assert result == mock_response

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_llm_call_missing_config_returns_none(self):
        """Test handle missing llm_config (return None)."""
        conn = LLMConnection()
        assert conn.llm_config is None

        result = await conn.llm_call([{"role": "user", "content": "test"}])
        assert result is None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_llm_call_handles_exception(self):
        """Test handle exceptions in llm_call."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()

            with patch(
                "omnimemory.core.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion:
                mock_acompletion.side_effect = Exception("API error")
                result = await conn.llm_call([{"role": "user", "content": "test"}])
                assert result is None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_embedding_call_success_string(self):
        """Test call embedding API successfully with string input."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.return_value = mock_response
                result = await conn.embedding_call("test text")
                assert result == mock_response

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_embedding_call_success_list(self):
        """Test call embedding API successfully with list input."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.return_value = mock_response
                result = await conn.embedding_call(["text1", "text2"])
                assert result == mock_response

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_embedding_call_missing_config_returns_none(self):
        """Test handle missing embedding_config (return None)."""
        conn = LLMConnection()
        assert conn.embedding_config is None

        result = await conn.embedding_call("test")
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_call_sync_success(self):
        """Test synchronous embedding call."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.embedding", return_value=mock_response
            ):
                result = conn.embedding_call_sync("test text")
                assert result == mock_response

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_call_sync_success(self):
        """Test synchronous LLM call."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.completion", return_value=mock_response
            ):
                result = conn.llm_call_sync([{"role": "user", "content": "test"}])
                assert result == mock_response

    @patch.dict(os.environ, {}, clear=True)
    def test_is_embedding_available_true(self):
        """Test is_embedding_available returns True when configured."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            assert conn.is_embedding_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_embedding_available_false(self):
        """Test is_embedding_available returns False when not configured."""
        conn = LLMConnection()
        assert conn.is_embedding_available() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_llm_available_true(self):
        """Test is_llm_available returns True when configured."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()
            assert conn.is_llm_available() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_llm_available_false(self):
        """Test is_llm_available returns False when not configured."""
        conn = LLMConnection()
        assert conn.is_llm_available() is False

    def test_to_dict_pydantic_model(self):
        """Test convert Pydantic model to dict."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            role: str
            content: str

        conn = LLMConnection()
        msg = TestModel(role="user", content="test")
        result = conn.to_dict(msg)
        assert isinstance(result, dict)
        assert result["role"] == "user"
        assert result["content"] == "test"

    def test_to_dict_dict(self):
        """Test convert dict to dict (no-op)."""
        conn = LLMConnection()
        msg = {"role": "user", "content": "test"}
        result = conn.to_dict(msg)
        assert result == msg

    def test_to_dict_object_with_dict(self):
        """Test convert object with __dict__ to dict."""

        class TestObj:
            def __init__(self):
                self.role = "user"
                self.content = "test"

        conn = LLMConnection()
        msg = TestObj()
        result = conn.to_dict(msg)
        assert isinstance(result, dict)
        assert result["role"] == "user"

    def test_to_dict_handle_timestamp(self):
        """Test handle timestamp conversion."""
        from datetime import datetime
        from pydantic import BaseModel

        class TestModel(BaseModel):
            role: str
            timestamp: datetime

        conn = LLMConnection()
        msg = TestModel(role="user", timestamp=datetime.now())
        result = conn.to_dict(msg)
        assert isinstance(result["timestamp"], float)

    def test_str_representation(self):
        """Test __str__ returns readable representation."""
        conn = LLMConnection()
        result = str(conn)
        assert "LLMConnection" in result

    def test_repr_representation(self):
        """Test __repr__ returns detailed representation."""
        conn = LLMConnection()
        result = repr(conn)
        assert "LLMConnection" in result

    @patch.dict(os.environ, {}, clear=True)
    def test_load_llm_configuration_exception_coverage_line_192(self):
        """Test exception handling in _load_llm_configuration (line 192)."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            with patch("omnimemory.core.llm.logger") as mock_logger:
                with patch.object(
                    LLMConnection,
                    "_set_llm_api_key",
                    side_effect=Exception("Config error"),
                ):
                    conn = LLMConnection()
                    assert conn.llm_config is None
                    mock_logger.error.assert_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_missing_provider_coverage_line_208(self):
        """Test missing provider/model in embedding config (line 208)."""
        with patch.dict(os.environ, {"EMBEDDING_API_KEY": "test-key"}):
            conn = LLMConnection()
            assert conn.embedding_config is None

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_invalid_dimensions_coverage_line_214(self):
        """Test invalid dimensions in embedding config (line 214)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "0",
            },
        ):
            conn = LLMConnection()
            assert conn.embedding_config is None

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_azure_coverage_line_255(self):
        """Test Azure embedding configuration (line 255)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "azure",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
                "EMBEDDING_AZURE_ENDPOINT": "https://test.openai.azure.com",
                "EMBEDDING_AZURE_API_VERSION": "2023-05-15",
                "EMBEDDING_AZURE_DEPLOYMENT": "embedding-deployment",
            },
        ):
            conn = LLMConnection()
            assert conn.embedding_config is not None
            assert os.environ.get("AZURE_API_BASE") == "https://test.openai.azure.com"

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_vertex_ai_coverage_line_266(self):
        """Test Vertex AI embedding configuration (line 266)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "vertex_ai",
                "EMBEDDING_MODEL": "textembedding-gecko",
                "EMBEDDING_DIMENSIONS": "768",
                "EMBEDDING_VERTEX_PROJECT": "test-project",
                "EMBEDDING_VERTEX_LOCATION": "us-central1",
            },
        ):
            conn = LLMConnection()
            assert conn.embedding_config is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_nvidia_nim_coverage_line_273(self):
        """Test NVIDIA NIM embedding configuration (line 273)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "nvidia_nim",
                "EMBEDDING_MODEL": "nvidia/nv-embedqa-e5-v5",
                "EMBEDDING_DIMENSIONS": "1024",
                "EMBEDDING_NVIDIA_NIM_API_BASE": "https://integrate.api.nvidia.com/v1",
            },
        ):
            conn = LLMConnection()
            assert conn.embedding_config is not None
            assert (
                os.environ.get("NVIDIA_NIM_API_BASE")
                == "https://integrate.api.nvidia.com/v1"
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_bedrock_coverage_line_278(self):
        """Test Bedrock embedding configuration (line 278)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "bedrock",
                "EMBEDDING_MODEL": "amazon.titan-embed-text-v1",
                "EMBEDDING_DIMENSIONS": "1024",
                "EMBEDDING_AWS_REGION": "us-east-1",
            },
        ):
            conn = LLMConnection()
            assert conn.embedding_config is not None
            assert os.environ.get("AWS_REGION_NAME") == "us-east-1"

    @patch.dict(os.environ, {}, clear=True)
    def test_load_embedding_configuration_exception_coverage_line_285(self):
        """Test exception handling in _load_embedding_configuration (line 285)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            with patch("omnimemory.core.llm.logger") as mock_logger:
                with patch.object(
                    LLMConnection,
                    "_set_embedding_api_key",
                    side_effect=Exception("Error"),
                ):
                    conn = LLMConnection()
                    assert conn.embedding_config is None
                    mock_logger.error.assert_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_call_missing_config_coverage_line_414(self):
        """Test embedding_call_sync missing config (line 414)."""
        conn = LLMConnection()
        result = conn.embedding_call_sync("test")
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_call_missing_provider_model_coverage_line_420(self):
        """Test embedding_call_sync missing provider/model (line 420)."""
        conn = LLMConnection()
        conn.embedding_config = {}
        result = conn.embedding_call_sync("test")
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_embedding_call_with_input_type_coverage_line_372(self):
        """Test embedding_call with input_type parameter (line 372)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.return_value = mock_response
                result = await conn.embedding_call("test", input_type="search_document")
                assert result == mock_response

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_embedding_call_with_metadata_coverage_line_375(self):
        """Test embedding_call with metadata parameter (line 375)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.return_value = mock_response
                result = await conn.embedding_call("test", metadata={"key": "value"})
                assert result == mock_response

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_embedding_call_with_user_coverage_line_378(self):
        """Test embedding_call with user parameter (line 378)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.return_value = mock_response
                result = await conn.embedding_call("test", user="user123")
                assert result == mock_response

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_embedding_call_cohere_input_type_coverage_line_383(self):
        """Test embedding_call auto-set input_type for Cohere (line 383)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "cohere",
                "EMBEDDING_MODEL": "embed-english-v3.0",
                "EMBEDDING_DIMENSIONS": "1024",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.return_value = mock_response
                _ = await conn.embedding_call("test")
                call_args = mock_aembedding.call_args
                assert call_args[1].get("input_type") == "search_document"

    # @patch.dict(os.environ, {}, clear=True)
    # @pytest.mark.asyncio
    # async def test_embedding_call_openai_text_embedding_3_coverage_line_391(self):
    #     """Test embedding_call remove dimensions for text-embedding-3 (line 391)."""
    #     with patch.dict(
    #         os.environ,
    #         {
    #             "EMBEDDING_API_KEY": "test-key",
    #             "EMBEDDING_PROVIDER": "openai",
    #             "EMBEDDING_MODEL": "text-embedding-3-small",
    #             "EMBEDDING_DIMENSIONS": "1536",
    #         },
    #     ):
    #         conn = LLMConnection()
    #         mock_response = Mock()
    #
    #         with patch(
    #             "omnimemory.core.llm.litellm.aembedding", new_callable=AsyncMock
    #         ) as mock_aembedding:
    #             mock_aembedding.return_value = mock_response
    #             _ = await conn.embedding_call("test")
    #             call_args = mock_aembedding.call_args
    #             assert "dimensions" not in call_args[1]

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_embedding_call_exception_coverage_line_398(self):
        """Test embedding_call exception handling (line 398)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()

            with patch(
                "omnimemory.core.llm.litellm.aembedding",
                new_callable=AsyncMock,
                side_effect=Exception("API error"),
            ):
                with patch("omnimemory.core.llm.logger") as mock_logger:
                    result = await conn.embedding_call("test")
                    assert result is None
                    mock_logger.error.assert_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_call_sync_with_parameters_coverage(self):
        """Test embedding_call_sync with all optional parameters."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.embedding", return_value=mock_response
            ) as mock_embedding:
                result = conn.embedding_call_sync(
                    "test",
                    input_type="search_document",
                    metadata={"key": "value"},
                    user="user123",
                )
                assert result == mock_response
                call_args = mock_embedding.call_args
                assert call_args[1].get("input_type") == "search_document"
                assert call_args[1].get("metadata") == {"key": "value"}
                assert call_args[1].get("user") == "user123"

    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_call_sync_exception_coverage_line_465(self):
        """Test embedding_call_sync exception handling (line 465)."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "test-key",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSIONS": "1536",
            },
        ):
            conn = LLMConnection()

            with patch(
                "omnimemory.core.llm.litellm.embedding",
                side_effect=Exception("API error"),
            ):
                with patch("omnimemory.core.llm.logger") as mock_logger:
                    result = conn.embedding_call_sync("test")
                    assert result is None
                    mock_logger.error.assert_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_to_dict_timestamp_with_tzinfo_coverage_line_489(self):
        """Test to_dict with timestamp that has tzinfo (line 489)."""
        from datetime import datetime, timezone
        from pydantic import BaseModel

        class TestModel(BaseModel):
            role: str
            timestamp: datetime

        conn = LLMConnection()
        msg = TestModel(role="user", timestamp=datetime.now(timezone.utc))
        result = conn.to_dict(msg)
        assert isinstance(result["timestamp"], float)

    @patch.dict(os.environ, {}, clear=True)
    def test_to_dict_fallback_coverage_line_496(self):
        """Test to_dict fallback for non-dict, non-model objects (line 496)."""
        conn = LLMConnection()
        result = conn.to_dict("just a string")
        assert result == "just a string"

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_llm_call_with_tools_coverage_line_527(self):
        """Test llm_call with tools parameter (line 527)."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()
            mock_response = Mock()
            tools = [{"type": "function", "function": {"name": "test_func"}}]

            with patch(
                "omnimemory.core.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion:
                mock_acompletion.return_value = mock_response
                _ = await conn.llm_call(
                    [{"role": "user", "content": "test"}], tools=tools
                )
                call_args = mock_acompletion.call_args
                assert call_args[1].get("tools") == tools
                assert call_args[1].get("tool_choice") == "auto"

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_llm_call_openrouter_stop_coverage_line_531(self):
        """Test llm_call with OpenRouter stop parameter (line 531)."""
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "test-key",
                "LLM_PROVIDER": "openrouter",
                "LLM_MODEL": "openai/gpt-4",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.acompletion", new_callable=AsyncMock
            ) as mock_acompletion:
                mock_acompletion.return_value = mock_response
                _ = await conn.llm_call([{"role": "user", "content": "test"}])
                call_args = mock_acompletion.call_args
                assert call_args[1].get("stop") == ["\n\nObservation:"]

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_llm_call_exception_coverage_line_539(self):
        """Test llm_call exception handling (line 539)."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()

            with patch(
                "omnimemory.core.llm.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=Exception("API error"),
            ):
                with patch("omnimemory.core.llm.logger") as mock_logger:
                    result = await conn.llm_call([{"role": "user", "content": "test"}])
                    assert result is None
                    mock_logger.error.assert_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_call_sync_with_tools_coverage_line_575(self):
        """Test llm_call_sync with tools parameter (line 575)."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()
            mock_response = Mock()
            tools = [{"type": "function", "function": {"name": "test_func"}}]

            with patch(
                "omnimemory.core.llm.litellm.completion", return_value=mock_response
            ) as mock_completion:
                _ = conn.llm_call_sync(
                    [{"role": "user", "content": "test"}], tools=tools
                )
                call_args = mock_completion.call_args
                assert call_args[1].get("tools") == tools
                assert call_args[1].get("tool_choice") == "auto"

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_call_sync_openrouter_stop_coverage_line_579(self):
        """Test llm_call_sync with OpenRouter stop parameter (line 579)."""
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "test-key",
                "LLM_PROVIDER": "openrouter",
                "LLM_MODEL": "openai/gpt-4",
            },
        ):
            conn = LLMConnection()
            mock_response = Mock()

            with patch(
                "omnimemory.core.llm.litellm.completion", return_value=mock_response
            ) as mock_completion:
                _ = conn.llm_call_sync([{"role": "user", "content": "test"}])
                call_args = mock_completion.call_args
                assert call_args[1].get("stop") == ["\n\nObservation:"]

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_call_sync_exception_coverage_line_587(self):
        """Test llm_call_sync exception handling (line 587)."""
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4"},
        ):
            conn = LLMConnection()

            with patch(
                "omnimemory.core.llm.litellm.completion",
                side_effect=Exception("API error"),
            ):
                with patch("omnimemory.core.llm.logger") as mock_logger:
                    result = conn.llm_call_sync([{"role": "user", "content": "test"}])
                    assert result is None
                    mock_logger.error.assert_called()
