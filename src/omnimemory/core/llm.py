import asyncio
import os
import random
import time
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import litellm

from omnimemory.core.logger_utils import get_logger

logger = get_logger(name="omnimemory.core.llm")

FuncType = TypeVar("FuncType", bound=Callable[..., Any])


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1,
    max_delay: float = 60,
    backoff_factor: float = 2,
) -> Callable[[FuncType], FuncType]:
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        backoff_factor: Multiplier for delay increase.

    Returns:
        Decorator that wraps callables with retry semantics.
    """

    def decorator(func: FuncType) -> FuncType:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception: Optional[Exception] = None
                async_func = cast(Callable[..., Awaitable[Any]], func)

                for attempt in range(max_retries + 1):
                    try:
                        result = await async_func(*args, **kwargs)
                        return result
                    except Exception as exc:
                        last_exception = exc
                        error_msg = str(exc).lower()
                        retryable = any(
                            keyword in error_msg
                            for keyword in [
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
                        )
                        if retryable and attempt < max_retries:
                            delay = min(
                                base_delay * (backoff_factor**attempt), max_delay
                            )
                            jitter = random.uniform(0, 0.1 * delay)
                            total_delay = delay + jitter
                            logger.warning(
                                "Retryable error on attempt %s/%s: %s",
                                attempt + 1,
                                max_retries + 1,
                                exc,
                            )
                            logger.info("Retrying in %.2f seconds...", total_delay)
                            await asyncio.sleep(total_delay)
                            continue
                        if retryable:
                            logger.error(
                                "Max retries (%s) exceeded. Last error: %s",
                                max_retries,
                                exc,
                            )
                        else:
                            logger.error("Non-retryable error: %s", exc)
                        break

                if last_exception is None:
                    raise RuntimeError("Retry loop exited without exception")
                raise last_exception

            return cast(FuncType, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            sync_func = cast(Callable[..., Any], func)

            for attempt in range(max_retries + 1):
                try:
                    result = sync_func(*args, **kwargs)
                    return result
                except Exception as exc:
                    last_exception = exc
                    error_msg = str(exc).lower()
                    retryable = any(
                        keyword in error_msg
                        for keyword in [
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
                    )
                    if retryable and attempt < max_retries:
                        delay = min(base_delay * (backoff_factor**attempt), max_delay)
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter
                        logger.warning(
                            "Retryable error on attempt %s/%s: %s",
                            attempt + 1,
                            max_retries + 1,
                            exc,
                        )
                        logger.info("Retrying in %.2f seconds...", total_delay)
                        time.sleep(total_delay)
                        continue
                    if retryable:
                        logger.error(
                            "Max retries (%s) exceeded. Last error: %s",
                            max_retries,
                            exc,
                        )
                    else:
                        logger.error("Non-retryable error: %s", exc)
                    break

            if last_exception is None:
                raise RuntimeError("Retry loop exited without exception")
            raise last_exception

        return cast(FuncType, sync_wrapper)

    return decorator


class LLMConnection:
    """Manages LLM connections using LiteLLM with environment variable configuration."""

    def __init__(self) -> None:
        """Initialize LLMConnection with environment variables."""
        self.llm_config: Optional[Dict[str, Any]] = None
        self.embedding_config: Optional[Dict[str, Any]] = None
        self._load_llm_configuration()
        self._load_embedding_configuration()

    def __str__(self) -> str:
        """Return a readable string representation of the LLMConnection."""
        llm_status = "configured" if self.llm_config else "not configured"
        embedding_status = "configured" if self.embedding_config else "not configured"
        return f"LLMConnection(LLM={llm_status}, EMBEDDING={embedding_status})"

    def __repr__(self) -> str:
        """Return a detailed representation of the LLMConnection."""
        return self.__str__()

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value."""
        return os.getenv(key, default)

    def _get_env_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get environment variable as integer."""
        value = self._get_env(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            logger.warning("Invalid integer value for %s: %s", key, value)
            return default

    def _get_env_float(
        self, key: str, default: Optional[float] = None
    ) -> Optional[float]:
        """Get environment variable as float."""
        value = self._get_env(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            logger.warning("Invalid float value for %s: %s", key, value)
            return default

    def _load_llm_configuration(self) -> None:
        """Load LLM configuration from environment variables."""
        llm_api_key = self._get_env("LLM_API_KEY")
        if not llm_api_key:
            logger.debug("LLM_API_KEY not set, skipping LLM configuration")
            return

        provider = self._get_env("LLM_PROVIDER")
        model = self._get_env("LLM_MODEL")

        if not provider or not model:
            logger.warning(
                "LLM_PROVIDER and LLM_MODEL are required when LLM_API_KEY is set"
            )
            return

        try:
            provider_lower = provider.lower()

            provider_model_map = {
                "openai": f"openai/{model}",
                "anthropic": f"anthropic/{model}",
                "groq": f"groq/{model}",
                "gemini": f"gemini/{model}",
                "azure": f"azure/{model}",
                "azureopenai": f"azure/{model}",
                "ollama": f"ollama/{model}",
                "mistral": f"mistral/{model}",
                "deepseek": f"deepseek/{model}",
                "openrouter": f"openrouter/{model}",
            }

            full_model = provider_model_map.get(provider_lower, model)

            self.llm_config = {
                "provider": provider,
                "model": full_model,
                "temperature": self._get_env_float("LLM_TEMPERATURE"),
                "max_tokens": self._get_env_int("LLM_MAX_TOKENS"),
                "top_p": self._get_env_float("LLM_TOP_P"),
            }

            if provider_lower in ["azure", "azureopenai"]:
                azure_endpoint = self._get_env("LLM_AZURE_ENDPOINT")
                azure_api_version = self._get_env("LLM_AZURE_API_VERSION")
                azure_deployment = self._get_env("LLM_AZURE_DEPLOYMENT")

                if azure_endpoint:
                    os.environ["AZURE_API_BASE"] = azure_endpoint
                if azure_api_version:
                    os.environ["AZURE_API_VERSION"] = azure_api_version
                if azure_deployment:
                    self.llm_config["model"] = f"azure/{azure_deployment}"

            if provider_lower == "ollama":
                ollama_host = self._get_env("LLM_OLLAMA_HOST")
                if ollama_host:
                    os.environ["OLLAMA_API_BASE"] = ollama_host

            os.environ["LITELLM_LOGGING"] = "False"
            os.environ["LITELLM_LOGGING_FAILSAFE"] = "False"

            self._set_llm_api_key(provider_lower, llm_api_key)

            logger.info(f"LLM configuration loaded: provider={provider}, model={model}")
        except Exception as e:
            logger.error(f"Error loading LLM configuration: {e}")
            self.llm_config = None

    def _load_embedding_configuration(self) -> None:
        """Load Embedding configuration from environment variables."""
        embedding_api_key = self._get_env("EMBEDDING_API_KEY")
        if not embedding_api_key:
            logger.debug("EMBEDDING_API_KEY not set, skipping embedding configuration")
            return

        provider = self._get_env("EMBEDDING_PROVIDER")
        model = self._get_env("EMBEDDING_MODEL")
        dimensions = self._get_env_int("EMBEDDING_DIMENSIONS")

        if not provider or not model:
            logger.warning(
                "EMBEDDING_PROVIDER and EMBEDDING_MODEL are required when EMBEDDING_API_KEY is set"
            )
            return

        if not dimensions or dimensions <= 0:
            logger.warning("EMBEDDING_DIMENSIONS must be a positive integer")
            return

        try:
            provider_lower = provider.lower()

            provider_model_map = {
                "openai": f"openai/{model}",
                "cohere": f"cohere/{model}",
                "mistral": f"mistral/{model}",
                "gemini": f"gemini/{model}",
                "vertex_ai": f"vertex_ai/{model}",
                "voyage": f"voyage/{model}",
                "nebius": f"nebius/{model}",
                "nvidia_nim": f"nvidia_nim/{model}",
                "bedrock": f"bedrock/{model}",
                "huggingface": f"huggingface/{model}",
                "azure": f"azure/{model}",
                "azureopenai": f"azure/{model}",
            }

            full_model = provider_model_map.get(provider_lower, model)

            encoding_format = self._get_env("EMBEDDING_ENCODING_FORMAT")
            if not encoding_format:
                encoding_format = "base64" if provider_lower == "voyage" else "float"

            self.embedding_config = {
                "provider": provider,
                "model": full_model,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
                "timeout": self._get_env_int("EMBEDDING_TIMEOUT"),
            }

            if provider_lower in ["azure", "azureopenai"]:
                azure_endpoint = self._get_env("EMBEDDING_AZURE_ENDPOINT")
                azure_api_version = self._get_env("EMBEDDING_AZURE_API_VERSION")
                azure_deployment = self._get_env("EMBEDDING_AZURE_DEPLOYMENT")

                if azure_endpoint:
                    os.environ["AZURE_API_BASE"] = azure_endpoint
                if azure_api_version:
                    os.environ["AZURE_API_VERSION"] = azure_api_version
                if azure_deployment:
                    self.embedding_config["model"] = f"azure/{azure_deployment}"

            elif provider_lower == "vertex_ai":
                vertex_project = self._get_env("EMBEDDING_VERTEX_PROJECT")
                vertex_location = self._get_env("EMBEDDING_VERTEX_LOCATION")

                if vertex_project:
                    litellm.vertex_project = vertex_project
                if vertex_location:
                    litellm.vertex_location = vertex_location

            elif provider_lower == "nvidia_nim":
                nvidia_nim_api_base = self._get_env("EMBEDDING_NVIDIA_NIM_API_BASE")
                if nvidia_nim_api_base:
                    os.environ["NVIDIA_NIM_API_BASE"] = nvidia_nim_api_base

            elif provider_lower == "bedrock":
                aws_region = self._get_env("EMBEDDING_AWS_REGION")
                if aws_region:
                    os.environ["AWS_REGION_NAME"] = aws_region

            self._set_embedding_api_key(provider_lower, embedding_api_key)

            logger.info(
                f"Embedding configuration loaded: provider={provider}, model={model}, dimensions={dimensions}"
            )
        except Exception as e:
            logger.error(f"Error loading embedding configuration: {e}")
            self.embedding_config = None

    def _set_llm_api_key(self, provider: str, api_key: str) -> None:
        """Set API key environment variable for LLM provider."""
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "azure": "AZURE_API_KEY",
            "azureopenai": "AZURE_API_KEY",
        }

        env_var = provider_key_map.get(provider)
        if env_var:
            os.environ[env_var] = api_key
            logger.debug(f"Set {env_var} for LLM provider: {provider}")
        else:
            logger.warning(f"Unknown LLM provider: {provider}, API key not set")

    def _set_embedding_api_key(self, provider: str, api_key: str) -> None:
        """Set API key environment variable for embedding provider."""
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "voyage": "VOYAGE_API_KEY",
            "nebius": "NEBIUS_API_KEY",
            "nvidia_nim": "NVIDIA_NIM_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "azure": "AZURE_API_KEY",
            "azureopenai": "AZURE_API_KEY",
            "bedrock": None,
            "vertex_ai": None,
        }

        env_var = provider_key_map.get(provider)
        if env_var:
            os.environ[env_var] = api_key
            logger.debug(f"Set {env_var} for embedding provider: {provider}")
        elif env_var is None and provider in ["bedrock", "vertex_ai"]:
            logger.debug(f"Provider {provider} uses different authentication method")
        else:
            logger.warning(f"Unknown embedding provider: {provider}, API key not set")

    @retry_with_backoff(max_retries=3, base_delay=1, max_delay=30)
    async def embedding_call(
        self,
        input_text: Union[str, Sequence[str]],
        input_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Call the embedding service using LiteLLM.

        Args:
            input_text: Text or batch of texts to embed.
            input_type: Provider-specific input type hint.
            metadata: Optional metadata forwarded to providers that support it.
            user: Optional user identifier for auditing.

        Returns:
            Provider embedding response payload or None on failure.
        """
        try:
            if not self.embedding_config:
                logger.error("Embedding configuration not loaded")
                return None

            if not self.embedding_config.get(
                "provider"
            ) or not self.embedding_config.get("model"):
                logger.error("Embedding provider or model not configured")
                return None

            params = {
                "model": self.embedding_config["model"],
                "input": input_text,
            }

            if self.embedding_config.get("dimensions") is not None:
                params["dimensions"] = self.embedding_config["dimensions"]

            params["encoding_format"] = self.embedding_config.get(
                "encoding_format", "float"
            )

            if self.embedding_config.get("timeout") is not None:
                params["timeout"] = self.embedding_config["timeout"]

            if input_type:
                params["input_type"] = input_type

            if metadata:
                params["metadata"] = metadata

            if user:
                params["user"] = user

            provider = self.embedding_config["provider"].lower()

            if provider == "cohere" and not input_type:
                params["input_type"] = "search_document"

            if provider == "openai":
                model_name = self.embedding_config.get("model", "").lower()
                if (
                    "text-embedding-3" in model_name
                    or "text-embedding-small-3" in model_name
                ):
                    params.pop("dimensions", None)

            litellm.drop_params = True

            response = await litellm.aembedding(**params)
            return response

        except Exception as e:
            model_name = (
                self.embedding_config.get("model")
                if self.embedding_config
                else "unknown"
            )
            error_message = (
                f"Error calling embedding service with model {model_name}: {e}"
            )
            logger.error(error_message)
            return None

    @retry_with_backoff(max_retries=3, base_delay=1, max_delay=30)
    def embedding_call_sync(
        self,
        input_text: Union[str, Sequence[str]],
        input_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Synchronous call to the embedding service using LiteLLM.

        Args mirror embedding_call.
        """
        try:
            if not self.embedding_config:
                logger.error("Embedding configuration not loaded")
                return None

            if not self.embedding_config.get(
                "provider"
            ) or not self.embedding_config.get("model"):
                logger.error("Embedding provider or model not configured")
                return None

            params = {
                "model": self.embedding_config["model"],
                "input": input_text,
            }

            if self.embedding_config.get("dimensions") is not None:
                params["dimensions"] = self.embedding_config["dimensions"]

            params["encoding_format"] = self.embedding_config.get(
                "encoding_format", "float"
            )

            if self.embedding_config.get("timeout") is not None:
                params["timeout"] = self.embedding_config["timeout"]

            if input_type:
                params["input_type"] = input_type

            if metadata:
                params["metadata"] = metadata

            if user:
                params["user"] = user

            provider = self.embedding_config["provider"].lower()

            if provider == "cohere" and not input_type:
                params["input_type"] = "search_document"

            if provider == "openai":
                model_name = self.embedding_config.get("model", "").lower()
                if (
                    "text-embedding-3" in model_name
                    or "text-embedding-small-3" in model_name
                ):
                    params.pop("dimensions", None)

            litellm.drop_params = True

            response = litellm.embedding(**params)
            return response

        except Exception as e:
            model_name = (
                self.embedding_config.get("model")
                if self.embedding_config
                else "unknown"
            )
            error_message = (
                f"Error calling embedding service with model {model_name}: {e}"
            )
            logger.error(error_message)
            return None

    def is_embedding_available(self) -> bool:
        """Return True when embedding configuration and API key are available."""
        return (
            self._get_env("EMBEDDING_API_KEY") is not None
            and self.embedding_config is not None
        )

    def is_llm_available(self) -> bool:
        """Return True when chat completion configuration and API key exist."""
        return self._get_env("LLM_API_KEY") is not None and self.llm_config is not None

    def to_dict(self, msg: Any) -> Any:
        """Convert message to dictionary format for LiteLLM consumption."""
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump(exclude_none=True)

            if "timestamp" in msg_dict and hasattr(msg_dict["timestamp"], "timestamp"):
                msg_dict["timestamp"] = msg_dict["timestamp"].timestamp()
            elif "timestamp" in msg_dict and hasattr(msg_dict["timestamp"], "tzinfo"):
                msg_dict["timestamp"] = msg_dict["timestamp"].timestamp()
            return msg_dict
        elif isinstance(msg, dict):
            return msg
        elif hasattr(msg, "__dict__"):
            return {k: v for k, v in msg.__dict__.items() if v is not None}
        else:
            return msg

    @retry_with_backoff(max_retries=3, base_delay=1, max_delay=30)
    async def llm_call(
        self,
        messages: Sequence[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Any]:
        """
        Call the LLM using LiteLLM.

        Args:
            messages: Chat completion messages (dict or pydantic models).
            tools: Optional tool definitions for tool-calling models.

        Returns:
            LiteLLM completion response or None when unavailable.
        """
        try:
            if not self.llm_config:
                logger.debug("LLM configuration not loaded, skipping LLM call")
                return None

            messages_dicts = [self.to_dict(m) for m in messages]

            params = {
                "model": self.llm_config["model"],
                "messages": messages_dicts,
            }

            if self.llm_config.get("temperature") is not None:
                params["temperature"] = self.llm_config["temperature"]

            if self.llm_config.get("max_tokens") is not None:
                params["max_tokens"] = self.llm_config["max_tokens"]

            if self.llm_config.get("top_p") is not None:
                params["top_p"] = self.llm_config["top_p"]

            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            if self.llm_config["provider"].lower() == "openrouter":
                if not tools:
                    params["stop"] = ["\n\nObservation:"]

            litellm.drop_params = True

            response = await litellm.acompletion(**params)
            return response

        except Exception as e:
            error_message = f"Error calling LLM with model {self.llm_config.get('model') if self.llm_config else 'unknown'}: {e}"
            logger.error(error_message)
            return None

    @retry_with_backoff(max_retries=3, base_delay=1, max_delay=30)
    def llm_call_sync(
        self,
        messages: Sequence[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Any]:
        """Synchronous variant of llm_call."""
        try:
            if not self.llm_config:
                logger.debug("LLM configuration not loaded, skipping LLM call")
                return None

            messages_dicts = [self.to_dict(m) for m in messages]

            params = {
                "model": self.llm_config["model"],
                "messages": messages_dicts,
            }

            if self.llm_config.get("temperature") is not None:
                params["temperature"] = self.llm_config["temperature"]

            if self.llm_config.get("max_tokens") is not None:
                params["max_tokens"] = self.llm_config["max_tokens"]

            if self.llm_config.get("top_p") is not None:
                params["top_p"] = self.llm_config["top_p"]

            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            if self.llm_config["provider"].lower() == "openrouter":
                if not tools:
                    params["stop"] = ["\n\nObservation:"]

            litellm.drop_params = True

            response = litellm.completion(**params)
            return response

        except Exception as e:
            error_message = f"Error calling LLM with model {self.llm_config.get('model') if self.llm_config else 'unknown'}: {e}"
            logger.error(error_message)
            return None
