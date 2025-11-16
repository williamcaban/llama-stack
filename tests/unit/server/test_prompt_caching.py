# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for prompt caching middleware.

These tests validate the PromptCachingMiddleware implementation including:
- Cache hit/miss detection
- Token counting integration
- Multi-tenant isolation
- Circuit breaker behavior
- Streaming bypass
- Configuration validation
- Error handling and graceful degradation
"""

from unittest.mock import AsyncMock, patch

import pytest

from llama_stack.core.server.prompt_caching import (
    CachingConfig,
    CircuitBreakerConfig,
    MemoryCacheConfig,
    PromptCachingMiddleware,
    RedisCacheConfig,
)
from llama_stack.providers.utils.cache.memory import MemoryCacheStore
from llama_stack_api.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionUsage,
    OpenAIChoice,
    OpenAISystemMessageParam,
    OpenAIUserMessageParam,
)


class TestCachingConfig:
    """Tests for CachingConfig pydantic model."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = CachingConfig()

        assert config.enabled is True
        assert config.cache_backend == "memory"
        assert config.min_cacheable_tokens == 1024
        assert config.cache_ttl_seconds == 600
        assert config.max_cache_ttl_seconds == 3600
        assert config.disable_for_streaming is True

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = CachingConfig(
            enabled=False,
            cache_backend="redis",
            min_cacheable_tokens=512,
            cache_ttl_seconds=300,
        )

        assert config.enabled is False
        assert config.cache_backend == "redis"
        assert config.min_cacheable_tokens == 512
        assert config.cache_ttl_seconds == 300

    def test_invalid_cache_backend(self):
        """Test validation of invalid cache backend."""
        with pytest.raises(ValueError, match="cache_backend must be one of"):
            CachingConfig(cache_backend="invalid")

    def test_invalid_eviction_policy(self):
        """Test validation of invalid eviction policy."""
        with pytest.raises(ValueError, match="eviction_policy must be one of"):
            CachingConfig(memory_cache=MemoryCacheConfig(eviction_policy="invalid"))

    def test_max_ttl_validation(self):
        """Test that max_cache_ttl_seconds must be >= cache_ttl_seconds."""
        with pytest.raises(ValueError, match="max_cache_ttl_seconds must be >="):
            CachingConfig(
                cache_ttl_seconds=600,
                max_cache_ttl_seconds=300,  # Less than cache_ttl_seconds
            )

    def test_valid_ttl_range(self):
        """Test valid TTL range configuration."""
        config = CachingConfig(
            cache_ttl_seconds=600,
            max_cache_ttl_seconds=1200,
        )

        assert config.cache_ttl_seconds == 600
        assert config.max_cache_ttl_seconds == 1200


class TestMemoryCacheConfig:
    """Tests for MemoryCacheConfig."""

    def test_default_memory_config(self):
        """Test default memory cache configuration."""
        config = MemoryCacheConfig()

        assert config.max_entries == 1000
        assert config.max_memory_mb == 512
        assert config.eviction_policy == "lru"

    def test_custom_memory_config(self):
        """Test custom memory cache configuration."""
        config = MemoryCacheConfig(
            max_entries=5000,
            max_memory_mb=1024,
            eviction_policy="lfu",
        )

        assert config.max_entries == 5000
        assert config.max_memory_mb == 1024
        assert config.eviction_policy == "lfu"


class TestRedisCacheConfig:
    """Tests for RedisCacheConfig."""

    def test_default_redis_config(self):
        """Test default Redis cache configuration."""
        config = RedisCacheConfig()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.connection_pool_size == 10

    def test_custom_redis_config(self):
        """Test custom Redis cache configuration."""
        config = RedisCacheConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            connection_pool_size=20,
        )

        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        assert config.connection_pool_size == 20


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_circuit_breaker_config(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 10
        assert config.recovery_timeout_seconds == 60

    def test_custom_circuit_breaker_config(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_seconds=30,
        )

        assert config.failure_threshold == 5
        assert config.recovery_timeout_seconds == 30


class TestPromptCachingMiddleware:
    """Tests for PromptCachingMiddleware."""

    @pytest.fixture
    def cache_store(self):
        """Create in-memory cache store for testing."""
        return MemoryCacheStore(max_entries=100, default_ttl=600)

    @pytest.fixture
    def config(self):
        """Create caching configuration for testing."""
        return CachingConfig(
            enabled=True,
            cache_backend="memory",
            min_cacheable_tokens=1024,
            cache_ttl_seconds=600,
        )

    @pytest.fixture
    def middleware(self, cache_store, config):
        """Create middleware instance for testing."""
        return PromptCachingMiddleware(cache_store=cache_store, config=config)

    @pytest.fixture
    def sample_request(self):
        """Create sample chat completion request."""
        return OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4",
            messages=[
                OpenAISystemMessageParam(
                    content="You are a helpful assistant. " * 200  # ~400 words = ~500 tokens
                ),
                OpenAIUserMessageParam(content="What is the capital of France?"),
            ],
            stream=False,
        )

    @pytest.fixture
    def sample_response(self):
        """Create sample chat completion response."""
        return OpenAIChatCompletion(
            id="chatcmpl-test123",
            created=1234567890,
            model="gpt-4",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(content="Paris"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=OpenAIChatCompletionUsage(
                prompt_tokens=1200,
                completion_tokens=10,
                total_tokens=1210,
            ),
        )

    async def test_middleware_initialization(self, middleware, config):
        """Test middleware initialization."""
        assert middleware.cache is not None
        assert middleware.config == config
        assert middleware.circuit_breaker is not None
        assert middleware.circuit_breaker.is_closed()

    async def test_caching_disabled(self, cache_store, sample_request, sample_response):
        """Test that middleware bypasses cache when disabled."""
        config = CachingConfig(enabled=False)
        middleware = PromptCachingMiddleware(cache_store=cache_store, config=config)

        execute_fn = AsyncMock(return_value=sample_response)

        response = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
        )

        assert response == sample_response
        execute_fn.assert_called_once()
        # Cache should not be accessed
        cache_size = await cache_store.size()
        assert cache_size == 0

    async def test_streaming_bypass(self, middleware, sample_request, sample_response):
        """Test that streaming requests bypass cache."""
        sample_request.stream = True
        execute_fn = AsyncMock(return_value=sample_response)

        response = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
        )

        assert response == sample_response
        execute_fn.assert_called_once()
        # Cache should not be accessed
        cache_size = await middleware.cache.size()
        assert cache_size == 0

    async def test_insufficient_messages(self, middleware, sample_response):
        """Test that requests with <2 messages bypass cache."""
        request = OpenAIChatCompletionRequestWithExtraBody(
            model="gpt-4",
            messages=[OpenAIUserMessageParam(content="Hello")],
            stream=False,
        )
        execute_fn = AsyncMock(return_value=sample_response)

        response = await middleware.process_chat_completion(
            request=request,
            execute_fn=execute_fn,
        )

        assert response == sample_response
        cache_size = await middleware.cache.size()
        assert cache_size == 0

    @patch("llama_stack.core.server.prompt_caching.count_tokens")
    async def test_below_token_threshold(self, mock_count_tokens, middleware, sample_request, sample_response):
        """Test that prompts below token threshold bypass cache."""
        # Mock token count below threshold
        mock_count_tokens.return_value = 500  # Below 1024 threshold

        execute_fn = AsyncMock(return_value=sample_response)

        response = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
        )

        assert response == sample_response
        cache_size = await middleware.cache.size()
        assert cache_size == 0

    @patch("llama_stack.core.server.prompt_caching.count_tokens")
    async def test_cache_miss_then_hit(self, mock_count_tokens, middleware, sample_request, sample_response):
        """Test cache miss followed by cache hit."""
        # Mock token count above threshold
        mock_count_tokens.return_value = 1200  # Above 1024 threshold

        execute_fn = AsyncMock(return_value=sample_response)

        # First request - cache miss
        response1 = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
            tenant_id="tenant1",
            user_id="user1",
        )

        assert response1 == sample_response
        execute_fn.assert_called_once()

        # Verify cache entry was stored
        cache_size = await middleware.cache.size()
        assert cache_size == 1

        # Second request - cache hit
        execute_fn.reset_mock()
        response2 = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
            tenant_id="tenant1",
            user_id="user1",
        )

        assert response2.id == sample_response.id
        # Verify cached_tokens was set
        assert response2.usage.prompt_tokens_details is not None
        assert response2.usage.prompt_tokens_details.cached_tokens == 1200

    @patch("llama_stack.core.server.prompt_caching.count_tokens")
    async def test_multi_tenant_isolation(self, mock_count_tokens, middleware, sample_request, sample_response):
        """Test that cache keys are isolated by tenant and user."""
        mock_count_tokens.return_value = 1200

        execute_fn = AsyncMock(return_value=sample_response)

        # Request from tenant1/user1
        await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
            tenant_id="tenant1",
            user_id="user1",
        )

        # Request from tenant2/user1 (different tenant)
        execute_fn.reset_mock()
        await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
            tenant_id="tenant2",
            user_id="user1",
        )

        # Should be cache miss (different tenant)
        assert execute_fn.call_count == 1

        # Verify 2 separate cache entries
        cache_size = await middleware.cache.size()
        assert cache_size == 2

    @patch("llama_stack.core.server.prompt_caching.count_tokens")
    async def test_cache_failures_graceful_degradation(
        self, mock_count_tokens, middleware, sample_request, sample_response
    ):
        """Test that cache failures don't block inference (graceful degradation)."""
        mock_count_tokens.return_value = 1200

        # Simulate cache failures
        middleware.cache.get = AsyncMock(side_effect=Exception("Cache backend failure"))
        middleware.cache.set = AsyncMock(side_effect=Exception("Cache backend failure"))

        execute_fn = AsyncMock(return_value=sample_response)

        # Make request despite cache failures
        response = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
        )

        # Verify inference still works (graceful degradation)
        assert response == sample_response
        execute_fn.assert_called_once()

        # Response should not have cached_tokens (cache miss with failure)
        assert (
            response.usage.prompt_tokens_details is None or response.usage.prompt_tokens_details.cached_tokens is None
        )

    async def test_cache_key_computation(self, middleware):
        """Test cache key computation."""
        messages = [
            OpenAISystemMessageParam(content="System message"),
            OpenAIUserMessageParam(content="User message"),
        ]

        # Same inputs should produce same key
        key1 = middleware._compute_cache_key(
            tenant_id="tenant1",
            user_id="user1",
            model="gpt-4",
            messages=messages,
        )
        key2 = middleware._compute_cache_key(
            tenant_id="tenant1",
            user_id="user1",
            model="gpt-4",
            messages=messages,
        )

        assert key1 == key2
        assert key1.startswith("prompt_cache:")

        # Different tenant should produce different key
        key3 = middleware._compute_cache_key(
            tenant_id="tenant2",
            user_id="user1",
            model="gpt-4",
            messages=messages,
        )

        assert key3 != key1

        # Different model should produce different key
        key4 = middleware._compute_cache_key(
            tenant_id="tenant1",
            user_id="user1",
            model="gpt-3.5-turbo",
            messages=messages,
        )

        assert key4 != key1

    async def test_clear_cache(self, middleware, cache_store):
        """Test cache clearing."""
        # Add some entries
        await cache_store.set(key="key1", value="value1")
        await cache_store.set(key="key2", value="value2")

        cache_size = await cache_store.size()
        assert cache_size == 2

        # Clear cache
        await middleware.clear_cache()

        cache_size = await cache_store.size()
        assert cache_size == 0

    async def test_get_cache_stats(self, middleware):
        """Test cache statistics retrieval."""
        stats = await middleware.get_cache_stats()

        assert "size" in stats
        assert "circuit_breaker_state" in stats
        assert "enabled" in stats
        assert "backend" in stats
        assert stats["enabled"] is True
        assert stats["backend"] == "memory"
        assert stats["circuit_breaker_state"] == "CLOSED"

    @patch("llama_stack.core.server.prompt_caching.count_tokens")
    async def test_token_counting_failure(self, mock_count_tokens, middleware, sample_request, sample_response):
        """Test graceful handling of token counting failures."""
        # Simulate token counting failure
        mock_count_tokens.side_effect = Exception("Tokenizer error")

        execute_fn = AsyncMock(return_value=sample_response)

        # Should continue without caching
        response = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
        )

        assert response == sample_response
        execute_fn.assert_called_once()

        # Cache should be empty
        cache_size = await middleware.cache.size()
        assert cache_size == 0

    @patch("llama_stack.core.server.prompt_caching.count_tokens")
    async def test_response_without_usage(self, mock_count_tokens, middleware, sample_request):
        """Test handling response without usage field."""
        mock_count_tokens.return_value = 1200

        # Response without usage
        response_no_usage = OpenAIChatCompletion(
            id="chatcmpl-test456",
            created=1234567890,
            model="gpt-4",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(content="Test"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=None,  # No usage field
        )

        execute_fn = AsyncMock(return_value=response_no_usage)

        # First request - cache miss
        await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
        )

        # Second request - cache hit
        execute_fn.reset_mock()
        execute_fn.return_value = response_no_usage
        response = await middleware.process_chat_completion(
            request=sample_request,
            execute_fn=execute_fn,
        )

        # Should create usage field with cached_tokens
        assert response.usage is not None
        assert response.usage.prompt_tokens_details is not None
        assert response.usage.prompt_tokens_details.cached_tokens == 1200
