# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Prompt caching middleware for OpenAI-compatible chat completions.

This module implements automatic prompt caching for chat completion requests
following OpenAI's caching behavior:
- Caches prompt prefixes ≥1024 tokens (configurable)
- Uses SHA-256 hashing for cache keys (FIPS-compliant)
- Supports multi-tenant isolation (tenant_id + user_id in cache key)
- Circuit breaker pattern for cache backend failures
- Bypasses caching for streaming requests
- Updates response usage with cached_tokens count

Example:
    config = CachingConfig(
        enabled=True,
        cache_backend="memory",
        min_cacheable_tokens=1024,
        cache_ttl_seconds=600,
    )
    caching = PromptCachingMiddleware(cache_store=cache, config=config)

    # Wrap chat completion
    response = await caching.process_chat_completion(
        request=request,
        execute_fn=lambda req: provider.openai_chat_completion(req),
        tenant_id="tenant1",
        user_id="user1",
    )
"""

import asyncio
import hashlib
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, field_validator

from llama_stack.log import get_logger
from llama_stack.providers.utils.cache.cache_store import CacheStore, CircuitBreaker
from llama_stack.providers.utils.inference.tokenization import count_tokens
from llama_stack_api.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIChatCompletionUsage,
    OpenAIChatCompletionUsagePromptTokensDetails,
    OpenAIMessageParam,
)

logger = get_logger(__name__)


class MemoryCacheConfig(BaseModel):
    """Configuration for in-memory cache backend.

    Attributes:
        max_entries: Maximum number of cache entries to store
        max_memory_mb: Maximum memory usage in MB (None = unlimited)
        eviction_policy: Eviction policy (lru, lfu, ttl-only)
    """

    max_entries: int = Field(default=1000, ge=1, description="Maximum number of cache entries")
    max_memory_mb: int | None = Field(default=512, ge=1, description="Maximum memory usage in MB")
    eviction_policy: str = Field(default="lru", description="Eviction policy: lru, lfu, or ttl-only")

    @field_validator("eviction_policy")
    @classmethod
    def validate_eviction_policy(cls, v: str) -> str:
        allowed = {"lru", "lfu", "ttl-only"}
        if v not in allowed:
            raise ValueError(f"eviction_policy must be one of {allowed}, got: {v}")
        return v


class RedisCacheConfig(BaseModel):
    """Configuration for Redis cache backend.

    Attributes:
        host: Redis server hostname
        port: Redis server port
        db: Redis database number
        password: Redis password (optional, supports env var expansion)
        connection_pool_size: Connection pool size
        timeout_ms: Operation timeout in milliseconds
    """

    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis server port")
    db: int = Field(default=0, ge=0, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password (optional)")
    connection_pool_size: int = Field(default=10, ge=1, description="Connection pool size")
    timeout_ms: int = Field(default=100, ge=1, description="Operation timeout in milliseconds")


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker pattern.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit
        recovery_timeout_seconds: Time to wait before attempting recovery
    """

    failure_threshold: int = Field(default=10, ge=1, description="Consecutive failures before opening circuit")
    recovery_timeout_seconds: int = Field(default=60, ge=1, description="Recovery timeout in seconds")


class CachingConfig(BaseModel):
    """Configuration for prompt caching middleware.

    Attributes:
        enabled: Enable/disable caching
        cache_backend: Cache backend type (memory or redis)
        min_cacheable_tokens: Minimum tokens required for caching
        cache_ttl_seconds: Cache entry TTL in seconds
        max_cache_ttl_seconds: Maximum cache entry TTL
        cache_prefix_increment: Token increment for prefix caching
        memory_cache: Memory cache configuration
        redis_cache: Redis cache configuration
        circuit_breaker: Circuit breaker configuration
        disable_for_streaming: Disable caching for streaming requests
    """

    enabled: bool = Field(default=True, description="Enable/disable caching")
    cache_backend: str = Field(default="memory", description="Cache backend: memory or redis")
    min_cacheable_tokens: int = Field(default=1024, ge=1, description="Minimum tokens for caching")
    cache_ttl_seconds: int = Field(default=600, ge=1, description="Cache TTL in seconds")
    max_cache_ttl_seconds: int = Field(default=3600, ge=1, description="Maximum cache TTL")
    cache_prefix_increment: int = Field(default=128, ge=1, description="Token increment for prefix caching")
    memory_cache: MemoryCacheConfig = Field(default_factory=MemoryCacheConfig)
    redis_cache: RedisCacheConfig = Field(default_factory=RedisCacheConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    disable_for_streaming: bool = Field(default=True, description="Disable caching for streaming requests")

    @field_validator("cache_backend")
    @classmethod
    def validate_cache_backend(cls, v: str) -> str:
        allowed = {"memory", "redis"}
        if v not in allowed:
            raise ValueError(f"cache_backend must be one of {allowed}, got: {v}")
        return v

    @field_validator("max_cache_ttl_seconds")
    @classmethod
    def validate_max_ttl(cls, v: int, info) -> int:
        if "cache_ttl_seconds" in info.data and v < info.data["cache_ttl_seconds"]:
            raise ValueError("max_cache_ttl_seconds must be >= cache_ttl_seconds")
        return v


class PromptCachingMiddleware:
    """Middleware for caching prompt prefixes in chat completion requests.

    This middleware automatically caches prompt prefixes to reduce latency and costs
    following OpenAI's prompt caching behavior. It integrates with the inference router
    to intercept chat completion requests and check/update cache entries.

    Attributes:
        cache: Cache store instance (memory or Redis)
        config: Caching configuration
        circuit_breaker: Circuit breaker for cache backend failures
    """

    def __init__(self, cache_store: CacheStore, config: CachingConfig) -> None:
        """Initialize prompt caching middleware.

        Args:
            cache_store: Cache store implementation (MemoryCacheStore or RedisCacheStore)
            config: Caching configuration

        Raises:
            ValueError: If config validation fails
        """
        self.cache = cache_store
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker.failure_threshold,
            recovery_timeout=config.circuit_breaker.recovery_timeout_seconds,
        )
        logger.info(
            f"PromptCachingMiddleware initialized: backend={config.cache_backend}, "
            f"min_tokens={config.min_cacheable_tokens}, ttl={config.cache_ttl_seconds}s"
        )

    async def process_chat_completion(
        self,
        request: OpenAIChatCompletionRequestWithExtraBody,
        execute_fn: Callable[[OpenAIChatCompletionRequestWithExtraBody], Any],
        tenant_id: str = "default",
        user_id: str = "default",
    ) -> OpenAIChatCompletion:  # type: ignore[return]
        """Process chat completion request with caching.

        This method implements the core caching logic:
        1. Check if caching is enabled and applicable
        2. Extract and tokenize prompt prefix
        3. Compute cache key with multi-tenant isolation
        4. Check cache for existing entry
        5. Execute inference
        6. Update response with cached_tokens
        7. Store cache entry for future requests

        Args:
            request: Chat completion request
            execute_fn: Function to execute inference (e.g., provider.openai_chat_completion)
            tenant_id: Tenant identifier for multi-tenant isolation
            user_id: User identifier for multi-tenant isolation

        Returns:
            Chat completion response with updated usage statistics

        Note:
            - Streaming requests bypass caching if disable_for_streaming=True
            - Cache failures are logged but do not block inference
            - Circuit breaker prevents cascade failures
        """
        # 0. Check if caching is enabled
        if not self.config.enabled:
            return await execute_fn(request)  # type: ignore[no-any-return]

        # 1. Skip caching for streaming requests
        if request.stream and self.config.disable_for_streaming:
            logger.debug("Bypassing cache for streaming request")
            return await execute_fn(request)  # type: ignore[no-any-return]

        # 2. Extract prefix (all messages except last) and count tokens
        if not request.messages or len(request.messages) < 2:
            # Need at least 2 messages for prefix caching (system + user)
            logger.debug(f"Insufficient messages for caching: {len(request.messages) if request.messages else 0}")
            return await execute_fn(request)  # type: ignore[no-any-return]

        prefix_messages = request.messages[:-1]

        # 3. Count tokens in prefix
        try:
            # Convert Pydantic models to dicts for tokenization
            prefix_messages_dicts = [
                msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else msg for msg in prefix_messages
            ]
            token_count = count_tokens(
                messages=prefix_messages_dicts,  # type: ignore[arg-type]
                model=request.model,
                exact=True,  # Use exact tokenization when possible
            )
        except Exception as e:
            logger.warning(f"Failed to count tokens for caching: {e}")
            return await execute_fn(request)  # type: ignore[no-any-return]

        # 4. Check if prefix is cacheable
        if token_count < self.config.min_cacheable_tokens:
            logger.debug(f"Prefix too short for caching: {token_count} < {self.config.min_cacheable_tokens} tokens")
            return await execute_fn(request)  # type: ignore[no-any-return]

        # 5. Compute cache key
        cache_key = self._compute_cache_key(
            tenant_id=tenant_id,
            user_id=user_id,
            model=request.model,
            messages=prefix_messages,
        )

        # 6. Check cache (with circuit breaker and timeout)
        cache_hit = False
        cached_entry = None

        if self.circuit_breaker.is_closed():
            try:
                cached_entry = await asyncio.wait_for(
                    self.cache.get(key=cache_key),
                    timeout=self.config.redis_cache.timeout_ms / 1000.0,  # Convert ms to seconds
                )
                if cached_entry is not None:
                    cache_hit = True
                    logger.debug(f"Cache hit: {cache_key[:16]}... ({token_count} tokens)")
                    self.circuit_breaker.record_success()
                else:
                    logger.debug(f"Cache miss: {cache_key[:16]}... ({token_count} tokens)")
            except TimeoutError:
                logger.warning(f"Cache lookup timeout for key: {cache_key[:16]}...")
                self.circuit_breaker.record_failure()
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
                self.circuit_breaker.record_failure()
        else:
            logger.debug("Circuit breaker open, skipping cache lookup")

        # 7. Execute inference
        response = await execute_fn(request)

        # 8. Update response with cached_tokens if cache hit
        if cache_hit and isinstance(response, OpenAIChatCompletion):
            # Ensure usage field exists
            if response.usage is None:
                response.usage = OpenAIChatCompletionUsage(
                    prompt_tokens=token_count,
                    completion_tokens=0,
                    total_tokens=token_count,
                )

            # Create or update prompt_tokens_details
            if response.usage.prompt_tokens_details is None:
                response.usage.prompt_tokens_details = OpenAIChatCompletionUsagePromptTokensDetails()

            response.usage.prompt_tokens_details.cached_tokens = token_count

            logger.info(
                f"Updated response with cached_tokens: {token_count} "
                f"(prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens})"
            )

        # 9. Store cache entry for future requests (cache miss only)
        if not cache_hit and self.circuit_breaker.is_closed():
            try:
                cache_value = {
                    "token_count": token_count,
                    "model": request.model,
                    "message_count": len(prefix_messages),
                }
                await self.cache.set(
                    key=cache_key,
                    value=cache_value,
                    ttl=self.config.cache_ttl_seconds,
                )
                logger.debug(f"Stored cache entry: {cache_key[:16]}... ({token_count} tokens)")
                self.circuit_breaker.record_success()
            except Exception as e:
                logger.warning(f"Failed to store cache entry: {e}")
                self.circuit_breaker.record_failure()

        return response  # type: ignore[no-any-return]

    def _compute_cache_key(
        self,
        tenant_id: str,
        user_id: str,
        model: str,
        messages: list[OpenAIMessageParam],
    ) -> str:
        """Compute FIPS-compliant cache key with multi-tenant isolation.

        The cache key is computed as SHA-256 hash of:
            tenant_id:user_id:model:serialized_messages

        Messages are serialized as JSON with sorted keys for consistency.

        Args:
            tenant_id: Tenant identifier for isolation
            user_id: User identifier for isolation
            model: Model name to prevent cross-model cache pollution
            messages: Message list (prefix only)

        Returns:
            Cache key in format: "prompt_cache:<sha256_hash>"

        Note:
            - Uses SHA-256 (FIPS-compliant, not MD5/SHA1)
            - Messages serialized with sorted keys for deterministic hashing
            - Includes tenant_id and user_id for multi-tenant isolation
        """
        # Serialize messages with sorted keys for consistency
        # Convert Pydantic models to dicts for serialization
        serializable_messages: list[dict[str, Any]] = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                serializable_messages.append(msg.model_dump(exclude_none=True))  # type: ignore[arg-type]
            else:
                serializable_messages.append(msg)  # type: ignore[arg-type]

        serialized_messages = json.dumps(
            serializable_messages,
            sort_keys=True,
            separators=(",", ":"),  # Compact JSON
        )

        # Construct cache key components
        key_components = [
            tenant_id,
            user_id,
            model,
            serialized_messages,
        ]

        # Join with delimiter and hash (FIPS-compliant SHA-256)
        key_string = ":".join(key_components)
        cache_key_hash = hashlib.sha256(key_string.encode("utf-8")).hexdigest()

        return f"prompt_cache:{cache_key_hash}"

    async def clear_cache(self) -> None:
        """Clear all cached entries.

        This method clears the entire cache. Use with caution in production.

        Note:
            Circuit breaker state is not affected by cache clearing.
        """
        try:
            await self.cache.clear()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
                - size: Number of entries in cache
                - circuit_breaker_state: Circuit breaker state (CLOSED, OPEN, HALF_OPEN)
                - enabled: Whether caching is enabled
                - backend: Cache backend type

        Note:
            Some backends may not support all statistics.
        """
        try:
            cache_size = await self.cache.size()
        except Exception as e:
            logger.warning(f"Failed to get cache size: {e}")
            cache_size = -1

        return {
            "size": cache_size,
            "circuit_breaker_state": self.circuit_breaker.state,
            "enabled": self.config.enabled,
            "backend": self.config.cache_backend,
            "min_cacheable_tokens": self.config.min_cacheable_tokens,
            "cache_ttl_seconds": self.config.cache_ttl_seconds,
        }
