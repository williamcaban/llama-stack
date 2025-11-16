# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for OpenAI chat completion usage schema with cached_tokens support.

These tests validate that the OpenAIChatCompletionUsage and related models
correctly support the cached_tokens field for prompt caching, while maintaining
backward compatibility with responses that don't include these fields.
"""

import json

import pytest
from pydantic import ValidationError

from llama_stack_api.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionUsage,
    OpenAIChatCompletionUsageCompletionTokensDetails,
    OpenAIChatCompletionUsagePromptTokensDetails,
    OpenAIChoice,
)
from llama_stack_api.inference import OpenAIAssistantMessageParam


class TestOpenAIChatCompletionUsagePromptTokensDetails:
    """Tests for OpenAIChatCompletionUsagePromptTokensDetails model."""

    def test_create_with_cached_tokens(self):
        """Test creating PromptTokensDetails with cached_tokens field."""
        details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=100)

        assert details.cached_tokens == 100

    def test_create_without_cached_tokens(self):
        """Test creating PromptTokensDetails without cached_tokens (defaults to None)."""
        details = OpenAIChatCompletionUsagePromptTokensDetails()

        assert details.cached_tokens is None

    def test_create_with_none_cached_tokens(self):
        """Test creating PromptTokensDetails with explicit None for cached_tokens."""
        details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=None)

        assert details.cached_tokens is None

    def test_create_with_zero_cached_tokens(self):
        """Test creating PromptTokensDetails with zero cached tokens."""
        details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=0)

        assert details.cached_tokens == 0

    def test_json_serialization_with_cached_tokens(self):
        """Test JSON serialization when cached_tokens is set."""
        details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=150)

        json_str = details.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["cached_tokens"] == 150

    def test_json_serialization_without_cached_tokens(self):
        """Test JSON serialization when cached_tokens is None."""
        details = OpenAIChatCompletionUsagePromptTokensDetails()

        json_str = details.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["cached_tokens"] is None

    def test_json_deserialization_with_cached_tokens(self):
        """Test JSON deserialization with cached_tokens field."""
        json_data = {"cached_tokens": 200}

        details = OpenAIChatCompletionUsagePromptTokensDetails(**json_data)

        assert details.cached_tokens == 200

    def test_json_deserialization_without_cached_tokens(self):
        """Test JSON deserialization without cached_tokens field (backward compatibility)."""
        json_data = {}  # Empty dict - all fields optional

        details = OpenAIChatCompletionUsagePromptTokensDetails(**json_data)

        assert details.cached_tokens is None


class TestOpenAIChatCompletionUsageCompletionTokensDetails:
    """Tests for OpenAIChatCompletionUsageCompletionTokensDetails model."""

    def test_create_with_reasoning_tokens(self):
        """Test creating CompletionTokensDetails with reasoning_tokens field."""
        details = OpenAIChatCompletionUsageCompletionTokensDetails(reasoning_tokens=50)

        assert details.reasoning_tokens == 50

    def test_create_without_reasoning_tokens(self):
        """Test creating CompletionTokensDetails without reasoning_tokens."""
        details = OpenAIChatCompletionUsageCompletionTokensDetails()

        assert details.reasoning_tokens is None


class TestOpenAIChatCompletionUsage:
    """Tests for OpenAIChatCompletionUsage model with prompt_tokens_details."""

    def test_create_basic_usage(self):
        """Test creating basic usage without details fields."""
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.prompt_tokens_details is None
        assert usage.completion_tokens_details is None

    def test_create_usage_with_prompt_tokens_details(self):
        """Test creating usage with prompt_tokens_details including cached_tokens."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=80)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=prompt_details,
        )

        assert usage.prompt_tokens == 100
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 80

    def test_create_usage_with_completion_tokens_details(self):
        """Test creating usage with completion_tokens_details including reasoning_tokens."""
        completion_details = OpenAIChatCompletionUsageCompletionTokensDetails(reasoning_tokens=30)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            completion_tokens_details=completion_details,
        )

        assert usage.completion_tokens == 50
        assert usage.completion_tokens_details is not None
        assert usage.completion_tokens_details.reasoning_tokens == 30

    def test_create_usage_with_both_details(self):
        """Test creating usage with both prompt and completion token details."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=80)
        completion_details = OpenAIChatCompletionUsageCompletionTokensDetails(reasoning_tokens=30)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=prompt_details,
            completion_tokens_details=completion_details,
        )

        assert usage.prompt_tokens_details.cached_tokens == 80
        assert usage.completion_tokens_details.reasoning_tokens == 30

    def test_json_serialization_with_prompt_tokens_details(self):
        """Test JSON serialization of usage with prompt_tokens_details."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=80)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=prompt_details,
        )

        json_str = usage.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["prompt_tokens"] == 100
        assert parsed["prompt_tokens_details"]["cached_tokens"] == 80

    def test_json_serialization_without_details(self):
        """Test JSON serialization of usage without details fields."""
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        json_str = usage.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["prompt_tokens"] == 100
        assert parsed["prompt_tokens_details"] is None
        assert parsed["completion_tokens_details"] is None

    def test_json_deserialization_with_prompt_tokens_details(self):
        """Test JSON deserialization with prompt_tokens_details field."""
        json_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80},
        }

        usage = OpenAIChatCompletionUsage(**json_data)

        assert usage.prompt_tokens == 100
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 80

    def test_json_deserialization_backward_compatibility(self):
        """Test backward compatibility: old JSON without details fields still works."""
        json_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        usage = OpenAIChatCompletionUsage(**json_data)

        assert usage.prompt_tokens == 100
        assert usage.prompt_tokens_details is None
        assert usage.completion_tokens_details is None

    def test_required_fields_validation(self):
        """Test that required fields (prompt_tokens, completion_tokens, total_tokens) are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatCompletionUsage()  # Missing required fields

        errors = exc_info.value.errors()
        required_fields = {error["loc"][0] for error in errors}
        assert "prompt_tokens" in required_fields
        assert "completion_tokens" in required_fields
        assert "total_tokens" in required_fields


class TestOpenAIChatCompletionWithUsage:
    """Integration tests for OpenAIChatCompletion with usage field."""

    def test_create_completion_with_cached_tokens(self):
        """Test creating a full chat completion response with cached_tokens."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=512)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=600,
            completion_tokens=100,
            total_tokens=700,
            prompt_tokens_details=prompt_details,
        )

        completion = OpenAIChatCompletion(
            id="chatcmpl-test123",
            created=1234567890,
            model="gpt-4",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(
                        content="This is a test response using cached tokens.",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=usage,
        )

        assert completion.usage is not None
        assert completion.usage.prompt_tokens == 600
        assert completion.usage.prompt_tokens_details is not None
        assert completion.usage.prompt_tokens_details.cached_tokens == 512

    def test_create_completion_without_cached_tokens(self):
        """Test creating a chat completion without cached_tokens (cache miss)."""
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=600,
            completion_tokens=100,
            total_tokens=700,
        )

        completion = OpenAIChatCompletion(
            id="chatcmpl-test456",
            created=1234567890,
            model="gpt-4",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(
                        content="This is a test response without caching.",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=usage,
        )

        assert completion.usage is not None
        assert completion.usage.prompt_tokens == 600
        assert completion.usage.prompt_tokens_details is None

    def test_create_completion_with_zero_cached_tokens(self):
        """Test creating a chat completion with zero cached tokens (explicit cache miss)."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=0)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=600,
            completion_tokens=100,
            total_tokens=700,
            prompt_tokens_details=prompt_details,
        )

        completion = OpenAIChatCompletion(
            id="chatcmpl-test789",
            created=1234567890,
            model="gpt-4",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(
                        content="Cache miss explicitly indicated.",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=usage,
        )

        assert completion.usage.prompt_tokens_details is not None
        assert completion.usage.prompt_tokens_details.cached_tokens == 0

    def test_json_round_trip_with_cached_tokens(self):
        """Test JSON serialization and deserialization round trip with cached_tokens."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=1024)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=1200,
            completion_tokens=200,
            total_tokens=1400,
            prompt_tokens_details=prompt_details,
        )

        completion = OpenAIChatCompletion(
            id="chatcmpl-roundtrip",
            created=1234567890,
            model="gpt-4",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(
                        content="Round trip test.",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=usage,
        )

        # Serialize to JSON
        json_str = completion.model_dump_json()
        json_data = json.loads(json_str)

        # Verify JSON structure
        assert json_data["usage"]["prompt_tokens"] == 1200
        assert json_data["usage"]["prompt_tokens_details"]["cached_tokens"] == 1024

        # Deserialize back to model
        restored = OpenAIChatCompletion(**json_data)

        # Verify restored object
        assert restored.usage.prompt_tokens == 1200
        assert restored.usage.prompt_tokens_details.cached_tokens == 1024

    def test_backward_compatibility_old_response(self):
        """Test that old API responses without prompt_tokens_details still work."""
        json_data = {
            "id": "chatcmpl-old",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Old response format.",
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
            },
        }

        completion = OpenAIChatCompletion(**json_data)

        assert completion.usage.prompt_tokens == 50
        assert completion.usage.prompt_tokens_details is None
        assert completion.usage.completion_tokens_details is None


class TestCachedTokensSemantics:
    """Tests for cached_tokens semantic edge cases."""

    def test_cached_tokens_cannot_exceed_prompt_tokens(self):
        """Test that cached_tokens can logically be up to prompt_tokens (not validated by schema)."""
        # Note: The schema doesn't enforce cached_tokens <= prompt_tokens,
        # but this test documents the expected semantic behavior
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=1000)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
            prompt_tokens_details=prompt_details,
        )

        assert usage.prompt_tokens_details.cached_tokens == usage.prompt_tokens

    def test_partial_cache_hit(self):
        """Test representing a partial cache hit (some tokens cached, some not)."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=800)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=1200,  # 800 cached + 400 uncached
            completion_tokens=100,
            total_tokens=1300,
            prompt_tokens_details=prompt_details,
        )

        assert usage.prompt_tokens_details.cached_tokens == 800
        uncached_tokens = usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens
        assert uncached_tokens == 400

    def test_full_cache_hit(self):
        """Test representing a full cache hit (all prompt tokens cached)."""
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=1200)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=1200,
            completion_tokens=100,
            total_tokens=1300,
            prompt_tokens_details=prompt_details,
        )

        assert usage.prompt_tokens_details.cached_tokens == usage.prompt_tokens

    def test_cache_miss(self):
        """Test representing a cache miss (zero or None cached_tokens)."""
        # Explicit zero
        prompt_details = OpenAIChatCompletionUsagePromptTokensDetails(cached_tokens=0)
        usage = OpenAIChatCompletionUsage(
            prompt_tokens=1200,
            completion_tokens=100,
            total_tokens=1300,
            prompt_tokens_details=prompt_details,
        )

        assert usage.prompt_tokens_details.cached_tokens == 0

        # Implicit None (field omitted)
        usage_no_cache = OpenAIChatCompletionUsage(
            prompt_tokens=1200,
            completion_tokens=100,
            total_tokens=1300,
        )

        assert usage_no_cache.prompt_tokens_details is None
