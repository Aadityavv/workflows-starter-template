"""Provider selection."""

from __future__ import annotations

import os

from djmix.llm.base import LLMProvider

PROVIDERS = ("groq", "anthropic", "mock")
DEFAULT_PROVIDER = "groq"


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    choice = (name or os.environ.get("DJMIX_LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if choice == "groq":
        from djmix.llm.groq import GroqProvider

        return GroqProvider(**kwargs)
    if choice == "anthropic":
        from djmix.llm.anthropic import AnthropicProvider

        return AnthropicProvider(**kwargs)
    if choice == "mock":
        from djmix.llm.mock import MockProvider

        return MockProvider(**kwargs)
    raise ValueError(f"unknown LLM provider {choice!r}; known: {PROVIDERS}")
