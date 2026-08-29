"""The one interface every LLM provider implements.

Deliberately minimal -- `call_llm(prompt) -> str`. The planner never sees a
provider-specific type, so swapping Groq for Claude (or for the offline mock)
touches nothing outside this package.
"""

from __future__ import annotations

from typing import Protocol


class LLMUnavailable(RuntimeError):
    """The provider could not answer: no key, network failure, rate limit,
    timeout, empty body, or a refusal. Always recoverable -- the caller falls
    back to the deterministic rule-based planner."""


class LLMProvider(Protocol):
    name: str

    def call_llm(self, prompt: str, system: str | None = None) -> str: ...
