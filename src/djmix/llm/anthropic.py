"""Anthropic (Claude) provider."""

from __future__ import annotations

import os

from djmix.llm.base import LLMUnavailable

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, timeout: float = 120.0):
        self.model = model or os.environ.get("DJMIX_ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    def call_llm(self, prompt: str, system: str | None = None) -> str:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable(f"the anthropic package is not installed: {exc}") from exc

        client = anthropic.Anthropic(timeout=self.timeout)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            # Server-side refusal fallbacks: if a safety classifier declines,
            # the request is routed to another model rather than failing. If the
            # installed SDK or endpoint doesn't know the beta, fall back to the
            # plain call rather than losing the provider entirely.
            try:
                response = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            except (TypeError, AttributeError, anthropic.BadRequestError):
                response = client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(f"anthropic request failed ({exc.status_code}): {exc}") from exc
        except anthropic.APIError as exc:
            raise LLMUnavailable(f"anthropic request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMUnavailable(f"anthropic declined the request: {details}")

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise LLMUnavailable("anthropic returned an empty response")
        return text
