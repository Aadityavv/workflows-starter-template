"""Groq provider (the default)."""

from __future__ import annotations

import os

from djmix.llm.base import LLMUnavailable

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider:
    name = "groq"

    def __init__(self, model: str | None = None, timeout: float = 60.0):
        self.model = model or os.environ.get("DJMIX_GROQ_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    def call_llm(self, prompt: str, system: str | None = None) -> str:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMUnavailable("GROQ_API_KEY is not set")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMUnavailable(f"the groq package is not installed: {exc}") from exc

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            client = Groq(api_key=api_key, timeout=self.timeout)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                # Groq speaks the OpenAI schema and supports a JSON-object mode,
                # which removes a whole class of "here is your plan:" preambles.
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMUnavailable(f"groq request failed: {exc}") from exc

        if not response.choices:
            raise LLMUnavailable("groq returned no choices")
        return response.choices[0].message.content or ""
