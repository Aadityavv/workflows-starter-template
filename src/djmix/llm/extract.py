"""Defensive extraction of a JSON object from model output.

Models wrap JSON in fences, prepend "Sure! Here you go:", and occasionally get
cut off mid-object. Recovering from those is worth doing. Guessing at output
that is *structurally* wrong -- Python dict literals, single-quoted keys -- is
not: a model emitting those is malfunctioning, and coercing them is how invalid
plans get laundered into the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JSONExtractionError(ValueError):
    pass


def _balanced_object(text: str) -> str | None:
    """First complete top-level {...}, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_truncated(text: str) -> str | None:
    """Close a response that was cut off mid-object.

    Only structural repair: drop a dangling partial entry, then close the open
    brackets. If it still does not parse, give up rather than improvise.
    """
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]
    if fragment.count('"') % 2:  # cut inside a string -- unrecoverable
        return None
    fragment = re.sub(r",\s*(\"[^\"]*\"\s*:?\s*)?$", "", fragment.rstrip())
    stack: list[str] = []
    in_string, escaped = False, False
    for ch in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    closing = "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return fragment + closing


def extract_json(text: str) -> tuple[dict[str, Any], bool]:
    """Return (parsed_object, was_repaired) or raise JSONExtractionError."""
    if not text or not text.strip():
        raise JSONExtractionError("model returned an empty response")

    candidates: list[str] = []
    for match in FENCE.findall(text):
        candidates.append(match.strip())
    candidates.append(text.strip())

    for candidate in candidates:
        balanced = _balanced_object(candidate)
        for attempt in filter(None, (balanced, candidate)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed, False

    for candidate in candidates:
        repaired = _repair_truncated(candidate)
        if not repaired:
            continue
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, True

    raise JSONExtractionError(
        f"could not extract a JSON object from the response (first 200 chars: {text[:200]!r})"
    )
