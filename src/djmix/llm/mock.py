"""Deterministic offline provider.

Two jobs: let the whole LLM code path run in CI with no key and no network, and
let tests drive the failure modes on purpose -- malformed JSON, fenced JSON,
truncation, and hallucinated numbers that must be rejected by the validator.
"""

from __future__ import annotations

import json
import re

from djmix.llm.base import LLMUnavailable

MODES = (
    "valid",
    "fenced",
    "prose",
    "malformed",
    "truncated",
    "empty",
    "hallucinated_time",
    "hallucinated_field",
    "wrong_chain",
    "unavailable",
)


class MockProvider:
    name = "mock"

    def __init__(self, mode: str = "valid", canned: str | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown mock mode {mode!r}; known: {MODES}")
        self.mode = mode
        self.canned = canned
        self.calls: list[str] = []

    def call_llm(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self.canned is not None:
            return self.canned
        if self.mode == "unavailable":
            raise LLMUnavailable("mock provider configured to fail")
        if self.mode == "empty":
            return ""
        if self.mode == "malformed":
            return '{"occasion": "party", "plan": [ {"track_id": '

        plan = self._plan_from_prompt(prompt)
        if self.mode == "hallucinated_time":
            # A plausible-looking but entirely invented timestamp.
            plan["plan"][0]["transition"]["out_at"] = 137.913
        if self.mode == "hallucinated_field":
            plan["plan"][0]["transition"]["bpm"] = 128.0
        if self.mode == "wrong_chain":
            plan["plan"][0]["transition"]["into"] = "not-a-real-track-id"

        body = json.dumps(plan, indent=2)
        if self.mode == "fenced":
            return f"```json\n{body}\n```"
        if self.mode == "prose":
            return f"Sure! Here is the mix plan you asked for:\n\n```json\n{body}\n```\n\nEnjoy."
        if self.mode == "truncated":
            return body[: int(len(body) * 0.7)]
        return body

    @staticmethod
    def _plan_from_prompt(prompt: str) -> dict:
        """Build a valid plan by reading the menus out of the prompt.

        The mock deliberately behaves like a compliant model: it only ever picks
        values from the `allowed_*` lists it was given, so a passing test proves
        the pipeline works rather than proving the mock got lucky.
        """
        # Scan with a real JSON decoder rather than a regex: the fact blocks
        # contain nested objects, so any non-greedy brace pattern stops at the
        # first inner closing brace.
        decoder = json.JSONDecoder()
        tracks = []
        for match in re.finditer(r'\{\s*"track_id"', prompt):
            try:
                obj, _ = decoder.raw_decode(prompt[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "allowed_out_at" in obj:
                tracks.append(obj)
        if len(tracks) < 2:
            raise LLMUnavailable("mock could not read the track facts from the prompt")

        occasion_match = re.search(r"OCCASION:\s*(.+)", prompt)
        occasion = occasion_match.group(1).strip() if occasion_match else "road trip"

        steps = []
        for i, track in enumerate(tracks):
            if i == len(tracks) - 1:
                steps.append({"track_id": track["track_id"]})
                break
            nxt = tracks[i + 1]
            steps.append(
                {
                    "track_id": track["track_id"],
                    "transition": {
                        "type": "crossfade",
                        "out_at": track["allowed_out_at"][-1],
                        "len_sec": track["allowed_fade_lengths"][0],
                        "into": nxt["track_id"],
                        "in_at": nxt["allowed_in_at"][0],
                    },
                }
            )
        return {"occasion": occasion, "plan": steps}
