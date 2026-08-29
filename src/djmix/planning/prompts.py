"""Prompt construction for the LLM planner.

The central design choice: the model is given an explicit MENU of legal
timestamps, computed in Python from the measured beat grid, and told to copy
values from it. That converts its task from generating numbers into selecting
them -- which is the single biggest lever on first-pass validity, because a
class of error you cannot prompt away is one you can design away.

The validator does not trust the menu. It independently re-derives provenance
from the analysis, so a bug in the menu builder cannot launder an illegal
number through validation.
"""

from __future__ import annotations

import json

from djmix.models import TrackAnalysis
from djmix.planning.transitions import (
    entry_candidates,
    exit_candidates,
    fade_length,
    fade_length_candidates,
)

SYSTEM_PROMPT = """\
You are a DJ set designer. You are given MEASURED audio analysis facts for a \
set of tracks, and you decide the running order and how each track hands off to \
the next.

Do not invent BPM, key, or timestamp values. Use only the numbers provided in \
the input JSON. Every measurement you see was computed from the actual audio \
waveform; you have no way to measure anything yourself, and a number you \
produce that was not given to you is wrong by definition.

Rules, all of which are enforced by a validator that will reject your answer:

1. Output ONLY a single JSON object. No prose, no markdown code fences.
2. The object has exactly two keys: "occasion" (string) and "plan" (array).
3. Each plan item has "track_id". Every item except the last also has a \
"transition" object with exactly these keys: "type", "out_at", "len_sec", \
"into", "in_at".
4. "type" is one of "crossfade", "cut", "fade_out_in".
5. Every timestamp MUST be copied verbatim from the menus given for that track.
6. "out_at" must come from that item's own track's "allowed_out_at" list.
7. "in_at" must come from the "allowed_in_at" list of the track named in \
"into" -- it is a position in the INCOMING track, not the outgoing one.
8. "len_sec" must be one of the outgoing track's "allowed_fade_lengths".
9. "into" must equal the next item's "track_id".
10. No track may appear twice. The last item must have no "transition" key.
11. Do not output BPM, key, energy, or any other measurement. There is no \
field for them and any such key will be rejected.

Order the tracks so the energy shape suits the occasion, tempos stay close \
between neighbours, and keys are harmonically compatible where possible.\
"""


def track_facts(a: TrackAnalysis, max_options: int = 8) -> dict:
    """The measured facts and legal choices for one track."""
    fade = fade_length(a)
    return {
        "track_id": a.track_id,
        "duration_sec": round(a.duration_sec, 2),
        "bpm": round(a.bpm, 1),
        "key": a.key,
        "camelot": a.camelot,
        "energy": round(a.energy_score, 3),
        "mood_tags": {k: round(v, 2) for k, v in a.mood_tags.items()},
        "energetic_section": [round(v, 2) for v in a.energetic_section],
        "chorus_estimate": [round(v, 2) for v in a.chorus_estimate],
        "allowed_out_at": exit_candidates(a, fade, limit=max_options),
        "allowed_in_at": entry_candidates(a, fade, limit=max_options),
        "allowed_fade_lengths": fade_length_candidates(a),
    }


def build_prompt(
    analyses: list[TrackAnalysis],
    occasion: str,
    user_prompt: str | None = None,
    target_minutes: float | None = None,
    previous_errors: list[str] | None = None,
) -> str:
    facts = [track_facts(a) for a in analyses]
    lines = [f"OCCASION: {occasion}"]
    if user_prompt:
        lines.append(f"USER REQUEST: {user_prompt}")
    if target_minutes:
        lines.append(f"TARGET LENGTH: about {target_minutes:.0f} minutes")
    lines.append(f"\nTRACKS ({len(facts)} available, use all of them):\n")
    for fact in facts:
        lines.append(json.dumps(fact, indent=2))
    lines.append('\nReturn the mix plan as a single JSON object with keys "occasion" and "plan".')

    if previous_errors:
        # Feed the concrete violations back. The validator's messages include
        # the nearest legal value, which is exactly what a repair turn needs.
        lines.append(
            "\nYour previous answer was REJECTED by the validator for these reasons:\n"
            + "\n".join(f"  - {error}" for error in previous_errors)
            + "\n\nReturn corrected JSON only, choosing values from the menus above."
        )
    return "\n".join(lines)
