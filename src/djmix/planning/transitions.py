"""Choosing where a transition happens, from the measured beat grid.

Shared by the rule planner and by the LLM prompt builder, which offers the model
a menu of these same candidate times. Because every value produced here is read
off a measured grid, anything built from them is provenance-traceable by
construction -- the validator still re-derives it independently, so a bug in
this file cannot launder an illegal number through validation.
"""

from __future__ import annotations

import bisect

from djmix.models import TrackAnalysis, Transition
from djmix.planning.validation import MAX_FADE_SEC, MIN_FADE_SEC

DEFAULT_FADE_BARS = 2
MIN_LEAD_IN_SEC = 4.0
TAIL_MARGIN_SEC = 2.0


def _grid(a: TrackAnalysis) -> list[float]:
    """Prefer downbeats; fall back to beats when the downbeat phase is shaky.

    Low downbeat confidence is common on half-time material, and snapping to a
    confidently-wrong downbeat is worse than snapping to a merely-correct beat.
    """
    if len(a.downbeat_times) >= 4:
        return a.downbeat_times
    return a.beat_times


def snap_before(a: TrackAnalysis, t: float) -> float:
    grid = _grid(a)
    if not grid:
        return t
    i = bisect.bisect_right(grid, t) - 1
    return float(grid[max(0, i)])


def snap_after(a: TrackAnalysis, t: float) -> float:
    grid = _grid(a)
    if not grid:
        return t
    i = bisect.bisect_left(grid, t)
    return float(grid[min(i, len(grid) - 1)])


def fade_length(a: TrackAnalysis, bars: int = DEFAULT_FADE_BARS) -> float:
    """A whole number of bars at the measured tempo, clamped to sane limits."""
    length = bars * a.bar_period
    return round(min(max(length, MIN_FADE_SEC), MAX_FADE_SEC), 3)


def exit_candidates(a: TrackAnalysis, fade_len: float, limit: int = 8) -> list[float]:
    """Plausible points to start fading OUT of this track.

    Anchored on the end of the chorus and the end of the peak-energy section --
    where a DJ would leave -- then snapped to the grid.
    """
    latest = a.duration_sec - fade_len - TAIL_MARGIN_SEC
    raw = [a.chorus_estimate[1], a.energetic_section[1], a.duration_sec * 0.75]
    out: list[float] = []
    for value in raw:
        snapped = snap_before(a, min(value, latest))
        if MIN_FADE_SEC < snapped <= latest:
            out.append(round(snapped, 3))
    grid = _grid(a)
    for t in grid:
        if len(out) >= limit:
            break
        if a.duration_sec * 0.4 <= t <= latest:
            out.append(round(float(t), 3))
    return sorted(dict.fromkeys(out))[:limit]


def entry_candidates(a: TrackAnalysis, fade_len: float, limit: int = 8) -> list[float]:
    """Plausible points to start the incoming track from.

    Anchored so that the fade lands just before the track's strongest section,
    which is what makes a transition feel intentional rather than arbitrary.
    """
    lead_in = max(fade_len, MIN_LEAD_IN_SEC)
    raw = [
        a.energetic_section[0] - lead_in,
        a.chorus_estimate[0] - lead_in,
        0.0,
    ]
    out: list[float] = []
    for value in raw:
        snapped = snap_after(a, max(0.0, value))
        if 0.0 <= snapped <= a.duration_sec - fade_len - TAIL_MARGIN_SEC:
            out.append(round(snapped, 3))
    grid = _grid(a)
    for t in grid:
        if len(out) >= limit:
            break
        if 0.0 <= t <= min(a.duration_sec * 0.5, a.duration_sec - fade_len - TAIL_MARGIN_SEC):
            out.append(round(float(t), 3))
    return sorted(dict.fromkeys(out))[:limit]


def fade_length_candidates(a: TrackAnalysis) -> list[float]:
    lengths = {fade_length(a, bars) for bars in (1, 2, 4)}
    lengths |= {8.0, 16.0}
    return sorted(v for v in lengths if MIN_FADE_SEC <= v <= MAX_FADE_SEC)


def relative_bpm_gap(a: TrackAnalysis, b: TrackAnalysis) -> float:
    """Smallest relative tempo difference, allowing half- and double-time.

    An 87 BPM reading and a 174 BPM reading describe the same groove, so they
    must not be scored as a huge jump.
    """
    return min(abs(m * b.bpm - a.bpm) / a.bpm for m in (0.5, 1.0, 2.0))


def choose_transition(
    a: TrackAnalysis,
    b: TrackAnalysis,
    entry_of_a: float,
    bars: int = DEFAULT_FADE_BARS,
    min_play_sec: float = 20.0,
) -> Transition:
    """Build a transition from A into B using only measured values.

    Rather than taking the first candidate off a list, this aims at the point a
    DJ would actually pick -- leave at the end of A's chorus, arrive so that B's
    strongest section starts just after the fade completes -- and then snaps
    that intention onto the measured grid. Picking `entry_candidates()[0]` would
    put every track's entry at 0.0, which defeats the whole point of trimming to
    the strongest section.
    """
    fade_len = fade_length(a, bars)

    # Leave A at the end of its chorus, falling back to the end of its peak
    # section, but never before it has been audible for min_play_sec.
    latest_out = a.duration_sec - fade_len - TAIL_MARGIN_SEC
    earliest_out = min(entry_of_a + min_play_sec, max(latest_out, 0.0))
    ideal_out = (
        a.chorus_estimate[1] if a.chorus_estimate[1] > earliest_out else a.energetic_section[1]
    )
    ideal_out = min(max(ideal_out, earliest_out), latest_out)
    out_at = snap_before(a, ideal_out)
    if out_at < earliest_out:
        out_at = snap_after(a, earliest_out)
    out_at = min(max(out_at, 0.0), max(latest_out, 0.0))

    # Arrive in B so that the fade finishes right as its strongest section lands.
    latest_in = max(0.0, b.duration_sec - fade_len - TAIL_MARGIN_SEC)
    ideal_in = max(0.0, b.energetic_section[0] - fade_len)
    in_at = snap_after(b, min(ideal_in, latest_in))
    if in_at > latest_in:
        in_at = snap_before(b, latest_in)
    in_at = min(max(in_at, 0.0), latest_in)

    gap = relative_bpm_gap(a, b)
    # Beyond ~12% the stretch needed to beat-match is audible, so don't pretend:
    # use a longer, non-beatmatched blend instead of mangling the audio.
    kind = "crossfade" if gap <= 0.12 else "fade_out_in"

    return Transition(
        type=kind,
        out_at=round(float(out_at), 3),
        len_sec=fade_len,
        into=b.track_id,
        in_at=round(float(in_at), 3),
    )


def plan_segment_seconds(a: TrackAnalysis, start_at: float, out_at: float | None) -> float:
    """How long this track is audible in the mix (before crossfade overlap)."""
    end = out_at if out_at is not None else a.duration_sec
    return max(0.0, end - start_at)
