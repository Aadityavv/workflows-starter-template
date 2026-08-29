"""Deterministic rule-based planner.

This is the floor the system never drops below: it needs no network, no API key,
and no model. It is also the fallback whenever the LLM planner produces
something that fails validation, so it must always produce a plan that passes
validation itself -- a property test asserts exactly that over randomised pools.

Ordering is a beam search over an energy arc, with tempo and harmonic
compatibility as penalties. It is deterministic given the same input order.
"""

from __future__ import annotations

from djmix.config import OccasionProfile, get_occasion, occasion_from_prompt, planner_weights
from djmix.models import MixPlan, PlanStep, TrackAnalysis
from djmix.planning.base import MixRequest, PlanResult
from djmix.planning.camelot import camelot_distance
from djmix.planning.transitions import (
    choose_transition,
    plan_segment_seconds,
    relative_bpm_gap,
)
from djmix.planning.validation import validate_plan

BEAM_WIDTH = 8
BPM_FREE_GAP = 0.06
BPM_MAX_GAP = 0.12
AVERAGE_SLICE_SEC = 95.0


def pool_energy(tracks: list[TrackAnalysis]) -> dict[str, float]:
    """Min-max scale loudness across the candidate pool.

    `energy_score` on a TrackAnalysis is an absolute dB-derived figure, and on a
    consistently-mastered library every track lands within a few hundredths of
    every other -- which leaves the occasion's energy arc with nothing to
    discriminate on, and makes every occasion produce the same running order.
    Scaling within the pool restores the contrast the arc needs, and it is the
    right frame anyway: what matters is which of *these* tracks is the loud one.
    """
    if not tracks:
        return {}
    values = [a.energy_mean_db for a in tracks]
    low, high = min(values), max(values)
    if high - low < 1e-6:
        return {a.track_id: 0.5 for a in tracks}
    return {a.track_id: (a.energy_mean_db - low) / (high - low) for a in tracks}


def _mood_fit(a: TrackAnalysis, occasion: OccasionProfile) -> float:
    """How well the measured mood distribution matches what the occasion wants."""
    if not occasion.moods:
        return 0.5
    return sum(weight * a.mood_tags.get(mood, 0.0) for mood, weight in occasion.moods.items())


def _bpm_fit(a: TrackAnalysis, occasion: OccasionProfile) -> float:
    low, high = occasion.bpm_band
    for bpm in [a.bpm, *a.bpm_alternatives]:
        if low <= bpm <= high:
            return 1.0
    nearest = min(abs(a.bpm - low), abs(a.bpm - high))
    return float(max(0.0, 1.0 - nearest / 60.0))


def _bpm_penalty(a: TrackAnalysis, b: TrackAnalysis) -> float:
    gap = relative_bpm_gap(a, b)
    if gap <= BPM_FREE_GAP:
        return 0.0
    if gap <= BPM_MAX_GAP:
        return (gap - BPM_FREE_GAP) / (BPM_MAX_GAP - BPM_FREE_GAP)
    return 1.0 + (gap - BPM_MAX_GAP) * 4.0


def _key_penalty(a: TrackAnalysis, b: TrackAnalysis) -> float:
    # A key we are not confident in should not drive the ordering; a flat
    # middling penalty is more honest than trusting a coin-flip reading.
    if a.key_confidence < 0.05 or b.key_confidence < 0.05:
        return 0.4
    return camelot_distance(a.camelot, b.camelot)


def select_tracks(
    analyses: list[TrackAnalysis],
    occasion: OccasionProfile,
    target_minutes: float | None,
    max_tracks: int | None,
) -> list[TrackAnalysis]:
    """Pick which tracks are in, and how many.

    With a duration target, the count comes from the target divided by the
    typical trimmed slice length -- because each track contributes its strongest
    section rather than its full runtime.
    """
    ranked = sorted(
        analyses,
        key=lambda a: (-(0.6 * _mood_fit(a, occasion) + 0.4 * _bpm_fit(a, occasion)), a.track_id),
    )
    if target_minutes is not None:
        wanted = max(2, round(target_minutes * 60.0 / AVERAGE_SLICE_SEC))
    else:
        wanted = len(ranked)
    if max_tracks is not None:
        wanted = min(wanted, max_tracks)
    wanted = min(wanted, len(ranked))
    return ranked[:wanted]


def order_tracks(tracks: list[TrackAnalysis], occasion: OccasionProfile) -> list[TrackAnalysis]:
    """Beam search minimising deviation from the arc plus transition penalties."""
    energy = pool_energy(tracks)
    if len(tracks) <= 2:
        return sorted(tracks, key=lambda a: (energy[a.track_id], a.track_id))

    weights = planner_weights()
    total = len(tracks)
    by_id = {a.track_id: a for a in tracks}

    # (cost, ordered_ids)
    beams: list[tuple[float, list[str]]] = [(0.0, [])]
    for slot in range(total):
        target = occasion.target_energy(slot, total)
        candidates: list[tuple[float, list[str]]] = []
        for cost, chosen in beams:
            remaining = [t for t in tracks if t.track_id not in chosen]
            for track in remaining:
                step = weights.get("energy_fit", 1.0) * abs(energy[track.track_id] - target)
                step += weights.get("mood_fit", 0.7) * (1.0 - _mood_fit(track, occasion))
                if chosen:
                    previous = by_id[chosen[-1]]
                    step += weights.get("bpm_jump", 0.8) * _bpm_penalty(previous, track)
                    step += weights.get("key_clash", 0.6) * _key_penalty(previous, track)
                    # Tie the arc's slope to tempo direction. Without this the
                    # BPM-continuity penalty swamps the energy term and every
                    # occasion yields the same running order -- a workout mix
                    # would descend in tempo exactly like a wind-down mix. A
                    # rising arc should want tempo to rise.
                    slope = target - occasion.target_energy(max(slot - 1, 0), total)
                    tempo_delta = (track.bpm - previous.bpm) / max(previous.bpm, 1.0)
                    if slope * tempo_delta < 0:
                        step += weights.get("tempo_direction", 0.5) * min(abs(slope) * 4.0, 1.0)
                candidates.append((cost + step, [*chosen, track.track_id]))
        # Sort by cost then by the id sequence, so ties break deterministically.
        candidates.sort(key=lambda c: (c[0], c[1]))
        beams = candidates[:BEAM_WIDTH]

    best_ids = beams[0][1]
    return [by_id[i] for i in best_ids]


def plan_duration_sec(plan: MixPlan, analyses: dict[str, TrackAnalysis]) -> float:
    """Rendered length of a plan: audible segments minus crossfade overlaps."""
    total = 0.0
    for step in plan.plan:
        a = analyses[step.track_id]
        start = step.start_at or 0.0
        out_at = step.transition.out_at if step.transition else None
        total += plan_segment_seconds(a, start, out_at)
        if step.transition:
            total -= 0.0  # the fade is counted once, inside the outgoing segment
    return total


def build_plan(tracks: list[TrackAnalysis], occasion_name: str) -> MixPlan:
    steps: list[PlanStep] = []
    entry = 0.0
    for i, track in enumerate(tracks):
        if i == len(tracks) - 1:
            steps.append(PlanStep(track_id=track.track_id, start_at=round(entry, 3)))
            break
        transition = choose_transition(track, tracks[i + 1], entry)
        steps.append(
            PlanStep(track_id=track.track_id, start_at=round(entry, 3), transition=transition)
        )
        entry = transition.in_at
    return MixPlan(occasion=occasion_name, plan=steps)


class RulePlanner:
    name = "rule"

    def plan(self, request: MixRequest) -> PlanResult:
        occasion_name = request.occasion or (
            occasion_from_prompt(request.prompt) if request.prompt else None
        )
        occasion = get_occasion(occasion_name)

        analyses = {a.track_id: a for a in request.analyses}
        selected = select_tracks(
            request.analyses, occasion, request.target_minutes, request.max_tracks
        )
        if len(selected) < 2:
            raise ValueError("a mix needs at least two analysable tracks")

        ordered = order_tracks(selected, occasion)
        plan = build_plan(ordered, occasion.name)

        # Close the loop on the duration target: the first guess at track count
        # uses an assumed slice length, but the real slices come from each
        # track's measured structure, so re-measure and adjust.
        notes: list[str] = []
        if request.target_minutes is not None:
            target_sec = request.target_minutes * 60.0
            for _ in range(4):
                actual = plan_duration_sec(plan, analyses)
                if abs(actual - target_sec) <= 0.05 * target_sec:
                    break
                step_change = 1 if actual < target_sec else -1
                new_count = len(ordered) + step_change
                if not (2 <= new_count <= len(request.analyses)):
                    break
                selected = select_tracks(request.analyses, occasion, None, new_count)
                ordered = order_tracks(selected, occasion)
                plan = build_plan(ordered, occasion.name)
            actual = plan_duration_sec(plan, analyses)
            notes.append(f"target={target_sec / 60:.1f}min actual={actual / 60:.1f}min")
        validated, report = validate_plan(
            plan, analyses, planner=self.name, selected={a.track_id for a in ordered}
        )
        if validated is None:
            # The rule planner producing an invalid plan is a bug in this file,
            # not user error -- fail loudly rather than quietly shipping it.
            raise AssertionError(
                "rule planner produced a plan that fails validation: "
                + "; ".join(report.messages())
            )
        return PlanResult(
            plan=validated,
            planner=self.name,
            notes=[f"occasion={occasion.name}", f"tracks={len(ordered)}", *notes],
        )
