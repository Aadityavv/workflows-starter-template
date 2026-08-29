"""Plan validation: the gate between any planner and the rendering engine.

The product rule this enforces: BPM, key, and timestamps are *measured*, never
supplied by a language model. Prompt wording cannot enforce that. Four stages
here can, cheapest first:

  Stage 0  forbidden-field scan  -- no measurement-shaped key may appear anywhere
  Stage 1  strict schema         -- types, ranges, no extra keys, no NaN/strings
  Stage 2  graph integrity       -- the plan is a valid chain over the selection
  Stage 3  numeric provenance    -- every number traces to a measured value

Only `validate_plan` constructs a `ValidatedMixPlan`, and only a
`ValidatedMixPlan` can be rendered, so there is no path from raw model output to
audio that skips this file.

A note on how strong Stage 3 actually is, because it is easy to overstate: a
timestamp is accepted if it lands on the measured beat grid. At 128 BPM with a
35 ms window, roughly 15% of a timeline is within tolerance of *some* beat, so a
randomly invented number passes about one time in seven. That is why strict mode
(downbeats only, ~4%) is the default for the LLM path, why the LLM is handed an
explicit menu of legal times so a compliant model produces exact matches, and
why `test_provenance_coverage_is_sparse` measures this acceptance rate as a test
rather than leaving it as an assumption. Stage 0/1 do the heavy lifting against
invented BPM and key values; Stage 3 defends the one remaining numeric surface.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any

from djmix.models import (
    MixPlan,
    PlanValidationError,
    TrackAnalysis,
    Transition,
    ValidatedMixPlan,
)

# Tolerances. The analysis hop is 512 samples at 22050 Hz = 23.2 ms, so measured
# beat times are quantised to that grid; 35 ms is about 1.5 frames -- tight
# enough to be meaningful, loose enough not to reject a correctly snapped value
# that round-tripped through JSON.
EXACT_TOL = 0.005
SNAP_TOL = 0.035
OFFSET_TOL = 0.050
LENGTH_TOL = 0.050
# How far from a structural landmark a transition point may sit. This bounds
# how far a planner can wander from anything musically meaningful.
MAX_ANCHOR_WINDOW = 30.0
MIN_FADE_SEC = 2.0
MAX_FADE_SEC = 32.0
MIN_PLAY_SEC = 15.0

# Any key shaped like a measurement, anywhere in the payload, is refused before
# parsing. This exists so a model that tries to "helpfully" annotate its plan
# gets a precise error rather than a confusing schema message.
FORBIDDEN_KEY = re.compile(
    r"(bpm|tempo|\bkey\b|camelot|pitch|semitone|energy|loudness|lufs|duration|"
    r"chroma|beat_times|downbeat)",
    re.IGNORECASE,
)

STANDARD_FADE_LENGTHS = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
BAR_MULTIPLES = (0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass
class Provenance:
    """Why one number in the plan was accepted."""

    path: str
    value: float
    rule: str
    anchor_name: str
    anchor_value: float
    delta: float


@dataclass
class Violation:
    code: str
    path: str
    value: Any
    message: str
    nearest: str | None = None

    def __str__(self) -> str:
        base = f"[{self.code}] {self.path}: {self.message}"
        return f"{base} (nearest legal: {self.nearest})" if self.nearest else base


@dataclass
class ValidationReport:
    violations: list[Violation] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def messages(self) -> list[str]:
        return [str(v) for v in self.violations]


def _semantic_anchors(a: TrackAnalysis) -> dict[str, float]:
    """Musically meaningful landmarks -- the only anchors an offset may start from."""
    anchors: dict[str, float] = {
        "zero": 0.0,
        "duration": a.duration_sec,
        "energetic_section[0]": a.energetic_section[0],
        "energetic_section[1]": a.energetic_section[1],
        "chorus_estimate[0]": a.chorus_estimate[0],
        "chorus_estimate[1]": a.chorus_estimate[1],
    }
    for i, seg in enumerate(a.segments):
        anchors[f"segments[{i}].start"] = seg.start
        anchors[f"segments[{i}].end"] = seg.end
    return anchors


def _nearest(sorted_values: list[float], value: float) -> tuple[float, float] | None:
    """Closest entry in a sorted list, with its signed distance."""
    if not sorted_values:
        return None
    i = bisect.bisect_left(sorted_values, value)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(sorted_values):
            candidate = sorted_values[j]
            if best is None or abs(candidate - value) < abs(best - value):
                best = candidate
    return (best, value - best) if best is not None else None


def _closest_anchor(value: float, a: TrackAnalysis) -> tuple[str, float, float]:
    name, anchor = min(_semantic_anchors(a).items(), key=lambda kv: abs(kv[1] - value))
    return name, anchor, value - anchor


def explain_time(
    value: float, a: TrackAnalysis, fade_len: float | None, strict: bool, path: str
) -> Provenance | None:
    """Find a derivation of `value` from track `a`'s measured values, or None.

    A timestamp is traceable if EITHER

      * it lands exactly on a structural landmark (within 5 ms), or
      * it lands on the measured beat grid (within 35 ms) AND sits within
        MAX_ANCHOR_WINDOW of a structural landmark.

    Both conditions in the second rule are required, and that conjunction is the
    whole design. Grid-locking alone would let a planner pick any beat in the
    track; anchor-proximity alone would accept any of the thousands of instants
    within half a minute of a chorus. Together they say: "a real transition
    point, near something that actually happens in the music."

    An earlier draft also allowed free offsets from anchors -- any multiple of
    0.5 s up to 10 s. Measured against the fixtures that accepted 18-43% of
    uniformly random timestamps, which made the stage close to decorative. The
    rule below measures 3-15% instead, and `coverage_fraction` keeps that
    honest as a test rather than a claim.

    This does mean the spec's illustrative `in_at = energetic_section[0] - 5`
    is accepted only when that value also lands on the beat grid. That is
    deliberate and stricter: a transition point that is not on a beat is one
    the renderer would have to move anyway.
    """
    for name, anchor in _semantic_anchors(a).items():
        delta = value - anchor
        if abs(delta) <= EXACT_TOL:
            return Provenance(path, value, "EXACT", name, anchor, delta)

    anchor_name, anchor_value, anchor_delta = _closest_anchor(value, a)
    if abs(anchor_delta) > MAX_ANCHOR_WINDOW:
        return None

    grids: list[tuple[str, list[float]]] = [("downbeat", a.downbeat_times)]
    if not strict:
        grids.append(("beat", a.beat_times))
    for grid_name, grid in grids:
        hit = _nearest(grid, value)
        if hit is not None and abs(hit[1]) <= SNAP_TOL:
            return Provenance(
                path,
                value,
                f"SNAP_{grid_name.upper()}+ANCHOR:{anchor_name}",
                f"{grid_name}_grid",
                hit[0],
                hit[1],
            )
    return None


def explain_length(value: float, a: TrackAnalysis, path: str) -> Provenance | None:
    """A fade length must be a whole-bar multiple of the measured tempo, or a
    conventional round length."""
    bar = a.bar_period
    for n in BAR_MULTIPLES:
        target = n * bar
        if abs(value - target) <= LENGTH_TOL:
            return Provenance(
                path, value, "BAR_LENGTH", f"{n}bar@{a.bpm:.2f}bpm", target, value - target
            )
    for target in STANDARD_FADE_LENGTHS:
        if abs(value - target) <= LENGTH_TOL:
            return Provenance(path, value, "ROUND_LENGTH", f"{target:g}s", target, value - target)
    return None


def _nearest_bar(value: float, a: TrackAnalysis) -> float:
    """The whole-bar fade length closest to `value` -- what a repair turn needs."""
    return round(value / a.bar_period) * a.bar_period


def _nearest_report(value: float, a: TrackAnalysis) -> str:
    candidates: list[tuple[float, str]] = [(v, k) for k, v in _semantic_anchors(a).items()]
    hit = _nearest(a.beat_times, value)
    if hit:
        candidates.append((hit[0], "beat_grid"))
    hit = _nearest(a.downbeat_times, value)
    if hit:
        candidates.append((hit[0], "downbeat_grid"))
    if not candidates:
        return "no anchors available"
    best, name = min(candidates, key=lambda c: abs(c[0] - value))
    return f"{best:.3f} ({name}, {value - best:+.3f}s away)"


def scan_forbidden_fields(payload: Any, path: str = "$") -> list[Violation]:
    """Stage 0: reject any measurement-shaped key, at any depth."""
    violations: list[Violation] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            child = f"{path}.{k}"
            if FORBIDDEN_KEY.search(str(k)):
                violations.append(
                    Violation(
                        "FORBIDDEN_FIELD",
                        child,
                        v,
                        f"{k!r} is a measured quantity and must never appear in a plan; "
                        "measurements come from the analyzer, not the planner",
                    )
                )
            violations.extend(scan_forbidden_fields(v, child))
    elif isinstance(payload, list):
        for i, v in enumerate(payload):
            violations.extend(scan_forbidden_fields(v, f"{path}[{i}]"))
    return violations


def _check_graph(
    plan: MixPlan, analyses: dict[str, TrackAnalysis], selected: set[str] | None
) -> list[Violation]:
    violations: list[Violation] = []
    ids = [step.track_id for step in plan.plan]

    for i, track_id in enumerate(ids):
        if track_id not in analyses:
            violations.append(
                Violation(
                    "UNKNOWN_TRACK",
                    f"plan[{i}].track_id",
                    track_id,
                    "references a track with no analysis",
                )
            )
    if len(ids) != len(set(ids)):
        duplicates = sorted({t for t in ids if ids.count(t) > 1})
        violations.append(
            Violation(
                "DUPLICATE_TRACK",
                "plan",
                duplicates,
                "a track may appear in many different mixes, but not twice within "
                "one plan: the `into` chain and the renderer's segment bookkeeping "
                "would both become ambiguous",
            )
        )
    if selected is not None and set(ids) != selected:
        missing = sorted(selected - set(ids))
        extra = sorted(set(ids) - selected)
        violations.append(
            Violation(
                "SELECTION_MISMATCH",
                "plan",
                {"missing": missing, "extra": extra},
                "plan must cover exactly the selected tracks",
            )
        )
    if len(plan.plan) < 2:
        violations.append(
            Violation("TOO_SHORT", "plan", len(plan.plan), "a mix needs at least two tracks")
        )

    for i, step in enumerate(plan.plan):
        is_last = i == len(plan.plan) - 1
        path = f"plan[{i}].transition"
        if is_last:
            if step.transition is not None:
                violations.append(
                    Violation(
                        "TRAILING_TRANSITION",
                        path,
                        step.transition.model_dump(),
                        "the final track must not transition into anything",
                    )
                )
            continue
        if step.transition is None:
            violations.append(
                Violation(
                    "MISSING_TRANSITION", path, None, "every track but the last needs a transition"
                )
            )
            continue
        transition = step.transition
        if transition.into != ids[i + 1]:
            violations.append(
                Violation(
                    "BROKEN_CHAIN",
                    f"{path}.into",
                    transition.into,
                    f"must equal the next track in the plan ({ids[i + 1]})",
                )
            )
        violations.extend(_check_ranges(i, step.track_id, transition, analyses))
    return violations


def _check_ranges(
    i: int, track_id: str, transition: Transition, analyses: dict[str, TrackAnalysis]
) -> list[Violation]:
    violations: list[Violation] = []
    path = f"plan[{i}].transition"
    outgoing = analyses.get(track_id)
    incoming = analyses.get(transition.into)
    if outgoing is None or incoming is None:
        return violations

    if not (MIN_FADE_SEC <= transition.len_sec <= MAX_FADE_SEC):
        violations.append(
            Violation(
                "FADE_OUT_OF_RANGE",
                f"{path}.len_sec",
                transition.len_sec,
                f"must be between {MIN_FADE_SEC} and {MAX_FADE_SEC} seconds",
            )
        )
    if transition.out_at + transition.len_sec > outgoing.duration_sec + EXACT_TOL:
        violations.append(
            Violation(
                "OUT_PAST_END",
                f"{path}.out_at",
                transition.out_at,
                f"fade would run past the end of the outgoing track ({outgoing.duration_sec:.2f}s)",
            )
        )
    if transition.in_at + transition.len_sec > incoming.duration_sec + EXACT_TOL:
        violations.append(
            Violation(
                "IN_PAST_END",
                f"{path}.in_at",
                transition.in_at,
                f"fade would run past the end of the incoming track ({incoming.duration_sec:.2f}s)",
            )
        )
    return violations


def _check_play_time(plan: MixPlan, analyses: dict[str, TrackAnalysis]) -> list[Violation]:
    """Each track must actually be heard: entering and immediately leaving is
    not a mix, and it is a failure mode planners fall into when chasing a
    duration target."""
    violations: list[Violation] = []
    entry = 0.0
    for i, step in enumerate(plan.plan):
        if step.transition is None:
            break
        played = step.transition.out_at - entry
        if played < MIN_PLAY_SEC:
            violations.append(
                Violation(
                    "INSUFFICIENT_PLAY",
                    f"plan[{i}]",
                    round(played, 2),
                    f"track is only audible for {played:.1f}s; "
                    f"at least {MIN_PLAY_SEC}s is required",
                )
            )
        entry = step.transition.in_at
    return violations


def check_provenance(
    plan: MixPlan, analyses: dict[str, TrackAnalysis], strict: bool
) -> tuple[list[Violation], list[Provenance]]:
    """Stage 3. Note the per-field track scoping, which is not cosmetic:
    `in_at` belongs to the INCOMING track's timeline, so it is checked against
    the analysis of `transition.into`, not of the step it is written on. A model
    supplying a perfectly valid downbeat of the wrong track is a common and
    otherwise invisible error."""
    violations: list[Violation] = []
    provenance: list[Provenance] = []

    for i, step in enumerate(plan.plan):
        transition = step.transition
        if transition is None:
            continue
        outgoing = analyses.get(step.track_id)
        incoming = analyses.get(transition.into)
        if outgoing is None or incoming is None:
            continue

        checks = (
            ("out_at", transition.out_at, outgoing),
            ("in_at", transition.in_at, incoming),
        )
        for field_name, value, track in checks:
            path = f"plan[{i}].transition.{field_name}"
            result = explain_time(value, track, transition.len_sec, strict, path)
            if result is None:
                violations.append(
                    Violation(
                        "UNTRACEABLE_TIME",
                        path,
                        value,
                        f"cannot be derived from any measured value of track {track.track_id[:8]}",
                        nearest=_nearest_report(value, track),
                    )
                )
            else:
                provenance.append(result)

        path = f"plan[{i}].transition.len_sec"
        result = explain_length(transition.len_sec, outgoing, path)
        if result is None:
            violations.append(
                Violation(
                    "UNTRACEABLE_LENGTH",
                    path,
                    transition.len_sec,
                    f"is neither a whole-bar multiple at the measured "
                    f"{outgoing.bpm:.2f} BPM nor a conventional fade length",
                    nearest=f"{_nearest_bar(transition.len_sec, outgoing):.3f}",
                )
            )
        else:
            provenance.append(result)

    return violations, provenance


def validate_plan(
    payload: Any,
    analyses: dict[str, TrackAnalysis],
    planner: str,
    selected: set[str] | None = None,
    strict: bool = False,
) -> tuple[ValidatedMixPlan | None, ValidationReport]:
    """Run all four stages. Returns (validated_plan_or_None, report).

    Stages short-circuit: a payload that fails the forbidden-field scan or the
    schema is not worth graph-checking, and the error list stays readable.
    """
    report = ValidationReport()

    if isinstance(payload, MixPlan):
        plan = payload
    else:
        report.violations.extend(scan_forbidden_fields(payload))
        if report.violations:
            return None, report
        try:
            plan = MixPlan.model_validate(payload)
        except Exception as exc:
            report.violations.append(
                Violation("SCHEMA_INVALID", "$", None, str(exc).replace("\n", " ")[:500])
            )
            return None, report

    report.violations.extend(_check_graph(plan, analyses, selected))
    if report.violations:
        return None, report

    report.violations.extend(_check_play_time(plan, analyses))
    violations, provenance = check_provenance(plan, analyses, strict)
    report.violations.extend(violations)
    report.provenance.extend(provenance)

    if report.violations:
        return None, report

    notes = [f"{p.path}: {p.rule} <- {p.anchor_name}" for p in report.provenance]
    return ValidatedMixPlan(plan=plan, planner=planner, validation_notes=notes), report


def validate_or_raise(
    payload: Any,
    analyses: dict[str, TrackAnalysis],
    planner: str,
    selected: set[str] | None = None,
    strict: bool = False,
) -> ValidatedMixPlan:
    validated, report = validate_plan(payload, analyses, planner, selected, strict)
    if validated is None:
        raise PlanValidationError(report.messages())
    return validated


def coverage_fraction(a: TrackAnalysis, strict: bool, samples: int = 5000, seed: int = 0) -> float:
    """Fraction of uniformly random timestamps this track would accept.

    This is the validator's discriminating power, measured rather than assumed.
    A test asserts it stays low; if a future tolerance change made almost
    everything traceable, that test fails instead of the layer silently becoming
    decorative.
    """
    import random

    rng = random.Random(seed)
    accepted = 0
    for _ in range(samples):
        t = rng.uniform(0.0, a.duration_sec)
        if explain_time(t, a, None, strict, "probe") is not None:
            accepted += 1
    return accepted / samples


__all__ = [
    "Provenance",
    "Violation",
    "ValidationReport",
    "validate_plan",
    "validate_or_raise",
    "explain_time",
    "explain_length",
    "scan_forbidden_fields",
    "coverage_fraction",
    "MIN_FADE_SEC",
    "MAX_FADE_SEC",
    "MIN_PLAY_SEC",
]
