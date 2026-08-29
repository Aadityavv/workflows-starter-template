"""Core data shapes.

These models are the only contract between the three modules (audio, planning,
render).  Keeping the contract here is what lets analysis and rendering be
unit-tested with no LLM in the loop, and lets LLM providers be swapped without
touching audio code.

The single most important design decision in this file is what `Transition`
*cannot* express: it has no `bpm` and no `key` field, and `extra="forbid"`.
A planner that tries to emit a measurement is rejected structurally, before any
semantic validation runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANALYZER_VERSION = "1"

TransitionType = Literal["crossfade", "cut", "fade_out_in"]


class Segment(BaseModel):
    """One labelled structural section, measured by recurrence-matrix clustering."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    label: int

    @model_validator(mode="after")
    def _ordered(self) -> Segment:
        if self.end <= self.start:
            raise ValueError("segment end must be after start")
        return self


class TrackAnalysis(BaseModel):
    """Everything measured from one waveform. No value here is ever LLM-supplied."""

    model_config = ConfigDict(extra="forbid")

    track_id: str
    source_path: str
    content_hash: str
    analyzer_version: str = ANALYZER_VERSION

    duration_sec: float = Field(gt=0)

    # Tempo. `bpm_alternatives` records half/double-time candidates rather than
    # silently "correcting" the detected value -- octave errors are the single
    # most common beat-tracking failure and hiding them loses information.
    bpm: float = Field(gt=0)
    bpm_confidence: float = Field(ge=0, le=1)
    bpm_alternatives: list[float] = Field(default_factory=list)
    beat_times: list[float] = Field(default_factory=list)
    downbeat_times: list[float] = Field(default_factory=list)

    key: str
    key_confidence: float = Field(ge=0, le=1)
    camelot: str

    energetic_section: tuple[float, float]
    chorus_estimate: tuple[float, float]
    segments: list[Segment] = Field(default_factory=list)

    energy_curve: list[float] = Field(default_factory=list)
    energy_curve_hz: float = Field(default=10.0, gt=0)
    energy_mean_db: float = 0.0
    energy_score: float = Field(default=0.0, ge=0, le=1)

    mood_tags: dict[str, float] = Field(default_factory=dict)

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_period(self) -> float:
        """One bar, assuming 4/4 -- the meter the downbeat estimator also assumes."""
        return 4 * self.beat_period

    def public_summary(self) -> dict:
        """The spec's per-track analysis shape, for CLI display and LLM input."""
        return {
            "track_id": self.track_id,
            "duration_sec": round(self.duration_sec, 1),
            "bpm": round(self.bpm, 1),
            "key": self.key,
            "energetic_section": [round(v, 1) for v in self.energetic_section],
            "chorus_estimate": [round(v, 1) for v in self.chorus_estimate],
            "mood_tags": {k: round(v, 2) for k, v in self.mood_tags.items()},
        }


class Transition(BaseModel):
    """How one track hands off to the next.

    No bpm/key fields exist by design, and extra keys are forbidden: this is the
    structural half of the guarantee that measurements never come from an LLM.

    `strict=True` matters more than it looks: language models routinely emit
    `"out_at": "183.2"` as a string, and lax coercion would quietly accept it.
    `allow_inf_nan=False` rejects `NaN`/`Infinity`, which JSON does not define
    but several models emit anyway.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    type: TransitionType = "crossfade"
    out_at: float = Field(ge=0, description="Seconds into the outgoing track where the fade begins")
    len_sec: float = Field(gt=0, description="Crossfade length in seconds")
    into: str = Field(description="track_id of the incoming track")
    in_at: float = Field(ge=0, description="Seconds into the incoming track where it is heard from")


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    track_id: str
    start_at: float | None = Field(
        default=None,
        ge=0,
        description="Where this track's contribution starts. Defaults to in_at of the "
        "incoming transition, or 0.0 for the first track.",
    )
    transition: Transition | None = None


class MixPlan(BaseModel):
    """An ordered sequence of tracks plus the transitions between them.

    Unvalidated. Nothing renders from this type -- see ValidatedMixPlan.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    occasion: str
    plan: list[PlanStep] = Field(min_length=1)
    notes: str | None = None


class ValidatedMixPlan(BaseModel):
    """A MixPlan that has passed schema, integrity, and numeric-provenance checks.

    `djmix.render.engine.render()` accepts only this type. The validator in
    `djmix.planning.validation` is the only place that constructs it, so there is
    no code path from raw model output to audio.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: MixPlan
    planner: str
    validation_notes: list[str] = Field(default_factory=list)

    @property
    def occasion(self) -> str:
        return self.plan.occasion

    @property
    def steps(self) -> list[PlanStep]:
        return self.plan.plan


class PlanValidationError(ValueError):
    """Raised when a candidate plan fails any gate. Carries every reason at once,
    so the LLM retry can be given the full list rather than one error at a time."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))
