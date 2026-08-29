"""The planner interface both implementations satisfy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from djmix.models import TrackAnalysis, ValidatedMixPlan


@dataclass
class MixRequest:
    """What the user asked for. Measurements are not part of this -- they come
    from the analyses, never from the request."""

    analyses: list[TrackAnalysis]
    occasion: str | None = None
    prompt: str | None = None
    target_minutes: float | None = None
    max_tracks: int | None = None


@dataclass
class PlanResult:
    plan: ValidatedMixPlan
    planner: str
    notes: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    llm_attempts: int = 0


class Planner(Protocol):
    name: str

    def plan(self, request: MixRequest) -> PlanResult: ...
