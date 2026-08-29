"""LLM planner: retry/fallback state machine around a validated plan.

The model contributes ordering and transition style. It contributes no
measurements: everything numeric it is allowed to emit must trace back to the
analysis, and the validator -- not this file, and certainly not the prompt --
is what enforces that.

Strict provenance is ON by default here. The prompt hands the model an explicit
menu of legal times, so a compliant model produces exact matches; strict mode
then rejects anything that merely lands near a beat by luck.
"""

from __future__ import annotations

import logging

from djmix.config import get_occasion, occasion_from_prompt
from djmix.llm.base import LLMProvider, LLMUnavailable
from djmix.llm.extract import JSONExtractionError, extract_json
from djmix.planning.base import MixRequest, PlanResult
from djmix.planning.prompts import SYSTEM_PROMPT, build_prompt
from djmix.planning.rules import RulePlanner, select_tracks
from djmix.planning.validation import validate_plan

log = logging.getLogger(__name__)

DEFAULT_RETRIES = 1


class LLMPlanner:
    name = "llm"

    def __init__(
        self,
        provider: LLMProvider,
        retries: int = DEFAULT_RETRIES,
        strict: bool = True,
        allow_fallback: bool = True,
    ):
        self.provider = provider
        self.retries = retries
        self.strict = strict
        self.allow_fallback = allow_fallback

    def plan(self, request: MixRequest) -> PlanResult:
        occasion_name = request.occasion or (
            occasion_from_prompt(request.prompt) if request.prompt else None
        )
        occasion = get_occasion(occasion_name)

        # The model orders the tracks; it does not get to invent the shortlist,
        # and the duration target is honoured by the same measured logic the
        # rule planner uses.
        selected = select_tracks(
            request.analyses, occasion, request.target_minutes, request.max_tracks
        )
        if len(selected) < 2:
            raise ValueError("a mix needs at least two analysable tracks")

        analyses = {a.track_id: a for a in request.analyses}
        selected_ids = {a.track_id for a in selected}

        errors: list[str] = []
        attempts = 0
        notes: list[str] = [f"occasion={occasion.name}", f"provider={self.provider.name}"]

        for attempt in range(self.retries + 1):
            prompt = build_prompt(
                selected,
                occasion.name,
                request.prompt,
                request.target_minutes,
                previous_errors=errors or None,
            )
            attempts += 1
            try:
                raw = self.provider.call_llm(prompt, system=SYSTEM_PROMPT)
            except LLMUnavailable as exc:
                return self._fallback(request, f"provider unavailable: {exc}", attempts, notes)

            try:
                payload, repaired = extract_json(raw)
            except JSONExtractionError as exc:
                errors = [str(exc)]
                log.warning("LLM attempt %d: %s", attempt + 1, exc)
                continue

            if repaired:
                notes.append("response was truncated and structurally repaired")

            validated, report = validate_plan(
                payload,
                analyses,
                planner=self.name,
                selected=selected_ids,
                strict=self.strict,
            )
            if validated is not None:
                notes.append(f"validated {len(report.provenance)} numeric values")
                return PlanResult(
                    plan=validated, planner=self.name, notes=notes, llm_attempts=attempts
                )

            errors = report.messages()
            log.warning(
                "LLM attempt %d rejected by validator: %s", attempt + 1, "; ".join(errors[:3])
            )

        return self._fallback(
            request, "plan failed validation: " + "; ".join(errors[:5]), attempts, notes
        )

    def _fallback(
        self, request: MixRequest, reason: str, attempts: int, notes: list[str]
    ) -> PlanResult:
        if not self.allow_fallback:
            raise LLMUnavailable(f"LLM planning failed and fallback is disabled: {reason}")
        # Never silent. A quiet downgrade would hide exactly the failure this
        # whole design exists to catch.
        log.warning("falling back to the rule-based planner: %s", reason)
        result = RulePlanner().plan(request)
        return PlanResult(
            plan=result.plan,
            planner="rule",
            notes=[*notes, *result.notes, "fell back from the LLM planner"],
            fallback_reason=reason,
            llm_attempts=attempts,
        )
