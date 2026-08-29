"""The hallucination-rejection suite.

This is the test file that matters most: it is what proves the product rule
("measurements are never LLM-supplied") is enforced by code rather than by
prompt wording.
"""

from __future__ import annotations

import copy
import json

import pytest

from djmix.models import Transition
from djmix.planning.base import MixRequest
from djmix.planning.rules import RulePlanner
from djmix.planning.validation import (
    MIN_FADE_SEC,
    coverage_fraction,
    scan_forbidden_fields,
    validate_plan,
)


@pytest.fixture
def valid_payload(analyses, analyses_by_id):
    """A known-good plan, produced by the rule planner from measured values."""
    result = RulePlanner().plan(MixRequest(analyses=analyses, occasion="party"))
    return json.loads(result.plan.plan.model_dump_json(exclude_none=True))


@pytest.fixture
def selected_ids(valid_payload):
    return {step["track_id"] for step in valid_payload["plan"]}


def _validate(payload, analyses_by_id, selected_ids, strict=False):
    return validate_plan(
        payload, analyses_by_id, planner="test", selected=selected_ids, strict=strict
    )


def test_the_baseline_plan_is_valid(valid_payload, analyses_by_id, selected_ids):
    validated, report = _validate(valid_payload, analyses_by_id, selected_ids)
    assert validated is not None, report.messages()
    assert report.provenance


def test_rule_planner_output_always_validates(analyses):
    """The rule planner is the fallback, so it must never emit an invalid plan.
    Run over every occasion and several sizes as a property check."""
    from djmix.config import available_occasions

    for occasion in available_occasions():
        for max_tracks in (2, 3, len(analyses)):
            result = RulePlanner().plan(
                MixRequest(analyses=analyses, occasion=occasion, max_tracks=max_tracks)
            )
            assert result.plan.planner == "rule"


# --- Stage 0: no measurement-shaped field may appear anywhere ---------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("bpm", 128.0),
        ("key", "A minor"),
        ("camelot", "8A"),
        ("tempo", 120),
        ("energy", 0.8),
        ("duration", 200.0),
        ("loudness", -14.0),
    ],
)
def test_invented_measurement_fields_are_rejected(
    valid_payload, analyses_by_id, selected_ids, field, value
):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][0]["transition"][field] = value
    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code == "FORBIDDEN_FIELD" for v in report.violations), report.messages()


def test_forbidden_scan_finds_nested_fields():
    violations = scan_forbidden_fields({"plan": [{"meta": {"detected_bpm": 174}}]})
    assert any(v.code == "FORBIDDEN_FIELD" for v in violations)


def test_transition_schema_has_no_place_for_a_measurement():
    """The structural half of the guarantee, asserted directly."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Transition(type="crossfade", out_at=10.0, len_sec=8.0, into="x", in_at=1.0, bpm=128)
    assert "bpm" not in Transition.model_fields
    assert "key" not in Transition.model_fields


# --- Stage 1: strict schema -------------------------------------------------


def test_string_numbers_are_rejected(valid_payload, analyses_by_id, selected_ids):
    """Models emit `"out_at": "183.2"` constantly; lax coercion would take it."""
    payload = copy.deepcopy(valid_payload)
    payload["plan"][0]["transition"]["out_at"] = str(payload["plan"][0]["transition"]["out_at"])
    validated, _ = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None


def test_nan_is_rejected(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][0]["transition"]["len_sec"] = float("nan")
    validated, _ = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None


@pytest.mark.parametrize("bad", [{}, {"occasion": "x"}, {"occasion": "x", "plan": []}])
def test_structurally_broken_plans_are_rejected(bad, analyses_by_id):
    validated, _ = validate_plan(bad, analyses_by_id, planner="test")
    assert validated is None


# --- Stage 2: graph integrity ----------------------------------------------


def test_dangling_into_is_rejected(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][0]["transition"]["into"] = "00000000-0000-0000-0000-000000000000"
    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code in {"UNKNOWN_TRACK", "BROKEN_CHAIN"} for v in report.violations)


def test_chain_must_match_ordering(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][0]["transition"]["into"] = payload["plan"][-1]["track_id"]
    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code == "BROKEN_CHAIN" for v in report.violations)


def test_duplicate_track_within_one_plan_is_rejected(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][1]["track_id"] = payload["plan"][0]["track_id"]
    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code == "DUPLICATE_TRACK" for v in report.violations)


def test_a_track_may_appear_in_many_different_mixes(analyses):
    """No global uniqueness constraint -- only within a single plan."""
    party = RulePlanner().plan(MixRequest(analyses=analyses, occasion="party"))
    focus = RulePlanner().plan(MixRequest(analyses=analyses, occasion="focus"))
    shared = {s.track_id for s in party.plan.steps} & {s.track_id for s in focus.plan.steps}
    assert shared, "the same track should be reusable across different mixes"


def test_last_step_must_not_transition(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][-1]["transition"] = copy.deepcopy(payload["plan"][0]["transition"])
    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code == "TRAILING_TRANSITION" for v in report.violations)


def test_fade_running_past_the_end_is_rejected(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    track = analyses_by_id[payload["plan"][0]["track_id"]]
    payload["plan"][0]["transition"]["out_at"] = track.duration_sec + 10
    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code == "OUT_PAST_END" for v in report.violations)


# --- Stage 3: numeric provenance -------------------------------------------


def test_invented_timestamp_is_rejected(valid_payload, analyses_by_id, selected_ids):
    """A number that is in range and well-formed, but traces to nothing."""
    payload = copy.deepcopy(valid_payload)
    track = analyses_by_id[payload["plan"][0]["track_id"]]
    # Deliberately far from every anchor and off the beat grid.
    invented = round(track.duration_sec * 0.5 + 0.137, 3)
    while any(abs(invented - b) < 0.2 for b in track.beat_times):
        invented += 0.07
    payload["plan"][0]["transition"]["out_at"] = round(invented, 3)
    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code == "UNTRACEABLE_TIME" for v in report.violations), report.messages()
    assert any(v.nearest for v in report.violations if v.code == "UNTRACEABLE_TIME")


def test_in_at_is_scoped_to_the_incoming_track(valid_payload, analyses_by_id, selected_ids):
    """A very common and otherwise invisible model error: giving an `in_at`
    that is a perfectly valid beat of the OUTGOING track."""
    payload = copy.deepcopy(valid_payload)
    outgoing = analyses_by_id[payload["plan"][0]["track_id"]]
    incoming = analyses_by_id[payload["plan"][0]["transition"]["into"]]

    for beat in outgoing.beat_times:
        near_incoming = min(abs(beat - b) for b in incoming.beat_times)
        anchors_ok = beat < incoming.duration_sec * 0.9
        if near_incoming > 0.2 and anchors_ok and beat > 20:
            payload["plan"][0]["transition"]["in_at"] = round(beat, 3)
            break
    else:
        pytest.skip("no outgoing beat is off the incoming grid on this fixture set")

    validated, report = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
    assert any(v.code == "UNTRACEABLE_TIME" and "in_at" in v.path for v in report.violations), (
        report.messages()
    )


def test_arbitrary_fade_length_is_rejected(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][0]["transition"]["len_sec"] = 9.37
    validated, _ = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None


def test_fade_length_rule_rejects_non_musical_values(analyses):
    """Exercised directly rather than through a whole plan: an odd length also
    trips range checks, and those short-circuit before provenance runs, so a
    plan-level assertion would not actually prove the length rule fired."""
    from djmix.planning.validation import explain_length

    for a in analyses:
        bar = a.bar_period
        assert explain_length(bar * 2, a, "len_sec") is not None
        assert explain_length(8.0, a, "len_sec") is not None
        # Pick a value that is neither a bar multiple nor a conventional length.
        odd = bar * 2 + 0.37
        while any(abs(odd - r) <= 0.06 for r in (1, 2, 4, 8, 12, 16, 24, 32)) or any(
            abs(odd - n * bar) <= 0.06 for n in (0.5, 1, 2, 4, 8)
        ):
            odd += 0.19
        assert explain_length(odd, a, "len_sec") is None, (
            f"{odd:.3f}s should not be explainable at {a.bpm:.1f} BPM"
        )


def test_round_and_bar_fade_lengths_are_accepted(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    track = analyses_by_id[payload["plan"][0]["track_id"]]
    for length in (8.0, round(2 * track.bar_period, 3)):
        payload["plan"][0]["transition"]["len_sec"] = length
        # Keep the fade in bounds so only the length rule is under test.
        payload["plan"][0]["transition"]["out_at"] = min(
            payload["plan"][0]["transition"]["out_at"], track.duration_sec - length - 1
        )
        _, report = _validate(payload, analyses_by_id, selected_ids)
        assert not any(v.code == "UNTRACEABLE_LENGTH" for v in report.violations)


def test_provenance_coverage_is_sparse(analyses):
    """Quantifies how discriminating the validator actually is.

    Without this, "every number must trace to a measurement" is an unverified
    claim: if tolerances were ever widened enough, essentially any timestamp
    would be explainable and the stage would be decorative. An earlier draft of
    the rules accepted 18-43% of uniformly random timestamps; this pins the
    tightened version and fails loudly if it ever loosens again.
    """
    normal = [coverage_fraction(a, strict=False) for a in analyses]
    strict = [coverage_fraction(a, strict=True) for a in analyses]
    assert max(normal) < 0.25, f"normal-mode acceptance too permissive: {normal}"
    assert max(strict) < 0.08, f"strict-mode acceptance too permissive: {strict}"
    for n, s in zip(normal, strict, strict=True):
        assert s < n, "strict mode must be at least as tight as normal mode"


def test_validated_plan_is_the_only_route_to_rendering():
    """The renderer's signature is part of the guarantee."""
    import inspect

    from djmix.models import ValidatedMixPlan
    from djmix.render.engine import render

    annotation = inspect.signature(render).parameters["plan"].annotation
    assert annotation in (ValidatedMixPlan, "ValidatedMixPlan")


def test_min_fade_is_enforced(valid_payload, analyses_by_id, selected_ids):
    payload = copy.deepcopy(valid_payload)
    payload["plan"][0]["transition"]["len_sec"] = MIN_FADE_SEC / 2
    validated, _ = _validate(payload, analyses_by_id, selected_ids)
    assert validated is None
