"""LLM extraction, and the retry/fallback state machine.

Everything here runs offline against the mock provider; conftest blocks the
network and strips provider keys, so a passing run proves the pipeline rather
than proving a provider was reachable.
"""

from __future__ import annotations

import json

import pytest

from djmix.llm.base import LLMUnavailable
from djmix.llm.extract import JSONExtractionError, extract_json
from djmix.llm.factory import get_provider
from djmix.llm.mock import MockProvider
from djmix.planning.base import MixRequest
from djmix.planning.llm import LLMPlanner

# --- defensive JSON extraction ---------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here you go:\n```json\n{"a": 1}\n```\nHope that helps.',
        'Here is the plan: {"a": 1}',
        '{"a": 1}\n{"b": 2}',
    ],
)
def test_extracts_json_from_common_wrappers(raw):
    parsed, repaired = extract_json(raw)
    assert parsed["a"] == 1
    assert repaired is False


def test_braces_inside_strings_do_not_confuse_the_scanner():
    parsed, _ = extract_json('{"occasion": "a } weird { name", "plan": []}')
    assert parsed["occasion"] == "a } weird { name"


def test_truncation_between_values_is_structurally_repaired():
    full = json.dumps({"occasion": "party", "plan": [{"track_id": "abc"}, {"track_id": "def"}]})
    cut = full.index('{"track_id": "def"')
    parsed, repaired = extract_json(full[:cut])
    assert repaired is True
    assert parsed["occasion"] == "party"
    assert parsed["plan"][0]["track_id"] == "abc"


def test_truncation_inside_a_string_is_refused():
    """Deliberate: closing an unterminated string means inventing its contents,
    and a track_id guessed that way would be silently wrong."""
    full = json.dumps({"occasion": "party", "plan": [{"track_id": "abcdef"}]})
    cut = full.index("abcdef") + 3
    with pytest.raises(JSONExtractionError):
        extract_json(full[:cut])


@pytest.mark.parametrize("raw", ["", "   ", "no json here at all", "{'a': 1}"])
def test_unrecoverable_output_raises(raw):
    """Note the single-quoted case: a model emitting Python dict literals is
    malfunctioning, and coercing it is how invalid plans get laundered in."""
    with pytest.raises(JSONExtractionError):
        extract_json(raw)


# --- provider layer ---------------------------------------------------------


def test_providers_are_pluggable():
    for name in ("groq", "anthropic", "mock"):
        assert get_provider(name) is not None
    with pytest.raises(ValueError):
        get_provider("nope")


@pytest.mark.parametrize("name", ["groq", "anthropic"])
def test_missing_credentials_degrade_rather_than_crash(name):
    with pytest.raises(LLMUnavailable):
        get_provider(name).call_llm("hello")


# --- the planner state machine ---------------------------------------------


@pytest.mark.parametrize("mode", ["valid", "fenced", "prose"])
def test_well_formed_responses_are_accepted(analyses, mode):
    result = LLMPlanner(MockProvider(mode)).plan(MixRequest(analyses=analyses, occasion="party"))
    assert result.planner == "llm"
    assert result.fallback_reason is None
    assert result.llm_attempts == 1


@pytest.mark.parametrize(
    "mode",
    ["malformed", "truncated", "empty", "hallucinated_time", "hallucinated_field", "wrong_chain"],
)
def test_bad_responses_fall_back_to_the_rule_planner(analyses, mode):
    provider = MockProvider(mode)
    result = LLMPlanner(provider).plan(MixRequest(analyses=analyses, occasion="party"))
    assert result.planner == "rule"
    assert result.fallback_reason, "a fallback must always record why"
    assert result.llm_attempts == 2, "the model gets exactly one repair attempt"
    assert len(provider.calls) == 2


def test_provider_failure_falls_back_immediately(analyses):
    provider = MockProvider("unavailable")
    result = LLMPlanner(provider).plan(MixRequest(analyses=analyses, occasion="party"))
    assert result.planner == "rule"
    assert result.llm_attempts == 1, "an unreachable provider should not be retried"


def test_repair_turn_is_told_what_was_wrong(analyses):
    provider = MockProvider("hallucinated_field")
    LLMPlanner(provider).plan(MixRequest(analyses=analyses, occasion="party"))
    assert "REJECTED" in provider.calls[1]
    assert "FORBIDDEN_FIELD" in provider.calls[1]


def test_strict_mode_refuses_to_fall_back(analyses):
    planner = LLMPlanner(MockProvider("malformed"), allow_fallback=False)
    with pytest.raises(LLMUnavailable):
        planner.plan(MixRequest(analyses=analyses, occasion="party"))


def test_prompt_states_the_no_invented_numbers_rule():
    from djmix.planning.prompts import SYSTEM_PROMPT

    assert "Do not invent BPM, key, or timestamp values" in SYSTEM_PROMPT
    assert "Use only the numbers provided in the input JSON" in SYSTEM_PROMPT


def test_prompt_offers_only_measured_options(analyses):
    """Every candidate time handed to the model must come off the measured grid,
    so a compliant model cannot help but be traceable."""
    from djmix.planning.prompts import track_facts

    for a in analyses:
        facts = track_facts(a)
        grid = set(round(t, 3) for t in a.beat_times)
        for value in facts["allowed_out_at"] + facts["allowed_in_at"]:
            assert any(abs(value - t) < 0.05 for t in grid), (
                f"offered {value} is not on the measured beat grid"
            )
        assert facts["allowed_fade_lengths"]
        assert "bpm" in facts and "allowed_out_at" in facts


def test_llm_plan_would_also_pass_independent_validation(analyses, analyses_by_id):
    """The validator must not trust the menu builder: re-validate the accepted
    plan from scratch against the analyses."""
    from djmix.planning.validation import validate_plan

    result = LLMPlanner(MockProvider("valid")).plan(MixRequest(analyses=analyses, occasion="party"))
    assert result.planner == "llm"
    payload = json.loads(result.plan.plan.model_dump_json(exclude_none=True))
    validated, report = validate_plan(payload, analyses_by_id, planner="recheck", strict=True)
    assert validated is not None, report.messages()
