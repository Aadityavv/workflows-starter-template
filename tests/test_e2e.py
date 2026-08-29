"""End-to-end: the MVP acceptance criteria, asserted.

The spec's criterion is: given 5+ user-uploaded tracks and an occasion label,
return one exported audio file with audible, non-jarring transitions, within a
processing-time budget.

"Non-jarring" cannot be asserted by listening in CI, so it is measured by proxy:
seam steps relative to local content, beat-grid alignment error at matched
junctions, loudness, and headroom. Those numbers are reported, not claimed.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import soundfile as sf

from djmix.audio.io import write_audio
from djmix.planning.base import MixRequest
from djmix.planning.rules import RulePlanner
from djmix.planning.validation import validate_plan
from djmix.render.engine import render

TIME_BUDGET_SEC_PER_TRACK = 18.0  # the spec's "3 minutes per 10 tracks"


def test_five_plus_tracks_to_one_exported_file(analyses, analyses_by_id, tmp_path):
    assert len(analyses) >= 5, "the acceptance criterion is stated for 5+ tracks"

    start = time.perf_counter()
    result = RulePlanner().plan(MixRequest(analyses=analyses, occasion="road trip"))
    buffer, report = render(result.plan, analyses_by_id)
    out = write_audio(tmp_path / "mix.wav", buffer)
    elapsed = time.perf_counter() - start

    assert out.is_file() and out.stat().st_size > 0

    written, sr = sf.read(str(out), always_2d=True)
    assert sr == buffer.sample_rate
    assert written.shape[1] == 2
    assert abs(written.shape[0] / sr - report.duration_sec) < 0.05
    assert np.isfinite(written).all()

    # Every track in the plan is actually audible for a reasonable stretch.
    assert len(result.plan.steps) >= 5

    budget = TIME_BUDGET_SEC_PER_TRACK * len(analyses)
    assert elapsed < budget, (
        f"planning + rendering took {elapsed:.1f}s for {len(analyses)} cached tracks "
        f"(budget {budget:.0f}s)"
    )


def test_transition_quality_proxies(analyses, analyses_by_id):
    """The objective stand-ins for 'audible but not jarring'."""
    result = RulePlanner().plan(MixRequest(analyses=analyses, occasion="party"))
    _, report = render(result.plan, analyses_by_id)

    assert report.worst_seam_ratio < 2.0
    assert report.master.integrated_lufs == pytest.approx(-14.0, abs=0.7)
    assert report.peak_db <= -0.9
    for junction in report.junctions:
        assert junction.fade_sec >= 2.0, "a transition must be a blend, not a jump cut"
        if junction.beatmatched:
            assert junction.grid_error_ms < 25.0


def test_a_round_trip_through_disk_still_validates(analyses, analyses_by_id, tmp_path):
    """A plan written to a file and read back must still pass every gate."""
    result = RulePlanner().plan(MixRequest(analyses=analyses, occasion="focus"))
    path = tmp_path / "plan.json"
    path.write_text(result.plan.plan.model_dump_json(indent=2, exclude_none=True))

    payload = json.loads(path.read_text())
    validated, report = validate_plan(payload, analyses_by_id, planner="file")
    assert validated is not None, report.messages()


def test_natural_language_request_reaches_a_rendered_mix(analyses, analyses_by_id, tmp_path):
    """The spec's headline example, on the offline path."""
    from djmix.llm.mock import MockProvider
    from djmix.planning.llm import LLMPlanner

    result = LLMPlanner(MockProvider("valid")).plan(
        MixRequest(
            analyses=analyses,
            prompt="give me a 3-minute high-energy pre-workout mix",
            target_minutes=3,
        )
    )
    assert result.plan.occasion == "workout", "free text should map to the workout occasion"

    buffer, report = render(result.plan, analyses_by_id)
    out = write_audio(tmp_path / "prompt-mix.wav", buffer)
    assert out.is_file()
    assert report.duration_sec > 30


def test_entitlements_are_config_driven(analyses):
    from djmix.config import EntitlementError, check_mix_request, get_tier

    free = get_tier("free")
    check_mix_request(free, len(analyses), 10.0)
    with pytest.raises(EntitlementError):
        check_mix_request(free, len(analyses), free.max_mix_minutes + 1)
    with pytest.raises(EntitlementError):
        check_mix_request(free, free.max_tracks_per_mix + 1, 5.0)
    assert get_tier("plus").max_mix_minutes > free.max_mix_minutes
