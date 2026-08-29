"""Rendering: determinism, loudness, seam quality, and crossfade maths."""

from __future__ import annotations

import numpy as np
import pytest

from djmix.planning.base import MixRequest
from djmix.planning.rules import RulePlanner
from djmix.render.crossfade import align_phase, equal_power_curves
from djmix.render.engine import render
from djmix.render.master import master
from djmix.render.timestretch import build_tempo_groups, stretch_ratio, time_stretch


def test_equal_power_curves_hold_constant_power():
    """The reason for cos/sin rather than a linear fade: a linear crossfade of
    two uncorrelated signals dips about 3 dB in the middle."""
    out, incoming = equal_power_curves(1024)
    assert np.allclose(out**2 + incoming**2, 1.0, atol=1e-9)
    assert out[0] == pytest.approx(1.0)
    assert incoming[0] == pytest.approx(0.0)


def test_align_phase_never_moves_more_than_half_a_beat():
    beat = 0.5
    outgoing = [10.0 + i * beat for i in range(20)]
    incoming = [0.13 + i * beat for i in range(20)]
    aligned = align_phase(10.2, 5.0, outgoing, incoming, beat)
    assert abs(aligned - 5.0) <= beat / 2 + 1e-9


def test_tempo_groups_respect_the_stretch_budget():
    groups = build_tempo_groups(
        [("a", 128.0), ("b", 130.0), ("c", 132.0), ("d", 174.0), ("e", 176.0)]
    )
    assert len(groups) == 2, "a 174 BPM track cannot join a ~130 BPM group within +/-6%"
    assert [g.member_ids for g in groups] == [["a", "b", "c"], ["d", "e"]]
    assert groups[0].tempo == pytest.approx(130.0)
    assert groups[-1].tempo == pytest.approx(175.0)
    # Every member must be reachable within the stretch budget of its group.
    for group, bpms in zip(groups, ([128.0, 130.0, 132.0], [174.0, 176.0]), strict=True):
        for bpm in bpms:
            assert abs(group.tempo - bpm) / bpm <= 0.06


def test_stretch_ratio_is_clamped_and_bypassed():
    assert stretch_ratio(128.0, 128.0) == 1.0
    assert stretch_ratio(128.0, 128.1) == 1.0, "a negligible ratio should bypass the stretcher"
    assert stretch_ratio(100.0, 200.0) == pytest.approx(1.06)


def test_time_stretch_changes_length_in_the_right_direction():
    sr = 22050
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((2, sr * 2)).astype(np.float32) * 0.1
    slower = time_stretch(audio, sr, 0.94)
    assert slower.shape[0] == 2
    assert slower.shape[1] > audio.shape[1] * 1.02


def test_master_hits_the_loudness_target():
    sr = 44100
    t = np.arange(sr * 5) / sr
    quiet = np.vstack([np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32) * 0.02
    out, report = master(quiet, sr, target_lufs=-14.0)
    assert report.integrated_lufs == pytest.approx(-14.0, abs=0.5)
    assert report.peak_db <= -0.9


def test_master_survives_silence():
    sr = 44100
    silence = np.zeros((2, sr), dtype=np.float32)
    out, report = master(silence, sr)
    assert out.shape == silence.shape
    assert not np.isfinite(report.integrated_lufs)


@pytest.fixture(scope="module")
def rendered(analyses, analyses_by_id):
    plan = RulePlanner().plan(MixRequest(analyses=analyses, occasion="workout")).plan
    return render(plan, analyses_by_id)


def test_render_produces_sane_audio(rendered):
    buffer, report = rendered
    assert buffer.channels == 2
    assert report.duration_sec > 30
    assert np.isfinite(buffer.samples).all()
    assert float(np.abs(buffer.samples).max()) <= 1.0


def test_render_hits_the_loudness_target(rendered):
    _, report = rendered
    assert report.master.integrated_lufs == pytest.approx(-14.0, abs=0.7)
    assert report.peak_db <= -0.9, "output must stay under the true-peak ceiling"


def test_seams_are_not_audible_clicks(rendered):
    """Measured relative to local content: percussive audio legitimately has
    huge single-sample steps, so an absolute threshold would flag real music."""
    _, report = rendered
    assert report.worst_seam_ratio < 2.0, (
        f"a junction steps {report.worst_seam_ratio:.1f}x harder than the audio around it"
    )


def test_beatmatched_junctions_are_grid_locked(rendered):
    _, report = rendered
    matched = [j for j in report.junctions if j.beatmatched]
    assert matched, "at least one junction should be beat-matched"
    for junction in matched:
        assert junction.grid_error_ms < 25.0


def test_render_is_deterministic(analyses, analyses_by_id):
    """Same plan, same output. A determinism claim is only meaningful if it is
    tested, and it is what allows a rendered mix to be reproduced from a plan."""
    plan = RulePlanner().plan(MixRequest(analyses=analyses, occasion="romantic")).plan
    first, _ = render(plan, analyses_by_id)
    second, _ = render(plan, analyses_by_id)
    assert first.samples.shape == second.samples.shape
    assert float(np.max(np.abs(first.samples - second.samples))) < 1e-6


def test_planning_is_deterministic(analyses):
    a = RulePlanner().plan(MixRequest(analyses=analyses, occasion="party")).plan
    b = RulePlanner().plan(MixRequest(analyses=analyses, occasion="party")).plan
    assert [s.track_id for s in a.steps] == [s.track_id for s in b.steps]
