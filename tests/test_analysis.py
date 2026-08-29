"""Analyzer accuracy against exactly-known ground truth."""

from __future__ import annotations

import numpy as np

from djmix.audio.tempo import analyze_tempo
from djmix.planning.camelot import camelot_distance, to_camelot

BPM_TOLERANCE = 0.02


def _octave_aware_error(measured: float, truth: float) -> tuple[float, str]:
    """Smallest relative error over half/normal/double time, and which one won."""
    options = {"1x": measured, "2x": measured * 2, "0.5x": measured / 2}
    name, value = min(options.items(), key=lambda kv: abs(kv[1] - truth))
    return abs(value - truth) / truth, name


def test_bpm_within_tolerance(analyses, truth_for):
    octave_cases = []
    for a in analyses:
        truth = truth_for[a.track_id]
        error, which = _octave_aware_error(a.bpm, truth["bpm"])
        assert error <= BPM_TOLERANCE, (
            f"{truth['path']}: measured {a.bpm:.2f}, expected {truth['bpm']} "
            f"(best match {which}, error {error:.1%})"
        )
        if which != "1x":
            octave_cases.append((truth["path"], a.bpm, truth["bpm"], which))
    # Octave readings are legitimate but should stay visible, not be absorbed
    # silently -- if this list grows, the tempo estimator has regressed.
    assert len(octave_cases) <= 1, f"too many octave-mismatched readings: {octave_cases}"


def test_bpm_is_never_zero(analyses):
    for a in analyses:
        assert a.bpm > 0, "a measured BPM of 0.0 must never be produced"


def test_beatless_audio_is_refused_not_guessed():
    """The prototyping bug, pinned: a tone must yield None, never 0.0."""
    sr = 22050
    tone = np.sin(2 * np.pi * 440 * np.arange(0, 20, 1 / sr)).astype(np.float32)
    result = analyze_tempo(tone, sr)
    assert result.ok is False
    assert result.bpm is None
    assert result.bpm != 0.0
    assert result.warnings


def test_beat_grid_matches_reported_bpm(analyses):
    """bpm and beat_times must describe the same grid -- the validator's
    bar-multiple check and its beat-snap check would otherwise contradict."""
    for a in analyses:
        intervals = np.diff(a.beat_times)
        implied = 60.0 / float(np.median(intervals))
        assert abs(implied - a.bpm) / a.bpm < 0.03


def test_key_exact_match(analyses, truth_for):
    misses = []
    for a in analyses:
        truth = truth_for[a.track_id]
        if a.key != truth["key"]:
            misses.append((truth["path"], a.key, truth["key"]))
    assert not misses, f"key mismatches: {misses}"


def test_key_misses_would_be_near_misses(analyses, truth_for):
    """If the estimator does regress, it should confuse related keys rather than
    unrelated ones. Asserting the failure *mode* keeps a regression legible."""
    for a in analyses:
        truth = truth_for[a.track_id]
        expected = to_camelot(*truth["key"].split())
        assert camelot_distance(a.camelot, expected) <= 0.15


def test_energetic_section_lands_in_a_loud_section(analyses, truth_for):
    for a in analyses:
        truth = truth_for[a.track_id]
        choruses = [s for s in truth["sections"] if s["label"] == "chorus"]
        start, end = a.energetic_section
        assert end > start
        assert any(s["start"] - 8 <= start and end <= s["end"] + 8 for s in choruses), (
            f"{truth['path']}: energetic section {a.energetic_section} is not in a chorus "
            f"({[(s['start'], s['end']) for s in choruses]})"
        )


def test_chorus_estimate_starts_inside_a_chorus(analyses, truth_for):
    for a in analyses:
        truth = truth_for[a.track_id]
        choruses = [s for s in truth["sections"] if s["label"] == "chorus"]
        start = a.chorus_estimate[0]
        assert any(s["start"] - 8 <= start <= s["end"] + 8 for s in choruses), (
            f"{truth['path']}: chorus estimate {a.chorus_estimate} misses every true chorus"
        )


def test_downbeats_are_a_subset_of_beats(analyses):
    # Compared with a tolerance rather than by set membership: both lists are
    # rounded floats, and numpy and Python disagree on exact .5 ties.
    for a in analyses:
        beats = np.asarray(a.beat_times)
        for downbeat in a.downbeat_times:
            assert np.min(np.abs(beats - downbeat)) < 1e-3


def test_mood_tags_form_a_distribution(analyses):
    for a in analyses:
        assert a.mood_tags
        assert abs(sum(a.mood_tags.values()) - 1.0) < 0.01
        assert all(0.0 <= v <= 1.0 for v in a.mood_tags.values())


def test_analysis_is_cached_and_reused(tmp_path, ground_truth):
    import time

    import generate as G

    from djmix.audio.analysis import analyze_file
    from djmix.audio.cache import AnalysisCache

    cache = AnalysisCache(tmp_path / "cache")
    path = G.FIXTURE_DIR / ground_truth[0]["path"]

    start = time.perf_counter()
    first = analyze_file(path, cache=cache)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    second = analyze_file(path, cache=cache)
    warm = time.perf_counter() - start

    assert first.model_dump() == second.model_dump()
    assert warm < cold / 5, f"cache did not help: {cold:.2f}s cold vs {warm:.2f}s warm"


def test_track_id_is_content_derived(tmp_path, ground_truth):
    """Same bytes in a different place must give the same id, so cache entries
    and mix plans agree regardless of where a file lives."""
    import shutil

    import generate as G

    from djmix.audio.analysis import analyze_file

    source = G.FIXTURE_DIR / ground_truth[0]["path"]
    copy = tmp_path / "renamed.flac"
    shutil.copy(source, copy)
    assert analyze_file(source).track_id == analyze_file(copy).track_id
