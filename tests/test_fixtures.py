"""The fixtures themselves must be trustworthy before they can grade anything."""

from __future__ import annotations

from pathlib import Path

import generate as G
import numpy as np

from djmix.audio.io import load_mono_for_analysis


def test_fixtures_exist_and_are_audible(ground_truth):
    assert len(ground_truth) >= 5, "the MVP criterion is stated for 5+ tracks"
    for truth in ground_truth:
        path = G.FIXTURE_DIR / truth["path"]
        assert path.is_file()
        y, sr = load_mono_for_analysis(path)
        assert np.isfinite(y).all()
        assert float(np.abs(y).max()) > 0.1, f"{truth['path']} is effectively silent"


def test_generation_is_deterministic_in_process():
    """Same seed, same bytes -- otherwise ground truth drifts under the tests."""
    first, _ = G.render_fixture(G.SPECS[0])
    second, _ = G.render_fixture(G.SPECS[0])
    assert np.array_equal(first, second)


def test_generation_is_deterministic_across_processes(tmp_path):
    """The in-process check above is not sufficient, and this is not a
    theoretical gap: seeding from Python's built-in hash() passed it while
    producing different audio on every run, because string hashing is
    randomised per process. Two separate interpreters, with deliberately
    different PYTHONHASHSEED values, must produce byte-identical fixtures.
    """
    import hashlib
    import os
    import subprocess
    import sys

    def generate_into(directory: Path, hash_seed: str) -> dict[str, str]:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        subprocess.run(
            [sys.executable, str(Path(G.__file__)), "--out", str(directory)],
            check=True,
            capture_output=True,
            env=env,
        )
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.flac"))
        }

    first = generate_into(tmp_path / "a", "1")
    second = generate_into(tmp_path / "b", "2")
    assert first and first == second, "fixture bytes depend on interpreter state"


def test_fixtures_have_real_percussive_transients(ground_truth):
    """The regression guard for the false 0.0 BPM.

    A signal with no broadband attack transients gives the onset detector a flat
    envelope and nothing to lock onto. This asserts the fixtures have plenty of
    real onsets BEFORE any tempo test runs, so a fixture regression cannot be
    misdiagnosed as an analyzer regression.
    """
    import librosa

    for truth in ground_truth:
        y, sr = load_mono_for_analysis(G.FIXTURE_DIR / truth["path"])
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        assert float(onset.std()) > 1e-3, f"{truth['path']} has a flat onset envelope"
        peaks = librosa.util.peak_pick(
            onset, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.2, wait=2
        )
        expected_beats = truth["duration_sec"] / (60.0 / truth["bpm"])
        assert len(peaks) > expected_beats * 0.5, (
            f"{truth['path']}: only {len(peaks)} onsets for ~{expected_beats:.0f} beats"
        )


def test_a_pure_tone_gives_the_detector_almost_nothing(ground_truth):
    """Documents why these fixtures are synthesised the way they are.

    A steady tone is not literally onset-free -- it has one attack where it
    starts. What it lacks is *recurring* transients, which is what a beat
    tracker needs, so it yields orders of magnitude fewer onset peaks than real
    percussive material of the same length.
    """
    import librosa

    sr = 22050
    seconds = 10
    tone = np.sin(2 * np.pi * 440 * np.arange(0, seconds, 1 / sr)).astype(np.float32)
    tone_peaks = librosa.util.peak_pick(
        librosa.onset.onset_strength(y=tone, sr=sr),
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=0.2,
        wait=2,
    )
    y, real_sr = load_mono_for_analysis(G.FIXTURE_DIR / ground_truth[0]["path"])
    real_peaks = librosa.util.peak_pick(
        librosa.onset.onset_strength(y=y[: seconds * real_sr], sr=real_sr),
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=0.2,
        wait=2,
    )
    assert len(tone_peaks) < 5
    assert len(real_peaks) > 10 * max(len(tone_peaks), 1)
