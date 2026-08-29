"""Analysis orchestrator: one audio file in, one TrackAnalysis out.

This is the only place the individual DSP modules are composed, and the only
producer of TrackAnalysis objects. Everything downstream -- planners, validator,
renderer -- consumes its output and never touches audio for measurement again.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import librosa

from djmix.audio import io as audio_io
from djmix.audio.cache import AnalysisCache
from djmix.audio.energy import analyze_energy
from djmix.audio.key import estimate_key
from djmix.audio.structure import analyze_structure
from djmix.audio.tagging import tag_track
from djmix.audio.tempo import analyze_tempo
from djmix.models import ANALYZER_VERSION, Segment, TrackAnalysis

# Stable per-content id: analysing the same file twice, from any directory,
# yields the same track_id, so cache entries and mix plans always agree.
TRACK_NAMESPACE = uuid.UUID("6f1b0f4a-1f2c-4f9e-9b3d-0f1d2c3b4a59")


class AnalysisError(RuntimeError):
    pass


def track_id_for(content_hash: str) -> str:
    return str(uuid.uuid5(TRACK_NAMESPACE, content_hash))


def analyze_file(
    path: str | Path,
    cache: AnalysisCache | None = None,
    force: bool = False,
) -> TrackAnalysis:
    path = Path(path)
    audio_io.check_uploadable(path)
    content_hash = audio_io.content_hash(path)

    if cache is not None and not force:
        cached = cache.get(content_hash)
        if cached is not None:
            return cached

    y, sr = audio_io.load_mono_for_analysis(path)

    # One harmonic/percussive separation for the whole pipeline. HPSS dominates
    # analysis cost, and tempo, key, structure, and tagging each want a piece of
    # it; computing it per-module made a 90 s track take ~24 s instead of ~7 s.
    # The asymmetric margins give key detection a clean harmonic signal (4.0)
    # and beat tracking crisp transients (2.0).
    harmonic, percussive = librosa.effects.hpss(y, margin=(4.0, 2.0))

    tempo = analyze_tempo(y, sr, percussive=percussive, harmonic=harmonic)
    if not tempo.ok or tempo.bpm is None:
        # Refusing is the correct outcome: a track with no trackable beat cannot
        # be beat-matched, and inventing a tempo here would be exactly the kind
        # of unmeasured number the whole design exists to prevent.
        raise AnalysisError(
            f"{path.name}: could not measure a beat grid ({'; '.join(tempo.warnings)})"
        )

    key = estimate_key(y, sr, harmonic=harmonic)
    energy = analyze_energy(y, sr, tempo.beat_times, tempo.downbeat_times)
    structure = analyze_structure(
        y, sr, tempo.beat_times, energy.energetic_section, harmonic=harmonic
    )
    mood_tags = tag_track(y, sr, tempo.bpm, key.mode, harmonic=harmonic, percussive=percussive)

    analysis = TrackAnalysis(
        track_id=track_id_for(content_hash),
        source_path=str(path.resolve()),
        content_hash=content_hash,
        analyzer_version=ANALYZER_VERSION,
        duration_sec=round(len(y) / sr, 3),
        bpm=tempo.bpm,
        bpm_confidence=tempo.confidence,
        bpm_alternatives=tempo.alternatives,
        beat_times=[round(float(t), 4) for t in tempo.beat_times],
        downbeat_times=[round(float(t), 4) for t in tempo.downbeat_times],
        key=key.name,
        key_confidence=key.confidence,
        camelot=key.camelot,
        energetic_section=energy.energetic_section,
        chorus_estimate=structure.chorus_estimate,
        segments=[Segment(start=s, end=e, label=lbl) for s, e, lbl in structure.segments],
        energy_curve=[round(float(v), 5) for v in energy.curve],
        energy_curve_hz=energy.curve_hz,
        energy_mean_db=energy.mean_db,
        energy_score=energy.score,
        mood_tags=mood_tags,
    )

    if cache is not None:
        cache.put(analysis)
    return analysis


def analyze_directory(
    directory: str | Path,
    cache: AnalysisCache | None = None,
    force: bool = False,
    on_progress=None,
) -> tuple[list[TrackAnalysis], list[tuple[Path, str]]]:
    """Analyse every supported file in a directory.

    Returns (analyses, failures). One unreadable or beatless file must not abort
    a library scan, so failures are collected and reported rather than raised.
    """
    directory = Path(directory)
    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in audio_io.SUPPORTED_SUFFIXES
    )
    analyses: list[TrackAnalysis] = []
    failures: list[tuple[Path, str]] = []
    for i, path in enumerate(files):
        if on_progress:
            on_progress(i, len(files), path)
        try:
            analyses.append(analyze_file(path, cache=cache, force=force))
        except Exception as exc:
            failures.append((path, str(exc)))
    if on_progress:
        on_progress(len(files), len(files), None)
    return analyses, failures
