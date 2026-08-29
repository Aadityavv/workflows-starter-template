"""Mood / character tags derived from measured features.

IMPORTANT: this is a *measured* tagger, not an LLM one. Every input is a DSP
feature computed from the waveform; the output is a probability-like
distribution over occasion-relevant moods, never a single hard label.

It is deliberately a placeholder for the CLAP zero-shot audio-text model planned
for Phase 2, and it is honestly much cruder than CLAP: it is a hand-weighted
function of tempo, brightness, percussiveness, mode, and dynamic range. The
point is that the *interface* is the one CLAP will implement, so swapping the
model changes this file and nothing else -- and that the no-LLM rule holds even
for the soft, subjective part of the analysis.
"""

from __future__ import annotations

import librosa
import numpy as np

MOODS = ("energetic", "romantic", "spiritual", "focus", "melancholic", "uplifting")


def _bell(x: float, center: float, width: float) -> float:
    return float(np.exp(-(((x - center) / width) ** 2)))


def compute_features(
    y: np.ndarray,
    sr: int,
    bpm: float,
    mode: str,
    harmonic: np.ndarray | None = None,
    percussive: np.ndarray | None = None,
) -> dict[str, float]:
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    if harmonic is None or percussive is None:
        harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.sum(harmonic**2))
    p_energy = float(np.sum(percussive**2))
    percussiveness = p_energy / max(h_energy + p_energy, 1e-12)
    rms = librosa.feature.rms(y=y)[0]
    crest = float(np.max(rms) / max(float(np.mean(rms)), 1e-12))
    return {
        "bpm": bpm,
        "brightness": float(np.clip(centroid / 4000.0, 0.0, 1.0)),
        "rolloff": float(np.clip(rolloff / 11000.0, 0.0, 1.0)),
        "percussiveness": float(np.clip(percussiveness, 0.0, 1.0)),
        "crest": float(np.clip(crest / 6.0, 0.0, 1.0)),
        "is_minor": 1.0 if mode.lower().startswith("min") else 0.0,
    }


def tag_track(
    y: np.ndarray,
    sr: int,
    bpm: float,
    mode: str,
    harmonic: np.ndarray | None = None,
    percussive: np.ndarray | None = None,
) -> dict[str, float]:
    """Return a normalised distribution over MOODS."""
    f = compute_features(y, sr, bpm, mode, harmonic=harmonic, percussive=percussive)
    fast = _bell(f["bpm"], 140, 45)
    mid = _bell(f["bpm"], 105, 30)
    slow = _bell(f["bpm"], 72, 28)
    bright, minor, punch = f["brightness"], f["is_minor"], f["percussiveness"]

    raw = {
        "energetic": 0.55 * fast + 0.30 * punch + 0.15 * bright,
        "romantic": 0.45 * slow + 0.25 * (1 - punch) + 0.30 * (1 - bright),
        "spiritual": 0.40 * slow + 0.35 * (1 - punch) + 0.25 * f["crest"],
        "focus": 0.40 * mid + 0.35 * (1 - f["crest"]) + 0.25 * (1 - bright),
        "melancholic": 0.50 * minor + 0.30 * slow + 0.20 * (1 - bright),
        "uplifting": 0.40 * (1 - minor) + 0.35 * bright + 0.25 * mid,
    }
    total = sum(raw.values())
    if total <= 1e-12:
        return dict.fromkeys(MOODS, 1.0 / len(MOODS))
    return {k: round(v / total, 4) for k, v in raw.items()}
