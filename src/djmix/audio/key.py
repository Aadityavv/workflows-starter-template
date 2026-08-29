"""Musical key estimation: chroma + Krumhansl-Schmuckler profile correlation.

Measured from the waveform, never looked up. The estimator reports its
`runner_up` and a confidence margin because relative major/minor confusion is
the dominant real-world failure mode -- the planner distrusts low-confidence
keys rather than mixing confidently on a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from djmix.planning.camelot import PITCH_CLASSES, to_camelot

# Krumhansl-Kessler probe-tone profiles: the perceived stability of each scale
# degree, averaged over listeners. Correlating an observed chroma vector against
# all 24 rotations is the standard K-S key-finding method.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class KeyResult:
    tonic: str
    mode: str
    camelot: str
    confidence: float
    runner_up: tuple[str, str]
    correlations: dict[str, float]

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"


def _correlate(vec: np.ndarray, profile: np.ndarray) -> float:
    v = vec - vec.mean()
    p = profile - profile.mean()
    denom = np.linalg.norm(v) * np.linalg.norm(p)
    return float(np.dot(v, p) / denom) if denom > 1e-12 else 0.0


# Alternative profile sets, selectable for comparison. K-S is the default
# because it is the method the design calls for; Temperley and
# Albrecht-Shanahan are kept because they are measurably more robust to
# percussive bleed and are useful when diagnosing a suspicious key reading.
PROFILES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "krumhansl": (KS_MAJOR, KS_MINOR),
    "temperley": (
        np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
        np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
    ),
    "albrecht_shanahan": (
        np.array(
            [0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081]
        ),
        np.array(
            [0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052]
        ),
    ),
}


def chroma_vector(
    y: np.ndarray, sr: int, hop_length: int = 512, harmonic: np.ndarray | None = None
) -> np.ndarray:
    """Loudness-weighted average CENS chroma of the harmonic component.

    Two choices matter here, both settled by measurement against the labelled
    fixtures rather than by preference:

    * Percussive energy is removed first (HPSS), because a drum hit adds roughly
      equal energy to every chroma bin and washes out the profile correlation.
    * CENS rather than plain CQT chroma. CENS quantises and smooths each frame
      and L2-normalises it, which discards exactly the loudness and timbre
      variation that percussive bleed introduces. On the fixture set, K-S over
      raw CQT chroma scored 4/6 -- confusing two tracks with their parallel
      major -- while K-S over CENS scored 6/6. Every other profile set scored
      6/6 either way, so CQT was the single point of failure, not the profiles.

    Frames are then weighted by squared RMS so loud, harmonically definite
    passages count for more than quiet ambiguous ones.
    """
    if harmonic is None:
        harmonic = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cens(
        y=harmonic, sr=sr, hop_length=hop_length, bins_per_octave=36
    )
    rms = librosa.feature.rms(y=harmonic, hop_length=hop_length)[0]
    n = min(chroma.shape[1], len(rms))
    weights = rms[:n] ** 2
    if weights.sum() <= 1e-12:
        weights = np.ones(n)
    vec = (chroma[:, :n] * weights).sum(axis=1)
    total = vec.sum()
    return vec / total if total > 1e-12 else vec


def estimate_key(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    profile: str = "krumhansl",
    harmonic: np.ndarray | None = None,
) -> KeyResult:
    major_profile, minor_profile = PROFILES[profile]
    vec = chroma_vector(y, sr, hop_length, harmonic=harmonic)
    correlations: dict[str, float] = {}
    for i, pitch in enumerate(PITCH_CLASSES):
        rotated = np.roll(vec, -i)
        correlations[f"{pitch} major"] = _correlate(rotated, major_profile)
        correlations[f"{pitch} minor"] = _correlate(rotated, minor_profile)

    ranked = sorted(correlations.items(), key=lambda kv: kv[1], reverse=True)
    (best_name, best_r), (second_name, second_r) = ranked[0], ranked[1]
    worst_r = ranked[-1][1]

    spread = best_r - worst_r
    confidence = float(np.clip((best_r - second_r) / spread, 0.0, 1.0)) if spread > 1e-9 else 0.0

    tonic, mode = best_name.split(" ")
    runner_tonic, runner_mode = second_name.split(" ")
    return KeyResult(
        tonic=tonic,
        mode=mode,
        camelot=to_camelot(tonic, mode),
        confidence=round(confidence, 3),
        runner_up=(runner_tonic, runner_mode),
        correlations={k: round(v, 4) for k, v in correlations.items()},
    )
