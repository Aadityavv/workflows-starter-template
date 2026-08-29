"""Pitch-preserving time stretch, and the tempo plan that drives it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_STRETCH = 0.06  # ±6%: beyond this, Rubber Band starts to be audible.
BYPASS_THRESHOLD = 0.002  # Below this, skip the stretcher entirely.


@dataclass
class TempoGroup:
    tempo: float
    member_ids: list[str]


def build_tempo_groups(bpms: list[tuple[str, float]], max_stretch: float = MAX_STRETCH):
    """Partition an ordered track list into runs that can share one tempo.

    Why groups rather than a per-transition tempo ramp: inside a group every
    track is stretched by a single constant factor to the group's tempo, so the
    beat grids line up *exactly* and a crossfade is genuinely beat-locked.
    Ramping tempo through a fade would sound smoother in principle, but Rubber
    Band's offline API takes a constant factor -- a ramp means chunked
    processing and seam discontinuities, and it destroys the bit-level
    reproducibility that the determinism test depends on.

    When the next track is too far away in tempo to stretch into the current
    group, a new group opens and the junction between them is rendered as a
    non-beatmatched blend rather than mangling the audio to force a match.
    """
    groups: list[TempoGroup] = []
    current_ids: list[str] = []
    current_bpms: list[float] = []

    def close() -> None:
        if current_ids:
            groups.append(TempoGroup(float(np.median(current_bpms)), list(current_ids)))

    for track_id, bpm in bpms:
        if not current_ids:
            current_ids, current_bpms = [track_id], [bpm]
            continue
        candidate = float(np.median([*current_bpms, bpm]))
        # Every member, including the newcomer, must stay inside the budget.
        if all(abs(candidate - b) / b <= max_stretch for b in [*current_bpms, bpm]):
            current_ids.append(track_id)
            current_bpms.append(bpm)
        else:
            close()
            current_ids, current_bpms = [track_id], [bpm]
    close()
    return groups


def stretch_ratio(source_bpm: float, target_bpm: float, max_stretch: float = MAX_STRETCH) -> float:
    """Playback-rate ratio to move `source_bpm` to `target_bpm`, clamped."""
    if source_bpm <= 0:
        return 1.0
    ratio = target_bpm / source_bpm
    ratio = float(np.clip(ratio, 1.0 - max_stretch, 1.0 + max_stretch))
    return 1.0 if abs(ratio - 1.0) < BYPASS_THRESHOLD else ratio


def time_stretch(samples: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Stretch (channels, n) audio so it plays `ratio` times faster.

    Uses pedalboard's Rubber Band binding when available -- it is a pip wheel
    with no system dependency and is transparent at these ratios -- and falls
    back to librosa's phase vocoder otherwise, which smears transients but keeps
    the pipeline working without the optional wheel.
    """
    if abs(ratio - 1.0) < BYPASS_THRESHOLD:
        return samples
    try:
        import pedalboard

        out = pedalboard.time_stretch(
            samples.astype(np.float32),
            sr,
            stretch_factor=float(ratio),
            pitch_shift_in_semitones=0.0,
            high_quality=True,
        )
        return np.ascontiguousarray(np.atleast_2d(out).astype(np.float32))
    except Exception:
        import librosa

        stretched = [
            librosa.effects.time_stretch(channel.astype(np.float32), rate=float(ratio))
            for channel in samples
        ]
        width = min(len(c) for c in stretched)
        return np.ascontiguousarray(np.vstack([c[:width] for c in stretched]).astype(np.float32))
