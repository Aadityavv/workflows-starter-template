"""Loudness normalisation and true-peak protection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_LUFS = -14.0
TRUE_PEAK_CEILING_DB = -1.0


@dataclass
class MasterReport:
    integrated_lufs: float
    gain_db: float
    peak_db: float
    limited: bool


def measure_lufs(samples: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 integrated loudness. `samples` is (channels, n)."""
    import pyloudnorm

    meter = pyloudnorm.Meter(sr)
    return float(meter.integrated_loudness(samples.T.astype(np.float64)))


def normalize_track(samples: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Pre-gain one track toward the target before it enters the mix.

    Doing this per track as well as on the final mix matters: a final-pass-only
    normalisation lets a quiet track get buried underneath a loud one *during*
    the crossfade, which is precisely where it is most audible.
    """
    # Loudness measurement needs at least one 400 ms block.
    if samples.shape[1] < int(0.5 * sr):
        return samples
    loudness = measure_lufs(samples, sr)
    if not np.isfinite(loudness):
        return samples
    gain = 10 ** ((target_lufs - loudness) / 20.0)
    # Bound the correction: a near-silent intro should not be dragged up 40 dB.
    gain = float(np.clip(gain, 0.1, 4.0))
    out = samples * gain
    # Guard the peak too. Most source material is already mastered near full
    # scale, so a gain toward -14 LUFS is often >1 and would push it over -- and
    # several such tracks then sum during a crossfade. Clipping before the mix
    # bus is not recoverable later.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def master(
    samples: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS
) -> tuple[np.ndarray, MasterReport]:
    """Normalise the finished mix and keep it under the true-peak ceiling."""
    audio = samples.astype(np.float64)
    loudness = measure_lufs(audio, sr)

    if not np.isfinite(loudness):
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return samples, MasterReport(
            integrated_lufs=float("-inf"),
            gain_db=0.0,
            peak_db=20 * np.log10(max(peak, 1e-10)),
            limited=False,
        )

    gain_db = target_lufs - loudness
    audio = audio * (10 ** (gain_db / 20.0))

    ceiling = 10 ** (TRUE_PEAK_CEILING_DB / 20.0)
    peak = float(np.max(np.abs(audio)))
    limited = False
    if peak > ceiling:
        try:
            import pedalboard

            limiter = pedalboard.Limiter(threshold_db=TRUE_PEAK_CEILING_DB, release_ms=100.0)
            audio = limiter(audio.astype(np.float32), sr).astype(np.float64)
        except Exception:
            audio = audio * (ceiling / peak)
        limited = True
        # A limiter changes perceived loudness, and not always downward -- it
        # raises density while cutting peaks. Re-measure and correct ONCE.
        # Iterating to convergence would be non-deterministic in the last bit,
        # and determinism is an acceptance criterion for this engine.
        corrected = measure_lufs(audio, sr)
        if np.isfinite(corrected) and abs(corrected - target_lufs) > 0.5:
            trim = 10 ** ((target_lufs - corrected) / 20.0)
            audio = audio * trim
            gain_db += 20 * np.log10(trim)
            peak = float(np.max(np.abs(audio)))
            if peak > ceiling:
                audio = audio * (ceiling / peak)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio.astype(np.float32), MasterReport(
        integrated_lufs=round(measure_lufs(audio, sr), 2),
        gain_db=round(gain_db, 2),
        peak_db=round(20 * np.log10(max(peak, 1e-10)), 2),
        limited=limited,
    )
