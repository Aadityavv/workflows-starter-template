"""RMS energy curve and peak-energy window.

The energetic section is the anchor the planner reaches for most often, and its
edges are snapped to the measured beat grid so that a transition placed there is
automatically provenance-traceable.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

CURVE_HZ = 10.0
SMOOTH_SEC = 3.0
DEFAULT_WINDOW_SEC = 15.0


@dataclass
class EnergyResult:
    curve: np.ndarray
    curve_hz: float
    energetic_section: tuple[float, float]
    mean_db: float
    peak_time: float
    score: float
    crest_factor: float


def _snap(value: float, grid: np.ndarray, limit: float) -> float:
    """Move `value` onto the nearest grid point, but only if one is close enough."""
    if grid.size == 0:
        return value
    idx = int(np.argmin(np.abs(grid - value)))
    return float(grid[idx]) if abs(grid[idx] - value) <= limit else value


def analyze_energy(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray | None = None,
    downbeat_times: np.ndarray | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_length: int = 512,
) -> EnergyResult:
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Resample the frame-rate curve onto a fixed 10 Hz grid so that the stored
    # curve has the same meaning regardless of hop length or sample rate.
    n_points = max(2, int(duration * CURVE_HZ))
    grid_times = np.linspace(0, duration, n_points)
    curve = np.interp(grid_times, times, rms)

    smooth_n = max(1, int(SMOOTH_SEC * CURVE_HZ))
    kernel = np.ones(smooth_n) / smooth_n
    smoothed = np.convolve(curve, kernel, mode="same")

    window_sec = min(window_sec, max(1.0, duration / 3))
    win_n = max(1, int(window_sec * CURVE_HZ))
    if win_n >= len(smoothed):
        start_idx = 0
        win_n = len(smoothed)
    else:
        rolling = np.convolve(smoothed, np.ones(win_n) / win_n, mode="valid")
        start_idx = int(np.argmax(rolling))

    start = float(grid_times[start_idx])
    end = float(min(duration, start + window_sec))

    # Snap to the measured grid: downbeats if we have them, else beats. This is
    # what lets the planner use these edges directly as transition points and
    # still satisfy the provenance validator.
    grid = downbeat_times if downbeat_times is not None and len(downbeat_times) else beat_times
    if grid is not None and len(grid):
        grid = np.asarray(grid, dtype=float)
        tolerance = max(2.0, window_sec / 4)
        start = _snap(start, grid, tolerance)
        end = _snap(end, grid, tolerance)
        if end <= start:
            end = min(duration, start + window_sec)

    ref = max(float(np.max(rms)), 1e-10)
    mean_db = float(20 * np.log10(max(float(np.mean(rms)), 1e-10)))
    peak_db = float(20 * np.log10(ref))
    peak_time = float(grid_times[int(np.argmax(smoothed))])

    # Normalised loudness score in [0,1], used by the planner's energy arc.
    quiet_floor_db = -40.0
    score = float(np.clip((mean_db - quiet_floor_db) / (0 - quiet_floor_db), 0.0, 1.0))

    return EnergyResult(
        curve=curve.astype(np.float32),
        curve_hz=CURVE_HZ,
        energetic_section=(round(start, 3), round(end, 3)),
        mean_db=round(mean_db, 2),
        peak_time=round(peak_time, 3),
        score=round(score, 4),
        crest_factor=round(peak_db - mean_db, 2),
    )
