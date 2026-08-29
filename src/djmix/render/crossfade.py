"""Equal-power crossfade curves and beat-phase alignment."""

from __future__ import annotations

import bisect

import numpy as np


def equal_power_curves(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine/sine gain pair whose squares sum to exactly 1 at every sample.

    A linear crossfade dips about 3 dB in the middle, because two uncorrelated
    signals sum in power, not amplitude. Equal-power holds total power constant,
    which is why the transition does not audibly duck.
    """
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)


def align_phase(
    out_at: float,
    in_at: float,
    outgoing_grid: list[float],
    incoming_grid: list[float],
    beat_period: float,
) -> float:
    """Nudge the incoming entry so its next beat coincides with the outgoing one.

    The planner already snaps to the grid, so this is a safety net -- it mostly
    protects the LLM path, where a timestamp may be beat-snapped but land on a
    different phase. The correction is wrapped to at most half a beat so it can
    never shift the entry into a different bar.
    """
    if not outgoing_grid or not incoming_grid or beat_period <= 0:
        return in_at

    def next_grid_point(grid: list[float], t: float) -> float | None:
        i = bisect.bisect_left(grid, t)
        return float(grid[i]) if i < len(grid) else None

    out_next = next_grid_point(outgoing_grid, out_at)
    in_next = next_grid_point(incoming_grid, in_at)
    if out_next is None or in_next is None:
        return in_at

    phase_out = out_next - out_at
    phase_in = in_next - in_at
    shift = phase_out - phase_in
    # Wrap into (-beat/2, +beat/2].
    shift = (shift + beat_period / 2) % beat_period - beat_period / 2
    return max(0.0, in_at + shift)
