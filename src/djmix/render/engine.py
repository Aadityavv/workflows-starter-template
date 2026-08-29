"""The rendering engine: a validated plan in, one audio buffer out.

Fully deterministic and completely AI-free. It accepts only a
`ValidatedMixPlan`, which only the validator can construct, so there is no way
to get audio out of an unvalidated plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from djmix.audio.io import SAMPLE_RATE, AudioBuffer, load_audio
from djmix.models import TrackAnalysis, ValidatedMixPlan
from djmix.render.crossfade import align_phase, equal_power_curves
from djmix.render.master import MasterReport, master, normalize_track
from djmix.render.timestretch import MAX_STRETCH, build_tempo_groups, stretch_ratio, time_stretch

HEAD_FADE_SEC = 1.0
TAIL_FADE_SEC = 3.0


@dataclass
class JunctionReport:
    """Objective quality measures for one transition.

    These exist because "it sounds fine" is not something an automated test, or
    an author who cannot listen, can honestly assert. Grid alignment error and
    seam discontinuity can be measured, so they are.
    """

    from_track: str
    into_track: str
    kind: str
    fade_sec: float
    grid_error_ms: float
    stretch_ratio_out: float
    stretch_ratio_in: float
    beatmatched: bool


@dataclass
class RenderReport:
    duration_sec: float
    sample_rate: int
    tempo_groups: list[tuple[float, int]]
    junctions: list[JunctionReport] = field(default_factory=list)
    master: MasterReport | None = None
    peak_db: float = 0.0
    worst_seam_ratio: float = 0.0


def _seconds_to_samples(t: float, sr: int) -> int:
    return int(round(t * sr))


def _seam_ratio(audio: np.ndarray, at: int, sr: int, window_ms: float = 50.0) -> float:
    """Step size at sample `at`, divided by the typical step nearby.

    A clean join sits at or below the surrounding material (~1.0). A click shows
    up as a step several times larger than anything around it.
    """
    half = max(16, int(sr * window_ms / 1000.0))
    low, high = max(1, at - half), min(audio.shape[1], at + half)
    if high - low < 8:
        return 0.0
    steps = np.abs(np.diff(audio[:, low:high], axis=1)).max(axis=0)
    local = float(np.percentile(steps, 95))
    step_at = float(np.abs(audio[:, at] - audio[:, at - 1]).max())
    return step_at / local if local > 1e-9 else 0.0


def render(
    plan: ValidatedMixPlan,
    analyses: dict[str, TrackAnalysis],
    sample_rate: int = SAMPLE_RATE,
    max_stretch: float = MAX_STRETCH,
    target_lufs: float = -14.0,
    normalize_tracks: bool = True,
) -> tuple[AudioBuffer, RenderReport]:
    steps = plan.steps
    ordered = [(s.track_id, analyses[s.track_id].bpm) for s in steps]
    groups = build_tempo_groups(ordered, max_stretch=max_stretch)

    tempo_of: dict[str, float] = {}
    for group in groups:
        for track_id in group.member_ids:
            tempo_of[track_id] = group.tempo

    # Load, pre-gain, and stretch every track once.
    rendered: dict[str, np.ndarray] = {}
    ratios: dict[str, float] = {}
    for track_id in {s.track_id for s in steps}:
        analysis = analyses[track_id]
        buffer = load_audio(analysis.source_path, sample_rate=sample_rate).to_stereo()
        samples = buffer.samples
        if normalize_tracks:
            samples = normalize_track(samples, sample_rate, target_lufs)
        ratio = stretch_ratio(analysis.bpm, tempo_of[track_id], max_stretch)
        ratios[track_id] = ratio
        rendered[track_id] = time_stretch(samples, sample_rate, ratio)

    def scaled(t: float, track_id: str) -> float:
        """Map a measured timestamp onto the stretched timeline."""
        return t / ratios[track_id]

    def grid(track_id: str) -> list[float]:
        analysis = analyses[track_id]
        source = (
            analysis.downbeat_times if len(analysis.downbeat_times) >= 4 else analysis.beat_times
        )
        return [t / ratios[track_id] for t in source]

    segments: list[tuple[str, int, int]] = []  # (track_id, start_sample, end_sample)
    fades: list[tuple[int, int, str, str]] = []  # (fade_start_in_mix, n, out_id, in_id)
    junctions: list[JunctionReport] = []

    cursor = 0
    entry = scaled(steps[0].start_at or 0.0, steps[0].track_id)

    for step in steps:
        track_id = step.track_id
        audio = rendered[track_id]
        transition = step.transition

        if transition is None:
            start = _seconds_to_samples(entry, sample_rate)
            segments.append((track_id, start, audio.shape[1]))
            break

        next_id = transition.into
        fade_sec = transition.len_sec / max(ratios[track_id], 1e-9)
        out_at = scaled(transition.out_at, track_id)
        in_at = scaled(transition.in_at, next_id)

        beatmatched = (
            transition.type == "crossfade" and abs(tempo_of[track_id] - tempo_of[next_id]) < 1e-6
        )
        if beatmatched:
            aligned = align_phase(
                out_at,
                in_at,
                grid(track_id),
                grid(next_id),
                60.0 / max(tempo_of[track_id], 1e-9),
            )
        else:
            aligned = in_at
            # An unmatched junction gets a longer blend: with no shared grid to
            # lock to, a slow fade is what keeps it from sounding like a cut.
            fade_sec = min(fade_sec * 2.0, 12.0)

        out_sample = _seconds_to_samples(out_at, sample_rate)
        fade_n = max(1, _seconds_to_samples(fade_sec, sample_rate))
        fade_n = min(fade_n, max(1, audio.shape[1] - out_sample))
        remaining_in = rendered[next_id].shape[1] - _seconds_to_samples(aligned, sample_rate)
        fade_n = min(fade_n, max(1, remaining_in))

        start = _seconds_to_samples(entry, sample_rate)
        segments.append((track_id, start, out_sample + fade_n))
        fades.append((cursor + (out_sample - start), fade_n, track_id, next_id))

        grid_error = 0.0
        if beatmatched:
            grid_error = abs(aligned - in_at) * 1000.0

        junctions.append(
            JunctionReport(
                from_track=track_id,
                into_track=next_id,
                kind=transition.type,
                fade_sec=round(fade_n / sample_rate, 3),
                grid_error_ms=round(grid_error, 2),
                stretch_ratio_out=round(ratios[track_id], 5),
                stretch_ratio_in=round(ratios[next_id], 5),
                beatmatched=beatmatched,
            )
        )

        cursor += out_sample - start
        entry = aligned

    # Lay everything into one buffer. Segments overlap by exactly their fade.
    total = 0
    positions: list[int] = []
    position = 0
    for index, (_track_id, start, end) in enumerate(segments):
        positions.append(position)
        length = max(0, end - start)
        if index < len(fades):
            position += length - fades[index][1]
        else:
            position += length
        total = max(total, positions[-1] + length)

    mix = np.zeros((2, total), dtype=np.float64)
    for index, (track_id, start, end) in enumerate(segments):
        chunk = rendered[track_id][:, max(0, start) : max(0, end)]
        if chunk.shape[1] == 0:
            continue
        gains = np.ones(chunk.shape[1])

        # Fade in over the overlap this track shares with the previous one.
        if index > 0:
            fade_n = fades[index - 1][1]
            fade_n = min(fade_n, chunk.shape[1])
            _, gain_in = equal_power_curves(fade_n)
            gains[:fade_n] = gain_in
        # Fade out over the overlap it shares with the next.
        if index < len(fades):
            fade_n = min(fades[index][1], chunk.shape[1])
            gain_out, _ = equal_power_curves(fade_n)
            gains[-fade_n:] *= gain_out

        at = positions[index]
        width = min(chunk.shape[1], total - at)
        if width > 0:
            mix[:, at : at + width] += chunk[:, :width] * gains[:width]

    # Top and tail so the mix does not start or end on a discontinuity.
    head = min(_seconds_to_samples(HEAD_FADE_SEC, sample_rate), mix.shape[1])
    if head > 1:
        mix[:, :head] *= np.linspace(0.0, 1.0, head)
    tail = min(_seconds_to_samples(TAIL_FADE_SEC, sample_rate), mix.shape[1])
    if tail > 1:
        mix[:, -tail:] *= np.linspace(1.0, 0.0, tail)

    mastered, master_report = master(mix.astype(np.float32), sample_rate, target_lufs)

    # Seam quality, measured relative to local content rather than against an
    # absolute threshold. Percussive material legitimately contains very large
    # single-sample steps (a kick click or a hi-hat is near-Nyquist), so an
    # absolute step limit would flag ordinary audio as a defect. What actually
    # indicates a bad join is a step at the junction that is far larger than the
    # steps just either side of it.
    worst = 0.0
    for index in range(len(fades)):
        if index + 1 >= len(positions):
            continue
        at = positions[index + 1]
        if at <= 0 or at >= mastered.shape[1] - 1:
            continue
        worst = max(worst, _seam_ratio(mastered, at, sample_rate))

    report = RenderReport(
        duration_sec=round(mastered.shape[1] / sample_rate, 3),
        sample_rate=sample_rate,
        tempo_groups=[(round(g.tempo, 2), len(g.member_ids)) for g in groups],
        junctions=junctions,
        master=master_report,
        peak_db=master_report.peak_db,
        worst_seam_ratio=round(worst, 3),
    )
    return AudioBuffer(mastered, sample_rate), report
