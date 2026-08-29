"""Structural segmentation via a recurrence (self-similarity) matrix.

Follows the McFee-Ellis spectral-clustering recipe: build a beat-synchronous
feature stack, form an affinity matrix combining long-range repetition with
local timbral continuity, then cluster its Laplacian eigenvectors.

This is the least reliable of the four analyzers, and the design deliberately
limits the blast radius: the chorus estimate is one anchor among many (beats,
downbeats, energetic section), so a poor estimate shifts where a transition
lands rather than breaking the mix. When clustering degenerates, the module
says so and falls back to the energy peak instead of inventing a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

MIN_CHORUS_SEC = 8.0
MAX_CHORUS_SEC = 60.0
# A chorus that covers most of a track is not a chorus, it is a segmentation
# failure. When clustering collapses the song into one long block the absolute
# cap above is not enough -- 60 s of an 80 s track is still 75% of it -- so the
# estimate is also capped as a fraction of the duration, and an estimate that
# needs that much clamping is treated as unreliable rather than trusted.
MAX_CHORUS_FRACTION = 0.4
N_SEGMENT_TYPES = 5
# Raw per-beat cluster labels flicker, which yields dozens of one-beat
# "sections". That is not just untidy: every segment boundary becomes an anchor
# for the provenance validator, and a hundred spurious anchors would make almost
# any timestamp explainable. Smoothing and merging keeps the anchor set small
# and musically meaningful, which is what gives the validator its teeth.
LABEL_SMOOTH_BEATS = 9
MIN_SEGMENT_SEC = 6.0
# Never merge a track down past this many sections: the cascade below always has
# a shortest segment to absorb, so without a floor it happily collapses a whole
# song into one block and silently trips the energy-peak fallback.
MIN_SEGMENT_COUNT = 4


@dataclass
class StructureResult:
    segments: list[tuple[float, float, int]]
    chorus_estimate: tuple[float, float]
    confident: bool
    warnings: list[str] = field(default_factory=list)


def _laplacian_labels(affinity: np.ndarray, k: int) -> np.ndarray:
    """Spectral clustering over the normalised graph Laplacian."""
    degree = affinity.sum(axis=1)
    degree[degree <= 0] = 1e-9
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = affinity * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1][:k]
    embedding = eigenvectors[:, order]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-9)
    return _kmeans(embedding, k)


def _kmeans(points: np.ndarray, k: int, iterations: int = 50) -> np.ndarray:
    """k-means++ with a fixed seed. Deterministic by construction -- the whole
    analysis pipeline must give the same answer on the same bytes."""
    rng = np.random.default_rng(0)
    n = len(points)
    k = min(k, n)
    centers = [points[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 1e-12 else np.full(n, 1.0 / n)
        centers.append(points[rng.choice(n, p=probs)])
    centroids = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = points[labels == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return labels


def _smooth_labels(labels: np.ndarray, width: int) -> np.ndarray:
    """Median-filter the per-beat label sequence to remove single-beat flicker."""
    if width < 3 or len(labels) < width:
        return labels
    half = width // 2
    padded = np.pad(labels, half, mode="edge")
    out = np.empty_like(labels)
    for i in range(len(labels)):
        window = padded[i : i + width]
        values, counts = np.unique(window, return_counts=True)
        out[i] = values[np.argmax(counts)]
    return out


def _merge_short_segments(
    segments: list[tuple[float, float, int]], min_sec: float
) -> list[tuple[float, float, int]]:
    """Absorb sub-minimum segments into whichever neighbour they are closer to,
    then coalesce adjacent runs that ended up sharing a label."""
    if len(segments) <= 1:
        return segments
    changed = True
    while changed and len(segments) > MIN_SEGMENT_COUNT:
        changed = False
        for i, (start, end, _label) in enumerate(segments):
            if end - start >= min_sec:
                continue
            prev_len = segments[i - 1][1] - segments[i - 1][0] if i > 0 else -1.0
            next_len = segments[i + 1][1] - segments[i + 1][0] if i + 1 < len(segments) else -1.0
            if prev_len >= next_len and i > 0:
                segments[i - 1] = (segments[i - 1][0], end, segments[i - 1][2])
            elif i + 1 < len(segments):
                segments[i + 1] = (start, segments[i + 1][1], segments[i + 1][2])
            else:
                break
            segments.pop(i)
            changed = True
            break

    coalesced = [segments[0]]
    for start, end, label in segments[1:]:
        if label == coalesced[-1][2]:
            coalesced[-1] = (coalesced[-1][0], end, label)
        else:
            coalesced.append((start, end, label))
    return [(round(a, 3), round(b, 3), c) for a, b, c in coalesced]


def analyze_structure(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    energetic_section: tuple[float, float],
    hop_length: int = 512,
    harmonic: np.ndarray | None = None,
) -> StructureResult:
    duration = len(y) / sr
    warnings: list[str] = []

    beats = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
    beats = np.asarray([b for b in beats if b >= 0], dtype=int)
    if len(beats) < 16:
        return StructureResult(
            [(0.0, duration, 0)],
            energetic_section,
            False,
            ["too few beats for structural analysis; using the energy peak"],
        )

    try:
        if harmonic is None:
            harmonic = librosa.effects.harmonic(y, margin=2.0)
        chroma = librosa.feature.chroma_cens(y=harmonic, sr=sr, hop_length=hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beats, aggregate=np.mean)

        stacked = librosa.feature.stack_memory(chroma_sync, n_steps=4, delay=2)
        # Long-range repetition: which beats sound like which other beats.
        recurrence = librosa.segment.recurrence_matrix(stacked, width=3, mode="affinity", sym=True)
        recurrence = librosa.segment.path_enhance(recurrence, n=9)
        # Local continuity: consecutive beats with similar timbre belong together.
        path = librosa.segment.recurrence_matrix(
            mfcc_sync, width=3, mode="affinity", sym=True, metric="cosine"
        )
        n = min(recurrence.shape[0], path.shape[0])
        affinity = 0.6 * recurrence[:n, :n] + 0.4 * path[:n, :n]
        labels = _laplacian_labels(affinity, N_SEGMENT_TYPES)
    except Exception as exc:  # pragma: no cover - degenerate audio
        return StructureResult(
            [(0.0, duration, 0)],
            energetic_section,
            False,
            [f"structural clustering failed ({type(exc).__name__}); using the energy peak"],
        )

    # librosa.util.sync emits one column per inter-beat interval, which is one
    # more than the number of beat instants, so labels can outrun beat_seconds.
    beat_seconds = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    usable = min(len(labels), len(beat_seconds))
    labels = labels[:usable]
    beat_seconds = beat_seconds[:usable]

    labels = _smooth_labels(labels, LABEL_SMOOTH_BEATS)

    segments: list[tuple[float, float, int]] = []
    start_i = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_i]:
            start = float(beat_seconds[start_i])
            end = float(beat_seconds[i]) if i < len(beat_seconds) else duration
            if end > start:
                segments.append((round(start, 3), round(end, 3), int(labels[start_i])))
            start_i = i

    segments = _merge_short_segments(segments, MIN_SEGMENT_SEC)

    if len(segments) < 2 or len({s[2] for s in segments}) < 2:
        return StructureResult(
            segments or [(0.0, duration, 0)],
            energetic_section,
            False,
            ["segmentation produced a single section; using the energy peak as chorus"],
        )

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    def mean_rms(start: float, end: float) -> float:
        mask = (rms_times >= start) & (rms_times < end)
        return float(rms[mask].mean()) if mask.any() else 0.0

    # Score each *label class*: the chorus is the section type that recurs, covers
    # a lot of the track, and is loud. Then pick that class's loudest instance.
    class_scores: dict[int, float] = {}
    for label in sorted({s[2] for s in segments}):
        instances = [s for s in segments if s[2] == label]
        coverage = sum(e - st for st, e, _ in instances) / duration
        repeats = len(instances)
        loudness = float(np.mean([mean_rms(st, e) for st, e, _ in instances]))
        class_scores[label] = coverage * repeats * loudness

    best_label = max(class_scores, key=class_scores.get)
    instances = [s for s in segments if s[2] == best_label]
    # Skip a chorus "instance" in the first 10% -- that is nearly always the intro.
    late = [s for s in instances if s[0] >= 0.10 * duration] or instances
    chorus = max(late, key=lambda s: mean_rms(s[0], s[1]))

    start, end = chorus[0], chorus[1]
    span_limit = min(MAX_CHORUS_SEC, MAX_CHORUS_FRACTION * duration)

    if end - start > span_limit:
        # The chosen "chorus" spans an implausible share of the track, which
        # means the clustering did not really separate the sections. Trusting a
        # truncated version of it would put the transition somewhere arbitrary;
        # the measured energy peak is the better answer and is what the caller
        # already falls back to elsewhere.
        return StructureResult(
            segments=segments,
            chorus_estimate=energetic_section,
            confident=False,
            warnings=[
                *warnings,
                f"chorus candidate spans {end - start:.0f}s of a {duration:.0f}s track; "
                "segmentation did not separate sections, using the energy peak",
            ],
        )

    if end - start < MIN_CHORUS_SEC:
        end = min(duration, start + MIN_CHORUS_SEC)
        warnings.append("chorus candidate shorter than the minimum; extended")

    return StructureResult(
        segments=segments,
        chorus_estimate=(round(start, 3), round(end, 3)),
        confident=True,
        warnings=warnings,
    )
