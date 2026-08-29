"""Deterministic percussive test fixtures with exact known ground truth.

Why this file exists: a synthetic sine or pulse train is NOT a valid test signal
for beat tracking. During prototyping such a signal produced a false 0.0 BPM
reading, because onset detection keys off broadband percussive attack
transients, which a steady tone does not have. So these fixtures synthesize a
real drum kit -- a pitch-swept kick with a click transient, a noise-burst snare
with a tonal body, and hi-hats -- over a harmonic bed in a known key.

Everything is seeded, so fixtures regenerate byte-identically and the ground
truth (tempo, key, and which section is loudest) is exact rather than annotated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050
FIXTURE_DIR = Path(__file__).parent / "audio"

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Per-bar chord progressions, as (semitones above root, quality), keyed by
# section role. Two properties are deliberate and were both arrived at by
# watching an analyzer fail without them:
#
# 1. Tonal unambiguity. Every progression resolves to the tonic, and the minor
#    mode uses a MAJOR dominant so its leading tone pins the key down. An
#    earlier, more "interesting" progression (Am-Em-G-Dm) weighted G exactly as
#    heavily as A, and the key estimator quite correctly answered G major. A
#    fixture that does not establish a key cannot grade a key detector.
#
# 2. Verses and choruses use DIFFERENT harmony. When every section shared one
#    4-bar loop, the recurrence matrix locked onto the loop instead of the song
#    form -- it produced a clean period-16-beat labelling that was completely
#    unrelated to intro/verse/chorus. Sections differing only in gain give
#    chroma nothing to separate, which is not how real music behaves.
SECTION_PROGRESSIONS: dict[str, dict[str, list[tuple[int, str]]]] = {
    "maj": {
        "tonic": [(0, "maj"), (0, "maj"), (0, "maj"), (7, "maj")],
        "verse": [(0, "maj"), (9, "min"), (5, "maj"), (7, "maj")],
        "chorus": [(5, "maj"), (7, "maj"), (0, "maj"), (0, "maj")],
    },
    "min": {
        "tonic": [(0, "min"), (0, "min"), (0, "min"), (7, "maj")],
        "verse": [(0, "min"), (8, "maj"), (3, "maj"), (10, "maj")],
        "chorus": [(5, "min"), (7, "maj"), (0, "min"), (0, "min")],
    },
}
# Which progression each structural role uses.
ROLE_PROGRESSION = {
    "intro": "tonic",
    "verse": "verse",
    "chorus": "chorus",
    "outro": "tonic",
}
CHORD_INTERVALS = {"maj": (0, 4, 7), "min": (0, 3, 7)}


def _env(n: int, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    """Percussive envelope: near-instant attack, exponential decay."""
    a = max(1, int(attack * sr))
    t = np.arange(n, dtype=np.float64)
    env = np.exp(-t / max(1.0, decay * sr))
    env[:a] *= np.linspace(0.0, 1.0, a)
    return env


def _kick(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Pitch-swept sine plus a broadband click.

    The click is the part that matters: it is what gives the onset detector a
    sharp spectral-flux spike to latch onto.
    """
    n = int(0.28 * sr)
    t = np.arange(n) / sr
    freq = 42 + 110 * np.exp(-t * 38)  # 152 Hz -> 42 Hz sweep
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * _env(n, 0.0005, 0.075, sr)
    click_n = int(0.006 * sr)
    click = rng.standard_normal(click_n) * np.linspace(1.0, 0.0, click_n) ** 2
    out = body
    out[:click_n] += click * 0.55
    return (out * 0.95).astype(np.float64)


def _snare(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Filtered noise burst plus two tonal modes, as a real snare has."""
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    # One-pole high-pass so it sits above the kick.
    hp = np.zeros(n)
    prev_x = prev_y = 0.0
    alpha = 0.86
    for i in range(n):
        hp[i] = alpha * (prev_y + noise[i] - prev_x)
        prev_x, prev_y = noise[i], hp[i]
    tone = 0.5 * np.sin(2 * np.pi * 185 * t) + 0.3 * np.sin(2 * np.pi * 331 * t)
    return ((hp * 0.8 + tone * 0.4) * _env(n, 0.0005, 0.055, sr) * 0.7).astype(np.float64)


def _hat(rng: np.random.Generator, closed: bool = True, sr: int = SR) -> np.ndarray:
    n = int((0.055 if closed else 0.19) * sr)
    noise = rng.standard_normal(n)
    # Crude high-shelf: differencing emphasises the top end.
    shaped = np.diff(noise, prepend=0.0)
    return (shaped * _env(n, 0.0003, 0.014 if closed else 0.075, sr) * 0.3).astype(np.float64)


def _lead(midi: int, dur: float, sr: int = SR) -> np.ndarray:
    """Triangle-ish melody voice, present only in choruses.

    Gives the timbre (MFCC) side of the segmentation something to separate, in
    addition to the harmonic change -- real choruses differ in arrangement, not
    just in level.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    out = np.zeros(n)
    for harmonic, amp in ((1, 1.0), (3, 1 / 9), (5, 1 / 25), (7, 1 / 49)):
        out += amp * np.sin(2 * np.pi * f0 * harmonic * t)
    env = np.ones(n)
    ramp = int(0.03 * sr)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return out / max(1e-9, np.abs(out).max()) * env * 0.20


def _chord(root_midi: int, quality: str, dur: float, sr: int = SR) -> np.ndarray:
    """Sawtooth-ish pad. Gives the chroma/key estimator real harmonic content."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for interval in CHORD_INTERVALS[quality]:
        f0 = 440.0 * 2 ** ((root_midi + interval - 69) / 12)
        for harmonic in range(1, 7):
            out += np.sin(2 * np.pi * f0 * harmonic * t) / (harmonic**1.6)
    env = np.ones(n)
    ramp = int(0.02 * sr)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return out / max(1e-9, np.abs(out).max()) * env * 0.16


def _bass(root_midi: int, dur: float, sr: int = SR) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = 440.0 * 2 ** ((root_midi - 12 - 69) / 12)
    out = np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(4 * np.pi * f0 * t)
    env = np.ones(n)
    ramp = int(0.01 * sr)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return out * env * 0.22


@dataclass
class Section:
    """One structural block. `gain` is what makes the energy curve meaningful,
    and repeating a `label` is what gives the recurrence matrix something to find."""

    label: str
    bars: int
    gain: float
    drums: bool = True
    snare: bool = True
    hats: bool = True
    lead: bool = False


@dataclass
class FixtureSpec:
    name: str
    bpm: float
    root: str
    mode: str  # "maj" | "min"
    sections: list[Section] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.root} {'major' if self.mode == 'maj' else 'minor'}"


def _root_midi(root: str) -> int:
    return 60 + NOTE_NAMES.index(root)


def render_fixture(spec: FixtureSpec, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Render one fixture and return the audio plus its exact ground truth."""
    rng = np.random.default_rng(abs(hash(spec.name)) % (2**31))
    beat = 60.0 / spec.bpm
    bar = 4 * beat

    kick, snare, hat = _kick(rng), _snare(rng), _hat(rng)
    mode_progressions = SECTION_PROGRESSIONS[spec.mode]
    root_midi = _root_midi(spec.root)

    total_bars = sum(s.bars for s in spec.sections)
    total = int(np.ceil(total_bars * bar * sr)) + sr
    out = np.zeros(total)

    def place(sample: np.ndarray, at_sec: float, gain: float) -> None:
        i = int(round(at_sec * sr))
        j = min(total, i + len(sample))
        if i < total:
            out[i:j] += sample[: j - i] * gain

    truth_sections, bar_index = [], 0
    for section in spec.sections:
        s_start = bar_index * bar
        progression = mode_progressions[ROLE_PROGRESSION[section.label]]
        for b in range(section.bars):
            bar_start = (bar_index + b) * bar
            degree, quality = progression[b % len(progression)]
            place(_chord(root_midi + degree, quality, bar, sr), bar_start, section.gain)
            place(_bass(root_midi + degree, bar, sr), bar_start, section.gain)
            if section.lead:
                place(_lead(root_midi + degree + 12, bar, sr), bar_start, section.gain * 0.5)
            if not section.drums:
                continue
            # Kick on 1 and 3 with an off-beat push; snare on 2 and 4; hats on 8ths.
            for beat_pos in (0.0, 2.0, 2.75):
                place(kick, bar_start + beat_pos * beat, section.gain)
            if section.snare:
                for beat_pos in (1.0, 3.0):
                    place(snare, bar_start + beat_pos * beat, section.gain)
            if section.hats:
                for eighth in range(8):
                    if eighth % 2 == 1 or True:
                        place(hat, bar_start + eighth * 0.5 * beat, section.gain * 0.6)
        bar_index += section.bars
        truth_sections.append(
            {
                "label": section.label,
                "start": round(s_start, 3),
                "end": round(bar_index * bar, 3),
                "gain": section.gain,
            }
        )

    peak = np.abs(out).max()
    if peak > 0:
        out = out / peak * 0.89

    loudest = max(truth_sections, key=lambda s: s["gain"])
    truth = {
        "name": spec.name,
        "bpm": spec.bpm,
        "key": spec.key,
        "duration_sec": round(len(out) / sr, 3),
        "sections": truth_sections,
        "loudest_section": [loudest["start"], loudest["end"]],
        "bar_sec": round(bar, 4),
    }
    return out.astype(np.float32), truth


def _verse_chorus(bars_intro: int = 4) -> list[Section]:
    """Intro / verse / chorus / verse / chorus / outro.

    The chorus label repeats with identical instrumentation so the recurrence
    matrix has a genuine repeat to detect, and the gain difference gives the RMS
    energy curve an unambiguous peak.
    """
    return [
        Section("intro", bars_intro, 0.42, drums=True, snare=False, hats=True),
        Section("verse", 8, 0.62),
        Section("chorus", 8, 1.00, lead=True),
        Section("verse", 8, 0.62),
        Section("chorus", 8, 1.00, lead=True),
        Section("outro", 4, 0.40, snare=False),
    ]


SPECS = [
    FixtureSpec("track_090_amin", 90.0, "A", "min", _verse_chorus()),
    FixtureSpec("track_100_cmaj", 100.0, "C", "maj", _verse_chorus()),
    FixtureSpec("track_120_gmaj", 120.0, "G", "maj", _verse_chorus()),
    FixtureSpec("track_128_fmin", 128.0, "F", "min", _verse_chorus()),
    FixtureSpec("track_140_dmin", 140.0, "D", "min", _verse_chorus()),
    FixtureSpec("track_174_emaj", 174.0, "E", "maj", _verse_chorus(8)),
]


def generate_all(out_dir: Path = FIXTURE_DIR) -> list[dict]:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    truths = []
    for spec in SPECS:
        audio, truth = render_fixture(spec)
        path = out_dir / f"{spec.name}.flac"
        sf.write(str(path), audio, SR)
        truth["path"] = path.name
        truths.append(truth)
    (out_dir / "ground_truth.json").write_text(json.dumps(truths, indent=2) + "\n")
    return truths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=FIXTURE_DIR)
    args = ap.parse_args()
    for t in generate_all(args.out):
        print(f"{t['path']:24s} {t['bpm']:6.1f} BPM  {t['key']:10s} {t['duration_sec']:7.1f}s")
