"""Camelot wheel: harmonic-compatibility arithmetic for key names.

Pure functions over strings -- no audio, no state -- so this is exhaustively
testable and shared by the analyzer (which labels tracks) and the planner (which
scores adjacent pairs).
"""

from __future__ import annotations

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "Fb": "E",
    "E#": "F",
    "B#": "C",
}

# Standard Camelot numbering: majors are the "B" ring, minors the "A" ring, and
# stepping +1 around the ring is a move of a perfect fifth.
_MAJOR_CAMELOT = {
    "C": 8,
    "G": 9,
    "D": 10,
    "A": 11,
    "E": 12,
    "B": 1,
    "F#": 2,
    "C#": 3,
    "G#": 4,
    "D#": 5,
    "A#": 6,
    "F": 7,
}
_MINOR_CAMELOT = {
    "A": 8,
    "E": 9,
    "B": 10,
    "F#": 11,
    "C#": 12,
    "G#": 1,
    "D#": 2,
    "A#": 3,
    "F": 4,
    "C": 5,
    "G": 6,
    "D": 7,
}


def normalize_tonic(tonic: str) -> str:
    t = tonic.strip()
    t = t[0].upper() + t[1:]
    return ENHARMONIC.get(t, t)


def to_camelot(tonic: str, mode: str) -> str:
    """'A', 'minor' -> '8A'. Raises on an unknown pitch class."""
    t = normalize_tonic(tonic)
    is_minor = mode.lower().startswith("min")
    table = _MINOR_CAMELOT if is_minor else _MAJOR_CAMELOT
    if t not in table:
        raise ValueError(f"unknown tonic {tonic!r}")
    return f"{table[t]}{'A' if is_minor else 'B'}"


def parse_camelot(code: str) -> tuple[int, str]:
    code = code.strip().upper()
    return int(code[:-1]), code[-1]


def camelot_distance(a: str, b: str) -> float:
    """Harmonic distance in [0, 1.5]. Lower is a smoother blend.

    0.00  same key
    0.15  adjacent on the wheel (a fifth apart), or relative major/minor
    0.50  two steps -- the classic "energy boost" move, usable but noticeable
    1.00  unrelated
    1.50  tritone -- the worst case, and worth flagging distinctly
    """
    if a == b:
        return 0.0
    na, la = parse_camelot(a)
    nb, lb = parse_camelot(b)
    ring = min((na - nb) % 12, (nb - na) % 12)
    if la == lb:
        return {0: 0.0, 1: 0.15, 2: 0.5}.get(ring, 1.5 if ring == 6 else 1.0)
    # Crossing rings: same number is the relative major/minor pair.
    if ring == 0:
        return 0.15
    if ring == 1:
        return 0.6
    return 1.5 if ring == 6 else 1.0


def is_compatible(a: str, b: str, threshold: float = 0.15) -> bool:
    return camelot_distance(a, b) <= threshold
