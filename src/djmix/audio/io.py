"""Audio decoding and file identity.

Decoding order: libsndfile (wav/flac/mp3/ogg natively), then ffmpeg for anything
it rejects -- in practice m4a/aac. The ffmpeg binary comes from imageio-ffmpeg,
a pip wheel, so there is no system-package prerequisite.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
ANALYSIS_SAMPLE_RATE = 22050  # librosa's default; halves analysis cost with no
# meaningful loss for beat/chroma/RMS features.
SUPPORTED_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class AudioLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioBuffer:
    """Interleaved-free audio: shape (channels, samples), float32, -1..1."""

    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    @property
    def duration_sec(self) -> float:
        return self.samples.shape[1] / self.sample_rate

    def to_mono(self) -> np.ndarray:
        return self.samples.mean(axis=0) if self.channels > 1 else self.samples[0]

    def to_stereo(self) -> AudioBuffer:
        if self.channels == 2:
            return self
        if self.channels == 1:
            return AudioBuffer(np.vstack([self.samples[0], self.samples[0]]), self.sample_rate)
        return AudioBuffer(np.vstack([self.samples[:2]]), self.sample_rate)


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """Prefer a system ffmpeg; fall back to the wheel-shipped static binary."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def content_hash(path: str | Path) -> str:
    """SHA-256 of the file bytes. Half the analysis cache key -- the other half is
    the analyzer version, so bumping the DSP invalidates stale results."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_uploadable(path: str | Path) -> None:
    """Ingestion guard: format and size limits, enforced before any decode."""
    p = Path(path)
    if not p.is_file():
        raise AudioLoadError(f"not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise AudioLoadError(
            f"unsupported format {p.suffix!r}; allowed: {sorted(SUPPORTED_SUFFIXES)}"
        )
    size = p.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise AudioLoadError(
            f"{p.name} is {size / 1e6:.0f} MB; limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB"
        )
    if size == 0:
        raise AudioLoadError(f"{p.name} is empty")


def _decode_with_ffmpeg(path: Path, sample_rate: int) -> AudioBuffer:
    exe = ffmpeg_path()
    if exe is None:
        raise AudioLoadError(
            f"cannot decode {path.name}: libsndfile rejected it and no ffmpeg is available"
        )
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "decoded.wav"
        proc = subprocess.run(
            [
                exe,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_f32le",
                "-y",
                str(wav),
            ],
            capture_output=True,
        )
        if proc.returncode != 0 or not wav.exists():
            raise AudioLoadError(
                f"ffmpeg failed to decode {path.name}: {proc.stderr.decode()[:300]}"
            )
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    return AudioBuffer(np.ascontiguousarray(data.T), sr)


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> AudioBuffer:
    """Decode to float32 at `sample_rate`, preserving channel count."""
    p = Path(path)
    check_uploadable(p)
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        buf = AudioBuffer(np.ascontiguousarray(data.T), sr)
    except Exception:
        buf = _decode_with_ffmpeg(p, sample_rate)
    if buf.sample_rate != sample_rate:
        buf = resample(buf, sample_rate)
    if not np.isfinite(buf.samples).all():
        raise AudioLoadError(f"{p.name} decoded to non-finite samples")
    return buf


def resample(buf: AudioBuffer, target_sr: int) -> AudioBuffer:
    if buf.sample_rate == target_sr:
        return buf
    import librosa

    out = librosa.resample(buf.samples, orig_sr=buf.sample_rate, target_sr=target_sr, axis=-1)
    return AudioBuffer(np.ascontiguousarray(out.astype(np.float32)), target_sr)


def load_mono_for_analysis(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono at the analysis rate -- what every DSP routine in audio/ consumes."""
    buf = load_audio(path, sample_rate=ANALYSIS_SAMPLE_RATE)
    return buf.to_mono(), buf.sample_rate


def write_audio(path: str | Path, buf: AudioBuffer, bitrate: str = "320k") -> Path:
    """Write WAV/FLAC via libsndfile; compressed formats via ffmpeg."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    interleaved = buf.samples.T
    if p.suffix.lower() in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        sf.write(str(p), interleaved, buf.sample_rate)
        return p
    exe = ffmpeg_path()
    if exe is None:
        raise AudioLoadError(f"writing {p.suffix} needs ffmpeg, which is unavailable")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "render.wav"
        sf.write(str(wav), interleaved, buf.sample_rate)
        proc = subprocess.run(
            [exe, "-nostdin", "-v", "error", "-i", str(wav), "-b:a", bitrate, "-y", str(p)],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise AudioLoadError(f"ffmpeg failed to encode {p.name}: {proc.stderr.decode()[:300]}")
    return p
