"""On-disk analysis cache.

Analysis is by far the most expensive step (HPSS + CQT + a recurrence matrix),
and mixes are generated repeatedly over the same library, so results must
survive between runs. The key is the file's content hash plus the analyzer
version, so re-encoding a file invalidates it and bumping the DSP invalidates
everything -- neither mtime nor path is trusted.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from djmix.models import ANALYZER_VERSION, TrackAnalysis

DEFAULT_CACHE_DIR = Path(".djmix-cache")


class AnalysisCache:
    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIR):
        self.directory = Path(directory)

    def _path(self, content_hash: str) -> Path:
        return self.directory / f"{content_hash}-v{ANALYZER_VERSION}.json"

    def get(self, content_hash: str) -> TrackAnalysis | None:
        path = self._path(content_hash)
        if not path.is_file():
            return None
        try:
            return TrackAnalysis.model_validate_json(path.read_text())
        except Exception:
            # A corrupt or stale-shaped entry is a cache miss, never a crash.
            return None

    def put(self, analysis: TrackAnalysis) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(analysis.content_hash)
        # Atomic replace: a killed process must not leave a half-written entry
        # that later reads as valid JSON.
        fd, tmp = tempfile.mkstemp(dir=str(self.directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(analysis.model_dump_json(indent=2))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def clear(self) -> int:
        if not self.directory.is_dir():
            return 0
        removed = 0
        for path in self.directory.glob("*.json"):
            path.unlink()
            removed += 1
        return removed

    def entries(self) -> list[Path]:
        return sorted(self.directory.glob("*.json")) if self.directory.is_dir() else []
