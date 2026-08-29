"""Shared fixtures.

Two rules enforced here for every test:

* No network. An autouse fixture makes socket creation raise, so an accidental
  live API call fails loudly instead of silently costing money or making CI
  depend on a provider being up.
* No API keys. Provider keys are stripped from the environment so the LLM tests
  genuinely exercise the mock path.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

FIXTURE_AUDIO = Path(__file__).parent / "fixtures" / "audio"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("network access is not allowed in tests")

    monkeypatch.setattr(socket, "socket", blocked)
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _fixture_audio() -> None:
    """Materialise the test audio if it is not already on disk.

    The audio is generated rather than committed: generate.py is seeded and
    byte-deterministic (asserted in test_fixtures.py), so storing ~14 MB of
    FLAC would add weight without adding reproducibility.
    """
    import generate as G

    expected = {f"{spec.name}.flac" for spec in G.SPECS}
    present = {p.name for p in FIXTURE_AUDIO.glob("*.flac")} if FIXTURE_AUDIO.is_dir() else set()
    if not expected.issubset(present) or not (FIXTURE_AUDIO / "ground_truth.json").is_file():
        G.generate_all(FIXTURE_AUDIO)


@pytest.fixture(scope="session")
def ground_truth(_fixture_audio) -> list[dict]:
    return json.loads((FIXTURE_AUDIO / "ground_truth.json").read_text())


@pytest.fixture(scope="session")
def analyses(tmp_path_factory, _fixture_audio) -> list:
    """Analyse the fixture library once per session -- it is the slow step."""
    from djmix.audio.analysis import analyze_directory
    from djmix.audio.cache import AnalysisCache

    cache = AnalysisCache(tmp_path_factory.mktemp("djmix-cache"))
    result, failures = analyze_directory(FIXTURE_AUDIO, cache=cache)
    assert not failures, f"fixture analysis failed: {failures}"
    return result


@pytest.fixture(scope="session")
def analyses_by_id(analyses) -> dict:
    return {a.track_id: a for a in analyses}


@pytest.fixture(scope="session")
def truth_for(analyses, ground_truth) -> dict:
    """Map each analysis to the ground truth of the file it came from."""
    by_name = {t["path"]: t for t in ground_truth}
    return {a.track_id: by_name[Path(a.source_path).name] for a in analyses}
