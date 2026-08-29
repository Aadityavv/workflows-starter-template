"""Web layer tests.

These exercise the HTTP surface against the real analysis and render code --
no mocking of the audio pipeline, because the point of the API is that it is a
thin client of those modules.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from djmix.web.app import create_app

FIXTURE_AUDIO = Path(__file__).parent / "fixtures" / "audio"


def _wait(client: TestClient, job_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.fixture(scope="module")
def client(tmp_path_factory, analyses) -> TestClient:
    """A server whose cache is pre-warmed from the session analyses.

    Re-analysing inside the web tests would double the suite's runtime for no
    extra coverage -- the analysis path is tested directly elsewhere.
    """
    root = tmp_path_factory.mktemp("web")
    app = create_app(root)
    uploads = root / "uploads"
    from djmix.audio.cache import AnalysisCache

    cache = AnalysisCache(root / "cache")
    for analysis in analyses:
        # Copy the audio in and repoint the analysis at its new home, so the
        # server's library entries reference files that actually exist.
        destination = uploads / Path(analysis.source_path).name
        shutil.copy(analysis.source_path, destination)
        cache.put(analysis.model_copy(update={"source_path": str(destination)}))
    return TestClient(app)


def test_config_lists_occasions_and_limits(client):
    body = client.get("/api/config").json()
    assert "workout" in body["occasions"]
    assert ".mp3" in body["supported_uploads"]
    assert body["tier"]["max_tracks_per_mix"] > 0


def test_library_reports_measured_values(client):
    tracks = client.get("/api/tracks").json()["tracks"]
    assert len(tracks) >= 5
    for track in tracks:
        assert track["bpm"] > 0
        assert track["camelot"]
        assert 0.0 <= track["energy"] <= 1.0
        assert isinstance(track["bpm_ambiguous"], bool)


def test_index_page_serves(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "djmix" in response.text
    assert "<script>" in response.text


def test_favicon_does_not_404(client):
    assert client.get("/favicon.ico").status_code == 200


def test_rejects_unsupported_upload(client):
    response = client.post(
        "/api/upload", files={"files": ("notes.txt", b"not audio", "text/plain")}
    )
    body = response.json()
    assert body["job_id"] is None
    assert body["rejected"][0]["reason"].startswith("unsupported format")


def test_upload_analyses_in_the_background(client, tmp_path):
    source = sorted(FIXTURE_AUDIO.glob("*.flac"))[0]
    with source.open("rb") as fh:
        response = client.post("/api/upload", files={"files": (source.name, fh, "audio/flac")})
    body = response.json()
    assert body["saved"] == [source.name]
    # The request returned a job rather than blocking on the DSP.
    assert body["job_id"]
    job = _wait(client, body["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["analysed"]


def test_mix_requires_at_least_two_tracks(client):
    tracks = client.get("/api/tracks").json()["tracks"]
    response = client.post("/api/mix", json={"track_ids": [tracks[0]["track_id"]]})
    assert response.status_code == 400


def test_unknown_job_and_mix_are_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/mixes/nope/audio").status_code == 404


def test_full_mix_flow(client):
    tracks = client.get("/api/tracks").json()["tracks"]
    ids = [t["track_id"] for t in tracks]

    started = client.post(
        "/api/mix",
        json={"track_ids": ids, "occasion": "party", "planner": "rule", "output_format": "wav"},
    ).json()
    job = _wait(client, started["job_id"])
    assert job["status"] == "done", job

    result = job["result"]
    assert result["planner"] == "rule"
    assert result["occasion"] == "party"
    assert len(result["steps"]) >= 2
    assert result["render"]["duration_sec"] > 30
    assert result["render"]["lufs"] == pytest.approx(-14.0, abs=0.7)
    assert result["render"]["peak_db"] <= -0.9
    assert result["render"]["worst_seam_ratio"] < 2.0

    audio = client.get(result["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert len(audio.content) > 100_000

    plan = client.get(result["plan_url"]).json()
    assert plan["occasion"] == "party"
    # The served plan must not carry any measurement -- same rule as everywhere.
    assert "bpm" not in client.get(result["plan_url"]).text


def test_llm_planner_falls_back_over_http(client, monkeypatch):
    """A bad model response must degrade to the rule planner, and the API must
    report that it happened rather than passing it off as an LLM plan."""
    from djmix.llm import factory
    from djmix.llm.mock import MockProvider

    monkeypatch.setattr(
        factory, "get_provider", lambda name=None, **kw: MockProvider("hallucinated_field")
    )
    ids = [t["track_id"] for t in client.get("/api/tracks").json()["tracks"]]
    started = client.post(
        "/api/mix",
        json={"track_ids": ids, "occasion": "focus", "planner": "llm", "output_format": "wav"},
    ).json()
    job = _wait(client, started["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["planner"] == "rule"
    assert "FORBIDDEN_FIELD" in job["result"]["fallback_reason"]


def test_job_failure_is_reported_not_swallowed(client, monkeypatch):
    """A crash inside a job must surface as an error status, not a spinner that
    never resolves."""
    import djmix.web.app as web_app

    def boom(*args, **kwargs):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(web_app, "render", boom)
    ids = [t["track_id"] for t in client.get("/api/tracks").json()["tracks"]]
    started = client.post("/api/mix", json={"track_ids": ids, "occasion": "party"}).json()
    job = _wait(client, started["job_id"])
    assert job["status"] == "error"
    assert "render exploded" in job["message"]


def test_purge_deletes_everything(client):
    removed = client.delete("/api/data").json()
    assert removed["analyses"] >= 1
    assert client.get("/api/tracks").json()["tracks"] == []
