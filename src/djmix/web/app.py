"""FastAPI application.

Every long operation returns a job id immediately and is polled, because
analysis and rendering take real time and the UI must show progress rather than
hang. The audio work itself is the same code the CLI calls.

This is a LOCAL app: it runs on your machine, reads files you give it, and
writes results next to them. It is not a hosted service, and deliberately so --
see LEGAL.md on why uploading other people's music to a server is a different
proposition from mixing your own library on your own computer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from djmix.audio import io as audio_io
from djmix.audio.analysis import analyze_file
from djmix.audio.cache import AnalysisCache
from djmix.audio.tempo import PREFERRED_BPM_HI, PREFERRED_BPM_LO
from djmix.config import (
    EntitlementError,
    available_occasions,
    check_mix_request,
    get_tier,
)
from djmix.models import TrackAnalysis
from djmix.planning.base import MixRequest
from djmix.planning.rules import RulePlanner, plan_duration_sec
from djmix.render.engine import render
from djmix.web.jobs import Job, JobRegistry

STATIC_DIR = Path(__file__).parent / "static"


class Workspace:
    """Where uploads, analyses, and rendered mixes live."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.uploads = self.root / "uploads"
        self.renders = self.root / "renders"
        self.cache_dir = self.root / "cache"
        for directory in (self.uploads, self.renders, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def cache(self) -> AnalysisCache:
        return AnalysisCache(self.cache_dir)

    def analyses(self) -> list[TrackAnalysis]:
        out: list[TrackAnalysis] = []
        for path in self.cache.entries():
            try:
                analysis = TrackAnalysis.model_validate_json(path.read_text())
            except Exception:
                continue
            # Drop entries whose audio has since been deleted, so the library
            # never lists a track the renderer cannot actually load.
            if Path(analysis.source_path).is_file():
                out.append(analysis)
        return sorted(out, key=lambda a: Path(a.source_path).name.lower())


class MixOptions(BaseModel):
    track_ids: list[str] = Field(default_factory=list)
    occasion: str | None = None
    prompt: str | None = None
    duration_minutes: float | None = None
    planner: str = "rule"
    provider: str | None = None
    output_format: str = "mp3"


def create_app(workspace_root: Path | str = "djmix-data") -> FastAPI:
    workspace = Workspace(Path(workspace_root))
    jobs = JobRegistry()
    app = FastAPI(title="djmix", docs_url="/api/docs")

    def track_summary(a: TrackAnalysis) -> dict:
        top_mood = max(a.mood_tags, key=a.mood_tags.get) if a.mood_tags else None
        # Half- and double-time readings always exist arithmetically, so their
        # mere presence says nothing. What is worth surfacing is whether an
        # alternative is *also* a plausible DJ tempo -- that is the case where
        # 87 and 174 BPM are both honest descriptions of the same groove.
        ambiguous = any(PREFERRED_BPM_LO <= alt <= PREFERRED_BPM_HI for alt in a.bpm_alternatives)
        return {
            **a.public_summary(),
            "filename": Path(a.source_path).name,
            "camelot": a.camelot,
            "key_confidence": a.key_confidence,
            "energy": a.energy_score,
            "top_mood": top_mood,
            "bpm_alternatives": a.bpm_alternatives,
            "bpm_ambiguous": ambiguous,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        # A 1x1 transparent GIF. Cheaper than shipping an icon file, and it
        # stops the browser logging a 404 on every page load.
        return Response(
            content=bytes.fromhex(
                "47494638396101000100800000000000ffffff21f90401000000"
                "002c00000000010001000002024401003b"
            ),
            media_type="image/gif",
        )

    @app.get("/api/config")
    def config() -> dict:
        tier = get_tier()
        return {
            "occasions": available_occasions(),
            "formats": ["mp3", "wav", "flac"],
            "supported_uploads": sorted(audio_io.SUPPORTED_SUFFIXES),
            "max_upload_mb": audio_io.MAX_UPLOAD_BYTES // (1024 * 1024),
            "tier": {
                "name": tier.name,
                "max_mix_minutes": tier.max_mix_minutes,
                "max_tracks_per_mix": tier.max_tracks_per_mix,
            },
        }

    @app.get("/api/tracks")
    def list_tracks() -> dict:
        return {"tracks": [track_summary(a) for a in workspace.analyses()]}

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)) -> dict:
        """Store uploads, then analyse them in the background.

        The upload itself is synchronous (it is just I/O), but analysis is not:
        it returns a job id the browser polls.
        """
        saved: list[Path] = []
        rejected: list[dict] = []
        for upload_file in files:
            name = Path(upload_file.filename or "unnamed").name
            suffix = Path(name).suffix.lower()
            if suffix not in audio_io.SUPPORTED_SUFFIXES:
                rejected.append({"filename": name, "reason": f"unsupported format {suffix}"})
                await upload_file.close()
                continue
            destination = workspace.uploads / name
            with destination.open("wb") as fh:
                shutil.copyfileobj(upload_file.file, fh)
            await upload_file.close()
            try:
                audio_io.check_uploadable(destination)
            except audio_io.AudioLoadError as exc:
                destination.unlink(missing_ok=True)
                rejected.append({"filename": name, "reason": str(exc)})
                continue
            saved.append(destination)

        if not saved:
            return {"job_id": None, "saved": [], "rejected": rejected}

        def work(job: Job) -> dict:
            analysed, failed = [], []
            for i, path in enumerate(saved):
                jobs.update(
                    job,
                    progress=i / len(saved),
                    message=f"analysing {path.name} ({i + 1} of {len(saved)})",
                )
                try:
                    analysis = analyze_file(path, cache=workspace.cache)
                    analysed.append(track_summary(analysis))
                except Exception as exc:
                    # One unreadable or beatless file must not sink the batch.
                    failed.append({"filename": path.name, "reason": str(exc)})
            return {"analysed": analysed, "failed": failed}

        job = jobs.submit("analyze", work)
        return {
            "job_id": job.id,
            "saved": [p.name for p in saved],
            "rejected": rejected,
        }

    @app.post("/api/mix")
    def create_mix(options: MixOptions) -> dict:
        library = workspace.analyses()
        if options.track_ids:
            wanted = set(options.track_ids)
            library = [a for a in library if a.track_id in wanted]
        if len(library) < 2:
            raise HTTPException(400, "select at least two analysed tracks")

        try:
            check_mix_request(get_tier(), len(library), options.duration_minutes)
        except EntitlementError as exc:
            raise HTTPException(402, str(exc)) from exc

        if options.output_format not in {"mp3", "wav", "flac"}:
            raise HTTPException(400, f"unsupported format {options.output_format}")

        def work(job: Job) -> dict:
            jobs.update(job, progress=0.1, message="planning")
            request = MixRequest(
                analyses=library,
                occasion=options.occasion,
                prompt=options.prompt,
                target_minutes=options.duration_minutes,
            )

            if options.planner == "llm":
                from djmix.llm.factory import get_provider
                from djmix.planning.llm import LLMPlanner

                result = LLMPlanner(get_provider(options.provider)).plan(request)
            else:
                result = RulePlanner().plan(request)

            by_id = {a.track_id: a for a in library}
            jobs.update(job, progress=0.35, message="rendering audio")
            buffer, report = render(result.plan, by_id)

            jobs.update(job, progress=0.9, message="writing file")
            out_path = workspace.renders / f"{job.id}.{options.output_format}"
            audio_io.write_audio(out_path, buffer)
            plan_path = out_path.with_suffix(".plan.json")
            plan_path.write_text(
                result.plan.plan.model_dump_json(indent=2, exclude_none=True) + "\n"
            )

            steps = []
            for step in result.plan.steps:
                a = by_id[step.track_id]
                steps.append(
                    {
                        "track_id": step.track_id,
                        "filename": Path(a.source_path).name,
                        "bpm": round(a.bpm, 1),
                        "key": a.key,
                        "camelot": a.camelot,
                        "start_at": step.start_at or 0.0,
                        "transition": step.transition.model_dump() if step.transition else None,
                    }
                )

            return {
                "mix_id": job.id,
                "planner": result.planner,
                "occasion": result.plan.occasion,
                "notes": result.notes,
                "fallback_reason": result.fallback_reason,
                "llm_attempts": result.llm_attempts,
                "planned_minutes": round(plan_duration_sec(result.plan.plan, by_id) / 60, 2),
                "steps": steps,
                "audio_url": f"/api/mixes/{job.id}/audio",
                "plan_url": f"/api/mixes/{job.id}/plan",
                "render": {
                    "duration_sec": report.duration_sec,
                    "lufs": report.master.integrated_lufs if report.master else None,
                    "peak_db": report.peak_db,
                    "worst_seam_ratio": report.worst_seam_ratio,
                    "tempo_groups": report.tempo_groups,
                    "junctions": [
                        {
                            "kind": j.kind,
                            "fade_sec": j.fade_sec,
                            "beatmatched": j.beatmatched,
                            "grid_error_ms": j.grid_error_ms,
                        }
                        for j in report.junctions
                    ],
                },
            }

        job = jobs.submit("mix", work)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.public()

    def _mix_file(mix_id: str, suffixes: tuple[str, ...]) -> Path:
        for suffix in suffixes:
            candidate = workspace.renders / f"{mix_id}{suffix}"
            if candidate.is_file():
                return candidate
        raise HTTPException(404, "unknown mix")

    @app.get("/api/mixes/{mix_id}/audio")
    def mix_audio(mix_id: str) -> FileResponse:
        path = _mix_file(mix_id, (".mp3", ".wav", ".flac"))
        media = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac"}[path.suffix]
        return FileResponse(path, media_type=media, filename=f"djmix-{mix_id[:8]}{path.suffix}")

    @app.get("/api/mixes/{mix_id}/plan")
    def mix_plan(mix_id: str) -> dict:
        path = _mix_file(mix_id, (".plan.json",))
        return json.loads(path.read_text())

    @app.delete("/api/tracks/{track_id}")
    def delete_track(track_id: str) -> dict:
        for analysis in workspace.analyses():
            if analysis.track_id == track_id:
                Path(analysis.source_path).unlink(missing_ok=True)
                for entry in workspace.cache.entries():
                    if analysis.content_hash in entry.name:
                        entry.unlink(missing_ok=True)
                return {"deleted": track_id}
        raise HTTPException(404, "unknown track")

    @app.delete("/api/data")
    def purge() -> dict:
        """Delete everything: uploads, analyses, and rendered mixes."""
        removed = {"uploads": 0, "renders": 0, "analyses": 0}
        for path in workspace.uploads.iterdir():
            if path.is_file():
                path.unlink()
                removed["uploads"] += 1
        for path in workspace.renders.iterdir():
            if path.is_file():
                path.unlink()
                removed["renders"] += 1
        removed["analyses"] = workspace.cache.clear()
        return removed

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
