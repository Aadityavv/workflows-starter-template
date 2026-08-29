"""Background jobs.

Analysis and rendering are CPU-bound and take real time -- seconds per track for
analysis, tens of seconds for a render. They must never run on the request
thread, so every long operation becomes a job the browser polls.

This is an in-process worker, not Celery+Redis. For a single-user app running on
your own machine that is the right trade: no broker to install, no second
process to supervise. The interface below (submit -> job id -> poll status) is
the same one a Celery backend would expose, so swapping it out for a distributed
queue when there are real users means replacing this file, not the API layer.

A single worker thread is deliberate: librosa releases the GIL inside its native
code, but running several analyses at once on a laptop mostly just thrashes the
CPU and makes progress reporting meaningless.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = "queued"
    progress: float = 0.0
    message: str = "queued"
    result: Any = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
        }


class JobRegistry:
    def __init__(self, max_workers: int = 1):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="djmix")

    def submit(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        self.update(job, status="running", message="starting", progress=0.0)
        try:
            result = fn(job)
        except Exception as exc:
            # A failed job must not take the server down, and the browser needs
            # something better than a spinner that never stops.
            self.update(
                job,
                status="error",
                message=str(exc) or exc.__class__.__name__,
                error=traceback.format_exc(limit=4),
                progress=1.0,
            )
            return
        self.update(job, status="done", message="complete", progress=1.0, result=result)

    def update(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def prune(self, keep: int = 50) -> None:
        with self._lock:
            done = [j for j in self._jobs.values() if j.status in ("done", "error")]
            for job in sorted(done, key=lambda j: j.created_at)[: -keep or None]:
                self._jobs.pop(job.id, None)
