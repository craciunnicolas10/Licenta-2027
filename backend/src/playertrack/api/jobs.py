"""In-memory registry of analysis jobs.

Good enough for a single-process deployment; a persistent queue (e.g. Redis +
a worker) can replace it later without changing the HTTP contract.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    frames_processed: int = 0
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        """A copy of the job's state, safe to serialise while the job keeps running."""
        with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job is not None else None

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id].status = JobStatus.RUNNING

    def set_progress(self, job_id: str, frames_processed: int) -> None:
        with self._lock:
            self._jobs[job_id].frames_processed = frames_processed

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.DONE
            job.result = result
            job.finished_at = time.time()

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.FAILED
            job.error = error
            job.finished_at = time.time()
