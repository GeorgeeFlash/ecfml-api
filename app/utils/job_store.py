from datetime import datetime, timezone
from threading import Lock
from typing import Any


class JobStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def init_job(self, job_id: str, status: str = "PENDING") -> dict[str, Any]:
        with self._lock:
            job = {
                "status": status,
                "progress": None,
                "error": None,
                "meta": {"events": []},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._jobs[job_id] = job
            return job

    def update_job(
        self,
        job_id: str,
        status: str | None = None,
        progress: float | None = None,
        error: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.setdefault(
                job_id,
                {
                    "status": "PENDING",
                    "progress": None,
                    "error": None,
                    "meta": {"events": []},
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if status is not None:
                job["status"] = status
            if progress is not None:
                job["progress"] = progress
            if error is not None:
                job["error"] = error
            if meta is not None:
                job["meta"] = meta
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            return job

    def append_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.setdefault(
                job_id,
                {
                    "status": "PENDING",
                    "progress": None,
                    "error": None,
                    "meta": {"events": []},
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            events = job["meta"].setdefault("events", [])
            events.append(event)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._jobs.values())


job_store = JobStore()
