from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List
from uuid import uuid4

from app.chains.rag import RAGPipeline
from app.config import Settings
from app.utils.models import IndexJobStatus, IndexProgress


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackgroundIndexManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-index")
        self._jobs: Dict[str, IndexJobStatus] = {}
        self._lock = Lock()

    def start_index_job(
        self,
        settings: Settings,
        paths: Iterable[Path],
        collection_name: str,
    ) -> str:
        path_list = [Path(path) for path in paths]
        job_id = uuid4().hex[:12]
        job = IndexJobStatus(
            job_id=job_id,
            status="queued",
            collection_name=collection_name,
            total_files=len(path_list),
            started_at=_utc_now(),
            message="Waiting to start indexing.",
        )
        with self._lock:
            self._jobs[job_id] = job

        settings_json = settings.model_dump_json()
        self._executor.submit(
            self._run_index_job,
            job_id,
            settings_json,
            [str(path) for path in path_list],
            collection_name,
        )
        return job_id

    def get_job(self, job_id: str) -> IndexJobStatus | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job is not None else None

    def list_jobs(self) -> List[IndexJobStatus]:
        with self._lock:
            jobs = [deepcopy(job) for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job.started_at, reverse=True)

    def _run_index_job(
        self,
        job_id: str,
        settings_json: str,
        raw_paths: List[str],
        collection_name: str,
    ) -> None:
        settings = Settings.model_validate_json(settings_json)
        settings.ensure_directories()
        pipeline = RAGPipeline(settings=settings)
        self._update_job(
            job_id,
            status="running",
            stage="loading",
            message="Preparing indexing job.",
        )

        def progress_callback(progress: IndexProgress) -> None:
            self._update_job(
                job_id,
                stage=progress.stage,
                processed_files=progress.current,
                message=progress.message,
            )

        try:
            summary = pipeline.index_paths(
                [Path(path) for path in raw_paths],
                collection_name=collection_name,
                progress_callback=progress_callback,
            )
        except Exception as error:
            self._update_job(
                job_id,
                status="failed",
                stage="failed",
                message="Indexing job failed.",
                error=str(error),
                finished_at=_utc_now(),
            )
            return

        self._update_job(
            job_id,
            status="completed",
            stage="completed",
            processed_files=summary.files_indexed,
            message="Indexing job completed.",
            summary=summary,
            finished_at=_utc_now(),
        )

    def _update_job(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
