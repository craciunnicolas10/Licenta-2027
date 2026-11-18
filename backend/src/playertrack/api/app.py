"""FastAPI application.

    POST /api/analyses        upload a video + calibration + click, returns a job (202)
    GET  /api/analyses/{id}   job status, progress and, when done, the result
    GET  /api/health          liveness check

Run with ``playertrack serve`` or ``uvicorn playertrack.api.app:app``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..homography import PitchHomography
from ..pipeline import AnalysisPipeline, AnalysisResult, TargetSelection
from ..video import iter_frames, read_video_info
from .jobs import JobStore

logger = logging.getLogger(__name__)

Analyzer = Callable[[Path, PitchHomography, TargetSelection, Callable[[int], None]], AnalysisResult]

DEFAULT_CORS_ORIGINS = "http://localhost:5173"


def make_default_analyzer(
    weights: str = "yolo11m.pt", stride: int = 1, device: str | None = None
) -> Analyzer:
    """YOLO-backed analyzer. The model is loaded on first use and shared by all jobs.

    Jobs run one at a time: a single model instance is not safe to call from
    several threads, and one GPU would not run them faster anyway.
    """
    detector = None
    lock = threading.Lock()

    def analyze(
        video_path: Path,
        homography: PitchHomography,
        selection: TargetSelection,
        progress: Callable[[int], None],
    ) -> AnalysisResult:
        nonlocal detector
        with lock:
            if detector is None:
                from ..detection import YoloDetector

                detector = YoloDetector(weights, device=device)
            info = read_video_info(video_path)
            frames = iter_frames(video_path, stride=stride, start=selection.frame_index)
            return AnalysisPipeline(detector, homography).run(frames, info.fps, selection, progress)

    return analyze


def _run_job(
    store: JobStore,
    job_id: str,
    analyzer: Analyzer,
    video_path: Path,
    homography: PitchHomography,
    selection: TargetSelection,
) -> None:
    store.mark_running(job_id)
    try:
        result = analyzer(
            video_path, homography, selection, lambda n: store.set_progress(job_id, n)
        )
        if result.target_found:
            store.finish(job_id, result.to_dict())
        else:
            store.fail(job_id, "No player found at the selected position")
    except Exception as exc:  # report any analysis failure through the job, not a crash
        logger.exception("Analysis %s failed", job_id)
        store.fail(job_id, f"{type(exc).__name__}: {exc}")
    finally:
        video_path.unlink(missing_ok=True)


def create_app(analyzer: Analyzer | None = None, upload_dir: Path | None = None) -> FastAPI:
    analyzer = analyzer or make_default_analyzer()
    upload_dir = upload_dir or Path(tempfile.mkdtemp(prefix="playertrack-"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore()

    app = FastAPI(title="playertrack", version=__version__)
    app.state.jobs = store
    origins = os.environ.get("PLAYERTRACK_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/api/analyses", status_code=202)
    def create_analysis(
        background: BackgroundTasks,
        video: Annotated[UploadFile, File()],
        calibration: Annotated[str, Form(description="Calibration JSON, see docs/calibration.md")],
        click_x: Annotated[float, Form()],
        click_y: Annotated[float, Form()],
        click_frame: Annotated[int, Form(ge=0)] = 0,
    ) -> dict[str, Any]:
        try:
            homography = PitchHomography.from_calibration(json.loads(calibration))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid calibration: {exc}") from exc

        job = store.create()
        suffix = Path(video.filename or "").suffix.lower() or ".mp4"
        video_path = upload_dir / f"{job.id}{suffix}"
        with video_path.open("wb") as target:
            shutil.copyfileobj(video.file, target)

        selection = TargetSelection(click_frame, (click_x, click_y))
        background.add_task(_run_job, store, job.id, analyzer, video_path, homography, selection)
        return store.get(job.id)

    @app.get("/api/analyses/{job_id}")
    def get_analysis(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return job

    return app


def __getattr__(name: str) -> Any:
    # ``uvicorn playertrack.api.app:app`` - build the default app only when asked for.
    if name == "app":
        return create_app()
    raise AttributeError(name)
