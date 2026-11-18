import json

import pytest
from fastapi.testclient import TestClient
from synthetic import FPS, ScriptedDetector, frames, passing_scene

from playertrack.api import create_app
from playertrack.pipeline import AnalysisPipeline

# 10 px = 1 m, matching the synthetic scene
CALIBRATION = {
    "points": [
        {"image": [0, 0], "landmark": "top_left_corner"},
        {"image": [1050, 0], "landmark": "top_right_corner"},
        {"image": [0, 680], "landmark": "bottom_left_corner"},
        {"image": [1050, 680], "landmark": "bottom_right_corner"},
    ]
}


def synthetic_analyzer(video_path, homography, selection, progress):
    assert video_path.exists()
    pipeline = AnalysisPipeline(ScriptedDetector(passing_scene()), homography)
    return pipeline.run(frames(100), FPS, selection, progress)


def failing_analyzer(video_path, homography, selection, progress):
    raise RuntimeError("decoder exploded")


@pytest.fixture
def make_client(tmp_path):
    def factory(analyzer=synthetic_analyzer):
        return TestClient(create_app(analyzer, upload_dir=tmp_path)), tmp_path

    return factory


def submit(client, calibration=CALIBRATION, click=(110, 320)):
    return client.post(
        "/api/analyses",
        files={"video": ("match.mp4", b"fake video bytes", "video/mp4")},
        data={
            "calibration": json.dumps(calibration),
            "click_x": str(click[0]),
            "click_y": str(click[1]),
        },
    )


def test_health(make_client):
    client, _ = make_client()
    assert client.get("/api/health").json()["status"] == "ok"


def test_analysis_runs_to_completion(make_client):
    client, upload_dir = make_client()
    response = submit(client)
    assert response.status_code == 202
    job_id = response.json()["id"]

    # TestClient runs background tasks before returning, so the job is already done
    job = client.get(f"/api/analyses/{job_id}").json()
    assert job["status"] == "done"
    assert job["frames_processed"] == 100
    assert job["result"]["movement"]["distance_m"] == pytest.approx(19.8, rel=0.05)
    assert job["result"]["possession"]["passes_attempted"] == 1
    assert list(upload_dir.iterdir()) == []  # upload removed after the run


def test_click_on_nobody_fails_job(make_client):
    client, _ = make_client()
    job_id = submit(client, click=(1000, 650)).json()["id"]
    job = client.get(f"/api/analyses/{job_id}").json()
    assert job["status"] == "failed"
    assert "No player" in job["error"]


def test_analyzer_exception_is_reported(make_client):
    client, _ = make_client(failing_analyzer)
    job_id = submit(client).json()["id"]
    job = client.get(f"/api/analyses/{job_id}").json()
    assert job["status"] == "failed"
    assert job["error"] == "RuntimeError: decoder exploded"


def test_invalid_calibration_rejected(make_client):
    client, upload_dir = make_client()
    response = submit(client, calibration={"points": CALIBRATION["points"][:2]})
    assert response.status_code == 422
    assert "At least 4" in response.json()["detail"]
    assert list(upload_dir.iterdir()) == []


def test_unknown_job(make_client):
    client, _ = make_client()
    assert client.get("/api/analyses/does-not-exist").status_code == 404
