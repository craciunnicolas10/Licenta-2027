import json

import pytest
from synthetic import FPS, ScriptedDetector, frames, passing_scene, scale_homography

from playertrack.pipeline import TARGET_ID, AnalysisPipeline, TargetSelection


def run_scene(scene, selection):
    pipeline = AnalysisPipeline(ScriptedDetector(scene), scale_homography())
    return pipeline.run(frames(len(scene)), FPS, selection)


def test_end_to_end_passing_scene():
    result = run_scene(passing_scene(), TargetSelection(0, (110, 320)))

    assert result.target_found
    assert result.frames_processed == 100
    assert result.trajectory.coverage == 1.0
    # 99 steps of 0.2 m; smoothing trims a little at both ends
    assert result.movement.distance_m == pytest.approx(19.8, rel=0.05)
    assert result.movement.top_speed_kmh == pytest.approx(18.0, rel=0.05)

    assert result.possession.touches == 1
    assert result.possession.possession_time_s == pytest.approx(49 / FPS)
    (pass_,) = result.passes
    assert pass_.passer_id == TARGET_ID
    assert pass_.receiver_id == 2
    assert result.thirds[0] == pytest.approx(1.0)  # ran from x=11 m to x=30.8 m


def test_result_is_json_serialisable():
    result = run_scene(passing_scene(20), TargetSelection(0, (110, 320)))
    data = json.loads(json.dumps(result.to_dict()))
    assert len(data["trajectory"]) == 20
    assert data["trajectory"][0] == [0.0, 11.0, 34.0]
    assert len(data["heatmap"]) == 14


def test_target_lost_and_reacquired_keeps_one_identity():
    scene = passing_scene()
    for f in range(30, 70):  # target occluded long enough for the tracker to drop it
        scene[f] = scene[f][1:]
    pipeline = AnalysisPipeline(ScriptedDetector(scene), scale_homography())
    pipeline.config = type(pipeline.config)(tracker_max_misses=5, reacquire_radius_px=100)
    result = pipeline.run(frames(100), FPS, TargetSelection(0, (110, 320)))
    assert len(result.target_track_ids) == 2
    assert result.trajectory.coverage == pytest.approx(0.6)


def test_click_on_empty_grass_finds_nobody():
    result = run_scene(passing_scene(30), TargetSelection(0, (1000, 600)))
    assert not result.target_found
    assert result.trajectory.timestamps.size == 0
    assert result.movement.distance_m == 0.0
