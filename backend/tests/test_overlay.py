import json

import cv2
import numpy as np
from synthetic import ball_box, player_box, scale_homography

from playertrack.cli import main
from playertrack.overlay import draw_detections, draw_pitch, pitch_line_segments
from playertrack.structures import Detection, ObjectClass

BLANK = np.zeros((720, 1100, 3), dtype=np.uint8)


def test_pitch_segments_stay_on_the_pitch():
    for segment in pitch_line_segments(105, 68):
        assert segment.min() >= -1e-9
        assert segment[:, 0].max() <= 105 + 1e-9
        assert segment[:, 1].max() <= 68 + 1e-9


def test_draw_pitch_puts_lines_where_the_homography_says():
    out = draw_pitch(BLANK, scale_homography())  # 10 px per metre
    assert out[340, 525].any()  # halfway line at x = 52.5 m
    assert out[0:5, 0:1050].any()  # far touchline
    assert not out[300, 300].any()  # open grass
    assert not BLANK.any()  # input left untouched


def test_draw_detections_marks_each_box():
    detections = [
        Detection(player_box(200, 400), 0.9, ObjectClass.PLAYER),
        Detection(ball_box(600, 300), 0.5, ObjectClass.BALL),
    ]
    out = draw_detections(BLANK, detections)
    assert out[360:400, 190].any()  # left edge of the player box
    assert out[297:303, 597].any()  # ball box


def write_video(path, n_frames=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (1100, 720))
    for _ in range(n_frames):
        writer.write(BLANK)
    writer.release()


def test_snapshot_command_with_calibration(tmp_path):
    video = tmp_path / "clip.avi"
    write_video(video)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "points": [
                    {"image": [0, 0], "landmark": "top_left_corner"},
                    {"image": [1050, 0], "landmark": "top_right_corner"},
                    {"image": [0, 680], "landmark": "bottom_left_corner"},
                    {"image": [1050, 680], "landmark": "bottom_right_corner"},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "frame.png"
    code = main(
        [
            "snapshot",
            str(video),
            "--frame",
            "2",
            "--calibration",
            str(calibration),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    image = cv2.imread(str(out))
    assert image.shape == (720, 1100, 3)
    assert image[340, 525].any()


def test_snapshot_from_an_image(tmp_path):
    image = tmp_path / "photo.jpg"
    cv2.imwrite(str(image), BLANK)
    out = tmp_path / "result" / "photo-out.png"
    assert main(["snapshot", str(image), "--out", str(out)]) == 0
    assert cv2.imread(str(out)).shape == (720, 1100, 3)


def test_snapshot_past_the_end(tmp_path):
    video = tmp_path / "clip.avi"
    write_video(video, n_frames=3)
    assert main(["snapshot", str(video), "--frame", "10", "--out", str(tmp_path / "x.png")]) == 1
