import json

import numpy as np
import pytest

from playertrack.homography import PitchHomography, pitch_landmarks

# Perspective view of the pitch: a trapezoid in the image maps to the full rectangle.
IMAGE_CORNERS = [[300, 200], [980, 200], [80, 650], [1200, 650]]
PITCH_CORNERS = [[0, 0], [105, 0], [0, 68], [105, 68]]


def test_corners_map_exactly():
    h = PitchHomography.from_correspondences(IMAGE_CORNERS, PITCH_CORNERS)
    np.testing.assert_allclose(h.image_to_pitch(IMAGE_CORNERS), PITCH_CORNERS, atol=1e-6)
    assert h.reprojection_error(IMAGE_CORNERS, PITCH_CORNERS) < 1e-6


def test_round_trip_image_pitch_image():
    h = PitchHomography.from_correspondences(IMAGE_CORNERS, PITCH_CORNERS)
    points = np.array([[640, 400], [500, 300], [1000, 600]], dtype=float)
    np.testing.assert_allclose(h.pitch_to_image(h.image_to_pitch(points)), points, atol=1e-6)


def test_perspective_far_side_is_compressed():
    h = PitchHomography.from_correspondences(IMAGE_CORNERS, PITCH_CORNERS)
    near_y = h.image_to_pitch([[640, 640]])[0, 1]
    mid_y = h.image_to_pitch([[640, 425]])[0, 1]
    # the far half of the pitch looks smaller, so the image midline lies past the halfway mark
    assert 34 < mid_y < 68
    assert near_y > 60


def test_needs_four_points():
    with pytest.raises(ValueError):
        PitchHomography.from_correspondences(IMAGE_CORNERS[:3], PITCH_CORNERS[:3])


def test_landmarks_follow_pitch_size():
    marks = pitch_landmarks(100, 64)
    assert marks["center_spot"] == (50, 32)
    assert marks["right_penalty_spot"] == (89, 32)
    assert marks["left_penalty_box_top_corner"] == pytest.approx((16.5, 32 - 20.16))


def test_calibration_file_with_landmarks_and_raw_points(tmp_path):
    names = ["top_left_corner", "top_right_corner", "bottom_left_corner"]
    data = {
        "pitch": {"length": 105, "width": 68},
        "points": [
            *(
                {"image": img, "landmark": name}
                for img, name in zip(IMAGE_CORNERS[:3], names, strict=True)
            ),
            {"image": IMAGE_CORNERS[3], "pitch": PITCH_CORNERS[3]},
        ],
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    h = PitchHomography.from_calibration_file(path)
    np.testing.assert_allclose(h.image_to_pitch(IMAGE_CORNERS), PITCH_CORNERS, atol=1e-6)


def test_calibration_unknown_landmark():
    data = {"points": [{"image": [0, 0], "landmark": "corner_flag_of_narnia"}] * 4}
    with pytest.raises(ValueError, match="Unknown landmark"):
        PitchHomography.from_calibration(data)
