"""Mapping image pixels to pitch coordinates in metres.

Pitch coordinate system: origin at the top-left corner flag, x along the touchline
(0 .. length), y along the goal line (0 .. width). A homography is valid for the
camera position it was estimated from, so this module assumes a fixed camera for
now; per-frame calibration from detected pitch keypoints is planned.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0

PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
PENALTY_SPOT_DISTANCE_M = 11.0
CENTER_CIRCLE_RADIUS_M = 9.15


def pitch_landmarks(
    length: float = DEFAULT_PITCH_LENGTH_M, width: float = DEFAULT_PITCH_WIDTH_M
) -> dict[str, tuple[float, float]]:
    """Named pitch points (in metres) that are easy to click in a video frame."""
    mid_x, mid_y = length / 2.0, width / 2.0
    box_top, box_bottom = mid_y - PENALTY_AREA_WIDTH_M / 2.0, mid_y + PENALTY_AREA_WIDTH_M / 2.0
    six_top, six_bottom = mid_y - GOAL_AREA_WIDTH_M / 2.0, mid_y + GOAL_AREA_WIDTH_M / 2.0
    right_box_x = length - PENALTY_AREA_DEPTH_M
    right_six_x = length - GOAL_AREA_DEPTH_M
    return {
        "top_left_corner": (0.0, 0.0),
        "top_right_corner": (length, 0.0),
        "bottom_left_corner": (0.0, width),
        "bottom_right_corner": (length, width),
        "halfway_top": (mid_x, 0.0),
        "halfway_bottom": (mid_x, width),
        "center_spot": (mid_x, mid_y),
        "center_circle_top": (mid_x, mid_y - CENTER_CIRCLE_RADIUS_M),
        "center_circle_bottom": (mid_x, mid_y + CENTER_CIRCLE_RADIUS_M),
        "left_penalty_box_top_goal_line": (0.0, box_top),
        "left_penalty_box_top_corner": (PENALTY_AREA_DEPTH_M, box_top),
        "left_penalty_box_bottom_corner": (PENALTY_AREA_DEPTH_M, box_bottom),
        "left_penalty_box_bottom_goal_line": (0.0, box_bottom),
        "left_goal_area_top_corner": (GOAL_AREA_DEPTH_M, six_top),
        "left_goal_area_bottom_corner": (GOAL_AREA_DEPTH_M, six_bottom),
        "left_penalty_spot": (PENALTY_SPOT_DISTANCE_M, mid_y),
        "right_penalty_box_top_goal_line": (length, box_top),
        "right_penalty_box_top_corner": (right_box_x, box_top),
        "right_penalty_box_bottom_corner": (right_box_x, box_bottom),
        "right_penalty_box_bottom_goal_line": (length, box_bottom),
        "right_goal_area_top_corner": (right_six_x, six_top),
        "right_goal_area_bottom_corner": (right_six_x, six_bottom),
        "right_penalty_spot": (length - PENALTY_SPOT_DISTANCE_M, mid_y),
    }


def _apply(matrix: np.ndarray, points: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


class PitchHomography:
    def __init__(
        self,
        matrix: Any,
        pitch_length: float = DEFAULT_PITCH_LENGTH_M,
        pitch_width: float = DEFAULT_PITCH_WIDTH_M,
    ) -> None:
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"Homography must be 3x3, got shape {m.shape}")
        self.matrix = m
        self._inverse = np.linalg.inv(m)
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    @classmethod
    def from_correspondences(
        cls,
        image_points: Any,
        pitch_points: Any,
        ransac_threshold_m: float | None = None,
        pitch_length: float = DEFAULT_PITCH_LENGTH_M,
        pitch_width: float = DEFAULT_PITCH_WIDTH_M,
    ) -> PitchHomography:
        """Estimate the image -> pitch homography from at least 4 point pairs.

        With more than 4 pairs and ``ransac_threshold_m`` set, RANSAC discards badly
        clicked points; otherwise a least-squares fit over all points is used.
        """
        image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        pitch = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
        if len(image) != len(pitch):
            raise ValueError("image_points and pitch_points must have the same length")
        if len(image) < 4:
            raise ValueError("At least 4 point correspondences are needed")
        use_ransac = ransac_threshold_m is not None and len(image) > 4
        matrix, _ = cv2.findHomography(
            image,
            pitch,
            cv2.RANSAC if use_ransac else 0,
            ransac_threshold_m if use_ransac else 3.0,
        )
        if matrix is None:
            raise ValueError("Degenerate point configuration (are 3 of the points collinear?)")
        return cls(matrix, pitch_length, pitch_width)

    @classmethod
    def from_calibration(cls, data: Mapping[str, Any]) -> PitchHomography:
        """Build from a calibration mapping (see ``docs/calibration.md`` for the format)."""
        pitch = data.get("pitch", {})
        length = float(pitch.get("length", DEFAULT_PITCH_LENGTH_M))
        width = float(pitch.get("width", DEFAULT_PITCH_WIDTH_M))
        landmarks = pitch_landmarks(length, width)
        image_points, pitch_points = [], []
        for entry in data.get("points", []):
            image_points.append(entry["image"])
            if "pitch" in entry:
                pitch_points.append(entry["pitch"])
            elif "landmark" in entry:
                name = entry["landmark"]
                if name not in landmarks:
                    raise ValueError(f"Unknown landmark '{name}'")
                pitch_points.append(landmarks[name])
            else:
                raise ValueError("Each calibration point needs a 'pitch' or 'landmark' key")
        ransac = data.get("ransac_threshold_m")
        return cls.from_correspondences(
            image_points,
            pitch_points,
            None if ransac is None else float(ransac),
            pitch_length=length,
            pitch_width=width,
        )

    @classmethod
    def from_calibration_file(cls, path: str | Path) -> PitchHomography:
        return cls.from_calibration(json.loads(Path(path).read_text(encoding="utf-8")))

    def image_to_pitch(self, points: Any) -> np.ndarray:
        return _apply(self.matrix, points)

    def pitch_to_image(self, points: Any) -> np.ndarray:
        return _apply(self._inverse, points)

    def reprojection_error(self, image_points: Any, pitch_points: Any) -> float:
        """Mean distance in metres between projected image points and their true positions."""
        projected = self.image_to_pitch(image_points)
        expected = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
        return float(np.linalg.norm(projected - expected, axis=1).mean())
