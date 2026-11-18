"""Drawing helpers for checking detections and calibration on a video frame."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .homography import (
    CENTER_CIRCLE_RADIUS_M,
    GOAL_AREA_DEPTH_M,
    GOAL_AREA_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PitchHomography,
)
from .structures import Detection, ObjectClass

# BGR colours
CLASS_COLORS: dict[ObjectClass, tuple[int, int, int]] = {
    ObjectClass.PLAYER: (80, 200, 80),
    ObjectClass.GOALKEEPER: (200, 160, 40),
    ObjectClass.REFEREE: (40, 40, 220),
    ObjectClass.BALL: (0, 215, 255),
}
PITCH_LINE_COLOR = (255, 255, 255)


def pitch_line_segments(length: float, width: float) -> list[np.ndarray]:
    """Pitch markings as polylines in metres (each an (N, 2) array)."""
    mid_x, mid_y = length / 2.0, width / 2.0

    def rect(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], dtype=np.float64)

    box_half, six_half = PENALTY_AREA_WIDTH_M / 2.0, GOAL_AREA_WIDTH_M / 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, 65)
    circle = np.column_stack(
        [
            mid_x + CENTER_CIRCLE_RADIUS_M * np.cos(angles),
            mid_y + CENTER_CIRCLE_RADIUS_M * np.sin(angles),
        ]
    )
    return [
        rect(0.0, 0.0, length, width),
        np.array([[mid_x, 0.0], [mid_x, width]]),
        rect(0.0, mid_y - box_half, PENALTY_AREA_DEPTH_M, mid_y + box_half),
        rect(length - PENALTY_AREA_DEPTH_M, mid_y - box_half, length, mid_y + box_half),
        rect(0.0, mid_y - six_half, GOAL_AREA_DEPTH_M, mid_y + six_half),
        rect(length - GOAL_AREA_DEPTH_M, mid_y - six_half, length, mid_y + six_half),
        circle,
    ]


def draw_pitch(frame: np.ndarray, homography: PitchHomography, thickness: int = 2) -> np.ndarray:
    """Project the pitch markings into the image.

    If the drawn lines do not sit on the real ones, the calibration points are wrong.
    """
    out = frame.copy()
    for segment in pitch_line_segments(homography.pitch_length, homography.pitch_width):
        pixels = homography.pitch_to_image(segment)
        if not np.isfinite(pixels).all():
            continue
        cv2.polylines(
            out,
            [np.round(pixels).astype(np.int32)],
            False,
            PITCH_LINE_COLOR,
            thickness,
            cv2.LINE_AA,
        )
    return out


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Boxes with class and confidence. People are labelled with their box centre,
    a point that can be passed to ``analyze --click`` to select them."""
    out = frame.copy()
    for det in detections:
        color = CLASS_COLORS[det.cls]
        x1, y1, x2, y2 = (
            int(round(v)) for v in (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        fx, fy = det.bbox.foot_point
        cv2.circle(out, (int(round(fx)), int(round(fy))), 3, color, -1)
        label = f"{det.cls.name.lower()} {det.confidence:.2f}"
        if det.cls is not ObjectClass.BALL:
            cx, cy = det.bbox.center
            label += f" ({cx:.0f},{cy:.0f})"
        cv2.putText(
            out, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
    return out
