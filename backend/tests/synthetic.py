"""Synthetic match scene for tests: no model weights or real footage needed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from playertrack.homography import PitchHomography
from playertrack.structures import BBox, Detection, ObjectClass

FPS = 25.0
PX_PER_M = 10.0
FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def scale_homography() -> PitchHomography:
    """Top-down camera: 10 pixels = 1 metre."""
    return PitchHomography(np.diag([1 / PX_PER_M, 1 / PX_PER_M, 1.0]))


def player_box(foot_x: float, foot_y: float) -> BBox:
    return BBox(foot_x - 10, foot_y - 40, foot_x + 10, foot_y)


def ball_box(x: float, y: float) -> BBox:
    return BBox(x - 3, y - 3, x + 3, y + 3)


@dataclass
class ScriptedDetector:
    """Returns pre-computed detections, one list per call (frame)."""

    frames: list[list[Detection]]
    calls: int = field(default=0, init=False)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections = self.frames[self.calls] if self.calls < len(self.frames) else []
        self.calls += 1
        return detections


def passing_scene(n_frames: int = 100) -> list[list[Detection]]:
    """Target runs right at 5 m/s with the ball, passes it to a static team-mate.

    Frames 0-49: ball at the target's feet. 50-59: ball in flight. 60+: ball at
    the team-mate's feet.
    """
    frames = []
    for f in range(n_frames):
        target_x = 110 + 2 * f  # 2 px/frame = 0.2 m/frame = 5 m/s at 25 fps
        detections = [
            Detection(player_box(target_x, 340), 0.9, ObjectClass.PLAYER),
            Detection(player_box(600, 340), 0.9, ObjectClass.PLAYER),
            Detection(player_box(900, 100), 0.9, ObjectClass.REFEREE),
        ]
        if f < 50:
            ball = (target_x + 5, 338)
        elif f < 60:
            ball = (420, 500)
        else:
            ball = (604, 338)
        detections.append(Detection(ball_box(*ball), 0.6, ObjectClass.BALL))
        frames.append(detections)
    return frames


def frames(n: int):
    return ((i, FRAME) for i in range(n))
