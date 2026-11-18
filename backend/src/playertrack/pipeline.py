"""End-to-end analysis: video frames in, metrics for one selected player out.

Per frame: detect objects, update the tracker, follow the selected player and
project everyone's foot point onto the pitch. After the last frame the target's
trajectory gives the movement metrics and heatmap, and the ball / player
positions give possession spells and passes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .detection import Detector
from .homography import PitchHomography
from .metrics import (
    MovementMetrics,
    Trajectory,
    compute_heatmap,
    compute_movement_metrics,
    thirds_share,
)
from .possession import (
    FrameState,
    Pass,
    PossessionSummary,
    detect_passes,
    possession_spells,
    summarize_possession,
)
from .selection import TargetFollower, select_track_at
from .structures import PERSON_CLASSES, ObjectClass
from .tracking import IouTracker

# Stable key for the selected player in possession data; its tracker ID may change.
TARGET_ID = -1


@dataclass(frozen=True, slots=True)
class TargetSelection:
    """Where the user clicked on the player to analyse (image pixels)."""

    frame_index: int
    point: tuple[float, float]


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    tracker_iou_threshold: float = 0.3
    tracker_max_misses: int = 30
    selection_window_frames: int = 10
    reacquire_radius_px: float = 60.0
    smoothing_window: int = 5
    heatmap_bins: tuple[int, int] = (21, 14)
    heatmap_blur_sigma: float = 1.0
    control_radius_m: float = 1.5
    min_possession_frames: int = 3
    max_pass_flight_s: float = 4.0


@dataclass
class AnalysisResult:
    fps: float
    frames_processed: int
    target_found: bool
    target_track_ids: list[int]
    trajectory: Trajectory
    movement: MovementMetrics
    heatmap: np.ndarray
    thirds: tuple[float, float, float]
    possession: PossessionSummary
    passes: list[Pass] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view of the result (NaN positions become ``null``)."""

        def _num(value: float) -> float | None:
            return None if math.isnan(value) else round(float(value), 3)

        return {
            "fps": self.fps,
            "frames_processed": self.frames_processed,
            "target_found": self.target_found,
            "target_track_ids": self.target_track_ids,
            "coverage": round(self.trajectory.coverage, 4),
            "movement": self.movement.to_dict(),
            "thirds": {"left": self.thirds[0], "middle": self.thirds[1], "right": self.thirds[2]},
            "possession": self.possession.to_dict(),
            "passes": [p.to_dict() for p in self.passes],
            "heatmap": self.heatmap.round(5).tolist(),
            "trajectory": [
                [round(float(t), 3), _num(x), _num(y)]
                for t, (x, y) in zip(
                    self.trajectory.timestamps, self.trajectory.positions, strict=True
                )
            ],
        }


class AnalysisPipeline:
    def __init__(
        self,
        detector: Detector,
        homography: PitchHomography,
        config: AnalysisConfig | None = None,
    ) -> None:
        self.detector = detector
        self.homography = homography
        self.config = config or AnalysisConfig()

    def run(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        fps: float,
        selection: TargetSelection,
        progress: Callable[[int], None] | None = None,
    ) -> AnalysisResult:
        if fps <= 0:
            raise ValueError("fps must be positive")
        cfg = self.config
        tracker = IouTracker(
            iou_threshold=cfg.tracker_iou_threshold, max_misses=cfg.tracker_max_misses
        )
        follower: TargetFollower | None = None
        timestamps: list[float] = []
        positions: list[tuple[float, float]] = []
        states: list[FrameState] = []
        processed = 0

        for frame_index, frame in frames:
            processed += 1
            if progress is not None:
                progress(processed)

            detections = self.detector.detect(frame)
            people = [d for d in detections if d.cls in PERSON_CLASSES]
            balls = [d for d in detections if d.cls is ObjectClass.BALL]
            tracks = tracker.update(people)

            if follower is None:
                in_window = (
                    selection.frame_index
                    <= frame_index
                    <= selection.frame_index + cfg.selection_window_frames
                )
                chosen = select_track_at(tracks, selection.point) if in_window else None
                if chosen is None:
                    continue
                follower = TargetFollower(
                    chosen,
                    frame_index,
                    known_ids=[t.track_id for t in tracks],
                    reacquire_radius_px=cfg.reacquire_radius_px,
                )

            target = follower.update(frame_index, tracks)
            field_players = [t for t in tracks if t.cls is not ObjectClass.REFEREE]
            keys = [
                TARGET_ID if target is not None and t.track_id == target.track_id else t.track_id
                for t in field_players
            ]
            pitch_points = (
                self.homography.image_to_pitch([t.bbox.foot_point for t in field_players])
                if field_players
                else np.empty((0, 2))
            )
            players = {
                k: (float(p[0]), float(p[1])) for k, p in zip(keys, pitch_points, strict=True)
            }

            ball = None
            if balls:
                best_ball = max(balls, key=lambda d: d.confidence)
                bx, by = self.homography.image_to_pitch([best_ball.bbox.center])[0]
                ball = (float(bx), float(by))

            timestamp = frame_index / fps
            timestamps.append(timestamp)
            positions.append(players.get(TARGET_ID, (math.nan, math.nan)))
            states.append(FrameState(timestamp, ball, players))

        trajectory = Trajectory(
            np.asarray(timestamps, dtype=np.float64),
            np.asarray(positions, dtype=np.float64).reshape(-1, 2),
        )
        length, width = self.homography.pitch_length, self.homography.pitch_width
        spells = possession_spells(
            states, cfg.control_radius_m, min_frames=cfg.min_possession_frames
        )
        passes = detect_passes(spells, TARGET_ID, cfg.max_pass_flight_s)
        return AnalysisResult(
            fps=fps,
            frames_processed=processed,
            target_found=follower is not None,
            target_track_ids=follower.track_ids if follower is not None else [],
            trajectory=trajectory,
            movement=compute_movement_metrics(trajectory, cfg.smoothing_window),
            heatmap=compute_heatmap(
                trajectory.positions, length, width, cfg.heatmap_bins, cfg.heatmap_blur_sigma
            ),
            thirds=thirds_share(trajectory.positions, length),
            possession=summarize_possession(spells, passes, TARGET_ID),
            passes=passes,
        )
