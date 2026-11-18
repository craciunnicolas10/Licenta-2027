"""Physical metrics computed from the target player's trajectory on the pitch."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .homography import DEFAULT_PITCH_LENGTH_M, DEFAULT_PITCH_WIDTH_M

MS_TO_KMH = 3.6
# 25.2 km/h: the sprint threshold most commonly used in football tracking literature.
SPRINT_THRESHOLD_MS = 7.0
# ~45 km/h. Faster steps are detection or homography noise, not running.
MAX_PLAUSIBLE_SPEED_MS = 12.5


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Positions over time. Rows where the player was not visible are NaN."""

    timestamps: np.ndarray  # seconds, shape (N,)
    positions: np.ndarray  # metres, shape (N, 2)

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if self.positions.shape != (len(self.timestamps), 2):
            raise ValueError("positions must have shape (len(timestamps), 2)")

    @property
    def valid_mask(self) -> np.ndarray:
        return ~np.isnan(self.positions).any(axis=1)

    @property
    def coverage(self) -> float:
        """Fraction of samples in which the player was located."""
        return float(self.valid_mask.mean()) if len(self.timestamps) else 0.0


@dataclass(frozen=True, slots=True)
class MovementMetrics:
    distance_m: float
    top_speed_kmh: float
    average_speed_kmh: float
    tracked_time_s: float
    sprint_count: int
    sprint_distance_m: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def smooth_positions(positions: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that ignores missing (NaN) samples.

    Raw foot points jitter by a few pixels every frame; without smoothing the
    jitter adds up to a large overestimate of distance and top speed.
    """
    n = len(positions)
    if window <= 1 or n == 0:
        return positions.astype(np.float64, copy=True)
    valid = ~np.isnan(positions).any(axis=1)
    filled = np.where(valid[:, None], positions, 0.0)
    kernel = np.ones(window)
    offset = (window - 1) // 2

    def centred(signal: np.ndarray) -> np.ndarray:
        # mode="same" would return max(n, window) samples when n < window
        return np.convolve(signal, kernel, mode="full")[offset : offset + n]

    sums = np.stack([centred(filled[:, k]) for k in range(2)], axis=1)
    counts = centred(valid.astype(np.float64))
    smoothed = np.full_like(positions, np.nan, dtype=np.float64)
    smoothed[valid] = sums[valid] / counts[valid, None]
    return smoothed


def compute_movement_metrics(
    trajectory: Trajectory,
    smoothing_window: int = 5,
    max_gap_s: float = 0.5,
    max_speed_ms: float = MAX_PLAUSIBLE_SPEED_MS,
    sprint_threshold_ms: float = SPRINT_THRESHOLD_MS,
    min_sprint_duration_s: float = 1.0,
) -> MovementMetrics:
    """Distance, speeds and sprints.

    Steps across gaps longer than ``max_gap_s`` (the player was lost) and steps
    faster than ``max_speed_ms`` (tracking glitches) are excluded.
    """
    positions = smooth_positions(trajectory.positions, smoothing_window)
    valid = ~np.isnan(positions).any(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return MovementMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)

    times = trajectory.timestamps[indices]
    points = positions[indices]
    dt = np.diff(times)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.inf)
    usable = (dt > 0) & (dt <= max_gap_s) & (speed <= max_speed_ms)

    distance = float(step[usable].sum())
    tracked_time = float(dt[usable].sum())
    top_speed = float(speed[usable].max()) if usable.any() else 0.0
    average_speed = distance / tracked_time if tracked_time > 0 else 0.0

    sprint_count = 0
    sprint_distance = 0.0
    run_time = run_distance = 0.0
    for fast, step_dt, step_len in zip(
        usable & (speed >= sprint_threshold_ms), dt, step, strict=True
    ):
        if fast:
            run_time += step_dt
            run_distance += step_len
            continue
        if run_time >= min_sprint_duration_s:
            sprint_count += 1
            sprint_distance += run_distance
        run_time = run_distance = 0.0
    if run_time >= min_sprint_duration_s:
        sprint_count += 1
        sprint_distance += run_distance

    return MovementMetrics(
        distance_m=distance,
        top_speed_kmh=top_speed * MS_TO_KMH,
        average_speed_kmh=average_speed * MS_TO_KMH,
        tracked_time_s=tracked_time,
        sprint_count=sprint_count,
        sprint_distance_m=float(sprint_distance),
    )


def compute_heatmap(
    positions: np.ndarray,
    pitch_length: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width: float = DEFAULT_PITCH_WIDTH_M,
    bins: tuple[int, int] = (21, 14),
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Share of time spent in each pitch cell.

    ``bins`` is (cells along the length, cells along the width). The result has
    shape ``(bins[1], bins[0])`` - row = y, column = x - and sums to 1 (or is all
    zeros when there are no valid positions). Points slightly outside the pitch
    (homography error near the lines) are clamped onto it.
    """
    valid = positions[~np.isnan(positions).any(axis=1)]
    x = np.clip(valid[:, 0], 0.0, pitch_length)
    y = np.clip(valid[:, 1], 0.0, pitch_width)
    grid, _, _ = np.histogram2d(
        y, x, bins=(bins[1], bins[0]), range=[[0.0, pitch_width], [0.0, pitch_length]]
    )
    if blur_sigma > 0:
        grid = cv2.GaussianBlur(grid, (0, 0), blur_sigma)
    total = grid.sum()
    return grid / total if total > 0 else grid


def thirds_share(
    positions: np.ndarray, pitch_length: float = DEFAULT_PITCH_LENGTH_M
) -> tuple[float, float, float]:
    """Share of time in the left, middle and right third of the pitch (camera view)."""
    x = positions[~np.isnan(positions).any(axis=1), 0]
    if len(x) == 0:
        return (0.0, 0.0, 0.0)
    third = pitch_length / 3.0
    left = float((x < third).mean())
    right = float((x >= 2 * third).mean())
    return (left, 1.0 - left - right, right)
