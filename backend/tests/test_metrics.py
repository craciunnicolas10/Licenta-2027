import numpy as np
import pytest

from playertrack.metrics import (
    Trajectory,
    compute_heatmap,
    compute_movement_metrics,
    smooth_positions,
    thirds_share,
)

FPS = 25.0


def straight_run(speed_ms: float, seconds: float, start=(10.0, 30.0)) -> Trajectory:
    n = int(seconds * FPS) + 1
    t = np.arange(n) / FPS
    positions = np.column_stack([start[0] + speed_ms * t, np.full(n, start[1])])
    return Trajectory(t, positions)


def test_constant_speed_run():
    metrics = compute_movement_metrics(straight_run(5.0, 10.0), smoothing_window=1)
    assert metrics.distance_m == pytest.approx(50.0)
    assert metrics.top_speed_kmh == pytest.approx(18.0)
    assert metrics.average_speed_kmh == pytest.approx(18.0)
    assert metrics.tracked_time_s == pytest.approx(10.0)
    assert metrics.sprint_count == 0


def test_smoothing_removes_jitter_inflation():
    rng = np.random.default_rng(0)
    trajectory = straight_run(3.0, 20.0)
    noisy = Trajectory(trajectory.timestamps, trajectory.positions + rng.normal(0, 0.15, (501, 2)))
    raw = compute_movement_metrics(noisy, smoothing_window=1)
    smoothed = compute_movement_metrics(noisy, smoothing_window=9)
    assert raw.distance_m > 70  # jitter almost doubles the true 60 m
    assert smoothed.distance_m == pytest.approx(60.0, rel=0.1)


def test_gaps_and_teleports_are_not_counted():
    trajectory = straight_run(4.0, 4.0)
    positions = trajectory.positions.copy()
    positions[40:70] = np.nan  # player lost for 1.2 s
    positions[90:] += [30.0, 0.0]  # identity switch: 30 m jump
    metrics = compute_movement_metrics(Trajectory(trajectory.timestamps, positions), 1)
    assert metrics.top_speed_kmh == pytest.approx(14.4)
    # 39 + 19 + 10 steps of 0.16 m
    assert metrics.distance_m == pytest.approx(68 * 0.16)


def test_sprint_detection():
    jog = straight_run(3.0, 2.0)
    sprint = straight_run(8.0, 1.5, start=(16.0, 30.0))
    t = np.concatenate([jog.timestamps, jog.timestamps[-1] + 1 / FPS + sprint.timestamps])
    positions = np.concatenate([jog.positions, sprint.positions])
    metrics = compute_movement_metrics(Trajectory(t, positions), smoothing_window=1)
    assert metrics.sprint_count == 1
    assert metrics.sprint_distance_m == pytest.approx(12.0, abs=0.5)
    assert metrics.top_speed_kmh == pytest.approx(28.8)


def test_too_few_points():
    trajectory = Trajectory(np.array([0.0]), np.array([[1.0, 1.0]]))
    assert compute_movement_metrics(trajectory).distance_m == 0.0


def test_smooth_positions_keeps_nans():
    positions = np.array([[0, 0], [1, 0], [np.nan, np.nan], [3, 0], [4, 0]], dtype=float)
    smoothed = smooth_positions(positions, 3)
    assert np.isnan(smoothed[2]).all()
    assert smoothed[1, 0] == pytest.approx(0.5)  # mean of 0 and 1; the NaN neighbour is skipped


def test_heatmap_sums_to_one_and_peaks_where_player_stood():
    positions = np.array([[2.0, 2.0]] * 9 + [[100.0, 60.0]] + [[np.nan, np.nan]] * 5)
    grid = compute_heatmap(positions, bins=(21, 14))
    assert grid.shape == (14, 21)
    assert grid.sum() == pytest.approx(1.0)
    assert grid[0, 0] == pytest.approx(0.9)
    assert grid[12, 20] == pytest.approx(0.1)


def test_heatmap_clamps_points_outside_pitch():
    grid = compute_heatmap(np.array([[-1.0, -1.0], [106.0, 70.0]]), bins=(21, 14))
    assert grid[0, 0] == pytest.approx(0.5)
    assert grid[-1, -1] == pytest.approx(0.5)


def test_empty_heatmap_is_zero():
    assert compute_heatmap(np.full((3, 2), np.nan)).sum() == 0.0


def test_thirds_share():
    positions = np.array([[10.0, 0], [20.0, 0], [50.0, 0], [90.0, 0]])
    assert thirds_share(positions) == pytest.approx((0.5, 0.25, 0.25))
