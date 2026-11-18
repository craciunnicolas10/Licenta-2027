"""Choosing the player to analyse and keeping hold of them when track IDs change.

The user clicks on a player in one frame. From then on we follow that track ID.
Trackers lose identities during occlusions, so when the target's ID disappears we
re-acquire it: the first *new* track that appears close to where the target was
last seen is taken to be the same player. Tracks that already existed are never
taken over, which stops the target from jumping onto a nearby team-mate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .structures import ObjectClass, Track


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def select_track_at(
    tracks: Sequence[Track],
    point: tuple[float, float],
    max_distance_px: float = 40.0,
) -> Track | None:
    """Return the player track under ``point`` (a click in image pixels).

    Prefers boxes that contain the point; otherwise falls back to the nearest box
    centre within ``max_distance_px``.
    """
    people = [t for t in tracks if t.cls is not ObjectClass.BALL]
    containing = [t for t in people if t.bbox.contains(*point)]
    if containing:
        return min(containing, key=lambda t: _distance(t.bbox.center, point))
    nearest = min(people, key=lambda t: _distance(t.bbox.center, point), default=None)
    if nearest is not None and _distance(nearest.bbox.center, point) <= max_distance_px:
        return nearest
    return None


class TargetFollower:
    def __init__(
        self,
        target: Track,
        frame_index: int,
        known_ids: Iterable[int] = (),
        reacquire_radius_px: float = 60.0,
        radius_growth_px: float = 4.0,
        max_gap_frames: int = 60,
    ) -> None:
        self.reacquire_radius_px = reacquire_radius_px
        self.radius_growth_px = radius_growth_px
        self.max_gap_frames = max_gap_frames
        self.track_ids: list[int] = [target.track_id]
        self._current_id = target.track_id
        self._last_point = target.bbox.foot_point
        self._last_frame = frame_index
        self._known_ids: set[int] = set(known_ids) | {target.track_id}

    @property
    def current_id(self) -> int:
        return self._current_id

    @property
    def last_seen_frame(self) -> int:
        return self._last_frame

    def update(self, frame_index: int, tracks: Sequence[Track]) -> Track | None:
        """Return the target's track in this frame, or ``None`` if it is not visible."""
        target = next((t for t in tracks if t.track_id == self._current_id), None)
        if target is None:
            target = self._reacquire(frame_index, tracks)
        self._known_ids.update(t.track_id for t in tracks)
        if target is not None:
            self._last_point = target.bbox.foot_point
            self._last_frame = frame_index
        return target

    def _reacquire(self, frame_index: int, tracks: Sequence[Track]) -> Track | None:
        gap = frame_index - self._last_frame
        if gap > self.max_gap_frames:
            return None
        radius = self.reacquire_radius_px + self.radius_growth_px * gap
        candidates = [
            t for t in tracks if t.track_id not in self._known_ids and t.cls is not ObjectClass.BALL
        ]
        best = min(
            candidates,
            key=lambda t: _distance(t.bbox.foot_point, self._last_point),
            default=None,
        )
        if best is None or _distance(best.bbox.foot_point, self._last_point) > radius:
            return None
        self._current_id = best.track_id
        self.track_ids.append(best.track_id)
        return best
