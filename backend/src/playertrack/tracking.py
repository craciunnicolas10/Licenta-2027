"""Multi-object tracking by IoU association with constant-velocity prediction.

A lightweight ByteTrack-style tracker: high-confidence detections are matched to
existing tracks first, then the tracks that are still unmatched get a second chance
with the low-confidence detections (partially occluded players often score low).
Only high-confidence detections may start new tracks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .structures import BBox, Detection, ObjectClass, Track


def greedy_iou_match(
    boxes_a: Sequence[BBox],
    boxes_b: Sequence[BBox],
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match boxes by descending IoU. Returns (matches, unmatched_a, unmatched_b)."""
    pairs = [(a.iou(b), i, j) for i, a in enumerate(boxes_a) for j, b in enumerate(boxes_b)]
    pairs = sorted((p for p in pairs if p[0] >= threshold), key=lambda p: p[0], reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        matches.append((i, j))
        used_a.add(i)
        used_b.add(j)
    unmatched_a = [i for i in range(len(boxes_a)) if i not in used_a]
    unmatched_b = [j for j in range(len(boxes_b)) if j not in used_b]
    return matches, unmatched_a, unmatched_b


@dataclass
class _TrackState:
    track_id: int
    bbox: BBox
    cls: ObjectClass
    confidence: float
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    misses: int = 0

    def predict(self) -> BBox:
        steps = self.misses + 1
        return self.bbox.shifted(self.velocity[0] * steps, self.velocity[1] * steps)

    def update(self, detection: Detection, momentum: float) -> None:
        steps = self.misses + 1
        old_x, old_y = self.bbox.center
        new_x, new_y = detection.bbox.center
        vx = (new_x - old_x) / steps
        vy = (new_y - old_y) / steps
        self.velocity = (
            momentum * self.velocity[0] + (1.0 - momentum) * vx,
            momentum * self.velocity[1] + (1.0 - momentum) * vy,
        )
        self.bbox = detection.bbox
        self.cls = detection.cls
        self.confidence = detection.confidence
        self.hits += 1
        self.misses = 0

    def to_track(self) -> Track:
        return Track(self.track_id, self.bbox, self.cls, self.confidence)


class IouTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        high_confidence: float = 0.5,
        low_confidence: float = 0.1,
        max_misses: int = 30,
        min_hits: int = 1,
        velocity_momentum: float = 0.7,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.velocity_momentum = velocity_momentum
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Advance the tracker by one frame and return the tracks visible in it."""
        high = [d for d in detections if d.confidence >= self.high_confidence]
        low = [d for d in detections if self.low_confidence <= d.confidence < self.high_confidence]
        predicted = [t.predict() for t in self._tracks]

        matches, unmatched_tracks, unmatched_high = greedy_iou_match(
            predicted, [d.bbox for d in high], self.iou_threshold
        )
        for track_index, det_index in matches:
            self._tracks[track_index].update(high[det_index], self.velocity_momentum)

        second_matches, still_unmatched, _ = greedy_iou_match(
            [predicted[i] for i in unmatched_tracks], [d.bbox for d in low], self.iou_threshold
        )
        for local_index, det_index in second_matches:
            self._tracks[unmatched_tracks[local_index]].update(
                low[det_index], self.velocity_momentum
            )
        for local_index in still_unmatched:
            self._tracks[unmatched_tracks[local_index]].misses += 1

        for det_index in unmatched_high:
            det = high[det_index]
            self._tracks.append(_TrackState(self._next_id, det.bbox, det.cls, det.confidence))
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        return [t.to_track() for t in self._tracks if t.misses == 0 and t.hits >= self.min_hits]
