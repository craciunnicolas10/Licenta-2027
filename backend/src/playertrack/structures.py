"""Core data structures shared by the analysis modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ObjectClass(IntEnum):
    PLAYER = 0
    GOALKEEPER = 1
    REFEREE = 2
    BALL = 3


PERSON_CLASSES = frozenset({ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE})


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in image pixels (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid box coordinates: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where a standing player touches the pitch."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def shifted(self, dx: float, dy: float) -> BBox:
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def iou(self, other: BBox) -> float:
        inter_w = min(self.x2, other.x2) - max(self.x1, other.x1)
        inter_h = min(self.y2, other.y2) - max(self.y1, other.y1)
        if inter_w <= 0 or inter_h <= 0:
            return 0.0
        inter = inter_w * inter_h
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    confidence: float
    cls: ObjectClass


@dataclass(frozen=True, slots=True)
class Track:
    """A detection that the tracker has linked to a persistent identity."""

    track_id: int
    bbox: BBox
    cls: ObjectClass
    confidence: float
