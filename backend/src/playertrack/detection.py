"""Object detection: players, goalkeepers, referees and the ball."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .structures import BBox, Detection, ObjectClass

# Class names used by football-specific models (players / goalkeeper / referee / ball)
# and by the generic COCO models (person / sports ball).
CLASS_NAME_MAP: dict[str, ObjectClass] = {
    "player": ObjectClass.PLAYER,
    "person": ObjectClass.PLAYER,
    "goalkeeper": ObjectClass.GOALKEEPER,
    "referee": ObjectClass.REFEREE,
    "ball": ObjectClass.BALL,
    "sports ball": ObjectClass.BALL,
}


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    """Ultralytics YOLO model wrapped behind the :class:`Detector` interface.

    Works with the stock COCO weights, but a model fine-tuned on football footage
    (separate player / goalkeeper / referee / ball classes) gives much better results,
    especially for the ball.
    """

    def __init__(
        self,
        weights: str = "yolo11m.pt",
        confidence: float = 0.1,
        image_size: int = 1280,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "YOLO detection needs the 'detect' extra: pip install -e 'backend[detect]'"
            ) from exc
        self._model = YOLO(weights)
        self._confidence = confidence
        self._image_size = image_size
        self._device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            frame,
            conf=self._confidence,
            imgsz=self._image_size,
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        detections: list[Detection] = []
        for (x1, y1, x2, y2), confidence, class_id in zip(
            xyxy, confidences, class_ids, strict=True
        ):
            cls = CLASS_NAME_MAP.get(str(result.names[class_id]).lower())
            if cls is None:
                continue
            detections.append(
                Detection(BBox(float(x1), float(y1), float(x2), float(y2)), float(confidence), cls)
            )
        return detections
