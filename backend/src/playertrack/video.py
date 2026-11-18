"""Reading match videos frame by frame."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image as BGR. Unlike ``cv2.imread`` this also handles non-ASCII paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image, creating parent folders; the format follows the file extension."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return True


def _open(path: str | Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return capture


def read_video_info(path: str | Path) -> VideoInfo:
    capture = _open(path)
    try:
        return VideoInfo(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def iter_frames(
    path: str | Path,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR frame)`` for every ``stride``-th frame in ``[start, end)``.

    Skipped frames are only grabbed, not decoded, so a large stride is cheap.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    capture = _open(path)
    try:
        if start:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        while end is None or index < end:
            if not capture.grab():
                break
            if (index - start) % stride == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                yield index, frame
            index += 1
    finally:
        capture.release()
