import cv2
import numpy as np
import pytest

from playertrack.video import iter_frames, read_video_info


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def test_video_info(tiny_video):
    info = read_video_info(tiny_video)
    assert info.fps == pytest.approx(25.0)
    assert (info.width, info.height) == (64, 48)
    assert info.frame_count == 10
    assert info.duration_s == pytest.approx(0.4)


def test_iter_frames_with_stride(tiny_video):
    indices = [i for i, _ in iter_frames(tiny_video, stride=3)]
    assert indices == [0, 3, 6, 9]


def test_iter_frames_range(tiny_video):
    indices = [i for i, _ in iter_frames(tiny_video, start=2, end=5)]
    assert indices == [2, 3, 4]


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_video_info(tmp_path / "nope.mp4")
