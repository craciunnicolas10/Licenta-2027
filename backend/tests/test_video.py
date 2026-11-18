import cv2
import numpy as np
import pytest

from playertrack.video import is_image, iter_frames, read_image, read_video_info, write_image


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


def test_image_round_trip_with_non_ascii_path(tmp_path):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[5:10, 5:10] = (0, 0, 255)
    path = tmp_path / "terenul_de_fotbal_ț.png"
    assert write_image(path, image)
    assert is_image(path)
    np.testing.assert_array_equal(read_image(path), image)


def test_read_missing_image(tmp_path):
    assert read_image(tmp_path / "nope.png") is None
    assert not is_image("match.mp4")


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_video_info(tmp_path / "nope.mp4")
