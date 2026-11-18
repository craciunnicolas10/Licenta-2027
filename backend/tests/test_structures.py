import pytest

from playertrack.structures import BBox


def test_box_geometry():
    box = BBox(10, 20, 30, 60)
    assert box.width == 20
    assert box.height == 40
    assert box.center == (20, 40)
    assert box.foot_point == (20, 60)
    assert box.contains(15, 59)
    assert not box.contains(31, 30)


def test_iou_identical_disjoint_and_partial():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0
    # 5x10 overlap, union 150
    assert a.iou(BBox(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_invalid_box_rejected():
    with pytest.raises(ValueError):
        BBox(10, 0, 5, 10)
