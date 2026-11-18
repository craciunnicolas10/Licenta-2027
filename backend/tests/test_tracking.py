from playertrack.structures import BBox, Detection, ObjectClass
from playertrack.tracking import IouTracker, greedy_iou_match


def det(x: float, y: float = 100, confidence: float = 0.9) -> Detection:
    return Detection(BBox(x, y, x + 20, y + 40), confidence, ObjectClass.PLAYER)


def ids_by_x(tracks):
    return {round(t.bbox.x1): t.track_id for t in tracks}


def test_greedy_match_prefers_highest_iou():
    a = [BBox(0, 0, 10, 10), BBox(100, 0, 110, 10)]
    b = [BBox(101, 0, 111, 10), BBox(1, 0, 11, 10), BBox(500, 0, 510, 10)]
    matches, unmatched_a, unmatched_b = greedy_iou_match(a, b, 0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert unmatched_a == []
    assert unmatched_b == [2]


def test_ids_stay_stable_for_moving_players():
    tracker = IouTracker()
    first = ids_by_x(tracker.update([det(0), det(200)]))
    for step in range(1, 20):
        tracks = tracker.update([det(0 + 3 * step), det(200 - 3 * step)])
        current = {t.track_id for t in tracks}
        assert current == set(first.values())
    by_x = ids_by_x(tracks)
    assert by_x[57] == first[0]
    assert by_x[143] == first[200]


def test_track_survives_short_occlusion():
    tracker = IouTracker(max_misses=5)
    (track,) = tracker.update([det(0)])
    for step in range(1, 5):
        tracker.update([det(2 * step)])
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    (back,) = tracker.update([det(14)])
    assert back.track_id == track.track_id


def test_track_dropped_after_max_misses():
    tracker = IouTracker(max_misses=2)
    (track,) = tracker.update([det(0)])
    for _ in range(3):
        tracker.update([])
    (new,) = tracker.update([det(0)])
    assert new.track_id != track.track_id


def test_low_confidence_detection_continues_but_cannot_start_track():
    tracker = IouTracker(high_confidence=0.5, low_confidence=0.1)
    (track,) = tracker.update([det(0)])
    (continued,) = tracker.update([det(2, confidence=0.3)])
    assert continued.track_id == track.track_id

    tracks = tracker.update([det(4, confidence=0.3), det(400, confidence=0.3)])
    assert [t.track_id for t in tracks] == [track.track_id]
