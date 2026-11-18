from playertrack.selection import TargetFollower, select_track_at
from playertrack.structures import BBox, ObjectClass, Track


def track(track_id: int, foot_x: float, foot_y: float = 140, cls=ObjectClass.PLAYER) -> Track:
    return Track(track_id, BBox(foot_x - 10, foot_y - 40, foot_x + 10, foot_y), cls, 0.9)


def test_click_inside_box_selects_player():
    tracks = [track(1, 100), track(2, 300)]
    assert select_track_at(tracks, (305, 120)).track_id == 2


def test_click_near_box_within_tolerance():
    tracks = [track(1, 100)]
    assert select_track_at(tracks, (125, 120), max_distance_px=40).track_id == 1
    assert select_track_at(tracks, (200, 120), max_distance_px=40) is None


def test_ball_is_never_selected():
    ball = Track(9, BBox(95, 95, 105, 105), ObjectClass.BALL, 0.9)
    assert select_track_at([ball], (100, 100)) is None


def test_follower_keeps_current_id():
    follower = TargetFollower(track(1, 100), frame_index=0, known_ids=[1, 2])
    assert follower.update(1, [track(1, 102), track(2, 300)]).track_id == 1
    assert follower.track_ids == [1]


def test_follower_reacquires_new_id_near_last_position():
    follower = TargetFollower(track(1, 100), frame_index=0, known_ids=[1, 2])
    assert follower.update(1, [track(2, 300)]) is None  # target occluded
    reacquired = follower.update(2, [track(2, 300), track(7, 110)])
    assert reacquired.track_id == 7
    assert follower.current_id == 7
    assert follower.track_ids == [1, 7]


def test_follower_does_not_jump_to_existing_neighbour():
    follower = TargetFollower(track(1, 100), frame_index=0, known_ids=[1, 2])
    # team-mate 2 is right next to where the target disappeared, but was already tracked
    assert follower.update(1, [track(2, 105)]) is None
    assert follower.current_id == 1


def test_follower_gives_up_after_long_gap():
    follower = TargetFollower(track(1, 100), frame_index=0, max_gap_frames=10)
    assert follower.update(50, [track(5, 100)]) is None
