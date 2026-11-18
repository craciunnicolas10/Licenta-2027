import pytest

from playertrack.possession import (
    FrameState,
    ball_holder,
    detect_passes,
    possession_spells,
    summarize_possession,
)

FPS = 25.0
A, B, C = 1, 2, 3
PLAYERS = {A: (10.0, 10.0), B: (30.0, 10.0), C: (50.0, 10.0)}
NOWHERE = (20.0, 40.0)


def scene(ball_owners: list[int | None]) -> list[FrameState]:
    """One frame per entry: the ball sits at that player's feet, or far from everyone."""
    return [
        FrameState(i / FPS, PLAYERS[owner] if owner else NOWHERE, PLAYERS)
        for i, owner in enumerate(ball_owners)
    ]


def test_ball_holder_respects_radius():
    state = FrameState(0.0, (11.0, 10.0), PLAYERS)
    assert ball_holder(state, 1.5) == A
    assert ball_holder(state, 0.5) is None
    assert ball_holder(FrameState(0.0, None, PLAYERS), 1.5) is None


def test_spells_drop_flicker_and_merge_dribbles():
    owners = [A] * 10 + [B] * 2 + [A] * 10 + [None] * 5 + [A] * 10 + [None] * 20 + [B] * 10
    spells = possession_spells(scene(owners), min_frames=3, merge_gap_s=0.6)
    assert [s.player_id for s in spells] == [A, B]
    assert spells[0].start_time == 0.0
    assert spells[0].end_time == pytest.approx(36 / FPS)


def test_pass_detection_with_and_without_teams():
    owners = [A] * 10 + [None] * 10 + [B] * 10 + [None] * 5 + [A] * 10 + [None] * 5 + [C] * 10
    spells = possession_spells(scene(owners))
    passes = detect_passes(spells, passer_id=A)
    assert [(p.passer_id, p.receiver_id) for p in passes] == [(A, B), (A, C)]
    assert all(p.completed is None for p in passes)

    teams = {A: 0, B: 0, C: 1}
    with_teams = detect_passes(spells, passer_id=A, team_of=teams.get)
    assert [p.completed for p in with_teams] == [True, False]

    summary = summarize_possession(spells, with_teams, A)
    assert summary.touches == 2
    assert summary.passes_attempted == 2
    assert summary.passes_completed == 1


def test_long_ball_flight_is_not_a_pass():
    owners = [A] * 10 + [None] * 150 + [B] * 10
    assert detect_passes(possession_spells(scene(owners)), A, max_flight_s=4.0) == []
