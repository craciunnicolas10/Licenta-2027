"""Ball possession and pass detection from per-frame player and ball positions.

A player is in control of the ball in a frame when they are the closest player to
it and within ``control_radius_m``. Frame-level decisions flicker, so very short
runs are discarded and consecutive runs of the same player separated by a small
gap (the ball pushed a bit ahead while dribbling) are merged into one spell.
A pass is a spell of the passer followed, within ``max_flight_s``, by a spell of
another player.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FrameState:
    timestamp: float
    ball: tuple[float, float] | None
    players: Mapping[int, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class PossessionSpell:
    player_id: int
    start_time: float
    end_time: float

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True, slots=True)
class Pass:
    passer_id: int
    receiver_id: int
    release_time: float
    reception_time: float
    completed: bool | None  # None while team membership is unknown

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PossessionSummary:
    possession_time_s: float
    touches: int
    passes_attempted: int
    passes_completed: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def ball_holder(state: FrameState, control_radius_m: float) -> int | None:
    if state.ball is None or not state.players:
        return None
    bx, by = state.ball
    player_id, (px, py) = min(
        state.players.items(), key=lambda item: math.hypot(item[1][0] - bx, item[1][1] - by)
    )
    return player_id if math.hypot(px - bx, py - by) <= control_radius_m else None


def possession_spells(
    states: Sequence[FrameState],
    control_radius_m: float = 1.5,
    min_frames: int = 3,
    merge_gap_s: float = 0.6,
) -> list[PossessionSpell]:
    runs: list[list[int]] = []  # [player_id, first_index, last_index]
    for index, state in enumerate(states):
        holder = ball_holder(state, control_radius_m)
        if holder is None:
            continue
        if runs and runs[-1][0] == holder and runs[-1][2] == index - 1:
            runs[-1][2] = index
        else:
            runs.append([holder, index, index])

    merged: list[list[int]] = []
    for run in (r for r in runs if r[2] - r[1] + 1 >= min_frames):
        if (
            merged
            and merged[-1][0] == run[0]
            and states[run[1]].timestamp - states[merged[-1][2]].timestamp <= merge_gap_s
        ):
            merged[-1][2] = run[2]
        else:
            merged.append(list(run))

    return [
        PossessionSpell(player_id, states[first].timestamp, states[last].timestamp)
        for player_id, first, last in merged
    ]


def detect_passes(
    spells: Sequence[PossessionSpell],
    passer_id: int,
    max_flight_s: float = 4.0,
    team_of: Callable[[int], int | None] | None = None,
) -> list[Pass]:
    passes: list[Pass] = []
    for current, following in zip(spells, spells[1:], strict=False):
        if current.player_id != passer_id or following.player_id == passer_id:
            continue
        if following.start_time - current.end_time > max_flight_s:
            continue
        completed: bool | None = None
        if team_of is not None:
            passer_team, receiver_team = team_of(current.player_id), team_of(following.player_id)
            if passer_team is not None and receiver_team is not None:
                completed = passer_team == receiver_team
        passes.append(
            Pass(
                passer_id=current.player_id,
                receiver_id=following.player_id,
                release_time=current.end_time,
                reception_time=following.start_time,
                completed=completed,
            )
        )
    return passes


def summarize_possession(
    spells: Sequence[PossessionSpell], passes: Sequence[Pass], player_id: int
) -> PossessionSummary:
    own = [s for s in spells if s.player_id == player_id]
    own_passes = [p for p in passes if p.passer_id == player_id]
    known = [p for p in own_passes if p.completed is not None]
    return PossessionSummary(
        possession_time_s=float(sum(s.duration_s for s in own)),
        touches=len(own),
        passes_attempted=len(own_passes),
        passes_completed=sum(p.completed for p in known) if known else None,
    )
