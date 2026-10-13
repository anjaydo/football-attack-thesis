from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod

from .models import AttackSequence


FINAL_THIRD_X = 70.0
PENALTY_BOX_X = 88.5
PENALTY_BOX_Y_MIN = 13.84
PENALTY_BOX_Y_MAX = 54.16
SHOT_EVENT_TYPES = frozenset({"shot", "headed_attempt"})


@dataclass(frozen=True)
class AttackMetrics:
    duration_sec: float
    event_count: int
    progression_m: float
    progression_ratio: float
    entered_final_third: bool
    entered_penalty_box: bool
    shots: int
    shots_on_target: int
    goals: int
    xg_sum: float
    attack_goal_probability: float
    positive_xt_gain: float
    ended_in_turnover: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _first_x(sequence: AttackSequence) -> float | None:
    for event in sequence.events:
        if event.x is not None:
            return event.x
    return None


def _last_x(sequence: AttackSequence) -> float | None:
    for event in reversed(sequence.events):
        # A shot's end point is the ball trajectory, not controlled progression.
        if event.event_type in SHOT_EVENT_TYPES and event.x is not None:
            return event.x
        if event.end_x is not None:
            return event.end_x
        if event.x is not None:
            return event.x
    return None


def _reached_zone(sequence: AttackSequence, min_x: float, y_range: tuple[float, float] | None = None) -> bool:
    for event in sequence.events:
        points = ((event.x, event.y), (event.end_x, event.end_y))
        for x, y in points:
            if x is None or x < min_x:
                continue
            if y_range is None or (y is not None and y_range[0] <= y <= y_range[1]):
                return True
    return False


def compute_metrics(sequence: AttackSequence) -> AttackMetrics:
    start_x = _first_x(sequence)
    end_x = _last_x(sequence)
    progression = max(0.0, (end_x or 0.0) - (start_x or 0.0))
    distance_available = max(1.0, 105.0 - (start_x or 0.0))

    shots = [event for event in sequence.events if event.event_type in SHOT_EVENT_TYPES]
    xg_values = [event.xg for event in shots if event.xg is not None]
    on_target_outcomes = {"goal", "saved", "on_target"}

    xt_gains = [
        event.xt_after - event.xt_before
        for event in sequence.events
        if event.xt_before is not None and event.xt_after is not None
    ]
    final_event = sequence.events[-1]

    return AttackMetrics(
        duration_sec=round(sequence.end_sec - sequence.start_sec, 3),
        event_count=len(sequence.events),
        progression_m=round(progression, 3),
        progression_ratio=round(min(1.0, progression / distance_available), 4),
        entered_final_third=_reached_zone(sequence, FINAL_THIRD_X),
        entered_penalty_box=_reached_zone(
            sequence,
            PENALTY_BOX_X,
            (PENALTY_BOX_Y_MIN, PENALTY_BOX_Y_MAX),
        ),
        shots=len(shots),
        shots_on_target=sum(event.outcome in on_target_outcomes for event in shots),
        goals=sum(event.outcome == "goal" for event in shots),
        xg_sum=round(sum(xg_values), 4),
        # This independence approximation is useful as a transparent baseline.
        attack_goal_probability=round(1.0 - prod(1.0 - value for value in xg_values), 4),
        positive_xt_gain=round(sum(max(0.0, gain) for gain in xt_gains), 4),
        ended_in_turnover=(
            final_event.event_type == "turnover" or final_event.outcome in {"lost", "failed"}
        ),
    )
