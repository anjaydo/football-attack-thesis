from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _optional_number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


@dataclass(frozen=True)
class Event:
    timestamp: float
    event_type: str
    team: str
    x: float | None = None
    y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    outcome: str | None = None
    xg: float | None = None
    xt_before: float | None = None
    xt_after: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        event = cls(
            timestamp=float(data["timestamp"]),
            event_type=str(data["event_type"]).lower(),
            team=str(data["team"]),
            x=_optional_number(data, "x"),
            y=_optional_number(data, "y"),
            end_x=_optional_number(data, "end_x"),
            end_y=_optional_number(data, "end_y"),
            outcome=(str(data["outcome"]).lower() if data.get("outcome") else None),
            xg=_optional_number(data, "xg"),
            xt_before=_optional_number(data, "xt_before"),
            xt_after=_optional_number(data, "xt_after"),
        )
        event.validate()
        return event

    def validate(self) -> None:
        for key, value, upper in (
            ("x", self.x, PITCH_LENGTH),
            ("end_x", self.end_x, PITCH_LENGTH),
            ("y", self.y, PITCH_WIDTH),
            ("end_y", self.end_y, PITCH_WIDTH),
        ):
            if value is not None and not 0.0 <= value <= upper:
                raise ValueError(f"{key} must be between 0 and {upper}")
        if self.xg is not None and not 0.0 <= self.xg <= 1.0:
            raise ValueError("xg must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class AttackSequence:
    attack_id: str
    team: str
    start_sec: float
    end_sec: float
    events: tuple[Event, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackSequence":
        sequence = cls(
            attack_id=str(data["attack_id"]),
            team=str(data["team"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            events=tuple(Event.from_dict(item) for item in data["events"]),
        )
        sequence.validate()
        return sequence

    def validate(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be greater than or equal to start_sec")
        if not self.events:
            raise ValueError("an attack must contain at least one event")
        if any(event.team != self.team for event in self.events):
            raise ValueError("all events must belong to the attacking team")
        timestamps = [event.timestamp for event in self.events]
        if timestamps != sorted(timestamps):
            raise ValueError("events must be ordered by timestamp")
        if timestamps[0] < self.start_sec or timestamps[-1] > self.end_sec:
            raise ValueError("event timestamps must fall within the attack interval")

    def to_dict(self) -> dict[str, object]:
        return {
            "attack_id": self.attack_id,
            "team": self.team,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "events": [event.to_dict() for event in self.events],
        }
