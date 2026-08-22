from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import AttackSequence, Event


@dataclass(frozen=True)
class SegmentationConfig:
    max_gap_sec: float = 15.0
    min_events: int = 1
    terminal_event_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"shot", "turnover", "ball_out", "offside"})
    )

    def __post_init__(self) -> None:
        if self.max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be positive")
        if self.min_events < 1:
            raise ValueError("min_events must be at least 1")


def segment_attacks(
    events: Iterable[Event],
    config: SegmentationConfig | None = None,
) -> tuple[AttackSequence, ...]:
    settings = config or SegmentationConfig()
    ordered = tuple(events)
    timestamps = [event.timestamp for event in ordered]
    if timestamps != sorted(timestamps):
        raise ValueError("match events must be ordered by timestamp")

    attacks: list[AttackSequence] = []
    current: list[Event] = []
    attack_number = 1

    def flush() -> None:
        nonlocal attack_number
        if len(current) >= settings.min_events:
            sequence = AttackSequence(
                attack_id=f"attack-{attack_number:04d}",
                team=current[0].team,
                start_sec=current[0].timestamp,
                end_sec=current[-1].timestamp,
                events=tuple(current),
            )
            sequence.validate()
            attacks.append(sequence)
            attack_number += 1
        current.clear()

    for event in ordered:
        if current:
            changed_team = event.team != current[-1].team
            exceeded_gap = event.timestamp - current[-1].timestamp > settings.max_gap_sec
            if changed_team or exceeded_gap:
                flush()

        current.append(event)
        if event.event_type in settings.terminal_event_types:
            flush()

    if current:
        flush()
    return tuple(attacks)
