from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import AttackSequence, Event, PITCH_LENGTH, PITCH_WIDTH
from .xg import heuristic_xg


FOOTPASS_ACTIONS = {
    1: "drive",
    2: "pass",
    3: "cross",
    4: "throw_in",
    5: "shot",
    6: "header",
    7: "tackle",
    8: "block",
}

FOOTPASS_ROLES = {
    1: "goalkeeper",
    2: "left_back",
    3: "left_centre_back",
    4: "mid_centre_back",
    5: "right_centre_back",
    6: "left_midfielder",
    7: "right_midfielder",
    8: "defensive_midfielder",
    9: "attacking_midfielder",
    10: "left_winger",
    11: "right_winger",
    12: "central_forward",
    13: "right_back",
}


@dataclass(frozen=True)
class FootpassSegmentationConfig:
    fps: float = 25.0
    max_gap_sec: float = 4.0
    max_duration_sec: float = 45.0
    min_events: int = 1

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.max_gap_sec <= 0 or self.max_duration_sec <= 0:
            raise ValueError("fps and duration thresholds must be positive")
        if self.min_events < 1:
            raise ValueError("min_events must be positive")


def _read_playbyplay(path: Path) -> dict[str, list[list[object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, dict):
        raise ValueError("FOOTPASS play-by-play JSON must contain an events object")
    return {str(key): list(value) for key, value in events.items()}


def footpass_match_keys(path: Path) -> tuple[str, ...]:
    return tuple(sorted(_read_playbyplay(path)))


def _load_tactical_rows(path: Path, match_key: str) -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'footpass' extra to read tactical HDF5 data") from error
    with h5py.File(path, "r") as handle:
        if match_key not in handle:
            raise KeyError(f"{match_key} is not present in {path}")
        return handle[match_key][:]


def _find_tactical_row(
    rows: Any,
    frame_values: Any,
    frame: int,
    team: int,
    jersey: int,
    action: int,
) -> Any | None:
    import numpy as np

    # FOOTPASS rows are ordered by frame. Search the small frame slice instead of
    # scanning ~1.6 million rows once for every event.
    start = int(np.searchsorted(frame_values, frame, side="left"))
    end = int(np.searchsorted(frame_values, frame, side="right"))
    frame_rows = rows[start:end]
    candidates = frame_rows[(frame_rows[:, 2] == team) & (frame_rows[:, 3] == jersey)]
    if len(candidates) == 0:
        return None
    exact = candidates[candidates[:, 13] == action]
    if len(exact):
        return exact[0]
    visible = candidates[~np.isnan(candidates[:, 9])]
    return visible[0] if len(visible) else candidates[0]


def load_footpass_events(
    playbyplay_path: Path,
    *,
    match_key: str,
    source: str,
    tactical_h5: Path | None = None,
    fps: float = 25.0,
    min_confidence: float = 0.0,
    model_name: str | None = None,
) -> tuple[Event, ...]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    normalized_model = None
    if model_name:
        normalized_model = "".join(
            character for character in model_name.casefold() if character.isalnum() or character == "_"
        )
        if not normalized_model:
            raise ValueError("model_name must contain at least one letter or number")
    source_name = f"footpass_{source}"
    if normalized_model:
        source_name = f"{source_name}_{normalized_model}"
    groups = _read_playbyplay(playbyplay_path)
    if match_key not in groups:
        raise KeyError(f"{match_key} is not present in {playbyplay_path}")
    rows = _load_tactical_rows(tactical_h5, match_key) if tactical_h5 else None
    frame_values = rows[:, 0].copy() if rows is not None else None

    converted: list[Event] = []
    for index, raw in enumerate(groups[match_key]):
        if len(raw) < 4:
            raise ValueError(f"event {index} in {match_key} has fewer than four fields")
        frame, team, jersey, action = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        if action not in FOOTPASS_ACTIONS:
            raise ValueError(f"unknown FOOTPASS action class: {action}")
        confidence = float(raw[4]) if source == "predicted" and len(raw) >= 5 else 1.0
        if source == "predicted" and confidence < min_confidence:
            continue
        row = (
            _find_tactical_row(rows, frame_values, frame, team, jersey, action)
            if rows is not None
            else None
        )
        player_id = role = None
        x = y = None
        if row is not None:
            player_id = str(int(row[1]))
            role = FOOTPASS_ROLES.get(int(row[4]), f"role_{int(row[4])}")
            raw_x, raw_y = float(row[5]), float(row[6])
            x = raw_x * PITCH_LENGTH if team == 1 else (1.0 - raw_x) * PITCH_LENGTH
            y = raw_y * PITCH_WIDTH
            x = min(PITCH_LENGTH, max(0.0, x))
            y = min(PITCH_WIDTH, max(0.0, y))

        evidence_id = f"event:{match_key}:{frame}:{team}:{jersey}:{action}:{index}"
        converted.append(
            Event(
                timestamp=frame / fps,
                event_type=FOOTPASS_ACTIONS[action],
                team=str(team),
                x=x,
                y=y,
                xg=(heuristic_xg((x, y)) if action == 5 and x is not None and y is not None else None),
                xg_source=(
                    "heuristic_geometry_v1"
                    if action == 5 and x is not None and y is not None
                    else None
                ),
                frame=frame,
                player_id=player_id,
                jersey=jersey,
                role=role,
                confidence=confidence,
                source=source_name,
                evidence_id=evidence_id,
            )
        )
    return tuple(sorted(converted, key=lambda item: item.timestamp))


def segment_footpass_attacks(
    events: Iterable[Event],
    *,
    match_id: str,
    config: FootpassSegmentationConfig | None = None,
) -> tuple[AttackSequence, ...]:
    settings = config or FootpassSegmentationConfig()
    ordered = tuple(sorted(events, key=lambda item: item.timestamp))
    attacks: list[AttackSequence] = []
    current: list[Event] = []
    attack_number = 1

    def flush(reason: str) -> None:
        nonlocal attack_number
        if len(current) >= settings.min_events:
            first, last = current[0], current[-1]
            sequence = AttackSequence(
                attack_id=f"{match_id}:attack-{attack_number:04d}",
                match_id=match_id,
                team=first.team,
                start_sec=first.timestamp,
                end_sec=last.timestamp,
                start_frame=first.frame,
                end_frame=last.frame,
                termination_reason=reason,
                source=first.source,
                events=tuple(current),
            )
            sequence.validate()
            attacks.append(sequence)
            attack_number += 1
        current.clear()

    for event in ordered:
        if current:
            gap = event.timestamp - current[-1].timestamp
            duration = event.timestamp - current[0].timestamp
            if event.team != current[-1].team:
                flush("team_change")
            elif gap > settings.max_gap_sec:
                flush("action_gap")
            elif duration > settings.max_duration_sec:
                flush("max_duration")
        current.append(event)
        if event.event_type == "shot":
            flush("shot")
        elif event.event_type == "throw_in":
            flush("restart")

    if current:
        flush("end_of_stream")
    return tuple(attacks)


def footpass_summary(events: Iterable[Event], attacks: Iterable[AttackSequence]) -> dict[str, object]:
    event_items = tuple(events)
    attack_items = tuple(attacks)
    return {
        "event_count": len(event_items),
        "attack_count": len(attack_items),
        "actions": dict(Counter(item.event_type for item in event_items)),
        "shot_attacks": sum(any(event.event_type == "shot" for event in item.events) for item in attack_items),
        "no_shot_attacks": sum(not any(event.event_type == "shot" for event in item.events) for item in attack_items),
    }
