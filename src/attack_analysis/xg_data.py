from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .models import PITCH_LENGTH, PITCH_WIDTH
from .xg import shot_angle, shot_distance


PRESSURE_RADIUS_M = 2.0
DEAD_BALL_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Goal Kick",
    "From Kick Off",
    "From Throw In",
}


def _pitch_point(location: Iterable[object] | None) -> tuple[float, float] | None:
    values = tuple(location or ())
    if len(values) < 2:
        return None
    return float(values[0]) / 120.0 * PITCH_LENGTH, float(values[1]) / 80.0 * PITCH_WIDTH


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def extract_statsbomb_shot_features(
    event: dict[str, Any],
    freeze_frame: Iterable[dict[str, Any]],
    *,
    match_id: str,
) -> dict[str, Any]:
    shooter = _pitch_point(event.get("location"))
    if shooter is None:
        raise ValueError("shot event is missing a two-dimensional location")
    visible = tuple(item for item in freeze_frame if isinstance(item, dict))
    opponents = tuple(item for item in visible if item.get("teammate") is False)
    allies = tuple(
        item for item in visible if item.get("teammate") is True and item.get("actor") is not True
    )
    goalkeeper = next((item for item in opponents if item.get("keeper") is True), None)
    goalkeeper_point = _pitch_point(goalkeeper.get("location")) if goalkeeper else None
    nearby_defenders = 0
    for opponent in opponents:
        if opponent.get("keeper") is True:
            continue
        point = _pitch_point(opponent.get("location"))
        if point is not None and _distance(shooter, point) <= PRESSURE_RADIUS_M:
            nearby_defenders += 1

    play_pattern = str((event.get("play_pattern") or {}).get("name") or "")
    shot = event.get("shot") or {}
    shot_type = str((shot.get("type") or {}).get("name") or "")
    outcome = str((shot.get("outcome") or {}).get("name") or "")
    timestamp = str(event.get("timestamp") or "00:00:00.000")
    parts = timestamp.replace(".", ":").split(":")
    second = int(float(parts[2])) if len(parts) >= 3 else int(event.get("second") or 0)

    return {
        "match_id": str(match_id),
        "event_id": str(event.get("id") or ""),
        "x_shooter": shooter[0],
        "y_shooter": shooter[1],
        "shot_angle": shot_angle(shooter),
        "distance_to_goal": shot_distance(shooter),
        "x_goalkeeper": goalkeeper_point[0] if goalkeeper_point else None,
        "y_goalkeeper": goalkeeper_point[1] if goalkeeper_point else None,
        "foe_players": len(opponents),
        "ally_players": len(allies),
        "defending_players": nearby_defenders,
        "under_pressure": int(bool(event.get("under_pressure")) or nearby_defenders > 0),
        "is_regular_play": int(play_pattern == "Regular Play"),
        "is_counter": int(play_pattern == "From Counter"),
        "is_dead_ball": int(play_pattern in DEAD_BALL_PATTERNS or shot_type != "Open Play"),
        "is_penalty": int(shot_type == "Penalty"),
        "minute": int(event.get("minute") or 0),
        "second": second,
        "goal": int(outcome == "Goal"),
        "statsbomb_xg": shot.get("statsbomb_xg"),
    }


def extract_statsbomb_shot_features_v2(
    event: dict[str, Any],
    freeze_frame: Iterable[dict[str, Any]] | None,
    *,
    match_id: str,
) -> dict[str, Any]:
    """Extract semantic xG-v2 inputs without replacing unknown context.

    Unlike the V1 compatibility exporter, missing pressure/play pattern/player
    visibility stays missing and is later represented by explicit masks.
    """

    shooter = _pitch_point(event.get("location"))
    if shooter is None:
        raise ValueError("shot event is missing a two-dimensional location")
    visibility_known = freeze_frame is not None
    visible = tuple(item for item in (freeze_frame or ()) if isinstance(item, dict))
    located = tuple(item for item in visible if _pitch_point(item.get("location")) is not None)
    opponents = tuple(item for item in located if item.get("teammate") is False)
    allies = tuple(
        item for item in located if item.get("teammate") is True and item.get("actor") is not True
    )
    goalkeeper = next((item for item in opponents if item.get("keeper") is True), None)
    goalkeeper_point = _pitch_point(goalkeeper.get("location")) if goalkeeper else None
    defenders = tuple(item for item in opponents if item.get("keeper") is not True)
    nearby_defenders = sum(
        _distance(shooter, point) <= PRESSURE_RADIUS_M
        for item in defenders
        if (point := _pitch_point(item.get("location"))) is not None
    )

    shot = event.get("shot") or {}
    body_part = str((shot.get("body_part") or {}).get("name") or "").strip() or None
    shot_type = str((shot.get("type") or {}).get("name") or "").strip() or None
    play_pattern = str((event.get("play_pattern") or {}).get("name") or "").strip() or None
    outcome = str((shot.get("outcome") or {}).get("name") or "")
    under_pressure = bool(event["under_pressure"]) if "under_pressure" in event else None
    return {
        "match_id": str(match_id),
        "event_id": str(event.get("id") or ""),
        "x_shooter": shooter[0],
        "y_shooter": shooter[1],
        "shot_angle": shot_angle(shooter),
        "distance_to_goal": shot_distance(shooter),
        "body_part": body_part,
        "under_pressure": under_pressure,
        "play_pattern": play_pattern,
        "shot_type": shot_type,
        "goalkeeper_visible": (goalkeeper is not None if visibility_known else None),
        "x_goalkeeper": goalkeeper_point[0] if goalkeeper_point else None,
        "y_goalkeeper": goalkeeper_point[1] if goalkeeper_point else None,
        "defenders_visible": (bool(defenders) if visibility_known else None),
        "observed_players": (len(located) if visibility_known else None),
        "foe_players": (len(opponents) if visibility_known else None),
        "ally_players": (len(allies) if visibility_known else None),
        "defending_players": (nearby_defenders if visibility_known else None),
        "goal": int(outcome == "Goal"),
        "provider_xg_reference": shot.get("statsbomb_xg"),
    }


def export_statsbomb_xg_dataset(open_data_root: Path, output_csv: Path) -> dict[str, Any]:
    events_directory = open_data_root / "events"
    freeze_directory = open_data_root / "three-sixty"
    if not events_directory.is_dir() or not freeze_directory.is_dir():
        raise FileNotFoundError("StatsBomb open-data root must contain events and three-sixty directories")

    rows: list[dict[str, Any]] = []
    matches = 0
    skipped_malformed: list[str] = []
    for freeze_path in sorted(freeze_directory.glob("*.json")):
        events_path = events_directory / freeze_path.name
        if not events_path.is_file():
            continue
        match_id = freeze_path.stem
        try:
            events = json.loads(events_path.read_text(encoding="utf-8"))
            freeze_records = json.loads(freeze_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped_malformed.append(match_id)
            continue
        freeze_by_event = {
            str(item.get("event_uuid")): item.get("freeze_frame", ())
            for item in freeze_records
            if item.get("event_uuid")
        }
        match_rows = 0
        for event in events:
            if str((event.get("type") or {}).get("name") or "") != "Shot":
                continue
            event_id = str(event.get("id") or "")
            if event_id not in freeze_by_event:
                continue
            try:
                rows.append(
                    extract_statsbomb_shot_features(
                        event,
                        freeze_by_event[event_id],
                        match_id=match_id,
                    )
                )
                match_rows += 1
            except ValueError:
                continue
        if match_rows:
            matches += 1

    if not rows:
        raise ValueError("no StatsBomb 360 shot rows were extracted")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "output": str(output_csv.resolve()),
        "matches": matches,
        "shots": len(rows),
        "goals": sum(int(row["goal"]) for row in rows),
        "goal_rate": sum(int(row["goal"]) for row in rows) / len(rows),
        "skipped_malformed_matches": skipped_malformed,
    }


def export_statsbomb_xg_v2_dataset(open_data_root: Path, output_csv: Path) -> dict[str, Any]:
    """Export all StatsBomb shots using the numeric xG-v2 feature contract.

    Matches or shots without 360 freeze frames are retained with visibility
    missingness masks instead of being silently treated as zero players.
    """

    from .xg_v2 import XG_V2_FEATURE_COLUMNS, normalize_xg_v2_features

    events_directory = open_data_root / "events"
    freeze_directory = open_data_root / "three-sixty"
    if not events_directory.is_dir():
        raise FileNotFoundError("StatsBomb open-data root must contain an events directory")

    rows: list[dict[str, Any]] = []
    matches = 0
    matches_with_360 = 0
    skipped_malformed: list[str] = []
    skipped_invalid_shots = 0
    for events_path in sorted(events_directory.glob("*.json")):
        match_id = events_path.stem
        try:
            events = json.loads(events_path.read_text(encoding="utf-8"))
            freeze_path = freeze_directory / events_path.name
            freeze_records = (
                json.loads(freeze_path.read_text(encoding="utf-8"))
                if freeze_path.is_file()
                else []
            )
        except (OSError, json.JSONDecodeError):
            skipped_malformed.append(match_id)
            continue
        freeze_by_event = {
            str(item.get("event_uuid")): item.get("freeze_frame", ())
            for item in freeze_records
            if item.get("event_uuid")
        }
        if freeze_by_event:
            matches_with_360 += 1
        match_rows = 0
        for event in events:
            if str((event.get("type") or {}).get("name") or "") != "Shot":
                continue
            event_id = str(event.get("id") or "")
            try:
                raw = extract_statsbomb_shot_features_v2(
                    event,
                    freeze_by_event.get(event_id),
                    match_id=match_id,
                )
            except ValueError:
                continue
            try:
                numeric = normalize_xg_v2_features(raw)
            except ValueError:
                skipped_invalid_shots += 1
                continue
            rows.append(
                {
                    "schema_version": "xg-features-v2",
                    "match_id": match_id,
                    "event_id": event_id,
                    **{name: numeric[name] for name in XG_V2_FEATURE_COLUMNS},
                    "body_part_raw": raw["body_part"],
                    "play_pattern_raw": raw["play_pattern"],
                    "shot_type_raw": raw["shot_type"],
                    "goal": raw["goal"],
                    "provider_xg_reference": raw["provider_xg_reference"],
                }
            )
            match_rows += 1
        if match_rows:
            matches += 1

    if not rows:
        raise ValueError("no StatsBomb shot rows were extracted")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema_version": "xg-features-v2",
        "output": str(output_csv.resolve()),
        "matches": matches,
        "matches_with_360": matches_with_360,
        "shots": len(rows),
        "goals": sum(int(row["goal"]) for row in rows),
        "goal_rate": sum(int(row["goal"]) for row in rows) / len(rows),
        "shots_with_missing_visibility": sum(
            int(float(row["observed_players_missing"]) == 1.0) for row in rows
        ),
        "feature_columns": list(XG_V2_FEATURE_COLUMNS),
        "skipped_invalid_shots": skipped_invalid_shots,
        "skipped_malformed_matches": skipped_malformed,
    }
