from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .observations import FrameObservation


EXPECTED_COCO_KEYPOINTS = 17


@dataclass(frozen=True)
class ObservationQualityReport:
    status: str
    quality_score: float
    frame_count: int
    duration_sec: float
    scene_cut_count: int
    scene_cut_ratio: float
    total_player_detections: int
    total_ball_detections: int
    frames_with_players_ratio: float
    frames_with_ball_ratio: float
    average_players_per_frame: float
    player_tracking_coverage: float
    ball_tracking_coverage: float
    unique_player_tracks: int
    unique_ball_tracks: int
    average_player_track_length_frames: float
    average_ball_track_length_frames: float
    keypoint_coverage: float
    team_assignment_coverage: float
    team_a_detections: int
    team_b_detections: int
    calibrated_position_coverage: float
    field_geometry_frame_coverage: float
    average_field_line_segments: float
    average_field_intersections: float
    player_speed_p95: float | None
    ball_speed_p95: float | None
    speed_unit: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_observations_jsonl(path: Path) -> tuple[FrameObservation, ...]:
    frames: list[FrameObservation] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                frames.append(FrameObservation.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid observation at line {line_number}: {error}") from error
    return tuple(frames)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction, 3)


def analyze_observation_quality(
    frames: Iterable[FrameObservation],
    keypoint_confidence: float = 0.25,
) -> ObservationQualityReport:
    if not 0.0 <= keypoint_confidence <= 1.0:
        raise ValueError("keypoint_confidence must be between 0 and 1")
    sequence = tuple(frames)
    frame_indices = [frame.frame_index for frame in sequence]
    timestamps = [frame.timestamp_sec for frame in sequence]
    if frame_indices != sorted(frame_indices) or len(frame_indices) != len(set(frame_indices)):
        raise ValueError("observation frames must have unique, increasing frame_index values")
    if timestamps != sorted(timestamps):
        raise ValueError("observation timestamps must be increasing")

    frame_count = len(sequence)
    scene_cut_count = sum(frame.scene_cut for frame in sequence)
    scene_cut_ratio = _ratio(scene_cut_count, max(1, frame_count - 1))
    players = [
        item
        for frame in sequence
        for item in frame.objects
        if item.object_type in {"player", "goalkeeper"}
    ]
    balls = [
        item for frame in sequence for item in frame.objects if item.object_type == "ball"
    ]
    all_objects = [item for frame in sequence for item in frame.objects]

    frames_with_players = sum(
        any(item.object_type in {"player", "goalkeeper"} for item in frame.objects)
        for frame in sequence
    )
    frames_with_ball = sum(
        any(item.object_type == "ball" for item in frame.objects) for frame in sequence
    )
    player_track_lengths = Counter(
        item.track_id for item in players if item.track_id is not None
    )
    ball_track_lengths = Counter(item.track_id for item in balls if item.track_id is not None)

    visible_keypoints = 0
    for player in players:
        for keypoint in player.keypoints[:EXPECTED_COCO_KEYPOINTS]:
            confidence = keypoint[2] if len(keypoint) == 3 else 1.0
            visible_keypoints += confidence >= keypoint_confidence
    possible_keypoints = len(players) * EXPECTED_COCO_KEYPOINTS

    speed_units = {
        item.speed_unit for item in all_objects if item.speed_unit is not None
    }
    speed_unit = next(iter(speed_units)) if len(speed_units) == 1 else (
        "mixed" if speed_units else "unknown"
    )
    player_speeds = [item.speed for item in players if item.speed is not None]
    ball_speeds = [item.speed for item in balls if item.speed is not None]

    player_frame_coverage = _ratio(frames_with_players, frame_count)
    ball_frame_coverage = _ratio(frames_with_ball, frame_count)
    average_players = round(len(players) / frame_count, 3) if frame_count else 0.0
    player_tracking = _ratio(sum(player_track_lengths.values()), len(players))
    ball_tracking = _ratio(sum(ball_track_lengths.values()), len(balls))
    keypoint_coverage = _ratio(visible_keypoints, possible_keypoints)
    team_a_detections = sum(item.team == "team_a" for item in players)
    team_b_detections = sum(item.team == "team_b" for item in players)
    team_assignment_coverage = _ratio(
        team_a_detections + team_b_detections,
        len(players),
    )
    calibrated_coverage = _ratio(
        sum(item.pitch_x is not None and item.pitch_y is not None for item in all_objects),
        len(all_objects),
    )
    geometry_frames = [frame.field_geometry for frame in sequence if frame.field_geometry]
    field_geometry_coverage = _ratio(len(geometry_frames), frame_count)
    average_field_segments = (
        round(sum(len(item.segments) for item in geometry_frames) / len(geometry_frames), 3)
        if geometry_frames
        else 0.0
    )
    average_field_intersections = (
        round(
            sum(len(item.intersections) for item in geometry_frames) / len(geometry_frames),
            3,
        )
        if geometry_frames
        else 0.0
    )

    quality_score = 100.0 * (
        0.15 * player_frame_coverage
        + 0.15 * min(1.0, average_players / 10.0)
        + 0.25 * ball_frame_coverage
        + 0.15 * player_tracking
        + 0.10 * ball_tracking
        + 0.20 * keypoint_coverage
    )
    quality_score = round(quality_score, 1)

    warnings: list[str] = []
    if not players:
        warnings.append("no_player_detections")
    elif average_players < 6.0:
        warnings.append("low_average_players_per_frame")
    if not balls:
        warnings.append("no_ball_detections")
    elif ball_frame_coverage < 0.5:
        warnings.append("low_ball_frame_coverage")
    if players and player_tracking < 0.8:
        warnings.append("low_player_tracking_coverage")
    if balls and ball_tracking < 0.5:
        warnings.append("low_ball_tracking_coverage")
    if players and keypoint_coverage < 0.5:
        warnings.append("low_keypoint_coverage")
    if 0.0 < team_assignment_coverage < 0.7:
        warnings.append("low_team_assignment_coverage")

    player_speed_p95 = _percentile([float(value) for value in player_speeds], 0.95)
    ball_speed_p95 = _percentile([float(value) for value in ball_speeds], 0.95)
    if speed_unit == "km/h" and player_speed_p95 is not None and player_speed_p95 > 45:
        warnings.append("implausible_player_speed_p95")
    if scene_cut_ratio > 0.02:
        warnings.append("high_scene_cut_rate")

    critical = {"no_player_detections", "no_ball_detections"}
    if critical.intersection(warnings) or quality_score < 40.0:
        status = "fail"
    elif warnings or quality_score < 75.0:
        status = "warn"
    else:
        status = "pass"

    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    return ObservationQualityReport(
        status=status,
        quality_score=quality_score,
        frame_count=frame_count,
        duration_sec=round(duration, 3),
        scene_cut_count=scene_cut_count,
        scene_cut_ratio=scene_cut_ratio,
        total_player_detections=len(players),
        total_ball_detections=len(balls),
        frames_with_players_ratio=player_frame_coverage,
        frames_with_ball_ratio=ball_frame_coverage,
        average_players_per_frame=average_players,
        player_tracking_coverage=player_tracking,
        ball_tracking_coverage=ball_tracking,
        unique_player_tracks=len(player_track_lengths),
        unique_ball_tracks=len(ball_track_lengths),
        average_player_track_length_frames=(
            round(sum(player_track_lengths.values()) / len(player_track_lengths), 3)
            if player_track_lengths
            else 0.0
        ),
        average_ball_track_length_frames=(
            round(sum(ball_track_lengths.values()) / len(ball_track_lengths), 3)
            if ball_track_lengths
            else 0.0
        ),
        keypoint_coverage=keypoint_coverage,
        team_assignment_coverage=team_assignment_coverage,
        team_a_detections=team_a_detections,
        team_b_detections=team_b_detections,
        calibrated_position_coverage=calibrated_coverage,
        field_geometry_frame_coverage=field_geometry_coverage,
        average_field_line_segments=average_field_segments,
        average_field_intersections=average_field_intersections,
        player_speed_p95=player_speed_p95,
        ball_speed_p95=ball_speed_p95,
        speed_unit=speed_unit,
        warnings=tuple(warnings),
    )
