from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Iterable

from .geometry import Point
from .models import Event, PITCH_LENGTH
from .observations import FrameObservation, TrackedObject
from .xg import heuristic_xg


@dataclass(frozen=True)
class EventInferenceConfig:
    max_possession_distance_m: float = 3.0
    carry_min_distance_m: float = 3.0
    max_transfer_gap_sec: float = 2.0
    max_transfer_speed_mps: float = 45.0
    shot_window_sec: float = 1.5
    shot_min_x: float = 70.0
    shot_min_progress_m: float = 8.0
    shot_min_speed_mps: float = 8.0
    attack_directions: dict[str, str] = field(
        default_factory=lambda: {"team_a": "right", "team_b": "left"}
    )

    def __post_init__(self) -> None:
        if self.max_possession_distance_m <= 0:
            raise ValueError("max_possession_distance_m must be positive")
        if self.carry_min_distance_m <= 0:
            raise ValueError("carry_min_distance_m must be positive")
        if self.max_transfer_gap_sec <= 0:
            raise ValueError("max_transfer_gap_sec must be positive")
        if self.max_transfer_speed_mps <= 0:
            raise ValueError("max_transfer_speed_mps must be positive")
        if self.shot_window_sec <= 0 or self.shot_min_progress_m <= 0 or self.shot_min_speed_mps <= 0:
            raise ValueError("shot thresholds must be positive")
        if not 0.0 <= self.shot_min_x <= PITCH_LENGTH:
            raise ValueError("shot_min_x must be on the normalized pitch")
        if any(direction not in {"left", "right"} for direction in self.attack_directions.values()):
            raise ValueError("attack directions must be left or right")


@dataclass(frozen=True)
class _OwnedSample:
    timestamp_sec: float
    team: str
    track_id: str
    ball: Point
    scene_index: int


@dataclass
class _PossessionSegment:
    team: str
    track_id: str
    samples: list[_OwnedSample]

    @property
    def start(self) -> _OwnedSample:
        return self.samples[0]

    @property
    def end(self) -> _OwnedSample:
        return self.samples[-1]


@dataclass(frozen=True)
class _BallSample:
    timestamp_sec: float
    point: Point
    scene_index: int


def _position(item: TrackedObject) -> Point | None:
    if item.pitch_x is None or item.pitch_y is None:
        return None
    return item.pitch_x, item.pitch_y


def _owned_sample(
    frame: FrameObservation,
    max_distance: float,
    scene_index: int,
) -> _OwnedSample | None:
    balls = [
        item
        for item in frame.objects
        if item.object_type == "ball" and _position(item) is not None
    ]
    if not balls:
        return None
    ball = max(balls, key=lambda item: item.confidence)
    ball_position = _position(ball)
    assert ball_position is not None

    candidates = []
    for player in frame.objects:
        if player.object_type not in {"player", "goalkeeper"}:
            continue
        if player.track_id is None or player.team is None:
            continue
        player_position = _position(player)
        if player_position is None:
            continue
        distance = hypot(
            player_position[0] - ball_position[0],
            player_position[1] - ball_position[1],
        )
        if distance <= max_distance:
            candidates.append((distance, player))
    if not candidates:
        return None
    _, owner = min(candidates, key=lambda candidate: candidate[0])
    return _OwnedSample(
        timestamp_sec=frame.timestamp_sec,
        team=owner.team,
        track_id=owner.track_id,
        ball=ball_position,
        scene_index=scene_index,
    )


def _normalize(point: Point, team: str, config: EventInferenceConfig) -> Point:
    direction = config.attack_directions.get(team, "right")
    return (PITCH_LENGTH - point[0], point[1]) if direction == "left" else point


def _infer_shot(
    segment: _PossessionSegment,
    ball_samples: list[_BallSample],
    following: _PossessionSegment | None,
    config: EventInferenceConfig,
) -> Event | None:
    start = _normalize(segment.end.ball, segment.team, config)
    if start[0] < config.shot_min_x:
        return None
    if following is not None and following.team == segment.team:
        return None

    future = [
        sample
        for sample in ball_samples
        if sample.scene_index == segment.end.scene_index
        and segment.end.timestamp_sec < sample.timestamp_sec
        <= segment.end.timestamp_sec + config.shot_window_sec
    ]
    if not future:
        return None
    normalized = [(_normalize(sample.point, segment.team, config), sample) for sample in future]
    end, end_sample = max(normalized, key=lambda item: item[0][0])
    elapsed = end_sample.timestamp_sec - segment.end.timestamp_sec
    progress = end[0] - start[0]
    speed = hypot(end[0] - start[0], end[1] - start[1]) / elapsed
    if progress < config.shot_min_progress_m or speed < config.shot_min_speed_mps:
        return None
    return Event(
        timestamp=segment.end.timestamp_sec,
        event_type="shot",
        team=segment.team,
        x=start[0],
        y=start[1],
        end_x=end[0],
        end_y=end[1],
        outcome="attempt",
        xg=heuristic_xg(start),
        xg_source="heuristic_geometry_v1",
    )


def infer_events_from_observations(
    frames: Iterable[FrameObservation],
    config: EventInferenceConfig | None = None,
) -> tuple[Event, ...]:
    settings = config or EventInferenceConfig()
    frame_sequence = tuple(frames)
    samples: list[_OwnedSample] = []
    ball_samples: list[_BallSample] = []
    scene_index = 0
    for frame in frame_sequence:
        if frame.scene_cut:
            scene_index += 1
        calibrated_balls = [
            item
            for item in frame.objects
            if item.object_type == "ball" and _position(item) is not None
        ]
        if calibrated_balls:
            ball = max(calibrated_balls, key=lambda item: item.confidence)
            ball_position = _position(ball)
            assert ball_position is not None
            ball_samples.append(_BallSample(frame.timestamp_sec, ball_position, scene_index))
        sample = _owned_sample(
            frame,
            settings.max_possession_distance_m,
            scene_index,
        )
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise ValueError(
            "no possession samples found; calibrated pitch coordinates and team labels are required"
        )

    segments: list[_PossessionSegment] = []
    for sample in samples:
        if (
            segments
            and segments[-1].team == sample.team
            and segments[-1].track_id == sample.track_id
            and segments[-1].end.scene_index == sample.scene_index
            and sample.timestamp_sec - segments[-1].end.timestamp_sec
            <= settings.max_transfer_gap_sec
        ):
            segments[-1].samples.append(sample)
        else:
            segments.append(_PossessionSegment(sample.team, sample.track_id, [sample]))

    events: list[Event] = []

    def possession_start(segment: _PossessionSegment) -> Event:
        x, y = _normalize(segment.start.ball, segment.team, settings)
        return Event(
            timestamp=segment.start.timestamp_sec,
            event_type="possession_start",
            team=segment.team,
            x=x,
            y=y,
            outcome="controlled",
        )

    events.append(possession_start(segments[0]))
    for index, segment in enumerate(segments):
        start = _normalize(segment.start.ball, segment.team, settings)
        end = _normalize(segment.end.ball, segment.team, settings)
        displacement = hypot(end[0] - start[0], end[1] - start[1])
        if displacement >= settings.carry_min_distance_m:
            events.append(
                Event(
                    timestamp=segment.end.timestamp_sec,
                    event_type="carry",
                    team=segment.team,
                    x=start[0],
                    y=start[1],
                    end_x=end[0],
                    end_y=end[1],
                    outcome="complete",
                )
            )

        following = segments[index + 1] if index + 1 < len(segments) else None
        shot = _infer_shot(segment, ball_samples, following, settings)
        if shot is not None:
            events.append(shot)
            if following is not None:
                events.append(possession_start(following))
            continue

        if following is None:
            continue
        transfer_gap = following.start.timestamp_sec - segment.end.timestamp_sec
        transfer_distance = hypot(
            following.start.ball[0] - segment.end.ball[0],
            following.start.ball[1] - segment.end.ball[1],
        )
        transfer_speed = (
            transfer_distance / transfer_gap if transfer_gap > 0 else float("inf")
        )
        if (
            following.start.scene_index != segment.end.scene_index
            or transfer_gap > settings.max_transfer_gap_sec
            or transfer_speed > settings.max_transfer_speed_mps
        ):
            events.append(possession_start(following))
            continue

        if following.team == segment.team:
            pass_end = _normalize(following.start.ball, segment.team, settings)
            events.append(
                Event(
                    timestamp=following.start.timestamp_sec,
                    event_type="pass",
                    team=segment.team,
                    x=end[0],
                    y=end[1],
                    end_x=pass_end[0],
                    end_y=pass_end[1],
                    outcome="complete",
                )
            )
        else:
            events.append(
                Event(
                    timestamp=segment.end.timestamp_sec,
                    event_type="turnover",
                    team=segment.team,
                    x=end[0],
                    y=end[1],
                    outcome="lost",
                )
            )
            events.append(possession_start(following))

    events.sort(key=lambda event: event.timestamp)
    return tuple(events)
