from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import hypot
from typing import Iterable

from .geometry import Point
from .models import Event, PITCH_LENGTH
from .observations import FrameObservation, TrackedObject
from .xg import shot_angle, shot_distance
from .xg import TrainedXGModel


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
    shot_min_end_x: float = 96.0
    min_ball_confidence: float = 0.35
    max_ball_speed_mps: float = 40.0
    ball_continuity_gap_sec: float = 0.50
    dense_frame_dt_sec: float = 0.15
    min_stable_run_samples: int = 4
    min_owner_dwell_samples: int = 3
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
        if not 0.0 <= self.shot_min_end_x <= PITCH_LENGTH:
            raise ValueError("shot_min_end_x must be on the normalized pitch")
        if not 0.0 <= self.min_ball_confidence <= 1.0:
            raise ValueError("min_ball_confidence must be between 0 and 1")
        if self.max_ball_speed_mps <= 0 or self.ball_continuity_gap_sec <= 0:
            raise ValueError("ball continuity thresholds must be positive")
        if self.dense_frame_dt_sec <= 0:
            raise ValueError("dense_frame_dt_sec must be positive")
        if self.min_stable_run_samples < 2 or self.min_owner_dwell_samples < 1:
            raise ValueError("stable-run and owner-dwell lengths must be positive")
        if any(direction not in {"left", "right"} for direction in self.attack_directions.values()):
            raise ValueError("attack directions must be left or right")


@dataclass(frozen=True)
class _OwnedSample:
    timestamp_sec: float
    team: str
    track_id: str
    ball: Point
    scene_index: int
    run_id: int = 0


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
    run_id: int


@dataclass(frozen=True)
class _TrackedBall:
    frame_index: int
    timestamp_sec: float
    point: Point
    scene_index: int
    run_id: int


def _position(item: TrackedObject) -> Point | None:
    if item.pitch_x is None or item.pitch_y is None:
        return None
    return item.pitch_x, item.pitch_y


def _video_is_dense(frames: tuple[FrameObservation, ...], config: EventInferenceConfig) -> bool:
    deltas = [
        frames[index].timestamp_sec - frames[index - 1].timestamp_sec
        for index in range(1, len(frames))
        if frames[index].timestamp_sec > frames[index - 1].timestamp_sec
    ]
    if not deltas:
        return False
    deltas.sort()
    return deltas[len(deltas) // 2] <= config.dense_frame_dt_sec


def _stable_ball_track(
    frames: tuple[FrameObservation, ...],
    config: EventInferenceConfig,
) -> tuple[_TrackedBall, ...]:
    """Keep ball points that can be the same physical ball.

    A broadcast frame rate makes a one-frame jump of tens of metres obvious.
    Those points never join a run, and a run shorter than
    ``min_stable_run_samples`` is dropped. Sparse observations (about 1 Hz in
    tests, or a ball that reappears after occlusion) are kept one point at a
    time because a real touch is not expected to fill every frame.
    """

    dense = _video_is_dense(frames, config)
    scene_index = 0
    raw: list[tuple[int, float, Point, int]] = []
    for frame_index, frame in enumerate(frames):
        if frame.scene_cut:
            scene_index += 1
        balls = [
            item
            for item in frame.objects
            if item.object_type == "ball"
            and item.confidence >= config.min_ball_confidence
            and _position(item) is not None
        ]
        if not balls:
            continue
        ball = max(balls, key=lambda item: item.confidence)
        point = _position(ball)
        assert point is not None
        raw.append((frame_index, frame.timestamp_sec, point, scene_index))

    runs: list[list[tuple[int, float, Point, int]]] = []
    current: list[tuple[int, float, Point, int]] = []
    for sample in raw:
        if not current:
            current = [sample]
            continue
        previous = current[-1]
        elapsed = sample[1] - previous[1]
        distance = hypot(sample[2][0] - previous[2][0], sample[2][1] - previous[2][1])
        speed = distance / elapsed if elapsed > 0 else float("inf")
        joins = (
            sample[3] == previous[3]
            and 0.0 < elapsed <= config.ball_continuity_gap_sec
            and speed <= config.max_ball_speed_mps + 1e-6
        )
        if joins:
            current.append(sample)
        else:
            runs.append(current)
            current = [sample]
    if current:
        runs.append(current)

    kept: list[_TrackedBall] = []
    for run_id, run in enumerate(runs):
        if dense and len(run) < config.min_stable_run_samples:
            continue
        kept.extend(
            _TrackedBall(
                frame_index=sample[0],
                timestamp_sec=sample[1],
                point=sample[2],
                scene_index=sample[3],
                run_id=run_id,
            )
            for sample in run
        )
    return tuple(kept)


def _commit_owner_dwell(
    samples: list[_OwnedSample],
    *,
    dense: bool,
    min_dwell: int,
) -> list[_OwnedSample]:
    """Drop owner labels that last only one or two frames on a video-rate track."""

    if not dense or min_dwell <= 1 or len(samples) <= 1:
        return samples
    kept: list[_OwnedSample] = [samples[0]]
    committed = (samples[0].team, samples[0].track_id, samples[0].scene_index)
    pending: list[_OwnedSample] = []
    for sample in samples[1:]:
        key = (sample.team, sample.track_id, sample.scene_index)
        if key == committed:
            pending.clear()
            kept.append(sample)
            continue
        if pending and (
            pending[0].team,
            pending[0].track_id,
            pending[0].scene_index,
        ) != key:
            pending.clear()
        pending.append(sample)
        if len(pending) >= min_dwell:
            kept.extend(pending)
            committed = key
            pending.clear()
    return kept


def _owned_sample(
    frame: FrameObservation,
    ball_position: Point,
    max_distance: float,
    scene_index: int,
    run_id: int,
) -> _OwnedSample | None:
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
        run_id=run_id,
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
    """Emit trajectory evidence only; never assert an attempt on goal.

    A fast forward ball can be a pass, clearance, cross, or shot.  The ball
    must stay on one continuous run and actually arrive near the goal.  A
    one-frame jump to the goal line is not a shot.
    """
    start = _normalize(segment.end.ball, segment.team, config)
    if start[0] < config.shot_min_x:
        return None
    if following is not None and following.team == segment.team:
        return None

    future = [
        sample
        for sample in ball_samples
        if sample.run_id == segment.end.run_id
        and sample.scene_index == segment.end.scene_index
        and segment.end.timestamp_sec < sample.timestamp_sec
        <= segment.end.timestamp_sec + config.shot_window_sec
    ]
    if len(future) < 2:
        return None
    normalized = [(_normalize(sample.point, segment.team, config), sample) for sample in future]
    end, end_sample = max(normalized, key=lambda item: item[0][0])
    elapsed = end_sample.timestamp_sec - segment.end.timestamp_sec
    if elapsed <= 0:
        return None
    progress = end[0] - start[0]
    speed = hypot(end[0] - start[0], end[1] - start[1]) / elapsed
    if progress < config.shot_min_progress_m or speed < config.shot_min_speed_mps:
        return None
    if speed > config.max_ball_speed_mps or end[0] < config.shot_min_end_x:
        return None
    return Event(
        timestamp=segment.end.timestamp_sec,
        event_type="shot_candidate",
        team=segment.team,
        x=start[0],
        y=start[1],
        end_x=end[0],
        end_y=end[1],
        outcome="unverified",
        source="video_forward_ball_candidate_v1",
    )


def infer_events_from_observations(
    frames: Iterable[FrameObservation],
    config: EventInferenceConfig | None = None,
) -> tuple[Event, ...]:
    settings = config or EventInferenceConfig()
    frame_sequence = tuple(frames)
    dense = _video_is_dense(frame_sequence, settings)
    tracked_balls = _stable_ball_track(frame_sequence, settings)
    samples: list[_OwnedSample] = []
    ball_samples: list[_BallSample] = []
    for tracked in tracked_balls:
        ball_samples.append(
            _BallSample(
                tracked.timestamp_sec,
                tracked.point,
                tracked.scene_index,
                tracked.run_id,
            )
        )
        sample = _owned_sample(
            frame_sequence[tracked.frame_index],
            tracked.point,
            settings.max_possession_distance_m,
            tracked.scene_index,
            tracked.run_id,
        )
        if sample is not None:
            samples.append(sample)
    samples = _commit_owner_dwell(
        samples,
        dense=dense,
        min_dwell=settings.min_owner_dwell_samples,
    )
    if not samples:
        raise ValueError(
            "no stable ball track; detections jump faster than a real ball, "
            "or no calibrated player is close enough to own it"
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

    if dense:
        segments = [
            segment
            for segment in segments
            if len(segment.samples) >= settings.min_owner_dwell_samples
        ]
    if not segments:
        raise ValueError(
            "no stable possession; owner labels flicker faster than a real touch"
        )

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


def enrich_shots_with_trained_xg(
    events: Iterable[Event],
    frames: Iterable[FrameObservation],
    model: TrainedXGModel,
    config: EventInferenceConfig | None = None,
) -> tuple[Event, ...]:
    """Replace geometry xG with the trained model using calibrated video context."""
    from .xg_data import PRESSURE_RADIUS_M

    settings = config or EventInferenceConfig()
    frame_sequence = tuple(frames)
    enriched: list[Event] = []
    for event in events:
        if event.event_type != "shot" or event.x is None or event.y is None:
            enriched.append(event)
            continue
        nearest = (
            min(
                frame_sequence,
                key=lambda frame: abs(frame.timestamp_sec - event.timestamp),
            )
            if frame_sequence
            else None
        )
        visible = []
        if nearest is not None:
            for item in nearest.objects:
                position = _position(item)
                if (
                    item.object_type not in {"player", "goalkeeper"}
                    or item.team is None
                    or position is None
                ):
                    continue
                visible.append((item, _normalize(position, event.team, settings)))
        opponents = [item for item in visible if item[0].team != event.team]
        allies = [item for item in visible if item[0].team == event.team]
        goalkeeper = next(
            (position for item, position in opponents if item.object_type == "goalkeeper"),
            None,
        )
        nearby_defenders = sum(
            hypot(position[0] - event.x, position[1] - event.y) <= PRESSURE_RADIUS_M
            for _, position in opponents
        )
        features: dict[str, float | int | None] = {
            "x_shooter": event.x,
            "y_shooter": event.y,
            "shot_angle": shot_angle((event.x, event.y)),
            "distance_to_goal": shot_distance((event.x, event.y)),
            "x_goalkeeper": goalkeeper[0] if goalkeeper else None,
            "y_goalkeeper": goalkeeper[1] if goalkeeper else None,
            "foe_players": len(opponents),
            "ally_players": max(0, len(allies) - 1),
            "defending_players": nearby_defenders,
            "under_pressure": int(nearby_defenders > 0),
            "is_regular_play": 1,
            "is_counter": 0,
            "is_dead_ball": 0,
            "is_penalty": 0,
            "minute": int(event.timestamp // 60),
            "second": int(event.timestamp % 60),
        }
        enriched.append(
            replace(
                event,
                xg=model.predict(features),
                xg_source=model.source,
                xg_inputs=features,
            )
        )
    return tuple(enriched)
