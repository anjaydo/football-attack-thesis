from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping

from .models import PITCH_LENGTH, PITCH_WIDTH, AttackSequenceV2, ObservedActionV2, TerminalType
from .observations import BoundingBox, FrameObservation, TrackedObject
from .xg_data import PRESSURE_RADIUS_M
from .xg_v2 import ConfirmedXGRecordV2, TrainedXGModelV2


PLAYER_TYPES = frozenset({"player", "goalkeeper"})
COCO_NOSE = 0
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16
PENALTY_BOX_DEPTH_M = 16.5


@dataclass(frozen=True)
class FusionAssociation:
    action_id: str
    association_id: str
    track_id: str | None
    method: str
    distance_m: float | None
    iou: float | None
    xg_raw: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "association_id": self.association_id,
            "track_id": self.track_id,
            "method": self.method,
            "distance_m": self.distance_m,
            "iou": self.iou,
            "xg_raw": dict(self.xg_raw),
        }


@dataclass(frozen=True)
class FusedXGResult:
    association: FusionAssociation
    xg_record: ConfirmedXGRecordV2 | None
    withheld_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "association": self.association.to_dict(),
            "xg_record": None if self.xg_record is None else self.xg_record.to_dict(),
            "withheld_reason": self.withheld_reason,
        }


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = (
        (first.x2 - first.x1) * (first.y2 - first.y1)
        + (second.x2 - second.x1) * (second.y2 - second.y1)
        - intersection
    )
    return intersection / union if union else 0.0


def _pitch_players(frame: FrameObservation) -> tuple[TrackedObject, ...]:
    return tuple(
        item
        for item in frame.objects
        if item.object_type in PLAYER_TYPES and item.pitch_x is not None and item.pitch_y is not None
    )


def _attack_view_xy(
    x: float,
    y: float,
    *,
    left_to_right: bool | None,
) -> tuple[float, float]:
    if left_to_right is False:
        return PITCH_LENGTH - x, y
    return x, y


def _with_attack_view(player: TrackedObject, *, left_to_right: bool | None) -> TrackedObject:
    if player.pitch_x is None or player.pitch_y is None:
        return player
    x, y = _attack_view_xy(player.pitch_x, player.pitch_y, left_to_right=left_to_right)
    if x == player.pitch_x and y == player.pitch_y:
        return player
    return TrackedObject(
        track_id=player.track_id,
        object_type=player.object_type,
        confidence=player.confidence,
        bbox=player.bbox,
        team=player.team,
        pitch_x=x,
        pitch_y=y,
        anchor_image=player.anchor_image,
        keypoints=player.keypoints,
        speed=player.speed,
        speed_unit=player.speed_unit,
    )


def _keypoint_xy(
    keypoints: tuple[tuple[float, ...], ...],
    index: int,
    *,
    threshold: float = 0.2,
) -> tuple[float, float] | None:
    if index >= len(keypoints):
        return None
    point = keypoints[index]
    confidence = point[2] if len(point) > 2 else 1.0
    if confidence < threshold:
        return None
    return float(point[0]), float(point[1])


def infer_body_part_from_pose(
    action: ObservedActionV2,
    shooter: TrackedObject | None,
    frame: FrameObservation,
) -> str | None:
    if action.body_part:
        return action.body_part
    if action.action_type in {"header", "headed_attempt"}:
        return "head"
    if shooter is None or not shooter.keypoints:
        return None
    balls = [item for item in frame.objects if item.object_type == "ball"]
    ball = balls[0] if balls else None
    if ball is None:
        return None
    ball_xy = (
        (ball.bbox.x1 + ball.bbox.x2) / 2.0,
        (ball.bbox.y1 + ball.bbox.y2) / 2.0,
    )
    ranked: list[tuple[float, str]] = []
    for name, index in (
        ("head", COCO_NOSE),
        ("left_foot", COCO_LEFT_ANKLE),
        ("right_foot", COCO_RIGHT_ANKLE),
        ("other", COCO_LEFT_KNEE),
        ("other", COCO_RIGHT_KNEE),
    ):
        point = _keypoint_xy(shooter.keypoints, index)
        if point is None:
            continue
        ranked.append((hypot(point[0] - ball_xy[0], point[1] - ball_xy[1]), name))
    if not ranked:
        return None
    return min(ranked, key=lambda item: item[0])[1]


def play_pattern_from_possession(actions: tuple[ObservedActionV2, ...]) -> str:
    if not actions:
        return "regular_play"
    types = tuple(item.action_type for item in actions)
    terminal = actions[-1]
    if types[0] == "throw_in" or any(
        item.action_type == "throw_in" and terminal.timestamp_sec - item.timestamp_sec <= 12.0
        for item in actions
    ):
        return "throw_in"
    start = actions[0]
    duration = terminal.timestamp_sec - start.timestamp_sec
    if (
        start.x is not None
        and start.x < 40.0
        and duration <= 10.0
        and types.count("pass") <= 3
    ):
        return "counter"
    return "regular_play"


def shot_type_from_possession(actions: tuple[ObservedActionV2, ...]) -> str:
    if not actions:
        return "open_play"
    if play_pattern_from_possession(actions) == "throw_in":
        return "open_play"
    terminal = actions[-1]
    if (
        terminal.x is not None
        and terminal.x >= 99.0
        and terminal.y is not None
        and abs(terminal.y - PITCH_WIDTH / 2.0) <= 4.0
        and terminal.action_type == "shot"
    ):
        return "penalty"
    return "open_play"


def infer_goalkeeper(
    players: tuple[TrackedObject, ...],
    *,
    attacking_team: str | None,
) -> TrackedObject | None:
    labeled = [item for item in players if item.object_type == "goalkeeper"]
    if attacking_team:
        labeled = [item for item in labeled if item.team != attacking_team] or labeled
    if labeled:
        return labeled[0]
    foes = (
        [item for item in players if item.team and item.team != attacking_team]
        if attacking_team
        else list(players)
    )
    box = [
        item
        for item in foes
        if item.pitch_x is not None and item.pitch_x >= PITCH_LENGTH - PENALTY_BOX_DEPTH_M
    ]
    pool = box or foes
    if not pool:
        return None
    return max(pool, key=lambda item: (item.pitch_x or 0.0, item.track_id or ""))


def associate_action_with_frame(
    action: ObservedActionV2,
    frame: FrameObservation,
    *,
    team_mapping: Mapping[str, str] | None = None,
    max_distance_m: float = 3.0,
    min_iou: float = 0.3,
    actor_bbox: BoundingBox | None = None,
    possession_actions: tuple[ObservedActionV2, ...] | None = None,
) -> FusionAssociation:
    """Link one FOOTPASS action to a YOLO track. Never relabels the action fused."""

    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    oriented = tuple(
        _with_attack_view(item, left_to_right=action.raw_left_to_right)
        for item in _pitch_players(frame)
    )
    preferred_team = None
    if action.actor_team_id and team_mapping:
        preferred_team = team_mapping.get(action.actor_team_id)

    scored: list[tuple[float, float, float, TrackedObject]] = []
    for player in oriented:
        distance = None
        if action.x is not None and action.y is not None:
            distance = hypot(player.pitch_x - action.x, player.pitch_y - action.y)
        iou = bbox_iou(actor_bbox, player.bbox) if actor_bbox is not None else 0.0
        team_penalty = (
            0.0
            if preferred_team is None or player.team in {preferred_team, None}
            else 2.0
        )
        if distance is not None and distance <= max_distance_m:
            scored.append((team_penalty, distance, -iou, player))
        elif iou >= min_iou:
            scored.append((team_penalty, max_distance_m + 1.0, -iou, player))
    shooter = None
    method = "unmatched"
    distance_m = None
    iou_value = None
    if scored:
        team_penalty, distance, neg_iou, shooter = min(
            scored,
            key=lambda item: (item[0], item[1], item[2], item[3].track_id or ""),
        )
        method = (
            "bbox_iou"
            if actor_bbox is not None and -neg_iou >= min_iou and distance > max_distance_m
            else "pitch_nearest"
        )
        distance_m = None if distance > max_distance_m else distance
        iou_value = -neg_iou if actor_bbox is not None else None
    attacking_team = shooter.team if shooter is not None and shooter.team else preferred_team
    return FusionAssociation(
        action_id=action.action_id,
        association_id=f"fusion:{action.action_id}:{shooter.track_id if shooter else 'unmatched'}",
        track_id=None if shooter is None else shooter.track_id,
        method=method,
        distance_m=distance_m,
        iou=iou_value,
        xg_raw=_xg_raw_from_frame(
            action,
            frame,
            attacking_team,
            oriented=oriented,
            shooter=shooter,
            possession_actions=possession_actions,
        ),
    )


def _xg_raw_from_frame(
    action: ObservedActionV2,
    frame: FrameObservation,
    attacking_yolo_team: str | None,
    *,
    oriented: tuple[TrackedObject, ...] | None = None,
    shooter: TrackedObject | None = None,
    possession_actions: tuple[ObservedActionV2, ...] | None = None,
) -> dict[str, object]:
    players = oriented if oriented is not None else tuple(
        _with_attack_view(item, left_to_right=action.raw_left_to_right)
        for item in _pitch_players(frame)
    )
    observed = len(players)
    location = (action.x, action.y) if action.x is not None and action.y is not None else None
    keeper = infer_goalkeeper(players, attacking_team=attacking_yolo_team)
    if attacking_yolo_team:
        foes = [item for item in players if item.team and item.team != attacking_yolo_team]
        allies = [item for item in players if item.team == attacking_yolo_team]
    else:
        foes = []
        allies = []
    defending = 0
    if location is not None:
        defending = sum(
            1
            for item in (foes or [player for player in players if player is not shooter])
            if hypot(item.pitch_x - location[0], item.pitch_y - location[1]) <= PRESSURE_RADIUS_M
            and item is not keeper
            and item is not shooter
        )
    actions = possession_actions or (action,)
    ally_count = None
    if attacking_yolo_team is not None:
        ally_count = max(0, len(allies) - (1 if shooter is not None and shooter.team == attacking_yolo_team else 0))
    return {
        "x_shooter": action.x,
        "y_shooter": action.y,
        "body_part": infer_body_part_from_pose(action, shooter, frame),
        "x_goalkeeper": None if keeper is None else keeper.pitch_x,
        "y_goalkeeper": None if keeper is None else keeper.pitch_y,
        "goalkeeper_visible": None if not players else keeper is not None,
        "foe_players": None if attacking_yolo_team is None else len(foes),
        "ally_players": ally_count,
        "defending_players": defending if location is not None else None,
        "under_pressure": None if location is None else int(defending > 0),
        "observed_players": observed if observed else None,
        "play_pattern": play_pattern_from_possession(actions),
        "shot_type": shot_type_from_possession(actions),
    }


def score_attack_with_observations(
    attack: AttackSequenceV2,
    frame: FrameObservation,
    *,
    model: TrainedXGModelV2 | None = None,
    team_mapping: Mapping[str, str] | None = None,
    ball_f1: float | None = None,
    min_ball_f1: float = 0.80,
) -> FusedXGResult:
    """Fill xG-v2 features from YOLO context. FOOTPASS remains the action stream."""

    terminal_action = next(
        (item for item in attack.possession.actions if item.action_id == attack.terminal.action_id),
        attack.possession.actions[-1],
    )
    association = associate_action_with_frame(
        terminal_action,
        frame,
        team_mapping=team_mapping,
        possession_actions=attack.possession.actions,
    )
    if ball_f1 is not None and ball_f1 < min_ball_f1:
        return FusedXGResult(
            association=association,
            xg_record=None,
            withheld_reason="ball_quality_gate",
        )
    if attack.terminal.terminal_type not in {
        TerminalType.SHOT_ATTEMPT,
        TerminalType.HEADED_ATTEMPT,
    }:
        return FusedXGResult(
            association=association,
            xg_record=None,
            withheld_reason="terminal_not_shot",
        )
    if model is None:
        return FusedXGResult(
            association=association,
            xg_record=None,
            withheld_reason="no_xg_v2_model",
        )
    record = model.score_attack(attack, association.xg_raw)
    return FusedXGResult(association=association, xg_record=record)


__all__ = [
    "FusionAssociation",
    "FusedXGResult",
    "associate_action_with_frame",
    "bbox_iou",
    "infer_body_part_from_pose",
    "infer_goalkeeper",
    "play_pattern_from_possession",
    "score_attack_with_observations",
]
