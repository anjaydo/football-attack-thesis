from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping

from .models import AttackSequenceV2, ObservedActionV2, TerminalType
from .observations import BoundingBox, FrameObservation, TrackedObject
from .xg_data import PRESSURE_RADIUS_M
from .xg_v2 import ConfirmedXGRecordV2, TrainedXGModelV2


PLAYER_TYPES = frozenset({"player", "goalkeeper"})


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


def associate_action_with_frame(
    action: ObservedActionV2,
    frame: FrameObservation,
    *,
    team_mapping: Mapping[str, str] | None = None,
    max_distance_m: float = 3.0,
    min_iou: float = 0.3,
    actor_bbox: BoundingBox | None = None,
) -> FusionAssociation:
    """Link one FOOTPASS action to a YOLO track. Never relabels the action fused."""

    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive")
    players = _pitch_players(frame)
    preferred_team = None
    if action.actor_team_id and team_mapping:
        preferred_team = team_mapping.get(action.actor_team_id)

    scored: list[tuple[float, float, TrackedObject]] = []
    for player in players:
        distance = None
        if action.x is not None and action.y is not None:
            distance = hypot(player.pitch_x - action.x, player.pitch_y - action.y)
        iou = bbox_iou(actor_bbox, player.bbox) if actor_bbox is not None else 0.0
        if preferred_team and player.team not in {preferred_team, None}:
            continue
        if distance is not None and distance <= max_distance_m:
            scored.append((distance, -iou, player))
        elif iou >= min_iou:
            scored.append((max_distance_m + 1.0, -iou, player))
    if scored:
        distance, neg_iou, player = min(scored, key=lambda item: (item[0], item[1], item[2].track_id or ""))
        method = "bbox_iou" if actor_bbox is not None and -neg_iou >= min_iou and distance > max_distance_m else "pitch_nearest"
        association_id = f"fusion:{action.action_id}:{player.track_id or 'untracked'}"
        return FusionAssociation(
            action_id=action.action_id,
            association_id=association_id,
            track_id=player.track_id,
            method=method,
            distance_m=None if distance > max_distance_m else distance,
            iou=-neg_iou if actor_bbox is not None else None,
            xg_raw=_xg_raw_from_frame(action, frame, preferred_team),
        )
    return FusionAssociation(
        action_id=action.action_id,
        association_id=f"fusion:{action.action_id}:unmatched",
        track_id=None,
        method="unmatched",
        distance_m=None,
        iou=None,
        xg_raw=_xg_raw_from_frame(action, frame, preferred_team),
    )


def _xg_raw_from_frame(
    action: ObservedActionV2,
    frame: FrameObservation,
    attacking_yolo_team: str | None,
) -> dict[str, object]:
    players = _pitch_players(frame)
    observed = len(players)
    shooter = (action.x, action.y) if action.x is not None and action.y is not None else None
    keepers = [item for item in players if item.object_type == "goalkeeper"]
    if attacking_yolo_team:
        foes = [item for item in players if item.team and item.team != attacking_yolo_team]
        allies = [
            item
            for item in players
            if item.team == attacking_yolo_team and item.object_type == "player"
        ]
        keepers = [item for item in keepers if item.team != attacking_yolo_team] or keepers
    else:
        foes = []
        allies = []
    keeper = keepers[0] if keepers else None
    defending = 0
    if shooter is not None:
        defending = sum(
            1
            for item in (foes or [player for player in players if player.object_type == "player"])
            if hypot(item.pitch_x - shooter[0], item.pitch_y - shooter[1]) <= PRESSURE_RADIUS_M
            and item.object_type != "goalkeeper"
        )
    return {
        "x_shooter": action.x,
        "y_shooter": action.y,
        "body_part": action.body_part,
        "x_goalkeeper": None if keeper is None else keeper.pitch_x,
        "y_goalkeeper": None if keeper is None else keeper.pitch_y,
        "goalkeeper_visible": None if not players else keeper is not None,
        "foe_players": None if attacking_yolo_team is None else len(foes),
        "ally_players": None if attacking_yolo_team is None else max(0, len(allies) - 1),
        "defending_players": defending if shooter is not None else None,
        "under_pressure": None if shooter is None else int(defending > 0),
        "observed_players": observed if observed else None,
        "play_pattern": "unknown",
        "shot_type": "unknown",
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
    "score_attack_with_observations",
]
