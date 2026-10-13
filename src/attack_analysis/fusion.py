from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import hypot
from typing import Mapping, Sequence

from .footpass import SHOT_CORRIDOR_M, _distance_to_segment, _in_attacking_penalty_box, actions_in_attack_view
from .models import PITCH_LENGTH, PITCH_WIDTH, AttackSequenceV2, ObservedActionV2, TerminalType
from .observations import BoundingBox, FrameObservation, TrackedObject
from .xg import GOAL_CENTER_Y, shot_angle, shot_distance
from .xg_data import PRESSURE_RADIUS_M
from .xg_v2 import ConfirmedXGRecordV2, TrainedXGModelV2


PLAYER_TYPES = frozenset({"player", "goalkeeper"})
COCO_NOSE = 0
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
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
    spatial_context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action_id": self.action_id,
            "association_id": self.association_id,
            "track_id": self.track_id,
            "method": self.method,
            "distance_m": self.distance_m,
            "iou": self.iou,
            "xg_raw": dict(self.xg_raw),
        }
        if self.spatial_context:
            payload["spatial_context"] = dict(self.spatial_context)
        return payload


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
    keypoints = player.keypoints
    if left_to_right is False:
        flipped: list[tuple[float, ...]] = []
        for point in keypoints:
            if len(point) >= 5:
                flipped.append(
                    (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]) if len(point) > 2 else 1.0,
                        PITCH_LENGTH - float(point[3]),
                        float(point[4]),
                    )
                )
            else:
                flipped.append(point)
        keypoints = tuple(flipped)
    elif x == player.pitch_x and y == player.pitch_y:
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
        keypoints=keypoints,
        speed=player.speed,
        speed_unit=player.speed_unit,
    )


def _attack_view_action(action: ObservedActionV2) -> ObservedActionV2:
    return actions_in_attack_view((action,))[0]


def _orient_players(
    action: ObservedActionV2,
    players: Sequence[TrackedObject],
) -> tuple[TrackedObject, ...]:
    """Mirror YOLO pitch coords into the same attack-view as the FOOTPASS action."""

    located = tuple(
        item for item in players if item.pitch_x is not None and item.pitch_y is not None
    )
    if action.raw_left_to_right is not None:
        return tuple(
            _with_attack_view(item, left_to_right=action.raw_left_to_right) for item in located
        )
    viewed = _attack_view_action(action)
    if viewed.x is None or viewed.y is None or not located:
        return located
    mirrored = tuple(_with_attack_view(item, left_to_right=False) for item in located)

    def nearest(group: tuple[TrackedObject, ...]) -> float:
        return min(
            hypot(item.pitch_x - viewed.x, item.pitch_y - viewed.y) for item in group
        )

    return mirrored if nearest(mirrored) + 0.25 < nearest(located) else located


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
    feet = ((shooter.bbox.x1 + shooter.bbox.x2) / 2.0, shooter.bbox.y2)
    height = max(1.0, shooter.bbox.y2 - shooter.bbox.y1)
    # Frames carry several false balls; only one near his body can be his.
    balls = [
        item
        for item in frame.objects
        if item.object_type == "ball"
        and hypot(
            (item.bbox.x1 + item.bbox.x2) / 2.0 - feet[0],
            (item.bbox.y1 + item.bbox.y2) / 2.0 - feet[1],
        )
        <= 1.3 * height
    ]
    ball = (
        min(
            balls,
            key=lambda item: hypot(
                (item.bbox.x1 + item.bbox.x2) / 2.0 - feet[0],
                (item.bbox.y1 + item.bbox.y2) / 2.0 - feet[1],
            ),
        )
        if balls
        else None
    )
    if ball is not None:
        ball_xy = (
            (ball.bbox.x1 + ball.bbox.x2) / 2.0,
            (ball.bbox.y1 + ball.bbox.y2) / 2.0,
        )
    else:
        # YOLO often misses the ball; the bbox floor is a last-touch proxy.
        ball_xy = (
            (shooter.bbox.x1 + shooter.bbox.x2) / 2.0,
            shooter.bbox.y2,
        )
    shoulders = [
        point[1]
        for point in (
            _keypoint_xy(shooter.keypoints, COCO_LEFT_SHOULDER),
            _keypoint_xy(shooter.keypoints, COCO_RIGHT_SHOULDER),
        )
        if point is not None
    ]
    # A ball bouncing at hip height sits nearer the nose than the ankles in 2D;
    # a header needs the ball above the shoulders. A knee-high ball is volleyed
    # with that leg's foot.
    head_allowed = bool(shoulders) and ball_xy[1] < min(shoulders)
    ranked: list[tuple[float, str]] = []
    for name, index in (
        ("head", COCO_NOSE),
        ("left_foot", COCO_LEFT_ANKLE),
        ("right_foot", COCO_RIGHT_ANKLE),
        ("left_foot", COCO_LEFT_KNEE),
        ("right_foot", COCO_RIGHT_KNEE),
    ):
        if name == "head" and not head_allowed:
            continue
        point = _keypoint_xy(shooter.keypoints, index)
        if point is None:
            continue
        ranked.append((hypot(point[0] - ball_xy[0], point[1] - ball_xy[1]), name))
    if not ranked:
        return None
    return min(ranked, key=lambda item: item[0])[1]


_BUILDUP_TYPES = frozenset({"pass", "cross", "throw_in", "drive", "header"})
_CORNER_FLAG_X = 99.0
_CORNER_FLAG_Y = 3.5
_PENALTY_BOX_LINE_X = PITCH_LENGTH - PENALTY_BOX_DEPTH_M
_STANDING_HEIGHT_M = 1.8


def _action_snapshot(action: ObservedActionV2, terminal: ObservedActionV2) -> dict[str, object]:
    return {
        "action_type": action.action_type,
        "x": action.x,
        "y": action.y,
        "dt_sec": round(terminal.timestamp_sec - action.timestamp_sec, 3),
        "jersey": action.jersey,
        "action_id": action.action_id,
        "frame": action.frame,
    }


def possession_buildup(actions: tuple[ObservedActionV2, ...]) -> dict[str, object]:
    """Summarise the last pass/cross/throw-in before the terminal, in attack view."""

    if not actions:
        return {
            "assist_type": None,
            "assist": None,
            "last_pass": None,
            "last_cross": None,
            "last_throw_in": None,
            "cross_then_header": False,
            "events": [],
        }
    viewed = actions_in_attack_view(actions)
    terminal = viewed[-1]
    prior = [item for item in viewed[:-1] if item.action_type in _BUILDUP_TYPES]

    def last_of(kind: str) -> dict[str, object] | None:
        matches = [item for item in prior if item.action_type == kind]
        return None if not matches else _action_snapshot(matches[-1], terminal)

    last_pass = last_of("pass")
    last_cross = last_of("cross")
    last_throw_in = last_of("throw_in")
    assists = [item for item in (last_cross, last_pass, last_throw_in) if item is not None]
    assist = min(assists, key=lambda item: float(item["dt_sec"])) if assists else None
    events = [
        {
            "action_type": item.action_type,
            "x": item.x,
            "y": item.y,
            "jersey": item.jersey,
            "is_terminal": item.action_id == terminal.action_id,
            "action_id": item.action_id,
        }
        for item in viewed
        if item.x is not None and item.y is not None
    ]
    headed = terminal.action_type in {"header", "headed_attempt"}
    return {
        "assist_type": None if assist is None else assist["action_type"],
        "assist": assist,
        "last_pass": last_pass,
        "last_cross": last_cross,
        "last_throw_in": last_throw_in,
        "cross_then_header": bool(
            headed and last_cross is not None and float(last_cross["dt_sec"]) <= 8.0
        ),
        "events": events,
    }


def play_pattern_evidence(actions: tuple[ObservedActionV2, ...]) -> dict[str, str]:
    """Infer StatsBomb-like play_pattern. FOOTPASS has no corner/FK class."""

    if not actions:
        return {
            "label": "regular_play",
            "source": "event_sequence",
            "reason": "empty possession",
        }
    viewed = actions_in_attack_view(actions)
    types = tuple(item.action_type for item in viewed)
    terminal = viewed[-1]
    start = viewed[0]
    if types[0] == "throw_in" or any(
        item.action_type == "throw_in" and terminal.timestamp_sec - item.timestamp_sec <= 12.0
        for item in viewed
    ):
        return {
            "label": "throw_in",
            "source": "event_sequence",
            "reason": "throw_in action within 12s of the terminal",
        }
    for item in reversed(viewed[:-1]):
        if terminal.timestamp_sec - item.timestamp_sec > 8.0:
            break
        if item.x is None or item.y is None:
            continue
        in_corner = item.x >= _CORNER_FLAG_X and (
            item.y <= _CORNER_FLAG_Y or item.y >= PITCH_WIDTH - _CORNER_FLAG_Y
        )
        if in_corner and item.action_type in {"cross", "pass", "header"}:
            return {
                "label": "corner",
                "source": "geometry_heuristic",
                "reason": "last delivery from an attacking corner flag",
            }
    if (
        terminal.action_type in {"shot", "header"}
        and terminal.x is not None
        and terminal.x < _PENALTY_BOX_LINE_X
        and len(viewed) <= 3
        and start.x is not None
        and start.y is not None
        and terminal.y is not None
        and hypot(start.x - terminal.x, start.y - terminal.y) <= 6.0
        and terminal.timestamp_sec - start.timestamp_sec <= 6.0
    ):
        return {
            "label": "free_kick",
            "source": "geometry_heuristic",
            "reason": "short sequence from a similar location outside the box",
        }
    duration = terminal.timestamp_sec - start.timestamp_sec
    if (
        start.x is not None
        and start.x < 40.0
        and duration <= 10.0
        and types.count("pass") <= 3
    ):
        return {
            "label": "counter",
            "source": "event_sequence",
            "reason": "started in own half with few passes",
        }
    return {
        "label": "regular_play",
        "source": "event_sequence",
        "reason": "default open play",
    }


def play_pattern_from_possession(actions: tuple[ObservedActionV2, ...]) -> str:
    return play_pattern_evidence(actions)["label"]


def shot_type_evidence(actions: tuple[ObservedActionV2, ...]) -> dict[str, str]:
    if not actions:
        return {
            "label": "open_play",
            "source": "event_sequence",
            "reason": "empty possession",
        }
    viewed = actions_in_attack_view(actions)
    terminal = viewed[-1]
    pattern = play_pattern_evidence(actions)
    if pattern["label"] == "throw_in":
        return {
            "label": "open_play",
            "source": "event_sequence",
            "reason": "throw-in is scored as open play",
        }
    if (
        terminal.action_type == "shot"
        and terminal.x is not None
        and terminal.x >= 99.0
        and terminal.y is not None
        and abs(terminal.y - PITCH_WIDTH / 2.0) <= 4.0
    ):
        return {
            "label": "penalty",
            "source": "geometry_heuristic",
            "reason": "shot from the penalty spot",
        }
    if pattern["label"] == "free_kick":
        return {
            "label": "free_kick",
            "source": "geometry_heuristic",
            "reason": pattern["reason"],
        }
    return {
        "label": "open_play",
        "source": "event_sequence",
        "reason": "default open play",
    }


def shot_type_from_possession(actions: tuple[ObservedActionV2, ...]) -> str:
    return shot_type_evidence(actions)["label"]


def infer_goalkeeper(
    players: tuple[TrackedObject, ...],
    *,
    attacking_team: str | None,
    shooter: TrackedObject | None = None,
) -> TrackedObject | None:
    # YOLO sometimes labels the shooter "goalkeeper"; he cannot keep his own shot.
    if shooter is not None:
        players = tuple(
            item
            for item in players
            if item is not shooter
            and (shooter.track_id is None or item.track_id != shooter.track_id)
        )
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


_FOOTPASS_COUNT_KEYS = (
    "x_goalkeeper",
    "y_goalkeeper",
    "foe_players",
    "ally_players",
    "defending_players",
    "under_pressure",
    "observed_players",
    "shot_angle",
    "distance_to_goal",
)


def vote_team_mapping(
    action: ObservedActionV2,
    frame: FrameObservation,
    *,
    max_distance_m: float = 3.0,
    opposing_team_id: str | None = None,
) -> dict[str, str] | None:
    """Map FOOTPASS team ids to YOLO clusters using tracks near the actor."""

    if not action.actor_team_id:
        return None
    viewed = _attack_view_action(action)
    if viewed.x is None or viewed.y is None:
        return None
    players = _orient_players(action, _pitch_players(frame))
    players = tuple(item for item in players if item.team)
    if not players:
        return None
    nearby = [
        item
        for item in players
        if hypot(item.pitch_x - viewed.x, item.pitch_y - viewed.y) <= max_distance_m
    ]
    pool = nearby or sorted(
        players,
        key=lambda item: hypot(item.pitch_x - viewed.x, item.pitch_y - viewed.y),
    )[:1]
    counts: dict[str, int] = {}
    for item in pool:
        counts[item.team] = counts.get(item.team, 0) + 1
    attacking = max(counts, key=lambda team: (counts[team], team))
    mapping = {action.actor_team_id: attacking}
    other_yolo = next((item.team for item in players if item.team != attacking), None)
    if opposing_team_id and other_yolo:
        mapping[opposing_team_id] = other_yolo
    return mapping


def project_pose_to_pitch(shooter: TrackedObject) -> list[dict[str, object]]:
    """Attach pitch metres to COCO keypoints.

    Homography-projected keypoints (5-tuples from the vision pipeline) are
    preferred. Otherwise a local similarity around the ground-plane anchor is
    used: image x offset → pitch y, small image-y offset → pitch x. Upper-body
    points are off-plane; this is a visualisation / nearest-foot helper, not a
    true per-keypoint homography.
    """

    keypoints: list[dict[str, object]] = []
    if shooter.anchor_image:
        anchor_x, anchor_y = shooter.anchor_image
    else:
        anchor_x = (shooter.bbox.x1 + shooter.bbox.x2) / 2.0
        anchor_y = shooter.bbox.y2
    height_px = max(shooter.bbox.y2 - shooter.bbox.y1, 1.0)
    metres_per_px = _STANDING_HEIGHT_M / height_px
    for index, point in enumerate(shooter.keypoints):
        confidence = point[2] if len(point) > 2 else 1.0
        if confidence < 0.2:
            continue
        item: dict[str, object] = {
            "index": index,
            "x": float(point[0]),
            "y": float(point[1]),
            "confidence": float(confidence),
        }
        if len(point) >= 5:
            item["pitch_x"] = float(point[3])
            item["pitch_y"] = float(point[4])
            item["pitch_source"] = "homography"
        elif shooter.pitch_x is not None and shooter.pitch_y is not None:
            dx = float(point[0]) - anchor_x
            dy = float(point[1]) - anchor_y
            item["pitch_x"] = max(
                0.0,
                min(PITCH_LENGTH, shooter.pitch_x - dy * metres_per_px * 0.15),
            )
            item["pitch_y"] = max(
                0.0,
                min(PITCH_WIDTH, shooter.pitch_y + dx * metres_per_px),
            )
            item["pitch_source"] = "local_affine"
        keypoints.append(item)
    return keypoints


def align_yolo_tracks_to_freeze(
    yolo_players: Sequence[TrackedObject],
    freeze_players: Sequence[Mapping[str, object]],
    *,
    max_distance_m: float = 4.0,
) -> dict[str, object]:
    """Nearest-neighbour map from YOLO pitch tracks onto FOOTPASS freeze-frame roles."""

    freeze_located = [
        item
        for item in freeze_players
        if item.get("x") is not None and item.get("y") is not None
    ]
    assignments: list[dict[str, object]] = []
    used: set[int] = set()
    for player in yolo_players:
        if player.pitch_x is None or player.pitch_y is None:
            continue
        best_index = None
        best_distance = max_distance_m
        for index, freeze in enumerate(freeze_located):
            if index in used:
                continue
            distance = hypot(
                player.pitch_x - float(freeze["x"]),
                player.pitch_y - float(freeze["y"]),
            )
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            continue
        used.add(best_index)
        freeze = freeze_located[best_index]
        assignments.append(
            {
                "track_id": player.track_id,
                "yolo_team": player.team,
                "freeze_side": freeze.get("side"),
                "freeze_role": freeze.get("role"),
                "is_goalkeeper": bool(freeze.get("is_goalkeeper")),
                "distance_m": round(best_distance, 3),
            }
        )
    ally_teams = [
        item["yolo_team"]
        for item in assignments
        if item.get("freeze_side") == "ally" and item.get("yolo_team")
    ]
    foe_teams = [
        item["yolo_team"]
        for item in assignments
        if item.get("freeze_side") == "foe" and item.get("yolo_team")
    ]
    labelled = len(ally_teams) + len(foe_teams)
    consistent = 0
    if ally_teams:
        ally_mode = max(set(ally_teams), key=ally_teams.count)
        consistent += sum(1 for team in ally_teams if team == ally_mode)
        consistent += sum(1 for team in foe_teams if team != ally_mode)
    elif foe_teams:
        foe_mode = max(set(foe_teams), key=foe_teams.count)
        consistent += sum(1 for team in foe_teams if team == foe_mode)
    agreement = None if not labelled else round(consistent / labelled, 3)
    freeze_gk = next(
        (
            item
            for item in freeze_located
            if item.get("is_goalkeeper") and item.get("side") == "foe"
        ),
        None,
    )
    gk_in_camera = None
    if freeze_gk is not None:
        gk_in_camera = any(
            hypot(player.pitch_x - float(freeze_gk["x"]), player.pitch_y - float(freeze_gk["y"]))
            <= max_distance_m
            for player in yolo_players
            if player.pitch_x is not None and player.pitch_y is not None
        )
    return {
        "n_matched": len(assignments),
        "agreement": agreement,
        "goalkeeper_in_camera": gk_in_camera,
        "assignments": assignments,
    }


def _yolo_spatial_extras(
    action: ObservedActionV2,
    players: Sequence[TrackedObject],
    *,
    attacking_team: str | None,
    shooter: TrackedObject | None,
    keeper: TrackedObject | None,
) -> dict[str, object]:
    viewed = _attack_view_action(action)
    location = (viewed.x, viewed.y) if viewed.x is not None and viewed.y is not None else None
    if attacking_team:
        foes = [item for item in players if item.team and item.team != attacking_team]
        allies = [item for item in players if item.team == attacking_team]
    else:
        foes = []
        allies = []
    extras: dict[str, object] = {
        "defenders_in_box": None,
        "allies_ahead": None,
        "foes_in_shot_corridor": None,
        "gk_distance_m": None,
        "goalkeeper_in_camera": keeper is not None if players else None,
        "ball_detected": False,
    }
    if shooter is not None:
        extras["yolo_shooter"] = {
            "x": shooter.pitch_x,
            "y": shooter.pitch_y,
            "track_id": shooter.track_id,
        }
        pose = project_pose_to_pitch(shooter)
        extras["shooter_keypoint_count"] = len(pose)
        extras["shooter_keypoints"] = pose
        extras["shooter_bbox"] = {
            "x1": shooter.bbox.x1,
            "y1": shooter.bbox.y1,
            "x2": shooter.bbox.x2,
            "y2": shooter.bbox.y2,
        }
    if location is not None:
        extras["defenders_in_box"] = sum(
            1 for item in foes if _in_attacking_penalty_box(item.pitch_x, item.pitch_y)
        )
        extras["allies_ahead"] = sum(
            1
            for item in allies
            if item is not shooter and item.pitch_x > location[0] + 0.5
        )
        extras["foes_in_shot_corridor"] = sum(
            1
            for item in foes
            if item is not keeper
            and _distance_to_segment(
                item.pitch_x,
                item.pitch_y,
                location[0],
                location[1],
                PITCH_LENGTH,
                GOAL_CENTER_Y,
            )
            <= SHOT_CORRIDOR_M
        )
        if keeper is not None and keeper.pitch_x is not None and keeper.pitch_y is not None:
            extras["gk_distance_m"] = hypot(
                keeper.pitch_x - location[0], keeper.pitch_y - location[1]
            )
    return extras


def merge_fusion_context(
    yolo_raw: Mapping[str, object],
    *,
    footpass_freeze: Mapping[str, object] | None = None,
    yolo_extras: Mapping[str, object] | None = None,
    frame: FrameObservation | None = None,
    shooter: TrackedObject | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Prefer FOOTPASS tactical counts/GK; keep YOLO body_part and camera checks."""

    provenance: dict[str, str] = {}
    xg_raw = dict(yolo_raw)
    for key, value in xg_raw.items():
        if value is not None:
            provenance[key] = "yolo"
    extras = dict(yolo_extras or {})
    extras["play_pattern_source"] = extras.get("play_pattern_source") or xg_raw.pop("play_pattern_source", None)
    extras["shot_type_source"] = extras.get("shot_type_source") or xg_raw.pop("shot_type_source", None)
    freeze = dict(footpass_freeze or {})
    if freeze:
        for key in _FOOTPASS_COUNT_KEYS:
            if freeze.get(key) is not None:
                xg_raw[key] = freeze[key]
                provenance[key] = "footpass_tactical"
        if freeze.get("x_shooter") is not None:
            xg_raw["x_shooter"] = freeze["x_shooter"]
            provenance["x_shooter"] = "footpass_tactical"
        if freeze.get("y_shooter") is not None:
            xg_raw["y_shooter"] = freeze["y_shooter"]
            provenance["y_shooter"] = "footpass_tactical"
        if freeze.get("x_goalkeeper") is not None:
            xg_raw["goalkeeper_visible"] = True
            provenance["goalkeeper_visible"] = "footpass_tactical"
        for key in (
            "defenders_in_box",
            "allies_ahead",
            "foes_in_shot_corridor",
            "gk_distance_m",
        ):
            if freeze.get(key) is not None:
                extras[key] = freeze[key]
                provenance[key] = "footpass_tactical"
        if freeze.get("players"):
            extras["players"] = list(freeze["players"])
            provenance["players"] = "footpass_tactical"
        extras["footpass_evidence_id"] = freeze.get("evidence_id")
    xg_raw.setdefault("play_pattern", "regular_play")
    provenance.setdefault("play_pattern", str(extras.get("play_pattern_source") or "event_sequence"))
    xg_raw.setdefault("shot_type", "open_play")
    provenance.setdefault("shot_type", str(extras.get("shot_type_source") or "event_sequence"))
    if xg_raw.get("body_part"):
        provenance["body_part"] = (
            "event" if shooter is None or not shooter.keypoints else "yolo_pose"
        )
    location = (
        (float(xg_raw["x_shooter"]), float(xg_raw["y_shooter"]))
        if xg_raw.get("x_shooter") is not None and xg_raw.get("y_shooter") is not None
        else None
    )
    if location is not None:
        xg_raw.setdefault("shot_angle", shot_angle(location))
        xg_raw.setdefault("distance_to_goal", shot_distance(location))
        provenance.setdefault("shot_angle", provenance.get("x_shooter", "event"))
        provenance.setdefault("distance_to_goal", provenance.get("x_shooter", "event"))
    extras.setdefault(
        "ball_detected",
        bool(frame is not None and any(item.object_type == "ball" for item in frame.objects)),
    )
    extras.setdefault(
        "calibration_confidence",
        None if frame is None else frame.calibration_confidence,
    )
    extras.setdefault(
        "yolo_player_count",
        None if frame is None else len(_pitch_players(frame)),
    )
    extras["ontology_note"] = (
        "FOOTPASS events are drive/pass/cross/throw_in/shot/header/tackle/block; "
        "no corner, free kick, goal kick, or kick-off class. "
        "corner/free_kick/penalty labels are geometry heuristics, not FOOTPASS labels."
    )
    if extras.get("play_pattern_source"):
        provenance["play_pattern"] = str(extras["play_pattern_source"])
    if extras.get("shot_type_source"):
        provenance["shot_type"] = str(extras["shot_type_source"])
    extras["provenance"] = provenance
    return xg_raw, extras


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
    viewed = _attack_view_action(action)
    oriented = _orient_players(action, _pitch_players(frame))
    preferred_team = None
    if action.actor_team_id and team_mapping:
        preferred_team = team_mapping.get(action.actor_team_id)

    scored: list[tuple[float, float, float, TrackedObject]] = []
    for player in oriented:
        distance = None
        if viewed.x is not None and viewed.y is not None:
            distance = hypot(player.pitch_x - viewed.x, player.pitch_y - viewed.y)
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
    players = oriented if oriented is not None else _orient_players(action, _pitch_players(frame))
    observed = len(players)
    viewed = _attack_view_action(action)
    location = (viewed.x, viewed.y) if viewed.x is not None and viewed.y is not None else None
    keeper = infer_goalkeeper(players, attacking_team=attacking_yolo_team, shooter=shooter)
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
    pattern = play_pattern_evidence(actions)
    shot = shot_type_evidence(actions)
    ally_count = None
    if attacking_yolo_team is not None:
        ally_count = max(0, len(allies) - (1 if shooter is not None and shooter.team == attacking_yolo_team else 0))
    return {
        "x_shooter": viewed.x,
        "y_shooter": viewed.y,
        "body_part": infer_body_part_from_pose(action, shooter, frame),
        "x_goalkeeper": None if keeper is None else keeper.pitch_x,
        "y_goalkeeper": None if keeper is None else keeper.pitch_y,
        "goalkeeper_visible": None if not players else keeper is not None,
        "foe_players": None if attacking_yolo_team is None else len(foes),
        "ally_players": ally_count,
        "defending_players": defending if location is not None else None,
        "under_pressure": None if location is None else int(defending > 0),
        "observed_players": observed if observed else None,
        "play_pattern": pattern["label"],
        "shot_type": shot["label"],
        "play_pattern_source": pattern["source"],
        "shot_type_source": shot["source"],
    }


def score_attack_with_observations(
    attack: AttackSequenceV2,
    frame: FrameObservation | None,
    *,
    model: TrainedXGModelV2 | None = None,
    team_mapping: Mapping[str, str] | None = None,
    ball_f1: float | None = None,
    min_ball_f1: float = 0.80,
    footpass_freeze: Mapping[str, object] | None = None,
) -> FusedXGResult:
    """Fill xG-v2 features from FOOTPASS freeze-frame, with YOLO as overlay evidence."""

    terminal_action = next(
        (item for item in attack.possession.actions if item.action_id == attack.terminal.action_id),
        attack.possession.actions[-1],
    )
    opposing = next(
        (
            item.actor_team_id
            for item in attack.possession.actions
            if item.actor_team_id and item.actor_team_id != terminal_action.actor_team_id
        ),
        None,
    )
    mapping = dict(team_mapping) if team_mapping else {}
    shooter = None
    extras: dict[str, object] = {}
    oriented: tuple[TrackedObject, ...] = ()
    viewed_terminal = _attack_view_action(terminal_action)
    pattern = play_pattern_evidence(attack.possession.actions)
    shot = shot_type_evidence(attack.possession.actions)
    extras["play_pattern_source"] = pattern["source"]
    extras["shot_type_source"] = shot["source"]
    extras["play_pattern_reason"] = pattern["reason"]
    extras["shot_type_reason"] = shot["reason"]
    if frame is not None:
        voted = vote_team_mapping(
            terminal_action,
            frame,
            opposing_team_id=opposing,
        )
        if voted:
            mapping = voted
        association = associate_action_with_frame(
            terminal_action,
            frame,
            team_mapping=mapping or None,
            possession_actions=attack.possession.actions,
        )
        oriented = _orient_players(terminal_action, _pitch_players(frame))
        shooter = next(
            (item for item in oriented if item.track_id and item.track_id == association.track_id),
            None,
        )
        attacking = (
            shooter.team
            if shooter is not None and shooter.team
            else mapping.get(terminal_action.actor_team_id or "")
        )
        keeper = infer_goalkeeper(oriented, attacking_team=attacking, shooter=shooter)
        extras.update(
            _yolo_spatial_extras(
                terminal_action,
                oriented,
                attacking_team=attacking,
                shooter=shooter,
                keeper=keeper,
            )
        )
        extras["ball_detected"] = any(item.object_type == "ball" for item in frame.objects)
        extras["play_pattern_source"] = pattern["source"]
        extras["shot_type_source"] = shot["source"]
    else:
        location_method = (
            "footpass_tactical"
            if footpass_freeze
            else "event_location"
            if viewed_terminal.x is not None
            else "unmatched"
        )
        association = FusionAssociation(
            action_id=terminal_action.action_id,
            association_id=f"fusion:{terminal_action.action_id}:footpass",
            track_id=None,
            method=location_method,
            distance_m=None,
            iou=None,
            xg_raw={
                "x_shooter": viewed_terminal.x,
                "y_shooter": viewed_terminal.y,
                "body_part": terminal_action.body_part
                or (
                    "head"
                    if terminal_action.action_type in {"header", "headed_attempt"}
                    else None
                ),
                "play_pattern": pattern["label"],
                "shot_type": shot["label"],
            },
        )
    xg_raw, spatial = merge_fusion_context(
        association.xg_raw,
        footpass_freeze=footpass_freeze,
        yolo_extras=extras,
        frame=frame,
        shooter=shooter,
    )
    buildup = possession_buildup(attack.possession.actions)
    last_cross = buildup.get("last_cross")
    if (
        attack.terminal.terminal_type is TerminalType.HEADED_ATTEMPT
        and isinstance(last_cross, dict)
        and float(last_cross.get("dt_sec") or 99.0) <= 8.0
    ):
        buildup["cross_then_header"] = True
    spatial["possession_buildup"] = buildup
    spatial.setdefault("provenance", {})
    spatial["provenance"]["possession_buildup"] = "event_sequence"
    if footpass_freeze and footpass_freeze.get("players") and oriented:
        aligned = align_yolo_tracks_to_freeze(
            oriented,
            list(footpass_freeze["players"]),  # type: ignore[arg-type]
        )
        spatial["yolo_role_align"] = aligned
        spatial["provenance"]["yolo_role_align"] = "yolo+footpass_tactical"
        if aligned.get("goalkeeper_in_camera") is not None:
            spatial["goalkeeper_in_camera"] = aligned["goalkeeper_in_camera"]
            spatial["provenance"]["goalkeeper_in_camera"] = "yolo+footpass_tactical"
    freeze_ready = bool(footpass_freeze and footpass_freeze.get("foe_players") is not None)
    if ball_f1 is not None and ball_f1 < min_ball_f1 and not freeze_ready:
        spatial["ball_quality_warning"] = f"ball F1 {ball_f1:.2f} < {min_ball_f1:.2f}"
        spatial["provenance"]["ball_quality_warning"] = "yolo"
    association = replace(association, xg_raw=xg_raw, spatial_context=spatial)
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
    "align_yolo_tracks_to_freeze",
    "associate_action_with_frame",
    "bbox_iou",
    "infer_body_part_from_pose",
    "infer_goalkeeper",
    "merge_fusion_context",
    "play_pattern_evidence",
    "play_pattern_from_possession",
    "possession_buildup",
    "project_pose_to_pitch",
    "score_attack_with_observations",
    "shot_type_from_possession",
    "vote_team_mapping",
]
