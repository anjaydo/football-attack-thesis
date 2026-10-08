from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import hypot, isfinite
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .attack_v2 import AttackV2DecodeResult, FootpassAttackV2Config, decode_footpass_attack_v2
from .footpass_terminal_inference import (
    FootpassTerminalInferenceConfig,
    TerminalInferenceResult,
    infer_footpass_terminals,
)
from .models import (
    ATTACK_V2_SCHEMA_VERSION,
    AttackSequence,
    FieldProvenanceV2,
    Event,
    ObservedActionV2,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SourceTier,
)
from .xg import GOAL_CENTER_Y, TrainedXGModel, heuristic_xg, load_trained_xg_model, shot_angle, shot_distance


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

_FOOTPASS_MATCH_KEY = re.compile(r"^(?P<match>.+)_H(?P<period>[1-9][0-9]*)$")
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
SHOT_CORRIDOR_M = 1.5
_FOOTPASS_SHOT_XG_KEYS = (
    "x_shooter",
    "y_shooter",
    "shot_angle",
    "distance_to_goal",
    "x_goalkeeper",
    "y_goalkeeper",
    "foe_players",
    "ally_players",
    "defending_players",
    "under_pressure",
    "observed_players",
    "is_regular_play",
    "is_counter",
    "is_dead_ball",
    "is_penalty",
    "minute",
    "second",
)


def split_footpass_match_key(match_key: str) -> tuple[str, int]:
    """Split a FOOTPASS half key into the stable match ID and period number."""

    match = _FOOTPASS_MATCH_KEY.fullmatch(match_key.strip())
    if match is None:
        raise ValueError("FOOTPASS match_key must end in _H<period>, for example game_18_H1")
    return match.group("match"), int(match.group("period"))


def normalize_footpass_position(
    raw_x: float,
    raw_y: float,
    direction_flag: int,
) -> tuple[float, float]:
    """Normalize FOOTPASS absolute coordinates so every team attacks toward x=105.

    In the released tactical files, direction flag 0 attacks the right goal and
    direction flag 1 attacks the left goal. The latter must therefore be mirrored.
    """
    x = (1.0 - raw_x) * PITCH_LENGTH if direction_flag == 1 else raw_x * PITCH_LENGTH
    y = raw_y * PITCH_WIDTH
    return min(PITCH_LENGTH, max(0.0, x)), min(PITCH_WIDTH, max(0.0, y))


def possession_relative_position(
    raw_x: float | None,
    raw_y: float | None,
    *,
    actor_team_id: str | None,
    possession_team_id: str | None,
) -> tuple[float | None, float | None]:
    """Keep one possession on a single attacking axis.

    Each FOOTPASS actor is independently mirrored so *that* team attacks
    toward x=105. Opponent actions therefore jump to the opposite half in
    the same possession. Remirror opponent X into the possession team's view.
    """

    if raw_x is None or raw_y is None:
        return None, None
    if not actor_team_id or not possession_team_id or actor_team_id == possession_team_id:
        return raw_x, raw_y
    return min(PITCH_LENGTH, max(0.0, PITCH_LENGTH - raw_x)), raw_y


def actions_in_attack_view(actions: Sequence[ObservedActionV2]) -> tuple[ObservedActionV2, ...]:
    """Return copies whose x/y sit in the possession team's attacking direction."""

    from dataclasses import replace

    remapped: list[ObservedActionV2] = []
    for action in actions:
        x, y = possession_relative_position(
            action.x,
            action.y,
            actor_team_id=action.actor_team_id,
            possession_team_id=action.possession_team_id,
        )
        remapped.append(replace(action, x=x, y=y))
    return tuple(remapped)


def attack_view_median_x(actions: Sequence[ObservedActionV2]) -> float | None:
    xs = [
        possession_relative_position(
            action.x,
            action.y,
            actor_team_id=action.actor_team_id,
            possession_team_id=action.possession_team_id,
        )[0]
        for action in actions
        if action.possession_team_id
        and action.actor_team_id == action.possession_team_id
        and action.x is not None
    ]
    values = [item for item in xs if item is not None]
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


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
    resolved = Path(path).resolve()
    return _cached_tactical_rows(str(resolved), match_key, resolved.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _cached_tactical_rows(path_str: str, match_key: str, mtime_ns: int) -> Any:
    del mtime_ns  # Included so a rewritten HDF5 is not served from cache.
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'footpass' extra to read tactical HDF5 data") from error
    with h5py.File(path_str, "r") as handle:
        if match_key not in handle:
            raise KeyError(f"{match_key} is not present in {path_str}")
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


def _frame_slice(rows: Any, frame_values: Any, frame: int) -> Any:
    import numpy as np

    start = int(np.searchsorted(frame_values, frame, side="left"))
    end = int(np.searchsorted(frame_values, frame, side="right"))
    return rows[start:end]


def _stable_team_ids_by_direction(rows: Any, match_id: str) -> dict[int, str]:
    """Resolve period-local direction flags to roster-backed team IDs.

    FOOTPASS player IDs use a match-stable roster namespace (100-series and
    200-series in the released files), while the 0/1 direction flag flips at
    half-time.  We infer the roster namespace from player IDs instead of
    treating the direction flag as team identity.
    """

    if rows is None or len(rows) == 0:
        return {}
    # Sampling is enough because each broadcast frame normally contains all
    # visible roster entries, and avoids a multi-million-row Counter allocation.
    stride = max(1, len(rows) // 20_000)
    counts: dict[int, Counter[int]] = {0: Counter(), 1: Counter()}
    for row in rows[::stride]:
        player_value = float(row[1])
        direction_value = float(row[2])
        if not isfinite(player_value) or not isfinite(direction_value):
            continue
        player_id = int(player_value)
        direction = int(direction_value)
        if direction not in counts:
            continue
        # The official files allocate players in roster blocks.  Retaining the
        # hundred-series makes substitutions share one stable team identity.
        roster_namespace = (player_id // 100) * 100 if abs(player_id) >= 100 else player_id
        counts[direction][roster_namespace] += 1
    result: dict[int, str] = {}
    for direction, candidates in counts.items():
        if candidates:
            namespace = candidates.most_common(1)[0][0]
            result[direction] = f"{match_id}:roster-{namespace}"
    if len(set(result.values())) != len(result):
        raise ValueError("FOOTPASS tactical rows do not resolve to two distinct stable teams")
    return result


def _resolve_footpass_source_tier(
    source: str,
    tactical_h5: Path | None,
    source_tier: SourceTier | str | None,
) -> SourceTier:
    if source not in {"oracle", "predicted"}:
        raise ValueError("source must be 'oracle' or 'predicted'")
    if source_tier is None:
        if tactical_h5 is not None:
            return (
                SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL
                if source == "oracle"
                else SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL
            )
        if source == "predicted":
            return SourceTier.VIDEO_ONLY
        raise ValueError(
            "oracle FOOTPASS actions without tactical association are not a supported v2 source tier"
        )
    resolved = source_tier if isinstance(source_tier, SourceTier) else SourceTier(str(source_tier))
    if resolved is SourceTier.FUSED:
        raise ValueError("the FOOTPASS adapter cannot claim fused without a real association stage")
    if resolved is SourceTier.LEGACY_UNVALIDATED:
        raise ValueError("legacy_unvalidated is reserved for the V1 compatibility converter")
    if resolved is SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL and (
        source != "oracle" or tactical_h5 is None
    ):
        raise ValueError("oracle_actions_oracle_tactical requires oracle actions and tactical HDF5")
    if resolved is SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL and (
        source != "predicted" or tactical_h5 is None
    ):
        raise ValueError(
            "predicted_actions_oracle_tactical requires predicted actions and tactical HDF5"
        )
    if resolved is SourceTier.VIDEO_ONLY and tactical_h5 is not None:
        raise ValueError("an oracle-tactical association cannot be rendered as video_only")
    return resolved


def _field_provenance(
    field_name: str,
    *,
    source_tier: SourceTier,
    source_record_id: str,
    confidence: float | None,
) -> FieldProvenanceV2:
    return FieldProvenanceV2(
        field_name=field_name,
        source_tier=source_tier,
        source_record_id=source_record_id,
        evidence_ids=(source_record_id,),
        confidence=confidence,
    )


def load_footpass_observed_actions_v2(
    playbyplay_path: Path,
    *,
    match_key: str,
    source: str,
    tactical_h5: Path | None = None,
    fps: float = 25.0,
    min_confidence: float = 0.0,
    model_name: str | None = None,
    class_confidence_thresholds: Mapping[str, float] | None = None,
    source_tier: SourceTier | str | None = None,
) -> tuple[ObservedActionV2, ...]:
    """Adapt instantaneous FOOTPASS records to the explicit Attack-v2 contract.

    No possession or terminal intent is invented here.  In particular, a raw
    ``header`` remains a generic header and never becomes a headed attempt.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    thresholds = dict(class_confidence_thresholds or {})
    if any(name not in FOOTPASS_ACTIONS.values() for name in thresholds):
        raise ValueError("class confidence thresholds contain an unknown action")
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("class confidence thresholds must be between 0 and 1")

    resolved_tier = _resolve_footpass_source_tier(source, tactical_h5, source_tier)
    match_id, period = split_footpass_match_key(match_key)
    groups = _read_playbyplay(playbyplay_path)
    if match_key not in groups:
        raise KeyError(f"{match_key} is not present in {playbyplay_path}")
    rows = _load_tactical_rows(tactical_h5, match_key) if tactical_h5 else None
    frame_values = rows[:, 0].copy() if rows is not None else None
    stable_teams = _stable_team_ids_by_direction(rows, match_id)
    normalized_model = ""
    if model_name:
        normalized_model = "".join(
            character for character in model_name.casefold()
            if character.isalnum() or character == "_"
        )
        if not normalized_model:
            raise ValueError("model_name must contain at least one letter or number")

    actions: list[ObservedActionV2] = []
    for index, raw in enumerate(groups[match_key]):
        if len(raw) < 4:
            raise ValueError(f"event {index} in {match_key} has fewer than four fields")
        frame, direction, jersey, action_code = (
            int(raw[0]),
            int(raw[1]),
            int(raw[2]),
            int(raw[3]),
        )
        if direction not in (0, 1):
            raise ValueError(f"event {index} has an invalid FOOTPASS direction flag")
        if action_code not in FOOTPASS_ACTIONS:
            raise ValueError(f"unknown FOOTPASS action class: {action_code}")
        action_type = FOOTPASS_ACTIONS[action_code]
        confidence = float(raw[4]) if source == "predicted" and len(raw) >= 5 else 1.0
        required_confidence = max(min_confidence, thresholds.get(action_type, 0.0))
        if source == "predicted" and confidence < required_confidence:
            continue

        row = (
            _find_tactical_row(rows, frame_values, frame, direction, jersey, action_code)
            if rows is not None
            else None
        )
        frame_rows = _frame_slice(rows, frame_values, frame) if rows is not None else ()
        evidence_id = (
            f"footpass:{source}:{normalized_model or 'official'}:"
            f"{match_key}:{frame}:{direction}:{jersey}:{action_code}:{index}"
        )
        tactical_record_id = f"footpass-hdf5:{match_key}:{frame}:{direction}:{jersey}"
        actor_id = role = None
        x = y = None
        if row is not None:
            player_id = int(row[1])
            actor_id = f"{match_id}:player-{player_id}"
            role_id = int(row[4])
            role = FOOTPASS_ROLES.get(role_id, f"role_{role_id}")
            raw_x, raw_y = float(row[5]), float(row[6])
            if isfinite(raw_x) and isfinite(raw_y):
                x, y = normalize_footpass_position(raw_x, raw_y, direction)

        actor_team_id = stable_teams.get(direction)
        bbox_visible: bool | None = None
        replay = False
        replay_known = False
        if source == "oracle" and len(raw) >= 6:
            bbox_visible = bool(int(raw[4]))
            replay = bool(int(raw[5]))
            replay_known = True
        elif row is not None:
            bbox_visible = isfinite(float(row[9]))
            if len(frame_rows):
                replay = not any(isfinite(float(item[9])) for item in frame_rows)
                replay_known = True

        qualifiers: list[str] = []
        if not replay_known:
            qualifiers.append("replay_status_unknown")
        beneficiary_team_id = actor_team_id if action_type == "throw_in" else None

        provenance: list[FieldProvenanceV2] = []
        for field_name in ("frame", "timestamp_sec", "action_type", "raw_left_to_right", "jersey"):
            provenance.append(
                _field_provenance(
                    field_name,
                    source_tier=resolved_tier,
                    source_record_id=evidence_id,
                    confidence=confidence,
                )
            )
        if replay_known:
            provenance.append(
                _field_provenance(
                    "replay",
                    source_tier=resolved_tier,
                    source_record_id=evidence_id if source == "oracle" else tactical_record_id,
                    confidence=1.0 if source == "oracle" else None,
                )
            )
        for field_name, value in (
            ("actor_team_id", actor_team_id),
            ("beneficiary_team_id", beneficiary_team_id),
            ("actor_id", actor_id),
            ("role", role),
            ("x", x),
            ("y", y),
            ("bbox_visible", bbox_visible),
        ):
            if value is not None:
                provenance.append(
                    _field_provenance(
                        field_name,
                        source_tier=resolved_tier,
                        source_record_id=tactical_record_id,
                        confidence=None,
                    )
                )

        actions.append(
            ObservedActionV2(
                match_id=match_id,
                period=period,
                frame=frame,
                timestamp_sec=frame / fps,
                action_type=action_type,
                source_tier=resolved_tier,
                actor_team_id=actor_team_id,
                beneficiary_team_id=beneficiary_team_id,
                raw_left_to_right=(direction == 0),
                x=x,
                y=y,
                actor_id=actor_id,
                jersey=jersey,
                role=role,
                bbox_visible=bbox_visible,
                replay=replay,
                qualifiers=tuple(qualifiers),
                action_probability=confidence,
                actor_probability=(1.0 if source == "oracle" and actor_id is not None else None),
                timestamp_probability=(1.0 if source == "oracle" else None),
                field_provenance=tuple(provenance),
                evidence_ids=(evidence_id,),
            )
        )
    return tuple(sorted(actions, key=lambda item: (item.frame, item.action_id)))


def decode_footpass_sequences_v2(
    playbyplay_path: Path,
    *,
    match_key: str,
    source: str,
    tactical_h5: Path | None = None,
    fps: float = 25.0,
    min_confidence: float = 0.0,
    model_name: str | None = None,
    class_confidence_thresholds: Mapping[str, float] | None = None,
    source_tier: SourceTier | str | None = None,
    decoder_config: FootpassAttackV2Config | None = None,
    infer_terminals: bool = True,
) -> AttackV2DecodeResult:
    prepared = prepare_footpass_actions_v2(
        playbyplay_path,
        match_key=match_key,
        source=source,
        tactical_h5=tactical_h5,
        fps=fps,
        min_confidence=min_confidence,
        model_name=model_name,
        class_confidence_thresholds=class_confidence_thresholds,
        source_tier=source_tier,
        decoder_config=decoder_config,
        infer_terminals=infer_terminals,
    )
    settings = decoder_config or FootpassAttackV2Config(fps=fps)
    return decode_footpass_attack_v2(prepared.actions, settings)


def prepare_footpass_actions_v2(
    playbyplay_path: Path,
    *,
    match_key: str,
    source: str,
    tactical_h5: Path | None = None,
    fps: float = 25.0,
    min_confidence: float = 0.0,
    model_name: str | None = None,
    class_confidence_thresholds: Mapping[str, float] | None = None,
    source_tier: SourceTier | str | None = None,
    decoder_config: FootpassAttackV2Config | None = None,
    infer_terminals: bool = True,
) -> TerminalInferenceResult:
    actions = load_footpass_observed_actions_v2(
        playbyplay_path,
        match_key=match_key,
        source=source,
        tactical_h5=tactical_h5,
        fps=fps,
        min_confidence=min_confidence,
        model_name=model_name,
        class_confidence_thresholds=class_confidence_thresholds,
        source_tier=source_tier,
    )
    if not infer_terminals:
        return TerminalInferenceResult(actions=actions)
    settings = decoder_config or FootpassAttackV2Config(fps=fps)
    return infer_footpass_terminals(
        actions,
        FootpassTerminalInferenceConfig(
            fps=settings.fps,
            event_cluster_sec=settings.event_cluster_sec,
            max_observation_gap_sec=settings.max_observation_gap_sec,
        ),
    )


def footpass_v2_summary(
    decoded: AttackV2DecodeResult,
    inference: TerminalInferenceResult | None = None,
) -> dict[str, object]:
    statuses = Counter(item.status.value for item in decoded.possessions)
    terminals = Counter(
        item.terminal.terminal_type.value
        for item in decoded.possessions
        if item.terminal is not None
    )
    summary: dict[str, object] = {
        "schema_version": ATTACK_V2_SCHEMA_VERSION,
        "possession_count": len(decoded.possessions),
        "attack_count": len(decoded.attacks),
        "possession_statuses": dict(sorted(statuses.items())),
        "terminal_types": dict(sorted(terminals.items())),
        "excluded_replay_actions": len(decoded.excluded_replay_action_ids),
        "duplicate_actions": len(decoded.duplicate_action_ids),
        "unassigned_actions": len(decoded.unassigned_action_ids),
        "complete_attack_terminal_invariant": all(
            attack.terminal.qualifies_as_attack for attack in decoded.attacks
        ),
    }
    if inference is not None:
        summary["terminal_inference"] = inference.to_summary_dict()
    return summary


def _distance_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length2 = dx * dx + dy * dy
    if length2 <= 1e-12:
        return hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length2))
    return hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _in_attacking_penalty_box(x: float, y: float) -> bool:
    half = PENALTY_BOX_WIDTH_M / 2.0
    return x >= PITCH_LENGTH - PENALTY_BOX_DEPTH_M and abs(y - GOAL_CENTER_Y) <= half


def footpass_freeze_frame(
    frame_rows: Any,
    *,
    attacking_direction: int,
    shooter_x: float,
    shooter_y: float,
    timestamp: float,
) -> dict[str, object]:
    """Oracle 11v11 snapshot at one tactical frame, in possession-relative metres."""

    from .xg_data import PRESSURE_RADIUS_M

    visible: list[tuple[int, int, float, float]] = []
    for row in frame_rows:
        raw_x, raw_y = float(row[5]), float(row[6])
        if raw_x != raw_x or raw_y != raw_y:
            continue
        x, y = normalize_footpass_position(raw_x, raw_y, attacking_direction)
        visible.append((int(row[2]), int(row[4]), x, y))
    opponents = [item for item in visible if item[0] != attacking_direction]
    allies = [item for item in visible if item[0] == attacking_direction]
    goalkeeper = next((item for item in opponents if item[1] == 1), None)
    nearby_defenders = sum(
        item[1] != 1
        and hypot(item[2] - shooter_x, item[3] - shooter_y) <= PRESSURE_RADIUS_M
        for item in opponents
    )
    shooter_index = None
    best = None
    for index, item in enumerate(allies):
        distance = hypot(item[2] - shooter_x, item[3] - shooter_y)
        if best is None or distance < best:
            best = distance
            shooter_index = index
    point = (shooter_x, shooter_y)
    gk_distance = None
    if goalkeeper is not None:
        gk_distance = hypot(goalkeeper[2] - shooter_x, goalkeeper[3] - shooter_y)
    players: list[dict[str, object]] = []
    for index, item in enumerate(allies):
        players.append(
            {
                "side": "ally",
                "role": FOOTPASS_ROLES.get(item[1], f"role_{item[1]}"),
                "x": item[2],
                "y": item[3],
                "is_goalkeeper": item[1] == 1,
                "is_shooter": index == shooter_index,
            }
        )
    for item in opponents:
        players.append(
            {
                "side": "foe",
                "role": FOOTPASS_ROLES.get(item[1], f"role_{item[1]}"),
                "x": item[2],
                "y": item[3],
                "is_goalkeeper": item[1] == 1,
                "is_shooter": False,
            }
        )
    ally_count = max(0, len(allies) - 1)
    foe_count = len(opponents)
    observed = 1 + ally_count + foe_count if visible else None
    return {
        "x_shooter": shooter_x,
        "y_shooter": shooter_y,
        "shot_angle": shot_angle(point),
        "distance_to_goal": shot_distance(point),
        "x_goalkeeper": goalkeeper[2] if goalkeeper else None,
        "y_goalkeeper": goalkeeper[3] if goalkeeper else None,
        "foe_players": foe_count,
        "ally_players": ally_count,
        "defending_players": nearby_defenders,
        "under_pressure": int(nearby_defenders > 0),
        "observed_players": observed,
        "defenders_in_box": sum(1 for item in opponents if _in_attacking_penalty_box(item[2], item[3])),
        "allies_ahead": sum(1 for item in allies if item[2] > shooter_x + 0.5),
        "foes_in_shot_corridor": sum(
            1
            for item in opponents
            if item[1] != 1
            and _distance_to_segment(
                item[2], item[3], shooter_x, shooter_y, PITCH_LENGTH, GOAL_CENTER_Y
            )
            <= SHOT_CORRIDOR_M
        ),
        "gk_distance_m": gk_distance,
        "is_regular_play": 1,
        "is_counter": 0,
        "is_dead_ball": 0,
        "is_penalty": 0,
        "minute": int(timestamp // 60),
        "second": int(timestamp % 60),
        "players": players,
        "source": "footpass_tactical",
    }


def _footpass_shot_features(
    frame_rows: Any,
    *,
    team: int,
    shooter_x: float,
    shooter_y: float,
    timestamp: float,
) -> dict[str, object]:
    freeze = footpass_freeze_frame(
        frame_rows,
        attacking_direction=team,
        shooter_x=shooter_x,
        shooter_y=shooter_y,
        timestamp=timestamp,
    )
    return {key: freeze[key] for key in _FOOTPASS_SHOT_XG_KEYS if key in freeze}


def load_footpass_freeze_frame(
    tactical_h5: Path,
    *,
    match_key: str,
    frame: int,
    attacking_direction: int,
    shooter_x: float,
    shooter_y: float,
    timestamp: float,
) -> dict[str, object]:
    rows = _load_tactical_rows(tactical_h5, match_key)
    frame_values = rows[:, 0].copy()
    freeze = footpass_freeze_frame(
        _frame_slice(rows, frame_values, frame),
        attacking_direction=attacking_direction,
        shooter_x=shooter_x,
        shooter_y=shooter_y,
        timestamp=timestamp,
    )
    freeze["match_key"] = match_key
    freeze["frame"] = frame
    freeze["evidence_id"] = f"footpass-hdf5:{match_key}:{frame}:freeze"
    return freeze


def load_footpass_events(
    playbyplay_path: Path,
    *,
    match_key: str,
    source: str,
    tactical_h5: Path | None = None,
    fps: float = 25.0,
    min_confidence: float = 0.0,
    model_name: str | None = None,
    xg_model_path: Path | None = None,
    class_confidence_thresholds: Mapping[str, float] | None = None,
) -> tuple[Event, ...]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    thresholds = dict(class_confidence_thresholds or {})
    if any(name not in FOOTPASS_ACTIONS.values() for name in thresholds):
        raise ValueError("class confidence thresholds contain an unknown action")
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("class confidence thresholds must be between 0 and 1")
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
    xg_model: TrainedXGModel | None = (
        load_trained_xg_model(str(xg_model_path.resolve())) if xg_model_path else None
    )

    converted: list[Event] = []
    for index, raw in enumerate(groups[match_key]):
        if len(raw) < 4:
            raise ValueError(f"event {index} in {match_key} has fewer than four fields")
        frame, team, jersey, action = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
        if action not in FOOTPASS_ACTIONS:
            raise ValueError(f"unknown FOOTPASS action class: {action}")
        confidence = float(raw[4]) if source == "predicted" and len(raw) >= 5 else 1.0
        action_name = FOOTPASS_ACTIONS[action]
        required_confidence = max(min_confidence, thresholds.get(action_name, 0.0))
        if source == "predicted" and confidence < required_confidence:
            continue
        row = (
            _find_tactical_row(rows, frame_values, frame, team, jersey, action)
            if rows is not None
            else None
        )
        player_id = role = None
        x = y = None
        frame_rows = rows[0:0] if rows is not None else ()
        if row is not None:
            player_id = str(int(row[1]))
            role = FOOTPASS_ROLES.get(int(row[4]), f"role_{int(row[4])}")
            raw_x, raw_y = float(row[5]), float(row[6])
            if isfinite(raw_x) and isfinite(raw_y):
                x, y = normalize_footpass_position(raw_x, raw_y, team)

        if rows is not None:
            frame_rows = _frame_slice(rows, frame_values, frame)

        event_xg = None
        event_xg_source = None
        event_xg_inputs = None
        if action == 5 and x is not None and y is not None:
            event_xg_inputs = _footpass_shot_features(
                frame_rows,
                team=team,
                shooter_x=x,
                shooter_y=y,
                timestamp=frame / fps,
            )
            if xg_model is not None:
                event_xg = xg_model.predict(event_xg_inputs)
                event_xg_source = xg_model.source
            else:
                event_xg = heuristic_xg((x, y))
                event_xg_source = "heuristic_geometry_v1"
                event_xg_inputs = {
                    key: event_xg_inputs[key]
                    for key in ("x_shooter", "y_shooter", "shot_angle", "distance_to_goal")
                }

        evidence_id = f"event:{match_key}:{frame}:{team}:{jersey}:{action}:{index}"
        converted.append(
            Event(
                timestamp=frame / fps,
                event_type=action_name,
                team=str(team),
                x=x,
                y=y,
                xg=event_xg,
                xg_source=event_xg_source,
                xg_inputs=event_xg_inputs,
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
    """Legacy V1 heuristic retained only for reproducible comparison.

    The returned objects are possession candidates, not validated attacks. New
    code must use :func:`decode_footpass_sequences_v2`.
    """

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
        "schema_version": "PossessionCandidateV1",
        "validation_status": "legacy_unvalidated",
        "event_count": len(event_items),
        "possession_candidate_count": len(attack_items),
        "actions": dict(Counter(item.event_type for item in event_items)),
        "candidates_with_shot": sum(
            any(event.event_type == "shot" for event in item.events) for item in attack_items
        ),
        "candidates_without_shot": sum(
            not any(event.event_type == "shot" for event in item.events) for item in attack_items
        ),
        "warning": "V1 heuristic segments are not AttackSequenceV2 and must not be scored as attacks",
    }
