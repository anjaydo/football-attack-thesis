from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .attack_features import build_attack_features
from .footpass import load_footpass_freeze_frame, possession_relative_position
from .fusion import FusedXGResult, score_attack_with_observations
from .models import PITCH_LENGTH, PITCH_WIDTH, AttackSequence, AttackSequenceV2, Event, TerminalType
from .observations import FrameObservation
from .quality import read_observations_jsonl
from .research import BenchmarkRecord, EvidenceBundle
from .xg_v2 import ConfirmedXGRecordV2, TrainedXGModelV2, load_trained_xg_v2_model


XG_V2_DEFAULT_RELATIVE = Path("models") / "xg-v2.joblib"
PITCH_CENTRE_Y = PITCH_WIDTH / 2.0
PENALTY_BOX_X = PITCH_LENGTH - 16.5
PENALTY_BOX_HALF_WIDTH = 20.16
CENTRAL_HALF_WIDTH = 9.16
FINAL_THIRD_X = PITCH_LENGTH * 2.0 / 3.0
MIDDLE_THIRD_X = PITCH_LENGTH / 3.0


@dataclass(frozen=True)
class AttackEvidenceBundle:
    """Unified FOOTPASS + YOLO + xG-v2 evidence for research-only conclusions."""

    schema_version: str
    attack: AttackSequenceV2
    fusion: FusedXGResult | None
    xg_record: ConfirmedXGRecordV2 | None
    withheld_reason: str | None
    quality_gates: dict[str, bool]
    quality_metrics: dict[str, float]
    benchmarks: tuple[BenchmarkRecord, ...]
    web_context: tuple[dict[str, object], ...]
    limitations: tuple[str, ...]
    observations_path: str | None = None
    xg_release_status: str = "research_only"

    def evidence_ids(self) -> frozenset[str]:
        ids = set(self.attack.evidence_ids)
        ids.add(self.attack.attack_id)
        ids.update(action.action_id for action in self.attack.possession.actions)
        if self.attack.terminal.action_id:
            ids.add(self.attack.terminal.action_id)
        if self.xg_record is not None:
            ids.add(self.xg_record.record_id)
            ids.add(f"xg:{self.xg_record.record_id}")
        if self.fusion is not None:
            ids.add(self.fusion.association.association_id)
            freeze_id = self.fusion.association.spatial_context.get("footpass_evidence_id")
            if freeze_id:
                ids.add(str(freeze_id))
        ids.update(record.benchmark_id for record in self.benchmarks)
        for index, snippet in enumerate(self.web_context):
            ids.add(str(snippet.get("snippet_id") or f"web:{index}"))
        return frozenset(ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attack": self.attack.to_dict(),
            "fusion": None if self.fusion is None else self.fusion.to_dict(),
            "xg_record": None if self.xg_record is None else self.xg_record.to_dict(),
            "xg_status": "scored" if self.xg_record is not None else "withheld",
            "xg_probability": None if self.xg_record is None else self.xg_record.xg_probability,
            "xg_release_status": self.xg_release_status,
            "withheld_reason": self.withheld_reason,
            "quality_gates": dict(self.quality_gates),
            "quality_metrics": dict(self.quality_metrics),
            "benchmarks": [item.to_dict() for item in self.benchmarks],
            "web_context": [dict(item) for item in self.web_context],
            "web_context_limitation": "Web snippets are non-numeric context only.",
            "limitations": list(self.limitations),
            "observations_path": self.observations_path,
            "evidence_ids": sorted(self.evidence_ids()),
        }

    def as_legacy_bundle(
        self,
        *,
        shot_outcome: str | None = None,
        outcome_source: str | None = None,
    ) -> EvidenceBundle:
        """Compatibility adapter so existing tool/verify_claims contracts still work."""

        preview = _attack_to_preview_sequence(self.attack)
        features = build_attack_features(preview)
        xg_value = 0.0 if self.xg_record is None else float(self.xg_record.xg_probability)
        extra_ids = tuple(sorted(self.evidence_ids() - set(features.evidence_ids)))
        features = replace(
            features,
            xg_sum=xg_value,
            xg_max=xg_value,
            evidence_ids=tuple(features.evidence_ids) + extra_ids,
        )
        if shot_outcome == "goal":
            features = replace(
                features,
                goals=max(1, features.goals),
                shots_on_target=max(1, features.shots_on_target),
            )
        spatial = None
        xg_raw = None
        if self.fusion is not None:
            spatial = dict(self.fusion.association.spatial_context)
            xg_raw = dict(self.fusion.association.xg_raw)
            freeze_id = spatial.get("footpass_evidence_id")
            if freeze_id:
                extra_ids = tuple(dict.fromkeys((*features.evidence_ids, str(freeze_id))))
                features = replace(features, evidence_ids=extra_ids)
        return EvidenceBundle(
            attack=features,
            quality_gates={**self.quality_gates, "xg_scored": self.xg_record is not None},
            quality_metrics=dict(self.quality_metrics),
            benchmarks=self.benchmarks,
            source_ids=(self.attack.attack_id,),
            limitations=self.limitations,
            spatial_context=spatial,
            xg_raw=xg_raw,
            sequence_context=sequence_timeline(
                self.attack,
                shot_outcome=shot_outcome,
                outcome_source=outcome_source,
            ),
        )


def _zone(x: float | None, y: float | None) -> str | None:
    if x is None:
        return None
    if x >= PENALTY_BOX_X and (y is None or abs(y - PITCH_CENTRE_Y) <= PENALTY_BOX_HALF_WIDTH):
        return "penalty_box"
    if x >= FINAL_THIRD_X:
        return "final_third"
    if x >= MIDDLE_THIRD_X:
        return "middle_third"
    return "defensive_third"


def _channel(y: float | None) -> str | None:
    if y is None:
        return None
    offset = abs(y - PITCH_CENTRE_Y)
    if offset <= CENTRAL_HALF_WIDTH:
        return "central"
    if offset <= PENALTY_BOX_HALF_WIDTH:
        return "half_space"
    return "wide"


def sequence_timeline(
    attack: AttackSequenceV2,
    *,
    shot_outcome: str | None = None,
    outcome_source: str | None = None,
) -> dict[str, object]:
    """Ordered, pre-interpreted event timeline for the language model.

    Coordinates are in the attacking team's view (x=105 is the goal being
    attacked) and zones/channels are computed here so the model never has to
    infer geometry from raw numbers.
    """

    possession = attack.possession
    team = attack.team_id
    terminal_id = attack.terminal.action_id
    start = possession.start_sec
    events: list[dict[str, object]] = []
    previous_x: float | None = None
    box_entry: dict[str, object] | None = None
    biggest_gain: dict[str, object] | None = None
    for step, action in enumerate(possession.actions, start=1):
        x, y = possession_relative_position(
            action.x,
            action.y,
            actor_team_id=action.actor_team_id,
            possession_team_id=team,
        )
        side = (
            "unknown"
            if not action.actor_team_id
            else "attacking" if action.actor_team_id == team else "opponent"
        )
        gain = None
        if side == "attacking" and x is not None:
            if previous_x is not None:
                gain = round(x - previous_x, 1)
            previous_x = x
        zone = _zone(x, y)
        item: dict[str, object] = {
            "step": step,
            "evidence_id": action.action_id,
            "t_sec": round(action.timestamp_sec - start, 1),
            "action": (
                attack.terminal.terminal_type.value if action.action_id == terminal_id else action.action_type
            ),
            "team": side,
            "x": None if x is None else round(x, 1),
            "y": None if y is None else round(y, 1),
            "zone": zone,
            "channel": _channel(y),
            "forward_gain_m": gain,
        }
        if action.jersey is not None:
            item["jersey"] = action.jersey
        if action.outcome:
            item["outcome"] = action.outcome
        if action.action_id == terminal_id:
            item["terminal"] = True
        events.append(item)
        if side == "attacking" and zone == "penalty_box" and box_entry is None:
            box_entry = {"step": step, "t_sec": item["t_sec"], "evidence_id": action.action_id}
        if gain is not None and gain > 0 and (biggest_gain is None or gain > float(biggest_gain["forward_gain_m"])):
            biggest_gain = {"step": step, "forward_gain_m": gain, "evidence_id": action.action_id}

    terminal_step = next((int(item["step"]) for item in events if item.get("terminal")), None)
    key_moments = _key_moments(events, biggest_gain, terminal_step, possession.status.value)
    attacking = [item for item in events if item["team"] == "attacking"]
    counts: dict[str, int] = {}
    for item in attacking:
        counts[str(item["action"])] = counts.get(str(item["action"]), 0) + 1
    first = events[0] if events else None
    status = possession.status.value
    return {
        "reading_note": (
            "events are in time order; x/y are in the attacking team's view (x=105 is the goal attacked); "
            "zone/channel/forward_gain_m are precomputed"
        ),
        "sequence_status": status,
        "start": {
            "observed_from_beginning": status == "complete",
            "note": (
                "build-up before the first event was not observed (restart, set piece or camera gap)"
                if status == "left_censored"
                else "possession start observed"
            ),
            "first_event": None if first is None else {
                key: first[key] for key in ("step", "evidence_id", "action", "team", "zone")
            },
        },
        "duration_sec": round(possession.end_sec - possession.start_sec, 1),
        "key_moments": key_moments,
        "events": events,
        "attacking_action_counts": counts,
        "opponent_touches": sum(1 for item in events if item["team"] == "opponent"),
        "penalty_box_entry": box_entry,
        "largest_forward_gain": biggest_gain,
        "terminal": {
            "type": attack.terminal.terminal_type.value,
            "evidence_id": terminal_id,
            "body_part": attack.terminal.body_part,
            "zone": next((item["zone"] for item in events if item.get("terminal")), None),
            "channel": next((item["channel"] for item in events if item.get("terminal")), None),
        },
        "shot_outcome": shot_outcome or "unknown",
        "shot_outcome_source": outcome_source if shot_outcome else None,
    }


_ZONE_RANK = {"defensive_third": 0, "middle_third": 1, "final_third": 2, "penalty_box": 3}


def _key_moments(
    events: list[dict[str, object]],
    biggest_gain: dict[str, object] | None,
    terminal_step: int | None,
    status: str,
) -> list[dict[str, object]]:
    """Authoritative step-by-step highlights so the model does not have to locate them itself."""

    moments: list[dict[str, object]] = []

    def add(item: dict[str, object], moment: str, **detail: object) -> None:
        moments.append(
            {
                "step": item["step"],
                "evidence_id": item["evidence_id"],
                "moment": moment,
                "action": item["action"],
                "team": item["team"],
                "zone": item["zone"],
                "channel": item["channel"],
                **detail,
            }
        )

    if events:
        add(events[0], "first_observed_event" if status == "left_censored" else "possession_start")
    furthest = _ZONE_RANK.get(str(events[0]["zone"]), -1) if events else -1
    for item in events[1:]:
        if item["team"] != "attacking":
            continue
        rank = _ZONE_RANK.get(str(item["zone"]), -1)
        if rank > furthest:
            furthest = rank
            add(item, f"first_entry_{item['zone']}")
    for item in events:
        step = int(item["step"])
        if biggest_gain is not None and step == biggest_gain["step"]:
            add(item, "largest_forward_gain", forward_gain_m=biggest_gain["forward_gain_m"])
        if item["team"] == "attacking" and item["action"] == "cross":
            add(item, "cross")
        if item["team"] == "opponent":
            add(item, "opponent_touch_after_terminal" if terminal_step and step > terminal_step else "opponent_touch")
        if item.get("terminal"):
            add(item, "terminal")
    return sorted(moments, key=lambda moment: int(moment["step"]))


def default_team_mapping(attack: AttackSequenceV2) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if attack.team_id:
        mapping[attack.team_id] = "team_a"
    other = next(
        (
            action.actor_team_id
            for action in attack.possession.actions
            if action.actor_team_id and action.actor_team_id != attack.team_id
        ),
        None,
    )
    if other:
        mapping[other] = "team_b"
    return mapping


def resolve_xg_v2_model_path(project_root: Path, configured: str | None = None) -> Path:
    if configured:
        return Path(configured)
    return project_root / XG_V2_DEFAULT_RELATIVE


def _attack_to_preview_sequence(attack: AttackSequenceV2) -> AttackSequence:
    # A generic FOOTPASS "header" is any headed touch; only the attack's headed
    # terminal is an attempt that metrics count as a shot.
    headed_terminal = (
        attack.terminal.action_id
        if attack.terminal.terminal_type is TerminalType.HEADED_ATTEMPT
        else None
    )
    events = tuple(
        Event(
            timestamp=action.timestamp_sec,
            event_type=(
                "headed_attempt" if action.action_id == headed_terminal else action.action_type
            ),
            team=attack.team_id,
            x=action.x,
            y=action.y,
            outcome=action.outcome,
            frame=action.frame,
            player_id=action.actor_id,
            jersey=action.jersey,
            role=action.role,
            confidence=action.action_probability if action.action_probability is not None else 1.0,
            source="attack-v2",
            evidence_id=action.action_id,
        )
        for action in attack.possession.actions
    )
    return AttackSequence(
        attack_id=attack.attack_id,
        team=attack.team_id,
        start_sec=attack.possession.start_sec,
        end_sec=attack.possession.end_sec,
        events=events,
        match_id=attack.match_id,
        start_frame=attack.possession.start_frame,
        end_frame=attack.possession.end_frame,
        source="attack-v2",
    )


def observations_path_from_manifest(window_manifest: Mapping[str, Any] | None) -> Path | None:
    if not window_manifest:
        return None
    outputs = dict(window_manifest.get("outputs") or {})
    raw = outputs.get("observations")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_file() else None


def video_frame_for_match_frame(
    match_frame: int,
    window_manifest: Mapping[str, Any] | None = None,
) -> int:
    clock = dict((window_manifest or {}).get("video_clock") or {})
    preview_fps = clock.get("preview_fps")
    if not preview_fps:
        return match_frame
    match_fps = float(clock.get("match_fps") or 25.0)
    offset = float(clock.get("preview_offset_sec") or 0.0)
    return max(0, int(round((match_frame / match_fps + offset) * float(preview_fps))))


def frame_for_terminal(
    frames: tuple[FrameObservation, ...],
    terminal_frame: int,
    *,
    window_manifest: Mapping[str, Any] | None = None,
) -> FrameObservation | None:
    if not frames:
        return None
    target = video_frame_for_match_frame(terminal_frame, window_manifest)
    exact = next((item for item in frames if item.frame_index == target), None)
    if exact is not None:
        return exact
    return min(frames, key=lambda item: abs(item.frame_index - target))


def _attacking_direction(action) -> int | None:
    if action.raw_left_to_right is True:
        return 0
    if action.raw_left_to_right is False:
        return 1
    return None


def _load_terminal_freeze(
    attack_v2: AttackSequenceV2,
    tactical_h5: Path | None,
) -> dict[str, object] | None:
    if tactical_h5 is None or not Path(tactical_h5).is_file():
        return None
    terminal = next(
        (item for item in attack_v2.possession.actions if item.action_id == attack_v2.terminal.action_id),
        attack_v2.possession.actions[-1],
    )
    direction = _attacking_direction(terminal)
    if direction is None or terminal.x is None or terminal.y is None:
        return None
    match_key = f"{attack_v2.match_id}_H{attack_v2.period}"
    try:
        return load_footpass_freeze_frame(
            Path(tactical_h5),
            match_key=match_key,
            frame=terminal.frame,
            attacking_direction=direction,
            shooter_x=float(terminal.x),
            shooter_y=float(terminal.y),
            timestamp=terminal.timestamp_sec,
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        return None


def _ball_f1_from_manifest(window_manifest: Mapping[str, Any] | None) -> float | None:
    if not window_manifest:
        return None
    quality = dict(window_manifest.get("quality") or {})
    for key in ("ball_f1", "ball_detection_f1", "ball_f1_score"):
        if quality.get(key) is not None:
            return float(quality[key])
    return None


def build_attack_evidence(
    attack_v2: AttackSequenceV2,
    *,
    window_manifest: Mapping[str, Any] | None = None,
    xg_model_path: Path | None = None,
    team_mapping: Mapping[str, str] | None = None,
    allow_research_only: bool = True,
    benchmarks: tuple[BenchmarkRecord, ...] = (),
    web_context: tuple[dict[str, object], ...] = (),
    model: TrainedXGModelV2 | None = None,
    frames: tuple[FrameObservation, ...] | None = None,
    tactical_h5: Path | None = None,
) -> AttackEvidenceBundle:
    limitations: list[str] = [
        "xG-v2 is research_only until FOOTPASS human-domain validation.",
        "Web search cannot support numeric claims.",
    ]
    observations_path = observations_path_from_manifest(window_manifest)
    loaded_frames = frames
    if loaded_frames is None and observations_path is not None:
        loaded_frames = read_observations_jsonl(observations_path)
    terminal = attack_v2.terminal
    quality_gates = {
        "events": True,
        "attack_v2": True,
        "shot_terminal": terminal.terminal_type
        in {TerminalType.SHOT_ATTEMPT, TerminalType.HEADED_ATTEMPT},
        "yolo_observations": loaded_frames is not None and bool(loaded_frames),
        "xg_model": False,
    }
    quality_metrics: dict[str, float] = {}
    if window_manifest:
        quality = dict(window_manifest.get("quality") or {})
        for key in (
            "frames_with_ball_ratio",
            "player_tracking_coverage",
            "calibrated_position_coverage",
        ):
            if quality.get(key) is not None:
                quality_metrics[key] = float(quality[key])
    fusion: FusedXGResult | None = None
    xg_record: ConfirmedXGRecordV2 | None = None
    withheld = None
    loaded_model = model
    if loaded_model is None and xg_model_path is not None and Path(xg_model_path).is_file():
        loaded_model = load_trained_xg_v2_model(
            xg_model_path,
            allow_research_only=allow_research_only,
        )
    quality_gates["xg_model"] = loaded_model is not None
    mapping = dict(team_mapping) if team_mapping else default_team_mapping(attack_v2)
    freeze = _load_terminal_freeze(attack_v2, tactical_h5)
    quality_gates["footpass_freeze"] = freeze is not None
    if freeze:
        freeze_id = str(freeze.get("evidence_id") or "")
        if freeze_id:
            limitations.append(f"FOOTPASS freeze-frame `{freeze_id}` supplies GK/foe/packing.")
    if loaded_frames:
        frame = frame_for_terminal(
            loaded_frames,
            terminal.frame,
            window_manifest=window_manifest,
        )
        if frame is None and freeze is None:
            withheld = "terminal_frame_missing"
            limitations.append("YOLO observations exist but the terminal frame was not found.")
        else:
            fusion = score_attack_with_observations(
                attack_v2,
                frame,
                model=loaded_model,
                team_mapping=mapping,
                ball_f1=_ball_f1_from_manifest(window_manifest),
                footpass_freeze=freeze,
            )
            xg_record = fusion.xg_record
            withheld = fusion.withheld_reason
    elif freeze is not None:
        fusion = score_attack_with_observations(
            attack_v2,
            None,
            model=loaded_model,
            team_mapping=mapping,
            footpass_freeze=freeze,
        )
        xg_record = fusion.xg_record
        withheld = fusion.withheld_reason
        limitations.append("YOLO window missing; spatial counts use FOOTPASS tactical freeze-frame.")
    elif not quality_gates["shot_terminal"]:
        withheld = "terminal_not_shot"
        limitations.append("xG-v2 scores only shot_attempt / headed_attempt terminals.")
    else:
        fusion = score_attack_with_observations(
            attack_v2,
            None,
            model=loaded_model,
            team_mapping=mapping,
            footpass_freeze=None,
        )
        xg_record = fusion.xg_record
        withheld = fusion.withheld_reason
        limitations.append(
            "No YOLO window or HDF5 freeze; xG uses shooter location and sequence "
            "pattern only (GK/counts missing, xG-v2 missingness masks)."
        )
        if loaded_model is None:
            limitations.append("models/xg-v2.joblib was not found.")
    if withheld and xg_record is None:
        limitations.append(f"xG withheld: {withheld}.")
    quality_gates["tracking"] = bool(quality_metrics.get("player_tracking_coverage", 0) > 0) or bool(
        loaded_frames
    )
    quality_gates["calibration"] = bool(quality_metrics.get("calibrated_position_coverage", 0) > 0) or bool(
        loaded_frames
    )
    if freeze:
        quality_gates["tracking"] = True
        quality_gates["calibration"] = True
    elif not loaded_frames:
        quality_gates["tracking"] = False
        quality_gates["calibration"] = False
    return AttackEvidenceBundle(
        schema_version="attack-evidence-v2/1.0",
        attack=attack_v2,
        fusion=fusion,
        xg_record=xg_record,
        withheld_reason=withheld,
        quality_gates=quality_gates,
        quality_metrics=quality_metrics,
        benchmarks=benchmarks,
        web_context=web_context,
        limitations=tuple(dict.fromkeys(limitations)),
        observations_path=None if observations_path is None else str(observations_path),
    )


def conclusion_ready(bundle: AttackEvidenceBundle) -> bool:
    return bool(bundle.quality_gates.get("attack_v2"))
