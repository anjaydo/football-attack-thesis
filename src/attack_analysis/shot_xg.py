"""Attach StatsBomb xG-v2 to every analysis sequence that ends on a shot."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .attack_evidence import build_attack_evidence
from .footpass import load_footpass_freeze_frame
from .fusion import merge_fusion_context, play_pattern_evidence, shot_type_evidence
from .models import AttackSequenceV2, ObservedActionV2, PossessionSequenceV2
from .xg_v2 import TrainedXGModelV2


SHOT_EVENT_TYPES = frozenset({"shot", "shot_attempt"})


class ShotXGRequiredError(ValueError):
    """A finished analysis ends on event type shot but has no StatsBomb xG."""


def ending_event_action(possession: PossessionSequenceV2) -> ObservedActionV2 | None:
    """The event that closes the sequence: the terminal action, else the last action."""

    terminal = possession.terminal
    if terminal is not None and terminal.action_id:
        linked = next(
            (item for item in possession.actions if item.action_id == terminal.action_id),
            None,
        )
        if linked is not None:
            return linked
    if not possession.actions:
        return None
    return possession.actions[-1]


def is_shot_event(action: ObservedActionV2 | None) -> bool:
    return action is not None and action.action_type in SHOT_EVENT_TYPES


def _attacking_direction(action: ObservedActionV2) -> int | None:
    if action.raw_left_to_right is True:
        return 0
    if action.raw_left_to_right is False:
        return 1
    return None


def _freeze_for_shot(
    possession: PossessionSequenceV2,
    action: ObservedActionV2,
    tactical_h5: Path | None,
) -> dict[str, object] | None:
    if tactical_h5 is None or not Path(tactical_h5).is_file():
        return None
    direction = _attacking_direction(action)
    if direction is None or action.x is None or action.y is None:
        return None
    match_key = f"{possession.match_id}_H{possession.period}"
    try:
        return load_footpass_freeze_frame(
            Path(tactical_h5),
            match_key=match_key,
            frame=action.frame,
            attacking_direction=direction,
            shooter_x=float(action.x),
            shooter_y=float(action.y),
            timestamp=action.timestamp_sec,
        )
    except (KeyError, OSError, RuntimeError, ValueError):
        return None


def _predict_shot_probability(
    possession: PossessionSequenceV2,
    action: ObservedActionV2,
    model: TrainedXGModelV2,
    tactical_h5: Path | None,
) -> float:
    pattern = play_pattern_evidence(possession.actions)
    shot = shot_type_evidence(possession.actions)
    raw = {
        "x_shooter": action.x,
        "y_shooter": action.y,
        "body_part": action.body_part,
        "play_pattern": pattern["label"],
        "shot_type": shot["label"],
        "play_pattern_source": pattern["source"],
        "shot_type_source": shot["source"],
    }
    merged, _spatial = merge_fusion_context(
        raw,
        footpass_freeze=_freeze_for_shot(possession, action, tactical_h5),
    )
    return model.predict(merged)


def _training_payload(model: TrainedXGModelV2) -> dict[str, object]:
    report = model.report if isinstance(model.report, Mapping) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), Mapping) else {}
    status = str(report.get("status") or "")
    task = report.get("task")
    if task not in (None, "xg-v2"):
        raise ShotXGRequiredError("xG model task must be xg-v2 trained on StatsBomb shots")
    if status and "statsbomb" not in status and status != "validated":
        raise ShotXGRequiredError(
            "xG model report does not identify a StatsBomb-trained xG-v2 artifact"
        )
    return {
        "source": "statsbomb_open_data",
        "feature_schema": "xg-features-v2",
        "model_status": status,
        "rows": dataset.get("rows"),
        "matches": dataset.get("matches"),
        "model_id": model.model_id,
    }


def score_analysis_shots(
    attacks: Sequence[AttackSequenceV2],
    possessions: Sequence[PossessionSequenceV2],
    *,
    model: TrainedXGModelV2 | None,
    tactical_h5: Path | None = None,
) -> dict[str, object]:
    """Score every sequence whose closing event type is shot.

    Complete attacks use the confirmed xG-v2 record. Censored possessions that
    still close on a shot get the same model probability, marked unconfirmed.
    A shot without a finite probability is an error, not a withheld field.
    """

    attacks_by_possession = {item.possession.possession_id: item for item in attacks}
    pending: list[tuple[PossessionSequenceV2, ObservedActionV2, AttackSequenceV2 | None, bool]] = []
    seen: set[str] = set()
    for possession in possessions:
        action = ending_event_action(possession)
        if not is_shot_event(action) or action is None:
            continue
        pending.append((possession, action, attacks_by_possession.get(possession.possession_id), True))
        seen.add(possession.possession_id)
    for attack in attacks:
        if attack.possession.possession_id in seen:
            continue
        action = ending_event_action(attack.possession)
        if not is_shot_event(action) or action is None:
            continue
        pending.append((attack.possession, action, attack, True))
    scored_actions = {action.action_id for _possession, action, _attack, _terminal in pending}
    rebound_shots: list[tuple[PossessionSequenceV2, ObservedActionV2, AttackSequenceV2 | None, bool]] = []
    for possession, _terminal_action, attack, _is_terminal in pending:
        for action in possession.actions:
            if action.action_id in scored_actions or not is_shot_event(action):
                continue
            scored_actions.add(action.action_id)
            rebound_shots.append((possession, action, attack, False))
    pending.extend(rebound_shots)

    if not pending:
        return {
            "status": "no_shot_terminals",
            "training_data": None if model is None else _training_payload(model),
            "model_id": None if model is None else model.model_id,
            "shot_terminals_scored": 0,
            "records": [],
        }
    if model is None:
        identifiers = [possession.possession_id for possession, _action, _attack, _terminal in pending]
        raise ShotXGRequiredError(
            "sequence ends on event type shot but models/xg-v2.joblib is missing: "
            + ", ".join(identifiers)
        )
    training = _training_payload(model)
    records: list[dict[str, object]] = []
    for possession, action, attack, is_terminal in pending:
        confirmed = attack is not None and is_terminal
        record_id = None
        if attack is not None and is_terminal:
            evidence = build_attack_evidence(
                attack,
                model=model,
                tactical_h5=tactical_h5,
                allow_research_only=True,
            )
            if evidence.xg_record is None:
                raise ShotXGRequiredError(
                    f"{attack.attack_id} ends on shot but xG was withheld "
                    f"({evidence.withheld_reason or 'unknown'})"
                )
            probability = float(evidence.xg_record.xg_probability)
            record_id = evidence.xg_record.record_id
        else:
            try:
                probability = float(
                    _predict_shot_probability(possession, action, model, tactical_h5)
                )
            except (TypeError, ValueError, RuntimeError) as error:
                raise ShotXGRequiredError(
                    f"{possession.possession_id} ends on shot but xG could not be scored: {error}"
                ) from error
        if not 0.0 <= probability <= 1.0:
            raise ShotXGRequiredError(
                f"{possession.possession_id} shot xG {probability} is outside [0, 1]"
            )
        records.append(
            {
                "event_type": "shot",
                "action_id": action.action_id,
                "possession_id": possession.possession_id,
                "attack_id": None if attack is None else attack.attack_id,
                "sequence_status": possession.status.value,
                "confirmed_attack": confirmed,
                "terminal_type": (
                    None
                    if possession.terminal is None
                    else possession.terminal.terminal_type.value
                ),
                "xg_probability": probability,
                "terminal_shot": is_terminal,
                "record_id": record_id,
                "model_id": model.model_id,
                "training_data": "statsbomb_open_data",
            }
        )
    return {
        "status": "scored",
        "training_data": training,
        "model_id": model.model_id,
        "shot_terminals_scored": sum(1 for item in records if item["terminal_shot"]),
        "records": records,
    }
