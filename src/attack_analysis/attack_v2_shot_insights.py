from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gold_annotations import GoldSequenceAnnotationV2, SHOT_OUTCOMES, read_gold_annotations_jsonl
from .models import (
    ATTACK_TERMINAL_ALLOWLIST,
    AttackSequenceV2,
    ObservedActionV2,
    PossessionSequenceV2,
    SequenceStatus,
    TerminalType,
)


SHOT_ACTION_TYPES = frozenset({"shot", "shot_attempt"})
ATTEMPT_TERMINAL_TYPES = frozenset(
    {
        TerminalType.SHOT_ATTEMPT.value,
        TerminalType.HEADED_ATTEMPT.value,
    }
)

SEQUENCE_FILTER_CONFIRMED = "Confirmed Attack-v2"
SEQUENCE_FILTER_CENSORED_SHOT = "Shot terminal · bị censor"
SEQUENCE_FILTER_ALL_POSSESSIONS = "All possessions"

SHOT_OUTCOME_LABELS = {
    "goal": "Bàn thắng",
    "on_target": "Trúng đích",
    "blocked": "Bị chặn",
    "off_target": "Ra ngoài",
    "woodwork": "Trúng khung thành",
    "unknown": "Chưa rõ",
}


def _as_possession(item: PossessionSequenceV2 | Mapping[str, Any]) -> PossessionSequenceV2:
    return item if isinstance(item, PossessionSequenceV2) else PossessionSequenceV2.from_dict(dict(item))


def _as_attack(item: AttackSequenceV2 | Mapping[str, Any]) -> AttackSequenceV2:
    return item if isinstance(item, AttackSequenceV2) else AttackSequenceV2.from_dict(dict(item))


def _as_action(item: ObservedActionV2 | Mapping[str, Any]) -> ObservedActionV2:
    return item if isinstance(item, ObservedActionV2) else ObservedActionV2.from_dict(dict(item))


@dataclass(frozen=True)
class GoldOutcomeIndex:
    """Frame-indexed gold shot outcomes for development/train splits."""

    annotations: tuple[GoldSequenceAnnotationV2, ...]

    @classmethod
    def load(cls, project_root: Path) -> "GoldOutcomeIndex":
        annotations: list[GoldSequenceAnnotationV2] = []
        for name in ("attack-v2-development.jsonl", "attack-v2-train.jsonl"):
            path = project_root / "data" / name
            if path.is_file():
                annotations.extend(read_gold_annotations_jsonl(path))
        return cls(tuple(annotations))

    def lookup(
        self,
        possession: PossessionSequenceV2,
    ) -> tuple[str | None, str | None]:
        if possession.terminal is None:
            return None, None
        terminal = possession.terminal
        if terminal.terminal_type.value not in ATTEMPT_TERMINAL_TYPES:
            return None, None
        terminal_frame = terminal.frame
        candidates = [
            item
            for item in self.annotations
            if item.match_id == possession.match_id
            and item.period == possession.period
            and item.stable_team_id == possession.possession_team_id
            and item.terminal_type in ATTEMPT_TERMINAL_TYPES
            and item.shot_outcome in SHOT_OUTCOMES
            and item.start_frame <= terminal_frame <= item.end_frame
        ]
        if not candidates:
            return None, None
        exact = [
            item
            for item in candidates
            if item.terminal_frame is not None and item.terminal_frame == terminal_frame
        ]
        chosen = exact[0] if exact else sorted(candidates, key=lambda item: item.start_frame)[-1]
        return chosen.shot_outcome, "gold_annotation"


def resolve_shot_outcome(
    possession: PossessionSequenceV2,
    *,
    gold_index: GoldOutcomeIndex | None = None,
) -> tuple[str | None, str | None]:
    """Resolve display outcome with explicit provenance priority."""

    if gold_index is not None:
        outcome, source = gold_index.lookup(possession)
        if outcome is not None:
            return outcome, source
    terminal = possession.terminal
    if terminal is not None and terminal.outcome:
        return terminal.outcome, "terminal_event"
    for action in reversed(possession.actions):
        if action.action_type in SHOT_ACTION_TYPES and action.outcome:
            return action.outcome, "observed_action"
    return None, None


def format_shot_outcome(outcome: str | None) -> str:
    if not outcome:
        return "—"
    return SHOT_OUTCOME_LABELS.get(outcome, outcome.replace("_", " ").title())


@dataclass(frozen=True)
class ShotPromotionRecord:
    action_id: str
    frame: int
    timestamp_sec: float
    team_id: str | None
    jersey: int | None
    promotion_status: str
    possession_id: str | None
    possession_status: str | None
    terminal_type: str | None
    attack_id: str | None
    censor_reason: str | None
    shot_outcome: str | None
    shot_outcome_source: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "frame": self.frame,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "team_id": self.team_id,
            "jersey": self.jersey,
            "promotion_status": self.promotion_status,
            "possession_id": self.possession_id,
            "possession_status": self.possession_status,
            "terminal_type": self.terminal_type,
            "attack_id": self.attack_id,
            "censor_reason": self.censor_reason,
            "shot_outcome": self.shot_outcome,
            "shot_outcome_source": self.shot_outcome_source,
        }


@dataclass(frozen=True)
class ShotInsights:
    records: tuple[ShotPromotionRecord, ...]
    promoted_action_ids: frozenset[str]
    attack_by_terminal_action: dict[str, str]
    censored_possession_ids: frozenset[str]
    gold_available: bool

    @property
    def summary(self) -> dict[str, object]:
        statuses = {}
        for record in self.records:
            statuses[record.promotion_status] = statuses.get(record.promotion_status, 0) + 1
        outcomes = {}
        for record in self.records:
            if record.shot_outcome:
                outcomes[record.shot_outcome] = outcomes.get(record.shot_outcome, 0) + 1
        return {
            "live_shots": len(self.records),
            "promoted_to_attack": statuses.get("promoted", 0),
            "censored_shot_terminal": statuses.get("censored", 0),
            "unassigned_shots": statuses.get("unassigned", 0),
            "promotion_statuses": statuses,
            "shot_outcomes": outcomes,
            "gold_outcomes_available": self.gold_available,
        }


def _possession_by_action(
    possessions: Sequence[PossessionSequenceV2 | Mapping[str, Any]],
) -> dict[str, PossessionSequenceV2]:
    mapping: dict[str, PossessionSequenceV2] = {}
    for item in possessions:
        possession = _as_possession(item)
        for action in possession.actions:
            mapping[action.action_id] = possession
    return mapping


def _attack_terminal_index(
    attacks: Sequence[AttackSequenceV2 | Mapping[str, Any]],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for item in attacks:
        attack = _as_attack(item)
        terminal = attack.terminal
        key = terminal.action_id or (terminal.evidence_ids[0] if terminal.evidence_ids else None)
        if key is not None:
            index[key] = attack.attack_id
    return index


def build_shot_insights(
    observed_actions: Sequence[ObservedActionV2 | Mapping[str, Any]],
    possessions: Sequence[PossessionSequenceV2 | Mapping[str, Any]],
    attacks: Sequence[AttackSequenceV2 | Mapping[str, Any]],
    *,
    excluded_replay_action_ids: Iterable[str] = (),
    gold_index: GoldOutcomeIndex | None = None,
) -> ShotInsights:
    possession_models = tuple(_as_possession(item) for item in possessions)
    attack_models = tuple(_as_attack(item) for item in attacks)
    action_to_possession = _possession_by_action(possession_models)
    attack_by_terminal = _attack_terminal_index(attack_models)
    promoted_terminal_ids = frozenset(attack_by_terminal)
    attack_possession_ids = frozenset(attack.possession.possession_id for attack in attack_models)
    replay_ids = frozenset(excluded_replay_action_ids)

    censored_ids: set[str] = set()
    for possession in possession_models:
        terminal = possession.terminal
        if terminal is None:
            continue
        if terminal.terminal_type.value not in ATTEMPT_TERMINAL_TYPES:
            continue
        if possession.possession_id in attack_possession_ids:
            continue
        censored_ids.add(possession.possession_id)

    records: list[ShotPromotionRecord] = []
    for raw_action in observed_actions:
        action = _as_action(raw_action)
        if action.action_type not in SHOT_ACTION_TYPES:
            continue
        if action.replay or action.action_id in replay_ids:
            continue
        possession = action_to_possession.get(action.action_id)
        terminal = possession.terminal if possession is not None else None
        terminal_type = terminal.terminal_type.value if terminal is not None else None
        attack_id = attack_by_terminal.get(action.action_id)
        if attack_id is not None:
            promotion_status = "promoted"
            censor_reason = None
        elif possession is not None and terminal_type in ATTEMPT_TERMINAL_TYPES:
            promotion_status = "censored"
            censor_reason = possession.status.value
        elif possession is not None:
            promotion_status = "unassigned"
            censor_reason = "terminal_not_allowlisted"
        else:
            promotion_status = "unassigned"
            censor_reason = "no_possession"
        outcome, outcome_source = (
            resolve_shot_outcome(possession, gold_index=gold_index)
            if possession is not None
            else (None, None)
        )
        records.append(
            ShotPromotionRecord(
                action_id=action.action_id,
                frame=action.frame,
                timestamp_sec=action.timestamp_sec,
                team_id=action.actor_team_id or action.possession_team_id,
                jersey=action.jersey,
                promotion_status=promotion_status,
                possession_id=possession.possession_id if possession is not None else None,
                possession_status=possession.status.value if possession is not None else None,
                terminal_type=terminal_type,
                attack_id=attack_id,
                censor_reason=censor_reason,
                shot_outcome=outcome,
                shot_outcome_source=outcome_source,
            )
        )
    records.sort(key=lambda item: (item.frame, item.action_id))
    return ShotInsights(
        records=tuple(records),
        promoted_action_ids=promoted_terminal_ids,
        attack_by_terminal_action=attack_by_terminal,
        censored_possession_ids=frozenset(censored_ids),
        gold_available=gold_index is not None and bool(gold_index.annotations),
    )


def filter_sequence_entries(
    *,
    sequence_filter: str,
    possessions: Sequence[Mapping[str, Any]],
    attacks: Sequence[Mapping[str, Any]],
    censored_possession_ids: Iterable[str],
) -> tuple[tuple[str, dict[str, object]], ...]:
    censored = frozenset(censored_possession_ids)
    if sequence_filter == SEQUENCE_FILTER_CONFIRMED:
        if attacks:
            return tuple(("attack", item) for item in attacks)
        return ()
    if sequence_filter == SEQUENCE_FILTER_CENSORED_SHOT:
        return tuple(
            ("possession", item)
            for item in possessions
            if str(item.get("possession_id") or "") in censored
        )
    return tuple(("possession", item) for item in possessions)


def possession_is_confirmed_attack(
    possession: PossessionSequenceV2,
    attack_possession_ids: Iterable[str],
) -> bool:
    return possession.possession_id in frozenset(attack_possession_ids)


def actions_with_resolved_outcome(
    possession: PossessionSequenceV2,
    *,
    shot_outcome: str | None,
    outcome_source: str | None,
) -> tuple[ObservedActionV2, ...]:
    if shot_outcome is None or possession.terminal is None:
        return possession.actions
    terminal_id = possession.terminal.action_id
    enriched: list[ObservedActionV2] = []
    for action in possession.actions:
        if action.action_id != terminal_id:
            enriched.append(action)
            continue
        qualifiers = tuple(action.qualifiers)
        if outcome_source and f"outcome_source:{outcome_source}" not in qualifiers:
            qualifiers = (*qualifiers, f"outcome_source:{outcome_source}")
        enriched.append(replace(action, outcome=shot_outcome, qualifiers=qualifiers))
    return tuple(enriched)


def attack_possession_ids(attacks: Sequence[AttackSequenceV2 | Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(_as_attack(item).possession.possession_id for item in attacks)
