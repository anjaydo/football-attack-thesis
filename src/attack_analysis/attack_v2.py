from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from typing import Iterable

from .models import (
    ATTACK_TERMINAL_ALLOWLIST,
    ATTACK_V2_RUBRIC_VERSION,
    FORCED_GOAL_LINE_OUT_CAUSES,
    AttackSequence,
    AttackSequenceV2,
    FieldProvenanceV2,
    ObservedActionV2,
    PossessionSequenceV2,
    PossessionState,
    PossessionStateTransitionV2,
    SequenceStatus,
    SourceTier,
    TerminalEventV2,
    TerminalType,
)


@dataclass(frozen=True)
class FootpassAttackV2Config:
    """Deterministic rules used before a learned structured decoder is available.

    A time gap is evidence loss, never a football terminal.  Consequently both
    sides of a large gap are censored and cannot become a validated attack.
    """

    max_observation_gap_sec: float = 8.0
    rebound_window_sec: float = 3.0
    event_cluster_sec: float = 0.50
    fps: float = 25.0
    rubric_version: str = ATTACK_V2_RUBRIC_VERSION
    controlled_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"drive", "pass", "cross"})
    )
    contested_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"tackle", "block", "header"})
    )
    restart_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"throw_in", "throw-in"})
    )

    def __post_init__(self) -> None:
        if self.max_observation_gap_sec <= 0:
            raise ValueError("max_observation_gap_sec must be positive")
        if self.rebound_window_sec < 0:
            raise ValueError("rebound_window_sec must be non-negative")
        if self.event_cluster_sec < 0:
            raise ValueError("event_cluster_sec must be non-negative")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not self.rubric_version.strip():
            raise ValueError("rubric_version must not be empty")
        for field_name in (
            "controlled_action_types",
            "contested_action_types",
            "restart_action_types",
        ):
            values = frozenset(str(value).strip().lower() for value in getattr(self, field_name))
            if not values or "" in values:
                raise ValueError(f"{field_name} must contain non-empty action names")
            object.__setattr__(self, field_name, values)


@dataclass(frozen=True)
class AttackV2DecodeResult:
    possessions: tuple[PossessionSequenceV2, ...]
    attacks: tuple[AttackSequenceV2, ...]
    excluded_replay_action_ids: tuple[str, ...] = ()
    duplicate_action_ids: tuple[str, ...] = ()
    unassigned_action_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        possession_ids = {item.possession_id for item in self.possessions}
        if len(possession_ids) != len(self.possessions):
            raise ValueError("decoder emitted duplicate possessions")
        if any(item.possession.possession_id not in possession_ids for item in self.attacks):
            raise ValueError("every attack must reference a decoded possession")
        if len({item.attack_id for item in self.attacks}) != len(self.attacks):
            raise ValueError("decoder emitted duplicate attacks")

    def to_dict(self) -> dict[str, object]:
        return {
            "possessions": [item.to_dict() for item in self.possessions],
            "attacks": [item.to_dict() for item in self.attacks],
            "excluded_replay_action_ids": list(self.excluded_replay_action_ids),
            "duplicate_action_ids": list(self.duplicate_action_ids),
            "unassigned_action_ids": list(self.unassigned_action_ids),
        }


@dataclass
class _PossessionBuilder:
    team_id: str
    initial_status: SequenceStatus
    actions: list[ObservedActionV2]
    transitions: list[PossessionStateTransitionV2]


@dataclass(frozen=True)
class _TerminalCandidate:
    action: ObservedActionV2
    terminal_type: TerminalType
    possession_team_id: str
    priority: int


_FORCED_OUT_CAUSE_ALIASES = {
    "save": "keeper_save_out",
    "saved": "keeper_save_out",
    "keeper_save": "keeper_save_out",
    "block": "defender_block_out",
    "blocked": "defender_block_out",
    "deflection": "defender_deflection_out",
    "deflected": "defender_deflection_out",
    "clearance": "defender_clearance_out",
    "cleared": "defender_clearance_out",
}


def _forced_out_cause(action: ObservedActionV2) -> str | None:
    evidence_labels = (action.action_type, action.outcome, *action.qualifiers)
    return next(
        (label for label in evidence_labels if label in FORCED_GOAL_LINE_OUT_CAUSES),
        None,
    ) or next(
        (
            _FORCED_OUT_CAUSE_ALIASES[label]
            for label in evidence_labels
            if label in _FORCED_OUT_CAUSE_ALIASES
        ),
        None,
    )


def _merge_source_tiers(tiers: Iterable[SourceTier]) -> SourceTier:
    unique = set(tiers)
    if not unique:
        raise ValueError("at least one source tier is required")
    if len(unique) == 1:
        return next(iter(unique))
    return SourceTier.FUSED


def _assign_possession(action: ObservedActionV2, team_id: str) -> ObservedActionV2:
    if action.possession_team_id == team_id:
        return action
    provenance = list(action.field_provenance)
    if not any(item.field_name == "possession_team_id" for item in provenance):
        provenance.append(
            FieldProvenanceV2(
                field_name="possession_team_id",
                source_tier=action.source_tier,
                source_record_id="attack-v2-deterministic-decoder",
                evidence_ids=(action.action_id,),
            )
        )
    return replace(
        action,
        possession_team_id=team_id,
        field_provenance=tuple(provenance),
    )


def _transition(
    builder: _PossessionBuilder,
    action: ObservedActionV2,
    state: PossessionState,
    team_id: str | None = None,
) -> None:
    candidate = PossessionStateTransitionV2(
        frame=action.frame,
        timestamp_sec=action.timestamp_sec,
        state=state,
        team_id=team_id,
        evidence_ids=(action.action_id,),
    )
    if builder.transitions:
        previous = builder.transitions[-1]
        if previous.state is candidate.state and previous.team_id == candidate.team_id:
            return
    builder.transitions.append(candidate)


def _start_builder(
    team_id: str,
    actions: Iterable[ObservedActionV2],
    initial_status: SequenceStatus,
    state: PossessionState = PossessionState.CONTROLLED,
) -> _PossessionBuilder:
    assigned = [_assign_possession(action, team_id) for action in actions]
    if not assigned:
        raise ValueError("cannot start a possession without actions")
    builder = _PossessionBuilder(
        team_id=team_id,
        initial_status=initial_status,
        actions=assigned,
        transitions=[],
    )
    _transition(
        builder,
        assigned[0],
        state,
        team_id if state is PossessionState.CONTROLLED else None,
    )
    return builder


def _append_actions(
    builder: _PossessionBuilder,
    actions: Iterable[ObservedActionV2],
) -> list[ObservedActionV2]:
    assigned = [_assign_possession(action, builder.team_id) for action in actions]
    builder.actions.extend(assigned)
    return assigned


def _combined_status(
    initial_status: SequenceStatus,
    end_status: SequenceStatus,
) -> SequenceStatus:
    if initial_status is SequenceStatus.UNCERTAIN or end_status is SequenceStatus.UNCERTAIN:
        return SequenceStatus.UNCERTAIN
    if initial_status is SequenceStatus.LEFT_CENSORED:
        if end_status is SequenceStatus.RIGHT_CENSORED:
            return SequenceStatus.UNCERTAIN
        return SequenceStatus.LEFT_CENSORED
    return end_status


def _close_builder(
    builder: _PossessionBuilder,
    *,
    end_status: SequenceStatus,
    terminal: TerminalEventV2 | None = None,
    rubric_version: str,
) -> PossessionSequenceV2:
    actions = tuple(sorted(builder.actions, key=lambda item: (item.frame, item.timestamp_sec, item.action_id)))
    first = actions[0]
    last = actions[-1]
    end_frame = max(last.frame, terminal.frame if terminal is not None else last.frame)
    end_sec = max(last.timestamp_sec, terminal.timestamp_sec if terminal is not None else last.timestamp_sec)
    transitions = tuple(
        sorted(builder.transitions, key=lambda item: (item.frame, item.timestamp_sec))
    )
    tiers = [item.source_tier for item in actions]
    if terminal is not None:
        tiers.append(terminal.source_tier)
    evidence_ids = tuple(
        evidence_id
        for action in actions
        for evidence_id in (action.action_id, *action.evidence_ids)
    )
    return PossessionSequenceV2(
        match_id=first.match_id,
        period=first.period,
        possession_team_id=builder.team_id,
        start_frame=first.frame,
        end_frame=end_frame,
        start_sec=first.timestamp_sec,
        end_sec=end_sec,
        actions=actions,
        status=_combined_status(builder.initial_status, end_status),
        source_tier=_merge_source_tiers(tiers),
        terminal=terminal,
        state_transitions=transitions,
        evidence_ids=evidence_ids,
        rubric_version=rubric_version,
    )


def _terminal_candidate(action: ObservedActionV2) -> _TerminalCandidate | None:
    action_type = action.action_type
    qualifiers = set(action.qualifiers)
    actor_team = action.actor_team_id or action.possession_team_id

    if action_type in {"headed_attempt", "headed_shot"} or (
        action_type in {"header", "shot", "shot_attempt"}
        and (
            "headed_attempt" in qualifiers
            or (action_type in {"shot", "shot_attempt"} and action.body_part in {"head", "header"})
        )
    ):
        if actor_team:
            return _TerminalCandidate(action, TerminalType.HEADED_ATTEMPT, actor_team, 100)
        return None
    if action_type in {"shot", "shot_attempt"}:
        if actor_team:
            return _TerminalCandidate(action, TerminalType.SHOT_ATTEMPT, actor_team, 90)
        return None
    if action_type == "forced_goal_line_out" or "forced_goal_line_out" in qualifiers:
        team_id = action.beneficiary_team_id or action.possession_team_id
        if (
            team_id
            and action.beneficiary_team_id == team_id
            and actor_team
            and actor_team != team_id
            and _forced_out_cause(action) is not None
        ):
            return _TerminalCandidate(
                action,
                TerminalType.FORCED_GOAL_LINE_OUT,
                team_id,
                80,
            )
        return None

    terminal_types = {
        "turnover": TerminalType.TURNOVER,
        "unforced_out": TerminalType.UNFORCED_OUT,
        "ball_out": TerminalType.UNFORCED_OUT,
        "foul": TerminalType.FOUL,
        "offside": TerminalType.OFFSIDE,
        "period_end": TerminalType.PERIOD_END,
    }
    terminal_type = terminal_types.get(action_type)
    team_id = action.possession_team_id or actor_team
    if terminal_type is not None and team_id:
        return _TerminalCandidate(action, terminal_type, team_id, 50)
    return None


def _pick_terminal_candidate(
    cluster: tuple[ObservedActionV2, ...],
    current_team_id: str | None,
) -> _TerminalCandidate | None:
    candidates = [candidate for action in cluster if (candidate := _terminal_candidate(action))]
    if not candidates:
        return None
    # A candidate for the established possession wins over a simultaneous
    # conflicting prediction.  Remaining ties are fully deterministic.
    candidates.sort(
        key=lambda item: (
            item.possession_team_id == current_team_id,
            item.priority,
            item.action.action_probability if item.action.action_probability is not None else -1.0,
            item.action.action_id,
        ),
        reverse=True,
    )
    return candidates[0]


def _make_terminal(candidate: _TerminalCandidate) -> TerminalEventV2:
    action = candidate.action
    cause = None
    if candidate.terminal_type is TerminalType.FORCED_GOAL_LINE_OUT:
        cause = _forced_out_cause(action)
        if cause is None:
            raise RuntimeError("validated forced-out candidate lost its causal evidence")
    return TerminalEventV2(
        match_id=action.match_id,
        period=action.period,
        frame=action.frame,
        timestamp_sec=action.timestamp_sec,
        terminal_type=candidate.terminal_type,
        possession_team_id=candidate.possession_team_id,
        source_tier=action.source_tier,
        beneficiary_team_id=action.beneficiary_team_id,
        last_touch_team_id=action.actor_team_id,
        cause=cause,
        body_part=("head" if candidate.terminal_type is TerminalType.HEADED_ATTEMPT else action.body_part),
        outcome=action.outcome,
        confidence=action.action_probability,
        action_id=action.action_id,
        evidence_ids=(action.action_id, *action.evidence_ids),
    )


def _make_turnover_terminal(builder: _PossessionBuilder, evidence: ObservedActionV2) -> TerminalEventV2:
    last = max(builder.actions, key=lambda item: (item.frame, item.timestamp_sec, item.action_id))
    return TerminalEventV2(
        match_id=last.match_id,
        period=last.period,
        frame=last.frame,
        timestamp_sec=last.timestamp_sec,
        terminal_type=TerminalType.TURNOVER,
        possession_team_id=builder.team_id,
        source_tier=_merge_source_tiers((last.source_tier, evidence.source_tier)),
        beneficiary_team_id=evidence.actor_team_id,
        last_touch_team_id=builder.team_id,
        cause="opponent_established_control",
        approximate_time=True,
        evidence_ids=(last.action_id, evidence.action_id),
    )


def _make_throw_in_terminal(
    builder: _PossessionBuilder,
    throw_in: ObservedActionV2,
) -> TerminalEventV2:
    return TerminalEventV2(
        match_id=throw_in.match_id,
        period=throw_in.period,
        frame=throw_in.frame,
        timestamp_sec=throw_in.timestamp_sec,
        terminal_type=TerminalType.UNFORCED_OUT,
        possession_team_id=builder.team_id,
        source_tier=throw_in.source_tier,
        beneficiary_team_id=throw_in.actor_team_id or throw_in.beneficiary_team_id,
        cause="throw_in_restart_observed",
        approximate_time=True,
        action_id=throw_in.action_id,
        evidence_ids=(throw_in.action_id, *throw_in.evidence_ids),
    )


def _cluster_actions(
    actions: tuple[ObservedActionV2, ...],
    config: FootpassAttackV2Config,
) -> tuple[tuple[ObservedActionV2, ...], ...]:
    if not actions:
        return ()
    clusters: list[list[ObservedActionV2]] = [[actions[0]]]
    max_cluster_frames = ceil(config.event_cluster_sec * config.fps) + 1
    for action in actions[1:]:
        previous = clusters[-1][-1]
        terminal_actions = [
            item for item in clusters[-1] if _terminal_candidate(item) is not None
        ]
        terminal_precedes_new_control = bool(terminal_actions) and (
            action.frame,
            action.timestamp_sec,
        ) > min((item.frame, item.timestamp_sec) for item in terminal_actions) and (
            action.action_type in config.controlled_action_types
            or action.action_type in config.restart_action_types
            or _terminal_candidate(action) is not None
        )
        same_match_period = action.match_id == previous.match_id and action.period == previous.period
        close_in_time = action.timestamp_sec - previous.timestamp_sec <= config.event_cluster_sec
        close_in_frames = action.frame - previous.frame <= max_cluster_frames
        if (
            same_match_period
            and close_in_time
            and close_in_frames
            and not terminal_precedes_new_control
        ):
            clusters[-1].append(action)
        else:
            clusters.append([action])
    return tuple(tuple(cluster) for cluster in clusters)


def qualify_attack_v2(
    possession: PossessionSequenceV2,
    *,
    calibrated_confidence: float | None = None,
    calibration_id: str | None = None,
) -> AttackSequenceV2 | None:
    if possession.status is not SequenceStatus.COMPLETE or possession.terminal is None:
        return None
    if possession.terminal.terminal_type not in ATTACK_TERMINAL_ALLOWLIST:
        return None
    return AttackSequenceV2(
        possession=possession,
        calibrated_confidence=calibrated_confidence,
        calibration_id=calibration_id,
    )


def _same_team_rebound_follows(
    clusters: tuple[tuple[ObservedActionV2, ...], ...],
    index: int,
    candidate: _TerminalCandidate,
    settings: FootpassAttackV2Config,
) -> bool:
    """Keep a blocked shot open when the same team shoots again immediately.

    A đá bồi is the same attacking move: shot, block, then the same team
    controls or shoots again before the opponent restarts or takes control.
    """

    if settings.rebound_window_sec <= 0:
        return False
    if candidate.terminal_type not in {TerminalType.SHOT_ATTEMPT, TerminalType.HEADED_ATTEMPT}:
        return False
    shot = candidate.action
    team = candidate.possession_team_id
    previous_end = clusters[index][-1]
    for later in clusters[index + 1 :]:
        first = later[0]
        if first.match_id != shot.match_id or first.period != shot.period:
            return False
        if first.timestamp_sec - shot.timestamp_sec > settings.rebound_window_sec:
            return False
        if first.timestamp_sec - previous_end.timestamp_sec > settings.max_observation_gap_sec:
            return False
        for item in later:
            if item.action_type in settings.restart_action_types:
                return False
            opponent_control = (
                item.action_type in settings.controlled_action_types
                and item.actor_team_id not in {None, team}
            )
            if opponent_control:
                return False
            later_candidate = _terminal_candidate(item)
            same_team_shot = (
                later_candidate is not None
                and later_candidate.possession_team_id == team
                and later_candidate.terminal_type
                in {TerminalType.SHOT_ATTEMPT, TerminalType.HEADED_ATTEMPT}
            )
            if item.actor_team_id == team and (
                item.action_type in settings.controlled_action_types or same_team_shot
            ):
                return True
        previous_end = later[-1]
    return False


def decode_footpass_attack_v2(
    actions: Iterable[ObservedActionV2],
    config: FootpassAttackV2Config | None = None,
) -> AttackV2DecodeResult:
    """Decode FOOTPASS-oriented observations into possessions and strict attacks.

    The decoder first clusters simultaneous evidence and classifies terminal
    signals, then applies constrained possession transitions.  Tackle, block,
    and generic header observations are contested touches: their actor team is
    not interpreted as a possession change.
    """

    settings = config or FootpassAttackV2Config()
    ordered_input = sorted(
        tuple(actions),
        key=lambda item: (item.match_id, item.period, item.frame, item.timestamp_sec, item.action_id),
    )
    replay_ids = tuple(sorted(item.action_id for item in ordered_input if item.replay))
    live_actions: list[ObservedActionV2] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    for action in ordered_input:
        if action.replay:
            continue
        if action.action_id in seen_ids:
            duplicate_ids.append(action.action_id)
            continue
        seen_ids.add(action.action_id)
        live_actions.append(action)

    possessions: list[PossessionSequenceV2] = []
    attacks: list[AttackSequenceV2] = []
    unassigned_ids: list[str] = []
    current: _PossessionBuilder | None = None
    previous_cluster_end: ObservedActionV2 | None = None
    next_initial_status = SequenceStatus.COMPLETE

    def emit(
        end_status: SequenceStatus,
        terminal: TerminalEventV2 | None = None,
    ) -> None:
        nonlocal current
        if current is None:
            return
        possession = _close_builder(
            current,
            end_status=end_status,
            terminal=terminal,
            rubric_version=settings.rubric_version,
        )
        possessions.append(possession)
        attack = qualify_attack_v2(possession)
        if attack is not None:
            attacks.append(attack)
        current = None

    clusters = _cluster_actions(tuple(live_actions), settings)
    for cluster_index, cluster in enumerate(clusters):
        first = cluster[0]
        last = cluster[-1]

        if previous_cluster_end is not None:
            changed_stream = (
                first.match_id != previous_cluster_end.match_id
                or first.period != previous_cluster_end.period
            )
            observation_gap = (
                first.timestamp_sec - previous_cluster_end.timestamp_sec
                > settings.max_observation_gap_sec
            )
            if changed_stream:
                emit(SequenceStatus.RIGHT_CENSORED)
                next_initial_status = SequenceStatus.LEFT_CENSORED
            elif observation_gap:
                emit(SequenceStatus.RIGHT_CENSORED)
                next_initial_status = SequenceStatus.LEFT_CENSORED

        if any(
            item.action_type == "scene_cut" or "scene_cut_boundary" in item.qualifiers
            for item in cluster
        ):
            emit(SequenceStatus.RIGHT_CENSORED)
            unassigned_ids.extend(item.action_id for item in cluster)
            next_initial_status = SequenceStatus.LEFT_CENSORED
            previous_cluster_end = last
            continue

        replay_status_unknown = any(
            "replay_status_unknown" in item.qualifiers for item in cluster
        )
        if replay_status_unknown:
            next_initial_status = SequenceStatus.UNCERTAIN
            if current is not None:
                current.initial_status = SequenceStatus.UNCERTAIN

        terminal_candidate = _pick_terminal_candidate(
            cluster,
            current.team_id if current is not None else None,
        )
        if terminal_candidate is not None:
            terminal_team = terminal_candidate.possession_team_id
            if current is not None and current.team_id != terminal_team:
                emit(
                    SequenceStatus.COMPLETE,
                    _make_turnover_terminal(current, terminal_candidate.action),
                )
            if current is None:
                established_control_in_cluster = any(
                    item.actor_team_id == terminal_team
                    and item.action_type in settings.controlled_action_types
                    and (item.frame, item.timestamp_sec)
                    <= (
                        terminal_candidate.action.frame,
                        terminal_candidate.action.timestamp_sec,
                    )
                    for item in cluster
                )
                current = _start_builder(
                    terminal_team,
                    cluster,
                    (
                        SequenceStatus.UNCERTAIN
                        if replay_status_unknown
                        else next_initial_status
                        if established_control_in_cluster
                        else SequenceStatus.LEFT_CENSORED
                    ),
                    PossessionState.CONTROLLED,
                )
            else:
                assigned = _append_actions(current, cluster)
                _transition(current, assigned[0], PossessionState.BALL_FLIGHT)
            terminal = _make_terminal(terminal_candidate)
            # The selected terminal's possession assignment is decoder output;
            # the terminal's stable observed action ID remains unchanged.
            if terminal.possession_team_id != current.team_id:
                raise RuntimeError("terminal team and possession decoder state diverged")
            if _same_team_rebound_follows(clusters, cluster_index, terminal_candidate, settings):
                next_initial_status = SequenceStatus.COMPLETE
                previous_cluster_end = last
                continue
            emit(SequenceStatus.COMPLETE, terminal)
            next_initial_status = SequenceStatus.COMPLETE
            previous_cluster_end = last
            continue

        throw_in = next(
            (item for item in cluster if item.action_type in settings.restart_action_types),
            None,
        )
        if throw_in is not None:
            if current is not None:
                emit(SequenceStatus.COMPLETE, _make_throw_in_terminal(current, throw_in))
            restart_team = throw_in.actor_team_id or throw_in.beneficiary_team_id
            if restart_team is None:
                unassigned_ids.extend(item.action_id for item in cluster)
                next_initial_status = SequenceStatus.UNCERTAIN
            else:
                current = _start_builder(
                    restart_team,
                    cluster,
                    SequenceStatus.COMPLETE,
                    PossessionState.DEAD_BALL,
                )
                _transition(current, current.actions[0], PossessionState.CONTROLLED, restart_team)
                next_initial_status = SequenceStatus.COMPLETE
            previous_cluster_end = last
            continue

        controlled_teams = {
            item.actor_team_id
            for item in cluster
            if item.action_type in settings.controlled_action_types and item.actor_team_id
        }
        if len(controlled_teams) == 1:
            controlled_team = next(iter(controlled_teams))
            if current is not None and current.team_id != controlled_team:
                emit(
                    SequenceStatus.COMPLETE,
                    _make_turnover_terminal(current, first),
                )
            if current is None:
                current = _start_builder(controlled_team, cluster, next_initial_status)
            else:
                assigned = _append_actions(current, cluster)
                _transition(current, assigned[0], PossessionState.CONTROLLED, current.team_id)
            next_initial_status = SequenceStatus.COMPLETE
        elif len(controlled_teams) > 1:
            if current is None:
                unassigned_ids.extend(item.action_id for item in cluster)
                next_initial_status = SequenceStatus.UNCERTAIN
            else:
                assigned = _append_actions(current, cluster)
                _transition(current, assigned[0], PossessionState.CONTESTED)
        elif current is not None:
            assigned = _append_actions(current, cluster)
            _transition(current, assigned[0], PossessionState.CONTESTED)
        else:
            unassigned_ids.extend(item.action_id for item in cluster)

        previous_cluster_end = last

    emit(SequenceStatus.RIGHT_CENSORED)
    return AttackV2DecodeResult(
        possessions=tuple(possessions),
        attacks=tuple(attacks),
        excluded_replay_action_ids=replay_ids,
        duplicate_action_ids=tuple(sorted(duplicate_ids)),
        unassigned_action_ids=tuple(sorted(set(unassigned_ids))),
    )


def legacy_attack_to_possession_v2(
    sequence: AttackSequence,
    *,
    match_id: str | None = None,
    period: int = 1,
    fps: float = 25.0,
) -> PossessionSequenceV2:
    """Read a legacy AttackSequence only as an unvalidated possession candidate.

    Even a legacy sequence ending in ``shot`` remains uncertain.  A human or a
    v2 terminal decoder must establish its boundary and terminal before it can
    be passed to :func:`qualify_attack_v2`.
    """

    resolved_match_id = match_id or sequence.match_id
    if not resolved_match_id:
        raise ValueError("match_id is required when the legacy sequence has none")
    if period < 1 or fps <= 0:
        raise ValueError("period and fps must be positive")

    converted: list[ObservedActionV2] = []
    for index, event in enumerate(sequence.events):
        frame = event.frame if event.frame is not None else round(event.timestamp * fps)
        source_record_id = event.evidence_id or f"{sequence.attack_id}:event:{index}"
        actor_id = f"{event.player_id or 'unknown'}@legacy-{index}"
        converted.append(
            ObservedActionV2(
                match_id=resolved_match_id,
                period=period,
                frame=frame,
                timestamp_sec=event.timestamp,
                action_type=event.event_type,
                source_tier=SourceTier.LEGACY_UNVALIDATED,
                actor_team_id=event.team,
                possession_team_id=sequence.team,
                x=event.x,
                y=event.y,
                actor_id=actor_id,
                jersey=event.jersey,
                role=event.role,
                outcome=event.outcome,
                action_probability=event.confidence,
                field_provenance=(
                    FieldProvenanceV2(
                        field_name="action_type",
                        source_tier=SourceTier.LEGACY_UNVALIDATED,
                        source_record_id=source_record_id,
                        evidence_ids=(source_record_id,),
                        confidence=event.confidence,
                    ),
                ),
                evidence_ids=(source_record_id,),
            )
        )
    actions = tuple(sorted(converted, key=lambda item: (item.frame, item.timestamp_sec, item.action_id)))
    if not actions:
        raise ValueError("legacy sequence contains no actions")
    return PossessionSequenceV2(
        match_id=resolved_match_id,
        period=period,
        possession_team_id=sequence.team,
        start_frame=actions[0].frame,
        end_frame=actions[-1].frame,
        start_sec=actions[0].timestamp_sec,
        end_sec=actions[-1].timestamp_sec,
        actions=actions,
        status=SequenceStatus.UNCERTAIN,
        source_tier=SourceTier.LEGACY_UNVALIDATED,
        terminal=None,
        state_transitions=(
            PossessionStateTransitionV2(
                frame=actions[0].frame,
                timestamp_sec=actions[0].timestamp_sec,
                state=PossessionState.UNKNOWN,
                team_id=sequence.team,
                evidence_ids=(actions[0].action_id,),
            ),
        ),
        evidence_ids=tuple(item for action in actions for item in action.evidence_ids),
    )


__all__ = [
    "AttackV2DecodeResult",
    "FootpassAttackV2Config",
    "decode_footpass_attack_v2",
    "legacy_attack_to_possession_v2",
    "qualify_attack_v2",
]
