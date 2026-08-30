from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .models import (
    FieldProvenanceV2,
    ObservedActionV2,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SourceTier,
)


INFERENCE_VERSION = "footpass-terminal-inference-v1"
CONTROLLED_ACTION_TYPES = frozenset({"drive", "pass", "cross"})
RESTART_ACTION_TYPES = frozenset({"throw_in", "throw-in"})
ATTEMPT_CONTEXT_TYPES = frozenset({"shot", "cross", "header"})
RARE_FOOTPASS_CLASSES = ("shot", "header", "block")


@dataclass(frozen=True)
class FootpassTerminalInferenceConfig:
    """Neighborhood rules that decorate FOOTPASS labels without changing classes."""

    fps: float = 25.0
    event_cluster_sec: float = 0.50
    max_observation_gap_sec: float = 8.0
    attacking_third_x: float = 70.0
    penalty_box_x: float = PITCH_LENGTH - 16.5
    inference_version: str = INFERENCE_VERSION

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.event_cluster_sec < 0 or self.max_observation_gap_sec <= 0:
            raise ValueError("fps and gap thresholds must be positive")
        if not 0.0 <= self.attacking_third_x <= PITCH_LENGTH:
            raise ValueError("attacking_third_x must lie on the pitch")
        if not 0.0 <= self.penalty_box_x <= PITCH_LENGTH:
            raise ValueError("penalty_box_x must lie on the pitch")
        if not self.inference_version.strip():
            raise ValueError("inference_version must not be empty")


@dataclass(frozen=True)
class TerminalInferenceResult:
    actions: tuple[ObservedActionV2, ...]
    headed_promoted: int = 0
    blocks_promoted: int = 0
    headers_left_generic: int = 0
    blocks_left_generic: int = 0
    skipped_incomplete: int = 0
    inference_version: str = INFERENCE_VERSION

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "inference_version": self.inference_version,
            "headed_promoted": self.headed_promoted,
            "blocks_promoted": self.blocks_promoted,
            "headers_left_generic": self.headers_left_generic,
            "blocks_left_generic": self.blocks_left_generic,
            "skipped_incomplete": self.skipped_incomplete,
        }


def _unknown_replay(action: ObservedActionV2) -> bool:
    return "replay_status_unknown" in action.qualifiers


def _live(action: ObservedActionV2) -> bool:
    return not action.replay and action.action_type != "scene_cut"


def _evidence(action: ObservedActionV2) -> tuple[str, ...]:
    return action.evidence_ids or (action.action_id,)


def _upsert_provenance(
    action: ObservedActionV2,
    field_name: str,
    *,
    inference_id: str,
    evidence_ids: tuple[str, ...],
) -> tuple[FieldProvenanceV2, ...]:
    remaining = tuple(
        item for item in action.field_provenance if item.field_name != field_name
    )
    return (
        *remaining,
        FieldProvenanceV2(
            field_name=field_name,
            source_tier=(
                action.source_tier
                if action.source_tier is not SourceTier.FUSED
                else next(
                    (
                        item.source_tier
                        for item in action.field_provenance
                        if item.source_tier is not SourceTier.FUSED
                    ),
                    SourceTier.VIDEO_ONLY,
                )
            ),
            source_record_id=inference_id,
            evidence_ids=evidence_ids,
        ),
    )


def _enrich(
    action: ObservedActionV2,
    *,
    inference_kind: str,
    extra_evidence: Iterable[str],
    extra_qualifiers: tuple[str, ...] = (),
    body_part: str | None = None,
    beneficiary_team_id: str | None = None,
) -> ObservedActionV2:
    inference_id = f"{INFERENCE_VERSION}:{inference_kind}:{action.action_id}"
    evidence = tuple(dict.fromkeys((*_evidence(action), *extra_evidence, inference_id)))
    updated = action
    if extra_qualifiers:
        updated = replace(
            updated,
            qualifiers=tuple(dict.fromkeys((*updated.qualifiers, *extra_qualifiers))),
            field_provenance=_upsert_provenance(
                updated,
                "qualifiers",
                inference_id=inference_id,
                evidence_ids=evidence,
            ),
        )
    if body_part is not None:
        updated = replace(
            updated,
            body_part=body_part,
            field_provenance=_upsert_provenance(
                updated,
                "body_part",
                inference_id=inference_id,
                evidence_ids=evidence,
            ),
        )
    if beneficiary_team_id is not None:
        updated = replace(
            updated,
            beneficiary_team_id=beneficiary_team_id,
            field_provenance=_upsert_provenance(
                updated,
                "beneficiary_team_id",
                inference_id=inference_id,
                evidence_ids=evidence,
            ),
        )
    return replace(updated, evidence_ids=evidence)


def _in_attacking_third(action: ObservedActionV2, config: FootpassTerminalInferenceConfig) -> bool:
    if action.x is None:
        return False
    if action.x >= config.penalty_box_x:
        if action.y is None:
            return True
        box_half = 40.32 / 2.0
        centre = PITCH_WIDTH / 2.0
        return centre - box_half <= action.y <= centre + box_half or action.x >= config.attacking_third_x
    return action.x >= config.attacking_third_x


def _previous_same_team_cross(
    actions: tuple[ObservedActionV2, ...],
    index: int,
    config: FootpassTerminalInferenceConfig,
) -> ObservedActionV2 | None:
    current = actions[index]
    team_id = current.actor_team_id
    if not team_id:
        return None
    for previous in reversed(actions[:index]):
        gap = current.timestamp_sec - previous.timestamp_sec
        if gap > config.max_observation_gap_sec:
            break
        if not _live(previous):
            continue
        if previous.action_type == "cross" and previous.actor_team_id == team_id:
            return previous
    return None


def _following_attempt_context(
    actions: tuple[ObservedActionV2, ...],
    index: int,
    config: FootpassTerminalInferenceConfig,
) -> ObservedActionV2 | None:
    current = actions[index]
    team_id = current.actor_team_id
    for following in actions[index + 1 :]:
        gap = following.timestamp_sec - current.timestamp_sec
        if gap > config.event_cluster_sec:
            break
        if not _live(following) or _unknown_replay(following):
            continue
        if following.action_type == "shot" and following.actor_team_id == team_id:
            return following
        if following.action_type == "block" and following.actor_team_id and following.actor_team_id != team_id:
            return following
    return None


def _should_promote_header(
    actions: tuple[ObservedActionV2, ...],
    index: int,
    last_controlled_team: str | None,
    config: FootpassTerminalInferenceConfig,
) -> tuple[bool, tuple[str, ...]]:
    action = actions[index]
    if action.action_type != "header" or not action.actor_team_id:
        return False, ()
    if "headed_attempt" in action.qualifiers:
        return False, ()
    if not _live(action) or _unknown_replay(action):
        return False, ()
    evidence: list[str] = []
    previous_cross = _previous_same_team_cross(actions, index, config)
    if previous_cross is not None:
        evidence.extend(_evidence(previous_cross))
    following = _following_attempt_context(actions, index, config)
    if following is not None:
        evidence.extend(_evidence(following))
    location_ok = (
        last_controlled_team == action.actor_team_id
        and _in_attacking_third(action, config)
    )
    if previous_cross is None and following is None and not location_ok:
        return False, ()
    return True, tuple(evidence)


def _preceding_attacking_attempt(
    actions: tuple[ObservedActionV2, ...],
    index: int,
    config: FootpassTerminalInferenceConfig,
) -> ObservedActionV2 | None:
    block = actions[index]
    if not block.actor_team_id:
        return None
    for previous in reversed(actions[:index]):
        gap = block.timestamp_sec - previous.timestamp_sec
        if gap > config.event_cluster_sec:
            break
        if not _live(previous) or _unknown_replay(previous):
            continue
        if previous.action_type not in ATTEMPT_CONTEXT_TYPES:
            continue
        if previous.actor_team_id and previous.actor_team_id != block.actor_team_id:
            if previous.action_type in {"shot", "cross"}:
                return previous
            if previous.action_type == "header" and (
                "headed_attempt" in previous.qualifiers or _in_attacking_third(previous, config)
            ):
                return previous
    return None


def _next_attacking_throw_in(
    actions: tuple[ObservedActionV2, ...],
    index: int,
    attacking_team_id: str,
    config: FootpassTerminalInferenceConfig,
) -> ObservedActionV2 | None:
    current = actions[index]
    for following in actions[index + 1 :]:
        gap = following.timestamp_sec - current.timestamp_sec
        if gap > config.max_observation_gap_sec:
            return None
        if not _live(following):
            continue
        if _unknown_replay(following):
            return None
        if following.action_type in RESTART_ACTION_TYPES:
            restart_team = following.actor_team_id or following.beneficiary_team_id
            if restart_team == attacking_team_id:
                return following
            return None
        if following.action_type in CONTROLLED_ACTION_TYPES:
            return None
    return None


def infer_footpass_terminals(
    actions: Iterable[ObservedActionV2],
    config: FootpassTerminalInferenceConfig | None = None,
) -> TerminalInferenceResult:
    """Decorate instantaneous FOOTPASS labels with Attack-v2 terminal evidence.

    Raw classes stay `header`/`block`. The decoder still owns qualification.
    Incomplete replay/gap evidence is left generic rather than invented.
    """

    settings = config or FootpassTerminalInferenceConfig()
    ordered = tuple(
        sorted(
            actions,
            key=lambda item: (item.match_id, item.period, item.frame, item.timestamp_sec, item.action_id),
        )
    )
    updated = list(ordered)
    headed_promoted = blocks_promoted = 0
    headers_left = blocks_left = skipped = 0
    last_controlled_team: str | None = None
    previous_live: ObservedActionV2 | None = None

    for index, action in enumerate(ordered):
        if action.match_id != (previous_live.match_id if previous_live else action.match_id) or (
            previous_live is not None and action.period != previous_live.period
        ):
            last_controlled_team = None
            previous_live = None
        if (
            previous_live is not None
            and action.timestamp_sec - previous_live.timestamp_sec > settings.max_observation_gap_sec
        ):
            last_controlled_team = None

        if action.action_type == "header":
            if not _live(action) or _unknown_replay(action):
                skipped += 1
                headers_left += 1
            else:
                promote, evidence = _should_promote_header(
                    ordered, index, last_controlled_team, settings
                )
                if promote:
                    updated[index] = _enrich(
                        action,
                        inference_kind="headed_attempt",
                        extra_evidence=evidence,
                        extra_qualifiers=("headed_attempt",),
                        body_part="head",
                    )
                    headed_promoted += 1
                else:
                    headers_left += 1

        if action.action_type in CONTROLLED_ACTION_TYPES and _live(action) and action.actor_team_id:
            last_controlled_team = action.actor_team_id
        if _live(action):
            previous_live = action

    decorated = tuple(updated)
    last_controlled_team = None
    previous_live = None
    for index, action in enumerate(decorated):
        if action.match_id != (previous_live.match_id if previous_live else action.match_id) or (
            previous_live is not None and action.period != previous_live.period
        ):
            last_controlled_team = None
            previous_live = None
        if (
            previous_live is not None
            and action.timestamp_sec - previous_live.timestamp_sec > settings.max_observation_gap_sec
        ):
            last_controlled_team = None

        if action.action_type == "block":
            if "forced_goal_line_out" in action.qualifiers:
                pass
            elif not _live(action) or _unknown_replay(action) or not action.actor_team_id:
                skipped += int(action.action_type == "block")
                blocks_left += 1
            else:
                attempt = _preceding_attacking_attempt(decorated, index, settings)
                throw_in = (
                    _next_attacking_throw_in(
                        decorated,
                        index,
                        attempt.actor_team_id,
                        settings,
                    )
                    if attempt is not None and attempt.actor_team_id
                    else None
                )
                if attempt is not None and throw_in is not None and attempt.actor_team_id:
                    updated[index] = _enrich(
                        action,
                        inference_kind="forced_goal_line_out",
                        extra_evidence=(*_evidence(attempt), *_evidence(throw_in)),
                        extra_qualifiers=("forced_goal_line_out",),
                        beneficiary_team_id=attempt.actor_team_id,
                    )
                    blocks_promoted += 1
                else:
                    blocks_left += 1

        if action.action_type in CONTROLLED_ACTION_TYPES and _live(action) and action.actor_team_id:
            last_controlled_team = action.actor_team_id
        if _live(action):
            previous_live = action

    return TerminalInferenceResult(
        actions=tuple(updated),
        headed_promoted=headed_promoted,
        blocks_promoted=blocks_promoted,
        headers_left_generic=headers_left,
        blocks_left_generic=blocks_left,
        skipped_incomplete=skipped,
        inference_version=settings.inference_version,
    )


__all__ = [
    "CONTROLLED_ACTION_TYPES",
    "INFERENCE_VERSION",
    "RARE_FOOTPASS_CLASSES",
    "FootpassTerminalInferenceConfig",
    "TerminalInferenceResult",
    "infer_footpass_terminals",
]
