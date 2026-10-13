from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Literal, Mapping

from .attack_v2 import AttackV2DecodeResult, FootpassAttackV2Config, decode_footpass_attack_v2
from .models import Event, FieldProvenanceV2, ObservedActionV2, SourceTier


def event_to_observed_action_v2(
    event: Event,
    *,
    match_id: str,
    period: int = 1,
    fps: float = 25.0,
    source_tier: SourceTier = SourceTier.VIDEO_ONLY,
    team_mapping: Mapping[str, str] | None = None,
    replay_status: Literal["live", "replay", "unknown"] = "unknown",
    index: int = 0,
    team_left_to_right: Mapping[str, bool] | None = None,
) -> ObservedActionV2:
    """Convert a legacy event without promoting its xG or heuristic semantics.

    This adapter is intended for the current video event stream.  An Event xG
    value is deliberately ignored: chance quality belongs to a separately
    confirmed AttackSequenceV2 terminal, not an instantaneous candidate.
    """

    if not match_id.strip():
        raise ValueError("match_id must not be empty")
    if period < 1 or fps <= 0:
        raise ValueError("period and fps must be positive")
    if source_tier is SourceTier.FUSED:
        raise ValueError("generic Event conversion cannot establish field-level fused association")
    if source_tier is SourceTier.LEGACY_UNVALIDATED:
        raise ValueError("use legacy_attack_to_possession_v2 for V1 compatibility artifacts")
    if replay_status not in {"live", "replay", "unknown"}:
        raise ValueError("replay_status must be live, replay, or unknown")

    frame = event.frame if event.frame is not None else round(event.timestamp * fps)
    actor_team_id = (team_mapping or {}).get(event.team, event.team)
    record_id = event.evidence_id or f"event:{match_id}:{period}:{frame}:{index}"
    provenance_fields = ["frame", "timestamp_sec", "action_type", "actor_team_id", "replay"]
    optional_values = {
        "x": event.x,
        "y": event.y,
        "actor_id": event.player_id,
        "jersey": event.jersey,
        "role": event.role,
        "outcome": event.outcome,
        "raw_left_to_right": (team_left_to_right or {}).get(event.team),
    }
    provenance_fields.extend(name for name, value in optional_values.items() if value is not None)
    provenance = tuple(
        FieldProvenanceV2(
            field_name=name,
            source_tier=source_tier,
            source_record_id=record_id,
            evidence_ids=(record_id,),
            confidence=event.confidence,
        )
        for name in provenance_fields
    )
    return ObservedActionV2(
        match_id=match_id,
        period=period,
        frame=frame,
        timestamp_sec=event.timestamp,
        action_type=event.event_type,
        source_tier=source_tier,
        actor_team_id=actor_team_id,
        raw_left_to_right=(team_left_to_right or {}).get(event.team),
        x=event.x,
        y=event.y,
        actor_id=event.player_id,
        jersey=event.jersey,
        role=event.role,
        replay=replay_status == "replay",
        outcome=event.outcome,
        qualifiers=(
            ("replay_status_unknown",)
            if replay_status == "unknown"
            else ("replay_verified",)
            if replay_status == "replay"
            else ("live_verified",)
        ),
        action_probability=event.confidence,
        field_provenance=provenance,
        evidence_ids=(record_id,),
    )


def event_stream_to_observed_actions_v2(
    events: Iterable[Event],
    *,
    match_id: str,
    period: int = 1,
    fps: float = 25.0,
    source_tier: SourceTier = SourceTier.VIDEO_ONLY,
    team_mapping: Mapping[str, str] | None = None,
    replay_status: Literal["live", "replay", "unknown"] = "unknown",
    scene_cut_frames: Iterable[int] = (),
    team_left_to_right: Mapping[str, bool] | None = None,
) -> tuple[ObservedActionV2, ...]:
    actions = tuple(
        event_to_observed_action_v2(
            event,
            match_id=match_id,
            period=period,
            fps=fps,
            source_tier=source_tier,
            team_mapping=team_mapping,
            replay_status=replay_status,
            index=index,
            team_left_to_right=team_left_to_right,
        )
        for index, event in enumerate(events)
    )
    scene_actions = tuple(
        ObservedActionV2(
            match_id=match_id,
            period=period,
            frame=int(frame),
            timestamp_sec=int(frame) / fps,
            action_type="scene_cut",
            source_tier=source_tier,
            replay=False,
            qualifiers=("scene_cut_boundary",),
            field_provenance=tuple(
                FieldProvenanceV2(
                    field_name=name,
                    source_tier=source_tier,
                    source_record_id=f"scene-cut:{match_id}:{period}:{int(frame)}",
                    evidence_ids=(f"scene-cut:{match_id}:{period}:{int(frame)}",),
                )
                for name in ("frame", "timestamp_sec", "action_type", "replay")
            ),
            evidence_ids=(f"scene-cut:{match_id}:{period}:{int(frame)}",),
        )
        for frame in sorted({int(value) for value in scene_cut_frames})
        if int(frame) >= 0
    )
    return tuple(
        sorted(
            (*actions, *scene_actions),
            key=lambda item: (item.frame, item.timestamp_sec, item.action_id),
        )
    )


def sequences_from_video_events(
    events: Iterable[Event],
    *,
    match_id: str,
    period: int = 1,
    fps: float = 25.0,
    scene_cut_frames: Iterable[int] = (),
) -> tuple[tuple[ObservedActionV2, ...], AttackV2DecodeResult]:
    """Turn an uploaded clip's event list into possessions and attacks.

    ``shot_candidate`` is trajectory evidence. For this exploratory path it is
    decoded as a shot so a fast ball into the box can be selected as an attack.
    The resulting attack stays ``video_only`` and is not a FOOTPASS label.
    """

    prepared = tuple(
        replace(event, event_type="shot") if event.event_type == "shot_candidate" else event
        for event in events
    )
    return decode_event_stream_v2(
        prepared,
        match_id=match_id,
        period=period,
        fps=fps,
        source_tier=SourceTier.VIDEO_ONLY,
        replay_status="live",
        scene_cut_frames=scene_cut_frames,
    )


def decode_event_stream_v2(
    events: Iterable[Event],
    *,
    match_id: str,
    period: int = 1,
    fps: float = 25.0,
    source_tier: SourceTier = SourceTier.VIDEO_ONLY,
    team_mapping: Mapping[str, str] | None = None,
    replay_status: Literal["live", "replay", "unknown"] = "unknown",
    scene_cut_frames: Iterable[int] = (),
    config: FootpassAttackV2Config | None = None,
    team_left_to_right: Mapping[str, bool] | None = None,
) -> tuple[tuple[ObservedActionV2, ...], AttackV2DecodeResult]:
    actions = event_stream_to_observed_actions_v2(
        events,
        match_id=match_id,
        period=period,
        fps=fps,
        source_tier=source_tier,
        team_mapping=team_mapping,
        replay_status=replay_status,
        scene_cut_frames=scene_cut_frames,
        team_left_to_right=team_left_to_right,
    )
    settings = config or FootpassAttackV2Config(
        fps=fps,
        controlled_action_types=frozenset(
            {"possession_start", "carry", "drive", "pass", "cross"}
        ),
    )
    return actions, decode_footpass_attack_v2(actions, settings)


__all__ = [
    "decode_event_stream_v2",
    "event_stream_to_observed_actions_v2",
    "event_to_observed_action_v2",
    "sequences_from_video_events",
]
