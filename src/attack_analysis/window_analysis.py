from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AttackSequence
from .provenance import (
    MANIFEST_VERSION,
    StaleArtifactError,
    file_sha256,
    mark_legacy_manifest,
)


_TERMINAL_TYPES = {
    "shot": "shot_attempt",
    "shot_attempt": "shot_attempt",
    "headed_attempt": "headed_attempt",
    "forced_goal_line_out": "forced_goal_line_out",
}
_SHOT_TERMINALS = {"shot_attempt", "headed_attempt"}
_BOUNDARY_EPSILON_SEC = 1e-6
_LEGACY_VALIDATION_STATUS = "legacy_unvalidated"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _terminal_type(sequence: AttackSequence, events: tuple[Any, ...]) -> str | None:
    if not events:
        return None
    final_event = events[-1]
    terminal = _TERMINAL_TYPES.get(str(final_event.event_type).casefold())
    if terminal is not None:
        return terminal
    outcome = str(final_event.outcome or "").casefold()
    if final_event.event_type == "header" and outcome in {"headed_attempt", "shot_attempt"}:
        return "headed_attempt"
    reason = str(sequence.termination_reason or "").casefold()
    return _TERMINAL_TYPES.get(reason) if final_event.timestamp >= sequence.end_sec - 0.08 else None


def _source_tier(source: str) -> str:
    normalized = source.casefold()
    if "fused" in normalized:
        return "fused"
    if "oracle" in normalized:
        return "oracle_actions_oracle_tactical"
    if "dst" in normalized or "predicted" in normalized:
        return "predicted_actions_oracle_tactical"
    return "video_only"


def _terminal_evidence(
    sequence: AttackSequence,
    events: tuple[Any, ...],
) -> tuple[str | None, bool, tuple[str, ...]]:
    """Audit a legacy terminal without promoting it to the v2 contract.

    ``Event`` has no independent beneficiary, last-touch team or causal evidence
    fields.  Consequently a legacy ``forced_goal_line_out`` can be retained as
    raw evidence but can never satisfy the Attack-v2 terminal contract.  Shot
    labels can be recognized for boundary inspection, but the surrounding
    ``AttackSequence`` still remains ``legacy_unvalidated``.
    """

    terminal_type = _terminal_type(sequence, events)
    if terminal_type is None:
        return None, False, ("allowlisted_terminal",)
    if terminal_type == "forced_goal_line_out":
        return terminal_type, False, (
            "beneficiary_team_id",
            "last_touch_team_id",
            "cause",
        )
    return terminal_type, terminal_type in _SHOT_TERMINALS, ()


def _window_sequence(
    sequence: AttackSequence,
    events: tuple[Any, ...],
    *,
    start_sec: float,
    end_sec: float,
) -> AttackSequence | None:
    if not events:
        return None
    event_frames = [event.frame for event in events if event.frame is not None]
    have_all_frames = len(event_frames) == len(events)
    return AttackSequence(
        attack_id=sequence.attack_id,
        match_id=sequence.match_id,
        team=sequence.team,
        start_sec=max(sequence.start_sec, start_sec),
        end_sec=min(sequence.end_sec, end_sec),
        start_frame=min(event_frames) if have_all_frames else None,
        end_frame=max(event_frames) if have_all_frames else None,
        termination_reason=(
            sequence.termination_reason
            if end_sec >= sequence.end_sec - _BOUNDARY_EPSILON_SEC
            else None
        ),
        source=sequence.source,
        events=events,
    )


def persist_footpass_window_stream(
    manifest: dict[str, Any],
    sequence: AttackSequence,
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    """Persist a bounded *legacy* FOOTPASS possession candidate.

    The input type is the V1 ``AttackSequence`` compatibility contract.  This
    function therefore never emits an Attack-v2 object, attack qualification,
    xG evaluation or effectiveness verdict, even when the final raw event is a
    shot.  Consumers that need validated attacks must decode
    ``ObservedActionV2`` into ``PossessionSequenceV2``/``AttackSequenceV2``.
    """
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("window end must be greater than its non-negative start")
    if manifest.get("provenance_status") == "stale":
        raise StaleArtifactError("cannot derive a window analysis from a stale manifest")
    run_directory_value = manifest.get("run_directory")
    if not run_directory_value:
        raise ValueError("vision manifest does not declare a run directory")
    run_directory = Path(str(run_directory_value))
    run_directory.mkdir(parents=True, exist_ok=True)

    events = tuple(
        event
        for event in sequence.events
        if start_sec <= event.timestamp <= end_sec
    )
    filtered_sequence = _window_sequence(
        sequence,
        events,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    left_censored = start_sec > sequence.start_sec + _BOUNDARY_EPSILON_SEC
    right_censored = end_sec < sequence.end_sec - _BOUNDARY_EPSILON_SEC
    if right_censored:
        terminal_type, terminal_evidence_complete, terminal_missing_fields = (
            None,
            False,
            ("terminal_inside_window",),
        )
    else:
        terminal_type, terminal_evidence_complete, terminal_missing_fields = (
            _terminal_evidence(sequence, events)
        )
    terminal_allowed = terminal_evidence_complete
    boundaries_complete = not left_censored and not right_censored
    # A V1 segment cannot establish the v2 possession start/terminal/aftermath
    # contract.  ``boundary_status`` only describes window coverage; it is not
    # an Attack-v2 completeness assertion.
    if left_censored and right_censored:
        boundary_status = "uncertain"
    elif left_censored:
        boundary_status = "left_censored"
    elif right_censored:
        boundary_status = "right_censored"
    elif terminal_type is not None:
        boundary_status = "complete"
    else:
        boundary_status = "uncertain"

    shots = tuple(
        event
        for event in events
        if _TERMINAL_TYPES.get(str(event.event_type).casefold()) in _SHOT_TERMINALS
        or (
            event.event_type == "header"
            and str(event.outcome or "").casefold() in {"headed_attempt", "shot_attempt"}
        )
    )
    calibration = dict(manifest.get("calibration") or {})
    claimed_source_tier = _source_tier(sequence.source)
    source_is_audited = claimed_source_tier in {
        "oracle_actions_oracle_tactical",
        "predicted_actions_oracle_tactical",
        "fused",
    }
    gates = {
        "dynamic_calibration": calibration.get("status") == "pass",
        "action_source": bool(events) and source_is_audited,
        "start_boundary": not left_censored,
        "end_boundary": not right_censored,
        "allowlisted_terminal": terminal_allowed if boundaries_complete else None,
        "terminal_evidence_complete": (
            terminal_evidence_complete if boundaries_complete else None
        ),
        "mandatory_aftermath_evidence": None,
        "v2_contract": False,
        "trained_xg": None,
        "attack_qualified": False if boundaries_complete else None,
    }
    if left_censored or right_censored:
        status = boundary_status
    else:
        status = _LEGACY_VALIDATION_STATUS
    event_stream_path = run_directory / "structured-event-stream.json"
    candidate_analysis_path = run_directory / "selected-possession-candidate-v1.json"
    candidate_manifest_path = run_directory / "possession-candidate-v1-manifest.json"
    event_payload = {
        "schema_version": "footpass-window-event-stream-legacy-v1",
        "manifest_version": MANIFEST_VERSION,
        "object_type": "PossessionCandidateV1",
        "validation_status": _LEGACY_VALIDATION_STATUS,
        "status": status,
        "boundary_status": boundary_status,
        "source_tier": _LEGACY_VALIDATION_STATUS,
        "claimed_upstream_source_tier": claimed_source_tier,
        "provenance_status": _LEGACY_VALIDATION_STATUS,
        "derived_from_artifact_identity": dict(manifest.get("provenance") or {}).get(
            "artifact_identity_sha256"
        ),
        "match_id": sequence.match_id,
        "candidate_id": sequence.attack_id,
        "legacy_attack_id": sequence.attack_id,
        "source": sequence.source,
        "window": {
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
            "start_frame": (manifest.get("vision") or {}).get("start_frame"),
            "end_frame": (manifest.get("vision") or {}).get("end_frame"),
        },
        "quality_gates": gates,
        "censoring": {
            "left_censored": left_censored,
            "right_censored": right_censored,
            "start_boundary_present": not left_censored,
            "end_boundary_present": not right_censored,
        },
        "terminal": {
            "type": terminal_type,
            "allowlisted": terminal_allowed,
            "evidence_complete": terminal_evidence_complete,
            "missing_fields": list(terminal_missing_fields),
        },
        "event_count": len(events),
        "mean_event_confidence": (
            round(sum(event.confidence for event in events) / len(events), 4)
            if events
            else 0.0
        ),
        "events": [event.to_dict() for event in events],
        "xg_status": "withheld_legacy_unvalidated",
        "xg_records": [],
        "legacy_xg_evidence_count": sum(event.xg is not None for event in shots),
    }
    analysis_payload = {
        "schema_version": "possession-candidate-window-v1",
        "manifest_version": MANIFEST_VERSION,
        "object_type": "PossessionCandidateV1",
        "validation_status": _LEGACY_VALIDATION_STATUS,
        "status": status,
        "boundary_status": boundary_status,
        "source_tier": _LEGACY_VALIDATION_STATUS,
        "claimed_upstream_source_tier": claimed_source_tier,
        "provenance_status": _LEGACY_VALIDATION_STATUS,
        "derived_from_artifact_identity": dict(manifest.get("provenance") or {}).get(
            "artifact_identity_sha256"
        ),
        "possession_candidate_v1": (
            filtered_sequence.to_dict() if filtered_sequence is not None else None
        ),
        "attack_sequence_v2": None,
        "possession_assessment": None,
        "deterministic_baseline": None,
        "xg_v2": None,
        "effectiveness_verdict": None,
        "verdict_status": "withheld_legacy_unvalidated",
        "event_stream_source": str(event_stream_path.resolve()),
    }
    _write_json(event_stream_path, event_payload)
    _write_json(candidate_analysis_path, analysis_payload)

    result = mark_legacy_manifest(manifest)
    result["upstream_provenance_status"] = manifest.get("provenance_status")
    result["provenance_status"] = _LEGACY_VALIDATION_STATUS
    result["validation_status"] = _LEGACY_VALIDATION_STATUS
    result["legacy_unvalidated"] = True
    result["upstream_source_tier"] = result.get("source_tier")
    result["analysis_source_tier"] = _LEGACY_VALIDATION_STATUS
    result["claimed_analysis_source_tier"] = claimed_source_tier
    outputs = dict(result.get("outputs") or {})
    upstream_manifest_path = outputs.get("manifest")
    if upstream_manifest_path:
        outputs["upstream_manifest"] = str(upstream_manifest_path)
    outputs["manifest"] = str(candidate_manifest_path.resolve())
    outputs["structured_event_stream"] = str(event_stream_path.resolve())
    outputs["selected_possession_candidate_v1"] = str(candidate_analysis_path.resolve())
    outputs.pop("selected_attack_analysis", None)
    result["outputs"] = outputs
    output_sha256 = dict(result.get("output_sha256") or {})
    output_sha256["structured_event_stream"] = file_sha256(event_stream_path)
    output_sha256["selected_possession_candidate_v1"] = file_sha256(
        candidate_analysis_path
    )
    output_sha256.pop("selected_attack_analysis", None)
    result["output_sha256"] = output_sha256
    result["structured_event_stream"] = {
        "status": status,
        "boundary_status": boundary_status,
        "object_type": "PossessionCandidateV1",
        "validation_status": _LEGACY_VALIDATION_STATUS,
        "source": sequence.source,
        "source_tier": _LEGACY_VALIDATION_STATUS,
        "claimed_upstream_source_tier": claimed_source_tier,
        "event_count": len(events),
        "shot_count": len(shots),
        "quality_gates": gates,
    }

    # Never overwrite the validated upstream vision manifest with a legacy V1
    # derivation.  The sidecar has its own manifest and is rejected by v2 cache
    # validation through ``provenance_status=legacy_unvalidated``.
    _write_json(candidate_manifest_path, result)
    return result
