from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .event_inference import EventInferenceConfig, infer_events_from_observations
from .geometry import Projector
from .models import PossessionValueV2, SourceTier
from .provenance import (
    MANIFEST_VERSION,
    build_provenance,
    content_sha256,
    file_sha256,
    validate_source_tier,
)
from .quality import (
    analyze_observation_quality,
    read_observations_jsonl,
    track_birth_limit_per_100_frames,
)
from .roi import PolygonROI
from .segmentation import segment_attacks
from .v2_adapters import decode_event_stream_v2
from .vision_pipeline import VisionConfig, annotate_video


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _model_fingerprint(reference: str | Path | None) -> str:
    if reference is None:
        return content_sha256(None)
    path = Path(reference)
    return file_sha256(path) if path.is_file() else content_sha256(str(reference))


def _source_tier_value(source_tier: SourceTier | str) -> str:
    value = source_tier.value if isinstance(source_tier, SourceTier) else str(source_tier)
    return validate_source_tier(value)


def _projector_assisted_source(projector: Projector | None) -> Path | None:
    """Return an external calibration source declared by an assisted projector."""

    report = getattr(projector, "report", None)
    if report is None:
        return None
    source = report.get("source") if isinstance(report, dict) else getattr(report, "source", None)
    if source is None or not str(source).strip():
        return None
    return Path(str(source)).resolve()


def validate_video_pipeline_source_contract(
    *,
    source_tier: str,
    projector: Projector | None,
    dataset_paths: tuple[Path, ...],
) -> None:
    """Prevent assisted inputs from being relabelled as video-only or fused."""

    assisted_source = _projector_assisted_source(projector)
    resolved_datasets = {path.resolve() for path in dataset_paths}
    if source_tier == SourceTier.FUSED.value:
        raise ValueError("run_video_pipeline has no field-level association stage and cannot claim fused")
    if source_tier == SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL.value:
        raise ValueError("video event inference produces predicted actions, not oracle actions")
    if source_tier == SourceTier.VIDEO_ONLY.value:
        if assisted_source is not None:
            raise ValueError("an assisted projector cannot be rendered as video_only")
        if dataset_paths:
            raise ValueError("video_only inference cannot consume oracle tactical datasets")
        return
    if source_tier == SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL.value:
        if projector is None:
            raise ValueError("predicted_actions_oracle_tactical requires an assisted projector")
        if not dataset_paths:
            raise ValueError("predicted_actions_oracle_tactical requires hashed dataset_paths")
        if assisted_source is not None and assisted_source not in resolved_datasets:
            raise ValueError("the assisted projector source must be included in dataset_paths")
        return
    raise ValueError(f"unsupported source tier: {source_tier}")


def _possession_value(possession: Any) -> PossessionValueV2:
    """Create an abstaining value record until a true ball/control trajectory is supplied.

    ``ObservedActionV2.x/y`` can be an actor/contact coordinate (FOOTPASS in
    particular stores the actor position).  Using the first and last action as
    if they were a ball trajectory manufactures progression and box entries, so
    both fields deliberately remain unknown in this pipeline.
    """

    return PossessionValueV2(
        possession_id=possession.possession_id,
        source_tier=possession.source_tier,
        # xT/EPV and short-horizon probabilities remain unknown until a
        # trajectory-trained value model is supplied.
        evidence_ids=possession.evidence_ids,
    )


def run_video_pipeline(
    input_path: Path,
    run_directory: Path,
    vision_config: VisionConfig | None = None,
    projector: Projector | None = None,
    roi: PolygonROI | None = None,
    event_config: EventInferenceConfig | None = None,
    xg_model_path: Path | None = None,
    *,
    source_tier: SourceTier | str = SourceTier.VIDEO_ONLY,
    dataset_paths: Iterable[Path] = (),
) -> dict[str, object]:
    resolved_source_tier = _source_tier_value(source_tier)
    resolved_dataset_paths = tuple(Path(path) for path in dataset_paths)
    validate_video_pipeline_source_contract(
        source_tier=resolved_source_tier,
        projector=projector,
        dataset_paths=resolved_dataset_paths,
    )
    model_source_tier = SourceTier(resolved_source_tier)
    settings = vision_config or VisionConfig()
    event_settings = event_config or EventInferenceConfig()
    run_directory.mkdir(parents=True, exist_ok=True)
    annotated_path = run_directory / "annotated.mp4"
    observations_path = run_directory / "observations.jsonl"
    quality_path = run_directory / "quality.json"
    events_path = run_directory / "events.json"
    sequences_path = run_directory / "sequences-v2.json"
    legacy_candidates_path = run_directory / "possession-candidates-v1.json"
    manifest_path = run_directory / "manifest.json"
    calibration_path = run_directory / "calibration.json"

    vision_summary = annotate_video(
        input_path,
        annotated_path,
        settings,
        projector,
        observations_path,
        roi,
    )
    frames = read_observations_jsonl(observations_path)
    quality = analyze_observation_quality(frames)
    _write_json(quality_path, quality.to_dict())

    provenance = build_provenance(
        source_tier=resolved_source_tier,
        video_sha256=file_sha256(input_path),
        config={"vision": settings, "event_inference": event_settings},
        model_fingerprints={
            "pose": _model_fingerprint(settings.pose_model),
            "detection": _model_fingerprint(settings.detection_model),
            "ball_detection": _model_fingerprint(settings.ball_detection_model),
            "xg_requested_but_withheld": _model_fingerprint(xg_model_path),
        },
        schema_versions={
            "run": "attack-run-v2.0",
            "domain": "2.0",
            "legacy_candidates": "PossessionCandidateV1",
        },
        postprocess={
            "decoder": "deterministic-terminal-first-v2",
            "xg_policy": "confirmed-terminal-only",
        },
        dataset_paths=resolved_dataset_paths,
        content_root=Path(__file__).resolve().parent,
    )
    manifest: dict[str, object] = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": "attack-run-v2.0",
        "source_tier": resolved_source_tier,
        "provenance": provenance,
        "provenance_status": "validated",
        "release_status": "research_only",
        "status": "vision_complete",
        "input_path": str(input_path.resolve()),
        "run_directory": str(run_directory.resolve()),
        "vision": vision_summary.to_dict(),
        "quality": quality.to_dict(),
        "outputs": {
            "annotated_video": str(annotated_path.resolve()),
            "observations": str(observations_path.resolve()),
            "quality_report": str(quality_path.resolve()),
        },
    }

    calibration_report = getattr(projector, "report", None)
    if calibration_report is not None:
        calibration_payload = calibration_report.to_dict()
        _write_json(calibration_path, calibration_payload)
        manifest["calibration"] = calibration_payload
        manifest["outputs"]["calibration_report"] = str(calibration_path.resolve())

    skip_reason: str | None = None
    ball_birth_limit = track_birth_limit_per_100_frames(quality.frame_count)
    if quality.status == "fail":
        skip_reason = "quality_report_failed"
    elif projector is None:
        skip_reason = "calibration_required_for_event_inference"
    elif calibration_report is not None and calibration_report.status != "pass":
        skip_reason = "dynamic_calibration_quality_gate_failed"
    elif quality.calibrated_player_coverage < 0.8:
        skip_reason = "calibrated_player_coverage_below_0.8"
    elif quality.calibrated_ball_coverage < 0.8:
        skip_reason = "calibrated_ball_coverage_below_0.8"
    elif quality.frames_with_ball_ratio < 0.5:
        skip_reason = "ball_frame_coverage_below_0.5"
    elif quality.ball_tracking_coverage < 0.5:
        skip_reason = "ball_tracking_coverage_below_0.5"
    elif quality.ball_track_continuity < 0.6:
        skip_reason = "ball_track_continuity_below_0.6"
    elif quality.ball_single_frame_track_ratio > 0.3:
        skip_reason = "ball_track_fragmentation_proxy_above_0.3"
    elif quality.ball_track_births_per_100_frames > ball_birth_limit:
        skip_reason = "ball_track_birth_rate_above_limit"
    elif quality.player_tracking_coverage < 0.8:
        skip_reason = "player_tracking_coverage_below_0.8"
    elif quality.player_single_frame_track_ratio > 0.75:
        skip_reason = "player_track_fragmentation_proxy_above_0.75"
    elif quality.team_assignment_coverage < 0.7:
        skip_reason = "team_assignment_coverage_below_0.7"
    elif quality.team_track_consistency < 0.9:
        skip_reason = "team_track_consistency_below_0.9"
    elif quality.player_speed_p95 is not None and quality.player_speed_p95 > 45:
        skip_reason = "implausible_player_speed_p95"

    manifest["event_quality_gates"] = {
        "calibration": projector is not None
        and (calibration_report is None or calibration_report.status == "pass")
        and quality.calibrated_player_coverage >= 0.8
        and quality.calibrated_ball_coverage >= 0.8,
        "ball_detection": quality.frames_with_ball_ratio >= 0.5,
        "ball_tracking": quality.ball_tracking_coverage >= 0.5
        and quality.ball_track_continuity >= 0.6
        and quality.ball_single_frame_track_ratio <= 0.3
        and quality.ball_track_births_per_100_frames <= ball_birth_limit,
        "player_tracking": quality.player_tracking_coverage >= 0.8
        and quality.player_single_frame_track_ratio <= 0.75,
        "team_assignment": quality.team_assignment_coverage >= 0.7
        and quality.team_track_consistency >= 0.9,
        "physical_speed": quality.player_speed_p95 is None or quality.player_speed_p95 <= 45,
        "thresholds": {
            "ball_single_frame_track_ratio_max": 0.3,
            "ball_track_births_per_100_frames_max": ball_birth_limit,
        },
    }

    if skip_reason is None:
        try:
            events = infer_events_from_observations(frames, event_settings)
            event_payload = {
                "source": str(observations_path.resolve()),
                "source_tier": resolved_source_tier,
                "provenance": provenance,
                "candidate_semantics": (
                    "shot_candidate is trajectory evidence only and cannot qualify an attack"
                ),
                "events": [
                    {
                        **event.to_dict(),
                        "source_tier": resolved_source_tier,
                        "provenance_artifact_id": provenance["artifact_identity_sha256"],
                    }
                    for event in events
                ],
            }
            _write_json(events_path, event_payload)

            observed_actions, decoded = decode_event_stream_v2(
                events,
                match_id=input_path.stem,
                period=1,
                fps=vision_summary.fps,
                source_tier=model_source_tier,
                replay_status="unknown",
                scene_cut_frames=(
                    frame.frame_index for frame in frames if frame.scene_cut
                ),
            )
            values = tuple(_possession_value(item) for item in decoded.possessions)
            sequence_payload = {
                "schema_version": "attack-v2-run/2.0",
                "source_tier": resolved_source_tier,
                "source": str(events_path.resolve()),
                "provenance": provenance,
                "observed_actions": [item.to_dict() for item in observed_actions],
                **decoded.to_dict(),
                "possession_values": [item.to_dict() for item in values],
                "xg": {
                    "status": "withheld",
                    "reason": "no independently confirmed v2 shot terminal",
                },
            }
            _write_json(sequences_path, sequence_payload)

            legacy_candidates = segment_attacks(events)
            _write_json(
                legacy_candidates_path,
                {
                    "schema_version": "PossessionCandidateV1",
                    "validation_status": "legacy_unvalidated",
                    "warning": "heuristic candidates are not attacks and receive no verdict",
                    "possession_candidates": [item.to_dict() for item in legacy_candidates],
                },
            )
            manifest["status"] = "analysis_complete"
            manifest["event_count"] = len(events)
            manifest["possession_count"] = len(decoded.possessions)
            manifest["attack_count"] = len(decoded.attacks)
            manifest["legacy_possession_candidate_count"] = len(legacy_candidates)
            manifest["xg_status"] = "withheld"
            if xg_model_path is not None:
                manifest["xg_warning"] = (
                    "requested xG-v1 model was fingerprinted but not applied before terminal confirmation"
                )
            manifest["outputs"].update(
                {
                    "events": str(events_path.resolve()),
                    "sequences_v2": str(sequences_path.resolve()),
                    "possession_candidates_v1": str(legacy_candidates_path.resolve()),
                }
            )
        except ValueError as error:
            skip_reason = str(error)

    if skip_reason is not None:
        manifest["analysis_status"] = "skipped"
        manifest["analysis_skip_reason"] = skip_reason
    else:
        manifest["analysis_status"] = "complete"

    manifest["outputs"]["manifest"] = str(manifest_path.resolve())
    manifest["output_sha256"] = {
        name: file_sha256(Path(str(path)))
        for name, path in manifest["outputs"].items()
        if name != "manifest" and Path(str(path)).is_file()
    }
    _write_json(manifest_path, manifest)
    return manifest
