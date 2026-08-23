from __future__ import annotations

import json
from pathlib import Path

from .event_inference import EventInferenceConfig, infer_events_from_observations
from .geometry import PitchProjector
from .quality import analyze_observation_quality, read_observations_jsonl
from .roi import PolygonROI
from .scoring import evaluate_attack
from .segmentation import segment_attacks
from .vision_pipeline import VisionConfig, annotate_video


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_video_pipeline(
    input_path: Path,
    run_directory: Path,
    vision_config: VisionConfig | None = None,
    projector: PitchProjector | None = None,
    roi: PolygonROI | None = None,
    event_config: EventInferenceConfig | None = None,
) -> dict[str, object]:
    run_directory.mkdir(parents=True, exist_ok=True)
    annotated_path = run_directory / "annotated.mp4"
    observations_path = run_directory / "observations.jsonl"
    quality_path = run_directory / "quality.json"
    events_path = run_directory / "events.json"
    attacks_path = run_directory / "attacks.json"
    manifest_path = run_directory / "manifest.json"

    vision_summary = annotate_video(
        input_path,
        annotated_path,
        vision_config,
        projector,
        observations_path,
        roi,
    )
    frames = read_observations_jsonl(observations_path)
    quality = analyze_observation_quality(frames)
    _write_json(quality_path, quality.to_dict())

    manifest: dict[str, object] = {
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

    skip_reason: str | None = None
    if quality.status == "fail":
        skip_reason = "quality_report_failed"
    elif projector is None:
        skip_reason = "calibration_required_for_event_inference"

    if skip_reason is None:
        try:
            events = infer_events_from_observations(frames, event_config)
            event_payload = {
                "source": str(observations_path.resolve()),
                "events": [event.to_dict() for event in events],
            }
            _write_json(events_path, event_payload)

            attacks = segment_attacks(events)
            attack_payload = {
                "source": str(events_path.resolve()),
                "attack_count": len(attacks),
                "attacks": [
                    {
                        "attack": attack.to_dict(),
                        "evaluation": evaluate_attack(attack).to_dict(),
                    }
                    for attack in attacks
                ],
            }
            _write_json(attacks_path, attack_payload)
            manifest["status"] = "analysis_complete"
            manifest["event_count"] = len(events)
            manifest["attack_count"] = len(attacks)
            manifest["outputs"].update(
                {
                    "events": str(events_path.resolve()),
                    "attacks": str(attacks_path.resolve()),
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
    _write_json(manifest_path, manifest)
    return manifest

