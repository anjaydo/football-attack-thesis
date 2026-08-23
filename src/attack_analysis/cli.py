from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .event_inference import EventInferenceConfig, infer_events_from_observations
from .detection_evaluation import (
    DEFAULT_BALL_THRESHOLDS,
    benchmark_ball_jsonl,
    evaluate_soccernet_jsonl,
)
from .environment import environment_report
from .field_geometry import FieldLineConfig, analyze_field_image
from .annotations import build_annotation_manifest, stratified_annotation_sample
from .attack_features import build_attack_features
from .agent_tools import EvidenceToolRegistry
from .benchmarks import build_footpass_role_benchmarks
from .footpass import (
    FootpassSegmentationConfig,
    footpass_match_keys,
    footpass_summary,
    load_footpass_events,
    segment_footpass_attacks,
)
from .research_evaluation import action_spotting_dataset_report
from .geometry import PitchProjector
from .llm import DeterministicEvidenceClient
from .ml import train_effectiveness_models, train_xg_models
from .models import AttackSequence, Event
from .pipeline import run_video_pipeline
from .quality import analyze_observation_quality, read_observations_jsonl
from .roi import PolygonROI
from .research import EvidenceBundle
from .scoring import evaluate_attack
from .segmentation import SegmentationConfig, segment_attacks
from .soccernet import analyze_gamestate_sequence, download_gamestate_sequence
from .training import train_ball_model
from .video import inspect_video
from .vision_pipeline import VisionConfig, annotate_video
from .yolo_dataset import export_soccernet_ball_dataset


COMMANDS = {
    "evaluate",
    "analyze-events",
    "video-info",
    "annotate-video",
    "quality-report",
    "infer-events",
    "run-pipeline",
    "analyze-clip",
    "doctor",
    "soccernet-download",
    "soccernet-analyze",
    "field-geometry",
    "evaluate-observations",
    "benchmark-ball",
    "soccernet-export-yolo",
    "train-ball-model",
    "footpass-ingest",
    "evaluate-footpass-actions",
    "build-annotations",
    "train-xg",
    "train-effectiveness",
    "evaluate-evidence",
    "serve-demo",
    "build-footpass-benchmarks",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate football attacks from structured events."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one pre-segmented attack JSON file"
    )
    evaluate_parser.add_argument("input", type=Path)

    events_parser = subparsers.add_parser(
        "analyze-events", help="Segment a match event stream and evaluate every attack"
    )
    events_parser.add_argument("input", type=Path)
    events_parser.add_argument("--max-gap", type=float, default=15.0)
    events_parser.add_argument("--min-events", type=int, default=1)

    video_parser = subparsers.add_parser(
        "video-info", help="Read video metadata through the optional OpenCV backend"
    )
    video_parser.add_argument("input", type=Path)

    annotate_parser = subparsers.add_parser(
        "annotate-video",
        help="Render player skeletons, player/ball boxes, track IDs and speed",
    )
    annotate_parser.add_argument("input", type=Path)
    annotate_parser.add_argument("output", type=Path)
    annotate_parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    annotate_parser.add_argument("--detection-model", default="yolo26n.pt")
    annotate_parser.add_argument("--ball-model")
    annotate_parser.add_argument("--ball-class-id", type=int, default=32)
    annotate_parser.add_argument("--player-tracker", default="botsort.yaml")
    annotate_parser.add_argument("--ball-tracker", default="bytetrack.yaml")
    annotate_parser.add_argument("--device", default="cpu")
    annotate_parser.add_argument("--imgsz", type=int, default=960)
    annotate_parser.add_argument("--player-conf", type=float, default=0.2)
    annotate_parser.add_argument("--ball-conf", type=float, default=0.15)
    annotate_parser.add_argument("--keypoint-conf", type=float, default=0.25)
    annotate_parser.add_argument("--speed-smoothing", type=float, default=0.35)
    annotate_parser.add_argument("--roi", type=Path, help="Optional image-space pitch polygon JSON")
    annotate_parser.add_argument("--no-team-classification", action="store_true")
    annotate_parser.add_argument("--team-min-tracks", type=int, default=4)
    annotate_parser.add_argument("--scene-cut-threshold", type=float, default=0.35)
    annotate_parser.add_argument("--no-scene-cut-reset", action="store_true")
    annotate_parser.add_argument("--calibration", type=Path)
    annotate_parser.add_argument(
        "--observations", type=Path, help="Optional per-frame JSONL output"
    )
    annotate_parser.add_argument("--max-frames", type=int)
    annotate_parser.add_argument(
        "--field-lines", action="store_true", help="Extract pitch-line segments and intersections"
    )
    annotate_parser.add_argument("--field-line-min-length", type=float, default=0.06)
    annotate_parser.add_argument("--field-line-max-segments", type=int, default=40)
    annotate_parser.add_argument(
        "--filter-ball-by-roi",
        action="store_true",
        help="Restrict ball centers to the pitch ROI; unsafe for high airborne balls",
    )

    quality_parser = subparsers.add_parser(
        "quality-report",
        help="Validate observation JSONL and report detection/tracking/pose coverage",
    )
    quality_parser.add_argument("input", type=Path)
    quality_parser.add_argument("--keypoint-conf", type=float, default=0.25)

    inference_parser = subparsers.add_parser(
        "infer-events",
        help="Infer conservative possession/carry/pass/turnover events from calibrated tracks",
    )
    inference_parser.add_argument("input", type=Path)
    inference_parser.add_argument("--output", type=Path)
    inference_parser.add_argument("--possession-distance", type=float, default=3.0)
    inference_parser.add_argument("--carry-distance", type=float, default=3.0)
    inference_parser.add_argument("--transfer-gap", type=float, default=2.0)
    inference_parser.add_argument("--max-transfer-speed", type=float, default=45.0)
    inference_parser.add_argument("--shot-window", type=float, default=1.5)
    inference_parser.add_argument("--shot-min-x", type=float, default=70.0)
    inference_parser.add_argument("--shot-min-progress", type=float, default=8.0)
    inference_parser.add_argument("--shot-min-speed", type=float, default=8.0)
    inference_parser.add_argument(
        "--team-a-direction", choices=("left", "right"), default="right"
    )
    inference_parser.add_argument(
        "--team-b-direction", choices=("left", "right"), default="left"
    )

    pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        aliases=("analyze-clip",),
        help="Run video annotation, quality gate and optional event/attack analysis",
    )
    pipeline_parser.add_argument("input", type=Path)
    pipeline_parser.add_argument("run_directory", type=Path)
    pipeline_parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    pipeline_parser.add_argument("--detection-model", default="yolo26n.pt")
    pipeline_parser.add_argument("--ball-model")
    pipeline_parser.add_argument("--ball-class-id", type=int, default=32)
    pipeline_parser.add_argument("--device", default="cpu")
    pipeline_parser.add_argument("--imgsz", type=int, default=960)
    pipeline_parser.add_argument("--player-conf", type=float, default=0.2)
    pipeline_parser.add_argument("--ball-conf", type=float, default=0.15)
    pipeline_parser.add_argument("--roi", type=Path)
    pipeline_parser.add_argument("--calibration", type=Path)
    pipeline_parser.add_argument("--no-team-classification", action="store_true")
    pipeline_parser.add_argument("--max-frames", type=int)
    pipeline_parser.add_argument("--field-lines", action="store_true")
    pipeline_parser.add_argument("--field-line-min-length", type=float, default=0.06)
    pipeline_parser.add_argument("--field-line-max-segments", type=int, default=40)
    pipeline_parser.add_argument("--filter-ball-by-roi", action="store_true")
    pipeline_parser.add_argument(
        "--team-a-direction", choices=("left", "right"), default="right"
    )
    pipeline_parser.add_argument(
        "--team-b-direction", choices=("left", "right"), default="left"
    )

    subparsers.add_parser("doctor", help="Check Python, vision packages, CUDA and tools")

    soccernet_download_parser = subparsers.add_parser(
        "soccernet-download",
        help="Download one SoccerNet GameState sequence without fetching the full split",
    )
    soccernet_download_parser.add_argument("output_directory", type=Path)
    soccernet_download_parser.add_argument(
        "--split", choices=("train", "valid", "test", "challenge"), default="valid"
    )
    soccernet_download_parser.add_argument(
        "--sequence", default="first", help="SNGS sequence name or 'first'"
    )

    soccernet_analyze_parser = subparsers.add_parser(
        "soccernet-analyze", help="Analyze SoccerNet GameState labels and pitch motion"
    )
    soccernet_analyze_parser.add_argument("sequence_directory", type=Path)
    soccernet_analyze_parser.add_argument(
        "--fps", type=float, help="Override frame rate from dataset metadata"
    )
    soccernet_analyze_parser.add_argument("--max-speed", type=float, default=45.0)
    soccernet_analyze_parser.add_argument("--pitch-margin", type=float, default=5.0)

    field_parser = subparsers.add_parser(
        "field-geometry", help="Extract pitch masks, line segments and intersections from one image"
    )
    field_parser.add_argument("input", type=Path)
    field_parser.add_argument("output_json", type=Path)
    field_parser.add_argument("--overlay", type=Path)
    field_parser.add_argument("--min-length", type=float, default=0.06)
    field_parser.add_argument("--max-segments", type=int, default=40)

    detection_evaluation_parser = subparsers.add_parser(
        "evaluate-observations",
        help="Compare observation boxes with SoccerNet GameState ground truth",
    )
    detection_evaluation_parser.add_argument("observations", type=Path)
    detection_evaluation_parser.add_argument("labels", type=Path)
    detection_evaluation_parser.add_argument("--start-frame", type=int, default=1)
    detection_evaluation_parser.add_argument("--iou", type=float, default=0.5)

    ball_benchmark_parser = subparsers.add_parser(
        "benchmark-ball",
        help="Sweep ball confidence and enforce an accuracy gate on SoccerNet labels",
    )
    ball_benchmark_parser.add_argument("observations", type=Path)
    ball_benchmark_parser.add_argument("labels", type=Path)
    ball_benchmark_parser.add_argument("--start-frame", type=int, default=1)
    ball_benchmark_parser.add_argument("--iou", type=float, default=0.5)
    ball_benchmark_parser.add_argument(
        "--threshold", action="append", type=float, dest="thresholds"
    )
    ball_benchmark_parser.add_argument("--min-f1", type=float, default=0.80)
    ball_benchmark_parser.add_argument("--min-precision", type=float, default=0.75)
    ball_benchmark_parser.add_argument("--min-recall", type=float, default=0.75)
    ball_benchmark_parser.add_argument("--output", type=Path)

    yolo_export_parser = subparsers.add_parser(
        "soccernet-export-yolo",
        help="Export SoccerNet GameState ball annotations as a YOLO dataset",
    )
    yolo_export_parser.add_argument("output_directory", type=Path)
    yolo_export_parser.add_argument(
        "--train-sequence", action="append", required=True, type=Path
    )
    yolo_export_parser.add_argument(
        "--val-sequence", action="append", required=True, type=Path
    )
    yolo_export_parser.add_argument(
        "--link-mode", choices=("hardlink", "copy"), default="hardlink"
    )

    ball_training_parser = subparsers.add_parser(
        "train-ball-model", help="Fine-tune a one-class YOLO football detector"
    )
    ball_training_parser.add_argument("data_yaml", type=Path)
    ball_training_parser.add_argument("output_directory", type=Path)
    ball_training_parser.add_argument("--model", default="yolo26s.pt")
    ball_training_parser.add_argument("--epochs", type=int, default=100)
    ball_training_parser.add_argument("--imgsz", type=int, default=1280)
    ball_training_parser.add_argument("--batch", type=int, default=4)
    ball_training_parser.add_argument("--device", default="cpu")
    ball_training_parser.add_argument("--workers", type=int, default=0)
    ball_training_parser.add_argument("--name", default="soccernet-ball")

    footpass_parser = subparsers.add_parser(
        "footpass-ingest",
        help="Normalize one FOOTPASS half, enrich tactical coordinates and segment attacks",
    )
    footpass_parser.add_argument("playbyplay", type=Path)
    footpass_parser.add_argument("match_key")
    footpass_parser.add_argument("--source", choices=("oracle", "predicted"), default="oracle")
    footpass_parser.add_argument("--tactical-h5", type=Path)
    footpass_parser.add_argument("--fps", type=float, default=25.0)
    footpass_parser.add_argument("--min-confidence", type=float, default=0.15)
    footpass_parser.add_argument("--model-name")
    footpass_parser.add_argument("--max-gap", type=float, default=4.0)
    footpass_parser.add_argument("--max-duration", type=float, default=45.0)
    footpass_parser.add_argument("--min-events", type=int, default=1)
    footpass_parser.add_argument("--output", type=Path)

    footpass_evaluation_parser = subparsers.add_parser(
        "evaluate-footpass-actions",
        help="Evaluate predicted FOOTPASS actions with strict match/team/jersey/class matching",
    )
    footpass_evaluation_parser.add_argument("ground_truth", type=Path)
    footpass_evaluation_parser.add_argument("predictions", type=Path)
    footpass_evaluation_parser.add_argument("--fps", type=float, default=25.0)
    footpass_evaluation_parser.add_argument("--delta", type=int, default=12)
    footpass_evaluation_parser.add_argument("--min-confidence", type=float, default=0.15)
    footpass_evaluation_parser.add_argument("--model-name", default="predicted")

    annotation_parser = subparsers.add_parser(
        "build-annotations", help="Create a stratified, unlabeled attack annotation manifest"
    )
    annotation_parser.add_argument("input", type=Path, help="JSON containing an attacks array")
    annotation_parser.add_argument("output", type=Path)
    annotation_parser.add_argument("--sample-size", type=int, default=300)
    annotation_parser.add_argument("--seed", type=int, default=42)

    xg_parser = subparsers.add_parser(
        "train-xg", help="Train leakage-safe xG candidates with match-grouped splits"
    )
    xg_parser.add_argument("input", type=Path)
    xg_parser.add_argument("output", type=Path)
    xg_parser.add_argument("--match-column", default="match_id")
    xg_parser.add_argument("--target", default="goal")
    xg_parser.add_argument("--seed", type=int, default=42)

    effectiveness_parser = subparsers.add_parser(
        "train-effectiveness", help="Train effectiveness baselines with match-grouped splits"
    )
    effectiveness_parser.add_argument("input", type=Path)
    effectiveness_parser.add_argument("output", type=Path)
    effectiveness_parser.add_argument("--seed", type=int, default=42)

    evidence_parser = subparsers.add_parser(
        "evaluate-evidence", help="Create evidence and run the offline grounded-verdict baseline"
    )
    evidence_parser.add_argument("input", type=Path, help="AttackSequence JSON")
    evidence_parser.add_argument("--events-ready", action="store_true")
    evidence_parser.add_argument("--tracking-ready", action="store_true")
    evidence_parser.add_argument("--calibration-ready", action="store_true")

    demo_parser = subparsers.add_parser("serve-demo", help="Launch the local Streamlit thesis demo")
    demo_parser.add_argument("--port", type=int, default=8501)

    benchmark_parser = subparsers.add_parser(
        "build-footpass-benchmarks", help="Build a frozen per-90 role benchmark snapshot from tactical HDF5"
    )
    benchmark_parser.add_argument("tactical_h5", type=Path)
    benchmark_parser.add_argument("output", type=Path)
    benchmark_parser.add_argument("--fps", type=float, default=25.0)
    benchmark_parser.add_argument("--season", default="2023-24")
    benchmark_parser.add_argument("--snapshot-date")
    return parser


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _evaluate_file(path: Path) -> dict[str, object]:
    return evaluate_attack(AttackSequence.from_dict(_load_json(path))).to_dict()


def _analyze_event_stream(path: Path, max_gap: float, min_events: int) -> dict[str, object]:
    payload = _load_json(path)
    raw_events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(raw_events, list):
        raise ValueError("event stream must be a JSON list or an object containing 'events'")

    events = tuple(Event.from_dict(item) for item in raw_events)
    attacks = segment_attacks(
        events,
        SegmentationConfig(max_gap_sec=max_gap, min_events=min_events),
    )
    return {
        "source": str(path),
        "attack_count": len(attacks),
        "attacks": [
            {
                "attack": attack.to_dict(),
                "evaluation": evaluate_attack(attack).to_dict(),
            }
            for attack in attacks
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Keep the original `attack-analysis file.json` command working.
    if arguments and arguments[0] not in COMMANDS and not arguments[0].startswith("-"):
        arguments.insert(0, "evaluate")

    args = build_parser().parse_args(arguments)
    if args.command == "evaluate":
        result = _evaluate_file(args.input)
    elif args.command == "doctor":
        result = environment_report()
    elif args.command == "soccernet-download":
        result = download_gamestate_sequence(
            args.output_directory, args.split, args.sequence
        )
    elif args.command == "soccernet-analyze":
        result = analyze_gamestate_sequence(
            args.sequence_directory, args.fps, args.max_speed, args.pitch_margin
        )
    elif args.command == "field-geometry":
        result = analyze_field_image(
            args.input,
            args.output_json,
            args.overlay,
            FieldLineConfig(
                min_length_ratio=args.min_length,
                max_segments=args.max_segments,
            ),
        )
    elif args.command == "evaluate-observations":
        result = evaluate_soccernet_jsonl(
            args.observations,
            args.labels,
            args.start_frame,
            args.iou,
        )
    elif args.command == "benchmark-ball":
        result = benchmark_ball_jsonl(
            args.observations,
            args.labels,
            args.thresholds if args.thresholds is not None else DEFAULT_BALL_THRESHOLDS,
            args.start_frame,
            args.iou,
            args.min_f1,
            args.min_precision,
            args.min_recall,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    elif args.command == "soccernet-export-yolo":
        result = export_soccernet_ball_dataset(
            args.output_directory,
            args.train_sequence,
            args.val_sequence,
            args.link_mode,
        )
    elif args.command == "train-ball-model":
        result = train_ball_model(
            args.data_yaml,
            args.output_directory,
            args.model,
            args.epochs,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.name,
        )
    elif args.command == "footpass-ingest":
        events = load_footpass_events(
            args.playbyplay,
            match_key=args.match_key,
            source=args.source,
            tactical_h5=args.tactical_h5,
            fps=args.fps,
            min_confidence=args.min_confidence,
            model_name=args.model_name,
        )
        attacks = segment_footpass_attacks(
            events,
            match_id=args.match_key,
            config=FootpassSegmentationConfig(
                fps=args.fps,
                max_gap_sec=args.max_gap,
                max_duration_sec=args.max_duration,
                min_events=args.min_events,
            ),
        )
        result = {
            "schema_version": "1.0",
            "source": args.source,
            "match_key": args.match_key,
            "summary": footpass_summary(events, attacks),
            "events": [item.to_dict() for item in events],
            "attacks": [item.to_dict() for item in attacks],
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.command == "evaluate-footpass-actions":
        ground_truth_keys = set(footpass_match_keys(args.ground_truth))
        prediction_keys = set(footpass_match_keys(args.predictions))
        keys = sorted(ground_truth_keys | prediction_keys)
        ground_truth = {
            key: load_footpass_events(
                args.ground_truth,
                match_key=key,
                source="oracle",
                fps=args.fps,
            )
            if key in ground_truth_keys
            else ()
            for key in keys
        }
        predictions = {
            key: load_footpass_events(
                args.predictions,
                match_key=key,
                source="predicted",
                fps=args.fps,
                min_confidence=args.min_confidence,
                model_name=args.model_name,
            )
            if key in prediction_keys
            else ()
            for key in keys
        }
        result = action_spotting_dataset_report(
            ground_truth,
            predictions,
            tolerance_frames=args.delta,
        )
    elif args.command == "build-annotations":
        payload = _load_json(args.input)
        raw_attacks = payload.get("attacks") if isinstance(payload, dict) else None
        if not isinstance(raw_attacks, list):
            raise ValueError("annotation input must contain an attacks array")
        attacks = tuple(AttackSequence.from_dict(item) for item in raw_attacks)
        sampled = stratified_annotation_sample(attacks, sample_size=args.sample_size, seed=args.seed)
        result = build_annotation_manifest(sampled)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.command == "train-xg":
        result = train_xg_models(
            args.input,
            args.output,
            match_column=args.match_column,
            target=args.target,
            seed=args.seed,
        )
    elif args.command == "train-effectiveness":
        result = train_effectiveness_models(args.input, args.output, seed=args.seed)
    elif args.command == "evaluate-evidence":
        sequence = AttackSequence.from_dict(_load_json(args.input))
        features = build_attack_features(sequence)
        quality_gates = {
            "events": args.events_ready,
            "tracking": args.tracking_ready,
            "calibration": args.calibration_ready,
        }
        failed = tuple(f"quality gate failed: {name}" for name, passed in quality_gates.items() if not passed)
        bundle = EvidenceBundle(
            attack=features,
            quality_gates=quality_gates,
            quality_metrics={},
            source_ids=(sequence.source,),
            limitations=failed,
        )
        registry = EvidenceToolRegistry({sequence.attack_id: bundle})
        verdict = DeterministicEvidenceClient().evaluate(bundle, registry)
        result = {"bundle": bundle.to_dict(), "verdict": verdict.to_dict()}
    elif args.command == "serve-demo":
        app_path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
        completed = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)],
            check=False,
        )
        result = {"exit_code": completed.returncode, "app": str(app_path)}
    elif args.command == "build-footpass-benchmarks":
        records = build_footpass_role_benchmarks(
            args.tactical_h5,
            args.output,
            fps=args.fps,
            season=args.season,
            snapshot_date=args.snapshot_date,
        )
        result = {"output": str(args.output), "record_count": len(records)}
    elif args.command == "analyze-events":
        result = _analyze_event_stream(args.input, args.max_gap, args.min_events)
    elif args.command == "video-info":
        result = inspect_video(args.input).to_dict()
    elif args.command == "quality-report":
        result = analyze_observation_quality(
            read_observations_jsonl(args.input),
            args.keypoint_conf,
        ).to_dict()
    elif args.command == "infer-events":
        events = infer_events_from_observations(
            read_observations_jsonl(args.input),
            EventInferenceConfig(
                max_possession_distance_m=args.possession_distance,
                carry_min_distance_m=args.carry_distance,
                max_transfer_gap_sec=args.transfer_gap,
                max_transfer_speed_mps=args.max_transfer_speed,
                shot_window_sec=args.shot_window,
                shot_min_x=args.shot_min_x,
                shot_min_progress_m=args.shot_min_progress,
                shot_min_speed_mps=args.shot_min_speed,
                attack_directions={
                    "team_a": args.team_a_direction,
                    "team_b": args.team_b_direction,
                },
            ),
        )
        result = {"source": str(args.input), "events": [event.to_dict() for event in events]}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    elif args.command in {"run-pipeline", "analyze-clip"}:
        projector = PitchProjector.from_json(args.calibration) if args.calibration else None
        roi = PolygonROI.from_json(args.roi) if args.roi else None
        result = run_video_pipeline(
            args.input,
            args.run_directory,
            VisionConfig(
                pose_model=args.pose_model,
                detection_model=args.detection_model,
                ball_detection_model=args.ball_model,
                ball_class_id=args.ball_class_id,
                device=args.device,
                image_size=args.imgsz,
                player_confidence=args.player_conf,
                ball_confidence=args.ball_conf,
                team_classification=not args.no_team_classification,
                max_frames=args.max_frames,
                field_lines=args.field_lines,
                field_line_min_length_ratio=args.field_line_min_length,
                field_line_max_segments=args.field_line_max_segments,
                filter_ball_by_roi=args.filter_ball_by_roi,
            ),
            projector,
            roi,
            EventInferenceConfig(
                attack_directions={
                    "team_a": args.team_a_direction,
                    "team_b": args.team_b_direction,
                }
            ),
        )
    else:
        projector = PitchProjector.from_json(args.calibration) if args.calibration else None
        roi = PolygonROI.from_json(args.roi) if args.roi else None
        result = annotate_video(
            args.input,
            args.output,
            VisionConfig(
                pose_model=args.pose_model,
                detection_model=args.detection_model,
                ball_detection_model=args.ball_model,
                ball_class_id=args.ball_class_id,
                player_tracker=args.player_tracker,
                ball_tracker=args.ball_tracker,
                device=args.device,
                image_size=args.imgsz,
                player_confidence=args.player_conf,
                ball_confidence=args.ball_conf,
                keypoint_confidence=args.keypoint_conf,
                speed_smoothing=args.speed_smoothing,
                team_classification=not args.no_team_classification,
                team_min_tracks=args.team_min_tracks,
                scene_cut_threshold=(
                    None if args.no_scene_cut_reset else args.scene_cut_threshold
                ),
                max_frames=args.max_frames,
                field_lines=args.field_lines,
                field_line_min_length_ratio=args.field_line_min_length,
                field_line_max_segments=args.field_line_max_segments,
                filter_ball_by_roi=args.filter_ball_by_roi,
            ),
            projector,
            args.observations,
            roi,
        ).to_dict()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
