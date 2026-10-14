from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .event_inference import EventInferenceConfig, infer_events_from_observations
from .event_calibration import (
    fit_event_threshold_profile,
    load_event_thresholds,
    save_event_threshold_profile,
)
from .detection_evaluation import (
    DEFAULT_BALL_THRESHOLDS,
    benchmark_ball_jsonl,
    evaluate_soccernet_jsonl,
)
from .environment import default_inference_imgsz, default_torch_device, environment_report
from .field_geometry import FieldLineConfig, analyze_field_image
from .annotations import build_annotation_manifest, stratified_annotation_sample
from .gold_annotations import (
    DEVELOPMENT_MATCHES,
    FROZEN_TEST_MATCHES,
    GOLD_SCHEMA_VERSION,
    GoldSequenceAnnotationV2,
    HUMAN_TRAIN_MATCHES,
    annotation_content_hash,
    build_gold_manifest,
    freeze_sealed_test_annotations,
    gold_progress_report,
    gold_split,
    intra_rater_report,
    read_gold_annotations_jsonl,
    read_gold_period_intervals,
    validate_continuous_full_match_coverage,
    verify_sealed_test_annotations,
)
from .attack_features import build_attack_features
from .agent_tools import EvidenceToolRegistry
from .benchmarks import FrozenBenchmarkStore, build_footpass_role_benchmarks, write_historical_snapshot
from .footpass_audit import audit_footpass_v2
from .footpass import (
    FootpassSegmentationConfig,
    footpass_match_keys,
    footpass_summary,
    footpass_v2_summary,
    prepare_footpass_actions_v2,
    load_footpass_events,
    segment_footpass_attacks,
)
from .attack_v2 import FootpassAttackV2Config, decode_footpass_attack_v2
from .scoreboard_goals import (
    ScoreboardGoalIndex,
    match_truth_from_restarts,
    match_truth_from_readings,
    read_scoreboard_readings,
)
from .action_spotting_v2 import OperatingPointMeasurement
from .release_control import (
    ReleaseComponentHashes,
    claim_final_test_once,
    complete_final_test_claim,
    final_test_evaluation_protocol_sha256,
    final_test_ledger_status,
    freeze_attack_v2_release,
    load_frozen_release_manifest,
    path_sha256,
    write_frozen_release_manifest,
)
from .provenance import content_sha256
from .prediction_sources import (
    DEFAULT_FOOTPASS_ROOT,
    PredictionSourceError,
    decide_role,
    load_registry,
    official_footpass_score,
    register_prediction_source,
    registry_report,
    resolve_prediction_source,
)
from .research_evaluation import action_spotting_dataset_report, rare_action_class_audit
from .sequence_evaluation import evaluate_acceptance_gates, sequence_evaluation_report
from .geometry import PitchProjector
from .llm import DeterministicEvidenceClient
from .ml import train_effectiveness_models, train_xg_models
from .env import xg_v2_model_path
from .shot_xg import ShotXGRequiredError, score_analysis_shots
from .xg_data import export_statsbomb_xg_dataset, export_statsbomb_xg_v2_dataset
from .xg_training_v2 import XGTrainingV2Config, train_xg_v2_models
from .xg_v2 import load_trained_xg_v2_model
from .models import AttackSequence, Event
from .models import (
    ATTACK_V2_RUBRIC_VERSION,
    ATTACK_V2_SCHEMA_VERSION,
    AttackSequenceV2,
)
from .pipeline import run_video_pipeline
from .quality import analyze_observation_quality, read_observations_jsonl
from .roi import PolygonROI
from .research import EvidenceBundle
from .scoring import evaluate_attack
from .segmentation import SegmentationConfig, segment_attacks
from .soccernet import analyze_gamestate_sequence, download_gamestate_sequence
from .training import train_ball_model
from .video import inspect_video
from .video_paths import operations_report
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
    "audit-footpass-v2",
    "build-scoreboard-goals",
    "evaluate-footpass-actions",
    "evaluate-attack-v2",
    "freeze-attack-v2-release",
    "verify-attack-v2-release",
    "attack-v2-final-test-status",
    "calibrate-footpass-events",
    "prediction-sources",
    "swap-prediction-source",
    "build-annotations",
    "build-gold-manifest",
    "validate-gold-annotations",
    "gold-intra-rater",
    "gold-status",
    "seal-gold-test",
    "verify-gold-test-seal",
    "build-xg-dataset",
    "build-xg-v2-dataset",
    "train-xg",
    "train-xg-v2",
    "train-effectiveness",
    "evaluate-evidence",
    "serve-demo",
    "build-footpass-benchmarks",
    "build-historical-snapshot",
    "attack-v2-overview",
    "inspect-video-sync",
    "serve-mcp",
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
    annotate_parser.add_argument("--device", default=default_torch_device())
    annotate_parser.add_argument("--imgsz", type=int, default=default_inference_imgsz())
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
    annotate_parser.add_argument("--start-frame", type=int, default=0)
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
    annotate_parser.add_argument(
        "--no-pose-crop",
        action="store_true",
        help="Skip the second pose pass on player crops",
    )
    annotate_parser.add_argument("--pose-crop-imgsz", type=int, default=320)
    annotate_parser.add_argument(
        "--ball-tile-size",
        type=int,
        default=0,
        help=(
            "Sliding-window side in pixels so a ~12px ball is upscaled. 0 (default) disables; "
            "tiles cut F1 to 0.136 with the COCO ball model on SNGS-033, so enable only with a "
            "fine-tuned --ball-model."
        ),
    )
    annotate_parser.add_argument("--ball-tile-overlap", type=float, default=0.1)
    annotate_parser.add_argument(
        "--ball-max-side",
        type=float,
        default=80.0,
        help="Drop ball boxes larger than this many pixels. 0 disables.",
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
    pipeline_parser.add_argument("--device", default=default_torch_device())
    pipeline_parser.add_argument("--imgsz", type=int, default=default_inference_imgsz())
    pipeline_parser.add_argument("--player-conf", type=float, default=0.2)
    pipeline_parser.add_argument("--ball-conf", type=float, default=0.15)
    pipeline_parser.add_argument("--roi", type=Path)
    pipeline_parser.add_argument("--calibration", type=Path)
    pipeline_parser.add_argument("--xg-model", type=Path)
    pipeline_parser.add_argument("--no-team-classification", action="store_true")
    pipeline_parser.add_argument("--max-frames", type=int)
    pipeline_parser.add_argument("--start-frame", type=int, default=0)
    pipeline_parser.add_argument("--field-lines", action="store_true")
    pipeline_parser.add_argument("--field-line-min-length", type=float, default=0.06)
    pipeline_parser.add_argument("--field-line-max-segments", type=int, default=40)
    pipeline_parser.add_argument("--filter-ball-by-roi", action="store_true")
    pipeline_parser.add_argument("--no-pose-crop", action="store_true")
    pipeline_parser.add_argument("--pose-crop-imgsz", type=int, default=320)
    pipeline_parser.add_argument("--ball-tile-size", type=int, default=0)
    pipeline_parser.add_argument("--ball-tile-overlap", type=float, default=0.1)
    pipeline_parser.add_argument("--ball-max-side", type=float, default=80.0)
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
    ball_training_parser.add_argument("--device", default=default_torch_device())
    ball_training_parser.add_argument("--workers", type=int, default=0)
    ball_training_parser.add_argument("--name", default="soccernet-ball")
    ball_training_parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit imgsz=1280 training on CPU (smoke only; hours per epoch)",
    )

    inspect_sync_parser = subparsers.add_parser(
        "inspect-video-sync",
        help="Check FOOTPASS dual-video roots, sync JSON, overlay cache key and missing files",
    )
    inspect_sync_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    footpass_parser = subparsers.add_parser(
        "footpass-ingest",
        help="Adapt one FOOTPASS half to v2 possessions/strict attacks; retain V1 as legacy candidates",
    )
    footpass_parser.add_argument(
        "playbyplay",
        type=Path,
        nargs="?",
        help="Play-by-play JSON; omit when --prediction-source picks a registered file",
    )
    footpass_parser.add_argument("match_key")
    footpass_parser.add_argument("--source", choices=("oracle", "predicted"), default="oracle")
    footpass_parser.add_argument(
        "--prediction-source",
        help="Registered predicted source id (models/prediction-sources.json); implies --source predicted",
    )
    footpass_parser.add_argument("--prediction-registry", type=Path)
    footpass_parser.add_argument("--tactical-h5", type=Path)
    footpass_parser.add_argument("--fps", type=float, default=25.0)
    footpass_parser.add_argument("--min-confidence", type=float, default=0.15)
    footpass_parser.add_argument("--model-name")
    footpass_parser.add_argument(
        "--xg-model",
        type=Path,
        help="StatsBomb-trained xG-v2 joblib. Sequences that end on event type shot are scored with it.",
    )
    footpass_parser.add_argument(
        "--event-threshold-profile", type=Path, help="Class-aware predicted-event calibration JSON"
    )
    footpass_parser.add_argument(
        "--max-gap",
        type=float,
        default=8.0,
        help="Observation gap that censors both sides; it is never a football terminal",
    )
    footpass_parser.add_argument("--max-duration", type=float, default=45.0)
    footpass_parser.add_argument("--min-events", type=int, default=1)
    footpass_parser.add_argument("--output", type=Path)

    footpass_audit_parser = subparsers.add_parser(
        "audit-footpass-v2",
        help="Regenerate the full validation oracle-assisted v2 geometry/invariant audit",
    )
    footpass_audit_parser.add_argument("playbyplay", type=Path)
    footpass_audit_parser.add_argument("tactical_h5", type=Path)
    footpass_audit_parser.add_argument("video_root", type=Path)
    footpass_audit_parser.add_argument("output", type=Path)

    scoreboard_goals_parser = subparsers.add_parser(
        "build-scoreboard-goals",
        help="Goal times per FOOTPASS match from the video score bug (event restart fallback without a score bug)",
    )
    scoreboard_goals_parser.add_argument(
        "playbyplay",
        type=Path,
        nargs="+",
        help="playbyplay JSON files; a sibling ../data/<split>_tactical_data.h5 is used for the restart fallback",
    )
    scoreboard_goals_parser.add_argument("--readings-dir", type=Path, default=Path("runs/goal-truth"))
    scoreboard_goals_parser.add_argument("--video-root", type=Path)
    scoreboard_goals_parser.add_argument("--exclude", default="game_0,game_36,game_50")
    scoreboard_goals_parser.add_argument(
        "--output", type=Path, default=Path("data/footpass-scoreboard-goals.json")
    )

    attack_v2_evaluation_parser = subparsers.add_parser(
        "evaluate-attack-v2",
        help="Evaluate strict attacks against continuous human gold with one-to-one tIoU matching",
    )
    attack_v2_evaluation_parser.add_argument("ground_truth", type=Path)
    attack_v2_evaluation_parser.add_argument("predictions", type=Path)
    attack_v2_evaluation_parser.add_argument("--output", type=Path)
    attack_v2_evaluation_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    attack_v2_evaluation_parser.add_argument("--seed", type=int, default=42)
    attack_v2_evaluation_parser.add_argument(
        "--allow-frozen-test",
        action="store_true",
        help=(
            "Consume the one permitted frozen-release evaluation on "
            "game_0/game_36/game_50"
        ),
    )
    attack_v2_evaluation_parser.add_argument(
        "--release-manifest",
        type=Path,
        help="Required immutable Attack-v2 release manifest for final-test evaluation",
    )
    attack_v2_evaluation_parser.add_argument(
        "--final-test-ledger",
        type=Path,
        help="Required append-only claim ledger for final-test evaluation",
    )
    attack_v2_evaluation_parser.add_argument(
        "--frozen-baseline",
        type=Path,
        help=(
            "Explicit frozen AttackSequenceV2 baseline used for the paired F1 "
            "improvement bootstrap; required for final-test evaluation"
        ),
    )

    release_freeze_parser = subparsers.add_parser(
        "freeze-attack-v2-release",
        help="Select the development operating point and freeze all release inputs",
    )
    release_freeze_parser.add_argument("release_id")
    release_freeze_parser.add_argument(
        "candidates",
        type=Path,
        help="JSON list (or candidates array) of threshold/precision/recall measurements",
    )
    release_freeze_parser.add_argument("config", type=Path)
    release_freeze_parser.add_argument("model", type=Path)
    release_freeze_parser.add_argument("schema", type=Path)
    release_freeze_parser.add_argument("rubric", type=Path)
    release_freeze_parser.add_argument("development_dataset", type=Path)
    release_freeze_parser.add_argument(
        "frozen_baseline",
        type=Path,
        help="Strict AttackSequenceV2 baseline bound to this release before final testing",
    )
    release_freeze_parser.add_argument("output", type=Path)

    release_verify_parser = subparsers.add_parser(
        "verify-attack-v2-release",
        help="Verify all self-hashes in an immutable Attack-v2 release manifest",
    )
    release_verify_parser.add_argument("manifest", type=Path)

    final_status_parser = subparsers.add_parser(
        "attack-v2-final-test-status",
        help="Read the immutable one-shot final-test ledger without changing it",
    )
    final_status_parser.add_argument("manifest", type=Path)
    final_status_parser.add_argument("ledger", type=Path)

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
    footpass_evaluation_parser.add_argument("--event-threshold-profile", type=Path)

    prediction_sources_parser = subparsers.add_parser(
        "prediction-sources",
        help="List registered FOOTPASS prediction files, their official F1 and hash status",
    )
    prediction_sources_parser.add_argument("--registry", type=Path)

    swap_source_parser = subparsers.add_parser(
        "swap-prediction-source",
        help=(
            "Score a new FOOTPASS prediction JSON with the official evaluator and register it; "
            "it becomes primary only if micro-F1 beats the current primary"
        ),
    )
    swap_source_parser.add_argument("predictions", type=Path)
    swap_source_parser.add_argument("--kind", choices=("TAAD", "DST"), required=True)
    swap_source_parser.add_argument("--id", dest="source_id", required=True)
    swap_source_parser.add_argument("--registry", type=Path)
    swap_source_parser.add_argument("--footpass-root", type=Path)
    swap_source_parser.add_argument("--ground-truth", type=Path)
    swap_source_parser.add_argument("--checkpoint", type=Path)
    swap_source_parser.add_argument("--threshold-profile", type=Path)
    swap_source_parser.add_argument("--note", default="")
    swap_source_parser.add_argument(
        "--dry-run", action="store_true", help="Score and report the gate decision without registering"
    )

    event_calibration_parser = subparsers.add_parser(
        "calibrate-footpass-events",
        help="Fit per-action confidence thresholds on named matches and report held-out results",
    )
    event_calibration_parser.add_argument("ground_truth", type=Path)
    event_calibration_parser.add_argument("predictions", type=Path)
    event_calibration_parser.add_argument("output", type=Path)
    event_calibration_parser.add_argument("--tuning-match", action="append", required=True)
    event_calibration_parser.add_argument("--fps", type=float, default=25.0)
    event_calibration_parser.add_argument("--delta", type=int, default=12)
    event_calibration_parser.add_argument("--baseline-threshold", type=float, default=0.15)

    annotation_parser = subparsers.add_parser(
        "build-annotations", help="Create a stratified, unlabeled attack annotation manifest"
    )
    annotation_parser.add_argument("input", type=Path, help="JSON containing an attacks array")
    annotation_parser.add_argument("output", type=Path)
    annotation_parser.add_argument("--sample-size", type=int, default=300)
    annotation_parser.add_argument("--seed", type=int, default=42)

    gold_manifest_parser = subparsers.add_parser(
        "build-gold-manifest",
        help="Freeze the twelve-match continuous Attack-v2 annotation manifest",
    )
    gold_manifest_parser.add_argument("video_root", type=Path)
    gold_manifest_parser.add_argument("output", type=Path)
    gold_manifest_parser.add_argument(
        "--sealed-directory",
        type=Path,
        help="Optional verified frozen-test seal to bind into the manifest",
    )

    gold_validation_parser = subparsers.add_parser(
        "validate-gold-annotations",
        help="Validate Attack-v2 JSONL labels without exposing the frozen test by default",
    )
    gold_validation_parser.add_argument("input", type=Path)
    gold_validation_parser.add_argument(
        "--allow-frozen-test",
        action="store_true",
        help="Final evaluator only: permit reading game_0/game_36/game_50 labels",
    )
    gold_validation_parser.add_argument(
        "--period-intervals",
        type=Path,
        help="Declared inclusive match/period bounds for continuous coverage validation",
    )
    gold_validation_parser.add_argument("--annotation-round", type=int, default=1)

    gold_agreement_parser = subparsers.add_parser(
        "gold-intra-rater",
        help="Compare two annotation rounds on stable Attack-v2 sequence IDs",
    )
    gold_agreement_parser.add_argument("first_round", type=Path)
    gold_agreement_parser.add_argument("second_round", type=Path)
    gold_agreement_parser.add_argument("--allow-frozen-test", action="store_true")

    gold_status_parser = subparsers.add_parser(
        "gold-status",
        help="Report 12-match gold progress without opening frozen-test labels (P11 stays open)",
    )
    gold_status_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing data/attack-v2-*.jsonl",
    )
    gold_status_parser.add_argument("--sealed-directory", type=Path)

    gold_seal_parser = subparsers.add_parser(
        "seal-gold-test",
        help="Hash and irreversibly freeze the isolated final-test annotation file",
    )
    gold_seal_parser.add_argument("sealed_directory", type=Path)
    gold_seal_parser.add_argument(
        "--period-intervals",
        type=Path,
        required=True,
        help="Declarations for both periods of exactly game_0, game_36 and game_50",
    )

    gold_verify_parser = subparsers.add_parser(
        "verify-gold-test-seal",
        help="Verify the frozen final-test label file against its content/file hashes",
    )
    gold_verify_parser.add_argument("sealed_directory", type=Path)

    xg_dataset_parser = subparsers.add_parser(
        "build-xg-dataset",
        help="Export match-grouped video-reproducible shot features from StatsBomb 360",
    )
    xg_dataset_parser.add_argument("open_data_root", type=Path)
    xg_dataset_parser.add_argument("output", type=Path)

    xg_v2_dataset_parser = subparsers.add_parser(
        "build-xg-v2-dataset",
        help="Export StatsBomb shots with explicit unknown/missingness feature semantics",
    )
    xg_v2_dataset_parser.add_argument("open_data_root", type=Path)
    xg_v2_dataset_parser.add_argument("output", type=Path)

    xg_parser = subparsers.add_parser(
        "train-xg", help="Train leakage-safe xG candidates with match-grouped splits"
    )
    xg_parser.add_argument("input", type=Path)
    xg_parser.add_argument("output", type=Path)
    xg_parser.add_argument("--match-column", default="match_id")
    xg_parser.add_argument("--target", default="goal")
    xg_parser.add_argument("--seed", type=int, default=42)

    xg_v2_parser = subparsers.add_parser(
        "train-xg-v2",
        help="Run grouped nested-CV xG-v2 selection, disjoint calibration and frozen test evaluation",
    )
    xg_v2_parser.add_argument("input", type=Path)
    xg_v2_parser.add_argument("output", type=Path)
    xg_v2_parser.add_argument("--match-column", default="match_id")
    xg_v2_parser.add_argument("--target", default="goal")
    xg_v2_parser.add_argument("--seed", type=int, default=42)
    xg_v2_parser.add_argument("--inner-folds", type=int, default=3)
    xg_v2_parser.add_argument("--bootstrap-samples", type=int, default=2000)

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

    mcp_parser = subparsers.add_parser(
        "serve-mcp",
        help="Serve Attack-v2 evidence tools over MCP stdio (Brave + frozen benchmarks)",
    )
    mcp_parser.add_argument("--bundle", type=Path, help="EvidenceBundle JSON for get_attack_evidence")
    mcp_parser.add_argument("--benchmarks", type=Path)

    benchmark_parser = subparsers.add_parser(
        "build-footpass-benchmarks", help="Build a frozen per-90 role benchmark snapshot from tactical HDF5"
    )
    benchmark_parser.add_argument("tactical_h5", type=Path)
    benchmark_parser.add_argument("output", type=Path)
    benchmark_parser.add_argument("--fps", type=float, default=25.0)
    benchmark_parser.add_argument("--season", default="2023-24")
    benchmark_parser.add_argument("--snapshot-date")

    snapshot_parser = subparsers.add_parser(
        "build-historical-snapshot",
        help="Write a frozen historical stats JSONL snapshot; live scrape is not accepted",
    )
    snapshot_parser.add_argument("input", type=Path, help="JSON list of benchmark records")
    snapshot_parser.add_argument("output", type=Path)
    snapshot_parser.add_argument("--snapshot-date", required=True)
    snapshot_parser.add_argument("--season", default="2023-24")
    snapshot_parser.add_argument("--source-url", required=True)

    overview_parser = subparsers.add_parser(
        "attack-v2-overview",
        help="Research-only Attack-v2 overview; never emits a categorical effectiveness verdict",
    )
    overview_parser.add_argument("input", type=Path, help="AttackSequenceV2 JSON or run JSON with attacks")
    overview_parser.add_argument("--xg-record", type=Path)
    overview_parser.add_argument("--benchmarks", type=Path)
    overview_parser.add_argument("--query")
    overview_parser.add_argument("--attack-id")
    return parser


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _operating_point_measurements(path: Path) -> tuple[OperatingPointMeasurement, ...]:
    payload = _load_json(path)
    values = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("operating-point input must be a non-empty candidates array")
    measurements: list[OperatingPointMeasurement] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("every operating-point candidate must be an object")
        try:
            precision = (
                value["attack_precision"]
                if "attack_precision" in value
                else value["precision"]
            )
        except KeyError as error:
            raise ValueError("operating-point candidate is missing precision") from error
        try:
            recall = value["attack_recall"] if "attack_recall" in value else value["recall"]
        except KeyError as error:
            raise ValueError("operating-point candidate is missing recall") from error
        measurements.append(
            OperatingPointMeasurement(
                threshold=float(value["threshold"]),
                attack_precision=float(precision),
                attack_recall=float(recall),
                label=str(value.get("label", "")),
            )
        )
    return tuple(measurements)


def _sequence_records(
    path: Path,
    *,
    human_gold: bool,
    allow_frozen_test: bool = False,
) -> tuple[object, ...]:
    if allow_frozen_test and not human_gold:
        raise ValueError("allow_frozen_test applies only to human ground truth")

    if path.suffix.casefold() == ".jsonl":
        raw_values: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid sequence JSONL at line {line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"sequence JSONL line {line_number} must contain an object"
                    )
                raw_values.append(value)
        values: list[object] = raw_values
    else:
        payload = _load_json(path)
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            container_key = "annotations" if human_gold else "attacks"
            candidate = payload.get(container_key)
            if isinstance(candidate, list):
                values = candidate
            elif (human_gold and "annotation_id" in payload) or (
                not human_gold and "attack_id" in payload
            ):
                values = [payload]
            else:
                raise ValueError(
                    f"strict sequence JSON must be a list or contain {container_key!r}"
                )
        else:
            raise ValueError("sequence input must contain JSON objects")

    records: list[object] = []
    if human_gold:
        if allow_frozen_test:
            if (
                path.suffix.casefold() != ".jsonl"
                or path.name != "attack-v2-frozen-test.jsonl"
            ):
                raise PermissionError(
                    "frozen final-test gold is accepted only through the sealed JSONL"
                )
            verify_sealed_test_annotations(path.parent)
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ValueError(f"human gold record {index} must be an object")
            if value.get("schema_version") != GOLD_SCHEMA_VERSION:
                raise ValueError(
                    f"human gold record {index} must declare {GOLD_SCHEMA_VERSION}"
                )
            try:
                annotation = GoldSequenceAnnotationV2.from_dict(value)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid human gold record {index}: {error}") from error
            split = gold_split(annotation.match_id)
            if split == "test" and not allow_frozen_test:
                raise PermissionError(
                    "frozen final-test gold requires the sealed one-shot evaluator"
                )
            if split != "test" and allow_frozen_test:
                raise PermissionError("sealed final-test JSONL may contain only test matches")
            if annotation.label == "attack":
                records.append(annotation)
        return tuple(records)

    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"prediction record {index} must be an object")
        if value.get("schema_version") != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(
                f"prediction record {index} must declare AttackSequenceV2 schema "
                f"{ATTACK_V2_SCHEMA_VERSION}"
            )
        if not isinstance(value.get("possession"), dict):
            raise ValueError(f"prediction record {index} must be an AttackSequenceV2")
        possession = value["possession"]
        if possession.get("schema_version") != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(
                f"prediction record {index} possession must declare schema "
                f"{ATTACK_V2_SCHEMA_VERSION}"
            )
        if possession.get("rubric_version") != ATTACK_V2_RUBRIC_VERSION:
            raise ValueError(
                f"prediction record {index} possession must declare rubric "
                f"{ATTACK_V2_RUBRIC_VERSION}"
            )
        actions = possession.get("actions")
        if not isinstance(actions, list) or any(
            not isinstance(action, dict)
            or action.get("schema_version") != ATTACK_V2_SCHEMA_VERSION
            for action in actions
        ):
            raise ValueError(
                f"prediction record {index} actions must explicitly declare v2 schema"
            )
        terminal = possession.get("terminal")
        if not isinstance(terminal, dict) or terminal.get(
            "schema_version"
        ) != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(
                f"prediction record {index} terminal must explicitly declare v2 schema"
            )
        try:
            records.append(AttackSequenceV2.from_dict(value))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid AttackSequenceV2 prediction {index}: {error}") from error
    return tuple(records)


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
        result["operations"] = operations_report(
            Path.cwd(),
            expected_match_ids=tuple(
                f"game_{number}"
                for number in HUMAN_TRAIN_MATCHES + DEVELOPMENT_MATCHES + FROZEN_TEST_MATCHES
            ),
        )
        result["gold"] = gold_progress_report(Path.cwd())
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
            allow_cpu=args.allow_cpu,
        )
    elif args.command == "footpass-ingest":
        prediction_source = None
        if args.prediction_source:
            if args.playbyplay is not None:
                raise SystemExit("pass either a playbyplay path or --prediction-source, not both")
            try:
                prediction_source, args.playbyplay = resolve_prediction_source(
                    args.prediction_source, args.prediction_registry
                )
            except PredictionSourceError as error:
                raise SystemExit(str(error)) from error
            args.source = "predicted"
            args.model_name = args.model_name or prediction_source.model_name
        elif args.playbyplay is None:
            raise SystemExit("footpass-ingest needs a playbyplay path or --prediction-source")
        thresholds = (
            load_event_thresholds(args.event_threshold_profile)
            if args.event_threshold_profile
            else None
        )
        decoder_config = FootpassAttackV2Config(
            fps=args.fps,
            max_observation_gap_sec=args.max_gap,
        )
        prepared = prepare_footpass_actions_v2(
            args.playbyplay,
            match_key=args.match_key,
            source=args.source,
            tactical_h5=args.tactical_h5,
            fps=args.fps,
            min_confidence=args.min_confidence,
            model_name=args.model_name,
            class_confidence_thresholds=thresholds,
            decoder_config=decoder_config,
        )
        observed_actions = prepared.actions
        decoded = decode_footpass_attack_v2(observed_actions, decoder_config)

        # Preserve the old heuristic strictly as a labelled compatibility
        # artifact. It is intentionally not evaluated or exposed as attacks.
        events = load_footpass_events(
            args.playbyplay,
            match_key=args.match_key,
            source=args.source,
            tactical_h5=args.tactical_h5,
            fps=args.fps,
            min_confidence=args.min_confidence,
            model_name=args.model_name,
            xg_model_path=None,
            class_confidence_thresholds=thresholds,
        )
        legacy_candidates = segment_footpass_attacks(
            events,
            match_id=args.match_key,
            config=FootpassSegmentationConfig(
                fps=args.fps,
                max_gap_sec=args.max_gap,
                max_duration_sec=args.max_duration,
                min_events=args.min_events,
            ),
        )
        model_path = args.xg_model if args.xg_model is not None else xg_v2_model_path(
            Path(__file__).resolve().parents[2]
        )
        loaded_xg = (
            load_trained_xg_v2_model(model_path, allow_research_only=True)
            if model_path.is_file()
            else None
        )
        try:
            xg_payload = score_analysis_shots(
                decoded.attacks,
                decoded.possessions,
                model=loaded_xg,
                tactical_h5=args.tactical_h5,
            )
        except ShotXGRequiredError as error:
            raise SystemExit(str(error)) from error
        result = {
            "schema_version": "attack-v2-run/2.0",
            "source_tier": (
                observed_actions[0].source_tier.value
                if observed_actions
                else (
                    "oracle_actions_oracle_tactical"
                    if args.source == "oracle"
                    else (
                        "predicted_actions_oracle_tactical"
                        if args.tactical_h5 is not None
                        else "video_only"
                    )
                )
            ),
            "match_key": args.match_key,
            "prediction_source": (
                None
                if prediction_source is None
                else {
                    "id": prediction_source.id,
                    "kind": prediction_source.kind,
                    "role": prediction_source.role,
                    "sha256": prediction_source.sha256,
                    "official_micro_f1": prediction_source.micro_f1,
                }
            ),
            "summary": {
                **footpass_v2_summary(decoded, prepared),
                "observed_action_count": len(observed_actions),
            },
            "observed_actions": [item.to_dict() for item in observed_actions],
            **decoded.to_dict(),
            "xg": xg_payload,
            "legacy_compatibility": {
                "output_type": "PossessionCandidateV1",
                "summary": footpass_summary(events, legacy_candidates),
                "events": [item.to_dict() for item in events],
                "possession_candidates": [item.to_dict() for item in legacy_candidates],
            },
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.command == "build-scoreboard-goals":
        excluded = {item for item in args.exclude.split(",") if item}
        index = ScoreboardGoalIndex.load(args.output)
        matches = dict(index.matches)
        for playbyplay in args.playbyplay:
            split = playbyplay.stem.rsplit("_", 1)[-1]
            tactical = playbyplay.parent.parent / "data" / f"{split}_tactical_data.h5"
            keys = footpass_match_keys(playbyplay)
            for match_id in sorted({key.rsplit("_H", 1)[0] for key in keys} - excluded):
                readings = read_scoreboard_readings(match_id, args.readings_dir, args.video_root)
                if readings is None:
                    print(f"{match_id}: no scoreboard readings, skipped", flush=True)
                    continue
                truth = match_truth_from_readings(match_id, readings)
                if truth is None:
                    halves = [
                        prepare_footpass_actions_v2(
                            playbyplay,
                            match_key=key,
                            source="oracle",
                            tactical_h5=tactical if tactical.is_file() else None,
                        ).actions
                        for key in keys
                        if key.rsplit("_H", 1)[0] == match_id
                    ]
                    truth = match_truth_from_restarts(match_id, halves)
                matches[match_id] = truth
                print(f"{match_id}: {truth.source} goals={len(truth.goal_times_sec)} final={truth.final_score}", flush=True)
        ScoreboardGoalIndex(matches).save(args.output)
        result = {"output": str(args.output), "matches": len(matches)}
    elif args.command == "audit-footpass-v2":
        result = audit_footpass_v2(
            args.playbyplay,
            args.tactical_h5,
            args.video_root,
            args.output,
        )
    elif args.command == "freeze-attack-v2-release":
        release = freeze_attack_v2_release(
            release_id=args.release_id,
            component_hashes=ReleaseComponentHashes(
                config_sha256=path_sha256(args.config),
                model_sha256=path_sha256(args.model),
                schema_sha256=path_sha256(args.schema),
                rubric_sha256=path_sha256(args.rubric),
                development_dataset_sha256=path_sha256(args.development_dataset),
                frozen_baseline_sha256=path_sha256(args.frozen_baseline),
            ),
            measurements=_operating_point_measurements(args.candidates),
        )
        write_frozen_release_manifest(release, args.output)
        result = release.to_dict()
    elif args.command == "verify-attack-v2-release":
        release = load_frozen_release_manifest(args.manifest)
        result = {
            "valid": True,
            "release_id": release.release_id,
            "release_sha256": release.release_sha256,
            "selected_operating_point": release.operating_point["selected"],
            "component_hashes": release.component_hashes.to_dict(),
        }
    elif args.command == "attack-v2-final-test-status":
        release = load_frozen_release_manifest(args.manifest)
        result = final_test_ledger_status(release, args.ledger)
    elif args.command == "evaluate-attack-v2":
        final_release = None
        final_claim = None
        if args.allow_frozen_test:
            if (
                args.release_manifest is None
                or args.final_test_ledger is None
                or args.frozen_baseline is None
            ):
                raise PermissionError(
                    "frozen final test requires --release-manifest, --final-test-ledger, "
                    "and --frozen-baseline"
                )
            final_release = load_frozen_release_manifest(args.release_manifest)
            protocol_hash = final_test_evaluation_protocol_sha256(
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.seed,
            )
            expected_label_name = "attack-v2-frozen-test.jsonl"
            if args.ground_truth.name != expected_label_name:
                raise PermissionError(
                    f"frozen final test must use the sealed {expected_label_name} file"
                )
            seal_path = args.ground_truth.parent / "attack-v2-frozen-test.seal.json"
            if not seal_path.is_file():
                raise PermissionError("frozen final-test labels have not been hash-sealed")
            seal = _load_json(seal_path)
            if not isinstance(seal, dict) or seal.get("state") != "frozen":
                raise PermissionError("frozen final-test label seal is invalid")
            expected_matches = {f"game_{number}" for number in FROZEN_TEST_MATCHES}
            if set(seal.get("matches", ())) != expected_matches:
                raise PermissionError(
                    "frozen final-test seal must cover game_0, game_36, and game_50"
                )
            baseline_path_hash = path_sha256(args.frozen_baseline)
            if (
                baseline_path_hash
                != final_release.component_hashes.frozen_baseline_sha256
            ):
                raise ValueError(
                    "supplied frozen baseline does not match the release manifest"
                )
            # The irreversible claim happens before the label file is opened.
            prediction_bundle_hash = content_sha256(
                {
                    "candidate_sha256": path_sha256(args.predictions),
                    "frozen_baseline_sha256": baseline_path_hash,
                }
            )
            final_claim = claim_final_test_once(
                final_release,
                args.final_test_ledger,
                final_test_labels_sha256=str(seal.get("file_sha256", "")),
                predictions_sha256=prediction_bundle_hash,
                evaluation_protocol_sha256=protocol_hash,
            )
        elif args.release_manifest is not None or args.final_test_ledger is not None:
            raise ValueError(
                "release manifest and final-test ledger are only valid with --allow-frozen-test"
            )
        ground_truth = _sequence_records(
            args.ground_truth,
            human_gold=True,
            allow_frozen_test=args.allow_frozen_test,
        )
        predictions = _sequence_records(args.predictions, human_gold=False)
        frozen_baseline = (
            None
            if args.frozen_baseline is None
            else _sequence_records(args.frozen_baseline, human_gold=False)
        )
        report = sequence_evaluation_report(
            ground_truth,
            predictions,
            terminal_compatibility="exact",
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
            frozen_baseline_predictions=frozen_baseline,
        )
        report["input_hashes"] = {
            "ground_truth_sha256": path_sha256(args.ground_truth),
            "predictions_sha256": path_sha256(args.predictions),
            "frozen_baseline_sha256": (
                None
                if args.frozen_baseline is None
                else path_sha256(args.frozen_baseline)
            ),
        }
        report["acceptance"] = evaluate_acceptance_gates(report)
        if final_release is not None and final_claim is not None:
            report["release_control"] = {
                "release_id": final_release.release_id,
                "release_sha256": final_release.release_sha256,
                "claim_sha256": final_claim["claim_sha256"],
                "final_test_policy": "one_shot_consumed_before_label_read",
            }
        result = report
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        if final_release is not None and final_claim is not None:
            complete_final_test_claim(
                final_release,
                args.final_test_ledger,
                claim_sha256=str(final_claim["claim_sha256"]),
                result_sha256=content_sha256(result),
            )
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
                class_confidence_thresholds=(
                    load_event_thresholds(args.event_threshold_profile)
                    if args.event_threshold_profile
                    else None
                ),
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
    elif args.command == "prediction-sources":
        result = registry_report(args.registry)
    elif args.command == "swap-prediction-source":
        footpass_root = args.footpass_root or DEFAULT_FOOTPASS_ROOT
        score = official_footpass_score(
            args.predictions,
            footpass_root=footpass_root,
            ground_truth=args.ground_truth,
        )
        if args.dry_run:
            primary = load_registry(args.registry).primary
            result = {
                "dry_run": True,
                "official": score.to_dict(),
                "would_be_role": decide_role(args.kind, score.micro_f1, primary),
                "current_primary": None if primary is None else primary.id,
                "current_primary_micro_f1": None if primary is None else primary.micro_f1,
            }
        else:
            result = register_prediction_source(
                args.predictions,
                source_id=args.source_id,
                kind=args.kind,
                score=score,
                registry_path=args.registry,
                store_directory=Path(footpass_root) / "playbyplay_PRED",
                checkpoint=args.checkpoint,
                threshold_profile=args.threshold_profile,
                note=args.note,
            )
    elif args.command == "calibrate-footpass-events":
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
                min_confidence=0.0,
                model_name="dst_uncalibrated",
            )
            if key in prediction_keys
            else ()
            for key in keys
        }
        result = fit_event_threshold_profile(
            ground_truth,
            predictions,
            tuning_matches=args.tuning_match,
            tolerance_frames=args.delta,
            baseline_threshold=args.baseline_threshold,
        )
        save_event_threshold_profile(result, args.output)
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
    elif args.command == "build-gold-manifest":
        result = build_gold_manifest(
            args.video_root,
            sealed_directory=args.sealed_directory,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif args.command == "validate-gold-annotations":
        annotations = read_gold_annotations_jsonl(
            args.input,
            allow_frozen_test=args.allow_frozen_test,
        )
        split_counts: dict[str, int] = {}
        for item in annotations:
            split = gold_split(item.match_id)
            split_counts[split] = split_counts.get(split, 0) + 1
        result = {
            "schema_version": GOLD_SCHEMA_VERSION,
            "valid": True,
            "annotation_count": len(annotations),
            "split_counts": split_counts,
            "content_sha256": annotation_content_hash(annotations),
        }
        if args.period_intervals is not None:
            result["continuous_coverage"] = validate_continuous_full_match_coverage(
                annotations,
                read_gold_period_intervals(args.period_intervals),
                annotation_round=args.annotation_round,
            )
    elif args.command == "gold-intra-rater":
        first_round = read_gold_annotations_jsonl(
            args.first_round,
            allow_frozen_test=args.allow_frozen_test,
        )
        second_round = read_gold_annotations_jsonl(
            args.second_round,
            allow_frozen_test=args.allow_frozen_test,
        )
        result = intra_rater_report(first_round, second_round)
    elif args.command == "gold-status":
        result = gold_progress_report(
            args.project_root,
            sealed_directory=args.sealed_directory,
        )
    elif args.command == "inspect-video-sync":
        result = operations_report(
            args.project_root,
            expected_match_ids=tuple(
                f"game_{number}"
                for number in HUMAN_TRAIN_MATCHES + DEVELOPMENT_MATCHES + FROZEN_TEST_MATCHES
            ),
        )
    elif args.command == "seal-gold-test":
        seal_path = freeze_sealed_test_annotations(
            args.sealed_directory,
            read_gold_period_intervals(args.period_intervals),
        )
        result = {
            "sealed": True,
            "seal_path": str(seal_path.resolve()),
            **verify_sealed_test_annotations(args.sealed_directory),
        }
    elif args.command == "verify-gold-test-seal":
        result = {
            "valid": True,
            **verify_sealed_test_annotations(args.sealed_directory),
        }
    elif args.command == "build-xg-dataset":
        result = export_statsbomb_xg_dataset(args.open_data_root, args.output)
    elif args.command == "build-xg-v2-dataset":
        result = export_statsbomb_xg_v2_dataset(args.open_data_root, args.output)
    elif args.command == "train-xg":
        result = train_xg_models(
            args.input,
            args.output,
            match_column=args.match_column,
            target=args.target,
            seed=args.seed,
        )
    elif args.command == "train-xg-v2":
        result = train_xg_v2_models(
            args.input,
            args.output,
            match_column=args.match_column,
            target=args.target,
            config=XGTrainingV2Config(
                seed=args.seed,
                inner_folds=args.inner_folds,
                bootstrap_samples=args.bootstrap_samples,
            ),
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
    elif args.command == "serve-mcp":
        from .mcp_server import serve_mcp_stdio

        serve_mcp_stdio(bundle_path=args.bundle, benchmarks_path=args.benchmarks)
        result = {"status": "stopped"}
    elif args.command == "build-footpass-benchmarks":
        records = build_footpass_role_benchmarks(
            args.tactical_h5,
            args.output,
            fps=args.fps,
            season=args.season,
            snapshot_date=args.snapshot_date,
        )
        result = {"output": str(args.output), "record_count": len(records)}
    elif args.command == "build-historical-snapshot":
        payload = _load_json(args.input)
        if not isinstance(payload, list):
            raise ValueError("historical snapshot input must be a JSON list of records")
        records = write_historical_snapshot(
            payload,
            args.output,
            snapshot_date=args.snapshot_date,
            default_season=args.season,
            default_source_url=args.source_url,
        )
        result = {"output": str(args.output), "record_count": len(records)}
    elif args.command == "attack-v2-overview":
        from .overview import overview_from_payload

        payload = _load_json(args.input)
        attack_payload = payload
        if "possession" not in payload and isinstance(payload.get("attacks"), list):
            attacks = payload["attacks"]
            if args.attack_id:
                attack_payload = next(
                    item for item in attacks if item.get("attack_id") == args.attack_id
                )
            elif attacks:
                attack_payload = attacks[0]
            else:
                raise ValueError("run JSON contains no AttackSequenceV2 objects")
        xg_payload = _load_json(args.xg_record) if args.xg_record else None
        store = FrozenBenchmarkStore.from_jsonl(args.benchmarks) if args.benchmarks else None
        result = overview_from_payload(
            attack_payload,
            xg_payload=xg_payload,
            benchmarks=store,
            query=args.query,
        )
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
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                field_lines=args.field_lines,
                field_line_min_length_ratio=args.field_line_min_length,
                field_line_max_segments=args.field_line_max_segments,
                filter_ball_by_roi=args.filter_ball_by_roi,
                pose_on_crop=not args.no_pose_crop,
                pose_crop_imgsz=args.pose_crop_imgsz,
                ball_tile_size=args.ball_tile_size,
                ball_tile_overlap=args.ball_tile_overlap,
                ball_max_side_px=args.ball_max_side,
            ),
            projector,
            roi,
            EventInferenceConfig(
                attack_directions={
                    "team_a": args.team_a_direction,
                    "team_b": args.team_b_direction,
                }
            ),
            args.xg_model,
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
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                field_lines=args.field_lines,
                field_line_min_length_ratio=args.field_line_min_length,
                field_line_max_segments=args.field_line_max_segments,
                filter_ball_by_roi=args.filter_ball_by_roi,
                pose_on_crop=not args.no_pose_crop,
                pose_crop_imgsz=args.pose_crop_imgsz,
                ball_tile_size=args.ball_tile_size,
                ball_tile_overlap=args.ball_tile_overlap,
                ball_max_side_px=args.ball_max_side,
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
