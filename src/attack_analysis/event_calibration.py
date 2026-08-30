from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .models import Event
from .research_evaluation import action_spotting_dataset_report


def apply_event_thresholds(
    events: Sequence[Event],
    thresholds: Mapping[str, float],
    *,
    minimum_confidence: float = 0.0,
) -> tuple[Event, ...]:
    return tuple(
        event
        for event in events
        if event.confidence >= max(minimum_confidence, thresholds.get(event.event_type, 0.0))
    )


def fit_event_threshold_profile(
    ground_truth: Mapping[str, Sequence[Event]],
    predictions: Mapping[str, Sequence[Event]],
    *,
    tuning_matches: Sequence[str],
    tolerance_frames: int = 12,
    baseline_threshold: float = 0.15,
    candidates: Sequence[float] = tuple(index / 100.0 for index in range(0, 96, 5)),
) -> dict[str, object]:
    tuning = tuple(sorted(set(tuning_matches)))
    all_matches = tuple(sorted(set(ground_truth) | set(predictions)))
    held_out = tuple(match_id for match_id in all_matches if match_id not in tuning)
    if not tuning or not set(tuning).issubset(all_matches):
        raise ValueError("tuning_matches must be a non-empty subset of available matches")
    if not held_out:
        raise ValueError("at least one match must remain held out")
    if not candidates or any(not 0.0 <= value <= 1.0 for value in candidates):
        raise ValueError("candidate thresholds must be between 0 and 1")

    classes = sorted(
        {event.event_type for match_id in tuning for event in ground_truth.get(match_id, ())}
    )
    thresholds: dict[str, float] = {}
    tuning_scores: dict[str, float] = {}
    tuning_ground_truth = {match_id: ground_truth.get(match_id, ()) for match_id in tuning}
    for event_type in classes:
        best_score = -1.0
        best_threshold = baseline_threshold
        for threshold in candidates:
            filtered = {
                match_id: tuple(
                    event
                    for event in predictions.get(match_id, ())
                    if event.event_type != event_type or event.confidence >= threshold
                )
                for match_id in tuning
            }
            report = action_spotting_dataset_report(
                tuning_ground_truth,
                filtered,
                tolerance_frames=tolerance_frames,
            )
            score = float(report["class_wise"].get(event_type, {}).get("f1", 0.0))
            if score > best_score or (score == best_score and threshold > best_threshold):
                best_score = score
                best_threshold = float(threshold)
        thresholds[event_type] = best_threshold
        tuning_scores[event_type] = best_score

    def evaluate(match_ids: Sequence[str], *, calibrated: bool) -> dict[str, object]:
        selected_ground_truth = {
            match_id: ground_truth.get(match_id, ()) for match_id in match_ids
        }
        selected_predictions = {
            match_id: (
                apply_event_thresholds(
                    predictions.get(match_id, ()),
                    thresholds,
                    minimum_confidence=baseline_threshold,
                )
                if calibrated
                else tuple(
                    event
                    for event in predictions.get(match_id, ())
                    if event.confidence >= baseline_threshold
                )
            )
            for match_id in match_ids
        }
        return action_spotting_dataset_report(
            selected_ground_truth,
            selected_predictions,
            tolerance_frames=tolerance_frames,
        )

    return {
        "schema_version": "dst-thresholds-v1",
        "baseline_threshold": baseline_threshold,
        "tolerance_frames": tolerance_frames,
        "tuning_matches": list(tuning),
        "held_out_matches": list(held_out),
        "thresholds": thresholds,
        "tuning_class_f1": tuning_scores,
        "tuning": {
            "baseline": evaluate(tuning, calibrated=False),
            "calibrated": evaluate(tuning, calibrated=True),
        },
        "held_out": {
            "baseline": evaluate(held_out, calibrated=False),
            "calibrated": evaluate(held_out, calibrated=True),
        },
        "all_matches": {
            "baseline": evaluate(all_matches, calibrated=False),
            "calibrated": evaluate(all_matches, calibrated=True),
        },
    }


def save_event_threshold_profile(report: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_event_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("event threshold profile is missing thresholds")
    result = {str(name): float(value) for name, value in thresholds.items()}
    if any(not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("event threshold profile contains an invalid threshold")
    return result
