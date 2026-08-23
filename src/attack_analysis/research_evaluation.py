from __future__ import annotations

import math
import random
from statistics import median
from typing import Callable, Mapping, Sequence

from .models import Event


def _prf(true_positive: int, predicted: int, actual: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "actual": actual,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def action_spotting_report(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    *,
    tolerance_frames: int = 12,
    match_player: bool = True,
) -> dict[str, object]:
    """Greedy one-to-one spotting metrics using FOOTPASS team/player/action identity."""
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be non-negative")
    classes = sorted({event.event_type for event in (*ground_truth, *predictions)})
    by_class: dict[str, dict[str, float | int]] = {}
    total_tp = total_predicted = total_actual = 0
    for event_type in classes:
        actual = [event for event in ground_truth if event.event_type == event_type]
        predicted = [event for event in predictions if event.event_type == event_type]
        remaining = set(range(len(actual)))
        true_positive = 0
        for prediction in sorted(predicted, key=lambda item: item.confidence or 0.0, reverse=True):
            if prediction.frame is None:
                continue
            candidates = [
                index
                for index in remaining
                if actual[index].frame is not None
                and actual[index].team == prediction.team
                and (not match_player or actual[index].jersey == prediction.jersey)
                and abs(actual[index].frame - prediction.frame) <= tolerance_frames
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda index: abs(actual[index].frame - prediction.frame))
            remaining.remove(best)
            true_positive += 1
        by_class[event_type] = _prf(true_positive, len(predicted), len(actual))
        total_tp += true_positive
        total_predicted += len(predicted)
        total_actual += len(actual)
    return {
        "tolerance_frames": tolerance_frames,
        "match_player": match_player,
        "class_wise": by_class,
        "micro": _prf(total_tp, total_predicted, total_actual),
        "macro_f1": sum(float(item["f1"]) for item in by_class.values()) / len(by_class)
        if by_class
        else 0.0,
    }


def action_spotting_dataset_report(
    ground_truth_by_match: Mapping[str, Sequence[Event]],
    predictions_by_match: Mapping[str, Sequence[Event]],
    *,
    tolerance_frames: int = 12,
    match_player: bool = True,
) -> dict[str, object]:
    """Aggregate spotting counts per match so events from different games can never match."""
    match_ids = sorted(set(ground_truth_by_match) | set(predictions_by_match))
    aggregate: dict[str, dict[str, int]] = {}
    match_wise: dict[str, dict[str, float | int]] = {}
    for match_id in match_ids:
        report = action_spotting_report(
            ground_truth_by_match.get(match_id, ()),
            predictions_by_match.get(match_id, ()),
            tolerance_frames=tolerance_frames,
            match_player=match_player,
        )
        micro = report["micro"]
        match_wise[match_id] = {
            "precision": float(micro["precision"]),
            "recall": float(micro["recall"]),
            "f1": float(micro["f1"]),
            "predicted": int(micro["predicted"]),
            "actual": int(micro["actual"]),
        }
        for event_type, metrics in report["class_wise"].items():
            totals = aggregate.setdefault(
                event_type, {"true_positive": 0, "predicted": 0, "actual": 0}
            )
            for key in totals:
                totals[key] += int(metrics[key])
    class_wise = {
        event_type: _prf(values["true_positive"], values["predicted"], values["actual"])
        for event_type, values in sorted(aggregate.items())
    }
    true_positive = sum(int(item["true_positive"]) for item in class_wise.values())
    predicted = sum(int(item["predicted"]) for item in class_wise.values())
    actual = sum(int(item["actual"]) for item in class_wise.values())
    return {
        "tolerance_frames": tolerance_frames,
        "match_player": match_player,
        "match_count": len(match_ids),
        "class_wise": class_wise,
        "micro": _prf(true_positive, predicted, actual),
        "macro_f1": sum(float(item["f1"]) for item in class_wise.values()) / len(class_wise)
        if class_wise
        else 0.0,
        "match_wise": match_wise,
    }


def boundary_f1(
    ground_truth_seconds: Sequence[float],
    predicted_seconds: Sequence[float],
    *,
    tolerance_seconds: float = 1.0,
) -> dict[str, float | int]:
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    remaining = set(range(len(ground_truth_seconds)))
    matched = 0
    for prediction in sorted(predicted_seconds):
        candidates = [
            index
            for index in remaining
            if abs(ground_truth_seconds[index] - prediction) <= tolerance_seconds
        ]
        if candidates:
            best = min(candidates, key=lambda index: abs(ground_truth_seconds[index] - prediction))
            remaining.remove(best)
            matched += 1
    return _prf(matched, len(predicted_seconds), len(ground_truth_seconds))


def top_view_error_report(
    ground_truth: Sequence[tuple[float, float]],
    predictions: Sequence[tuple[float, float]],
) -> dict[str, float | int]:
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have equal length")
    errors = sorted(
        math.hypot(actual_x - predicted_x, actual_y - predicted_y)
        for (actual_x, actual_y), (predicted_x, predicted_y) in zip(ground_truth, predictions)
    )
    if not errors:
        return {"count": 0, "median_error_m": 0.0, "p90_error_m": 0.0}
    p90_index = min(len(errors) - 1, math.ceil(0.9 * len(errors)) - 1)
    return {
        "count": len(errors),
        "median_error_m": median(errors),
        "p90_error_m": errors[p90_index],
    }


def match_bootstrap_interval(
    values_by_match: Mapping[str, Sequence[object]],
    metric: Callable[[Sequence[object]], float],
    *,
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Cluster bootstrap: resample whole matches, never individual sequences."""
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    match_ids = sorted(values_by_match)
    if not match_ids:
        raise ValueError("at least one match is required")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(match_ids) for _ in match_ids]
        observations = [item for match_id in selected for item in values_by_match[match_id]]
        estimates.append(float(metric(observations)))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = estimates[max(0, math.floor(alpha * samples))]
    upper = estimates[min(samples - 1, math.ceil((1.0 - alpha) * samples) - 1)]
    all_observations = [item for match_id in match_ids for item in values_by_match[match_id]]
    return {
        "estimate": float(metric(all_observations)),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "match_count": len(match_ids),
    }
