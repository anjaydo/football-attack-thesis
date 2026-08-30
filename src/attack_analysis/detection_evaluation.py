from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .observations import BoundingBox, FrameObservation
from .quality import read_observations_jsonl


PERSON_TYPES = {"player", "goalkeeper", "referee"}


@dataclass(frozen=True)
class ClassDetectionMetrics:
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float | None
    mean_center_error_px: float | None
    ap50: float
    map_50_95: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_error(first: BoundingBox, second: BoundingBox) -> float:
    first_center = ((first.x1 + first.x2) / 2.0, (first.y1 + first.y2) / 2.0)
    second_center = ((second.x1 + second.x2) / 2.0, (second.y1 + second.y2) / 2.0)
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def _metrics(
    frame_pairs: Iterable[
        tuple[list[tuple[float, BoundingBox]], list[BoundingBox]]
    ],
    iou_threshold: float,
) -> ClassDetectionMetrics:
    pairs = tuple(frame_pairs)
    prediction_count = 0
    ground_truth_count = 0
    true_positives = 0
    matched_ious: list[float] = []
    center_errors: list[float] = []
    for predictions, targets in pairs:
        prediction_count += len(predictions)
        ground_truth_count += len(targets)
        unmatched = set(range(len(targets)))
        for _, prediction in sorted(predictions, key=lambda item: item[0], reverse=True):
            candidates = [
                (index, _box_iou(prediction, targets[index])) for index in unmatched
            ]
            if not candidates:
                continue
            target_index, overlap = max(candidates, key=lambda item: item[1])
            if overlap < iou_threshold:
                continue
            unmatched.remove(target_index)
            true_positives += 1
            matched_ious.append(overlap)
            center_errors.append(_center_error(prediction, targets[target_index]))

    false_positives = prediction_count - true_positives
    false_negatives = ground_truth_count - true_positives
    precision = true_positives / prediction_count if prediction_count else 0.0
    recall = true_positives / ground_truth_count if ground_truth_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    aps = [_average_precision(pairs, 0.5 + 0.05 * index) for index in range(10)]
    return ClassDetectionMetrics(
        ground_truth=ground_truth_count,
        predictions=prediction_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_matched_iou=(
            round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else None
        ),
        mean_center_error_px=(
            round(sum(center_errors) / len(center_errors), 3) if center_errors else None
        ),
        ap50=round(aps[0], 4),
        map_50_95=round(sum(aps) / len(aps), 4),
    )


def _average_precision(
    frame_pairs: tuple[tuple[list[tuple[float, BoundingBox]], list[BoundingBox]], ...],
    iou_threshold: float,
) -> float:
    ground_truth_count = sum(len(targets) for _, targets in frame_pairs)
    if ground_truth_count == 0:
        return 0.0
    predictions = sorted(
        (
            (confidence, frame_index, box)
            for frame_index, (items, _) in enumerate(frame_pairs)
            for confidence, box in items
        ),
        key=lambda item: (-item[0], item[1]),
    )
    matched: dict[int, set[int]] = {index: set() for index in range(len(frame_pairs))}
    true_positive_flags: list[int] = []
    false_positive_flags: list[int] = []
    for _, frame_index, prediction in predictions:
        targets = frame_pairs[frame_index][1]
        candidates = [
            (target_index, _box_iou(prediction, target))
            for target_index, target in enumerate(targets)
            if target_index not in matched[frame_index]
        ]
        if not candidates:
            true_positive_flags.append(0)
            false_positive_flags.append(1)
            continue
        target_index, overlap = max(candidates, key=lambda item: item[1])
        if overlap >= iou_threshold:
            matched[frame_index].add(target_index)
            true_positive_flags.append(1)
            false_positive_flags.append(0)
        else:
            true_positive_flags.append(0)
            false_positive_flags.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for true_positive, false_positive in zip(true_positive_flags, false_positive_flags):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        recalls.append(cumulative_tp / ground_truth_count)
        precisions.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))
    return sum(
        max(
            (precision for recall, precision in zip(recalls, precisions) if recall >= level),
            default=0.0,
        )
        for level in (index / 100.0 for index in range(101))
    ) / 101.0


def _ground_truth_box(annotation: dict[str, object]) -> BoundingBox | None:
    raw = annotation.get("bbox_image")
    if not isinstance(raw, dict):
        return None
    x = float(raw["x"])
    y = float(raw["y"])
    return BoundingBox(x, y, x + float(raw["w"]), y + float(raw["h"]))


def evaluate_soccernet_observations(
    observations: Iterable[FrameObservation],
    labels_path: Path,
    start_frame_number: int = 1,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    if start_frame_number < 1:
        raise ValueError("start_frame_number must be positive")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    category_names = {
        int(item["id"]): str(item["name"]).lower()
        for item in labels.get("categories", [])
    }
    image_names = {
        str(item.get("image_id", item.get("id"))): Path(str(item["file_name"])).name
        for item in labels.get("images", [])
    }
    targets_by_file: dict[str, dict[str, list[BoundingBox]]] = {}
    for annotation in labels.get("annotations", []):
        if annotation.get("supercategory", "object") != "object":
            continue
        box = _ground_truth_box(annotation)
        image_name = image_names.get(str(annotation.get("image_id")))
        category = category_names.get(int(annotation.get("category_id", -1)), "")
        if box is None or image_name is None:
            continue
        metric_class = "person" if category in PERSON_TYPES else category
        if metric_class not in {"person", "ball"}:
            continue
        targets_by_file.setdefault(image_name, {"person": [], "ball": []})[
            metric_class
        ].append(box)

    person_pairs = []
    ball_pairs = []
    sequence = tuple(observations)
    for frame in sequence:
        file_name = f"{start_frame_number + frame.frame_index:06d}.jpg"
        targets = targets_by_file.get(file_name, {"person": [], "ball": []})
        person_predictions = [
            (item.confidence, item.bbox)
            for item in frame.objects
            if item.object_type in PERSON_TYPES
        ]
        ball_predictions = [
            (item.confidence, item.bbox)
            for item in frame.objects
            if item.object_type == "ball"
        ]
        person_pairs.append((person_predictions, targets["person"]))
        ball_pairs.append((ball_predictions, targets["ball"]))

    person = _metrics(person_pairs, iou_threshold)
    ball = _metrics(ball_pairs, iou_threshold)
    return {
        "labels": str(labels_path.resolve()),
        "frame_count": len(sequence),
        "start_frame_number": start_frame_number,
        "iou_threshold": iou_threshold,
        "person": person.to_dict(),
        "ball": ball.to_dict(),
    }


def evaluate_soccernet_jsonl(
    observations_path: Path,
    labels_path: Path,
    start_frame_number: int = 1,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    return evaluate_soccernet_observations(
        read_observations_jsonl(observations_path),
        labels_path,
        start_frame_number,
        iou_threshold,
    )


DEFAULT_BALL_THRESHOLDS = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)


def benchmark_ball_confidence(
    observations: Iterable[FrameObservation],
    labels_path: Path,
    thresholds: Iterable[float] = DEFAULT_BALL_THRESHOLDS,
    start_frame_number: int = 1,
    iou_threshold: float = 0.5,
    min_f1: float = 0.80,
    min_precision: float = 0.75,
    min_recall: float = 0.75,
) -> dict[str, object]:
    threshold_values = tuple(sorted(set(float(value) for value in thresholds)))
    if not threshold_values:
        raise ValueError("at least one confidence threshold is required")
    if any(value < 0.0 or value > 1.0 for value in threshold_values):
        raise ValueError("confidence thresholds must be between 0 and 1")
    if any(value < 0.0 or value > 1.0 for value in (min_f1, min_precision, min_recall)):
        raise ValueError("quality gate metrics must be between 0 and 1")

    frames = tuple(observations)
    sweep: list[dict[str, object]] = []
    for threshold in threshold_values:
        filtered_frames = tuple(
            replace(
                frame,
                objects=tuple(
                    item
                    for item in frame.objects
                    if item.object_type != "ball" or item.confidence >= threshold
                ),
            )
            for frame in frames
        )
        metrics = evaluate_soccernet_observations(
            filtered_frames,
            labels_path,
            start_frame_number,
            iou_threshold,
        )["ball"]
        sweep.append({"confidence": threshold, **metrics})

    best = max(
        sweep,
        key=lambda item: (
            float(item["f1"]),
            float(item["recall"]),
            float(item["precision"]),
            float(item["confidence"]),
        ),
    )
    failures = []
    for metric, minimum in (
        ("f1", min_f1),
        ("precision", min_precision),
        ("recall", min_recall),
    ):
        if float(best[metric]) < minimum:
            failures.append(
                f"ball_{metric}={float(best[metric]):.4f} is below {minimum:.4f}"
            )

    return {
        "labels": str(labels_path.resolve()),
        "frame_count": len(frames),
        "start_frame_number": start_frame_number,
        "iou_threshold": iou_threshold,
        "gate": {
            "min_f1": min_f1,
            "min_precision": min_precision,
            "min_recall": min_recall,
        },
        "accepted": not failures,
        "failures": failures,
        "best": best,
        "sweep": sweep,
    }


def benchmark_ball_jsonl(
    observations_path: Path,
    labels_path: Path,
    thresholds: Iterable[float] = DEFAULT_BALL_THRESHOLDS,
    start_frame_number: int = 1,
    iou_threshold: float = 0.5,
    min_f1: float = 0.80,
    min_precision: float = 0.75,
    min_recall: float = 0.75,
) -> dict[str, object]:
    report = benchmark_ball_confidence(
        read_observations_jsonl(observations_path),
        labels_path,
        thresholds,
        start_frame_number,
        iou_threshold,
        min_f1,
        min_precision,
        min_recall,
    )
    report["observations"] = str(observations_path.resolve())
    return report
