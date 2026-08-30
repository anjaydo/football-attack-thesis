from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Callable, Literal, Mapping, Sequence


DEFAULT_TIOU_THRESHOLDS = (0.25, 0.5, 0.75)
DEFAULT_REQUIRED_TERMINALS = (
    "shot_attempt",
    "headed_attempt",
    "forced_goal_line_out",
)
_MISSING = object()
_MISSED = "__missed__"
_SPURIOUS = "__spurious__"


def _normalise_identifier(value: object, field: str) -> str:
    if hasattr(value, "value"):
        value = getattr(value, "value")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _normalise_terminal(value: object | None) -> str:
    if value is None:
        return "unknown"
    if hasattr(value, "value"):
        value = getattr(value, "value")
    result = str(value).strip().casefold()
    return result or "unknown"


def _read(value: object, names: Sequence[str], default: object = _MISSING) -> object:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"sequence is missing required field; expected one of {tuple(names)}")


def _read_terminal(value: object) -> object | None:
    direct = _read(value, ("terminal_type", "termination_reason"), default=None)
    if direct is not None:
        return direct
    terminal = _read(value, ("terminal",), default=None)
    if terminal is None:
        return None
    return _read(terminal, ("terminal_type", "event_type", "type"), default=terminal)


def _read_with_possession(value: object, names: Sequence[str]) -> object:
    """Read a field from a sequence or its nested AttackSequenceV2 possession."""
    try:
        return _read(value, names)
    except ValueError:
        possession = _read(value, ("possession",), default=None)
        if possession is None:
            raise
        return _read(possession, names)


@dataclass(frozen=True)
class SequenceSpan:
    """Minimal, version-neutral sequence representation used by the evaluator."""

    sequence_id: str
    match_id: str
    period: str
    stable_team_id: str
    start_sec: float
    end_sec: float
    terminal_type: str = "unknown"

    def __post_init__(self) -> None:
        for field in ("sequence_id", "match_id", "period", "stable_team_id"):
            if not getattr(self, field):
                raise ValueError(f"{field} must not be empty")
        if not math.isfinite(self.start_sec) or not math.isfinite(self.end_sec):
            raise ValueError("sequence boundaries must be finite")
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be greater than or equal to start_sec")

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.match_id, self.period, self.stable_team_id)

    @classmethod
    def from_value(cls, value: object) -> "SequenceSpan":
        if isinstance(value, cls):
            return value
        terminal_value = _read_terminal(value)
        if terminal_value is None:
            possession = _read(value, ("possession",), default=None)
            if possession is not None:
                terminal_value = _read_terminal(possession)
        return cls(
            sequence_id=_normalise_identifier(
                _read(value, ("sequence_id", "attack_id", "possession_id", "annotation_id")),
                "sequence_id",
            ),
            match_id=_normalise_identifier(
                _read_with_possession(value, ("match_id",)),
                "match_id",
            ),
            period=_normalise_identifier(
                _read_with_possession(value, ("period", "half")),
                "period",
            ),
            stable_team_id=_normalise_identifier(
                _read_with_possession(
                    value,
                    ("stable_team_id", "possession_team_id", "team_id", "team"),
                ),
                "stable_team_id",
            ),
            start_sec=float(_read_with_possession(value, ("start_sec", "start_time"))),
            end_sec=float(_read_with_possession(value, ("end_sec", "end_time"))),
            terminal_type=_normalise_terminal(terminal_value),
        )


@dataclass(frozen=True)
class SequenceMatch:
    ground_truth_id: str
    prediction_id: str
    temporal_iou: float
    start_error_seconds: float
    end_error_seconds: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "ground_truth_id": self.ground_truth_id,
            "prediction_id": self.prediction_id,
            "temporal_iou": self.temporal_iou,
            "start_error_seconds": self.start_error_seconds,
            "end_error_seconds": self.end_error_seconds,
        }


@dataclass(frozen=True)
class SequenceMatchingResult:
    matches: tuple[SequenceMatch, ...]
    unmatched_ground_truth_ids: tuple[str, ...]
    unmatched_prediction_ids: tuple[str, ...]


TerminalCompatibility = Callable[[str, str], bool] | Literal["exact"] | None


def temporal_iou(first: object, second: object) -> float:
    """Return temporal intersection-over-union for two sequence-like values."""
    actual = SequenceSpan.from_value(first)
    predicted = SequenceSpan.from_value(second)
    intersection = max(
        0.0,
        min(actual.end_sec, predicted.end_sec) - max(actual.start_sec, predicted.start_sec),
    )
    union = max(actual.end_sec, predicted.end_sec) - min(actual.start_sec, predicted.start_sec)
    if union == 0.0:
        return 1.0
    return intersection / union


def _compatible_terminal(
    ground_truth: SequenceSpan,
    prediction: SequenceSpan,
    compatibility: TerminalCompatibility,
) -> bool:
    if compatibility is None:
        return True
    if compatibility == "exact":
        return ground_truth.terminal_type == prediction.terminal_type
    if callable(compatibility):
        return bool(compatibility(ground_truth.terminal_type, prediction.terminal_type))
    raise ValueError("terminal_compatibility must be None, 'exact', or callable")


def _coerce_unique(values: Sequence[object], side: str) -> tuple[SequenceSpan, ...]:
    spans = tuple(SequenceSpan.from_value(value) for value in values)
    identifiers = [span.sequence_id for span in spans]
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {side} sequence_id values: {duplicates}")
    return spans


def _hungarian_maximise(weights: Sequence[Sequence[float]]) -> list[int]:
    """Pure-Python square Hungarian solver; result maps each row to a column."""
    size = len(weights)
    if not size:
        return []
    if any(len(row) != size for row in weights):
        raise ValueError("Hungarian weight matrix must be square")
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    epsilon = 1e-12

    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate] - epsilon:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if (
                    minimum[candidate] < delta - epsilon
                    or (abs(minimum[candidate] - delta) <= epsilon and candidate < next_column)
                ):
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            next_column = previous_column[column]
            matched_row[column] = matched_row[next_column]
            column = next_column
            if column == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    return assignment


def match_sequence_spans(
    ground_truth: Sequence[object],
    predictions: Sequence[object],
    *,
    tiou_threshold: float = 0.5,
    terminal_compatibility: TerminalCompatibility = None,
) -> SequenceMatchingResult:
    """Optimally match sequences within exact match/period/stable-team groups.

    The objective first maximises the number of eligible matches and then total
    temporal IoU. Inputs are sorted before solving so ties are deterministic.
    """
    if not 0.0 <= tiou_threshold <= 1.0:
        raise ValueError("tiou_threshold must be between 0 and 1")
    actual = _coerce_unique(ground_truth, "ground-truth")
    predicted = _coerce_unique(predictions, "prediction")
    actual_groups: dict[tuple[str, str, str], list[SequenceSpan]] = defaultdict(list)
    predicted_groups: dict[tuple[str, str, str], list[SequenceSpan]] = defaultdict(list)
    for span in actual:
        actual_groups[span.group_key].append(span)
    for span in predicted:
        predicted_groups[span.group_key].append(span)

    matches: list[SequenceMatch] = []
    for key in sorted(set(actual_groups) & set(predicted_groups)):
        actual_group = sorted(
            actual_groups[key], key=lambda item: (item.start_sec, item.end_sec, item.sequence_id)
        )
        predicted_group = sorted(
            predicted_groups[key], key=lambda item: (item.start_sec, item.end_sec, item.sequence_id)
        )
        size = max(len(actual_group), len(predicted_group))
        cardinality_bonus = float(size + 1)
        weights = [[0.0] * size for _ in range(size)]
        eligible: dict[tuple[int, int], float] = {}
        for actual_index, actual_span in enumerate(actual_group):
            for predicted_index, predicted_span in enumerate(predicted_group):
                if not _compatible_terminal(actual_span, predicted_span, terminal_compatibility):
                    continue
                score = temporal_iou(actual_span, predicted_span)
                if score + 1e-12 < tiou_threshold:
                    continue
                eligible[(actual_index, predicted_index)] = score
                weights[actual_index][predicted_index] = cardinality_bonus + score
        for actual_index, predicted_index in enumerate(_hungarian_maximise(weights)):
            if (actual_index, predicted_index) not in eligible:
                continue
            actual_span = actual_group[actual_index]
            predicted_span = predicted_group[predicted_index]
            matches.append(
                SequenceMatch(
                    ground_truth_id=actual_span.sequence_id,
                    prediction_id=predicted_span.sequence_id,
                    temporal_iou=eligible[(actual_index, predicted_index)],
                    start_error_seconds=predicted_span.start_sec - actual_span.start_sec,
                    end_error_seconds=predicted_span.end_sec - actual_span.end_sec,
                )
            )

    matches.sort(key=lambda item: (item.ground_truth_id, item.prediction_id))
    matched_actual = {item.ground_truth_id for item in matches}
    matched_predicted = {item.prediction_id for item in matches}
    return SequenceMatchingResult(
        matches=tuple(matches),
        unmatched_ground_truth_ids=tuple(
            sorted(span.sequence_id for span in actual if span.sequence_id not in matched_actual)
        ),
        unmatched_prediction_ids=tuple(
            sorted(span.sequence_id for span in predicted if span.sequence_id not in matched_predicted)
        ),
    )


def _prf(true_positive: int, predicted: int, actual: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "actual": actual,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _error_report(errors: Sequence[float]) -> dict[str, float | int | None]:
    absolute = [abs(value) for value in errors]
    if not absolute:
        return {
            "count": 0,
            "mean_abs_error_seconds": None,
            "median_abs_error_seconds": None,
            "p90_abs_error_seconds": None,
            "max_abs_error_seconds": None,
        }
    return {
        "count": len(absolute),
        "mean_abs_error_seconds": mean(absolute),
        "median_abs_error_seconds": median(absolute),
        "p90_abs_error_seconds": _percentile(absolute, 0.9),
        "max_abs_error_seconds": max(absolute),
    }


def _terminal_report(
    actual: Sequence[SequenceSpan],
    predicted: Sequence[SequenceSpan],
    matching: SequenceMatchingResult,
    required_terminals: Sequence[str],
) -> dict[str, object]:
    actual_by_id = {span.sequence_id: span for span in actual}
    predicted_by_id = {span.sequence_id: span for span in predicted}
    actual_counts = Counter(span.terminal_type for span in actual)
    predicted_counts = Counter(span.terminal_type for span in predicted)
    true_positive: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for item in matching.matches:
        actual_terminal = actual_by_id[item.ground_truth_id].terminal_type
        predicted_terminal = predicted_by_id[item.prediction_id].terminal_type
        confusion[actual_terminal][predicted_terminal] += 1
        if actual_terminal == predicted_terminal:
            true_positive[actual_terminal] += 1
    for sequence_id in matching.unmatched_ground_truth_ids:
        confusion[actual_by_id[sequence_id].terminal_type][_MISSED] += 1
    for sequence_id in matching.unmatched_prediction_ids:
        confusion[_SPURIOUS][predicted_by_id[sequence_id].terminal_type] += 1

    classes = sorted(set(actual_counts) | set(predicted_counts))
    class_wise = {
        terminal: _prf(
            true_positive[terminal],
            predicted_counts[terminal],
            actual_counts[terminal],
        )
        for terminal in classes
    }
    required = {_normalise_terminal(value) for value in required_terminals}
    valid_predictions = sum(count for terminal, count in predicted_counts.items() if terminal in required)
    return {
        "class_wise": class_wise,
        "macro_f1": (
            sum(float(metrics["f1"]) for metrics in class_wise.values()) / len(class_wise)
            if class_wise
            else 0.0
        ),
        "confusion": {
            actual_terminal: dict(sorted(predicted_terminals.items()))
            for actual_terminal, predicted_terminals in sorted(confusion.items())
        },
        "required_terminals": sorted(required),
        "allowlisted_prediction_rate": valid_predictions / len(predicted) if predicted else 0.0,
    }


def _segmentation_report(
    actual: Sequence[SequenceSpan],
    predicted: Sequence[SequenceSpan],
    *,
    tiou_threshold: float,
    terminal_compatibility: TerminalCompatibility,
) -> dict[str, float | int]:
    compatible_predictions: Counter[str] = Counter()
    compatible_actual: Counter[str] = Counter()
    for actual_span in actual:
        for predicted_span in predicted:
            if actual_span.group_key != predicted_span.group_key:
                continue
            if not _compatible_terminal(actual_span, predicted_span, terminal_compatibility):
                continue
            if temporal_iou(actual_span, predicted_span) + 1e-12 < tiou_threshold:
                continue
            compatible_predictions[actual_span.sequence_id] += 1
            compatible_actual[predicted_span.sequence_id] += 1
    over = sum(count >= 2 for count in compatible_predictions.values())
    under = sum(count >= 2 for count in compatible_actual.values())
    return {
        "over_segmented_ground_truth": over,
        "under_segmented_predictions": under,
        "over_segmentation_excess": sum(max(0, count - 1) for count in compatible_predictions.values()),
        "under_segmentation_excess": sum(max(0, count - 1) for count in compatible_actual.values()),
        "over_segmentation_rate": over / len(actual) if actual else 0.0,
        "under_segmentation_rate": under / len(predicted) if predicted else 0.0,
    }


def sequence_metrics_at_threshold(
    ground_truth: Sequence[object],
    predictions: Sequence[object],
    *,
    tiou_threshold: float = 0.5,
    boundary_tolerance_seconds: float = 1.0,
    terminal_compatibility: TerminalCompatibility = None,
    required_terminals: Sequence[str] = DEFAULT_REQUIRED_TERMINALS,
) -> dict[str, object]:
    """Evaluate whole sequences at one temporal-IoU operating threshold."""
    if boundary_tolerance_seconds < 0:
        raise ValueError("boundary_tolerance_seconds must be non-negative")
    actual = _coerce_unique(ground_truth, "ground-truth")
    predicted = _coerce_unique(predictions, "prediction")
    matching = match_sequence_spans(
        actual,
        predicted,
        tiou_threshold=tiou_threshold,
        terminal_compatibility=terminal_compatibility,
    )
    counts = _prf(len(matching.matches), len(predicted), len(actual))
    start_true_positive = sum(
        abs(item.start_error_seconds) <= boundary_tolerance_seconds for item in matching.matches
    )
    end_true_positive = sum(
        abs(item.end_error_seconds) <= boundary_tolerance_seconds for item in matching.matches
    )
    temporal_ious = [item.temporal_iou for item in matching.matches]
    report: dict[str, object] = {
        "tiou_threshold": tiou_threshold,
        **counts,
        "median_tiou": median(temporal_ious) if temporal_ious else None,
        "mean_tiou": mean(temporal_ious) if temporal_ious else None,
        "start_boundary": {
            **_prf(start_true_positive, len(predicted), len(actual)),
            "tolerance_seconds": boundary_tolerance_seconds,
            "error": _error_report([item.start_error_seconds for item in matching.matches]),
        },
        "end_boundary": {
            **_prf(end_true_positive, len(predicted), len(actual)),
            "tolerance_seconds": boundary_tolerance_seconds,
            "error": _error_report([item.end_error_seconds for item in matching.matches]),
        },
        "terminal": _terminal_report(actual, predicted, matching, required_terminals),
        "segmentation": _segmentation_report(
            actual,
            predicted,
            tiou_threshold=tiou_threshold,
            terminal_compatibility=terminal_compatibility,
        ),
        "matches": [item.to_dict() for item in matching.matches],
        "unmatched_ground_truth_ids": list(matching.unmatched_ground_truth_ids),
        "unmatched_prediction_ids": list(matching.unmatched_prediction_ids),
    }
    return report


def _bootstrap_metric_values(report: Mapping[str, object]) -> dict[str, float | None]:
    terminal = report["terminal"]
    segmentation = report["segmentation"]
    start = report["start_boundary"]
    end = report["end_boundary"]
    return {
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
        "f1": float(report["f1"]),
        "median_tiou": None if report["median_tiou"] is None else float(report["median_tiou"]),
        "start_boundary_f1": float(start["f1"]),
        "end_boundary_f1": float(end["f1"]),
        "terminal_macro_f1": float(terminal["macro_f1"]),
        "over_segmentation_rate": float(segmentation["over_segmentation_rate"]),
        "under_segmentation_rate": float(segmentation["under_segmentation_rate"]),
    }


def _aggregate_match_reports(
    reports: Sequence[Mapping[str, object]],
    terminal_classes: Sequence[str] = (),
) -> dict[str, float | None]:
    true_positive = sum(int(report["true_positive"]) for report in reports)
    predicted = sum(int(report["predicted"]) for report in reports)
    actual = sum(int(report["actual"]) for report in reports)
    aggregate = _prf(true_positive, predicted, actual)
    start_true_positive = sum(int(report["start_boundary"]["true_positive"]) for report in reports)
    end_true_positive = sum(int(report["end_boundary"]["true_positive"]) for report in reports)
    start = _prf(start_true_positive, predicted, actual)
    end = _prf(end_true_positive, predicted, actual)

    terminal_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"true_positive": 0, "predicted": 0, "actual": 0}
    )
    for terminal in terminal_classes:
        terminal_counts[terminal]
    for report in reports:
        for terminal, metrics in report["terminal"]["class_wise"].items():
            for key in ("true_positive", "predicted", "actual"):
                terminal_counts[terminal][key] += int(metrics[key])
    terminal_f1 = [
        float(_prf(values["true_positive"], values["predicted"], values["actual"])["f1"])
        for values in terminal_counts.values()
    ]
    temporal_ious = [
        float(item["temporal_iou"])
        for report in reports
        for item in report["matches"]
    ]
    over = sum(int(report["segmentation"]["over_segmented_ground_truth"]) for report in reports)
    under = sum(int(report["segmentation"]["under_segmented_predictions"]) for report in reports)
    return {
        "precision": float(aggregate["precision"]),
        "recall": float(aggregate["recall"]),
        "f1": float(aggregate["f1"]),
        "median_tiou": median(temporal_ious) if temporal_ious else None,
        "start_boundary_f1": float(start["f1"]),
        "end_boundary_f1": float(end["f1"]),
        "terminal_macro_f1": mean(terminal_f1) if terminal_f1 else 0.0,
        "over_segmentation_rate": over / actual if actual else 0.0,
        "under_segmentation_rate": under / predicted if predicted else 0.0,
    }


def _bootstrap_interval(values: Sequence[float], confidence: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    return _percentile(values, alpha), _percentile(values, 1.0 - alpha)


def sequence_match_bootstrap_intervals(
    ground_truth: Sequence[object],
    predictions: Sequence[object],
    *,
    tiou_threshold: float = 0.5,
    boundary_tolerance_seconds: float = 1.0,
    terminal_compatibility: TerminalCompatibility = None,
    required_terminals: Sequence[str] = DEFAULT_REQUIRED_TERMINALS,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Match-cluster bootstrap confidence intervals for sequence metrics."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    actual = _coerce_unique(ground_truth, "ground-truth")
    predicted = _coerce_unique(predictions, "prediction")
    match_ids = sorted({span.match_id for span in (*actual, *predicted)})
    if not match_ids:
        raise ValueError("at least one match is required")
    reports = {
        match_id: sequence_metrics_at_threshold(
            tuple(span for span in actual if span.match_id == match_id),
            tuple(span for span in predicted if span.match_id == match_id),
            tiou_threshold=tiou_threshold,
            boundary_tolerance_seconds=boundary_tolerance_seconds,
            terminal_compatibility=terminal_compatibility,
            required_terminals=required_terminals,
        )
        for match_id in match_ids
    }
    full_report = sequence_metrics_at_threshold(
        actual,
        predicted,
        tiou_threshold=tiou_threshold,
        boundary_tolerance_seconds=boundary_tolerance_seconds,
        terminal_compatibility=terminal_compatibility,
        required_terminals=required_terminals,
    )
    estimates = _bootstrap_metric_values(full_report)
    terminal_classes = tuple(full_report["terminal"]["class_wise"])
    sampled_values: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        selected = [reports[rng.choice(match_ids)] for _ in match_ids]
        metrics = _aggregate_match_reports(selected, terminal_classes)
        for metric, value in metrics.items():
            if value is not None and math.isfinite(value):
                sampled_values[metric].append(value)

    intervals: dict[str, object] = {}
    for metric, estimate in estimates.items():
        lower, upper = _bootstrap_interval(sampled_values[metric], confidence)
        intervals[metric] = {
            "estimate": estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "effective_bootstrap_samples": len(sampled_values[metric]),
        }
    return {
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
        "match_count": len(match_ids),
        "metrics": intervals,
    }


def paired_f1_improvement_bootstrap(
    ground_truth: Sequence[object],
    predictions: Sequence[object],
    frozen_baseline_predictions: Sequence[object],
    *,
    tiou_threshold: float = 0.5,
    terminal_compatibility: TerminalCompatibility = None,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Paired match-cluster bootstrap for candidate-minus-baseline F1.

    Candidate and baseline are evaluated on the same resampled matches.  This
    preserves the pairing required by the release gate and avoids pretending
    that two independent confidence intervals establish an improvement.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    actual = _coerce_unique(ground_truth, "ground-truth")
    candidate = _coerce_unique(predictions, "prediction")
    baseline = _coerce_unique(frozen_baseline_predictions, "frozen-baseline")
    match_ids = sorted(
        {span.match_id for span in (*actual, *candidate, *baseline)}
    )
    if not match_ids:
        raise ValueError("at least one match is required")

    def reports_for(values: tuple[SequenceSpan, ...]) -> dict[str, dict[str, object]]:
        return {
            match_id: sequence_metrics_at_threshold(
                tuple(span for span in actual if span.match_id == match_id),
                tuple(span for span in values if span.match_id == match_id),
                tiou_threshold=tiou_threshold,
                terminal_compatibility=terminal_compatibility,
            )
            for match_id in match_ids
        }

    candidate_reports = reports_for(candidate)
    baseline_reports = reports_for(baseline)

    def aggregate_f1(reports: Sequence[Mapping[str, object]]) -> float:
        true_positive = sum(int(report["true_positive"]) for report in reports)
        predicted_count = sum(int(report["predicted"]) for report in reports)
        actual_count = sum(int(report["actual"]) for report in reports)
        return float(_prf(true_positive, predicted_count, actual_count)["f1"])

    candidate_f1 = aggregate_f1(tuple(candidate_reports.values()))
    baseline_f1 = aggregate_f1(tuple(baseline_reports.values()))
    deltas: list[float] = []
    rng = random.Random(seed)
    for _ in range(samples):
        selected = [rng.choice(match_ids) for _ in match_ids]
        candidate_sample = aggregate_f1([candidate_reports[key] for key in selected])
        baseline_sample = aggregate_f1([baseline_reports[key] for key in selected])
        deltas.append(candidate_sample - baseline_sample)
    lower, upper = _bootstrap_interval(deltas, confidence)
    return {
        "status": "computed_from_explicit_frozen_baseline",
        "tiou_threshold": tiou_threshold,
        "candidate_f1": candidate_f1,
        "baseline_f1": baseline_f1,
        "estimate": candidate_f1 - baseline_f1,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
        "match_count": len(match_ids),
        "per_match": {
            match_id: {
                "candidate_f1": float(candidate_reports[match_id]["f1"]),
                "baseline_f1": float(baseline_reports[match_id]["f1"]),
                "delta": (
                    float(candidate_reports[match_id]["f1"])
                    - float(baseline_reports[match_id]["f1"])
                ),
            }
            for match_id in match_ids
        },
    }


def _threshold_key(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def sequence_evaluation_report(
    ground_truth: Sequence[object],
    predictions: Sequence[object],
    *,
    tiou_thresholds: Sequence[float] = DEFAULT_TIOU_THRESHOLDS,
    primary_threshold: float = 0.5,
    boundary_tolerance_seconds: float = 1.0,
    terminal_compatibility: TerminalCompatibility = None,
    required_terminals: Sequence[str] = DEFAULT_REQUIRED_TERMINALS,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    bootstrap_confidence: float = 0.95,
    frozen_baseline_predictions: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build the serialisable Attack-v2 sequence metric registry report."""
    thresholds = tuple(float(value) for value in tiou_thresholds)
    if not thresholds:
        raise ValueError("at least one tiou threshold is required")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("tiou thresholds must be unique")
    if primary_threshold not in thresholds:
        raise ValueError("primary_threshold must be one of tiou_thresholds")
    actual = _coerce_unique(ground_truth, "ground-truth")
    predicted = _coerce_unique(predictions, "prediction")
    baseline = (
        None
        if frozen_baseline_predictions is None
        else _coerce_unique(frozen_baseline_predictions, "frozen-baseline")
    )
    match_ids = sorted({span.match_id for span in (*actual, *predicted)})
    threshold_reports: dict[str, object] = {}
    for threshold in thresholds:
        metrics = sequence_metrics_at_threshold(
            actual,
            predicted,
            tiou_threshold=threshold,
            boundary_tolerance_seconds=boundary_tolerance_seconds,
            terminal_compatibility=terminal_compatibility,
            required_terminals=required_terminals,
        )
        if bootstrap_samples:
            metrics["bootstrap"] = sequence_match_bootstrap_intervals(
                actual,
                predicted,
                tiou_threshold=threshold,
                boundary_tolerance_seconds=boundary_tolerance_seconds,
                terminal_compatibility=terminal_compatibility,
                required_terminals=required_terminals,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                confidence=bootstrap_confidence,
            )
        metrics["per_match"] = {
            match_id: sequence_metrics_at_threshold(
                tuple(span for span in actual if span.match_id == match_id),
                tuple(span for span in predicted if span.match_id == match_id),
                tiou_threshold=threshold,
                boundary_tolerance_seconds=boundary_tolerance_seconds,
                terminal_compatibility=terminal_compatibility,
                required_terminals=required_terminals,
            )
            for match_id in match_ids
        }
        threshold_reports[_threshold_key(threshold)] = metrics
    primary = threshold_reports[_threshold_key(primary_threshold)]
    paired_improvement: dict[str, object]
    if baseline is None:
        paired_improvement = {
            "status": "not_computed_missing_explicit_frozen_baseline",
            "tiou_threshold": primary_threshold,
            "candidate_f1": float(primary["f1"]),
            "baseline_f1": None,
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence": bootstrap_confidence,
            "bootstrap_samples": 0,
            "seed": bootstrap_seed,
            "match_count": len(match_ids),
            "per_match": {},
        }
    elif not bootstrap_samples:
        candidate_f1 = float(primary["f1"])
        baseline_metrics = sequence_metrics_at_threshold(
            actual,
            baseline,
            tiou_threshold=primary_threshold,
            boundary_tolerance_seconds=boundary_tolerance_seconds,
            terminal_compatibility=terminal_compatibility,
            required_terminals=required_terminals,
        )
        paired_improvement = {
            "status": "point_estimate_only_bootstrap_disabled",
            "tiou_threshold": primary_threshold,
            "candidate_f1": candidate_f1,
            "baseline_f1": float(baseline_metrics["f1"]),
            "estimate": candidate_f1 - float(baseline_metrics["f1"]),
            "ci_lower": None,
            "ci_upper": None,
            "confidence": bootstrap_confidence,
            "bootstrap_samples": 0,
            "seed": bootstrap_seed,
            "match_count": len(match_ids),
            "per_match": {},
        }
    else:
        paired_improvement = paired_f1_improvement_bootstrap(
            actual,
            predicted,
            baseline,
            tiou_threshold=primary_threshold,
            terminal_compatibility=terminal_compatibility,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence=bootstrap_confidence,
        )
    return {
        "schema_version": "attack-sequence-evaluation/v2",
        "primary_threshold": primary_threshold,
        "primary": primary,
        "thresholds": threshold_reports,
        "paired_f1_improvement": paired_improvement,
        "metric_registry": ATTACK_V2_METRIC_REGISTRY,
    }


def select_balanced_operating_point(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select max F1, then min |precision-recall|, then max min(P, R).

    Input order is the final deterministic tie-break, allowing callers to make
    an explicit preference (for example, lower confidence threshold first).
    """
    if not candidates:
        raise ValueError("at least one operating-point candidate is required")
    scored: list[tuple[tuple[float, float, float, int], dict[str, object]]] = []
    for index, candidate in enumerate(candidates):
        precision = float(candidate["precision"])
        recall = float(candidate["recall"])
        if not 0.0 <= precision <= 1.0 or not 0.0 <= recall <= 1.0:
            raise ValueError("precision and recall must be between 0 and 1")
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        enriched = dict(candidate)
        enriched.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "selection_rank": {
                    "absolute_precision_recall_gap": abs(precision - recall),
                    "minimum_precision_recall": min(precision, recall),
                },
            }
        )
        key = (f1, -abs(precision - recall), min(precision, recall), -index)
        scored.append((key, enriched))
    return max(scored, key=lambda item: item[0])[1]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    report_path: str
    comparator: Literal[">=", "<=", ">", "<", "=="]
    threshold: float
    description: str


ATTACK_V2_METRIC_REGISTRY: dict[str, dict[str, object]] = {
    "attack_precision_tiou_0_5": {
        "path": "thresholds.0.5.precision",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 0.75,
    },
    "attack_recall_tiou_0_5": {
        "path": "thresholds.0.5.recall",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 0.75,
    },
    "attack_f1_tiou_0_5": {
        "path": "thresholds.0.5.f1",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 0.80,
    },
    "start_boundary_f1_1s": {
        "path": "thresholds.0.5.start_boundary.f1",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 0.80,
    },
    "end_boundary_f1_1s": {
        "path": "thresholds.0.5.end_boundary.f1",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 0.80,
    },
    "median_tiou": {
        "path": "thresholds.0.5.median_tiou",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 0.80,
    },
    "over_segmentation_rate": {
        "path": "thresholds.0.5.segmentation.over_segmentation_rate",
        "unit": "ratio",
        "direction": "lower_is_better",
        "target": 0.10,
    },
    "under_segmentation_rate": {
        "path": "thresholds.0.5.segmentation.under_segmentation_rate",
        "unit": "ratio",
        "direction": "lower_is_better",
        "target": 0.10,
    },
    "terminal_macro_f1": {
        "path": "thresholds.0.5.terminal.macro_f1",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 0.85,
    },
    "allowlisted_prediction_rate": {
        "path": "thresholds.0.5.terminal.allowlisted_prediction_rate",
        "unit": "ratio",
        "direction": "higher_is_better",
        "target": 1.0,
    },
    "paired_f1_improvement_ci_lower": {
        "path": "paired_f1_improvement.ci_lower",
        "unit": "f1_delta",
        "direction": "higher_is_better_strict",
        "target": 0.0,
    },
}


def attack_v2_acceptance_specs(
    required_terminals: Sequence[str] = DEFAULT_REQUIRED_TERMINALS,
) -> tuple[MetricSpec, ...]:
    specs: list[MetricSpec] = []
    for key, definition in ATTACK_V2_METRIC_REGISTRY.items():
        direction = definition["direction"]
        comparator: Literal[">=", "<=", ">"] = (
            ">"
            if direction == "higher_is_better_strict"
            else ">="
            if direction == "higher_is_better"
            else "<="
        )
        specs.append(
            MetricSpec(
                key=key,
                report_path=str(definition["path"]),
                comparator=comparator,
                threshold=float(definition["target"]),
                description=key.replace("_", " "),
            )
        )
    for terminal in required_terminals:
        terminal_name = _normalise_terminal(terminal)
        specs.append(
            MetricSpec(
                key=f"{terminal_name}_recall",
                report_path=f"thresholds.0.5.terminal.class_wise.{terminal_name}.recall",
                comparator=">=",
                threshold=0.90,
                description=f"Recall for required terminal {terminal_name}",
            )
        )
    return tuple(specs)


def _path_value(report: Mapping[str, object], path: str) -> object:
    current: object = report
    components = path.split(".")
    index = 0
    while index < len(components):
        if not isinstance(current, Mapping):
            return _MISSING
        matched_key: str | None = None
        matched_end = index
        # Metric paths may address decimal threshold keys such as ``0.5``.
        # Prefer the longest actual mapping key rather than interpreting the
        # decimal point as a nesting separator.
        for end in range(len(components), index, -1):
            candidate = ".".join(components[index:end])
            if candidate in current:
                matched_key = candidate
                matched_end = end
                break
        if matched_key is None:
            return _MISSING
        current = current[matched_key]
        index = matched_end
    return current


def _compare(observed: float, comparator: str, threshold: float) -> bool:
    return {
        ">=": observed >= threshold,
        "<=": observed <= threshold,
        ">": observed > threshold,
        "<": observed < threshold,
        "==": math.isclose(observed, threshold, rel_tol=1e-12, abs_tol=1e-12),
    }[comparator]


def evaluate_acceptance_gates(
    report: Mapping[str, object],
    specs: Sequence[MetricSpec] | None = None,
) -> dict[str, object]:
    """Evaluate explicit numeric acceptance gates without treating missing as zero."""
    selected_specs = tuple(specs) if specs is not None else attack_v2_acceptance_specs()
    results: dict[str, object] = {}
    passed_count = 0
    for spec in selected_specs:
        value = _path_value(report, spec.report_path)
        valid = (
            value is not _MISSING
            and value is not None
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
        observed = float(value) if valid else None
        passed = bool(valid and _compare(observed, spec.comparator, spec.threshold))
        passed_count += int(passed)
        results[spec.key] = {
            "path": spec.report_path,
            "observed": observed,
            "comparator": spec.comparator,
            "threshold": spec.threshold,
            "passed": passed,
            "description": spec.description,
        }
    return {
        "passed": passed_count == len(selected_specs),
        "passed_count": passed_count,
        "gate_count": len(selected_specs),
        "gates": results,
    }


def report_to_json(report: Mapping[str, object], *, indent: int = 2) -> str:
    """Canonical JSON helper for persisted evaluation artifacts."""
    return json.dumps(report, indent=indent, sort_keys=True, allow_nan=False)
