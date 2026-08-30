"""Deterministic, dependency-free tracking evaluation for Attack-v2.

The module deliberately separates four questions:

* detection quality (precision/recall/F1 and an AP50-compatible summary),
* identity continuity (IDF1),
* higher-order detection/association quality (HOTA-style DetA/AssA), and
* football attributes (team and jersey number).

``bbox_gs_hota_style`` is inspired by SoccerNet GS-HOTA: matching is gated by
all team/jersey attributes that are known in the ground truth, and the normal
HOTA DetA/AssA aggregation is then applied.  It is intentionally *not* called
the official SoccerNet metric.  The official scorer uses calibrated pitch
points, a Gaussian localization similarity, role labels, and TrackEval's exact
matching/alignment rules.  This local metric uses image-space bbox IoU and a
deterministic maximum-cardinality/maximum-IoU assignment.  Release reporting
must therefore run the official evaluator in addition to this diagnostic.

Missing ground-truth identities and attributes are excluded from the relevant
denominators and their coverage is reported.  They are never silently mapped
to a numeric class such as zero.  A missing *prediction* for an available
ground-truth label is an error/abstention, as it should be.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence


PERSON_TYPE = "player"
BALL_TYPE = "ball"
OBJECT_TYPES = frozenset({PERSON_TYPE, BALL_TYPE})
DEFAULT_HOTA_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))

Label = str | int


@dataclass(frozen=True)
class TrackingBox:
    """Axis-aligned image-space box in ``(x1, y1, x2, y2)`` form."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("tracking box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("tracking box must have positive width and height")


@dataclass(frozen=True)
class TrackDetection:
    """One ground-truth or predicted object instance in one video frame.

    ``visibility=False`` means the record is explicitly outside the evaluation
    set.  ``visibility=None`` means visibility was not annotated; the bbox is
    still evaluated, while visibility-specific claims must not be made.
    """

    match_id: str
    frame_index: int
    bbox: TrackingBox
    object_type: str = PERSON_TYPE
    track_id: Label | None = None
    team_id: Label | None = None
    jersey_number: Label | None = None
    visibility: bool | None = True
    confidence: float = 1.0
    period: int = 1

    def __post_init__(self) -> None:
        if not str(self.match_id).strip():
            raise ValueError("match_id must be non-empty")
        if not isinstance(self.bbox, TrackingBox):
            raise TypeError("bbox must be a TrackingBox")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if not isinstance(self.period, int) or isinstance(self.period, bool) or self.period < 1:
            raise ValueError("period must be a positive integer")
        if self.object_type not in OBJECT_TYPES:
            raise ValueError(f"object_type must be one of {sorted(OBJECT_TYPES)}")
        if self.visibility is not True and self.visibility is not False and self.visibility is not None:
            raise ValueError("visibility must be True, False, or None")
        if isinstance(self.confidence, bool):
            raise ValueError("confidence must be finite and between zero and one")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between zero and one")


@dataclass(frozen=True)
class IoUMatch:
    """A matched pair, indexed against the caller's input sequences."""

    ground_truth_index: int
    prediction_index: int
    iou: float


@dataclass(frozen=True)
class DetectionSummary:
    ground_truth: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    ap50: float | None
    mean_matched_iou: float | None


@dataclass(frozen=True)
class IdentitySummary:
    labelled_ground_truth_detections: int
    identity_evaluable_predictions: int
    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int
    id_precision: float | None
    id_recall: float | None
    idf1: float | None
    ground_truth_identity_coverage: float | None


@dataclass(frozen=True)
class AttributeSummary:
    """Accuracy and coverage for team or jersey labels on localized players."""

    labelled_ground_truth_detections: int
    matched_labelled_detections: int
    predictions_with_label: int
    correct: int
    accuracy: float | None
    prediction_coverage: float | None
    matched_detection_coverage: float | None


@dataclass(frozen=True)
class HOTAThresholdSummary:
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    deta: float | None
    assa: float | None
    hota: float | None
    loca: float | None


@dataclass(frozen=True)
class HOTASummary:
    thresholds: tuple[HOTAThresholdSummary, ...]
    hota: float | None
    deta: float | None
    assa: float | None
    loca: float | None


@dataclass(frozen=True)
class TrackingEvaluationReport:
    """All tracking metrics at one frozen operating point."""

    primary_iou_threshold: float
    player_detection: DetectionSummary
    ball_detection: DetectionSummary
    player_identity: IdentitySummary
    player_hota: HOTASummary
    team: AttributeSummary
    jersey: AttributeSummary
    ball_track_recall: float | None
    ball_track_true_positives: int
    visible_ball_ground_truth: int
    bbox_gs_hota_style: HOTASummary
    gs_team_label_coverage: float | None
    gs_jersey_label_coverage: float | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _IndexedDetection:
    index: int
    detection: TrackDetection


@dataclass(frozen=True)
class _IndexedMatch:
    ground_truth: _IndexedDetection
    prediction: _IndexedDetection
    iou: float


def bbox_iou(first: TrackingBox, second: TrackingBox) -> float:
    """Return standard image-space intersection over union."""

    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _normalise_label(value: Label | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in {"", "-1", "?", "nan", "none", "null", "unk", "unknown"}:
        return None
    return text.casefold()


def _validate_threshold(threshold: float) -> None:
    if not math.isfinite(float(threshold)) or not 0.0 < threshold <= 1.0:
        raise ValueError("IoU thresholds must be finite and in (0, 1]")


def _mean(values: Iterable[float | None]) -> float | None:
    available = tuple(float(value) for value in values if value is not None)
    return sum(available) / len(available) if available else None


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _f1(precision: float | None, recall: float | None, *, has_items: bool) -> float | None:
    if not has_items:
        return None
    if precision is None or recall is None:
        return 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _visible(items: Sequence[TrackDetection], object_type: str) -> tuple[_IndexedDetection, ...]:
    return tuple(
        _IndexedDetection(index, item)
        for index, item in enumerate(items)
        if item.object_type == object_type and item.visibility is not False
    )


def _frame_key(item: TrackDetection) -> tuple[str, int, int]:
    return (str(item.match_id), int(item.period), int(item.frame_index))


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> tuple[tuple[int, int], ...]:
    """Solve a rectangular maximum-weight assignment with the Hungarian method.

    The implementation pads the matrix with zero-weight dummy rows/columns and
    uses the classic O(n^3) shortest augmenting-path formulation.  Returned
    pairs refer only to the original rectangular matrix.
    """

    row_count = len(weights)
    column_count = len(weights[0]) if row_count else 0
    if row_count == 0 or column_count == 0:
        return ()
    if any(len(row) != column_count for row in weights):
        raise ValueError("assignment matrix must be rectangular")

    size = max(row_count, column_count)
    maximum = max(max(float(value) for value in row) for row in weights)
    costs = [
        [
            maximum - (float(weights[row][column]) if row < row_count and column < column_count else 0.0)
            for column in range(size)
        ]
        for row in range(size)
    ]

    # 1-indexed Hungarian minimization, adapted from the standard potential
    # formulation.  Strict comparisons make tie handling deterministic.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    column_for_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        column_for_row[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        current_column = 0
        while True:
            used[current_column] = True
            current_row = column_for_row[current_column]
            delta = math.inf
            next_column = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                reduced = costs[current_row - 1][column - 1] - u[current_row] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(size + 1):
                if used[column]:
                    u[column_for_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_for_row[current_column] == 0:
                break
        while True:
            previous_column = predecessor[current_column]
            column_for_row[current_column] = column_for_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assigned_column = [-1] * size
    for column in range(1, size + 1):
        if column_for_row[column]:
            assigned_column[column_for_row[column] - 1] = column - 1
    return tuple(
        (row, column)
        for row, column in enumerate(assigned_column[:row_count])
        if 0 <= column < column_count
    )


def _attributes_compatible(ground_truth: TrackDetection, prediction: TrackDetection) -> bool:
    for ground_truth_value, prediction_value in (
        (ground_truth.team_id, prediction.team_id),
        (ground_truth.jersey_number, prediction.jersey_number),
    ):
        expected = _normalise_label(ground_truth_value)
        if expected is not None and _normalise_label(prediction_value) != expected:
            return False
    return True


def _match_group(
    ground_truth: Sequence[_IndexedDetection],
    predictions: Sequence[_IndexedDetection],
    iou_threshold: float,
    compatibility: Callable[[TrackDetection, TrackDetection], bool] | None = None,
) -> tuple[_IndexedMatch, ...]:
    if not ground_truth or not predictions:
        return ()
    size = max(len(ground_truth), len(predictions))
    cardinality_bonus = float(size + 1)
    overlaps: list[list[float]] = []
    weights: list[list[float]] = []
    for target in ground_truth:
        overlap_row: list[float] = []
        weight_row: list[float] = []
        for prediction in predictions:
            overlap = bbox_iou(target.detection.bbox, prediction.detection.bbox)
            compatible = compatibility is None or compatibility(target.detection, prediction.detection)
            eligible = overlap >= iou_threshold and compatible
            overlap_row.append(overlap)
            # The cardinality term is larger than the maximum possible sum of
            # IoU improvements.  Assignment therefore maximizes TP count first
            # and total IoU second.
            weight_row.append(cardinality_bonus + overlap if eligible else 0.0)
        overlaps.append(overlap_row)
        weights.append(weight_row)

    matches = []
    for target_index, prediction_index in _maximum_weight_assignment(weights):
        if weights[target_index][prediction_index] <= 0.0:
            continue
        matches.append(
            _IndexedMatch(
                ground_truth[target_index],
                predictions[prediction_index],
                overlaps[target_index][prediction_index],
            )
        )
    return tuple(matches)


def _match_indexed(
    ground_truth: Sequence[_IndexedDetection],
    predictions: Sequence[_IndexedDetection],
    iou_threshold: float,
    compatibility: Callable[[TrackDetection, TrackDetection], bool] | None = None,
) -> tuple[_IndexedMatch, ...]:
    targets_by_frame: dict[tuple[str, int, int], list[_IndexedDetection]] = {}
    predictions_by_frame: dict[tuple[str, int, int], list[_IndexedDetection]] = {}
    for item in ground_truth:
        targets_by_frame.setdefault(_frame_key(item.detection), []).append(item)
    for item in predictions:
        predictions_by_frame.setdefault(_frame_key(item.detection), []).append(item)

    matches: list[_IndexedMatch] = []
    for frame in sorted(set(targets_by_frame) | set(predictions_by_frame)):
        matches.extend(
            _match_group(
                targets_by_frame.get(frame, ()),
                predictions_by_frame.get(frame, ()),
                iou_threshold,
                compatibility,
            )
        )
    return tuple(matches)


def optimal_iou_matches(
    ground_truth: Iterable[TrackDetection],
    predictions: Iterable[TrackDetection],
    iou_threshold: float = 0.5,
    *,
    require_known_attributes: bool = False,
) -> tuple[IoUMatch, ...]:
    """Match like-typed records within each match/period/frame.

    Matching is globally optimal within a frame: first for match cardinality,
    then for summed IoU.  Explicitly invisible records are omitted.  When
    ``require_known_attributes`` is true, every team/jersey label available in
    the ground truth must match; missing ground-truth attributes are ignored.
    """

    _validate_threshold(iou_threshold)
    targets = tuple(ground_truth)
    estimates = tuple(predictions)
    all_matches: list[_IndexedMatch] = []
    compatibility = _attributes_compatible if require_known_attributes else None
    for object_type in sorted(OBJECT_TYPES):
        all_matches.extend(
            _match_indexed(
                _visible(targets, object_type),
                _visible(estimates, object_type),
                iou_threshold,
                compatibility,
            )
        )
    return tuple(
        IoUMatch(match.ground_truth.index, match.prediction.index, match.iou)
        for match in sorted(
            all_matches,
            key=lambda item: (item.ground_truth.index, item.prediction.index),
        )
    )


def _average_precision_50(
    ground_truth: Sequence[_IndexedDetection],
    predictions: Sequence[_IndexedDetection],
) -> float | None:
    if not ground_truth:
        return None
    targets_by_frame: dict[tuple[str, int, int], list[_IndexedDetection]] = {}
    for target in ground_truth:
        targets_by_frame.setdefault(_frame_key(target.detection), []).append(target)
    matched: dict[tuple[str, int, int], set[int]] = {
        frame: set() for frame in targets_by_frame
    }
    ranked_predictions = sorted(
        predictions,
        key=lambda item: (-item.detection.confidence, _frame_key(item.detection), item.index),
    )
    flags: list[int] = []
    for prediction in ranked_predictions:
        frame = _frame_key(prediction.detection)
        candidates = [
            (target.index, bbox_iou(target.detection.bbox, prediction.detection.bbox))
            for target in targets_by_frame.get(frame, ())
            if target.index not in matched.setdefault(frame, set())
        ]
        if not candidates:
            flags.append(0)
            continue
        target_index, overlap = max(candidates, key=lambda item: (item[1], -item[0]))
        if overlap >= 0.5:
            matched[frame].add(target_index)
            flags.append(1)
        else:
            flags.append(0)

    cumulative_true_positives = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for rank, flag in enumerate(flags, start=1):
        cumulative_true_positives += flag
        precisions.append(cumulative_true_positives / rank)
        recalls.append(cumulative_true_positives / len(ground_truth))
    return sum(
        max(
            (precision for precision, recall in zip(precisions, recalls) if recall >= level),
            default=0.0,
        )
        for level in (index / 100.0 for index in range(101))
    ) / 101.0


def _detection_summary(
    ground_truth: Sequence[_IndexedDetection],
    predictions: Sequence[_IndexedDetection],
    matches: Sequence[_IndexedMatch],
) -> DetectionSummary:
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    precision = _ratio(true_positives, len(predictions))
    recall = _ratio(true_positives, len(ground_truth))
    return DetectionSummary(
        ground_truth=len(ground_truth),
        predictions=len(predictions),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall, has_items=bool(ground_truth or predictions)),
        ap50=_average_precision_50(ground_truth, predictions),
        mean_matched_iou=_mean(match.iou for match in matches),
    )


def _identity_summary(
    ground_truth: Sequence[_IndexedDetection],
    predictions: Sequence[_IndexedDetection],
    matches: Sequence[_IndexedMatch],
) -> IdentitySummary:
    labelled_targets = tuple(
        target
        for target in ground_truth
        if _normalise_label(target.detection.track_id) is not None
    )
    matches_by_prediction = {match.prediction.index: match for match in matches}
    # A prediction localized to a target whose identity was not annotated is
    # ignored for ID metrics; it is not an identity false positive.
    absorbed_by_unknown_target = {
        match.prediction.index
        for match in matches
        if _normalise_label(match.ground_truth.detection.track_id) is None
    }
    identity_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.index not in absorbed_by_unknown_target
    )

    target_ids = sorted(
        {
            _normalise_label(target.detection.track_id)
            for target in labelled_targets
            if _normalise_label(target.detection.track_id) is not None
        }
    )
    prediction_ids = sorted(
        {
            _normalise_label(prediction.detection.track_id)
            for prediction in identity_predictions
            if _normalise_label(prediction.detection.track_id) is not None
        }
    )
    pair_counts: dict[tuple[str, str], int] = {}
    for prediction in identity_predictions:
        match = matches_by_prediction.get(prediction.index)
        if match is None:
            continue
        target_id = _normalise_label(match.ground_truth.detection.track_id)
        prediction_id = _normalise_label(prediction.detection.track_id)
        if target_id is None or prediction_id is None:
            continue
        pair_counts[(target_id, prediction_id)] = pair_counts.get((target_id, prediction_id), 0) + 1

    weights = [
        [float(pair_counts.get((target_id, prediction_id), 0)) for prediction_id in prediction_ids]
        for target_id in target_ids
    ]
    id_true_positives = sum(
        int(weights[row][column])
        for row, column in _maximum_weight_assignment(weights)
        if weights[row][column] > 0.0
    ) if weights and prediction_ids else 0
    id_false_negatives = len(labelled_targets) - id_true_positives
    id_false_positives = len(identity_predictions) - id_true_positives
    id_precision = _ratio(id_true_positives, len(identity_predictions))
    id_recall = _ratio(id_true_positives, len(labelled_targets))
    return IdentitySummary(
        labelled_ground_truth_detections=len(labelled_targets),
        identity_evaluable_predictions=len(identity_predictions),
        id_true_positives=id_true_positives,
        id_false_positives=id_false_positives,
        id_false_negatives=id_false_negatives,
        id_precision=id_precision,
        id_recall=id_recall,
        idf1=_f1(
            id_precision,
            id_recall,
            has_items=bool(labelled_targets or identity_predictions),
        ) if labelled_targets else None,
        ground_truth_identity_coverage=_ratio(len(labelled_targets), len(ground_truth)),
    )


def _attribute_summary(
    ground_truth: Sequence[_IndexedDetection],
    matches: Sequence[_IndexedMatch],
    attribute: str,
) -> AttributeSummary:
    labelled_targets = tuple(
        target
        for target in ground_truth
        if _normalise_label(getattr(target.detection, attribute)) is not None
    )
    labelled_target_indices = {target.index for target in labelled_targets}
    labelled_matches = tuple(
        match for match in matches if match.ground_truth.index in labelled_target_indices
    )
    predictions_with_label = sum(
        _normalise_label(getattr(match.prediction.detection, attribute)) is not None
        for match in labelled_matches
    )
    correct = sum(
        _normalise_label(getattr(match.prediction.detection, attribute))
        == _normalise_label(getattr(match.ground_truth.detection, attribute))
        for match in labelled_matches
    )
    return AttributeSummary(
        labelled_ground_truth_detections=len(labelled_targets),
        matched_labelled_detections=len(labelled_matches),
        predictions_with_label=predictions_with_label,
        correct=correct,
        accuracy=_ratio(correct, len(labelled_matches)),
        prediction_coverage=_ratio(predictions_with_label, len(labelled_matches)),
        matched_detection_coverage=_ratio(len(labelled_matches), len(labelled_targets)),
    )


def _hota_at_threshold(
    ground_truth: Sequence[_IndexedDetection],
    predictions: Sequence[_IndexedDetection],
    matches: Sequence[_IndexedMatch],
    iou_threshold: float,
) -> HOTAThresholdSummary:
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    detection_denominator = true_positives + false_positives + false_negatives
    deta = _ratio(true_positives, detection_denominator)

    labelled_target_count = sum(
        _normalise_label(target.detection.track_id) is not None for target in ground_truth
    )
    target_track_counts: dict[str, int] = {}
    for target in ground_truth:
        target_id = _normalise_label(target.detection.track_id)
        if target_id is not None:
            target_track_counts[target_id] = target_track_counts.get(target_id, 0) + 1

    # Ignore predictions localized to an identity-unlabelled ground-truth box.
    absorbed_by_unknown_target = {
        match.prediction.index
        for match in matches
        if _normalise_label(match.ground_truth.detection.track_id) is None
    }
    prediction_track_counts: dict[str, int] = {}
    for prediction in predictions:
        if prediction.index in absorbed_by_unknown_target:
            continue
        prediction_id = _normalise_label(prediction.detection.track_id)
        if prediction_id is not None:
            prediction_track_counts[prediction_id] = prediction_track_counts.get(prediction_id, 0) + 1

    pair_counts: dict[tuple[str, str], int] = {}
    association_matches = []
    for match in matches:
        target_id = _normalise_label(match.ground_truth.detection.track_id)
        if target_id is None:
            continue
        association_matches.append(match)
        prediction_id = _normalise_label(match.prediction.detection.track_id)
        if prediction_id is not None:
            pair_counts[(target_id, prediction_id)] = pair_counts.get((target_id, prediction_id), 0) + 1

    if labelled_target_count == 0:
        assa = None
    elif not association_matches:
        assa = 0.0
    else:
        association_total = 0.0
        for match in association_matches:
            target_id = _normalise_label(match.ground_truth.detection.track_id)
            prediction_id = _normalise_label(match.prediction.detection.track_id)
            if target_id is None or prediction_id is None:
                continue
            true_positive_associations = pair_counts[(target_id, prediction_id)]
            denominator = (
                target_track_counts[target_id]
                + prediction_track_counts[prediction_id]
                - true_positive_associations
            )
            association_total += true_positive_associations / denominator if denominator else 0.0
        # Missing predicted IDs contribute zero while remaining in the divisor.
        assa = association_total / len(association_matches)

    hota = math.sqrt(deta * assa) if deta is not None and assa is not None else None
    return HOTAThresholdSummary(
        iou_threshold=iou_threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        deta=deta,
        assa=assa,
        hota=hota,
        loca=_mean(match.iou for match in matches),
    )


def _hota_summary(
    ground_truth: Sequence[_IndexedDetection],
    predictions: Sequence[_IndexedDetection],
    thresholds: Sequence[float],
    compatibility: Callable[[TrackDetection, TrackDetection], bool] | None = None,
) -> HOTASummary:
    results = tuple(
        _hota_at_threshold(
            ground_truth,
            predictions,
            _match_indexed(ground_truth, predictions, threshold, compatibility),
            threshold,
        )
        for threshold in thresholds
    )
    return HOTASummary(
        thresholds=results,
        hota=_mean(result.hota for result in results),
        deta=_mean(result.deta for result in results),
        assa=_mean(result.assa for result in results),
        loca=_mean(result.loca for result in results),
    )


def evaluate_tracking(
    ground_truth: Iterable[TrackDetection],
    predictions: Iterable[TrackDetection],
    *,
    primary_iou_threshold: float = 0.5,
    hota_iou_thresholds: Iterable[float] = DEFAULT_HOTA_IOU_THRESHOLDS,
) -> TrackingEvaluationReport:
    """Evaluate player and ball detections/tracks without external packages.

    HOTA is evaluated for players. ``ball_track_recall`` is the fraction of
    visible ground-truth ball instances that are IoU-matched to a prediction
    carrying a non-missing track ID.  Thus it measures both ball localization
    and whether the temporal tracker emitted an identity, while ball detection
    recall remains available separately.
    """

    _validate_threshold(primary_iou_threshold)
    thresholds = tuple(float(value) for value in hota_iou_thresholds)
    if not thresholds:
        raise ValueError("at least one HOTA IoU threshold is required")
    for threshold in thresholds:
        _validate_threshold(threshold)
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("HOTA IoU thresholds must be unique")

    targets = tuple(ground_truth)
    estimates = tuple(predictions)
    target_players = _visible(targets, PERSON_TYPE)
    predicted_players = _visible(estimates, PERSON_TYPE)
    target_balls = _visible(targets, BALL_TYPE)
    predicted_balls = _visible(estimates, BALL_TYPE)

    player_matches = _match_indexed(
        target_players, predicted_players, primary_iou_threshold
    )
    ball_matches = _match_indexed(target_balls, predicted_balls, primary_iou_threshold)
    ball_track_true_positives = sum(
        _normalise_label(match.prediction.detection.track_id) is not None
        for match in ball_matches
    )
    team_labelled = sum(
        _normalise_label(item.detection.team_id) is not None for item in target_players
    )
    jersey_labelled = sum(
        _normalise_label(item.detection.jersey_number) is not None for item in target_players
    )

    return TrackingEvaluationReport(
        primary_iou_threshold=primary_iou_threshold,
        player_detection=_detection_summary(
            target_players, predicted_players, player_matches
        ),
        ball_detection=_detection_summary(target_balls, predicted_balls, ball_matches),
        player_identity=_identity_summary(
            target_players, predicted_players, player_matches
        ),
        player_hota=_hota_summary(
            target_players, predicted_players, thresholds
        ),
        team=_attribute_summary(target_players, player_matches, "team_id"),
        jersey=_attribute_summary(target_players, player_matches, "jersey_number"),
        ball_track_recall=_ratio(ball_track_true_positives, len(target_balls)),
        ball_track_true_positives=ball_track_true_positives,
        visible_ball_ground_truth=len(target_balls),
        bbox_gs_hota_style=_hota_summary(
            target_players,
            predicted_players,
            thresholds,
            _attributes_compatible,
        ),
        gs_team_label_coverage=_ratio(team_labelled, len(target_players)),
        gs_jersey_label_coverage=_ratio(jersey_labelled, len(target_players)),
        notes=(
            "bbox_gs_hota_style is an Attack-v2 image-space diagnostic, not the official SoccerNet GS-HOTA evaluator",
            "unknown ground-truth identity/team/jersey labels are excluded and coverage is reported",
            "AP50 uses confidence-ranked 101-point interpolation and is compatible in shape, not an official challenge score",
        ),
    )


__all__ = [
    "BALL_TYPE",
    "DEFAULT_HOTA_IOU_THRESHOLDS",
    "PERSON_TYPE",
    "AttributeSummary",
    "DetectionSummary",
    "HOTASummary",
    "HOTAThresholdSummary",
    "IdentitySummary",
    "IoUMatch",
    "TrackDetection",
    "TrackingBox",
    "TrackingEvaluationReport",
    "bbox_iou",
    "evaluate_tracking",
    "optimal_iou_matches",
]
