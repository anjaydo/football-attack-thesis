from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np

from .geometry import PitchProjector, Point
from .models import PITCH_LENGTH, PITCH_WIDTH


@dataclass(frozen=True)
class DynamicCalibrationConfig:
    source_width: int = 1920
    source_height: int = 1080
    sample_stride: int = 1
    min_correspondences: int = 6
    ransac_threshold_m: float = 2.0
    max_median_error_m: float = 0.5
    max_p90_error_m: float = 1.5
    min_frame_coverage: float = 0.9
    min_inlier_ratio: float = 0.6
    max_interpolation_gap_frames: int = 25
    max_projector_gap_frames: int = 8

    def validate(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source dimensions must be positive")
        if self.sample_stride <= 0 or self.min_correspondences < 6:
            raise ValueError("sample_stride must be positive and at least six points are required")
        if self.ransac_threshold_m <= 0:
            raise ValueError("RANSAC threshold must be positive")
        if self.max_median_error_m <= 0 or self.max_p90_error_m <= 0:
            raise ValueError("calibration error thresholds must be positive")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between 0 and 1")
        if not 0.0 <= self.min_frame_coverage <= 1.0:
            raise ValueError("min_frame_coverage must be between 0 and 1")
        if self.max_interpolation_gap_frames < 0 or self.max_projector_gap_frames < 0:
            raise ValueError("temporal calibration gaps cannot be negative")


@dataclass(frozen=True)
class CalibrationFrame:
    frame_index: int
    correspondences: int
    holdout_points: int
    inlier_ratio: float
    median_error_m: float
    p90_error_m: float
    confidence: float
    valid: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicCalibrationReport:
    source: str
    match_key: str
    start_frame: int
    end_frame: int
    sampled_frames: int
    valid_sampled_frames: int
    projected_frame_coverage: float
    direct_frame_coverage: float
    interpolated_frame_coverage: float
    extrapolated_frame_coverage: float
    unsupported_frames: int
    ineligible_frames: int
    eligible_frames: int
    median_holdout_error_m: float | None
    p90_holdout_error_m: float | None
    median_inlier_ratio: float | None
    status: str
    warnings: tuple[str, ...]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DynamicPitchProjector:
    """Quality-gated homographies with bounded temporal propagation."""

    def __init__(
        self,
        projectors: dict[int, PitchProjector],
        records: dict[int, CalibrationFrame],
        report: DynamicCalibrationReport,
        *,
        max_interpolation_gap_frames: int,
        max_gap_frames: int,
    ) -> None:
        self._projectors = dict(projectors)
        self._records = dict(records)
        self._frames = tuple(sorted(projectors))
        self.report = report
        self.max_interpolation_gap_frames = max_interpolation_gap_frames
        self.max_gap_frames = max_gap_frames

    def _nearest_frame(self, frame_index: int) -> int | None:
        if not self._frames:
            return None
        position = bisect_left(self._frames, frame_index)
        candidates = self._frames[max(0, position - 1) : position + 1]
        nearest = min(candidates, key=lambda value: abs(value - frame_index))
        return nearest if abs(nearest - frame_index) <= self.max_gap_frames else None

    def _bounding_frames(self, frame_index: int) -> tuple[int, int] | None:
        if not self._frames:
            return None
        position = bisect_left(self._frames, frame_index)
        if position < len(self._frames) and self._frames[position] == frame_index:
            return frame_index, frame_index
        left = self._frames[position - 1] if position > 0 else None
        right = self._frames[position] if position < len(self._frames) else None
        if left is not None and right is not None:
            missing_frames = right - left - 1
            if missing_frames <= self.max_interpolation_gap_frames:
                return left, right
        nearest = self._nearest_frame(frame_index)
        return (nearest, nearest) if nearest is not None else None

    def support_for_frame(self, frame_index: int) -> str:
        """Describe whether a projection is measured or temporally propagated."""

        if frame_index in self._projectors:
            return "direct"
        bounds = self._bounding_frames(frame_index)
        if bounds is None:
            return "unsupported"
        left, right = bounds
        return "interpolated" if left != right else "extrapolated"

    def for_frame(self, frame_index: int) -> PitchProjector | None:
        bounds = self._bounding_frames(frame_index)
        if bounds is None:
            return None
        left, right = bounds
        if left == right:
            return self._projectors[left]
        fraction = (frame_index - left) / (right - left)
        matrix = (
            (1.0 - fraction) * self._projectors[left].matrix
            + fraction * self._projectors[right].matrix
        )
        if abs(matrix[2, 2]) > 1e-9:
            matrix = matrix / matrix[2, 2]
        return PitchProjector(matrix)

    def confidence_for_frame(self, frame_index: int) -> float | None:
        bounds = self._bounding_frames(frame_index)
        if bounds is None:
            return None
        left, right = bounds
        if left == right:
            return self._records[left].confidence
        fraction = (frame_index - left) / (right - left)
        return (
            (1.0 - fraction) * self._records[left].confidence
            + fraction * self._records[right].confidence
        )

    def frame_record(self, frame_index: int) -> CalibrationFrame | None:
        nearest = self._nearest_frame(frame_index)
        return self._records.get(nearest) if nearest is not None else None


def _load_window_rows(
    tactical_h5: Path,
    match_key: str,
    start_frame: int,
    end_frame: int,
) -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install h5py to build FOOTPASS dynamic calibration") from error

    with h5py.File(tactical_h5, "r") as handle:
        if match_key not in handle:
            raise KeyError(f"{match_key} is not present in {tactical_h5}")
        dataset = handle[match_key]
        frame_values = dataset[:, 0]
        first = int(np.searchsorted(frame_values, start_frame, side="left"))
        last = int(np.searchsorted(frame_values, end_frame, side="right"))
        return dataset[first:last]


def _correspondences(
    rows: Any,
    *,
    video_width: int,
    video_height: int,
    settings: DynamicCalibrationConfig,
    min_box_width: float = 2.0,
    min_box_height: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(rows[:, [5, 6, 9, 10, 11, 12]]).all(axis=1)
    visible = rows[finite]
    if len(visible):
        visible = visible[(visible[:, 11] > min_box_width) & (visible[:, 12] > min_box_height)]
    if len(visible) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    image = np.column_stack(
        (
            (visible[:, 9] + visible[:, 11] / 2.0) * video_width / settings.source_width,
            (visible[:, 10] + visible[:, 12]) * video_height / settings.source_height,
        )
    ).astype(np.float32)
    pitch = np.column_stack(
        (visible[:, 5] * PITCH_LENGTH, visible[:, 6] * PITCH_WIDTH)
    ).astype(np.float32)
    return image, pitch


def _bbox_extents(rows: Any) -> tuple[float, float] | None:
    if len(rows) == 0:
        return None
    finite = np.isfinite(rows[:, [9, 10, 11, 12]]).all(axis=1)
    visible = rows[finite]
    if len(visible) == 0:
        return None
    rights = visible[:, 9] + visible[:, 11]
    bottoms = visible[:, 10] + visible[:, 12]
    return float(np.percentile(rights, 98)), float(np.percentile(bottoms, 98))


def candidate_bbox_source_sizes(
    rows: Any,
    *,
    video_width: int,
    video_height: int,
    settings: DynamicCalibrationConfig,
) -> tuple[tuple[int, int], ...]:
    """FOOTPASS bboxes may be Full HD, processing-res, or normalized 0–1."""

    candidates: set[tuple[int, int]] = {
        (settings.source_width, settings.source_height),
        (video_width, video_height),
    }
    extents = _bbox_extents(rows)
    if extents is not None:
        max_x, max_y = extents
        if 0.4 <= max_x <= 2.5 and 0.4 <= max_y <= 2.5:
            candidates.add((1, 1))
        else:
            for width, height in (
                (1920, 1080),
                (1280, 720),
                (1280, 736),
                (960, 540),
                (854, 480),
                (640, 360),
                (640, 352),
            ):
                if abs(max_x - width) / width <= 0.35:
                    candidates.add((width, height))
    return tuple(sorted(candidates))


def _fit_window(
    rows: Any,
    *,
    start_frame: int,
    video_width: int,
    video_height: int,
    settings: DynamicCalibrationConfig,
) -> tuple[dict[int, PitchProjector], dict[int, CalibrationFrame]]:
    projectors: dict[int, PitchProjector] = {}
    records: dict[int, CalibrationFrame] = {}
    for frame_index, frame_rows in _frame_groups(rows):
        if (frame_index - start_frame) % settings.sample_stride:
            continue
        projector, record = _fit_frame(
            frame_index,
            frame_rows,
            video_width=video_width,
            video_height=video_height,
            settings=settings,
        )
        records[frame_index] = record
        if projector is not None:
            projectors[frame_index] = projector
    return projectors, records


def _in_frame_ratio(
    rows: Any,
    *,
    video_width: int,
    video_height: int,
    settings: DynamicCalibrationConfig,
) -> float:
    image, _pitch = _correspondences(
        rows,
        video_width=video_width,
        video_height=video_height,
        settings=settings,
        min_box_width=0.0,
        min_box_height=0.0,
    )
    if len(image) == 0:
        return 0.0
    inside = (
        (image[:, 0] >= -40)
        & (image[:, 0] <= video_width + 40)
        & (image[:, 1] >= -40)
        & (image[:, 1] <= video_height + 40)
    )
    return float(np.mean(inside))


def _source_score(
    records: dict[int, CalibrationFrame],
    *,
    in_frame: float,
) -> tuple[float, int, float]:
    valid = [item for item in records.values() if item.valid]
    error = median(item.median_error_m for item in valid) if valid else float("inf")
    return (in_frame, len(valid), -error)


def select_bbox_source_size(
    rows: Any,
    *,
    start_frame: int,
    video_width: int,
    video_height: int,
    settings: DynamicCalibrationConfig,
) -> DynamicCalibrationConfig:
    candidates = candidate_bbox_source_sizes(
        rows,
        video_width=video_width,
        video_height=video_height,
        settings=settings,
    )
    if len(candidates) <= 1:
        return settings
    span = max(1, int(rows[-1, 0] - rows[0, 0]) + 1) if len(rows) else 1
    probe_stride = max(settings.sample_stride, max(1, span // 25))
    best_settings = settings
    best_score: tuple[float, int, float] | None = None
    for source_width, source_height in candidates:
        probe = replace(
            settings,
            source_width=source_width,
            source_height=source_height,
            sample_stride=probe_stride,
        )
        _projectors, records = _fit_window(
            rows,
            start_frame=start_frame,
            video_width=video_width,
            video_height=video_height,
            settings=probe,
        )
        score = _source_score(
            records,
            in_frame=_in_frame_ratio(
                rows,
                video_width=video_width,
                video_height=video_height,
                settings=probe,
            ),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_settings = replace(
                settings,
                source_width=source_width,
                source_height=source_height,
            )
    return best_settings


def _project_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
    transformed = homogeneous @ matrix.T
    return transformed[:, :2] / transformed[:, 2:3]


def _fit_frame(
    frame_index: int,
    rows: Any,
    *,
    video_width: int,
    video_height: int,
    settings: DynamicCalibrationConfig,
) -> tuple[PitchProjector | None, CalibrationFrame]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install OpenCV to fit dynamic homographies") from error

    image, pitch = _correspondences(
        rows,
        video_width=video_width,
        video_height=video_height,
        settings=settings,
    )
    if len(image) < settings.min_correspondences:
        image, pitch = _correspondences(
            rows,
            video_width=video_width,
            video_height=video_height,
            settings=settings,
            min_box_width=0.0,
            min_box_height=0.0,
        )
    count = len(image)
    invalid = CalibrationFrame(frame_index, count, 0, 0.0, float("inf"), float("inf"), 0.0, False)
    if count < 4:
        return None, invalid

    order = np.lexsort((image[:, 1], image[:, 0]))
    holdout = order[::4]
    train = np.setdiff1d(np.arange(count), holdout)
    if len(holdout) < 1 or len(train) < 4:
        holdout = order[-1:]
        train = order[:-1]
    if len(train) < 4 or len(holdout) < 1:
        return None, invalid

    validation_matrix, inliers = cv2.findHomography(
        image[train],
        pitch[train],
        cv2.RANSAC,
        settings.ransac_threshold_m,
    )
    if validation_matrix is None or not np.isfinite(validation_matrix).all():
        return None, invalid
    predicted = _project_points(validation_matrix, image[holdout])
    errors = np.linalg.norm(predicted - pitch[holdout], axis=1)
    finite_errors = errors[np.isfinite(errors)]
    if len(finite_errors) != len(errors):
        return None, invalid
    median_error = float(np.median(finite_errors))
    p90_error = float(np.percentile(finite_errors, 90))
    inlier_ratio = float(np.mean(inliers)) if inliers is not None and len(inliers) else 0.0
    valid = (
        median_error <= settings.max_median_error_m
        and p90_error <= settings.max_p90_error_m
        and inlier_ratio >= settings.min_inlier_ratio
    )
    confidence = min(
        1.0,
        max(
            0.0,
            inlier_ratio
            * (1.0 - median_error / (settings.max_median_error_m * 2.0))
            * (1.0 - p90_error / (settings.max_p90_error_m * 2.0)),
        ),
    )
    record = CalibrationFrame(
        frame_index=frame_index,
        correspondences=count,
        holdout_points=len(holdout),
        inlier_ratio=round(inlier_ratio, 4),
        median_error_m=round(median_error, 4),
        p90_error_m=round(p90_error, 4),
        confidence=round(confidence, 4),
        valid=valid,
    )
    if not valid:
        return None, record

    final_matrix, _ = cv2.findHomography(
        image,
        pitch,
        cv2.RANSAC,
        settings.ransac_threshold_m,
    )
    if final_matrix is None or not np.isfinite(final_matrix).all():
        return None, record
    return PitchProjector(final_matrix), record


def _frame_groups(rows: Any) -> Iterable[tuple[int, Any]]:
    if len(rows) == 0:
        return
    frame_values = rows[:, 0].astype(int)
    starts = np.flatnonzero(np.r_[True, frame_values[1:] != frame_values[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    for start, end in zip(starts, ends, strict=True):
        yield int(frame_values[start]), rows[start:end]


def build_footpass_dynamic_projector(
    tactical_h5: Path,
    match_key: str,
    *,
    start_frame: int,
    end_frame: int,
    video_width: int,
    video_height: int,
    config: DynamicCalibrationConfig | None = None,
) -> DynamicPitchProjector:
    settings = config or DynamicCalibrationConfig()
    settings.validate()
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("invalid calibration frame window")
    if video_width <= 0 or video_height <= 0:
        raise ValueError("video dimensions must be positive")
    rows = _load_window_rows(tactical_h5, match_key, start_frame, end_frame)
    if len(rows):
        settings = select_bbox_source_size(
            rows,
            start_frame=start_frame,
            video_width=video_width,
            video_height=video_height,
            settings=settings,
        )
    projectors, records = _fit_window(
        rows,
        start_frame=start_frame,
        video_width=video_width,
        video_height=video_height,
        settings=settings,
    )

    valid_records = [records[index] for index in projectors]
    requested_frames = end_frame - start_frame + 1
    provisional = DynamicPitchProjector(
        projectors,
        records,
        DynamicCalibrationReport(
            source=str(tactical_h5.resolve()),
            match_key=match_key,
            start_frame=start_frame,
            end_frame=end_frame,
            sampled_frames=len(records),
            valid_sampled_frames=len(valid_records),
            projected_frame_coverage=0.0,
            direct_frame_coverage=0.0,
            interpolated_frame_coverage=0.0,
            extrapolated_frame_coverage=0.0,
            unsupported_frames=requested_frames,
            ineligible_frames=0,
            eligible_frames=requested_frames,
            median_holdout_error_m=None,
            p90_holdout_error_m=None,
            median_inlier_ratio=None,
            status="fail",
            warnings=(),
            thresholds={},
        ),
        max_interpolation_gap_frames=settings.max_interpolation_gap_frames,
        max_gap_frames=settings.max_projector_gap_frames,
    )
    support_counts = {
        support: 0
        for support in ("direct", "interpolated", "extrapolated", "unsupported", "ineligible")
    }
    for frame_index in range(start_frame, end_frame + 1):
        support = provisional.support_for_frame(frame_index)
        if support == "unsupported":
            record = records.get(frame_index)
            if record is None or record.correspondences == 0:
                support = "ineligible"
        support_counts[support] += 1
    covered = (
        support_counts["direct"]
        + support_counts["interpolated"]
        + support_counts["extrapolated"]
    )
    eligible = requested_frames - support_counts["ineligible"]
    coverage = covered / eligible if eligible else 0.0
    warnings: list[str] = []
    if coverage < settings.min_frame_coverage:
        warnings.append("low_dynamic_calibration_coverage")
    if not valid_records:
        warnings.append("no_valid_dynamic_homography")
    if (settings.source_width, settings.source_height) != (1920, 1080):
        warnings.append("bbox_source_auto_selected")
    if support_counts["extrapolated"]:
        warnings.append("bounded_temporal_extrapolation_used")
    if support_counts["ineligible"]:
        warnings.append("ineligible_replay_or_missing_bbox_excluded")
    median_error = median(item.median_error_m for item in valid_records) if valid_records else None
    p90_error = (
        float(np.percentile([item.p90_error_m for item in valid_records], 90))
        if valid_records
        else None
    )
    median_inliers = median(item.inlier_ratio for item in valid_records) if valid_records else None
    status = (
        "pass"
        if eligible
        and coverage >= settings.min_frame_coverage
        and valid_records
        and median_error is not None
        and median_error <= settings.max_median_error_m
        and p90_error is not None
        and p90_error <= settings.max_p90_error_m
        else "fail"
    )
    report = DynamicCalibrationReport(
        source=str(tactical_h5.resolve()),
        match_key=match_key,
        start_frame=start_frame,
        end_frame=end_frame,
        sampled_frames=len(records),
        valid_sampled_frames=len(valid_records),
        projected_frame_coverage=round(coverage, 4),
        direct_frame_coverage=round(support_counts["direct"] / requested_frames, 4),
        interpolated_frame_coverage=round(
            support_counts["interpolated"] / requested_frames, 4
        ),
        extrapolated_frame_coverage=round(
            support_counts["extrapolated"] / requested_frames, 4
        ),
        unsupported_frames=support_counts["unsupported"],
        ineligible_frames=support_counts["ineligible"],
        eligible_frames=eligible,
        median_holdout_error_m=round(median_error, 4) if median_error is not None else None,
        p90_holdout_error_m=round(p90_error, 4) if p90_error is not None else None,
        median_inlier_ratio=round(median_inliers, 4) if median_inliers is not None else None,
        status=status,
        warnings=tuple(warnings),
        thresholds={
            "max_median_error_m": settings.max_median_error_m,
            "max_p90_error_m": settings.max_p90_error_m,
            "min_inlier_ratio": settings.min_inlier_ratio,
            "min_frame_coverage": settings.min_frame_coverage,
            "max_interpolation_gap_frames": settings.max_interpolation_gap_frames,
            "max_extrapolation_gap_frames": settings.max_projector_gap_frames,
            "min_correspondences": float(settings.min_correspondences),
            "bbox_source_width": float(settings.source_width),
            "bbox_source_height": float(settings.source_height),
        },
    )
    return DynamicPitchProjector(
        projectors,
        records,
        report,
        max_interpolation_gap_frames=settings.max_interpolation_gap_frames,
        max_gap_frames=settings.max_projector_gap_frames,
    )
