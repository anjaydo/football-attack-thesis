"""Per-frame pitch calibration from detected pitch landmarks.

A YOLO-pose pitch model (Roboflow ``football-pitch-detection``) predicts 32
semantic landmarks: pitch corners, penalty and goal box corners, the penalty
spots and the centre circle. Matching them to their metric positions gives an
image-to-pitch homography for every frame where at least four non-collinear
landmarks are visible. Frames without a fit borrow from neighbours in the same
camera shot.

Pitch frame: x along the length (0 = left goal in the main camera), y across
the width with y=0 on the far touchline. FOOTPASS tactical data uses the same
orientation (a team attacking +x has its left back near y=0).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .geometry import PitchProjector
from .models import PITCH_LENGTH, PITCH_WIDTH

CALIBRATION_SCHEMA = "pitch-keypoints-v1"

_PENALTY_BOX_WIDTH = 40.32
_PENALTY_BOX_LENGTH = 16.5
_GOAL_BOX_WIDTH = 18.32
_GOAL_BOX_LENGTH = 5.5
_CENTRE_CIRCLE_RADIUS = 9.15
_PENALTY_SPOT = 11.0


def pitch_keypoints(
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
) -> tuple[tuple[float, float], ...]:
    """Metric positions of the 32 landmarks, in the model's output order."""

    pb_low = (width - _PENALTY_BOX_WIDTH) / 2.0
    pb_high = (width + _PENALTY_BOX_WIDTH) / 2.0
    gb_low = (width - _GOAL_BOX_WIDTH) / 2.0
    gb_high = (width + _GOAL_BOX_WIDTH) / 2.0
    half = length / 2.0
    mid = width / 2.0
    return (
        (0.0, 0.0),
        (0.0, pb_low),
        (0.0, gb_low),
        (0.0, gb_high),
        (0.0, pb_high),
        (0.0, width),
        (_GOAL_BOX_LENGTH, gb_low),
        (_GOAL_BOX_LENGTH, gb_high),
        (_PENALTY_SPOT, mid),
        (_PENALTY_BOX_LENGTH, pb_low),
        (_PENALTY_BOX_LENGTH, gb_low),
        (_PENALTY_BOX_LENGTH, gb_high),
        (_PENALTY_BOX_LENGTH, pb_high),
        (half, 0.0),
        (half, mid - _CENTRE_CIRCLE_RADIUS),
        (half, mid + _CENTRE_CIRCLE_RADIUS),
        (half, width),
        (length - _PENALTY_BOX_LENGTH, pb_low),
        (length - _PENALTY_BOX_LENGTH, gb_low),
        (length - _PENALTY_BOX_LENGTH, gb_high),
        (length - _PENALTY_BOX_LENGTH, pb_high),
        (length - _PENALTY_SPOT, mid),
        (length - _GOAL_BOX_LENGTH, gb_low),
        (length - _GOAL_BOX_LENGTH, gb_high),
        (length, 0.0),
        (length, pb_low),
        (length, gb_low),
        (length, gb_high),
        (length, pb_high),
        (length, width),
        (half - _CENTRE_CIRCLE_RADIUS, mid),
        (half + _CENTRE_CIRCLE_RADIUS, mid),
    )


PITCH_KEYPOINTS = pitch_keypoints()


@dataclass(frozen=True)
class CalibrationFitConfig:
    """Fit acceptance rules.

    Four landmarks always fit a homography exactly, so a single mislabelled
    landmark goes unnoticed; ``min_inliers`` of 5 keeps the fit over-determined.
    """

    min_keypoint_confidence: float = 0.5
    ransac_threshold_m: float = 1.0
    max_rmse_m: float = 1.0
    max_image_error_ratio: float = 0.01
    min_inliers: int = 5
    min_spread_m: float = 2.0
    pitch_margin_m: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_keypoint_confidence <= 1.0:
            raise ValueError("min_keypoint_confidence must be between 0 and 1")
        if self.min_inliers < 4:
            raise ValueError("a homography needs at least four inliers")
        if (
            self.ransac_threshold_m <= 0
            or self.max_rmse_m <= 0
            or self.min_spread_m <= 0
            or self.max_image_error_ratio <= 0
        ):
            raise ValueError("thresholds must be positive")


@dataclass(frozen=True)
class FrameFit:
    """Homography fitted on one frame's landmarks (image pixels -> pitch metres)."""

    matrix: tuple[float, ...]
    inliers: int
    rmse_m: float
    keypoint_ids: tuple[int, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float).reshape(3, 3)


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def fit_frame_homography(
    image_xy: np.ndarray,
    confidence: np.ndarray,
    *,
    frame_size: tuple[float, float],
    config: CalibrationFitConfig | None = None,
) -> FrameFit | None:
    """Fit image->pitch on confident landmarks; ``None`` if the fit is unsafe."""

    import cv2

    settings = config or CalibrationFitConfig()
    image_xy = np.asarray(image_xy, dtype=float).reshape(-1, 2)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    if len(image_xy) != len(PITCH_KEYPOINTS) or len(confidence) != len(PITCH_KEYPOINTS):
        raise ValueError("expected one image point and confidence per pitch landmark")
    visible = np.where(confidence >= settings.min_keypoint_confidence)[0]
    if len(visible) < settings.min_inliers:
        return None
    source = image_xy[visible]
    target = np.asarray(PITCH_KEYPOINTS, dtype=float)[visible]
    # The calibration digest keys the vision cache, so RANSAC must not vary between runs.
    cv2.setRNGSeed(0)
    matrix, mask = cv2.findHomography(
        source, target, cv2.RANSAC, settings.ransac_threshold_m
    )
    if matrix is None or mask is None:
        return None
    inlier_index = np.where(mask.reshape(-1) > 0)[0]
    if len(inlier_index) < settings.min_inliers:
        return None
    inlier_target = target[inlier_index]
    centred = inlier_target - inlier_target.mean(axis=0)
    spread = np.linalg.svd(centred, compute_uv=False)
    if spread[-1] / np.sqrt(len(inlier_index)) < settings.min_spread_m:
        return None
    projected = _project(matrix, source[inlier_index])
    rmse = float(np.sqrt(np.mean(np.sum((projected - inlier_target) ** 2, axis=1))))
    if rmse > settings.max_rmse_m:
        return None
    width, height = frame_size
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        back = _project(inverse, inlier_target)
    image_error = np.sqrt(np.sum((back - source[inlier_index]) ** 2, axis=1))
    if not np.all(np.isfinite(image_error)) or float(np.sqrt(np.mean(image_error**2))) > (
        settings.max_image_error_ratio * float(np.hypot(width, height))
    ):
        return None
    probes = np.asarray(
        [[width * 0.5, height * 0.95], [width * 0.5, height * 0.5], [width * 0.1, height * 0.8], [width * 0.9, height * 0.8]]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        on_pitch = _project(matrix, probes)
    margin = settings.pitch_margin_m
    if not np.all(np.isfinite(on_pitch)) or np.any(
        (on_pitch[:, 0] < -margin)
        | (on_pitch[:, 0] > PITCH_LENGTH + margin)
        | (on_pitch[:, 1] < -margin)
        | (on_pitch[:, 1] > PITCH_WIDTH + margin)
    ):
        return None
    return FrameFit(
        matrix=tuple(float(value) for value in (matrix / matrix[2, 2]).reshape(-1)),
        inliers=int(len(inlier_index)),
        rmse_m=round(rmse, 4),
        keypoint_ids=tuple(int(visible[index]) + 1 for index in inlier_index),
    )


@dataclass(frozen=True)
class FrameCalibration:
    matrix: tuple[float, ...]
    confidence: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "matrix": [round(value, 10) for value in self.matrix],
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


def _anchor_grid(frame_size: tuple[float, float]) -> np.ndarray:
    width, height = frame_size
    return np.asarray(
        [
            (width * fx, height * fy)
            for fy in (0.4, 0.65, 0.95)
            for fx in (0.1, 0.5, 0.9)
        ],
        dtype=float,
    )


def _shot_ids(frame_count: int, cut_frames: set[int]) -> list[int]:
    shot = 0
    ids = []
    for index in range(frame_count):
        if index in cut_frames and index > 0:
            shot += 1
        ids.append(shot)
    return ids


def _fit_confidence(fit: FrameFit) -> float:
    return min(1.0, fit.inliers / 8.0) * max(0.0, 1.0 - fit.rmse_m / 3.0)


def smooth_calibrations(
    fits: Sequence[FrameFit | None],
    *,
    frame_size: tuple[float, float],
    cut_frames: set[int] | None = None,
    window: int = 3,
    max_gap: int = 25,
) -> list[FrameCalibration | None]:
    """Median-smooth fits inside each shot and bridge short gaps.

    Each fit maps a fixed image grid onto the pitch. The per-point median over
    neighbouring frames is robust to one bad fit; a homography is refit on
    the smoothed grid. Gaps up to ``max_gap`` frames are filled by blending
    the nearest fits on each side, with confidence decaying with distance.
    """

    import cv2

    anchors = _anchor_grid(frame_size)
    shots = _shot_ids(len(fits), cut_frames or set())
    mapped: list[np.ndarray | None] = []
    for fit in fits:
        if fit is None:
            mapped.append(None)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            points = _project(fit.as_array(), anchors)
        mapped.append(points if np.all(np.isfinite(points)) else None)
    valid = [index for index, points in enumerate(mapped) if points is not None]
    results: list[FrameCalibration | None] = []
    for index in range(len(fits)):
        shot = shots[index]
        near = [
            other
            for other in valid
            if abs(other - index) <= window and shots[other] == shot
        ]
        source = "keypoints"
        if near:
            stacked = np.stack([mapped[other] for other in near])
            target = np.median(stacked, axis=0)
            confidence = float(np.median([_fit_confidence(fits[other]) for other in near]))
            if mapped[index] is None:
                source = "keypoints_neighbour"
                confidence *= 0.8
        else:
            before = [other for other in valid if other < index and shots[other] == shot and index - other <= max_gap]
            after = [other for other in valid if other > index and shots[other] == shot and other - index <= max_gap]
            if not before and not after:
                results.append(None)
                continue
            source = "keypoints_bridged"
            if before and after:
                left, right = before[-1], after[0]
                weight = (index - left) / (right - left)
                target = (1.0 - weight) * mapped[left] + weight * mapped[right]
                gap = min(index - left, right - index)
                base = min(_fit_confidence(fits[left]), _fit_confidence(fits[right]))
            else:
                nearest = before[-1] if before else after[0]
                target = mapped[nearest]
                gap = abs(index - nearest)
                base = _fit_confidence(fits[nearest])
            confidence = base * max(0.2, 1.0 - gap / max_gap) * 0.8
        matrix, _mask = cv2.findHomography(anchors, target, 0)
        if matrix is None:
            results.append(None)
            continue
        matrix = matrix / matrix[2, 2]
        results.append(
            FrameCalibration(
                matrix=tuple(float(value) for value in matrix.reshape(-1)),
                confidence=round(min(1.0, max(0.0, confidence)), 4),
                source=source,
            )
        )
    return results


@dataclass
class PitchCalibrationTrack:
    """Per-frame image->pitch homographies for one video."""

    frame_size: tuple[float, float]
    start_frame: int
    frames: list[FrameCalibration | None]
    summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CALIBRATION_SCHEMA,
            "frame_size": list(self.frame_size),
            "start_frame": self.start_frame,
            "summary": self.summary,
            "frames": [None if item is None else item.to_dict() for item in self.frames],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PitchCalibrationTrack":
        if payload.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("unsupported pitch calibration schema")
        frames = [
            None
            if item is None
            else FrameCalibration(
                matrix=tuple(float(value) for value in item["matrix"]),
                confidence=float(item["confidence"]),
                source=str(item["source"]),
            )
            for item in payload.get("frames") or ()
        ]
        width, height = payload["frame_size"]
        return cls(
            frame_size=(float(width), float(height)),
            start_frame=int(payload.get("start_frame") or 0),
            frames=frames,
            summary=dict(payload.get("summary") or {}),
        )

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def record(self, frame_index: int) -> FrameCalibration | None:
        local = frame_index - self.start_frame
        if 0 <= local < len(self.frames):
            return self.frames[local]
        return None


class KeypointPitchProjector:
    """Frame-indexed projector over a :class:`PitchCalibrationTrack`.

    Video-only evidence: it declares no external calibration source, so the
    pipeline keeps the ``video_only`` tier.
    """

    overlay_label = "VIDEO-ONLY pitch (landmarks)"
    draws_pitch_model = True

    def __init__(self, track: PitchCalibrationTrack) -> None:
        self.track = track
        self.summary = dict(track.summary)
        self._cache: dict[int, PitchProjector | None] = {}

    def for_frame(self, frame_index: int) -> PitchProjector | None:
        if frame_index not in self._cache:
            record = self.track.record(frame_index)
            self._cache[frame_index] = (
                None
                if record is None
                else PitchProjector(np.asarray(record.matrix, dtype=float).reshape(3, 3))
            )
        return self._cache[frame_index]

    def confidence_for_frame(self, frame_index: int) -> float | None:
        record = self.track.record(frame_index)
        return None if record is None else record.confidence


def pitch_line_segments(
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
) -> list[list[tuple[float, float]]]:
    """Polylines of the pitch markings in metres."""

    from math import cos, pi, sin

    pb_low = (width - _PENALTY_BOX_WIDTH) / 2.0
    pb_high = (width + _PENALTY_BOX_WIDTH) / 2.0
    gb_low = (width - _GOAL_BOX_WIDTH) / 2.0
    gb_high = (width + _GOAL_BOX_WIDTH) / 2.0
    half = length / 2.0
    lines: list[list[tuple[float, float]]] = [
        [(0.0, 0.0), (length, 0.0), (length, width), (0.0, width), (0.0, 0.0)],
        [(half, 0.0), (half, width)],
        [(0.0, pb_low), (_PENALTY_BOX_LENGTH, pb_low), (_PENALTY_BOX_LENGTH, pb_high), (0.0, pb_high)],
        [(0.0, gb_low), (_GOAL_BOX_LENGTH, gb_low), (_GOAL_BOX_LENGTH, gb_high), (0.0, gb_high)],
        [(length, pb_low), (length - _PENALTY_BOX_LENGTH, pb_low), (length - _PENALTY_BOX_LENGTH, pb_high), (length, pb_high)],
        [(length, gb_low), (length - _GOAL_BOX_LENGTH, gb_low), (length - _GOAL_BOX_LENGTH, gb_high), (length, gb_high)],
    ]
    circle = [
        (half + _CENTRE_CIRCLE_RADIUS * cos(2 * pi * step / 48), width / 2 + _CENTRE_CIRCLE_RADIUS * sin(2 * pi * step / 48))
        for step in range(49)
    ]
    lines.append(circle)
    return lines


def draw_pitch_model(cv2: Any, frame: Any, projector: PitchProjector, color: tuple[int, int, int] = (255, 0, 255)) -> None:
    """Draw the calibrated pitch markings back onto the frame."""

    try:
        inverse = np.linalg.inv(projector.matrix)
    except np.linalg.LinAlgError:
        return
    height, width = frame.shape[:2]
    for polyline in pitch_line_segments():
        dense: list[tuple[float, float]] = []
        for start, end in zip(polyline, polyline[1:]):
            for step in range(12):
                t = step / 12.0
                dense.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
        dense.append(polyline[-1])
        points = np.hstack([np.asarray(dense), np.ones((len(dense), 1))]) @ inverse.T
        previous: tuple[int, int] | None = None
        for x, y, w in points:
            if w <= 1e-6:
                previous = None
                continue
            current = (int(round(x / w)), int(round(y / w)))
            if abs(current[0]) > 4 * width or abs(current[1]) > 4 * height:
                previous = None
                continue
            if previous is not None:
                cv2.line(frame, previous, current, color, 2, cv2.LINE_AA)
            previous = current


@dataclass(frozen=True)
class PitchModelConfig:
    model_path: str
    device: str = "cpu"
    image_sizes: tuple[int, ...] = (640, 1280)
    scene_cut_threshold: float = 0.35
    window: int = 3
    max_gap: int = 25
    fit: CalibrationFitConfig = CalibrationFitConfig()


@dataclass(frozen=True)
class KeypointDetections:
    """Raw landmark output: one 32-point set per frame and inference size."""

    frame_size: tuple[float, float]
    start_frame: int
    image_sizes: tuple[int, ...]
    xy: np.ndarray
    confidence: np.ndarray
    frame_difference: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.stem + ".partial.npz")
        with partial.open("wb") as handle:
            np.savez_compressed(
                handle,
                frame_size=np.asarray(self.frame_size, dtype=float),
                start_frame=np.asarray(self.start_frame),
                image_sizes=np.asarray(self.image_sizes, dtype=int),
                xy=self.xy.astype(np.float32),
                confidence=self.confidence.astype(np.float16),
                frame_difference=self.frame_difference.astype(np.float32),
            )
        partial.replace(path)

    @classmethod
    def load(cls, path: Path) -> "KeypointDetections":
        with np.load(path) as payload:
            width, height = payload["frame_size"].tolist()
            return cls(
                frame_size=(float(width), float(height)),
                start_frame=int(payload["start_frame"]),
                image_sizes=tuple(int(value) for value in payload["image_sizes"]),
                xy=payload["xy"].astype(float),
                confidence=payload["confidence"].astype(float),
                frame_difference=payload["frame_difference"].astype(float),
            )


def detect_pitch_keypoints(
    video_path: Path,
    config: PitchModelConfig,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> KeypointDetections:
    import cv2
    from ultralytics import YOLO

    model = YOLO(config.model_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    limit = max(0, total - start_frame) if max_frames is None else max_frames
    sizes = tuple(config.image_sizes)
    points: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    differences: list[float] = []
    previous: np.ndarray | None = None
    try:
        while len(points) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            sample = frame[::16, ::16].astype(np.float32)
            differences.append(
                0.0 if previous is None else float(np.mean(np.abs(sample - previous)) / 255.0)
            )
            previous = sample
            frame_xy = np.zeros((len(sizes), len(PITCH_KEYPOINTS), 2), dtype=np.float32)
            frame_conf = np.zeros((len(sizes), len(PITCH_KEYPOINTS)), dtype=np.float32)
            for slot, size in enumerate(sizes):
                result = model.predict(frame, imgsz=size, conf=0.25, device=config.device, verbose=False)[0]
                keypoints = result.keypoints
                if keypoints is None or keypoints.conf is None or len(keypoints) == 0:
                    continue
                best = (
                    int(np.argmax(result.boxes.conf.cpu().numpy()))
                    if result.boxes is not None and len(result.boxes)
                    else 0
                )
                frame_xy[slot] = keypoints.xy[best].cpu().numpy()
                frame_conf[slot] = keypoints.conf[best].cpu().numpy()
            points.append(frame_xy)
            scores.append(frame_conf)
            if progress is not None and len(points) % 50 == 0:
                progress(len(points), limit)
    finally:
        capture.release()
    empty_xy = np.zeros((0, len(sizes), len(PITCH_KEYPOINTS), 2))
    empty_conf = np.zeros((0, len(sizes), len(PITCH_KEYPOINTS)))
    return KeypointDetections(
        frame_size=(width, height),
        start_frame=start_frame,
        image_sizes=sizes,
        xy=np.stack(points) if points else empty_xy,
        confidence=np.stack(scores) if scores else empty_conf,
        frame_difference=np.asarray(differences, dtype=float),
    )


def calibration_from_detections(
    detections: KeypointDetections,
    config: PitchModelConfig,
) -> PitchCalibrationTrack:
    fits: list[FrameFit | None] = []
    for frame_xy, frame_conf in zip(detections.xy, detections.confidence):
        best: FrameFit | None = None
        for slot in range(len(detections.image_sizes)):
            fit = fit_frame_homography(
                frame_xy[slot],
                frame_conf[slot],
                frame_size=detections.frame_size,
                config=config.fit,
            )
            if fit is not None and (
                best is None or (fit.inliers, -fit.rmse_m) > (best.inliers, -best.rmse_m)
            ):
                best = fit
        fits.append(best)
    cuts = {
        index
        for index, value in enumerate(detections.frame_difference)
        if index > 0 and value >= config.scene_cut_threshold
    }
    frames = smooth_calibrations(
        fits,
        frame_size=detections.frame_size,
        cut_frames=cuts,
        window=config.window,
        max_gap=config.max_gap,
    )
    direct = [fit for fit in fits if fit is not None]
    covered = sum(item is not None for item in frames)
    summary = {
        "model": Path(config.model_path).name,
        "frames": len(fits),
        "direct_fits": len(direct),
        "calibrated_frames": covered,
        "coverage": round(covered / len(fits), 4) if fits else 0.0,
        "median_inliers": float(np.median([fit.inliers for fit in direct])) if direct else None,
        "median_rmse_m": float(np.median([fit.rmse_m for fit in direct])) if direct else None,
        "scene_cuts": len(cuts),
    }
    return PitchCalibrationTrack(
        frame_size=detections.frame_size,
        start_frame=detections.start_frame,
        frames=frames,
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_calibrate_video(
    video_path: Path,
    cache_dir: Path,
    config: PitchModelConfig,
    *,
    video_digest: str,
    start_frame: int = 0,
    max_frames: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PitchCalibrationTrack:
    """Detect landmarks once per video and model; refit whenever settings change."""

    key = hashlib.sha256(
        json.dumps(
            {
                "schema": CALIBRATION_SCHEMA,
                "video": video_digest,
                "model": _file_sha256(Path(config.model_path)),
                "image_sizes": list(config.image_sizes),
                "start_frame": start_frame,
                "max_frames": max_frames,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    path = cache_dir / f"pitch-keypoints-{key}.npz"
    if path.is_file():
        detections = KeypointDetections.load(path)
    else:
        detections = detect_pitch_keypoints(
            video_path,
            config,
            start_frame=start_frame,
            max_frames=max_frames,
            progress=progress,
        )
        detections.save(path)
    return calibration_from_detections(detections, config)


__all__ = [
    "CALIBRATION_SCHEMA",
    "CalibrationFitConfig",
    "FrameCalibration",
    "FrameFit",
    "KeypointDetections",
    "KeypointPitchProjector",
    "PITCH_KEYPOINTS",
    "PitchCalibrationTrack",
    "PitchModelConfig",
    "calibration_from_detections",
    "detect_pitch_keypoints",
    "draw_pitch_model",
    "fit_frame_homography",
    "load_or_calibrate_video",
    "pitch_keypoints",
    "pitch_line_segments",
    "smooth_calibrations",
]
