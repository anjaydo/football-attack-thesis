from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .geometry import (
    VIDEO_ONLY_CALIBRATION_CONFIDENCE,
    Point,
    Projector,
    projector_confidence,
    projector_for_frame,
    projector_from_pitch_polygon,
)
from .field_geometry import FieldGeometryDetector, FieldLineConfig, draw_field_geometry
from .models import PITCH_LENGTH, PITCH_WIDTH
from .motion import SpeedEstimator
from .roi import PolygonROI
from .scene import SceneCutDetector
from .team import OnlineTeamClassifier, extract_jersey_feature


COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32
# 6GB laptop GPUs already hold player, pose and ball weights. Keep extra batches small.
BALL_TILE_BATCH = 2
POSE_CROP_BATCH = 4
COCO_SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass(frozen=True)
class VisionConfig:
    pose_model: str = "yolo26n-pose.pt"
    detection_model: str = "yolo26n.pt"
    ball_detection_model: str | None = None
    ball_class_id: int = COCO_SPORTS_BALL_CLASS
    player_tracker: str = "botsort.yaml"
    ball_tracker: str = "bytetrack.yaml"
    device: str = "cpu"
    image_size: int = 960
    player_confidence: float = 0.2
    ball_confidence: float = 0.15
    keypoint_confidence: float = 0.25
    speed_smoothing: float = 0.35
    team_classification: bool = True
    team_min_tracks: int = 4
    scene_cut_threshold: float | None = 0.35
    start_frame: int = 0
    max_frames: int | None = None
    field_lines: bool = False
    field_line_min_length_ratio: float = 0.06
    field_line_max_segments: int = 40
    filter_ball_by_roi: bool = False
    pose_on_crop: bool = True
    pose_crop_pad: float = 0.35
    pose_crop_imgsz: int = 320
    ball_tile_size: int = 0
    ball_tile_overlap: float = 0.1
    ball_max_side_px: float = 80.0

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.ball_class_id < 0:
            raise ValueError("ball_class_id must be non-negative")
        for name, value in (
            ("player_confidence", self.player_confidence),
            ("ball_confidence", self.ball_confidence),
            ("keypoint_confidence", self.keypoint_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.start_frame < 0:
            raise ValueError("start_frame cannot be negative")
        if self.team_min_tracks < 2:
            raise ValueError("team_min_tracks must be at least 2")
        if self.scene_cut_threshold is not None and not 0.0 < self.scene_cut_threshold <= 1.0:
            raise ValueError("scene_cut_threshold must be in (0, 1]")
        FieldLineConfig(
            min_length_ratio=self.field_line_min_length_ratio,
            max_segments=self.field_line_max_segments,
        ).validate()
        if self.pose_crop_pad < 0:
            raise ValueError("pose_crop_pad must be non-negative")
        if self.pose_crop_imgsz <= 0:
            raise ValueError("pose_crop_imgsz must be positive")
        if self.ball_tile_size < 0:
            raise ValueError("ball_tile_size must be non-negative")
        if not 0.0 <= self.ball_tile_overlap < 0.75:
            raise ValueError("ball_tile_overlap must be in [0, 0.75)")
        if self.ball_max_side_px < 0:
            raise ValueError("ball_max_side_px must be non-negative")


@dataclass(frozen=True)
class VisionRunSummary:
    input_path: str
    output_path: str
    observations_path: str | None
    frames_processed: int
    fps: float
    width: int
    height: int
    unique_player_tracks: int
    unique_ball_tracks: int
    speed_unit: str
    pose_model: str
    detection_model: str
    roi_enabled: bool
    team_classification_enabled: bool
    scene_cuts_detected: int
    ball_detection_model: str | None = None
    field_geometry_enabled: bool = False
    field_geometry_frames: int = 0
    start_frame: int = 0
    end_frame: int | None = None
    calibrated_frames: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Vision dependencies are missing. Install with: "
            "python -m pip install -e \".[vision]\""
        ) from error
    return cv2, YOLO


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.numpy() if hasattr(value, "numpy") else value


def _track_ids(boxes: Any) -> list[int | None]:
    if boxes is None or boxes.id is None:
        count = 0 if boxes is None else len(boxes)
        return [None] * count
    return [int(value) for value in _to_numpy(boxes.id).reshape(-1)]


def _box_iou(first: Any, second: Any) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_keypoints(
    detection_box: Any,
    pose_boxes: Any,
    pose_keypoints: Any,
    used_pose_indexes: set[int],
    minimum_iou: float = 0.05,
) -> Any | None:
    if pose_keypoints is None:
        return None
    best_index = None
    best_iou = minimum_iou
    for index, pose_box in enumerate(pose_boxes):
        if index in used_pose_indexes:
            continue
        iou = _box_iou(detection_box, pose_box)
        if iou > best_iou:
            best_iou = iou
            best_index = index
    if best_index is None:
        return None
    used_pose_indexes.add(best_index)
    return pose_keypoints[best_index]


MIN_POSE_CROP_SIDE = 24
MIN_CROP_VISIBLE_KEYPOINTS = 5
BALL_TILE_TRACK_ID_START = 10_000


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    if tile >= length:
        return [0]
    starts = []
    position = 0
    while position + tile < length:
        starts.append(position)
        position += stride
    starts.append(length - tile)
    unique: list[int] = []
    for start in starts:
        if start not in unique:
            unique.append(start)
    return unique


def _tile_windows(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
    bounds: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bounds or (0, 0, width, height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    region_w = x2 - x1
    region_h = y2 - y1
    if tile_size >= region_w and tile_size >= region_h:
        if region_w < width or region_h < height:
            return [(x1, y1, x2, y2)]
        return []
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    tile_w = min(tile_size, region_w)
    tile_h = min(tile_size, region_h)
    windows: list[tuple[int, int, int, int]] = []
    for top in _axis_starts(region_h, tile_h, stride):
        for left in _axis_starts(region_w, tile_w, stride):
            windows.append(
                (
                    x1 + left,
                    y1 + top,
                    min(width, x1 + left + tile_w),
                    min(height, y1 + top + tile_h),
                )
            )
    return windows


def iter_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
    bounds: tuple[int, int, int, int] | None = None,
    max_windows: int = 12,
) -> list[tuple[int, int, int, int]]:
    """Overlapping windows. Empty when one tile already covers the whole frame.

    Overlap is reduced until the window count is at most ``max_windows`` so a
    1080p frame stays near a dozen ball forwards instead of a dense grid.
    """
    if tile_size <= 0 or width <= 0 or height <= 0:
        return []
    if tile_size >= width and tile_size >= height and bounds is None:
        return []
    current = overlap
    windows = _tile_windows(width, height, tile_size, current, bounds)
    while len(windows) > max_windows and current > 0:
        current = max(0.0, round(current - 0.05, 2))
        windows = _tile_windows(width, height, tile_size, current, bounds)
        if current == 0:
            break
    return windows


def _pitch_tile_bounds(
    field_geometry: Any | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    polygon = getattr(field_geometry, "pitch_polygon", None) if field_geometry else None
    if not polygon or len(polygon) < 3:
        return None
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    pad_x = 0.08 * width
    pad_y = 0.08 * height
    return (
        int(max(0, min(xs) - pad_x)),
        int(max(0, min(ys) - pad_y)),
        int(min(width, max(xs) + pad_x)),
        int(min(height, max(ys) + pad_y)),
    )


def _ball_shape_ok(box: tuple[float, float, float, float], max_side: float) -> bool:
    width = float(box[2]) - float(box[0])
    height = float(box[3]) - float(box[1])
    short = min(width, height)
    long = max(width, height)
    if short < 2.0 or long <= 0:
        return False
    if max_side > 0 and long > max_side:
        return False
    return long / short <= 3.0


def _center_inside(
    box: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
    margin: float = 4.0,
) -> bool:
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    return (
        other[0] - margin <= center_x <= other[2] + margin
        and other[1] - margin <= center_y <= other[3] + margin
    )


def _nms_indexes(
    boxes: list[tuple[float, float, float, float]],
    confidences: list[float],
    iou_threshold: float,
) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda index: confidences[index], reverse=True)
    kept: list[int] = []
    suppressed: set[int] = set()
    for index in order:
        if index in suppressed:
            continue
        kept.append(index)
        for other in order:
            if other == index or other in suppressed:
                continue
            if _box_iou(boxes[index], boxes[other]) >= iou_threshold:
                suppressed.add(other)
    return kept


def merge_ball_detections(
    frame_rows: list[tuple[tuple[float, float, float, float], float, int | None]],
    tile_rows: list[tuple[tuple[float, float, float, float], float, int | None]],
    max_side: float,
    duplicate_iou: float = 0.2,
) -> list[tuple[tuple[float, float, float, float], float, int | None]]:
    """Keep ByteTrack rows, then tile-only boxes that are not the same ball."""
    kept = [row for row in frame_rows if _ball_shape_ok(row[0], max_side)]
    candidates = [row for row in tile_rows if _ball_shape_ok(row[0], max_side)]
    extra_boxes: list[tuple[float, float, float, float]] = []
    extra_confidences: list[float] = []
    for box, confidence, _track_id in candidates:
        duplicate = False
        for kept_box, _kept_confidence, _kept_id in kept:
            if _box_iou(box, kept_box) >= duplicate_iou or _center_inside(box, kept_box):
                duplicate = True
                break
        if not duplicate:
            extra_boxes.append(box)
            extra_confidences.append(confidence)
    for index in _nms_indexes(extra_boxes, extra_confidences, duplicate_iou):
        kept.append((extra_boxes[index], extra_confidences[index], None))
    return kept


class TileBallAssociator:
    """Assign stable ids to tile-only balls. Never invents a box without a detection."""

    def __init__(self) -> None:
        self._next_id = BALL_TILE_TRACK_ID_START
        self._centers: dict[int, tuple[float, float]] = {}

    def reset(self) -> None:
        self._centers.clear()

    def assign(
        self, boxes: list[tuple[float, float, float, float]]
    ) -> list[int]:
        unused = set(self._centers)
        pairs: list[tuple[float, int, int]] = []
        for index, box in enumerate(boxes):
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            gate = max(40.0, 3.0 * max(box[2] - box[0], box[3] - box[1], 8.0))
            for track_id in unused:
                previous_x, previous_y = self._centers[track_id]
                distance = ((center_x - previous_x) ** 2 + (center_y - previous_y) ** 2) ** 0.5
                if distance <= gate:
                    pairs.append((distance, index, track_id))
        pairs.sort()
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        assigned = [0] * len(boxes)
        for _distance, index, track_id in pairs:
            if index in used_detections or track_id in used_tracks:
                continue
            assigned[index] = track_id
            used_detections.add(index)
            used_tracks.add(track_id)
        centers: dict[int, tuple[float, float]] = {}
        for index, box in enumerate(boxes):
            if index not in used_detections:
                assigned[index] = self._next_id
                self._next_id += 1
            centers[assigned[index]] = (
                (box[0] + box[2]) / 2.0,
                (box[1] + box[3]) / 2.0,
            )
        self._centers = centers
        return assigned


def _detection_rows(boxes: Any) -> list[tuple[tuple[float, float, float, float], float, int | None]]:
    if boxes is None:
        return []
    xyxy = _to_numpy(boxes.xyxy)
    if xyxy is None or len(xyxy) == 0:
        return []
    confidences = _to_numpy(boxes.conf)
    confidence_values = (
        confidences.reshape(-1) if confidences is not None else [1.0] * len(xyxy)
    )
    track_ids = _track_ids(boxes)
    rows = []
    for index, box in enumerate(xyxy):
        rows.append(
            (
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                float(confidence_values[index]),
                track_ids[index] if index < len(track_ids) else None,
            )
        )
    return rows


def _shift_keypoints(keypoints: Any, origin_x: float, origin_y: float) -> list[list[float]]:
    shifted = []
    for point in keypoints:
        values = [float(point[0]) + origin_x, float(point[1]) + origin_y]
        if len(point) > 2:
            values.append(float(point[2]))
        shifted.append(values)
    return shifted


def _visible_keypoint_count(keypoints: Any, threshold: float) -> int:
    count = 0
    for point in keypoints:
        confidence = float(point[2]) if len(point) > 2 else 1.0
        if confidence >= threshold:
            count += 1
    return count


def _padded_crop(
    frame: Any,
    box: Any,
    pad_ratio: float,
) -> tuple[Any, int, int] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    box_w = x2 - x1
    box_h = y2 - y1
    if min(box_w, box_h) < MIN_POSE_CROP_SIDE:
        return None
    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio
    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(width, int(x2 + pad_x + 0.999))
    crop_y2 = min(height, int(y2 + pad_y + 0.999))
    if crop_x2 - crop_x1 < MIN_POSE_CROP_SIDE or crop_y2 - crop_y1 < MIN_POSE_CROP_SIDE:
        return None
    return frame[crop_y1:crop_y2, crop_x1:crop_x2], crop_x1, crop_y1


def _select_crop_keypoints(
    result: Any,
    local_box: tuple[float, float, float, float],
    threshold: float,
) -> list[list[float]] | None:
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or keypoints is None:
        return None
    pose_boxes = _to_numpy(boxes.xyxy)
    keypoint_data = _to_numpy(keypoints.data)
    if pose_boxes is None or keypoint_data is None or len(pose_boxes) == 0:
        return None
    best_index = None
    best_iou = 0.05
    for index, pose_box in enumerate(pose_boxes):
        iou = _box_iou(local_box, pose_box)
        if iou > best_iou:
            best_iou = iou
            best_index = index
    if best_index is None:
        return None
    shifted = keypoint_data[best_index]
    if _visible_keypoint_count(shifted, threshold) < MIN_CROP_VISIBLE_KEYPOINTS:
        return None
    return [list(map(float, point)) for point in shifted]


def _pose_on_player_crops(
    pose_model: Any,
    frame: Any,
    boxes: list[Any],
    *,
    confidence: float,
    image_size: int,
    device: str,
    keypoint_confidence: float,
    pad_ratio: float,
) -> list[list[list[float]] | None]:
    crops: list[Any] = []
    origins: list[tuple[int, int]] = []
    local_boxes: list[tuple[float, float, float, float]] = []
    for box in boxes:
        cropped = _padded_crop(frame, box, pad_ratio)
        if cropped is None:
            crops.append(None)
            origins.append((0, 0))
            local_boxes.append((0.0, 0.0, 0.0, 0.0))
            continue
        image, origin_x, origin_y = cropped
        crops.append(image)
        origins.append((origin_x, origin_y))
        local_boxes.append(
            (
                float(box[0]) - origin_x,
                float(box[1]) - origin_y,
                float(box[2]) - origin_x,
                float(box[3]) - origin_y,
            )
        )
    valid = [image for image in crops if image is not None]
    if not valid:
        return [None] * len(boxes)
    results = list(
        pose_model.predict(
            valid,
            classes=[COCO_PERSON_CLASS],
            conf=confidence,
            imgsz=image_size,
            device=device,
            batch=POSE_CROP_BATCH,
            verbose=False,
        )
    )
    result_cursor = 0
    refined: list[list[list[float]] | None] = []
    for image, (origin_x, origin_y), local_box in zip(crops, origins, local_boxes):
        if image is None or result_cursor >= len(results):
            refined.append(None)
            continue
        result = results[result_cursor]
        result_cursor += 1
        selected = _select_crop_keypoints(result, local_box, keypoint_confidence)
        if selected is None:
            refined.append(None)
            continue
        refined.append(_shift_keypoints(selected, origin_x, origin_y))
    return refined


def _tiled_ball_detections(
    ball_model: Any,
    frame: Any,
    *,
    class_id: int,
    confidence: float,
    image_size: int,
    device: str,
    tile_size: int,
    overlap: float,
    bounds: tuple[int, int, int, int] | None,
) -> list[tuple[tuple[float, float, float, float], float, int | None]]:
    height, width = frame.shape[:2]
    windows = iter_tiles(width, height, tile_size, overlap, bounds)
    if not windows:
        return []
    crops = [frame[top:bottom, left:right] for left, top, right, bottom in windows]
    results = list(
        ball_model.predict(
            crops,
            classes=[class_id],
            conf=confidence,
            imgsz=image_size,
            device=device,
            batch=BALL_TILE_BATCH,
            verbose=False,
        )
    )
    rows: list[tuple[tuple[float, float, float, float], float, int | None]] = []
    for (left, top, _right, _bottom), result in zip(windows, results):
        for box, box_confidence, _track_id in _detection_rows(result.boxes):
            rows.append(
                (
                    (box[0] + left, box[1] + top, box[2] + left, box[3] + top),
                    box_confidence,
                    None,
                )
            )
    return rows


def _draw_label(cv2: Any, frame: Any, point: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = point
    y = max(height + baseline + 3, y)
    cv2.rectangle(frame, (x, y - height - baseline - 3), (x + width + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline - 1), font, scale, (15, 15, 15), thickness, cv2.LINE_AA)


def _player_anchor(box: Any, keypoints: Any | None, threshold: float) -> Point:
    if keypoints is not None and len(keypoints) >= 17:
        visible_ankles = []
        for index in (15, 16):
            confidence = float(keypoints[index][2]) if len(keypoints[index]) > 2 else 1.0
            if confidence >= threshold:
                visible_ankles.append((float(keypoints[index][0]), float(keypoints[index][1])))
        if visible_ankles:
            return (
                sum(point[0] for point in visible_ankles) / len(visible_ankles),
                sum(point[1] for point in visible_ankles) / len(visible_ankles),
            )
    x1, _, x2, y2 = map(float, box)
    return (x1 + x2) / 2.0, y2


def _goalkeeper_track_ids(
    player_items: list[tuple[int, Any, int | None, Any | None, Point, Point | None, str | None]],
) -> set[str]:
    located = [
        (str(track_id), team, pitch)
        for _index, _box, track_id, _keypoints, _anchor, pitch, team in player_items
        if track_id is not None and pitch is not None
    ]
    if not located:
        return set()
    keepers: set[str] = set()
    by_team: dict[str, list[tuple[str, Point]]] = {}
    unlabelled: list[tuple[str, Point]] = []
    for track_id, team, pitch in located:
        if team:
            by_team.setdefault(team, []).append((track_id, pitch))
        else:
            unlabelled.append((track_id, pitch))
    groups = list(by_team.values())
    if not groups and unlabelled:
        groups = [unlabelled]
    for members in groups:
        median_x = sorted(pitch[0] for _track, pitch in members)[len(members) // 2]
        own_goal = 0.0 if median_x < PITCH_LENGTH / 2.0 else PITCH_LENGTH
        track_id, pitch = min(members, key=lambda item: abs(item[1][0] - own_goal))
        if abs(pitch[0] - own_goal) <= 18.0:
            keepers.add(track_id)
    return keepers


def _speed_label(speed_per_sec: float, calibrated: bool) -> str:
    if calibrated:
        return f"{speed_per_sec * 3.6:.1f} km/h"
    return f"{speed_per_sec:.1f} px/s"


def _keypoints_payload(
    keypoints: Any,
    projector: Any | None,
    threshold: float,
) -> list[list[float]]:
    payload: list[list[float]] = []
    for keypoint in keypoints:
        x = float(keypoint[0])
        y = float(keypoint[1])
        confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
        row = [x, y, confidence]
        if projector is not None and confidence >= threshold:
            pitch = _project_on_pitch(projector, (x, y))
            if pitch is not None:
                row.extend([round(pitch[0], 3), round(pitch[1], 3)])
        payload.append(row)
    return payload


def _project_on_pitch(projector: Any | None, point: Point) -> Point | None:
    if projector is None:
        return None
    try:
        x, y = projector.project(point)
    except (ArithmeticError, ValueError):
        return None
    if not (0.0 <= x <= PITCH_LENGTH and 0.0 <= y <= PITCH_WIDTH):
        return None
    return x, y


def _draw_skeleton(cv2: Any, frame: Any, keypoints: Any, threshold: float) -> None:
    for first, second in COCO_SKELETON:
        first_conf = float(keypoints[first][2]) if len(keypoints[first]) > 2 else 1.0
        second_conf = float(keypoints[second][2]) if len(keypoints[second]) > 2 else 1.0
        if first_conf < threshold or second_conf < threshold:
            continue
        first_point = tuple(map(int, keypoints[first][:2]))
        second_point = tuple(map(int, keypoints[second][:2]))
        cv2.line(frame, first_point, second_point, (255, 210, 40), 2, cv2.LINE_AA)
    for keypoint in keypoints:
        confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
        if confidence >= threshold:
            cv2.circle(frame, tuple(map(int, keypoint[:2])), 2, (40, 240, 255), -1, cv2.LINE_AA)


def annotate_video(
    input_path: Path,
    output_path: Path,
    config: VisionConfig | None = None,
    projector: Projector | None = None,
    observations_path: Path | None = None,
    roi: PolygonROI | None = None,
) -> VisionRunSummary:
    settings = config or VisionConfig()
    settings.validate()
    if not input_path.is_file():
        raise FileNotFoundError(f"video not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output video paths must be different")
    if observations_path and observations_path.resolve() in {
        input_path.resolve(),
        output_path.resolve(),
    }:
        raise ValueError("observations path must differ from video paths")

    cv2, YOLO = _load_dependencies()
    pose_model = YOLO(settings.pose_model)
    player_model = YOLO(settings.detection_model)
    ball_model_name = settings.ball_detection_model or settings.detection_model
    ball_model = YOLO(ball_model_name)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("video has invalid FPS or dimensions")
    if settings.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, settings.start_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"cannot create output video: {output_path}")

    observations_file = None
    if observations_path:
        observations_path.parent.mkdir(parents=True, exist_ok=True)
        observations_file = observations_path.open("w", encoding="utf-8")

    speed = SpeedEstimator(settings.speed_smoothing)
    team_classifier = (
        OnlineTeamClassifier(min_tracks=settings.team_min_tracks)
        if settings.team_classification
        else None
    )
    scene_detector = (
        SceneCutDetector(settings.scene_cut_threshold)
        if settings.scene_cut_threshold is not None
        else None
    )
    scene_cuts_detected = 0
    field_geometry_frames = 0
    player_tracks: set[int] = set()
    ball_tracks: set[int] = set()
    ball_associator = TileBallAssociator()
    frame_index = settings.start_frame
    frames_processed = 0
    calibrated_frames = 0
    field_detector = (
        FieldGeometryDetector(
            cv2,
            FieldLineConfig(
                min_length_ratio=settings.field_line_min_length_ratio,
                max_segments=settings.field_line_max_segments,
            ),
        )
        if settings.field_lines
        else None
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if settings.max_frames is not None and frames_processed >= settings.max_frames:
                break
            timestamp_sec = frame_index / fps
            frame_projector = projector_for_frame(projector, frame_index)
            calibration_confidence = projector_confidence(projector, frame_index)
            projector_source = "oracle" if frame_projector is not None else None
            scene_cut = scene_detector.update(frame) if scene_detector else False
            if scene_cut:
                scene_cuts_detected += 1
                speed.reset()
                ball_associator.reset()
                if team_classifier is not None:
                    team_classifier.reset()

            field_geometry = field_detector.detect(frame) if field_detector else None
            if field_geometry is not None:
                field_geometry_frames += 1
                draw_field_geometry(cv2, frame, field_geometry)
                if frame_projector is None:
                    fallback = projector_from_pitch_polygon(field_geometry.pitch_polygon)
                    if fallback is not None:
                        frame_projector = fallback
                        calibration_confidence = VIDEO_ONLY_CALIBRATION_CONFIDENCE
                        projector_source = "video_only_polygon"
            if frame_projector is not None:
                calibrated_frames += 1
                if projector_source == "video_only_polygon":
                    overlay_label = (
                        f"VIDEO-ONLY pitch (grass polygon) | baseline {calibration_confidence:.2f}"
                    )
                    overlay_color = (40, 200, 255)
                else:
                    overlay_label = (
                        f"CALIBRATED pitch | confidence {calibration_confidence or 0.0:.2f}"
                    )
                    overlay_color = (40, 230, 100)
                cv2.putText(
                    frame,
                    overlay_label,
                    (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    overlay_color,
                    1,
                    cv2.LINE_AA,
                )
            automatic_roi = (
                PolygonROI(field_geometry.pitch_polygon)
                if field_geometry is not None and len(field_geometry.pitch_polygon) >= 3
                else None
            )
            player_roi = roi
            ball_roi = (roi or automatic_roi) if settings.filter_ball_by_roi else None
            pose_confidence = max(0.08, settings.player_confidence - 0.08)

            pose_result = pose_model.predict(
                frame,
                classes=[COCO_PERSON_CLASS],
                conf=pose_confidence,
                imgsz=settings.image_size,
                device=settings.device,
                verbose=False,
            )[0]
            player_result = player_model.track(
                frame,
                persist=True,
                tracker=settings.player_tracker,
                classes=[COCO_PERSON_CLASS],
                conf=settings.player_confidence,
                imgsz=settings.image_size,
                device=settings.device,
                verbose=False,
            )[0]
            ball_result = ball_model.track(
                frame,
                persist=True,
                tracker=settings.ball_tracker,
                classes=[settings.ball_class_id],
                conf=settings.ball_confidence,
                imgsz=settings.image_size,
                device=settings.device,
                verbose=False,
            )[0]

            player_boxes = player_result.boxes
            player_ids = _track_ids(player_boxes)
            player_xyxy = _to_numpy(player_boxes.xyxy) if player_boxes is not None else []
            player_confidences = (
                _to_numpy(player_boxes.conf).reshape(-1) if player_boxes is not None else []
            )
            pose_boxes = pose_result.boxes
            pose_xyxy = _to_numpy(pose_boxes.xyxy) if pose_boxes is not None else []
            keypoint_data = (
                _to_numpy(pose_result.keypoints.data)
                if pose_result.keypoints is not None
                else None
            )
            frame_objects: list[dict[str, object]] = []
            player_items: list[tuple[int, Any, int | None, Any | None, Point]] = []
            team_features: dict[str, Any] = {}
            used_pose_indexes: set[int] = set()
            matched_players: list[tuple[int, Any, int | None, Any | None]] = []
            for index, box in enumerate(player_xyxy):
                keypoints = _match_keypoints(
                    box, pose_xyxy, keypoint_data, used_pose_indexes
                )
                matched_players.append((index, box, player_ids[index], keypoints))
            if settings.pose_on_crop:
                missing_indexes = [
                    item_index
                    for item_index, item in enumerate(matched_players)
                    if item[3] is None
                ]
                if missing_indexes:
                    refined = _pose_on_player_crops(
                        pose_model,
                        frame,
                        [matched_players[item_index][1] for item_index in missing_indexes],
                        confidence=pose_confidence,
                        image_size=settings.pose_crop_imgsz,
                        device=settings.device,
                        keypoint_confidence=settings.keypoint_confidence,
                        pad_ratio=settings.pose_crop_pad,
                    )
                    for item_index, keypoints in zip(missing_indexes, refined):
                        if keypoints is None:
                            continue
                        player_index, box, track_id, _keypoints = matched_players[item_index]
                        matched_players[item_index] = (
                            player_index,
                            box,
                            track_id,
                            keypoints,
                        )
            for index, box, track_id, keypoints in matched_players:
                anchor = _player_anchor(box, keypoints, settings.keypoint_confidence)
                if player_roi is not None and not player_roi.contains(anchor):
                    continue
                player_items.append((index, box, track_id, keypoints, anchor))
                if team_classifier is not None and track_id is not None:
                    feature = extract_jersey_feature(frame, box, cv2)
                    if feature is not None:
                        team_features[str(track_id)] = feature

            team_labels = (
                team_classifier.update_frame(team_features)
                if team_classifier is not None
                else {}
            )
            team_colors = {
                "team_a": (70, 220, 90),
                "team_b": (255, 170, 60),
                None: (180, 180, 180),
            }
            located_items: list[
                tuple[int, Any, int | None, Any | None, Point, Point | None, str | None]
            ] = []
            for index, box, track_id, keypoints, anchor in player_items:
                pitch_position = _project_on_pitch(frame_projector, anchor)
                team = team_labels.get(str(track_id)) if track_id is not None else None
                located_items.append(
                    (index, box, track_id, keypoints, anchor, pitch_position, team)
                )
            keeper_ids = _goalkeeper_track_ids(located_items)
            for index, box, track_id, keypoints, anchor, pitch_position, team in located_items:
                calibrated = pitch_position is not None
                velocity = 0.0
                if track_id is not None:
                    player_tracks.add(track_id)
                    velocity = speed.update(
                        f"player:{track_id}:{'pitch' if calibrated else 'image'}",
                        pitch_position or anchor,
                        timestamp_sec,
                    )
                object_type = (
                    "goalkeeper"
                    if track_id is not None and str(track_id) in keeper_ids
                    else "player"
                )
                color = team_colors[team]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                identity = "?" if track_id is None else str(track_id)
                team_text = team or "unknown"
                _draw_label(
                    cv2,
                    frame,
                    (x1, y1),
                    f"{object_type} {team_text} #{identity} | {_speed_label(velocity, calibrated)}",
                    color,
                )
                if keypoints is not None:
                    _draw_skeleton(
                        cv2, frame, keypoints, settings.keypoint_confidence
                    )
                player_data: dict[str, object] = {
                    "track_id": str(track_id) if track_id is not None else None,
                    "object_type": object_type,
                    "confidence": float(player_confidences[index]),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    },
                    "anchor_image": [round(anchor[0], 3), round(anchor[1], 3)],
                    "speed": round(velocity * 3.6 if calibrated else velocity, 3),
                    "speed_unit": "km/h" if calibrated else "px/s",
                }
                if team is not None:
                    player_data["team"] = team
                if keypoints is not None:
                    player_data["keypoints"] = _keypoints_payload(
                        keypoints,
                        frame_projector,
                        settings.keypoint_confidence,
                    )
                if pitch_position is not None:
                    player_data["pitch_x"] = round(pitch_position[0], 3)
                    player_data["pitch_y"] = round(pitch_position[1], 3)
                frame_objects.append(player_data)

            ball_rows = merge_ball_detections(
                _detection_rows(ball_result.boxes),
                _tiled_ball_detections(
                    ball_model,
                    frame,
                    class_id=settings.ball_class_id,
                    confidence=settings.ball_confidence,
                    image_size=settings.image_size,
                    device=settings.device,
                    tile_size=settings.ball_tile_size,
                    overlap=settings.ball_tile_overlap,
                    bounds=_pitch_tile_bounds(field_geometry, width, height),
                ),
                max_side=settings.ball_max_side_px,
            )
            pending_indexes = [
                index for index, row in enumerate(ball_rows) if row[2] is None
            ]
            if pending_indexes:
                assigned_ids = ball_associator.assign(
                    [ball_rows[index][0] for index in pending_indexes]
                )
                for index, track_id in zip(pending_indexes, assigned_ids):
                    box, confidence, _track_id = ball_rows[index]
                    ball_rows[index] = (box, confidence, track_id)
            else:
                ball_associator.assign([])
            for box, box_confidence, track_id in ball_rows:
                x1, y1, x2, y2 = map(int, box)
                center = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
                if ball_roi is not None and not ball_roi.contains(center):
                    continue
                pitch_position = _project_on_pitch(frame_projector, center)
                calibrated = pitch_position is not None
                velocity = 0.0
                if track_id is not None:
                    ball_tracks.add(track_id)
                    velocity = speed.update(
                        f"ball:{track_id}:{'pitch' if calibrated else 'image'}",
                        pitch_position or center,
                        timestamp_sec,
                    )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 165, 255), 2)
                identity = "?" if track_id is None else str(track_id)
                _draw_label(
                    cv2,
                    frame,
                    (x1, y1),
                    f"ball #{identity} | {_speed_label(velocity, calibrated)}",
                    (40, 165, 255),
                )
                ball_data: dict[str, object] = {
                    "track_id": str(track_id) if track_id is not None else None,
                    "object_type": "ball",
                    "confidence": float(box_confidence),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    },
                    "anchor_image": [round(center[0], 3), round(center[1], 3)],
                    "speed": round(velocity * 3.6 if calibrated else velocity, 3),
                    "speed_unit": "km/h" if calibrated else "px/s",
                }
                if pitch_position is not None:
                    ball_data["pitch_x"] = round(pitch_position[0], 3)
                    ball_data["pitch_y"] = round(pitch_position[1], 3)
                frame_objects.append(ball_data)

            writer.write(frame)
            if observations_file:
                frame_payload: dict[str, object] = {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 4),
                    "scene_cut": scene_cut,
                    "objects": frame_objects,
                }
                if calibration_confidence is not None:
                    frame_payload["calibration_confidence"] = round(
                        calibration_confidence, 4
                    )
                if field_geometry is not None:
                    frame_payload["field_geometry"] = field_geometry.to_dict()
                observations_file.write(
                    json.dumps(frame_payload, ensure_ascii=False)
                    + "\n"
                )
            frame_index += 1
            frames_processed += 1
            if frames_processed % 100 == 0:
                print(f"processed {frames_processed} frames", file=sys.stderr)
    finally:
        if observations_file:
            observations_file.close()
        writer.release()
        capture.release()

    return VisionRunSummary(
        input_path=str(input_path.resolve()),
        output_path=str(output_path.resolve()),
        observations_path=(
            str(observations_path.resolve()) if observations_path else None
        ),
        frames_processed=frames_processed,
        fps=round(fps, 4),
        width=width,
        height=height,
        unique_player_tracks=len(player_tracks),
        unique_ball_tracks=len(ball_tracks),
        speed_unit="km/h" if projector else "px/s",
        pose_model=settings.pose_model,
        detection_model=settings.detection_model,
        ball_detection_model=ball_model_name,
        roi_enabled=roi is not None or settings.field_lines,
        team_classification_enabled=settings.team_classification,
        scene_cuts_detected=scene_cuts_detected,
        field_geometry_enabled=settings.field_lines,
        field_geometry_frames=field_geometry_frames,
        start_frame=settings.start_frame,
        end_frame=(frame_index - 1 if frames_processed else None),
        calibrated_frames=calibrated_frames,
    )
