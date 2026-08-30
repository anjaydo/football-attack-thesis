from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .geometry import Point, Projector, projector_confidence, projector_for_frame
from .field_geometry import FieldGeometryDetector, FieldLineConfig, draw_field_geometry
from .models import PITCH_LENGTH, PITCH_WIDTH
from .motion import SpeedEstimator
from .roi import PolygonROI
from .scene import SceneCutDetector
from .team import OnlineTeamClassifier, extract_jersey_feature


COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32
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
    minimum_iou: float = 0.25,
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


def _speed_label(speed_per_sec: float, calibrated: bool) -> str:
    if calibrated:
        return f"{speed_per_sec * 3.6:.1f} km/h"
    return f"{speed_per_sec:.1f} px/s"


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
            if frame_projector is not None:
                calibrated_frames += 1
                cv2.putText(
                    frame,
                    f"CALIBRATED pitch | confidence {calibration_confidence or 0.0:.2f}",
                    (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (40, 230, 100),
                    1,
                    cv2.LINE_AA,
                )
            scene_cut = scene_detector.update(frame) if scene_detector else False
            if scene_cut:
                scene_cuts_detected += 1
                speed.reset()
                if team_classifier is not None:
                    team_classifier.reset()

            field_geometry = field_detector.detect(frame) if field_detector else None
            if field_geometry is not None:
                field_geometry_frames += 1
                draw_field_geometry(cv2, frame, field_geometry)
            automatic_roi = (
                PolygonROI(field_geometry.pitch_polygon)
                if field_geometry is not None and len(field_geometry.pitch_polygon) >= 3
                else None
            )
            player_roi = roi or automatic_roi
            ball_roi = player_roi if settings.filter_ball_by_roi else None

            pose_result = pose_model.predict(
                frame,
                classes=[COCO_PERSON_CLASS],
                conf=settings.player_confidence,
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
            for index, box in enumerate(player_xyxy):
                track_id = player_ids[index]
                keypoints = _match_keypoints(
                    box, pose_xyxy, keypoint_data, used_pose_indexes
                )
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
            for index, box, track_id, keypoints, anchor in player_items:
                pitch_position = _project_on_pitch(frame_projector, anchor)
                calibrated = pitch_position is not None
                velocity = 0.0
                if track_id is not None:
                    player_tracks.add(track_id)
                    velocity = speed.update(
                        f"player:{track_id}:{'pitch' if calibrated else 'image'}",
                        pitch_position or anchor,
                        timestamp_sec,
                    )
                team = team_labels.get(str(track_id)) if track_id is not None else None
                color = team_colors[team]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                identity = "?" if track_id is None else str(track_id)
                team_text = team or "unknown"
                _draw_label(
                    cv2,
                    frame,
                    (x1, y1),
                    f"player {team_text} #{identity} | {_speed_label(velocity, calibrated)}",
                    color,
                )
                if keypoints is not None:
                    _draw_skeleton(
                        cv2, frame, keypoints, settings.keypoint_confidence
                    )
                player_data: dict[str, object] = {
                    "track_id": str(track_id) if track_id is not None else None,
                    "object_type": "player",
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
                    player_data["keypoints"] = [
                        [float(value) for value in keypoint] for keypoint in keypoints
                    ]
                if pitch_position is not None:
                    player_data["pitch_x"] = round(pitch_position[0], 3)
                    player_data["pitch_y"] = round(pitch_position[1], 3)
                frame_objects.append(player_data)

            ball_boxes = ball_result.boxes
            ball_ids = _track_ids(ball_boxes)
            ball_xyxy = _to_numpy(ball_boxes.xyxy) if ball_boxes is not None else []
            ball_confidences = (
                _to_numpy(ball_boxes.conf).reshape(-1) if ball_boxes is not None else []
            )
            for index, box in enumerate(ball_xyxy):
                track_id = ball_ids[index]
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
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
                    "confidence": float(ball_confidences[index]),
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
