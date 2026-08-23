from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldLineConfig:
    min_length_ratio: float = 0.06
    max_line_gap_ratio: float = 0.02
    hough_threshold: int = 60
    max_segments: int = 40
    green_hue_min: int = 25
    green_hue_max: int = 100
    white_saturation_max: int = 95
    white_value_min: int = 145

    def validate(self) -> None:
        if not 0.0 < self.min_length_ratio <= 1.0:
            raise ValueError("min_length_ratio must be in (0, 1]")
        if not 0.0 <= self.max_line_gap_ratio <= 1.0:
            raise ValueError("max_line_gap_ratio must be in [0, 1]")
        if self.hough_threshold <= 0 or self.max_segments <= 0:
            raise ValueError("Hough threshold and max segments must be positive")
        if not 0 <= self.green_hue_min < self.green_hue_max <= 180:
            raise ValueError("invalid green HSV hue range")
        if not 0 <= self.white_saturation_max <= 255:
            raise ValueError("white_saturation_max must be in [0, 255]")
        if not 0 <= self.white_value_min <= 255:
            raise ValueError("white_value_min must be in [0, 255]")


@dataclass(frozen=True)
class FieldLineSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float
    angle_deg: float
    confidence: float

    @classmethod
    def from_endpoints(
        cls, x1: float, y1: float, x2: float, y2: float, frame_width: int
    ) -> "FieldLineSegment":
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        confidence = min(1.0, length / max(frame_width * 0.4, 1.0))
        return cls(
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            round(length, 3),
            round(angle, 3),
            round(confidence, 4),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldLineSegment":
        segment = cls(*(float(data[key]) for key in asdict_fields(cls)))
        segment.validate()
        return segment

    def validate(self) -> None:
        if self.length_px <= 0:
            raise ValueError("field line length must be positive")
        if not 0.0 <= self.angle_deg < 180.0:
            raise ValueError("field line angle must be in [0, 180)")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("field line confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FieldIntersection:
    x: float
    y: float
    angle_deg: float
    kind: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldIntersection":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            angle_deg=float(data["angle_deg"]),
            kind=str(data["kind"]),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FieldGeometry:
    method: str
    frame_width: int
    frame_height: int
    pitch_mask_coverage: float
    line_mask_coverage: float
    segments: tuple[FieldLineSegment, ...]
    intersections: tuple[FieldIntersection, ...]
    pitch_polygon: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldGeometry":
        return cls(
            method=str(data["method"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            pitch_mask_coverage=float(data["pitch_mask_coverage"]),
            line_mask_coverage=float(data["line_mask_coverage"]),
            segments=tuple(
                FieldLineSegment.from_dict(item) for item in data.get("segments", [])
            ),
            intersections=tuple(
                FieldIntersection.from_dict(item)
                for item in data.get("intersections", [])
            ),
            pitch_polygon=tuple(
                tuple(float(value) for value in point)
                for point in data.get("pitch_polygon", [])
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "pitch_mask_coverage": self.pitch_mask_coverage,
            "line_mask_coverage": self.line_mask_coverage,
            "segments": [segment.to_dict() for segment in self.segments],
            "intersections": [point.to_dict() for point in self.intersections],
            "pitch_polygon": [list(point) for point in self.pitch_polygon],
        }


def asdict_fields(data_class: type[Any]) -> tuple[str, ...]:
    return tuple(data_class.__dataclass_fields__)


def _angle_distance(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _deduplicate_segments(
    segments: list[FieldLineSegment], max_segments: int
) -> tuple[FieldLineSegment, ...]:
    selected: list[FieldLineSegment] = []
    for candidate in sorted(segments, key=lambda item: item.length_px, reverse=True):
        midpoint = ((candidate.x1 + candidate.x2) / 2, (candidate.y1 + candidate.y2) / 2)
        duplicate = False
        for existing in selected:
            existing_midpoint = (
                (existing.x1 + existing.x2) / 2,
                (existing.y1 + existing.y2) / 2,
            )
            if _angle_distance(candidate.angle_deg, existing.angle_deg) <= 2.5 and math.hypot(
                midpoint[0] - existing_midpoint[0], midpoint[1] - existing_midpoint[1]
            ) <= 18.0:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_segments:
            break
    return tuple(selected)


def _segment_intersection(
    first: FieldLineSegment,
    second: FieldLineSegment,
    width: int,
    height: int,
) -> FieldIntersection | None:
    angle = _angle_distance(first.angle_deg, second.angle_deg)
    if angle < 12.0:
        return None
    x1, y1, x2, y2 = first.x1, first.y1, first.x2, first.y2
    x3, y3, x4, y4 = second.x1, second.y1, second.x2, second.y2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    x = (determinant_first * (x3 - x4) - (x1 - x2) * determinant_second) / denominator
    y = (determinant_first * (y3 - y4) - (y1 - y2) * determinant_second) / denominator
    if not (0.0 <= x < width and 0.0 <= y < height):
        return None

    def near_segment(segment: FieldLineSegment) -> bool:
        margin = 25.0
        return (
            min(segment.x1, segment.x2) - margin <= x <= max(segment.x1, segment.x2) + margin
            and min(segment.y1, segment.y2) - margin <= y <= max(segment.y1, segment.y2) + margin
        )

    if not near_segment(first) or not near_segment(second):
        return None
    kind = "cross" if 70.0 <= angle <= 110.0 else "junction"
    return FieldIntersection(round(x, 3), round(y, 3), round(angle, 3), kind)


def _find_intersections(
    segments: tuple[FieldLineSegment, ...], width: int, height: int
) -> tuple[FieldIntersection, ...]:
    points: list[FieldIntersection] = []
    for index, first in enumerate(segments):
        for second in segments[index + 1 :]:
            point = _segment_intersection(first, second, width, height)
            if point is None:
                continue
            if any(math.hypot(point.x - item.x, point.y - item.y) < 15.0 for item in points):
                continue
            points.append(point)
    return tuple(points[:40])


class FieldGeometryDetector:
    def __init__(self, cv2: Any, config: FieldLineConfig | None = None) -> None:
        self.cv2 = cv2
        self.config = config or FieldLineConfig()
        self.config.validate()

    def detect(self, frame: Any) -> FieldGeometry:
        cv2 = self.cv2
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pitch_mask = cv2.inRange(
            hsv,
            np.array([self.config.green_hue_min, 35, 30], dtype=np.uint8),
            np.array([self.config.green_hue_max, 255, 255], dtype=np.uint8),
        )
        kernel_size = max(3, round(min(width, height) * 0.008))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        pitch_polygon: tuple[tuple[float, float], ...] = ()
        if contours:
            largest_pitch = max(contours, key=cv2.contourArea)
            connected_pitch = np.zeros_like(pitch_mask)
            cv2.drawContours(connected_pitch, [largest_pitch], -1, 255, cv2.FILLED)
            pitch_mask = connected_pitch
            epsilon = 0.01 * cv2.arcLength(largest_pitch, True)
            approximated = cv2.approxPolyDP(largest_pitch, epsilon, True)
            pitch_polygon = tuple(
                (float(point[0][0]), float(point[0][1])) for point in approximated
            )
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.config.white_value_min], dtype=np.uint8),
            np.array([180, self.config.white_saturation_max, 255], dtype=np.uint8),
        )
        # The filled largest contour includes markings inside the grass region while
        # excluding similarly white advertising boards immediately outside it.
        boundary_kernel_size = max(3, round(min(width, height) * 0.02))
        if boundary_kernel_size % 2 == 0:
            boundary_kernel_size += 1
        interior_pitch_mask = cv2.erode(
            pitch_mask,
            np.ones((boundary_kernel_size, boundary_kernel_size), dtype=np.uint8),
            iterations=1,
        )
        line_mask = cv2.bitwise_and(white_mask, interior_pitch_mask)
        edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)
        raw_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=self.config.hough_threshold,
            minLineLength=max(20, round(width * self.config.min_length_ratio)),
            maxLineGap=max(5, round(width * self.config.max_line_gap_ratio)),
        )
        candidates: list[FieldLineSegment] = []
        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = (float(value) for value in line.reshape(-1)[:4])
                candidates.append(FieldLineSegment.from_endpoints(x1, y1, x2, y2, width))
        segments = _deduplicate_segments(candidates, self.config.max_segments)
        intersections = _find_intersections(segments, width, height)
        pixel_count = max(width * height, 1)
        return FieldGeometry(
            method="opencv_hsv_hough_v1",
            frame_width=width,
            frame_height=height,
            pitch_mask_coverage=round(float(np.count_nonzero(pitch_mask)) / pixel_count, 5),
            line_mask_coverage=round(float(np.count_nonzero(line_mask)) / pixel_count, 5),
            segments=segments,
            intersections=intersections,
            pitch_polygon=pitch_polygon,
        )


def draw_field_geometry(cv2: Any, frame: Any, geometry: FieldGeometry) -> None:
    for segment in geometry.segments:
        cv2.line(
            frame,
            (round(segment.x1), round(segment.y1)),
            (round(segment.x2), round(segment.y2)),
            (30, 70, 255),
            2,
            cv2.LINE_AA,
        )
    for point in geometry.intersections:
        cv2.circle(frame, (round(point.x), round(point.y)), 5, (255, 80, 30), -1)


def analyze_field_image(
    input_path: Path,
    output_json: Path,
    overlay_path: Path | None = None,
    config: FieldLineConfig | None = None,
) -> dict[str, object]:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'OpenCV is missing. Install with: python -m pip install -e ".[video]"'
        ) from error
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise ValueError(f"cannot read image: {input_path}")
    geometry = FieldGeometryDetector(cv2, config).detect(frame)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(geometry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_field_geometry(cv2, frame, geometry)
        if not cv2.imwrite(str(overlay_path), frame):
            raise ValueError(f"cannot write overlay image: {overlay_path}")
    return {
        "input": str(input_path.resolve()),
        "output_json": str(output_json.resolve()),
        "overlay": str(overlay_path.resolve()) if overlay_path else None,
        "segment_count": len(geometry.segments),
        "intersection_count": len(geometry.intersections),
        "pitch_mask_coverage": geometry.pitch_mask_coverage,
        "line_mask_coverage": geometry.line_mask_coverage,
        "method": geometry.method,
    }
