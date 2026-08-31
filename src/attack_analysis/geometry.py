from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


Point = tuple[float, float]


class FramePitchProjector(Protocol):
    """A calibration provider whose homography can change with the video frame."""

    def for_frame(self, frame_index: int) -> "PitchProjector | None": ...

    def confidence_for_frame(self, frame_index: int) -> float | None: ...


def compute_homography(source: Sequence[Point], destination: Sequence[Point]) -> np.ndarray:
    """Compute a projective transform from four point correspondences."""
    if len(source) != 4 or len(destination) != 4:
        raise ValueError("homography requires exactly four source and destination points")

    coefficients: list[list[float]] = []
    targets: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        coefficients.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        targets.append(u)
        coefficients.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        targets.append(v)

    try:
        solution = np.linalg.solve(
            np.asarray(coefficients, dtype=float),
            np.asarray(targets, dtype=float),
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("calibration points do not define a valid homography") from error
    return np.append(solution, 1.0).reshape(3, 3)


@dataclass(frozen=True)
class PitchProjector:
    matrix: np.ndarray

    @classmethod
    def from_points(
        cls,
        image_points: Sequence[Point],
        pitch_points: Sequence[Point],
    ) -> "PitchProjector":
        return cls(compute_homography(image_points, pitch_points))

    @classmethod
    def from_json(cls, path: Path) -> "PitchProjector":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        image_points = [tuple(map(float, point)) for point in payload["image_points"]]
        pitch_points = [tuple(map(float, point)) for point in payload["pitch_points"]]
        return cls.from_points(image_points, pitch_points)

    def project(self, point: Point) -> Point:
        vector = self.matrix @ np.asarray([point[0], point[1], 1.0], dtype=float)
        if abs(vector[2]) < 1e-9:
            raise ValueError("point projects to infinity")
        return float(vector[0] / vector[2]), float(vector[1] / vector[2])


Projector = PitchProjector | FramePitchProjector


def projector_for_frame(
    projector: Projector | None,
    frame_index: int,
) -> PitchProjector | None:
    if projector is None:
        return None
    resolver = getattr(projector, "for_frame", None)
    return resolver(frame_index) if callable(resolver) else projector


def projector_confidence(
    projector: Projector | None,
    frame_index: int,
) -> float | None:
    if projector is None:
        return None
    resolver = getattr(projector, "confidence_for_frame", None)
    if callable(resolver):
        value = resolver(frame_index)
        return None if value is None else min(1.0, max(0.0, float(value)))
    return 1.0


VIDEO_ONLY_CALIBRATION_CONFIDENCE = 0.25


def ordered_image_quad(
    polygon: Sequence[Point],
) -> tuple[Point, Point, Point, Point] | None:
    """Order a visible-pitch polygon as near-left, near-right, far-right, far-left.

    Near is the larger image-y (bottom of the frame). This is a video-only
    estimate: sideline y=0 vs y=68 may still be flipped relative to the
    FIFA axis, and the grass contour is not a semantic landmark set.
    """

    if len(polygon) < 4:
        return None
    points = [(float(x), float(y)) for x, y in polygon]
    by_y = sorted(points, key=lambda item: item[1])
    mid_y = by_y[len(by_y) // 2][1]
    top = [item for item in points if item[1] <= mid_y]
    bottom = [item for item in points if item[1] > mid_y]
    if len(top) < 2 or len(bottom) < 2:
        top = by_y[:2]
        bottom = by_y[-2:]
    far_left = min(top, key=lambda item: item[0])
    far_right = max(top, key=lambda item: item[0])
    near_left = min(bottom, key=lambda item: item[0])
    near_right = max(bottom, key=lambda item: item[0])
    if far_left[0] == far_right[0] or near_left[0] == near_right[0]:
        return None
    return near_left, near_right, far_right, far_left


def projector_from_pitch_polygon(
    polygon: Sequence[Point],
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> PitchProjector | None:
    """Homography from a grass-mask quad onto a 105×68 m pitch (research baseline)."""

    quad = ordered_image_quad(polygon)
    if quad is None:
        return None
    near_left, near_right, far_right, far_left = quad
    try:
        return PitchProjector.from_points(
            [near_left, near_right, far_right, far_left],
            [
                (0.0, 0.0),
                (pitch_length, 0.0),
                (pitch_length, pitch_width),
                (0.0, pitch_width),
            ],
        )
    except ValueError:
        return None
