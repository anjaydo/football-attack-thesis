from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


Point = tuple[float, float]


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

