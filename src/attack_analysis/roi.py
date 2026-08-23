from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .geometry import Point


@dataclass(frozen=True)
class PolygonROI:
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("ROI polygon requires at least three points")
        area = 0.0
        for first, second in zip(self.points, self.points[1:] + self.points[:1]):
            area += first[0] * second[1] - second[0] * first[1]
        if abs(area) < 1e-6:
            raise ValueError("ROI polygon must have non-zero area")

    @classmethod
    def from_json(cls, path: Path) -> "PolygonROI":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        raw_points = payload.get("polygon") or payload.get("image_points")
        if raw_points is None:
            raise ValueError("ROI JSON must contain polygon or image_points")
        return cls(tuple(tuple(map(float, point)) for point in raw_points))

    def contains(self, point: Point) -> bool:
        x, y = point
        inside = False
        for first, second in zip(self.points, self.points[1:] + self.points[:1]):
            x1, y1 = first
            x2, y2 = second

            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if abs(cross) < 1e-6 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
                return True

            if (y1 > y) != (y2 > y):
                intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < intersection_x:
                    inside = not inside
        return inside

