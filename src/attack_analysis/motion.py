from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Callable

from .geometry import Point


ProjectPoint = Callable[[Point], Point]


@dataclass
class _MotionState:
    point: Point
    timestamp_sec: float
    speed_per_sec: float | None = None


class SpeedEstimator:
    """Estimate smoothed per-track speed in pixels/s or pitch meters/s."""

    def __init__(self, smoothing: float = 0.35) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.smoothing = smoothing
        self._states: dict[str, _MotionState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(
        self,
        track_key: str,
        point: Point,
        timestamp_sec: float,
        project: ProjectPoint | None = None,
    ) -> float:
        measurement = project(point) if project else point
        previous = self._states.get(track_key)
        if previous is None:
            self._states[track_key] = _MotionState(measurement, timestamp_sec)
            return 0.0

        elapsed = timestamp_sec - previous.timestamp_sec
        if elapsed <= 0:
            return previous.speed_per_sec or 0.0

        instant_speed = hypot(
            measurement[0] - previous.point[0],
            measurement[1] - previous.point[1],
        ) / elapsed
        speed = (
            instant_speed
            if previous.speed_per_sec is None
            else self.smoothing * instant_speed
            + (1.0 - self.smoothing) * previous.speed_per_sec
        )
        self._states[track_key] = _MotionState(measurement, timestamp_sec, speed)
        return speed
