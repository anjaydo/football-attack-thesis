from __future__ import annotations

from math import atan2, exp, hypot

from .geometry import Point
from .models import PITCH_LENGTH, PITCH_WIDTH


GOAL_WIDTH = 7.32
GOAL_CENTER_Y = PITCH_WIDTH / 2.0


def shot_distance(point: Point) -> float:
    return hypot(PITCH_LENGTH - point[0], GOAL_CENTER_Y - point[1])


def shot_angle(point: Point) -> float:
    """Return the visible goal angle in radians from an attacking-right position."""
    dx = max(1e-6, PITCH_LENGTH - point[0])
    dy = GOAL_CENTER_Y - point[1]
    return abs(atan2(GOAL_WIDTH * dx, dx * dx + dy * dy - (GOAL_WIDTH / 2.0) ** 2))


def heuristic_xg(point: Point) -> float:
    """Transparent geometry-only baseline; not a fitted provider-grade xG model."""
    distance = shot_distance(point)
    angle = shot_angle(point)
    logit = -1.5 - 0.08 * (distance - 11.0) + 1.2 * angle
    probability = 1.0 / (1.0 + exp(-logit))
    return round(max(0.001, min(0.95, probability)), 4)

