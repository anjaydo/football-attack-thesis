from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import atan2, exp, hypot
from pathlib import Path
from typing import Any, Mapping

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


@dataclass(frozen=True)
class TrainedXGModel:
    model: Any
    medians: Any
    feature_columns: tuple[str, ...]
    source: str
    calibrator: Any | None = None

    def predict(self, features: Mapping[str, object]) -> float:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'ml' extra to run trained xG inference") from error
        if features.get("x_shooter") is None or features.get("y_shooter") is None:
            raise ValueError("trained xG requires shooter pitch coordinates")
        row = pd.DataFrame([{name: features.get(name) for name in self.feature_columns}])
        median_values = {
            name: float(self.medians.get(name, 0.0)) if hasattr(self.medians, "get") else 0.0
            for name in self.feature_columns
        }
        row = row.apply(pd.to_numeric, errors="coerce").fillna(median_values).fillna(0.0)
        raw = float(self.model.predict_proba(row.loc[:, self.feature_columns])[:, 1][0])
        if self.calibrator is not None:
            raw = float(self.calibrator.predict_proba(np.asarray([[raw]], dtype=float))[:, 1][0])
        return round(max(0.001, min(0.95, raw)), 4)


@lru_cache(maxsize=4)
def load_trained_xg_model(path: str) -> TrainedXGModel:
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'ml' extra to load a trained xG model") from error
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"trained xG artifact not found: {artifact_path}")
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict) or "model" not in payload or "medians" not in payload:
        raise ValueError("trained xG artifact has an unsupported format")
    report = dict(payload.get("report") or {})
    feature_columns = tuple(payload.get("feature_columns") or report.get("feature_columns") or ())
    if not feature_columns:
        raise ValueError("trained xG artifact does not declare feature columns")
    selected = str(report.get("selected_model") or type(payload["model"]).__name__)
    schema_version = str(payload.get("schema_version") or "legacy")
    return TrainedXGModel(
        model=payload["model"],
        medians=payload["medians"],
        feature_columns=feature_columns,
        calibrator=payload.get("calibrator"),
        source=f"trained_xg:{selected}:{schema_version}",
    )
