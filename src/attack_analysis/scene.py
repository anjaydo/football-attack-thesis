from __future__ import annotations

from typing import Any

import numpy as np


class SceneCutDetector:
    """Detect abrupt visual changes using a lightweight sampled-frame difference."""

    def __init__(self, threshold: float = 0.35, sample_step: int = 16) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("scene cut threshold must be in (0, 1]")
        if sample_step < 1:
            raise ValueError("sample_step must be positive")
        self.threshold = threshold
        self.sample_step = sample_step
        self.previous: np.ndarray | None = None

    def update(self, frame: Any) -> bool:
        sample = np.asarray(frame)[:: self.sample_step, :: self.sample_step].astype(
            np.float32
        )
        if self.previous is None:
            self.previous = sample
            return False
        if sample.shape != self.previous.shape:
            self.previous = sample
            return True
        difference = float(np.mean(np.abs(sample - self.previous)) / 255.0)
        self.previous = sample
        return difference >= self.threshold

