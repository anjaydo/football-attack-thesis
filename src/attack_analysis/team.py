from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _TrackColor:
    feature: np.ndarray
    last_seen_frame: int


def extract_jersey_feature(frame: Any, box: Any, cv2: Any) -> np.ndarray | None:
    """Extract a robust Lab-color feature from the central upper torso crop."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    left = max(0, int(x1 + 0.22 * box_width))
    right = min(width, int(x1 + 0.78 * box_width))
    top = max(0, int(y1 + 0.18 * box_height))
    bottom = min(height, int(y1 + 0.55 * box_height))
    if right - left < 2 or bottom - top < 2:
        return None

    crop = frame[top:bottom, left:right]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    if not len(lab):
        return None
    return np.median(lab, axis=0).astype(float) / 255.0


class OnlineTeamClassifier:
    """Cluster jersey colors into stable team_a/team_b labels across frames."""

    def __init__(
        self,
        min_tracks: int = 4,
        feature_smoothing: float = 0.25,
        memory_frames: int = 250,
        outlier_ratio: float = 0.75,
    ) -> None:
        if min_tracks < 2:
            raise ValueError("min_tracks must be at least 2")
        if not 0.0 < feature_smoothing <= 1.0:
            raise ValueError("feature_smoothing must be in (0, 1]")
        self.min_tracks = min_tracks
        self.feature_smoothing = feature_smoothing
        self.memory_frames = memory_frames
        self.outlier_ratio = outlier_ratio
        self.frame_index = -1
        self.track_colors: dict[str, _TrackColor] = {}
        self.centers: np.ndarray | None = None
        self.votes: dict[str, Counter[str]] = defaultdict(Counter)

    def reset(self) -> None:
        self.track_colors.clear()
        self.centers = None
        self.votes.clear()

    @staticmethod
    def _fit_two_clusters(features: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
        first, second = np.unravel_index(np.argmax(distances), distances.shape)
        centers = np.vstack([features[first], features[second]])
        for _ in range(20):
            assignment = np.argmin(
                np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2),
                axis=1,
            )
            updated = centers.copy()
            for index in range(2):
                members = features[assignment == index]
                if len(members):
                    updated[index] = members.mean(axis=0)
            if np.allclose(updated, centers, atol=1e-6):
                break
            centers = updated
        return centers

    def update_frame(self, features: dict[str, np.ndarray]) -> dict[str, str | None]:
        self.frame_index += 1
        for track_id, feature in features.items():
            previous = self.track_colors.get(track_id)
            smoothed = (
                feature
                if previous is None
                else self.feature_smoothing * feature
                + (1.0 - self.feature_smoothing) * previous.feature
            )
            self.track_colors[track_id] = _TrackColor(smoothed, self.frame_index)

        active = {
            track_id: state
            for track_id, state in self.track_colors.items()
            if self.frame_index - state.last_seen_frame <= self.memory_frames
        }
        if len(active) < self.min_tracks:
            return {track_id: None for track_id in features}

        matrix = np.vstack([state.feature for state in active.values()])
        new_centers = self._fit_two_clusters(matrix)
        if self.centers is not None:
            direct = np.linalg.norm(new_centers[0] - self.centers[0]) + np.linalg.norm(
                new_centers[1] - self.centers[1]
            )
            swapped = np.linalg.norm(new_centers[1] - self.centers[0]) + np.linalg.norm(
                new_centers[0] - self.centers[1]
            )
            if swapped < direct:
                new_centers = new_centers[::-1]
        self.centers = new_centers

        separation = float(np.linalg.norm(self.centers[0] - self.centers[1]))
        labels: dict[str, str | None] = {}
        for track_id, feature in features.items():
            distances = np.linalg.norm(self.centers - feature, axis=1)
            nearest = int(np.argmin(distances))
            if separation < 1e-6 or distances[nearest] > separation * self.outlier_ratio:
                labels[track_id] = None
                continue
            candidate = "team_a" if nearest == 0 else "team_b"
            self.votes[track_id][candidate] += 1
            labels[track_id] = self.votes[track_id].most_common(1)[0][0]
        return labels
