from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import PITCH_LENGTH, PITCH_WIDTH, _optional_number


OBJECT_TYPES = {"player", "goalkeeper", "referee", "ball", "other"}


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        box = cls(*(float(data[key]) for key in ("x1", "y1", "x2", "y2")))
        box.validate()
        return box

    def validate(self) -> None:
        if self.x1 < 0 or self.y1 < 0 or self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must be positive with x2>x1 and y2>y1")


@dataclass(frozen=True)
class TrackedObject:
    track_id: str
    object_type: str
    confidence: float
    bbox: BoundingBox
    team: str | None = None
    pitch_x: float | None = None
    pitch_y: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackedObject":
        observation = cls(
            track_id=str(data["track_id"]),
            object_type=str(data["object_type"]).lower(),
            confidence=float(data["confidence"]),
            bbox=BoundingBox.from_dict(data["bbox"]),
            team=str(data["team"]) if data.get("team") is not None else None,
            pitch_x=_optional_number(data, "pitch_x"),
            pitch_y=_optional_number(data, "pitch_y"),
        )
        observation.validate()
        return observation

    def validate(self) -> None:
        if self.object_type not in OBJECT_TYPES:
            raise ValueError(f"unsupported object_type: {self.object_type}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.pitch_x is not None and not 0.0 <= self.pitch_x <= PITCH_LENGTH:
            raise ValueError("pitch_x must be on the pitch")
        if self.pitch_y is not None and not 0.0 <= self.pitch_y <= PITCH_WIDTH:
            raise ValueError("pitch_y must be on the pitch")
        if (self.pitch_x is None) != (self.pitch_y is None):
            raise ValueError("pitch_x and pitch_y must be provided together")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    timestamp_sec: float
    objects: tuple[TrackedObject, ...]
    calibration_confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameObservation":
        frame = cls(
            frame_index=int(data["frame_index"]),
            timestamp_sec=float(data["timestamp_sec"]),
            objects=tuple(TrackedObject.from_dict(item) for item in data.get("objects", [])),
            calibration_confidence=_optional_number(data, "calibration_confidence"),
        )
        frame.validate()
        return frame

    def validate(self) -> None:
        if self.frame_index < 0 or self.timestamp_sec < 0:
            raise ValueError("frame index and timestamp must be non-negative")
        if self.calibration_confidence is not None and not 0.0 <= self.calibration_confidence <= 1.0:
            raise ValueError("calibration_confidence must be between 0 and 1")
        track_ids = [item.track_id for item in self.objects]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track_id values must be unique within a frame")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "objects": [item.to_dict() for item in self.objects],
        }
        if self.calibration_confidence is not None:
            result["calibration_confidence"] = self.calibration_confidence
        return result

