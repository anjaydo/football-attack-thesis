from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import PITCH_LENGTH, PITCH_WIDTH, _optional_number
from .field_geometry import FieldGeometry


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
    track_id: str | None
    object_type: str
    confidence: float
    bbox: BoundingBox
    team: str | None = None
    pitch_x: float | None = None
    pitch_y: float | None = None
    anchor_image: tuple[float, float] | None = None
    keypoints: tuple[tuple[float, ...], ...] = ()
    speed: float | None = None
    speed_unit: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackedObject":
        observation = cls(
            track_id=(str(data["track_id"]) if data.get("track_id") is not None else None),
            object_type=str(data["object_type"]).lower(),
            confidence=float(data["confidence"]),
            bbox=BoundingBox.from_dict(data["bbox"]),
            team=str(data["team"]) if data.get("team") is not None else None,
            pitch_x=_optional_number(data, "pitch_x"),
            pitch_y=_optional_number(data, "pitch_y"),
            anchor_image=(
                tuple(float(value) for value in data["anchor_image"])
                if data.get("anchor_image") is not None
                else None
            ),
            keypoints=tuple(
                tuple(float(value) for value in keypoint)
                for keypoint in data.get("keypoints", [])
            ),
            speed=_optional_number(data, "speed"),
            speed_unit=(str(data["speed_unit"]) if data.get("speed_unit") else None),
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
        if self.anchor_image is not None and len(self.anchor_image) != 2:
            raise ValueError("anchor_image must contain x and y")
        for keypoint in self.keypoints:
            if len(keypoint) not in {2, 3, 5}:
                raise ValueError(
                    "each keypoint must contain x,y[,confidence[,pitch_x,pitch_y]]"
                )
            if len(keypoint) >= 3 and not 0.0 <= keypoint[2] <= 1.0:
                raise ValueError("keypoint confidence must be between 0 and 1")
        if self.speed is not None and self.speed < 0:
            raise ValueError("speed cannot be negative")
        if self.speed_unit is not None and self.speed_unit not in {"px/s", "km/h"}:
            raise ValueError("speed_unit must be px/s or km/h")
        if (self.speed is None) != (self.speed_unit is None):
            raise ValueError("speed and speed_unit must be provided together")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    timestamp_sec: float
    objects: tuple[TrackedObject, ...]
    calibration_confidence: float | None = None
    scene_cut: bool = False
    field_geometry: FieldGeometry | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameObservation":
        frame = cls(
            frame_index=int(data["frame_index"]),
            timestamp_sec=float(data["timestamp_sec"]),
            objects=tuple(TrackedObject.from_dict(item) for item in data.get("objects", [])),
            calibration_confidence=_optional_number(data, "calibration_confidence"),
            scene_cut=bool(data.get("scene_cut", False)),
            field_geometry=(
                FieldGeometry.from_dict(data["field_geometry"])
                if data.get("field_geometry") is not None
                else None
            ),
        )
        frame.validate()
        return frame

    def validate(self) -> None:
        if self.frame_index < 0 or self.timestamp_sec < 0:
            raise ValueError("frame index and timestamp must be non-negative")
        if self.calibration_confidence is not None and not 0.0 <= self.calibration_confidence <= 1.0:
            raise ValueError("calibration_confidence must be between 0 and 1")
        track_keys = [
            (item.object_type, item.track_id)
            for item in self.objects
            if item.track_id is not None
        ]
        if len(track_keys) != len(set(track_keys)):
            raise ValueError("track IDs must be unique per object type within a frame")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "objects": [item.to_dict() for item in self.objects],
            "scene_cut": self.scene_cut,
        }
        if self.calibration_confidence is not None:
            result["calibration_confidence"] = self.calibration_confidence
        if self.field_geometry is not None:
            result["field_geometry"] = self.field_geometry.to_dict()
        return result
