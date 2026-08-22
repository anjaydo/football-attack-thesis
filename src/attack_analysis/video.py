from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_video_metadata(
    path: Path,
    width: float,
    height: float,
    fps: float,
    frame_count: float,
) -> VideoMetadata:
    if width <= 0 or height <= 0:
        raise ValueError("video width and height must be positive")
    if fps <= 0:
        raise ValueError("video FPS must be positive")
    if frame_count < 0:
        raise ValueError("video frame count cannot be negative")
    return VideoMetadata(
        path=str(path.resolve()),
        width=int(width),
        height=int(height),
        fps=round(float(fps), 4),
        frame_count=int(frame_count),
        duration_sec=round(float(frame_count) / float(fps), 3),
    )


def inspect_video(path: Path) -> VideoMetadata:
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenCV is required for video inspection. Install with: "
            "python -m pip install -e \".[video]\""
        ) from error

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        return build_video_metadata(
            path=path,
            width=capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            fps=capture.get(cv2.CAP_PROP_FPS),
            frame_count=capture.get(cv2.CAP_PROP_FRAME_COUNT),
        )
    finally:
        capture.release()
