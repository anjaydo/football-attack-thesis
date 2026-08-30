from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .footpass import split_footpass_match_key

DEFAULT_MATCH_FPS = 25.0
VIDEO_SYNC_SCHEMA = "footpass-video-sync/1.0"


@dataclass(frozen=True)
class MatchVideoSync:
    """Time/file mapping between FOOTPASS match clock and on-disk videos."""

    match_id: str
    preview_offset_sec: float = 0.0
    processing_offset_sec: float = 0.0
    preview_fps: float | None = None
    processing_fps: float = DEFAULT_MATCH_FPS
    preview_file: str | None = None
    processing_file: str | None = None

    @classmethod
    def from_dict(cls, match_id: str, payload: Mapping[str, object]) -> "MatchVideoSync":
        preview_fps = payload.get("preview_fps")
        processing_fps = payload.get("processing_fps", DEFAULT_MATCH_FPS)
        return cls(
            match_id=match_id,
            preview_offset_sec=float(payload.get("preview_offset_sec") or 0.0),
            processing_offset_sec=float(payload.get("processing_offset_sec") or 0.0),
            preview_fps=float(preview_fps) if preview_fps is not None else None,
            processing_fps=float(processing_fps or DEFAULT_MATCH_FPS),
            preview_file=str(payload["preview_file"]) if payload.get("preview_file") else None,
            processing_file=str(payload["processing_file"]) if payload.get("processing_file") else None,
        )


@dataclass(frozen=True)
class VideoPathResolver:
    processing_root: Path
    preview_root: Path | None
    sync_by_key: tuple[tuple[str, MatchVideoSync], ...]
    default_match_fps: float = DEFAULT_MATCH_FPS
    sync_path: Path | None = None

    @classmethod
    def from_environment(cls, project_root: Path) -> "VideoPathResolver":
        processing_root = Path(os.environ.get("FOOTPASS_VIDEOS_DIR", r"D:\videos"))
        preview_root = _configured_preview_root()
        sync_path = _configured_sync_path(project_root)
        sync_by_key: dict[str, MatchVideoSync] = {}
        default_match_fps = DEFAULT_MATCH_FPS
        if sync_path is not None and sync_path.is_file():
            payload = json.loads(sync_path.read_text(encoding="utf-8"))
            default_match_fps = float(payload.get("default_match_fps") or DEFAULT_MATCH_FPS)
            matches = payload.get("matches") or {}
            if isinstance(matches, Mapping):
                for match_id, item in matches.items():
                    if isinstance(item, Mapping):
                        sync_by_key[str(match_id)] = MatchVideoSync.from_dict(str(match_id), item)
        return cls(
            processing_root=processing_root,
            preview_root=preview_root,
            sync_by_key=tuple(sorted(sync_by_key.items())),
            default_match_fps=default_match_fps,
            sync_path=sync_path,
        )

    def _sync_lookup(self, match_key: str) -> MatchVideoSync | None:
        match_id, _period = split_footpass_match_key(match_key)
        indexed = dict(self.sync_by_key)
        if match_key in indexed:
            return indexed[match_key]
        if match_id in indexed:
            return indexed[match_id]
        return None

    def _video_filename(self, match_key: str, *, role: str) -> str:
        match_id, _period = split_footpass_match_key(match_key)
        sync = self._sync_lookup(match_key)
        if sync is not None:
            override = sync.preview_file if role == "preview" else sync.processing_file
            if override:
                return override
        return f"{match_id}.mp4"

    def processing_path(self, match_key: str) -> Path:
        return self.processing_root / self._video_filename(match_key, role="processing")

    def preview_path(self, match_key: str) -> Path:
        root = self.preview_root if self.preview_root is not None else self.processing_root
        return root / self._video_filename(match_key, role="preview")

    def dual_video_enabled(self, match_key: str) -> bool:
        if self.preview_root is None:
            return False
        preview = self.preview_path(match_key)
        processing = self.processing_path(match_key)
        if not preview.is_file() or not processing.is_file():
            return False
        try:
            return preview.resolve() != processing.resolve()
        except OSError:
            return str(preview) != str(processing)

    def match_sec_to_preview_sec(self, match_key: str, match_sec: float) -> float:
        sync = self._sync_lookup(match_key)
        offset = sync.preview_offset_sec if sync is not None else 0.0
        return max(0.0, float(match_sec) + offset)

    def match_sec_to_processing_sec(self, match_key: str, match_sec: float) -> float:
        sync = self._sync_lookup(match_key)
        offset = sync.processing_offset_sec if sync is not None else 0.0
        return max(0.0, float(match_sec) + offset)

    def preview_sec_to_match_sec(self, match_key: str, preview_sec: float) -> float:
        sync = self._sync_lookup(match_key)
        offset = sync.preview_offset_sec if sync is not None else 0.0
        return max(0.0, float(preview_sec) - offset)

    def match_sec_to_preview_frame(
        self,
        match_key: str,
        match_sec: float,
        preview_fps: float,
    ) -> int:
        if preview_fps <= 0:
            raise ValueError("preview_fps must be positive")
        return max(0, int(round(self.match_sec_to_preview_sec(match_key, match_sec) * preview_fps)))

    def preview_frame_to_match_frame(
        self,
        match_key: str,
        preview_frame: int,
        preview_fps: float,
    ) -> int:
        if preview_fps <= 0:
            raise ValueError("preview_fps must be positive")
        preview_sec = preview_frame / preview_fps
        sync = self._sync_lookup(match_key)
        offset = sync.preview_offset_sec if sync is not None else 0.0
        match_fps = sync.processing_fps if sync is not None else self.default_match_fps
        match_sec = max(0.0, preview_sec - offset)
        return max(0, int(round(match_sec * match_fps)))

    def yolo_video_path(self, match_key: str) -> Path:
        """Full HD overlay source when available; otherwise the processing file."""

        preview = self.preview_path(match_key)
        if preview.is_file():
            return preview
        return self.processing_path(match_key)

    def video_clock(self, match_key: str, *, preview_fps: float) -> dict[str, float | str]:
        sync = self._sync_lookup(match_key)
        return {
            "role": "fullhd" if self.preview_path(match_key).is_file() else "processing",
            "preview_fps": float(preview_fps),
            "match_fps": float(sync.processing_fps if sync is not None else self.default_match_fps),
            "preview_offset_sec": float(sync.preview_offset_sec if sync is not None else 0.0),
        }

    def describe_preview(self, match_key: str, *, start_sec: float, end_sec: float) -> str:
        preview_start = self.match_sec_to_preview_sec(match_key, start_sec)
        preview_end = self.match_sec_to_preview_sec(match_key, end_sec)
        preview_path = self.preview_path(match_key)
        if self.preview_path(match_key).is_file() and self.processing_path(match_key).is_file():
            try:
                dual = preview_path.resolve() != self.processing_path(match_key).resolve()
            except OSError:
                dual = True
            if dual:
                return (
                    f"{preview_path.name} (Full HD) · {preview_start:.1f}–{preview_end:.1f}s · "
                    f"YOLO overlay cũng xuất từ Full HD"
                )
        return f"{preview_path.name} · {preview_start:.1f}–{preview_end:.1f}s"


@dataclass(frozen=True)
class PreviewClockProjector:
    """Translate Full HD video frames into FOOTPASS match frames for homography lookup."""

    inner: object
    match_key: str
    resolver: VideoPathResolver
    preview_fps: float

    def for_frame(self, frame_index: int):
        return self.inner.for_frame(  # type: ignore[attr-defined]
            self.resolver.preview_frame_to_match_frame(
                self.match_key,
                frame_index,
                self.preview_fps,
            )
        )

    def confidence_for_frame(self, frame_index: int):
        return self.inner.confidence_for_frame(  # type: ignore[attr-defined]
            self.resolver.preview_frame_to_match_frame(
                self.match_key,
                frame_index,
                self.preview_fps,
            )
        )


def _configured_preview_root() -> Path | None:
    raw = os.environ.get("FOOTPASS_FULLHD_VIDEOS_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _configured_sync_path(project_root: Path) -> Path | None:
    raw = os.environ.get("FOOTPASS_VIDEO_SYNC_JSON", "").strip()
    if raw:
        return Path(raw)
    default = project_root / "data" / "footpass-video-sync.json"
    return default if default.is_file() else None
