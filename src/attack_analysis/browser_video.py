from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


BROWSER_CODECS = {"avc1", "h264", "H264"}
_FFMPEG_EXE: str | None | bool = False


def video_codec(path: Path) -> str:
    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required to inspect video codecs") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        value = int(capture.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
    finally:
        capture.release()


def _ffmpeg_executable() -> str | None:
    global _FFMPEG_EXE
    if _FFMPEG_EXE is not False:
        return _FFMPEG_EXE  # type: ignore[return-value]

    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG_EXE = found
        return found
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        _FFMPEG_EXE = None
        return None
    try:
        _FFMPEG_EXE = get_ffmpeg_exe()
    except Exception:
        _FFMPEG_EXE = None
    return _FFMPEG_EXE


def _transcode_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        return False
    temporary_path = output_path.with_name(f"{output_path.stem}.partial.mp4")
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        if temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
        return False
    temporary_path.replace(output_path)
    try:
        return video_codec(output_path) in BROWSER_CODECS
    except ValueError:
        return False


def _transcode_with_ffmpeg_pipe(input_path: Path, output_path: Path) -> bool:
    """Re-encode via raw BGR frames so OpenCV never touches OpenH264."""

    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        return False
    try:
        import cv2
    except ModuleNotFoundError:
        return False

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        return False
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        return False

    temporary_path = output_path.with_name(f"{output_path.stem}.partial.mp4")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    written = 0
    try:
        if process.stdin is None:
            return False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            process.stdin.write(frame.tobytes())
            written += 1
    except (BrokenPipeError, OSError, ValueError):
        written = 0
    finally:
        capture.release()
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        return_code = process.wait()

    if (
        return_code != 0
        or written == 0
        or not temporary_path.is_file()
        or temporary_path.stat().st_size == 0
    ):
        if temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
        return False
    temporary_path.replace(output_path)
    try:
        return video_codec(output_path) in BROWSER_CODECS
    except ValueError:
        return False


def sample_preview_frames(path: Path, *, max_frames: int = 8) -> tuple[Any, ...]:
    """Return BGR frames so Streamlit can still show YOLO overlay without H.264."""

    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required to preview annotated frames") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if frame_count <= 0:
            frames: list[Any] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                return ()
            step = max(1, len(frames) // max_frames)
            return tuple(frames[::step][:max_frames])
        indexes = [
            int(round(index * (frame_count - 1) / max(1, max_frames - 1)))
            for index in range(min(max_frames, frame_count))
        ]
        sampled: list[Any] = []
        for frame_index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                sampled.append(frame)
        return tuple(sampled)
    finally:
        capture.release()


def ensure_browser_compatible_video(input_path: Path) -> Path:
    """Prefer H.264 for Streamlit via FFmpeg; never use OpenCV OpenH264 writers."""

    if not input_path.is_file():
        raise FileNotFoundError(f"annotated video not found: {input_path}")
    if video_codec(input_path) in BROWSER_CODECS:
        return input_path

    output_path = input_path.with_name(f"{input_path.stem}-browser.mp4")
    if (
        output_path.is_file()
        and output_path.stat().st_mtime >= input_path.stat().st_mtime
        and video_codec(output_path) in BROWSER_CODECS
    ):
        return output_path

    if _transcode_with_ffmpeg(input_path, output_path):
        return output_path
    if _transcode_with_ffmpeg_pipe(input_path, output_path):
        return output_path
    return input_path
