from __future__ import annotations

from pathlib import Path


BROWSER_CODECS = {"avc1", "h264", "H264"}


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


def ensure_browser_compatible_video(input_path: Path) -> Path:
    """Return an H.264 MP4, transcoding OpenCV's FMP4 output only when needed."""
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

    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV is required to create browser-compatible video") from error

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("annotated video has invalid FPS or dimensions")

    temporary_path = output_path.with_name(f"{output_path.stem}.partial.mp4")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "H.264 encoder is unavailable; install an FFmpeg/OpenH264-enabled video backend"
        )

    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written == 0 or not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise ValueError("browser-compatible video encoder produced no frames")
    temporary_path.replace(output_path)
    if video_codec(output_path) not in BROWSER_CODECS:
        raise ValueError("browser-compatible video does not contain an H.264 stream")
    return output_path
