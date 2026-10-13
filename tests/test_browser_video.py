from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from attack_analysis.browser_video import (
    BROWSER_CODECS,
    _ffmpeg_executable,
    ensure_browser_compatible_video,
    extract_browser_clip,
    video_codec,
)


class BrowserVideoTests(unittest.TestCase):
    def test_falls_back_to_original_when_h264_is_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "annotated.mp4"
            raw.write_bytes(b"not-a-real-mp4")
            with (
                patch("attack_analysis.browser_video.video_codec", return_value="mp4v"),
                patch("attack_analysis.browser_video._transcode_with_ffmpeg", return_value=False),
                patch("attack_analysis.browser_video._transcode_with_ffmpeg_pipe", return_value=False),
            ):
                result = ensure_browser_compatible_video(raw)
        self.assertEqual(result, raw)

    def test_prefers_existing_h264_stream(self) -> None:
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "annotated.mp4"
            raw.write_bytes(b"h264")
            with patch("attack_analysis.browser_video.video_codec", return_value="avc1"):
                self.assertEqual(ensure_browser_compatible_video(raw), raw)

    def test_uses_ffmpeg_pipe_before_returning_original(self) -> None:
        with TemporaryDirectory() as directory:
            raw = Path(directory) / "annotated.mp4"
            raw.write_bytes(b"mp4v")
            browser = Path(directory) / "annotated-browser.mp4"
            with (
                patch("attack_analysis.browser_video.video_codec", side_effect=["mp4v", "avc1"]),
                patch("attack_analysis.browser_video._transcode_with_ffmpeg", return_value=False),
                patch(
                    "attack_analysis.browser_video._transcode_with_ffmpeg_pipe",
                    return_value=True,
                ) as pipe,
            ):
                result = ensure_browser_compatible_video(raw)
            pipe.assert_called_once()
            self.assertEqual(result, browser)

    def test_clip_keeps_only_the_requested_window(self) -> None:
        try:
            import cv2
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("OpenCV unavailable")
        if _ffmpeg_executable() is None:
            self.skipTest("FFmpeg unavailable")
        with TemporaryDirectory() as directory:
            source = Path(directory) / "match.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
            for index in range(125):
                writer.write(np.full((48, 64, 3), index * 2, dtype=np.uint8))
            writer.release()

            clip = extract_browser_clip(source, 1.0, 3.0, output_dir=Path(directory) / "clips")

            self.assertIsNotNone(clip)
            self.assertEqual(clip.parent, Path(directory) / "clips")
            capture = cv2.VideoCapture(str(clip))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            self.assertAlmostEqual(frames, 50, delta=2)
            self.assertIn(video_codec(clip), BROWSER_CODECS)
            self.assertEqual(extract_browser_clip(source, 1.0, 3.0, output_dir=Path(directory) / "clips"), clip)

    def test_clip_rejects_an_empty_window(self) -> None:
        with self.assertRaises(ValueError):
            extract_browser_clip(Path("missing.mp4"), 3.0, 3.0)


if __name__ == "__main__":
    unittest.main()
