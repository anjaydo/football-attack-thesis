from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from attack_analysis.browser_video import ensure_browser_compatible_video


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


if __name__ == "__main__":
    unittest.main()
