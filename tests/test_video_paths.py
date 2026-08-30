from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from attack_analysis.video_paths import PreviewClockProjector, VideoPathResolver


class VideoPathResolverTests(unittest.TestCase):
    def test_preview_and_processing_roots_split_by_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low = root / "low"
            high = root / "high"
            low.mkdir()
            high.mkdir()
            (low / "game_18.mp4").write_bytes(b"low")
            (high / "game_18.mp4").write_bytes(b"high")
            sync_path = root / "sync.json"
            sync_path.write_text(
                json.dumps(
                    {
                        "schema_version": "footpass-video-sync/1.0",
                        "matches": {
                            "game_18": {
                                "preview_offset_sec": 1.5,
                                "processing_offset_sec": 0.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous = {
                "FOOTPASS_VIDEOS_DIR": os.environ.get("FOOTPASS_VIDEOS_DIR"),
                "FOOTPASS_FULLHD_VIDEOS_DIR": os.environ.get("FOOTPASS_FULLHD_VIDEOS_DIR"),
                "FOOTPASS_VIDEO_SYNC_JSON": os.environ.get("FOOTPASS_VIDEO_SYNC_JSON"),
            }
            try:
                os.environ["FOOTPASS_VIDEOS_DIR"] = str(low)
                os.environ["FOOTPASS_FULLHD_VIDEOS_DIR"] = str(high)
                os.environ["FOOTPASS_VIDEO_SYNC_JSON"] = str(sync_path)
                resolver = VideoPathResolver.from_environment(root)
                self.assertTrue(resolver.dual_video_enabled("game_18_H1"))
                self.assertEqual(resolver.processing_path("game_18_H1"), low / "game_18.mp4")
                self.assertEqual(resolver.preview_path("game_18_H1"), high / "game_18.mp4")
                self.assertEqual(resolver.match_sec_to_preview_sec("game_18_H1", 10.0), 11.5)
                self.assertEqual(resolver.match_sec_to_processing_sec("game_18_H1", 10.0), 10.0)
                self.assertEqual(resolver.yolo_video_path("game_18_H1"), high / "game_18.mp4")
                self.assertEqual(
                    resolver.match_sec_to_preview_frame("game_18_H1", 10.0, 50.0),
                    575,
                )
                self.assertEqual(
                    resolver.preview_frame_to_match_frame("game_18_H1", 575, 50.0),
                    250,
                )
                self.assertEqual(resolver.preview_sec_to_match_sec("game_18_H1", 11.5), 10.0)
                clock = resolver.video_clock("game_18_H1", preview_fps=50.0)
                self.assertEqual(clock["role"], "fullhd")
                self.assertEqual(clock["preview_offset_sec"], 1.5)

                class _Inner:
                    def __init__(self) -> None:
                        self.seen: list[int] = []

                    def for_frame(self, frame_index: int) -> int:
                        self.seen.append(frame_index)
                        return frame_index

                    def confidence_for_frame(self, frame_index: int) -> float:
                        return float(frame_index)

                inner = _Inner()
                projector = PreviewClockProjector(
                    inner=inner,
                    match_key="game_18_H1",
                    resolver=resolver,
                    preview_fps=50.0,
                )
                self.assertEqual(projector.for_frame(575), 250)
                self.assertEqual(inner.seen, [250])
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
