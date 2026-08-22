import unittest
from pathlib import Path

from attack_analysis.observations import FrameObservation, TrackedObject
from attack_analysis.video import build_video_metadata


class ObservationTests(unittest.TestCase):
    def test_parses_frame_contract(self) -> None:
        frame = FrameObservation.from_dict(
            {
                "frame_index": 10,
                "timestamp_sec": 0.4,
                "calibration_confidence": 0.9,
                "objects": [
                    {
                        "track_id": "ball-1",
                        "object_type": "ball",
                        "confidence": 0.8,
                        "bbox": {"x1": 10, "y1": 20, "x2": 15, "y2": 25},
                        "pitch_x": 50,
                        "pitch_y": 34,
                    }
                ],
            }
        )
        self.assertEqual(frame.objects[0].track_id, "ball-1")
        self.assertEqual(frame.to_dict()["objects"][0]["pitch_x"], 50.0)

    def test_requires_pitch_coordinate_pair(self) -> None:
        with self.assertRaises(ValueError):
            TrackedObject.from_dict(
                {
                    "track_id": "player-1",
                    "object_type": "player",
                    "confidence": 0.9,
                    "bbox": {"x1": 10, "y1": 20, "x2": 30, "y2": 60},
                    "pitch_x": 50,
                }
            )

    def test_builds_video_metadata(self) -> None:
        metadata = build_video_metadata(Path("clip.mp4"), 1920, 1080, 25, 250)
        self.assertEqual(metadata.duration_sec, 10.0)
        self.assertEqual(metadata.frame_count, 250)

