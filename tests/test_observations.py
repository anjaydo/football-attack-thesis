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

    def test_parses_vision_runner_contract(self) -> None:
        frame = FrameObservation.from_dict(
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "objects": [
                    {
                        "track_id": "1",
                        "object_type": "player",
                        "confidence": 0.9,
                        "bbox": {"x1": 10, "y1": 20, "x2": 30, "y2": 60},
                        "anchor_image": [20, 60],
                        "keypoints": [[20, 30, 0.8]],
                        "speed": 12.5,
                        "speed_unit": "px/s",
                    },
                    {
                        "track_id": "1",
                        "object_type": "ball",
                        "confidence": 0.7,
                        "bbox": {"x1": 40, "y1": 50, "x2": 45, "y2": 55},
                        "anchor_image": [42.5, 52.5],
                        "speed": 20,
                        "speed_unit": "px/s",
                    },
                ],
            }
        )
        self.assertEqual(frame.objects[0].speed, 12.5)
        self.assertEqual(frame.objects[0].keypoints[0][2], 0.8)
        self.assertEqual(frame.objects[1].track_id, "1")

    def test_parses_homography_projected_keypoints(self) -> None:
        frame = FrameObservation.from_dict(
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "objects": [
                    {
                        "track_id": "1",
                        "object_type": "player",
                        "confidence": 0.9,
                        "bbox": {"x1": 10, "y1": 20, "x2": 30, "y2": 60},
                        "keypoints": [[20, 30, 0.8, 90.1, 34.2]],
                    }
                ],
            }
        )
        self.assertEqual(len(frame.objects[0].keypoints[0]), 5)
        self.assertAlmostEqual(frame.objects[0].keypoints[0][3], 90.1)

    def test_allows_missing_track_id(self) -> None:
        frame = FrameObservation.from_dict(
            {
                "frame_index": 0,
                "timestamp_sec": 0,
                "objects": [
                    {
                        "track_id": None,
                        "object_type": "ball",
                        "confidence": 0.5,
                        "bbox": {"x1": 1, "y1": 1, "x2": 2, "y2": 2},
                    }
                ],
            }
        )
        self.assertIsNone(frame.objects[0].track_id)

    def test_builds_video_metadata(self) -> None:
        metadata = build_video_metadata(Path("clip.mp4"), 1920, 1080, 25, 250)
        self.assertEqual(metadata.duration_sec, 10.0)
        self.assertEqual(metadata.frame_count, 250)
