import unittest

from attack_analysis.observations import FrameObservation
from attack_analysis.quality import analyze_observation_quality


def make_frame(
    frame_index: int,
    include_ball: bool = True,
    include_geometry: bool = False,
) -> FrameObservation:
    keypoints = [[10 + index, 20 + index, 0.9] for index in range(17)]
    objects = [
        {
            "track_id": str(player_id),
            "object_type": "player",
            "confidence": 0.9,
            "bbox": {
                "x1": 10 * player_id,
                "y1": 20,
                "x2": 10 * player_id + 5,
                "y2": 40,
            },
            "anchor_image": [10 * player_id + 2.5, 40],
            "keypoints": keypoints,
            "speed": 10,
            "speed_unit": "px/s",
        }
        for player_id in range(1, 7)
    ]
    if include_ball:
        objects.append(
            {
                "track_id": "1",
                "object_type": "ball",
                "confidence": 0.8,
                "bbox": {"x1": 80, "y1": 30, "x2": 84, "y2": 34},
                "anchor_image": [82, 32],
                "speed": 20,
                "speed_unit": "px/s",
            }
        )
    payload = {
        "frame_index": frame_index,
        "timestamp_sec": frame_index / 25,
        "objects": objects,
    }
    if include_geometry:
        payload["field_geometry"] = {
            "method": "test",
            "frame_width": 100,
            "frame_height": 50,
            "pitch_mask_coverage": 0.8,
            "line_mask_coverage": 0.02,
            "segments": [
                {
                    "x1": 0,
                    "y1": 25,
                    "x2": 100,
                    "y2": 25,
                    "length_px": 100,
                    "angle_deg": 0,
                    "confidence": 1,
                }
            ],
            "intersections": [],
        }
    return FrameObservation.from_dict(payload)


class QualityReportTests(unittest.TestCase):
    def test_good_observations_pass(self) -> None:
        report = analyze_observation_quality(make_frame(index) for index in range(5))
        self.assertEqual(report.status, "pass")
        self.assertGreaterEqual(report.quality_score, 90)
        self.assertEqual(report.frames_with_ball_ratio, 1.0)
        self.assertEqual(report.player_tracking_coverage, 1.0)
        self.assertEqual(report.keypoint_coverage, 1.0)

    def test_missing_ball_fails(self) -> None:
        report = analyze_observation_quality(
            make_frame(index, include_ball=False) for index in range(5)
        )
        self.assertEqual(report.status, "fail")
        self.assertIn("no_ball_detections", report.warnings)

    def test_reports_field_geometry_coverage(self) -> None:
        report = analyze_observation_quality(
            [make_frame(0, include_geometry=True), make_frame(1)]
        )
        self.assertEqual(report.field_geometry_frame_coverage, 0.5)
        self.assertEqual(report.average_field_line_segments, 1.0)
        self.assertEqual(report.average_field_intersections, 0.0)

    def test_rejects_duplicate_frame_indices(self) -> None:
        with self.assertRaises(ValueError):
            analyze_observation_quality([make_frame(0), make_frame(0)])
