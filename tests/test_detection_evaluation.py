import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.detection_evaluation import (
    benchmark_ball_confidence,
    evaluate_soccernet_observations,
)
from attack_analysis.observations import FrameObservation


class DetectionEvaluationTests(unittest.TestCase):
    def test_matches_people_and_ball_by_iou(self) -> None:
        labels = {
            "categories": [
                {"id": 1, "name": "player"},
                {"id": 3, "name": "referee"},
                {"id": 4, "name": "ball"},
            ],
            "images": [{"image_id": "10", "file_name": "000575.jpg"}],
            "annotations": [
                {
                    "image_id": "10",
                    "category_id": 1,
                    "supercategory": "object",
                    "bbox_image": {"x": 10, "y": 10, "w": 20, "h": 40},
                },
                {
                    "image_id": "10",
                    "category_id": 3,
                    "supercategory": "object",
                    "bbox_image": {"x": 40, "y": 10, "w": 20, "h": 40},
                },
                {
                    "image_id": "10",
                    "category_id": 4,
                    "supercategory": "object",
                    "bbox_image": {"x": 80, "y": 20, "w": 10, "h": 10},
                },
            ],
        }
        frame = FrameObservation.from_dict(
            {
                "frame_index": 0,
                "timestamp_sec": 0,
                "objects": [
                    {
                        "track_id": "1",
                        "object_type": "player",
                        "confidence": 0.9,
                        "bbox": {"x1": 10, "y1": 10, "x2": 30, "y2": 50},
                    },
                    {
                        "track_id": "2",
                        "object_type": "player",
                        "confidence": 0.8,
                        "bbox": {"x1": 40, "y1": 10, "x2": 60, "y2": 50},
                    },
                    {
                        "track_id": "1",
                        "object_type": "ball",
                        "confidence": 0.7,
                        "bbox": {"x1": 80, "y1": 20, "x2": 90, "y2": 30},
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            labels_path = Path(temporary_directory) / "labels.json"
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            report = evaluate_soccernet_observations([frame], labels_path, 575)

        self.assertEqual(report["person"]["true_positives"], 2)
        self.assertEqual(report["person"]["f1"], 1.0)
        self.assertEqual(report["person"]["ap50"], 1.0)
        self.assertEqual(report["person"]["map_50_95"], 1.0)
        self.assertEqual(report["ball"]["true_positives"], 1)
        self.assertEqual(report["ball"]["mean_center_error_px"], 0.0)

    def test_rejects_invalid_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            labels_path = Path(temporary_directory) / "labels.json"
            labels_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_soccernet_observations([], labels_path, iou_threshold=0)

    def test_ball_benchmark_selects_threshold_and_enforces_gate(self) -> None:
        labels = {
            "categories": [{"id": 4, "name": "ball"}],
            "images": [{"image_id": "10", "file_name": "000001.jpg"}],
            "annotations": [
                {
                    "image_id": "10",
                    "category_id": 4,
                    "supercategory": "object",
                    "bbox_image": {"x": 80, "y": 20, "w": 10, "h": 10},
                }
            ],
        }
        frame = FrameObservation.from_dict(
            {
                "frame_index": 0,
                "timestamp_sec": 0,
                "objects": [
                    {
                        "track_id": None,
                        "object_type": "ball",
                        "confidence": 0.9,
                        "bbox": {"x1": 80, "y1": 20, "x2": 90, "y2": 30},
                    },
                    {
                        "track_id": None,
                        "object_type": "ball",
                        "confidence": 0.2,
                        "bbox": {"x1": 10, "y1": 10, "x2": 20, "y2": 20},
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            labels_path = Path(temporary_directory) / "labels.json"
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            report = benchmark_ball_confidence(
                [frame], labels_path, thresholds=(0.1, 0.5)
            )

        self.assertTrue(report["accepted"])
        self.assertEqual(report["best"]["confidence"], 0.5)
        self.assertEqual(report["best"]["f1"], 1.0)

    def test_ball_benchmark_rejects_low_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            labels_path = Path(temporary_directory) / "labels.json"
            labels_path.write_text(
                json.dumps(
                    {
                        "categories": [{"id": 4, "name": "ball"}],
                        "images": [{"image_id": "1", "file_name": "000001.jpg"}],
                        "annotations": [
                            {
                                "image_id": "1",
                                "category_id": 4,
                                "bbox_image": {"x": 0, "y": 0, "w": 5, "h": 5},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = benchmark_ball_confidence([], labels_path, thresholds=(0.1,))

        self.assertFalse(report["accepted"])
        self.assertIn("ball_recall=0.0000 is below 0.7500", report["failures"])


if __name__ == "__main__":
    unittest.main()
