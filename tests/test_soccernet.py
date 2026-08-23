import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.soccernet import analyze_gamestate_sequence


class SoccerNetAnalysisTests(unittest.TestCase):
    def test_analyzes_tracks_pitch_coverage_and_speed(self) -> None:
        payload = {
            "info": {"version": "1.3"},
            "images": [
                {"id": 1, "file_name": "img1/000001.jpg"},
                {"id": 2, "file_name": "img1/000002.jpg"},
            ],
            "annotations": [
                {
                    "image_id": 1,
                    "track_id": 7,
                    "role": "player",
                    "team": "left",
                    "jersey_number": 10,
                    "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0},
                },
                {
                    "image_id": 2,
                    "track_id": 7,
                    "role": "player",
                    "team": "left",
                    "jersey_number": 10,
                    "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 0.0},
                },
                {
                    "image_id": 2,
                    "track_id": 8,
                    "role": "ball",
                    "team": None,
                    "jersey_number": None,
                    "bbox_pitch": None,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            sequence = Path(temporary_directory) / "SNGS-TEST"
            sequence.mkdir()
            (sequence / "Labels-GameState.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            report = analyze_gamestate_sequence(sequence, fps=25.0)

            self.assertEqual(report["dataset_version"], "1.3")
            self.assertEqual(report["frames"], 2)
            self.assertEqual(report["unique_tracks"], 2)
            self.assertEqual(report["roles"], {"ball": 1, "player": 2})
            self.assertAlmostEqual(report["pitch_coordinate_coverage"], 2 / 3, places=4)
            self.assertEqual(report["speed_by_role"]["player"]["mean_mps"], 25.0)
            self.assertEqual(report["attack_assessment"]["status"], "insufficient_context")
            self.assertTrue((sequence / "analysis-report.json").exists())

    def test_requires_labels_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                analyze_gamestate_sequence(Path(temporary_directory))

    def test_provisionally_marks_progress_without_chance_as_ineffective(self) -> None:
        payload = {
            "info": {"version": "1.3", "frame_rate": 25, "action_class": "Foul"},
            "categories": [
                {"id": 1, "name": "player", "supercategory": "object"},
                {"id": 4, "name": "ball", "supercategory": "object"},
            ],
            "images": [
                {"image_id": str(frame), "file_name": f"{frame:06d}.jpg"}
                for frame in range(1, 4)
            ],
            "annotations": [],
        }
        for frame, ball_x in enumerate((50.0, 0.0, -10.0), start=1):
            payload["annotations"].extend(
                [
                    {
                        "image_id": str(frame),
                        "track_id": 1,
                        "category_id": 1,
                        "attributes": {"role": "player", "team": "right"},
                        "bbox_pitch": {
                            "x_bottom_middle": ball_x + 0.5,
                            "y_bottom_middle": 0.0,
                        },
                    },
                    {
                        "image_id": str(frame),
                        "track_id": 2,
                        "category_id": 4,
                        "attributes": {"role": "ball", "team": None},
                        "bbox_pitch": {
                            "x_bottom_middle": ball_x,
                            "y_bottom_middle": 0.0,
                        },
                    },
                ]
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            sequence = Path(temporary_directory) / "SNGS-TEST"
            sequence.mkdir()
            (sequence / "Labels-GameState.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            assessment = analyze_gamestate_sequence(sequence)["attack_assessment"]

            self.assertEqual(assessment["status"], "provisional_ineffective_chance_creation")
            self.assertFalse(assessment["effective"])
            self.assertEqual(assessment["attacking_team"], "right")
            self.assertFalse(assessment["reached_final_third"])
            self.assertEqual(assessment["estimated_sequence_xg"], 0.0)


if __name__ == "__main__":
    unittest.main()
