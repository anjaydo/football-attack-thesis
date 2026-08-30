from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.xg_data import (
    export_statsbomb_xg_dataset,
    export_statsbomb_xg_v2_dataset,
    extract_statsbomb_shot_features,
    extract_statsbomb_shot_features_v2,
)


def _shot_event() -> dict[str, object]:
    return {
        "id": "shot-1",
        "type": {"name": "Shot"},
        "location": [108.0, 40.0],
        "minute": 12,
        "second": 34,
        "timestamp": "00:12:34.500",
        "under_pressure": True,
        "play_pattern": {"name": "Regular Play"},
        "shot": {
            "type": {"name": "Open Play"},
            "outcome": {"name": "Goal"},
            "statsbomb_xg": 0.42,
        },
    }


class XGDataTests(unittest.TestCase):
    def test_v2_semantics_keep_unknown_body_part_and_missing_visibility(self) -> None:
        event = _shot_event()
        event["shot"] = dict(event["shot"])
        row = extract_statsbomb_shot_features_v2(event, None, match_id="123")
        self.assertIsNone(row["body_part"])
        self.assertIsNone(row["goalkeeper_visible"])
        self.assertIsNone(row["observed_players"])
        self.assertEqual(row["play_pattern"], "Regular Play")

    def test_extracts_video_reproducible_features_without_target_leakage(self) -> None:
        row = extract_statsbomb_shot_features(
            _shot_event(),
            (
                {"teammate": False, "keeper": True, "location": [119.0, 40.0]},
                {"teammate": False, "keeper": False, "location": [109.0, 40.0]},
                {"teammate": True, "actor": False, "location": [100.0, 35.0]},
                {"teammate": True, "actor": True, "location": [108.0, 40.0]},
            ),
            match_id="123",
        )
        self.assertEqual(row["match_id"], "123")
        self.assertAlmostEqual(row["x_shooter"], 94.5)
        self.assertAlmostEqual(row["y_shooter"], 34.0)
        self.assertEqual(row["foe_players"], 2)
        self.assertEqual(row["ally_players"], 1)
        self.assertEqual(row["defending_players"], 1)
        self.assertEqual(row["under_pressure"], 1)
        self.assertEqual(row["goal"], 1)
        self.assertEqual(row["statsbomb_xg"], 0.42)

    def test_exports_only_shots_with_360_freeze_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events").mkdir()
            (root / "three-sixty").mkdir()
            (root / "events" / "123.json").write_text(
                json.dumps([_shot_event(), {"id": "pass-1", "type": {"name": "Pass"}}]),
                encoding="utf-8",
            )
            (root / "three-sixty" / "123.json").write_text(
                json.dumps(
                    [
                        {
                            "event_uuid": "shot-1",
                            "freeze_frame": [
                                {"teammate": False, "keeper": True, "location": [119, 40]}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "shots.csv"
            report = export_statsbomb_xg_dataset(root, output)
            contents = output.read_text(encoding="utf-8")
        self.assertEqual(report["matches"], 1)
        self.assertEqual(report["shots"], 1)
        self.assertEqual(report["goals"], 1)
        self.assertIn("match_id,event_id,x_shooter", contents)

    def test_v2_export_has_numeric_missingness_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events").mkdir()
            (root / "three-sixty").mkdir()
            (root / "events" / "123.json").write_text(
                json.dumps([_shot_event()]), encoding="utf-8"
            )
            output = root / "shots-v2.csv"
            report = export_statsbomb_xg_v2_dataset(root, output)
            contents = output.read_text(encoding="utf-8")
        self.assertEqual(report["schema_version"], "xg-features-v2")
        self.assertEqual(report["shots_with_missing_visibility"], 1)
        self.assertIn("body_part_unknown", contents)
        self.assertIn("observed_players_missing", contents)


if __name__ == "__main__":
    unittest.main()
