from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.annotations import weighted_kappa
from attack_analysis.benchmarks import build_footpass_role_benchmarks
from attack_analysis.footpass import (
    FootpassSegmentationConfig,
    load_footpass_events,
    segment_footpass_attacks,
)
from attack_analysis.ml import make_group_split, validate_no_target_leakage


class FootpassResearchTests(unittest.TestCase):
    def test_builds_anonymous_frozen_role_benchmarks(self) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest("FOOTPASS test dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            h5_path = Path(directory) / "tactical.h5"
            output_path = Path(directory) / "benchmarks.jsonl"
            rows = np.zeros((3, 14), dtype=float)
            rows[:, 0] = (0, 9000, 18000)
            rows[:, 1] = 7
            rows[:, 2] = 1
            rows[:, 4] = 2
            rows[:, 5] = (0.2, 0.6, 0.85)
            rows[:, 6] = 0.5
            rows[:, 13] = (2, 2, 5)
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("game_1_H1", data=rows)
            records = build_footpass_role_benchmarks(
                h5_path, output_path, snapshot_date="2026-01-01"
            )
        self.assertTrue(records)
        self.assertTrue(all(record.player_name.startswith("anonymous:") for record in records))
        self.assertEqual(records[0].snapshot_date, "2026-01-01")

    def test_loads_predicted_events_without_tactical_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.json"
            path.write_text(
                json.dumps({"events": {"game_1_H1": [[25, 0, 9, 2, 0.8], [50, 0, 9, 5, 0.7]]}}),
                encoding="utf-8",
            )
            events = load_footpass_events(
                path, match_key="game_1_H1", source="predicted"
            )
        self.assertEqual([item.event_type for item in events], ["pass", "shot"])
        self.assertEqual(events[0].confidence, 0.8)
        self.assertIsNone(events[1].xg)

    def test_prediction_confidence_filter_and_model_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.json"
            path.write_text(
                json.dumps(
                    {"events": {"game_1_H1": [[25, 0, 9, 2, 0.59], [50, 0, 9, 5, 0.8]]}}
                ),
                encoding="utf-8",
            )
            events = load_footpass_events(
                path,
                match_key="game_1_H1",
                source="predicted",
                min_confidence=0.6,
                model_name="DST",
            )
        self.assertEqual([item.event_type for item in events], ["shot"])
        self.assertEqual(events[0].source, "footpass_predicted_dst")

    def test_rejects_invalid_prediction_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.json"
            path.write_text(json.dumps({"events": {"game_1_H1": []}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "min_confidence"):
                load_footpass_events(
                    path, match_key="game_1_H1", source="predicted", min_confidence=1.1
                )

    def test_segments_team_change_shot_and_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.json"
            path.write_text(
                json.dumps(
                    {
                        "events": {
                            "game_1_H1": [
                                [0, 0, 9, 1],
                                [25, 0, 9, 2],
                                [50, 1, 8, 1],
                                [75, 1, 8, 5],
                                [250, 0, 9, 1],
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            events = load_footpass_events(path, match_key="game_1_H1", source="oracle")
        attacks = segment_footpass_attacks(
            events,
            match_id="game_1_H1",
            config=FootpassSegmentationConfig(max_gap_sec=4.0),
        )
        self.assertEqual(len(attacks), 3)
        self.assertEqual(attacks[0].termination_reason, "team_change")
        self.assertEqual(attacks[1].termination_reason, "shot")

    def test_group_split_has_no_match_overlap(self) -> None:
        split = make_group_split((f"game-{index}" for index in range(10)))
        self.assertFalse(set(split.train_groups) & set(split.validation_groups))
        self.assertFalse(set(split.train_groups) & set(split.test_groups))
        self.assertFalse(set(split.validation_groups) & set(split.test_groups))

    def test_target_leakage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target leakage"):
            validate_no_target_leakage(("distance", "goal"), "goal")

    def test_weighted_kappa(self) -> None:
        self.assertEqual(weighted_kappa((1, 2, 3, 4, 5), (1, 2, 3, 4, 5)), 1.0)
        self.assertLessEqual(weighted_kappa((1, 1, 2, 2), (4, 5, 4, 5)), 0.0)


if __name__ == "__main__":
    unittest.main()
