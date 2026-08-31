from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attack_analysis.annotations import weighted_kappa
from attack_analysis.benchmarks import build_footpass_role_benchmarks
from attack_analysis.footpass import (
    FootpassSegmentationConfig,
    decode_footpass_sequences_v2,
    footpass_summary,
    load_footpass_observed_actions_v2,
    load_footpass_events,
    normalize_footpass_position,
    segment_footpass_attacks,
    split_footpass_match_key,
)
from attack_analysis.models import SourceTier, TerminalType
from attack_analysis.ml import make_group_split, validate_no_target_leakage


class FootpassResearchTests(unittest.TestCase):
    def test_splits_period_from_stable_match_id(self) -> None:
        self.assertEqual(split_footpass_match_key("game_18_H2"), ("game_18", 2))
        with self.assertRaisesRegex(ValueError, "_H"):
            split_footpass_match_key("game_18")

    def test_normalizes_both_footpass_directions_to_attacking_right(self) -> None:
        self.assertAlmostEqual(normalize_footpass_position(0.8, 0.5, 0)[0], 84.0)
        self.assertAlmostEqual(normalize_footpass_position(0.2, 0.5, 1)[0], 84.0)

    def test_uses_trained_xg_for_tactical_shot(self) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest("FOOTPASS test dependencies are not installed")

        class FakeXGModel:
            source = "trained_xg:test:xg-v1"

            def predict(self, features: object) -> float:
                self.features = features
                return 0.37

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            playbyplay = root / "gt.json"
            tactical = root / "tactical.h5"
            playbyplay.write_text(
                json.dumps({"events": {"game_1_H1": [[25, 1, 9, 5]]}}),
                encoding="utf-8",
            )
            rows = np.zeros((22, 14), dtype=float)
            rows[:, 0] = 25
            rows[:, 1] = np.arange(22)
            rows[:11, 2] = 1
            rows[11:, 2] = 0
            rows[:, 3] = np.tile(np.arange(1, 12), 2)
            rows[:, 4] = 2
            rows[11, 4] = 1
            rows[:11, 5] = 0.2
            rows[11:, 5] = 0.02
            rows[:, 6] = 0.5
            rows[8, 13] = 5
            with h5py.File(tactical, "w") as handle:
                handle.create_dataset("game_1_H1", data=rows)
            fake_model = FakeXGModel()
            with patch("attack_analysis.footpass.load_trained_xg_model", return_value=fake_model):
                events = load_footpass_events(
                    playbyplay,
                    match_key="game_1_H1",
                    source="oracle",
                    tactical_h5=tactical,
                    xg_model_path=root / "model.joblib",
                )
        self.assertEqual(events[0].xg, 0.37)
        self.assertEqual(events[0].xg_source, "trained_xg:test:xg-v1")
        self.assertEqual(fake_model.features["foe_players"], 11)
        self.assertEqual(events[0].xg_inputs["foe_players"], 11)
        self.assertNotIn("goal", events[0].xg_inputs)
        self.assertEqual(events[0].xg_inputs["observed_players"], 22)

    def test_freeze_frame_counts_role_one_as_goalkeeper(self) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest("FOOTPASS test dependencies are not installed")
        from attack_analysis.footpass import footpass_freeze_frame

        rows = np.zeros((22, 14), dtype=float)
        rows[:, 0] = 25
        rows[:, 1] = np.arange(22)
        rows[:11, 2] = 1
        rows[11:, 2] = 0
        rows[:, 3] = np.tile(np.arange(1, 12), 2)
        rows[:, 4] = 2
        rows[11, 4] = 1
        rows[:11, 5] = 0.2
        rows[11:, 5] = 0.02
        rows[:, 6] = 0.5
        freeze = footpass_freeze_frame(
            rows,
            attacking_direction=1,
            shooter_x=21.0,
            shooter_y=34.0,
            timestamp=1.0,
        )
        self.assertEqual(freeze["foe_players"], 11)
        self.assertIsNotNone(freeze["x_goalkeeper"])
        self.assertEqual(sum(1 for item in freeze["players"] if item["is_goalkeeper"]), 1)

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

    def test_v2_video_only_does_not_invent_stable_team_or_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.json"
            path.write_text(
                json.dumps({"events": {"game_1_H1": [[25, 0, 9, 2, 0.8]]}}),
                encoding="utf-8",
            )
            actions = load_footpass_observed_actions_v2(
                path,
                match_key="game_1_H1",
                source="predicted",
            )
            decoded = decode_footpass_sequences_v2(
                path,
                match_key="game_1_H1",
                source="predicted",
            )
        self.assertEqual(actions[0].source_tier, SourceTier.VIDEO_ONLY)
        self.assertIsNone(actions[0].actor_team_id)
        self.assertIn("replay_status_unknown", actions[0].qualifiers)
        self.assertEqual(decoded.unassigned_action_ids, (actions[0].action_id,))
        self.assertFalse(decoded.attacks)

    def test_v2_oracle_adapter_preserves_bbox_replay_and_roster_team_across_halves(self) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError:
            self.skipTest("FOOTPASS test dependencies are not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            playbyplay = root / "gt.json"
            tactical = root / "tactical.h5"
            playbyplay.write_text(
                json.dumps(
                    {
                        "events": {
                            "game_1_H1": [
                                [0, 0, 9, 2, 1, 0],
                                [25, 1, 8, 6, 0, 0],
                                [50, 0, 9, 5, 1, 0],
                                [75, 0, 9, 2, 0, 1],
                            ],
                            "game_1_H2": [[0, 1, 9, 2, 1, 0]],
                        }
                    }
                ),
                encoding="utf-8",
            )

            def tactical_rows(period_two: bool) -> object:
                event_frames = (0,) if period_two else (0, 25, 50, 75)
                rows = np.zeros((len(event_frames) * 2, 14), dtype=float)
                for index, frame in enumerate(event_frames):
                    first = index * 2
                    rows[first, 0:7] = (
                        frame,
                        100,
                        1 if period_two else 0,
                        9,
                        12,
                        0.8,
                        0.5,
                    )
                    rows[first + 1, 0:7] = (
                        frame,
                        200,
                        0 if period_two else 1,
                        8,
                        3,
                        0.2,
                        0.5,
                    )
                    rows[first : first + 2, 9:13] = (10, 20, 30, 40)
                if not period_two:
                    rows[2, 13] = 6
                    rows[4, 13] = 5
                    rows[6:8, 9] = np.nan
                return rows

            with h5py.File(tactical, "w") as handle:
                handle.create_dataset("game_1_H1", data=tactical_rows(False))
                handle.create_dataset("game_1_H2", data=tactical_rows(True))

            first_half = load_footpass_observed_actions_v2(
                playbyplay,
                match_key="game_1_H1",
                source="oracle",
                tactical_h5=tactical,
            )
            second_half = load_footpass_observed_actions_v2(
                playbyplay,
                match_key="game_1_H2",
                source="oracle",
                tactical_h5=tactical,
            )
            decoded = decode_footpass_sequences_v2(
                playbyplay,
                match_key="game_1_H1",
                source="oracle",
                tactical_h5=tactical,
            )

        self.assertEqual(first_half[0].actor_team_id, second_half[0].actor_team_id)
        self.assertNotEqual(first_half[0].raw_left_to_right, second_half[0].raw_left_to_right)
        self.assertEqual(first_half[1].action_type, "header")
        self.assertFalse(first_half[1].bbox_visible)
        self.assertTrue(first_half[-1].replay)
        self.assertEqual(first_half[0].source_tier, SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL)
        self.assertEqual(len(decoded.attacks), 1)
        self.assertEqual(decoded.attacks[0].terminal.terminal_type, TerminalType.SHOT_ATTEMPT)
        self.assertIn(first_half[1].action_id, {item.action_id for item in decoded.possessions[0].actions})
        self.assertIn(first_half[-1].action_id, decoded.excluded_replay_action_ids)

    def test_v2_oracle_actions_require_tactical_association(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.json"
            path.write_text(
                json.dumps({"events": {"game_1_H1": [[25, 0, 9, 2, 1, 0]]}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a supported v2 source tier"):
                load_footpass_observed_actions_v2(
                    path,
                    match_key="game_1_H1",
                    source="oracle",
                )

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

    def test_decode_runs_neighborhood_terminal_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.json"
            path.write_text(
                json.dumps(
                    {
                        "events": {
                            "game_1_H1": [
                                [25, 0, 9, 2, 0.9],
                                [40, 0, 9, 3, 0.9],
                                [50, 0, 9, 6, 0.9],
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            decoded = decode_footpass_sequences_v2(
                path,
                match_key="game_1_H1",
                source="predicted",
            )
        self.assertFalse(decoded.attacks)
        self.assertTrue(decoded.unassigned_action_ids)

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
        summary = footpass_summary(events, attacks)
        self.assertEqual(summary["schema_version"], "PossessionCandidateV1")
        self.assertNotIn("attack_count", summary)

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
