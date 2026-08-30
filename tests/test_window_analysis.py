import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.models import AttackSequence, Event
from attack_analysis.provenance import MANIFEST_VERSION, StaleArtifactError
from attack_analysis.window_analysis import persist_footpass_window_stream


class WindowAnalysisTests(unittest.TestCase):
    def test_persists_legacy_evidence_but_withholds_attack_and_xg_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text('{"upstream": true}', encoding="utf-8")
            sequence = AttackSequence(
                attack_id="game_1_H1:attack-0001",
                match_id="game_1_H1",
                team="0",
                start_sec=1.0,
                end_sec=2.0,
                source="footpass_oracle",
                events=(
                    Event(
                        timestamp=2.0,
                        event_type="shot",
                        team="0",
                        x=90.0,
                        y=34.0,
                        xg=0.31,
                        xg_source="trained_xg:test:xg-v1",
                        xg_inputs={"x_shooter": 90.0, "y_shooter": 34.0},
                        source="footpass_oracle",
                        evidence_id="event:shot:1",
                    ),
                ),
            )
            manifest = {
                "run_directory": str(root),
                "vision": {"start_frame": 25, "end_frame": 50},
                "calibration": {"status": "pass"},
                "outputs": {"manifest": str(manifest_path)},
            }

            result = persist_footpass_window_stream(
                manifest,
                sequence,
                start_sec=1.0,
                end_sec=2.5,
            )

            self.assertEqual(
                result["structured_event_stream"]["status"], "legacy_unvalidated"
            )
            self.assertEqual(result["provenance_status"], "legacy_unvalidated")
            self.assertEqual(result["analysis_source_tier"], "legacy_unvalidated")
            event_stream = json.loads(
                (root / "structured-event-stream.json").read_text(encoding="utf-8")
            )
            self.assertEqual(event_stream["event_count"], 1)
            self.assertEqual(event_stream["object_type"], "PossessionCandidateV1")
            self.assertEqual(event_stream["xg_records"], [])
            self.assertEqual(event_stream["legacy_xg_evidence_count"], 1)
            self.assertEqual(event_stream["xg_status"], "withheld_legacy_unvalidated")
            analysis = json.loads(
                (root / "selected-possession-candidate-v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIsNone(analysis["attack_sequence_v2"])
            self.assertIsNone(analysis["possession_assessment"])
            self.assertIsNone(analysis["deterministic_baseline"])
            self.assertIsNone(analysis["xg_v2"])
            self.assertIsNone(analysis["effectiveness_verdict"])
            self.assertFalse(event_stream["quality_gates"]["attack_qualified"])
            self.assertFalse(event_stream["quality_gates"]["v2_contract"])
            self.assertFalse((root / "selected-attack-analysis.json").exists())
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                {"upstream": True},
            )
            derived_manifest = Path(result["outputs"]["manifest"])
            self.assertEqual(derived_manifest.name, "possession-candidate-v1-manifest.json")
            self.assertTrue(derived_manifest.is_file())
            self.assertEqual(
                result["outputs"]["upstream_manifest"], str(manifest_path)
            )

    def test_normal_possession_does_not_reach_effectiveness_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = AttackSequence(
                attack_id="game_1_H1:attack-0002",
                match_id="game_1_H1",
                team="0",
                start_sec=1.0,
                end_sec=5.0,
                source="footpass_oracle",
                events=(
                    Event(1.0, "pass", "0", x=30.0, y=20.0, source="footpass_oracle"),
                    Event(5.0, "pass", "0", x=34.0, y=48.0, source="footpass_oracle"),
                ),
            )
            result = persist_footpass_window_stream(
                {
                    "run_directory": str(root),
                    "calibration": {"status": "pass"},
                    "outputs": {},
                },
                sequence,
                start_sec=1.0,
                end_sec=5.0,
            )
            self.assertEqual(
                result["structured_event_stream"]["status"], "legacy_unvalidated"
            )
            self.assertFalse(
                result["structured_event_stream"]["quality_gates"]["attack_qualified"]
            )

    def test_current_vision_manifest_does_not_validate_legacy_sequence_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = persist_footpass_window_stream(
                {
                    "manifest_version": MANIFEST_VERSION,
                    "provenance_status": "current",
                    "source_tier": "oracle_actions_oracle_tactical",
                    "provenance": {"artifact_identity_sha256": "vision-artifact"},
                    "run_directory": str(root),
                    "calibration": {"status": "pass"},
                    "outputs": {},
                },
                AttackSequence(
                    attack_id="legacy-shot",
                    team="0",
                    start_sec=1.0,
                    end_sec=2.0,
                    source="footpass_oracle",
                    events=(Event(2.0, "shot", "0", source="footpass_oracle"),),
                ),
                start_sec=0.5,
                end_sec=2.5,
            )

        self.assertEqual(result["upstream_provenance_status"], "current")
        self.assertEqual(result["provenance_status"], "legacy_unvalidated")
        self.assertEqual(result["validation_status"], "legacy_unvalidated")
        self.assertEqual(result["upstream_source_tier"], "oracle_actions_oracle_tactical")
        self.assertEqual(result["analysis_source_tier"], "legacy_unvalidated")
        self.assertFalse(
            result["structured_event_stream"]["quality_gates"]["attack_qualified"]
        )

    def test_rejects_invalid_window(self) -> None:
        with self.assertRaises(ValueError):
            persist_footpass_window_stream(
                {"run_directory": "."},
                AttackSequence(
                    attack_id="a",
                    team="0",
                    start_sec=0,
                    end_sec=0,
                    events=(Event(0, "pass", "0"),),
                ),
                start_sec=2,
                end_sec=1,
            )

    def test_left_censored_window_uses_only_filtered_events_and_emits_no_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = AttackSequence(
                attack_id="game_1_H1:attack-0003",
                match_id="game_1_H1",
                team="0",
                start_sec=1.0,
                end_sec=4.0,
                source="footpass_oracle",
                events=(
                    Event(1.0, "pass", "0", x=30.0, y=20.0, source="footpass_oracle"),
                    Event(
                        4.0,
                        "shot",
                        "0",
                        x=91.0,
                        y=34.0,
                        xg=0.2,
                        xg_source="trained_xg:test:xg-v1",
                        xg_inputs={"x_shooter": 91.0},
                        source="footpass_oracle",
                    ),
                ),
            )
            result = persist_footpass_window_stream(
                {
                    "run_directory": str(root),
                    "calibration": {"status": "pass"},
                    "outputs": {},
                },
                sequence,
                start_sec=2.0,
                end_sec=4.5,
            )

            stream = json.loads(
                (root / "structured-event-stream.json").read_text(encoding="utf-8")
            )
            analysis = json.loads(
                (root / "selected-possession-candidate-v1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["structured_event_stream"]["status"], "left_censored")
        self.assertEqual(stream["event_count"], 1)
        self.assertEqual(len(analysis["possession_candidate_v1"]["events"]), 1)
        self.assertIsNone(stream["quality_gates"]["attack_qualified"])
        self.assertIsNone(stream["quality_gates"]["trained_xg"])
        self.assertEqual(stream["xg_records"], [])
        self.assertIsNone(analysis["possession_assessment"])
        self.assertIsNone(analysis["deterministic_baseline"])
        self.assertEqual(analysis["verdict_status"], "withheld_legacy_unvalidated")

    def test_right_censored_window_does_not_borrow_terminal_from_outside_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = AttackSequence(
                attack_id="game_1_H1:attack-right-censored",
                team="0",
                start_sec=1.0,
                end_sec=4.0,
                termination_reason="shot",
                source="footpass_oracle",
                events=(
                    Event(1.0, "pass", "0", x=30.0, y=20.0, source="footpass_oracle"),
                    Event(4.0, "shot", "0", x=90.0, y=34.0, source="footpass_oracle"),
                ),
            )
            result = persist_footpass_window_stream(
                {
                    "run_directory": str(root),
                    "calibration": {"status": "pass"},
                    "outputs": {},
                },
                sequence,
                start_sec=0.5,
                end_sec=2.0,
            )
            stream = json.loads(
                (root / "structured-event-stream.json").read_text(encoding="utf-8")
            )
            analysis = json.loads(
                (root / "selected-possession-candidate-v1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result["structured_event_stream"]["status"], "right_censored")
        self.assertIsNone(stream["terminal"]["type"])
        self.assertIsNone(stream["quality_gates"]["attack_qualified"])
        self.assertNotIn("termination_reason", analysis["possession_candidate_v1"])
        self.assertIsNone(analysis["deterministic_baseline"])

    def test_penetration_without_allowlisted_terminal_is_not_an_attack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = AttackSequence(
                attack_id="game_1_H1:attack-0004",
                team="0",
                start_sec=1.0,
                end_sec=3.0,
                source="footpass_oracle",
                events=(
                    Event(1.0, "drive", "0", x=40.0, y=30.0, source="footpass_oracle"),
                    Event(3.0, "cross", "0", x=95.0, y=34.0, source="footpass_oracle"),
                ),
            )
            result = persist_footpass_window_stream(
                {
                    "run_directory": str(root),
                    "calibration": {"status": "pass"},
                    "outputs": {},
                },
                sequence,
                start_sec=1.0,
                end_sec=3.0,
            )
            analysis = json.loads(
                (root / "selected-possession-candidate-v1.json").read_text(
                    encoding="utf-8"
                )
            )

        gates = result["structured_event_stream"]["quality_gates"]
        self.assertFalse(gates["allowlisted_terminal"])
        self.assertFalse(gates["attack_qualified"])
        self.assertEqual(
            result["structured_event_stream"]["status"], "legacy_unvalidated"
        )
        self.assertIsNone(analysis["deterministic_baseline"])

    def test_legacy_forced_goal_line_out_cannot_supply_v2_terminal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = AttackSequence(
                attack_id="game_1_H1:attack-0005",
                team="0",
                start_sec=1.0,
                end_sec=3.0,
                source="footpass_oracle",
                events=(
                    Event(1.0, "cross", "0", x=88.0, y=10.0, source="footpass_oracle"),
                    Event(
                        3.0,
                        "forced_goal_line_out",
                        "0",
                        x=104.0,
                        y=5.0,
                        source="footpass_oracle",
                    ),
                ),
            )
            result = persist_footpass_window_stream(
                {
                    "run_directory": str(root),
                    "calibration": {"status": "pass"},
                    "outputs": {},
                },
                sequence,
                start_sec=0.5,
                end_sec=3.5,
            )
            stream = json.loads(
                (root / "structured-event-stream.json").read_text(encoding="utf-8")
            )

        gates = result["structured_event_stream"]["quality_gates"]
        self.assertEqual(
            result["structured_event_stream"]["status"], "legacy_unvalidated"
        )
        self.assertFalse(gates["allowlisted_terminal"])
        self.assertFalse(gates["terminal_evidence_complete"])
        self.assertIsNone(gates["trained_xg"])
        self.assertFalse(gates["attack_qualified"])
        self.assertEqual(stream["terminal"]["type"], "forced_goal_line_out")
        self.assertEqual(
            set(stream["terminal"]["missing_fields"]),
            {"beneficiary_team_id", "last_touch_team_id", "cause"},
        )

    def test_generic_ball_out_does_not_imply_forced_goal_line_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = AttackSequence(
                attack_id="game_1_H1:attack-0006",
                team="0",
                start_sec=1.0,
                end_sec=2.0,
                source="footpass_oracle",
                events=(
                    Event(1.0, "cross", "0", x=90.0, y=8.0, source="footpass_oracle"),
                    Event(
                        2.0,
                        "ball_out",
                        "0",
                        x=105.0,
                        y=3.0,
                        outcome="forced_corner",
                        source="footpass_oracle",
                    ),
                ),
            )
            result = persist_footpass_window_stream(
                {
                    "run_directory": str(root),
                    "calibration": {"status": "pass"},
                    "outputs": {},
                },
                sequence,
                start_sec=1.0,
                end_sec=2.0,
            )

        gates = result["structured_event_stream"]["quality_gates"]
        self.assertFalse(gates["allowlisted_terminal"])
        self.assertFalse(gates["attack_qualified"])

    def test_rejects_manifest_already_marked_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(StaleArtifactError):
                persist_footpass_window_stream(
                    {
                        "run_directory": directory,
                        "provenance_status": "stale",
                    },
                    AttackSequence(
                        attack_id="a",
                        team="0",
                        start_sec=0.0,
                        end_sec=0.0,
                        events=(Event(0.0, "shot", "0"),),
                    ),
                    start_sec=0.0,
                    end_sec=1.0,
                )
