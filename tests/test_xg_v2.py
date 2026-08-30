from __future__ import annotations

import unittest

from attack_analysis.models import (
    AttackSequence,
    AttackSequenceV2,
    Event,
    ObservedActionV2,
    PossessionSequenceV2,
    SequenceStatus,
    SourceTier,
    TerminalEventV2,
    TerminalType,
)
from attack_analysis.xg_v2 import (
    ConfirmedXGRecordV2,
    XG_V2_FEATURE_COLUMNS,
    XGFeatureVectorV2,
    evaluate_xg_v2,
    normalize_xg_v2_features,
    paired_match_cluster_bootstrap_brier_delta,
    score_confirmed_attack_xg_v2,
    xg_v2_acceptance_gates,
)


SOURCE = SourceTier.VIDEO_ONLY


def _attack(
    terminal_type: TerminalType,
    *,
    outcome: str | None = None,
) -> AttackSequenceV2:
    action_type = (
        "header"
        if terminal_type is TerminalType.HEADED_ATTEMPT
        else "shot"
        if terminal_type is TerminalType.SHOT_ATTEMPT
        else "cross"
    )
    action = ObservedActionV2(
        match_id="match-1",
        period=1,
        frame=250,
        timestamp_sec=10.0,
        action_type=action_type,
        source_tier=SOURCE,
        actor_team_id="A",
        possession_team_id="A",
        body_part="head" if terminal_type is TerminalType.HEADED_ATTEMPT else None,
        outcome=outcome,
    )
    terminal = TerminalEventV2(
        match_id="match-1",
        period=1,
        frame=250,
        timestamp_sec=10.0,
        terminal_type=terminal_type,
        possession_team_id="A",
        source_tier=SOURCE,
        beneficiary_team_id=(
            "A" if terminal_type is TerminalType.FORCED_GOAL_LINE_OUT else None
        ),
        last_touch_team_id=(
            "B" if terminal_type is TerminalType.FORCED_GOAL_LINE_OUT else None
        ),
        cause=(
            "defender_deflection_out"
            if terminal_type is TerminalType.FORCED_GOAL_LINE_OUT
            else None
        ),
        body_part="head" if terminal_type is TerminalType.HEADED_ATTEMPT else None,
        outcome=outcome,
        action_id=action.action_id,
        evidence_ids=(action.action_id,),
    )
    possession = PossessionSequenceV2(
        match_id="match-1",
        period=1,
        possession_team_id="A",
        start_frame=250,
        end_frame=250,
        start_sec=10.0,
        end_sec=10.0,
        actions=(action,),
        status=SequenceStatus.COMPLETE,
        source_tier=SOURCE,
        terminal=terminal,
    )
    return AttackSequenceV2(possession)


class XGFeatureContractTests(unittest.TestCase):
    def test_unknown_context_is_not_regular_play_or_zero_semantics(self) -> None:
        row = normalize_xg_v2_features({})

        self.assertEqual(tuple(row), XG_V2_FEATURE_COLUMNS)
        self.assertEqual(row["shooter_location_missing"], 1.0)
        self.assertEqual(row["shot_angle_missing"], 1.0)
        self.assertEqual(row["distance_to_goal_missing"], 1.0)
        self.assertEqual(row["body_part_unknown"], 1.0)
        self.assertEqual(row["under_pressure_missing"], 1.0)
        self.assertEqual(row["goalkeeper_visibility_missing"], 1.0)
        self.assertEqual(row["defenders_visibility_missing"], 1.0)
        self.assertEqual(row["play_pattern_unknown"], 1.0)
        self.assertEqual(row["play_pattern_regular_play"], 0.0)
        self.assertEqual(row["shot_type_unknown"], 1.0)
        self.assertEqual(row["observed_players_missing"], 1.0)

    def test_normalizer_derives_geometry_and_normalizes_visible_counts(self) -> None:
        row = normalize_xg_v2_features(
            {
                "x_shooter": 94.5,
                "y_shooter": 34.0,
                "body_part": "Head",
                "under_pressure": False,
                "play_pattern": "From Corner",
                "shot_type": "Open Play",
                "goalkeeper_visible": True,
                "x_goalkeeper": 104.0,
                "y_goalkeeper": 34.0,
                "observed_players": 20,
                "foe_players": 9,
                "ally_players": 10,
                "defending_players": 2,
            }
        )

        self.assertAlmostEqual(row["x_shooter_norm"], 0.9)
        self.assertEqual(row["shot_angle_missing"], 0.0)
        self.assertGreater(row["shot_angle_norm"], 0.0)
        self.assertEqual(row["body_part_head"], 1.0)
        self.assertEqual(row["under_pressure"], 0.0)
        self.assertEqual(row["under_pressure_missing"], 0.0)
        self.assertEqual(row["play_pattern_corner"], 1.0)
        self.assertEqual(row["play_pattern_regular_play"], 0.0)
        self.assertAlmostEqual(row["observed_players_norm"], 20 / 22)
        self.assertAlmostEqual(row["foe_players_norm"], 9 / 20)
        self.assertAlmostEqual(row["ally_players_norm"], 10 / 20)
        self.assertAlmostEqual(row["defending_players_norm"], 2 / 20)

    def test_counts_can_derive_observed_players_but_conflicts_are_rejected(self) -> None:
        row = normalize_xg_v2_features({"foe_players": 8, "ally_players": 9})
        self.assertAlmostEqual(row["observed_players_norm"], 18 / 22)
        self.assertEqual(row["observed_players_missing"], 0.0)

        with self.assertRaisesRegex(ValueError, "conflict"):
            normalize_xg_v2_features(
                {
                    "goalkeeper_visible": False,
                    "x_goalkeeper": 104.0,
                    "y_goalkeeper": 34.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            normalize_xg_v2_features(
                {"observed_players": 5, "foe_players": 6}
            )

    def test_feature_vector_constructor_preserves_exact_schema(self) -> None:
        vector = XGFeatureVectorV2.from_raw(
            {"x_shooter": 90.0, "y_shooter": 30.0, "play_pattern": None}
        )
        self.assertEqual(tuple(vector.to_mapping()), XG_V2_FEATURE_COLUMNS)
        self.assertEqual(vector.to_mapping()["play_pattern_unknown"], 1.0)


class XGMetricsTests(unittest.TestCase):
    def test_perfect_predictions_have_zero_brier_and_ece(self) -> None:
        metrics = evaluate_xg_v2([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1])
        self.assertEqual(metrics.brier, 0.0)
        self.assertEqual(metrics.ece_10, 0.0)
        # Complete separation has no finite logistic calibration estimate.
        self.assertIsNone(metrics.calibration_slope)
        self.assertIsNone(metrics.calibration_intercept)

    def test_known_calibration_has_slope_one_and_intercept_zero(self) -> None:
        probabilities = [0.05] * 20 + [0.95] * 20
        outcomes = [1] + [0] * 19 + [0] + [1] * 19
        metrics = evaluate_xg_v2(probabilities, outcomes)

        self.assertAlmostEqual(metrics.brier, 0.0475)
        self.assertAlmostEqual(metrics.ece_10, 0.0)
        self.assertAlmostEqual(metrics.calibration_slope, 1.0, places=6)
        self.assertAlmostEqual(metrics.calibration_intercept, 0.0, places=6)

    def test_calibration_fit_is_finite_for_imbalanced_outcomes(self) -> None:
        probabilities = [0.05] * 100 + [0.5] * 20
        outcomes = [1] * 5 + [0] * 95 + [1] * 10 + [0] * 10
        metrics = evaluate_xg_v2(probabilities, outcomes)

        self.assertIsNotNone(metrics.calibration_slope)
        self.assertIsNotNone(metrics.calibration_intercept)
        self.assertAlmostEqual(metrics.calibration_slope, 1.0, places=6)
        self.assertAlmostEqual(metrics.calibration_intercept, 0.0, places=6)

    def test_paired_bootstrap_is_seeded_and_resamples_matches(self) -> None:
        probabilities = [0.05] * 20 + [0.95] * 20
        outcomes = [1] + [0] * 19 + [0] + [1] * 19
        match_ids = [f"match-{index // 10}" for index in range(40)]
        first = paired_match_cluster_bootstrap_brier_delta(
            match_ids,
            outcomes,
            probabilities,
            [0.5] * 40,
            samples=200,
            seed=42,
        )
        second = paired_match_cluster_bootstrap_brier_delta(
            match_ids,
            outcomes,
            probabilities,
            [0.5] * 40,
            samples=200,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertLess(first.upper_95, 0.0)
        self.assertEqual(first.matches, 4)

    def test_acceptance_gates_pass_only_with_complete_nonmissing_evidence(self) -> None:
        probabilities = [0.05] * 20 + [0.95] * 20
        outcomes = [1] + [0] * 19 + [0] + [1] * 19
        metrics = evaluate_xg_v2(probabilities, outcomes)
        bootstrap = paired_match_cluster_bootstrap_brier_delta(
            [f"match-{index // 10}" for index in range(40)],
            outcomes,
            probabilities,
            [0.5] * 40,
            samples=100,
        )
        self.assertTrue(xg_v2_acceptance_gates(metrics, bootstrap).passed)

        missing = evaluate_xg_v2([0.05, None], [0, 1])
        missing_gates = xg_v2_acceptance_gates(missing, None)
        self.assertFalse(missing_gates.passed)
        self.assertFalse(missing_gates.complete_coverage)
        self.assertFalse(missing_gates.calibration_slope)
        self.assertFalse(missing_gates.paired_brier_delta_upper_ci)

        unlabelled = evaluate_xg_v2([0.05, 0.95], [0, None])
        self.assertFalse(xg_v2_acceptance_gates(unlabelled, bootstrap).complete_coverage)


class ConfirmedXGGateTests(unittest.TestCase):
    def test_shot_and_headed_attempt_are_scoreable(self) -> None:
        shot_record = score_confirmed_attack_xg_v2(
            _attack(TerminalType.SHOT_ATTEMPT, outcome="goal"),
            0.35,
            model_id="model-checksum",
        )
        headed_record = score_confirmed_attack_xg_v2(
            _attack(TerminalType.HEADED_ATTEMPT, outcome="saved"),
            0.12,
            model_id="model-checksum",
        )
        self.assertEqual(shot_record.outcome, 1)
        self.assertEqual(headed_record.outcome, 0)
        self.assertEqual(headed_record.terminal_type, TerminalType.HEADED_ATTEMPT)
        self.assertEqual(ConfirmedXGRecordV2.from_dict(shot_record.to_dict()), shot_record)

    def test_missing_outcome_remains_unknown(self) -> None:
        record = score_confirmed_attack_xg_v2(
            _attack(TerminalType.SHOT_ATTEMPT),
            0.2,
            model_id="model-checksum",
            features={"x_shooter": 93.0, "y_shooter": 34.0},
        )
        self.assertIsNone(record.outcome)
        self.assertIsNotNone(record.features)
        self.assertIsNone(record.to_dict()["outcome"])

    def test_forced_out_censored_and_legacy_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shot_attempt or headed_attempt"):
            score_confirmed_attack_xg_v2(
                _attack(TerminalType.FORCED_GOAL_LINE_OUT),
                0.1,
                model_id="model-checksum",
            )

        action = ObservedActionV2(
            match_id="m",
            period=1,
            frame=1,
            timestamp_sec=0.04,
            action_type="shot",
            source_tier=SOURCE,
            actor_team_id="A",
            possession_team_id="A",
        )
        censored = PossessionSequenceV2(
            match_id="m",
            period=1,
            possession_team_id="A",
            start_frame=1,
            end_frame=1,
            start_sec=0.04,
            end_sec=0.04,
            actions=(action,),
            status=SequenceStatus.RIGHT_CENSORED,
            source_tier=SOURCE,
        )
        with self.assertRaisesRegex(TypeError, "AttackSequenceV2"):
            score_confirmed_attack_xg_v2(censored, 0.1, model_id="model-checksum")

        legacy = AttackSequence(
            attack_id="legacy-1",
            team="A",
            start_sec=0.0,
            end_sec=0.0,
            events=(Event(timestamp=0.0, event_type="shot", team="A"),),
        )
        with self.assertRaisesRegex(TypeError, "AttackSequenceV2"):
            score_confirmed_attack_xg_v2(legacy, 0.1, model_id="model-checksum")


if __name__ == "__main__":
    unittest.main()
