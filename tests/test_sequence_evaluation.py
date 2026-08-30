from __future__ import annotations

import unittest

from attack_analysis.sequence_evaluation import (
    MetricSpec,
    SequenceSpan,
    evaluate_acceptance_gates,
    match_sequence_spans,
    paired_f1_improvement_bootstrap,
    report_to_json,
    select_balanced_operating_point,
    sequence_evaluation_report,
    sequence_match_bootstrap_intervals,
    sequence_metrics_at_threshold,
    temporal_iou,
)


def span(
    sequence_id: str,
    start: float,
    end: float,
    *,
    match_id: str = "match-1",
    period: str = "1",
    team: str = "team-a",
    terminal: str = "shot_attempt",
) -> SequenceSpan:
    return SequenceSpan(
        sequence_id=sequence_id,
        match_id=match_id,
        period=period,
        stable_team_id=team,
        start_sec=start,
        end_sec=end,
        terminal_type=terminal,
    )


class SequenceEvaluationTests(unittest.TestCase):
    def test_temporal_iou_and_zero_duration_interval(self) -> None:
        self.assertAlmostEqual(temporal_iou(span("a", 0, 10), span("b", 5, 15)), 1 / 3)
        self.assertEqual(temporal_iou(span("a", 4, 4), span("b", 4, 4)), 1.0)

    def test_matching_is_constrained_by_match_period_and_stable_team(self) -> None:
        actual = (span("g", 0, 10),)
        predictions = (
            span("wrong-match", 0, 10, match_id="match-2"),
            span("wrong-period", 0, 10, period="2"),
            span("wrong-team", 0, 10, team="team-b"),
            span("valid", 1, 9),
        )
        result = match_sequence_spans(actual, predictions, tiou_threshold=0.5)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(result.matches[0].prediction_id, "valid")
        self.assertEqual(len(result.unmatched_prediction_ids), 3)

    def test_hungarian_matching_maximises_cardinality_before_total_iou(self) -> None:
        # Greedily assigning wide prediction x to g1 would strand g2. The
        # optimal solution assigns y->g1 and x->g2, yielding two matches.
        actual = (span("g1", 0, 10), span("g2", 8, 18))
        predictions = (span("x", 0, 18), span("y", 0, 7))
        result = match_sequence_spans(actual, predictions, tiou_threshold=0.5)
        pairs = {(item.ground_truth_id, item.prediction_id) for item in result.matches}
        self.assertEqual(pairs, {("g1", "y"), ("g2", "x")})

    def test_matching_can_require_compatible_terminals(self) -> None:
        actual = (span("g", 0, 10, terminal="shot_attempt"),)
        prediction = (span("p", 0, 10, terminal="headed_attempt"),)
        unconstrained = match_sequence_spans(actual, prediction)
        constrained = match_sequence_spans(
            actual,
            prediction,
            terminal_compatibility="exact",
        )
        self.assertEqual(len(unconstrained.matches), 1)
        self.assertEqual(len(constrained.matches), 0)

    def test_mapping_adapter_supports_v2_aliases_and_nested_terminal(self) -> None:
        value = {
            "attack_id": "attack-1",
            "match_id": "game-1",
            "half": 2,
            "possession_team_id": "blue",
            "start_time": 3,
            "end_time": 9,
            "terminal": {"type": "HEADED_ATTEMPT"},
        }
        converted = SequenceSpan.from_value(value)
        self.assertEqual(converted.sequence_id, "attack-1")
        self.assertEqual(converted.period, "2")
        self.assertEqual(converted.terminal_type, "headed_attempt")

    def test_mapping_adapter_accepts_gold_annotation_id(self) -> None:
        converted = SequenceSpan.from_value(
            {
                "annotation_id": "gold-1",
                "match_id": "game_18",
                "period": 1,
                "stable_team_id": "home",
                "start_sec": 4.0,
                "end_sec": 8.0,
                "terminal_type": "shot_attempt",
            }
        )
        self.assertEqual(converted.sequence_id, "gold-1")

    def test_mapping_adapter_supports_serialised_attack_v2(self) -> None:
        value = {
            "attack_id": "attack-2",
            "possession": {
                "possession_id": "possession-2",
                "match_id": "game-2",
                "period": 1,
                "possession_team_id": "red",
                "start_sec": 4.0,
                "end_sec": 8.0,
                "terminal": {"terminal_type": "shot_attempt"},
            },
        }
        converted = SequenceSpan.from_value(value)
        self.assertEqual(converted.sequence_id, "attack-2")
        self.assertEqual(converted.stable_team_id, "red")
        self.assertEqual(converted.start_sec, 4.0)
        self.assertEqual(converted.terminal_type, "shot_attempt")

    def test_metrics_report_paired_boundaries_and_terminal_confusion(self) -> None:
        actual = (
            span("g1", 0, 10, terminal="shot_attempt"),
            span("g2", 20, 30, terminal="headed_attempt"),
        )
        predictions = (
            span("p1", 0.5, 10.25, terminal="headed_attempt"),
            span("p2", 50, 60, terminal="forced_goal_line_out"),
        )
        report = sequence_metrics_at_threshold(
            actual,
            predictions,
            tiou_threshold=0.5,
            boundary_tolerance_seconds=0.4,
        )
        self.assertEqual(report["true_positive"], 1)
        self.assertEqual(report["f1"], 0.5)
        self.assertEqual(report["start_boundary"]["true_positive"], 0)
        self.assertEqual(report["end_boundary"]["true_positive"], 1)
        self.assertEqual(report["terminal"]["confusion"]["shot_attempt"]["headed_attempt"], 1)
        self.assertEqual(report["terminal"]["confusion"]["headed_attempt"]["__missed__"], 1)
        self.assertEqual(
            report["terminal"]["confusion"]["__spurious__"]["forced_goal_line_out"],
            1,
        )
        self.assertEqual(report["terminal"]["class_wise"]["headed_attempt"]["true_positive"], 0)

    def test_over_and_under_segmentation_are_reported(self) -> None:
        over = sequence_metrics_at_threshold(
            (span("g", 0, 10),),
            (span("p1", 0, 5), span("p2", 5, 10)),
            tiou_threshold=0.5,
        )
        self.assertEqual(over["segmentation"]["over_segmented_ground_truth"], 1)
        self.assertEqual(over["segmentation"]["over_segmentation_rate"], 1.0)

        under = sequence_metrics_at_threshold(
            (span("g1", 0, 5), span("g2", 5, 10)),
            (span("p", 0, 10),),
            tiou_threshold=0.5,
        )
        self.assertEqual(under["segmentation"]["under_segmented_predictions"], 1)
        self.assertEqual(under["segmentation"]["under_segmentation_rate"], 1.0)

    def test_cluster_bootstrap_is_seeded_and_resamples_whole_matches(self) -> None:
        actual = (
            span("g1", 0, 10, match_id="m1"),
            span("g2", 0, 10, match_id="m2"),
        )
        predicted = (span("p1", 0, 10, match_id="m1"),)
        first = sequence_match_bootstrap_intervals(actual, predicted, samples=100, seed=42)
        second = sequence_match_bootstrap_intervals(actual, predicted, samples=100, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["match_count"], 2)
        self.assertAlmostEqual(first["metrics"]["recall"]["estimate"], 0.5)
        self.assertEqual(first["metrics"]["recall"]["ci_lower"], 0.0)
        self.assertEqual(first["metrics"]["recall"]["ci_upper"], 1.0)

    def test_registry_report_has_all_required_tiou_thresholds(self) -> None:
        actual = (span("g", 0, 10),)
        predicted = (span("p", 1, 9),)
        report = sequence_evaluation_report(actual, predicted, bootstrap_samples=10)
        self.assertEqual(set(report["thresholds"]), {"0.25", "0.5", "0.75"})
        self.assertIs(report["primary"], report["thresholds"]["0.5"])
        self.assertEqual(report["primary"]["bootstrap"]["bootstrap_samples"], 10)
        self.assertEqual(set(report["primary"]["per_match"]), {"match-1"})
        self.assertIn("attack_f1_tiou_0_5", report["metric_registry"])

    def test_paired_f1_bootstrap_uses_same_resampled_matches(self) -> None:
        actual = (
            span("g1", 0, 10, match_id="m1"),
            span("g2", 0, 10, match_id="m2"),
        )
        candidate = (
            span("c1", 0, 10, match_id="m1"),
            span("c2", 0, 10, match_id="m2"),
        )
        baseline = (span("b1", 0, 10, match_id="m1"),)
        first = paired_f1_improvement_bootstrap(
            actual, candidate, baseline, samples=100, seed=42
        )
        second = paired_f1_improvement_bootstrap(
            actual, candidate, baseline, samples=100, seed=42
        )
        self.assertEqual(first, second)
        self.assertGreater(first["estimate"], 0.0)
        self.assertEqual(set(first["per_match"]), {"m1", "m2"})

    def test_balanced_operating_point_selection_uses_declared_tie_break(self) -> None:
        harmonic = 2 * 0.8 * 0.6 / (0.8 + 0.6)
        selected = select_balanced_operating_point(
            (
                {"name": "imbalanced", "precision": 0.8, "recall": 0.6},
                {"name": "balanced", "precision": harmonic, "recall": harmonic},
                {"name": "lower-f1", "precision": 0.67, "recall": 0.67},
            )
        )
        self.assertEqual(selected["name"], "balanced")
        self.assertAlmostEqual(selected["f1"], harmonic)

    def test_perfect_report_needs_and_passes_paired_baseline_gate(self) -> None:
        terminals = ("shot_attempt", "headed_attempt", "forced_goal_line_out")
        actual = tuple(span(f"g-{index}", index * 20, index * 20 + 10, terminal=value) for index, value in enumerate(terminals))
        predicted = tuple(span(f"p-{index}", index * 20, index * 20 + 10, terminal=value) for index, value in enumerate(terminals))
        missing_baseline = sequence_evaluation_report(actual, predicted, bootstrap_samples=10)
        missing_gates = evaluate_acceptance_gates(missing_baseline)
        self.assertFalse(missing_gates["passed"])
        self.assertIsNone(
            missing_gates["gates"]["paired_f1_improvement_ci_lower"]["observed"]
        )

        report = sequence_evaluation_report(
            actual,
            predicted,
            bootstrap_samples=10,
            frozen_baseline_predictions=(),
        )
        gates = evaluate_acceptance_gates(report)
        self.assertTrue(gates["passed"])
        self.assertEqual(gates["passed_count"], gates["gate_count"])
        self.assertNotIn("NaN", report_to_json(report))

    def test_missing_acceptance_metric_fails_instead_of_becoming_zero(self) -> None:
        specs = (MetricSpec("missing", "primary.nope", ">=", 0.5, "missing metric"),)
        gates = evaluate_acceptance_gates({"primary": {}}, specs)
        self.assertFalse(gates["passed"])
        self.assertIsNone(gates["gates"]["missing"]["observed"])

    def test_invalid_or_duplicate_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SequenceSpan("bad", "m", "1", "t", 2, 1)
        duplicate = span("duplicate", 0, 1)
        with self.assertRaisesRegex(ValueError, "duplicate ground-truth"):
            match_sequence_spans((duplicate, duplicate), ())


if __name__ == "__main__":
    unittest.main()
