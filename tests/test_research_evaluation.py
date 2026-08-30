from __future__ import annotations

import unittest

from attack_analysis.models import Event
from attack_analysis.research_evaluation import (
    action_spotting_report,
    action_spotting_dataset_report,
    boundary_f1,
    match_bootstrap_interval,
    top_view_error_report,
)


class ResearchEvaluationTests(unittest.TestCase):
    def test_action_spotting_is_one_to_one_and_class_wise(self) -> None:
        actual = (Event(timestamp=1, frame=25, event_type="pass", team="0"),)
        predicted = (
            Event(timestamp=1, frame=24, event_type="pass", team="0", confidence=0.9),
            Event(timestamp=1, frame=26, event_type="pass", team="0", confidence=0.8),
        )
        report = action_spotting_report(actual, predicted, tolerance_frames=12)
        self.assertEqual(report["class_wise"]["pass"]["true_positive"], 1)
        self.assertEqual(report["class_wise"]["pass"]["precision"], 0.5)

    def test_action_spotting_requires_player_identity(self) -> None:
        actual = (Event(timestamp=1, frame=25, event_type="pass", team="0", jersey=9),)
        predicted = (Event(timestamp=1, frame=25, event_type="pass", team="0", jersey=10),)
        report = action_spotting_report(actual, predicted)
        self.assertEqual(report["micro"]["true_positive"], 0)

    def test_action_spotting_uses_global_assignment_not_confidence_greedy(self) -> None:
        # Prediction at frame 10 can match either label. A confidence-first
        # greedy matcher takes frame 5 and strands the exact frame-5
        # prediction; the cardinality-first optimum recovers both events.
        actual = (
            Event(timestamp=0.2, frame=5, event_type="pass", team="0", jersey=9),
            Event(timestamp=0.6, frame=15, event_type="pass", team="0", jersey=9),
        )
        predicted = (
            Event(
                timestamp=0.4,
                frame=10,
                event_type="pass",
                team="0",
                jersey=9,
                confidence=0.99,
            ),
            Event(
                timestamp=0.2,
                frame=5,
                event_type="pass",
                team="0",
                jersey=9,
                confidence=0.10,
            ),
        )
        report = action_spotting_report(actual, predicted, tolerance_frames=5)
        self.assertEqual(report["micro"]["true_positive"], 2)
        self.assertEqual(report["micro"]["f1"], 1.0)

    def test_dataset_report_never_matches_events_across_matches(self) -> None:
        actual = Event(timestamp=1, frame=25, event_type="pass", team="0", jersey=9)
        predicted = Event(timestamp=1, frame=25, event_type="pass", team="0", jersey=9)
        report = action_spotting_dataset_report({"m1": (actual,)}, {"m2": (predicted,)})
        self.assertEqual(report["micro"]["true_positive"], 0)
        self.assertEqual(report["match_count"], 2)

    def test_boundary_and_projection_metrics(self) -> None:
        report = boundary_f1((10.0, 20.0), (10.5, 25.0), tolerance_seconds=1.0)
        self.assertEqual(report["f1"], 0.5)
        errors = top_view_error_report(((0, 0), (3, 4)), ((0, 0), (0, 0)))
        self.assertEqual(errors["median_error_m"], 2.5)
        self.assertEqual(errors["p90_error_m"], 5.0)

    def test_bootstrap_resamples_matches(self) -> None:
        report = match_bootstrap_interval(
            {"m1": (1.0, 1.0), "m2": (0.0,)},
            lambda values: sum(values) / len(values),
            samples=100,
        )
        self.assertEqual(report["match_count"], 2)
        self.assertAlmostEqual(report["estimate"], 2 / 3)

    def test_rare_classes_are_sliced_from_class_wise(self) -> None:
        actual = (Event(timestamp=1, frame=25, event_type="shot", team="0"),)
        predicted = (Event(timestamp=1, frame=25, event_type="shot", team="0"),)
        report = action_spotting_dataset_report({"m": actual}, {"m": predicted})
        self.assertEqual(report["rare_classes"]["shot"]["true_positive"], 1)
        self.assertEqual(report["rare_classes"]["header"]["actual"], 0)


if __name__ == "__main__":
    unittest.main()
