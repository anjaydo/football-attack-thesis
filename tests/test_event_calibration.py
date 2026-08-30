from __future__ import annotations

import unittest

from attack_analysis.event_calibration import apply_event_thresholds, fit_event_threshold_profile
from attack_analysis.models import Event


def _event(frame: int, confidence: float) -> Event:
    return Event(
        timestamp=frame / 25.0,
        frame=frame,
        event_type="shot",
        team="0",
        jersey=9,
        confidence=confidence,
        source="footpass_predicted_dst",
    )


class EventCalibrationTests(unittest.TestCase):
    def test_fits_on_named_match_and_reports_held_out_gain(self) -> None:
        ground_truth = {"tune": (_event(10, 1.0),), "test": (_event(10, 1.0),)}
        predictions = {
            "tune": (_event(10, 0.8), _event(100, 0.2)),
            "test": (_event(10, 0.8), _event(100, 0.2)),
        }
        report = fit_event_threshold_profile(
            ground_truth,
            predictions,
            tuning_matches=("tune",),
            baseline_threshold=0.1,
            candidates=(0.1, 0.5),
        )
        self.assertEqual(report["thresholds"]["shot"], 0.5)
        self.assertGreater(
            report["held_out"]["calibrated"]["micro"]["f1"],
            report["held_out"]["baseline"]["micro"]["f1"],
        )

    def test_applies_class_specific_and_global_minimum(self) -> None:
        kept = apply_event_thresholds(
            (_event(1, 0.4), _event(2, 0.8)),
            {"shot": 0.5},
            minimum_confidence=0.2,
        )
        self.assertEqual([event.frame for event in kept], [2])


if __name__ == "__main__":
    unittest.main()
