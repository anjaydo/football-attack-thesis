from __future__ import annotations

import unittest

from attack_analysis.models import AttackSequence, Event
from attack_analysis.possession import assess_possession


def _sequence(*events: Event) -> AttackSequence:
    return AttackSequence(
        attack_id="game_1_H1:attack-0001",
        team="1",
        start_sec=events[0].timestamp,
        end_sec=events[-1].timestamp,
        events=events,
    )


class PossessionAssessmentTests(unittest.TestCase):
    def test_long_build_up_into_box_and_cross_is_dangerous_attack(self) -> None:
        assessment = assess_possession(
            _sequence(
                Event(1.0, "pass", "1", x=6.0, y=38.0),
                Event(10.0, "drive", "1", x=72.0, y=12.0),
                Event(20.0, "cross", "1", x=94.0, y=16.0),
            )
        )
        self.assertEqual(assessment.label, "dangerous_attack")
        self.assertTrue(assessment.is_attack)
        self.assertEqual(assessment.penalty_box_touches, 1)
        self.assertGreater(assessment.peak_progression_m, 80.0)

    def test_lateral_build_up_is_normal_possession(self) -> None:
        assessment = assess_possession(
            _sequence(
                Event(1.0, "pass", "1", x=32.0, y=12.0),
                Event(6.0, "drive", "1", x=36.0, y=45.0),
                Event(12.0, "pass", "1", x=34.0, y=30.0),
            )
        )
        self.assertEqual(assessment.label, "normal_possession")
        self.assertFalse(assessment.is_attack)

    def test_shot_always_qualifies_as_attack(self) -> None:
        assessment = assess_possession(
            _sequence(Event(1.0, "shot", "1", x=58.0, y=34.0, xg=0.01))
        )
        self.assertEqual(assessment.label, "attack")
        self.assertTrue(assessment.is_attack)

    def test_missing_positions_abstains_without_explicit_chance(self) -> None:
        assessment = assess_possession(_sequence(Event(1.0, "pass", "1")))
        self.assertEqual(assessment.label, "insufficient_evidence")
        self.assertFalse(assessment.is_attack)


if __name__ == "__main__":
    unittest.main()
