import json
import unittest
from pathlib import Path

from attack_analysis.models import AttackSequence, Event
from attack_analysis.scoring import evaluate_attack


ROOT = Path(__file__).resolve().parents[1]


def load_example(name: str) -> AttackSequence:
    with (ROOT / "examples" / name).open("r", encoding="utf-8") as file:
        return AttackSequence.from_dict(json.load(file))


class AttackEvaluationTests(unittest.TestCase):
    def test_effective_attack_scores_high(self) -> None:
        result = evaluate_attack(load_example("attack_effective.json"))
        self.assertTrue(result.is_effective)
        self.assertEqual(result.label, "effective")
        self.assertGreaterEqual(result.score, 70.0)
        self.assertEqual(result.metrics.shots, 1)
        self.assertTrue(result.metrics.entered_penalty_box)
        self.assertEqual(result.metrics.progression_m, 43.0)
        self.assertEqual(result.is_effective, result.label == "effective")

    def test_ineffective_attack_scores_low(self) -> None:
        result = evaluate_attack(load_example("attack_ineffective.json"))
        self.assertFalse(result.is_effective)
        self.assertEqual(result.label, "ineffective")
        self.assertLess(result.score, 45.0)
        self.assertTrue(result.metrics.ended_in_turnover)

    def test_rejects_invalid_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            Event.from_dict(
                {
                    "timestamp": 0,
                    "event_type": "pass",
                    "team": "A",
                    "x": 106,
                    "y": 20,
                }
            )


if __name__ == "__main__":
    unittest.main()
