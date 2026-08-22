import json
import unittest
from pathlib import Path

from attack_analysis.models import Event
from attack_analysis.scoring import evaluate_attack
from attack_analysis.segmentation import SegmentationConfig, segment_attacks


ROOT = Path(__file__).resolve().parents[1]


class SegmentationTests(unittest.TestCase):
    def test_segments_terminal_events_and_team_changes(self) -> None:
        with (ROOT / "examples" / "match_events.json").open(encoding="utf-8") as file:
            events = [Event.from_dict(item) for item in json.load(file)["events"]]

        attacks = segment_attacks(events)

        self.assertEqual(len(attacks), 2)
        self.assertEqual(attacks[0].team, "Team A")
        self.assertEqual(attacks[0].events[-1].event_type, "shot")
        self.assertTrue(evaluate_attack(attacks[0]).is_effective)
        self.assertEqual(attacks[1].team, "Team B")
        self.assertFalse(evaluate_attack(attacks[1]).is_effective)

    def test_splits_on_large_time_gap(self) -> None:
        events = [
            Event.from_dict({"timestamp": 0, "event_type": "pass", "team": "A"}),
            Event.from_dict({"timestamp": 20, "event_type": "pass", "team": "A"}),
        ]
        attacks = segment_attacks(events, SegmentationConfig(max_gap_sec=10))
        self.assertEqual(len(attacks), 2)

    def test_rejects_unordered_events(self) -> None:
        events = [
            Event.from_dict({"timestamp": 2, "event_type": "pass", "team": "A"}),
            Event.from_dict({"timestamp": 1, "event_type": "shot", "team": "A"}),
        ]
        with self.assertRaises(ValueError):
            segment_attacks(events)

