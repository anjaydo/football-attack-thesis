from __future__ import annotations

import unittest

from attack_analysis.models import AttackSequence, Event
from attack_analysis.pitch_visualization import plot_attack_pitch


class PitchVisualizationTests(unittest.TestCase):
    def test_plots_ordered_events_on_metric_pitch(self) -> None:
        sequence = AttackSequence(
            attack_id="game_1_H1:attack-0001",
            team="1",
            start_sec=1.0,
            end_sec=3.0,
            events=(
                Event(1.0, "pass", "1", x=35.0, y=45.0, jersey=8),
                Event(2.0, "drive", "1", x=68.0, y=34.0, jersey=10),
                Event(3.0, "shot", "1", x=91.0, y=30.0, jersey=9, xg=0.24),
            ),
        )

        figure = plot_attack_pitch(sequence)
        try:
            self.assertEqual(len(figure.axes), 1)
            labels = {text.get_text() for text in figure.axes[0].texts}
            self.assertIn("1. Chuyền · #8", labels)
            self.assertIn("3. Sút · #9 · xG 0.240", labels)
            self.assertIn("HƯỚNG TẤN CÔNG  →", labels)
            self.assertGreaterEqual(len(figure.axes[0].collections), 3)
        finally:
            figure.clear()

    def test_rejects_attack_without_pitch_coordinates(self) -> None:
        sequence = AttackSequence(
            attack_id="game_1_H1:attack-0002",
            team="1",
            start_sec=1.0,
            end_sec=1.0,
            events=(Event(1.0, "pass", "1"),),
        )
        with self.assertRaisesRegex(ValueError, "no pitch coordinates"):
            plot_attack_pitch(sequence)


if __name__ == "__main__":
    unittest.main()
