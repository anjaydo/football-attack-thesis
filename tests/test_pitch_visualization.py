from __future__ import annotations

import unittest

from attack_analysis.models import AttackSequence, Event, ObservedActionV2, SourceTier
from attack_analysis.pitch_visualization import plot_attack_pitch, plot_pitch_events


class PitchVisualizationTests(unittest.TestCase):
    def test_plots_full_event_contents_on_metric_pitch(self) -> None:
        sequence = AttackSequence(
            attack_id="game_1_H1:attack-0001",
            team="1",
            start_sec=1.0,
            end_sec=3.0,
            events=(
                Event(1.0, "pass", "1", x=35.0, y=45.0, jersey=8, frame=25, outcome="complete"),
                Event(2.0, "drive", "1", x=68.0, y=34.0, jersey=10, player_id="p10"),
                Event(3.0, "shot", "1", x=91.0, y=30.0, jersey=9, xg=0.24, xg_source="trained_xg:v1"),
            ),
        )

        figure = plot_attack_pitch(sequence)
        try:
            self.assertEqual(len(figure.axes), 1)
            labels = "\n".join(text.get_text() for text in figure.axes[0].texts)
            self.assertIn("1. Chuyền", labels)
            self.assertIn("t = 1.00s", labels)
            self.assertIn("frame 25", labels)
            self.assertIn("#8", labels)
            self.assertIn("vị trí (35.0, 45.0)", labels)
            self.assertIn("kết quả: complete", labels)
            self.assertIn("3. Sút", labels)
            self.assertIn("xG 0.240 (trained_xg:v1)", labels)
            self.assertIn("HƯỚNG TẤN CÔNG  →", labels)
            self.assertGreaterEqual(len(figure.axes[0].collections), 3)
        finally:
            figure.clear()

    def test_plots_observed_action_fields(self) -> None:
        action = ObservedActionV2(
            match_id="game_1",
            period=1,
            frame=120,
            timestamp_sec=4.8,
            action_type="cross",
            source_tier=SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL,
            actor_team_id="1",
            possession_team_id="1",
            x=88.0,
            y=12.0,
            jersey=7,
            role="winger",
            body_part="foot",
            outcome="complete",
            bbox_visible=True,
        )
        figure = plot_pitch_events((action,))
        try:
            labels = "\n".join(text.get_text() for text in figure.axes[0].texts)
            self.assertIn("1. Tạt bóng", labels)
            self.assertIn("hiệp 1", labels)
            self.assertIn("bộ phận: foot", labels)
            self.assertIn("bbox: có", labels)
            self.assertIn("oracle_actions_oracle_tactical", labels)
        finally:
            figure.clear()

    def test_attack_relative_plot_keeps_opponent_on_same_half(self) -> None:
        from attack_analysis.pitch_visualization import pitch_event_views

        possession = ObservedActionV2(
            match_id="game_1",
            period=1,
            frame=1,
            timestamp_sec=1.0,
            action_type="pass",
            source_tier=SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL,
            actor_team_id="A",
            possession_team_id="A",
            x=84.0,
            y=34.0,
        )
        opponent = ObservedActionV2(
            match_id="game_1",
            period=1,
            frame=2,
            timestamp_sec=1.2,
            action_type="tackle",
            source_tier=SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL,
            actor_team_id="B",
            possession_team_id="A",
            x=21.0,
            y=34.0,
        )
        views = pitch_event_views((possession, opponent), attack_relative=True)
        self.assertAlmostEqual(views[0].x, 84.0)
        self.assertAlmostEqual(views[1].x, 84.0)
        self.assertTrue(views[1].opponent)

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
