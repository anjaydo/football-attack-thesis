from __future__ import annotations

import unittest

from attack_analysis.models import AttackSequence, Event, ObservedActionV2, SourceTier
from attack_analysis.pitch_visualization import (
    aligned_event_xy,
    pitch_event_views,
    plot_attack_pitch,
    plot_pitch_events,
)


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

    def test_interactive_pitch_puts_event_details_in_hover(self) -> None:
        from attack_analysis.pitch_visualization import interactive_pitch_events

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
        chart = interactive_pitch_events(sequence.events)
        spec = chart.to_dict()
        mark_types = []
        tooltips = []
        for layer in spec.get("layer", []):
            mark = layer.get("mark")
            mark_types.append(mark.get("type") if isinstance(mark, dict) else mark)
            tooltip = (layer.get("encoding") or {}).get("tooltip") or []
            if isinstance(tooltip, dict):
                tooltips.append(tooltip)
            else:
                tooltips.extend(tooltip)
        self.assertIn("rect", mark_types)
        self.assertIn("line", mark_types)
        self.assertIn("point", mark_types)
        self.assertIn("text", mark_types)
        self.assertNotIn("image", mark_types)
        width = float(spec.get("width") or spec.get("config", {}).get("view", {}).get("continuousWidth") or 0)
        height = float(spec.get("height") or spec.get("config", {}).get("view", {}).get("continuousHeight") or 0)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertAlmostEqual(width / height, (105.0 + 4.0) / (68.0 + 4.0), places=2)
        y_encoding = next(
            (layer.get("encoding") or {}).get("y")
            for layer in spec.get("layer", [])
            if (layer.get("encoding") or {}).get("y")
        )
        self.assertTrue(y_encoding.get("scale", {}).get("reverse"))
        self.assertTrue(any(item.get("title") in {"Hành động", "Chi tiết"} for item in tooltips if isinstance(item, dict)))
        event_values = []
        for dataset in (spec.get("datasets") or {}).values():
            if isinstance(dataset, list):
                event_values.extend(dataset)
        for layer in spec.get("layer", []):
            values = ((layer.get("data") or {}).get("values")) or []
            if isinstance(values, list):
                event_values.extend(values)
        titles = [row.get("title") for row in event_values if isinstance(row, dict) and row.get("title")]
        details = " ".join(str(row.get("detail") or "") for row in event_values if isinstance(row, dict))
        event_coords = {
            (round(float(row["x"]), 1), round(float(row["y"]), 1))
            for row in event_values
            if isinstance(row, dict) and "action_label" in row
        }
        self.assertEqual(event_coords, {(35.0, 45.0), (68.0, 34.0), (91.0, 30.0)})
        self.assertIn("1. Chuyền", titles)
        self.assertIn("3. Sút", titles)
        self.assertIn("frame 25", details)
        self.assertIn("#8", details)
        self.assertIn("xG 0.240", details)

    def test_freeze_frame_draws_event_trail_and_pose_inset(self) -> None:
        from attack_analysis.pitch_visualization import plot_freeze_frame

        figure = plot_freeze_frame(
            [
                {"side": "ally", "x": 90.0, "y": 34.0, "is_shooter": True, "is_goalkeeper": False},
                {"side": "foe", "x": 102.0, "y": 34.0, "is_shooter": False, "is_goalkeeper": True},
            ],
            x_shooter=90.0,
            y_shooter=34.0,
            x_goalkeeper=102.0,
            y_goalkeeper=34.0,
            events=(
                {"action_type": "pass", "x": 70.0, "y": 40.0},
                {"action_type": "shot", "x": 90.0, "y": 34.0, "is_terminal": True},
            ),
            yolo_shooter={"x": 90.4, "y": 34.2},
            shooter_keypoints=(
                {"index": 0, "x": 20.0, "y": 10.0, "confidence": 0.9, "pitch_x": 90.0, "pitch_y": 32.0},
                {"index": 5, "x": 18.0, "y": 22.0, "confidence": 0.8, "pitch_x": 89.5, "pitch_y": 33.0},
                {"index": 6, "x": 22.0, "y": 22.0, "confidence": 0.8, "pitch_x": 90.5, "pitch_y": 33.0},
            ),
            shooter_bbox={"x1": 10.0, "y1": 8.0, "x2": 30.0, "y2": 50.0},
        )
        try:
            self.assertGreaterEqual(len(figure.axes), 2)
        finally:
            figure.clear()


    def test_short_gap_on_the_opposite_half_is_reflected_for_display(self) -> None:
        events = (
            Event(106.84, "drive", "game_18:roster-200", x=90.9, y=38.5, jersey=9),
            Event(107.64, "shot", "game_18:roster-200", x=92.1, y=37.0, jersey=9),
            Event(107.76, "block", "game_18:roster-200", x=9.9, y=36.3, jersey=13),
        )
        aligned = aligned_event_xy(events)
        self.assertFalse(aligned[0][2])
        self.assertFalse(aligned[1][2])
        self.assertTrue(aligned[2][2])
        self.assertAlmostEqual(aligned[2][0], 105.0 - 9.9)
        self.assertAlmostEqual(aligned[2][1], 36.3)
        views = pitch_event_views(events, attack_relative=False)
        self.assertAlmostEqual(views[2].x, 95.1)
        self.assertTrue(views[2].mirrored_to_same_side)
        self.assertIn("cùng mặt sân", views[2].annotation)

    def test_observed_actions_without_end_point_can_be_reflected(self) -> None:
        actions = tuple(
            ObservedActionV2(
                match_id="clip",
                period=1,
                frame=frame,
                timestamp_sec=frame / 25.0,
                action_type=action_type,
                source_tier=SourceTier.VIDEO_ONLY,
                actor_team_id="team_a",
                possession_team_id="team_a",
                x=x,
                y=36.0,
            )
            for frame, action_type, x in ((2671, "drive", 90.9), (2691, "shot", 92.1), (2694, "block", 9.9))
        )
        views = pitch_event_views(actions, attack_relative=False)
        self.assertTrue(views[2].mirrored_to_same_side)

    def test_a_pass_across_the_pitch_is_not_reflected(self) -> None:
        events = (
            Event(10.0, "pass", "1", x=30.0, y=34.0),
            Event(11.2, "drive", "1", x=71.0, y=40.0),
        )
        aligned = aligned_event_xy(events)
        self.assertFalse(aligned[0][2])
        self.assertFalse(aligned[1][2])
        self.assertAlmostEqual(aligned[1][0], 71.0)


if __name__ == "__main__":
    unittest.main()
