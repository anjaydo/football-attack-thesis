from __future__ import annotations

import unittest

from attack_analysis.models import Event, SourceTier
from attack_analysis.v2_adapters import decode_event_stream_v2, event_to_observed_action_v2


class V2AdapterTests(unittest.TestCase):
    def test_event_xg_is_not_promoted_to_observed_action(self) -> None:
        action = event_to_observed_action_v2(
            Event(
                timestamp=1.0,
                event_type="shot_candidate",
                team="team_a",
                x=90.0,
                y=34.0,
                xg=0.9,
                xg_source="legacy",
            ),
            match_id="clip",
        )
        self.assertNotIn("xg", action.to_dict())
        self.assertEqual(action.source_tier, SourceTier.VIDEO_ONLY)
        self.assertIn("replay_status_unknown", action.qualifiers)

    def test_forward_ball_candidate_is_not_attack(self) -> None:
        events = (
            Event(0.0, "possession_start", "team_a", frame=0),
            Event(1.0, "carry", "team_a", frame=25),
            Event(2.0, "shot_candidate", "team_a", frame=50),
        )
        actions, decoded = decode_event_stream_v2(events, match_id="clip")
        self.assertEqual(len(actions), 3)
        self.assertFalse(decoded.attacks)
        self.assertEqual(decoded.possessions[0].status.value, "uncertain")

    def test_upload_flow_promotes_shot_candidate_into_a_selectable_attack(self) -> None:
        from attack_analysis.v2_adapters import sequences_from_video_events

        _actions, decoded = sequences_from_video_events(
            (
                Event(0.0, "possession_start", "team_a", x=40.0, y=34.0, frame=0),
                Event(1.0, "carry", "team_a", x=60.0, y=34.0, frame=25),
                Event(
                    2.0,
                    "shot_candidate",
                    "team_a",
                    x=88.0,
                    y=34.0,
                    frame=50,
                    outcome="unverified",
                ),
            ),
            match_id="upload-clip",
        )
        self.assertEqual(len(decoded.attacks), 1)
        attack = decoded.attacks[0]
        self.assertEqual(attack.terminal.terminal_type.value, "shot_attempt")
        self.assertEqual(attack.possession.actions[-1].action_type, "shot")
        self.assertEqual(attack.possession.actions[-1].outcome, "unverified")
        self.assertEqual(attack.possession.source_tier, SourceTier.VIDEO_ONLY)

    def test_explicit_shot_can_terminate_complete_possession(self) -> None:
        _, decoded = decode_event_stream_v2(
            (
                Event(0.0, "possession_start", "team_a", frame=0),
                Event(1.0, "shot", "team_a", frame=25),
            ),
            match_id="clip",
            replay_status="live",
        )
        self.assertEqual(len(decoded.attacks), 1)

    def test_explicit_shot_with_unknown_replay_status_is_withheld(self) -> None:
        _, decoded = decode_event_stream_v2(
            (
                Event(0.0, "possession_start", "team_a", frame=0),
                Event(1.0, "shot", "team_a", frame=25),
            ),
            match_id="clip",
        )
        self.assertEqual(len(decoded.attacks), 0)
        self.assertEqual(decoded.possessions[0].status.value, "uncertain")

    def test_scene_cut_frames_are_decoder_boundaries(self) -> None:
        actions, decoded = decode_event_stream_v2(
            (
                Event(0.0, "pass", "team_a", frame=0),
                Event(2.0, "pass", "team_a", frame=50),
                Event(3.0, "shot", "team_a", frame=75),
            ),
            match_id="clip",
            replay_status="live",
            scene_cut_frames=(25,),
        )
        self.assertTrue(any(item.action_type == "scene_cut" for item in actions))
        self.assertEqual(len(decoded.attacks), 1)
        self.assertEqual(decoded.attacks[0].possession.status.value, "left_censored")
        self.assertGreater(decoded.attacks[0].possession.start_frame, 25)

    def test_generic_converter_refuses_fused_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "fused"):
            event_to_observed_action_v2(
                Event(0.0, "pass", "team_a"),
                match_id="clip",
                source_tier=SourceTier.FUSED,
            )


if __name__ == "__main__":
    unittest.main()
