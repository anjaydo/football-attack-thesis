from __future__ import annotations

import unittest

from attack_analysis.attack_v2 import decode_footpass_attack_v2
from attack_analysis.footpass_terminal_inference import (
    FootpassTerminalInferenceConfig,
    infer_footpass_terminals,
)
from attack_analysis.models import ObservedActionV2, SequenceStatus, SourceTier, TerminalType


SOURCE = SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL


def _action(
    second: float,
    action_type: str,
    team: str | None,
    *,
    x: float | None = None,
    y: float | None = None,
    replay: bool = False,
    qualifiers: tuple[str, ...] = (),
) -> ObservedActionV2:
    return ObservedActionV2(
        match_id="game_18",
        period=1,
        frame=round(second * 25),
        timestamp_sec=second,
        action_type=action_type,
        source_tier=SOURCE,
        actor_team_id=team,
        x=x,
        y=y,
        replay=replay,
        qualifiers=qualifiers,
        action_probability=0.9,
        evidence_ids=(f"evidence-{second}-{action_type}-{team}",),
    )


class FootpassTerminalInferenceTests(unittest.TestCase):
    def test_cross_then_header_becomes_headed_attempt_attack(self) -> None:
        raw = (
            _action(1.0, "pass", "A", x=60.0, y=34.0),
            _action(2.0, "cross", "A", x=88.0, y=10.0),
            _action(2.4, "header", "A", x=92.0, y=34.0),
        )
        inferred = infer_footpass_terminals(raw)
        header = inferred.actions[2]
        decoded = decode_footpass_attack_v2(inferred.actions)

        self.assertEqual(inferred.headed_promoted, 1)
        self.assertIn("headed_attempt", header.qualifiers)
        self.assertEqual(header.body_part, "head")
        self.assertEqual(header.action_id, raw[2].action_id)
        self.assertTrue(
            any(item.field_name == "qualifiers" for item in header.field_provenance)
        )
        self.assertEqual(len(decoded.attacks), 1)
        self.assertEqual(decoded.attacks[0].terminal.terminal_type, TerminalType.HEADED_ATTEMPT)

    def test_midfield_header_stays_generic(self) -> None:
        raw = (
            _action(1.0, "pass", "A", x=50.0, y=34.0),
            _action(2.0, "header", "A", x=52.0, y=34.0),
        )
        inferred = infer_footpass_terminals(raw)
        decoded = decode_footpass_attack_v2(inferred.actions)

        self.assertEqual(inferred.headed_promoted, 0)
        self.assertEqual(inferred.headers_left_generic, 1)
        self.assertNotIn("headed_attempt", inferred.actions[1].qualifiers)
        self.assertEqual(len(decoded.attacks), 0)
        self.assertEqual(decoded.possessions[0].status, SequenceStatus.RIGHT_CENSORED)

    def test_attacking_third_header_requires_established_control(self) -> None:
        promoted = infer_footpass_terminals(
            (
                _action(1.0, "drive", "A", x=80.0, y=34.0),
                _action(2.0, "header", "A", x=92.0, y=34.0),
            )
        )
        contested = infer_footpass_terminals(
            (
                _action(1.0, "pass", "A", x=50.0, y=34.0),
                _action(2.0, "header", "B", x=92.0, y=34.0),
            )
        )

        self.assertEqual(promoted.headed_promoted, 1)
        self.assertEqual(contested.headed_promoted, 0)

    def test_shot_block_and_attacking_throw_in_promotes_forced_out(self) -> None:
        raw = (
            _action(1.0, "pass", "A", x=70.0, y=34.0),
            _action(2.0, "cross", "A", x=95.0, y=8.0),
            _action(2.2, "block", "B", x=100.0, y=34.0),
            _action(4.0, "throw_in", "A", x=105.0, y=20.0),
        )
        inferred = infer_footpass_terminals(raw)
        decoded = decode_footpass_attack_v2(inferred.actions)

        self.assertEqual(inferred.blocks_promoted, 1)
        self.assertIn("forced_goal_line_out", inferred.actions[2].qualifiers)
        self.assertEqual(inferred.actions[2].beneficiary_team_id, "A")
        self.assertEqual(len(decoded.attacks), 1)
        self.assertEqual(
            decoded.attacks[0].terminal.terminal_type,
            TerminalType.FORCED_GOAL_LINE_OUT,
        )

    def test_block_without_restart_is_not_forced_out(self) -> None:
        inferred = infer_footpass_terminals(
            (
                _action(1.0, "pass", "A"),
                _action(2.0, "shot", "A", x=90.0, y=34.0),
                _action(2.2, "block", "B", x=100.0, y=34.0),
                _action(2.8, "drive", "A", x=88.0, y=34.0),
            )
        )
        self.assertEqual(inferred.blocks_promoted, 0)
        self.assertNotIn("forced_goal_line_out", inferred.actions[2].qualifiers)

    def test_unknown_replay_and_large_gap_do_not_invent_terminals(self) -> None:
        unknown = infer_footpass_terminals(
            (
                _action(1.0, "cross", "A", x=90.0, y=10.0, qualifiers=("replay_status_unknown",)),
                _action(1.4, "header", "A", x=92.0, y=34.0, qualifiers=("replay_status_unknown",)),
            )
        )
        gapped = infer_footpass_terminals(
            (
                _action(1.0, "cross", "A", x=90.0, y=10.0),
                _action(12.0, "block", "B", x=100.0, y=34.0),
                _action(13.0, "throw_in", "A"),
            ),
            FootpassTerminalInferenceConfig(max_observation_gap_sec=8.0),
        )

        self.assertEqual(unknown.headed_promoted, 0)
        self.assertEqual(gapped.blocks_promoted, 0)
