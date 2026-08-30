from __future__ import annotations

import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attack_analysis.attack_v2 import decode_footpass_attack_v2
from attack_analysis.attack_v2_shot_insights import (
    SEQUENCE_FILTER_CENSORED_SHOT,
    SEQUENCE_FILTER_CONFIRMED,
    GoldOutcomeIndex,
    actions_with_resolved_outcome,
    build_shot_insights,
    filter_sequence_entries,
    format_shot_outcome,
    resolve_shot_outcome,
)
from attack_analysis.gold_annotations import GoldSequenceAnnotationV2
from attack_analysis.models import ObservedActionV2, SourceTier, TerminalType
from test_attack_v2 import _action


class AttackV2ShotInsightsTests(unittest.TestCase):
    def test_promoted_vs_censored_shot_audit(self) -> None:
        decoded = decode_footpass_attack_v2(
            [
                _action(0, "pass", "A"),
                _action(2, "shot", "A"),
                _action(10, "shot", "A"),
            ]
        )
        insights = build_shot_insights(
            decoded.possessions[0].actions + decoded.possessions[-1].actions,
            decoded.possessions,
            decoded.attacks,
        )
        self.assertEqual(insights.summary["live_shots"], 2)
        self.assertEqual(insights.summary["promoted_to_attack"], 1)
        self.assertEqual(insights.summary["censored_shot_terminal"], 1)
        censored = filter_sequence_entries(
            sequence_filter=SEQUENCE_FILTER_CENSORED_SHOT,
            possessions=[item.to_dict() for item in decoded.possessions],
            attacks=[item.to_dict() for item in decoded.attacks],
            censored_possession_ids=insights.censored_possession_ids,
        )
        self.assertEqual(len(censored), 1)
        confirmed = filter_sequence_entries(
            sequence_filter=SEQUENCE_FILTER_CONFIRMED,
            possessions=[item.to_dict() for item in decoded.possessions],
            attacks=[item.to_dict() for item in decoded.attacks],
            censored_possession_ids=insights.censored_possession_ids,
        )
        self.assertEqual(len(confirmed), len(decoded.attacks))

    def test_gold_outcome_overrides_unknown_terminal(self) -> None:
        decoded = decode_footpass_attack_v2([_action(0, "pass", "A"), _action(2, "shot", "A")])
        possession = decoded.attacks[0].possession
        gold = GoldOutcomeIndex(
            (
                GoldSequenceAnnotationV2(
                    match_id=possession.match_id,
                    period=possession.period,
                    stable_team_id=possession.possession_team_id,
                    start_frame=0,
                    end_frame=50,
                    start_sec=0.0,
                    end_sec=2.0,
                    label="attack",
                    terminal_type="shot_attempt",
                    terminal_frame=50,
                    terminal_sec=2.0,
                    terminal_cause=None,
                    shot_body_part="foot",
                    shot_outcome="goal",
                    beneficiary_team_id=None,
                    last_touch_team_id=None,
                    replay=False,
                    completeness="complete",
                    confidence=1.0,
                    reason_codes=(),
                    notes="",
                    annotator_id="test",
                    annotated_at="2026-08-30T00:00:00Z",
                ),
            )
        )
        outcome, source = resolve_shot_outcome(possession, gold_index=gold)
        self.assertEqual(outcome, "goal")
        self.assertEqual(source, "gold_annotation")
        self.assertEqual(format_shot_outcome(outcome), "Bàn thắng")
        enriched = actions_with_resolved_outcome(
            possession,
            shot_outcome=outcome,
            outcome_source=source,
        )
        terminal_action = next(item for item in enriched if item.action_id == possession.terminal.action_id)
        self.assertEqual(terminal_action.outcome, "goal")
        self.assertIn("outcome_source:gold_annotation", terminal_action.qualifiers)


if __name__ == "__main__":
    unittest.main()
