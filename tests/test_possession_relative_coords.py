from __future__ import annotations

import unittest

from attack_analysis.footpass import (
    actions_in_attack_view,
    attack_view_median_x,
    normalize_footpass_position,
    possession_relative_position,
)
from attack_analysis.models import ObservedActionV2, PITCH_LENGTH, SourceTier


class PossessionRelativeCoordsTests(unittest.TestCase):
    def test_opponent_action_is_remirrored_into_possession_view(self) -> None:
        possession_x, _ = normalize_footpass_position(0.85, 0.5, 0)
        opponent_stored, _ = normalize_footpass_position(0.85, 0.5, 1)
        self.assertGreater(possession_x, 80)
        self.assertLess(opponent_stored, 30)
        x_view, _ = possession_relative_position(
            opponent_stored,
            34.0,
            actor_team_id="team_b",
            possession_team_id="team_a",
        )
        self.assertAlmostEqual(x_view, PITCH_LENGTH - opponent_stored)
        self.assertAlmostEqual(x_view, possession_x, places=5)

    def test_actions_in_attack_view_keep_sequence_on_one_half(self) -> None:
        pass_action = ObservedActionV2(
            match_id="game_18",
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
        tackle = ObservedActionV2(
            match_id="game_18",
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
        remapped = actions_in_attack_view((pass_action, tackle))
        self.assertAlmostEqual(remapped[0].x, 84.0)
        self.assertAlmostEqual(remapped[1].x, 84.0)
        self.assertGreater(attack_view_median_x((pass_action, tackle)) or 0, 52.5)


if __name__ == "__main__":
    unittest.main()
