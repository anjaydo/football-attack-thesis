import tempfile
import unittest
from pathlib import Path

from attack_analysis.attack_v2 import FootpassAttackV2Config, decode_footpass_attack_v2
from attack_analysis.models import ObservedActionV2, SourceTier
from attack_analysis.scoreboard_goals import (
    MatchGoalTruth,
    ScoreboardGoalIndex,
    goal_times_from_changes,
    match_truth_from_readings,
    possession_outcomes,
    restart_goal_times,
    scoring_action,
    stable_score_changes,
)


def _action(
    second: float,
    action_type: str,
    team: str,
    *,
    x: float | None = None,
    y: float | None = None,
    replay: bool = False,
) -> ObservedActionV2:
    return ObservedActionV2(
        match_id="game_1_H1",
        period=1,
        frame=round(second * 25),
        timestamp_sec=second,
        action_type=action_type,
        source_tier=SourceTier.ORACLE_ACTIONS_ORACLE_TACTICAL,
        actor_team_id=team,
        x=x,
        y=y,
        replay=replay,
    )


def _readings(*scores: tuple[float, int, int]) -> list[dict[str, object]]:
    return [{"timestamp_sec": t, "home": home, "away": away} for t, home, away in scores]


class ScoreChangeTests(unittest.TestCase):
    def test_score_needs_consecutive_agreement(self) -> None:
        readings = _readings(
            (0, 0, 0), (2, 0, 0), (4, 0, 0),
            (6, 8, 0),
            (8, 0, 0),
            (10, 1, 0), (12, 1, 0), (14, 1, 0),
        )
        changes = stable_score_changes(readings)
        self.assertEqual([item["score"] for item in changes], [[0, 0], [1, 0]])
        self.assertEqual(changes[1]["timestamp_sec"], 10.0)

    def test_reverted_flip_is_not_a_goal(self) -> None:
        changes = [
            {"timestamp_sec": 0, "score": [0, 0]},
            {"timestamp_sec": 100, "score": [0, 1]},
            {"timestamp_sec": 200, "score": [0, 2]},
            {"timestamp_sec": 250, "score": [0, 1]},
            {"timestamp_sec": 900, "score": [1, 1]},
            {"timestamp_sec": 950, "score": [1, 4]},
        ]
        self.assertEqual(goal_times_from_changes(changes), [100.0, 900.0])

    def test_clean_feed_without_readings_has_no_scoreboard_truth(self) -> None:
        self.assertIsNone(match_truth_from_readings("game_3", []))
        truth = match_truth_from_readings("game_1", _readings((0, 0, 0), (2, 0, 0), (4, 0, 0)))
        self.assertEqual(truth.goal_times_sec, ())
        self.assertEqual(truth.final_score, (0, 0))


class ScoringActionTests(unittest.TestCase):
    def test_prefers_attempt_followed_by_celebration_stoppage(self) -> None:
        actions = [
            _action(10, "shot", "A"),
            _action(12, "pass", "B"),
            _action(30, "shot", "A"),
            _action(70, "pass", "B", x=52.5, y=34.0),
        ]
        self.assertEqual(scoring_action(actions, 90).timestamp_sec, 30)

    def test_ignores_replays_and_old_attempts(self) -> None:
        actions = [_action(10, "shot", "A"), _action(200, "shot", "A", replay=True)]
        self.assertIsNone(scoring_action(actions, 220))

    def test_restart_by_other_team_in_own_half_after_long_stoppage(self) -> None:
        actions = [
            _action(0, "pass", "A", x=52.5, y=34.0),
            _action(30, "shot", "A"),
            _action(90, "pass", "B", x=44.0, y=33.5),
            _action(100, "shot", "B"),
            _action(160, "pass", "A", x=6.0, y=34.0),
            _action(200, "shot", "A"),
            _action(260, "pass", "A", x=40.0, y=30.0),
        ]
        self.assertEqual(restart_goal_times(actions), [31.0])


class PossessionOutcomeTests(unittest.TestCase):
    def test_goal_and_no_goal_are_attached_to_attempt_possessions(self) -> None:
        actions = [
            _action(1, "pass", "A"),
            _action(3, "shot", "A"),
            _action(5, "pass", "B"),
            _action(7, "shot", "B"),
            _action(40, "pass", "A", x=52.5, y=34.0),
        ]
        decoded = decode_footpass_attack_v2(actions, FootpassAttackV2Config())
        index = ScoreboardGoalIndex(
            {"game_1": MatchGoalTruth("game_1", (35.0,), "video_scoreboard", (0, 1))}
        )
        outcomes = possession_outcomes("game_1_H1", actions, decoded.possessions, index)
        by_team = {
            item.possession_team_id: outcomes.get(item.possession_id)
            for item in decoded.possessions
            if item.possession_id in outcomes
        }
        self.assertEqual(by_team, {"A": ("no_goal", "video_scoreboard"), "B": ("goal", "video_scoreboard")})

    def test_index_round_trip(self) -> None:
        index = ScoreboardGoalIndex(
            {
                "game_3": MatchGoalTruth("game_3", (1200.0,), "event_restart"),
                "game_10": MatchGoalTruth("game_10", (1784.0, 1912.0), "video_scoreboard", (1, 2)),
            }
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "goals.json"
            index.save(path)
            loaded = ScoreboardGoalIndex.load(path)
        self.assertEqual(loaded, index)
        self.assertEqual(loaded.for_match_key("game_10_H2").goal_times_sec, (1784.0, 1912.0))
        self.assertIsNone(loaded.for_match_key("game_99_H1"))


if __name__ == "__main__":
    unittest.main()
