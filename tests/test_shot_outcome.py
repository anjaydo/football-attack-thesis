from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attack_analysis.attack_v2 import decode_footpass_attack_v2
from attack_analysis.metrics import compute_metrics
from attack_analysis.attack_evidence import _attack_to_preview_sequence
from attack_analysis.shot_outcome import (
    ScoreboardReader,
    ScoreReading,
    attack_with_video_outcome,
    ball_in_goal_times,
    infer_attack_outcomes,
    infer_shot_outcome,
    match_scoreboard_tokens,
    parse_scoreboard_tokens,
)
from test_attack_v2 import _action


def _token(text: str, left: int, width: int = 30) -> list:
    box = [[left, 60], [left + width, 60], [left + width, 90], [left, 90]]
    return [box, text, 0.95]


def _reads(*items: tuple[float, int, int]) -> list[ScoreReading]:
    return [ScoreReading(t, home, away) for t, home, away in items]


def _ball(t: float, x: float, y: float, confidence: float = 0.5) -> SimpleNamespace:
    ball = SimpleNamespace(object_type="ball", pitch_x=x, pitch_y=y, confidence=confidence)
    return SimpleNamespace(timestamp_sec=t, scene_cut=False, objects=(ball,))


class ScoreboardParsingTests(unittest.TestCase):
    def test_parses_name_score_score_name(self) -> None:
        tokens = [
            _token("SAB", 518),
            _token("0", 448),
            _token("1st HALF", 117, 120),
            _token("MUN", 272),
            _token("T", 394),
        ]
        self.assertEqual(parse_scoreboard_tokens(tokens), (1, 0))

    def test_parses_joined_score(self) -> None:
        tokens = [_token("ARS", 10), _token("2-1", 60), _token("CHE", 120)]
        score, anchor = match_scoreboard_tokens(tokens)
        self.assertEqual(score, (2, 1))
        self.assertEqual(len(anchor.score_boxes), 1)

    def test_rejects_text_without_score_layout(self) -> None:
        self.assertIsNone(parse_scoreboard_tokens([_token("ISIHALF", 10), _token("165", 80)]))
        self.assertIsNone(parse_scoreboard_tokens([_token("MUN", 10), _token("0", 60)]))


class ScoreboardReaderTests(unittest.TestCase):
    def test_anchored_read_needs_both_team_names(self) -> None:
        reader = ScoreboardReader(ocr=lambda *args, **kwargs: ([["MUN", 0.9]], None))
        frame = np.zeros((200, 800, 3), dtype=np.uint8)
        _score, anchor = match_scoreboard_tokens(
            [_token("MUN", 272), _token("0", 394), _token("0", 448), _token("SAB", 518)]
        )
        self.assertIsNone(reader.read_anchored(frame, anchor))

    def test_anchored_read_maps_lookalikes(self) -> None:
        answers = iter(["MUN", "SAB", "T", "O"])
        reader = ScoreboardReader(ocr=lambda *args, **kwargs: ([[next(answers), 0.9]], None))
        frame = np.zeros((200, 800, 3), dtype=np.uint8)
        _score, anchor = match_scoreboard_tokens(
            [_token("MUN", 272), _token("0", 394), _token("0", 448), _token("SAB", 518)]
        )
        self.assertEqual(reader.read_anchored(frame, anchor), (1, 0))


class ShotOutcomeTests(unittest.TestCase):
    def test_score_increase_after_shot_is_goal(self) -> None:
        readings = _reads((5, 0, 0), (12, 0, 0), (19, 1, 0), (20, 1, 0), (30, 1, 0))
        evidence = infer_shot_outcome(12.2, readings, after_limit_sec=34.6)
        self.assertEqual(evidence.outcome, "goal")
        self.assertEqual(evidence.source, "video_scoreboard")
        self.assertEqual(evidence.score_after, (1, 0))
        self.assertEqual(evidence.score_change_sec, 19)

    def test_single_misread_is_not_a_goal(self) -> None:
        readings = _reads((10, 1, 0), (20, 2, 0), (25, 1, 0), (40, 1, 0))
        self.assertEqual(infer_shot_outcome(12, readings).outcome, "no_goal")

    def test_unchanged_score_needs_settle_time(self) -> None:
        readings = _reads((97, 3, 0), (100, 3, 0), (104, 3, 0), (106, 3, 0))
        evidence = infer_shot_outcome(100.28, readings)
        self.assertEqual(evidence.outcome, "unknown")
        self.assertIsNone(evidence.source)

    def test_change_after_next_shot_belongs_to_next_shot(self) -> None:
        readings = _reads((30, 1, 0), (40, 1, 0), (48, 1, 0), (57, 2, 0), (58, 2, 0))
        self.assertEqual(infer_shot_outcome(34.6, readings, after_limit_sec=50.6).outcome, "no_goal")
        self.assertEqual(infer_shot_outcome(50.6, readings).outcome, "goal")

    def test_ball_in_goal_without_scoreboard(self) -> None:
        evidence = infer_shot_outcome(10.0, (), ball_times=(10.4, 10.6))
        self.assertEqual(evidence.outcome, "goal")
        self.assertEqual(evidence.source, "video_ball")
        self.assertEqual(infer_shot_outcome(10.0, (), ball_times=(10.4,)).outcome, "unknown")


class BallInGoalTests(unittest.TestCase):
    def test_mirrors_for_right_to_left_attack(self) -> None:
        frames = [
            _ball(10.2, 104.8, 34.0),
            _ball(10.4, 0.2, 33.0),
            _ball(10.6, 0.3, 50.0),
            _ball(10.8, 0.3, 34.0, confidence=0.1),
            _ball(14.0, 0.1, 34.0),
        ]
        self.assertEqual(
            ball_in_goal_times(frames, shot_sec=10.0, left_to_right=False),
            (10.4,),
        )
        self.assertEqual(
            ball_in_goal_times(frames, shot_sec=10.0, left_to_right=True),
            (10.2,),
        )
        self.assertEqual(ball_in_goal_times(frames, shot_sec=10.0, left_to_right=None), ())


class AttackOutcomeTests(unittest.TestCase):
    def test_goal_reaches_attack_metrics(self) -> None:
        decoded = decode_footpass_attack_v2([_action(0, "pass", "A"), _action(2, "shot", "A")])
        attack = decoded.attacks[0]
        readings = _reads((0, 0, 0), (1, 0, 0), (8, 1, 0), (9, 1, 0))
        outcomes = infer_attack_outcomes((attack,), readings)
        enriched = attack_with_video_outcome(attack, outcomes[attack.attack_id])
        self.assertEqual(enriched.attack_id, attack.attack_id)
        self.assertEqual(enriched.terminal.outcome, "goal")
        terminal = next(
            item for item in enriched.possession.actions if item.action_id == attack.terminal.action_id
        )
        self.assertIn("outcome_source:video_scoreboard", terminal.qualifiers)
        metrics = compute_metrics(_attack_to_preview_sequence(enriched))
        self.assertEqual(metrics.goals, 1)

    def test_headed_goal_counts_as_shot_and_goal(self) -> None:
        decoded = decode_footpass_attack_v2(
            [_action(0, "pass", "A"), _action(1, "header", "A"), _action(2, "headed_shot", "A")]
        )
        attack = decoded.attacks[0]
        self.assertEqual(attack.terminal.terminal_type.value, "headed_attempt")
        readings = _reads((0, 0, 0), (1, 0, 0), (8, 1, 0), (9, 1, 0))
        outcomes = infer_attack_outcomes((attack,), readings)
        metrics = compute_metrics(
            _attack_to_preview_sequence(attack_with_video_outcome(attack, outcomes[attack.attack_id]))
        )
        self.assertEqual((metrics.shots, metrics.shots_on_target, metrics.goals), (1, 1, 1))

    def test_unknown_leaves_attack_untouched(self) -> None:
        decoded = decode_footpass_attack_v2([_action(0, "pass", "A"), _action(2, "shot", "A")])
        attack = decoded.attacks[0]
        outcomes = infer_attack_outcomes((attack,), ())
        self.assertIs(attack_with_video_outcome(attack, outcomes[attack.attack_id]), attack)


if __name__ == "__main__":
    unittest.main()
