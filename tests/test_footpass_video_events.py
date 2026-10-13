import unittest
from dataclasses import replace

from attack_analysis.footpass_video_events import (
    TaadDetection,
    WideSegment,
    decode_highlight_attacks,
    highlight_attack_events,
    wide_view_segments,
)
from attack_analysis.observations import FrameObservation


def _player(track_id: str, height: float) -> dict[str, object]:
    return {
        "track_id": track_id,
        "object_type": "player",
        "confidence": 0.9,
        "bbox": {"x1": 100, "y1": 100, "x2": 130, "y2": 100 + height},
    }


def _frame(index: int, *, wide: bool, scene_cut: bool = False) -> FrameObservation:
    count, height = (10, 90.0) if wide else (3, 400.0)
    return FrameObservation.from_dict(
        {
            "frame_index": index,
            "timestamp_sec": index / 25.0,
            "scene_cut": scene_cut,
            "objects": [_player(f"p{slot}", height) for slot in range(count)],
        }
    )


def _detection(second: float, action: str, team: str | None, score: float = 0.6, segment: int = 0) -> TaadDetection:
    frame = round(second * 25)
    return TaadDetection(
        frame_index=frame,
        timestamp_sec=frame / 25.0,
        action=action,
        score=score,
        track_id=f"t{frame}",
        team=team,
        pitch_x=80.0,
        pitch_y=34.0,
        segment_index=segment,
    )


class WideViewSegmentTests(unittest.TestCase):
    def test_close_ups_split_wide_runs_and_short_runs_are_dropped(self) -> None:
        pattern = [True] * 125 + [False] * 75 + [True] * 50 + [False] * 50 + [True] * 100
        frames = [_frame(index, wide=wide) for index, wide in enumerate(pattern)]

        segments = wide_view_segments(frames, frame_height=1080)

        self.assertEqual([(item.start_index, item.end_index) for item in segments], [(0, 124), (300, 399)])

    def test_scene_cut_frame_breaks_a_wide_run(self) -> None:
        frames = [_frame(index, wide=True, scene_cut=index == 100) for index in range(200)]

        segments = wide_view_segments(frames, frame_height=1080)

        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0].end_index, 100)
        self.assertGreater(segments[1].start_index, 100)


class HighlightAttackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = [_frame(index, wide=True) for index in range(400)]
        self.segment = WideSegment(start_index=0, end_index=399, start_sec=0.0, end_sec=399 / 25.0)

    def test_segment_becomes_one_attack_despite_jersey_flicker(self) -> None:
        detections = [
            _detection(1.0, "drive", "team_a"),
            _detection(3.0, "pass", "team_b", score=0.3),
            _detection(5.0, "cross", "team_a"),
            _detection(6.0, "block", "team_b"),
            _detection(8.0, "shot", "team_b", score=0.5),
            _detection(9.0, "pass", "team_a"),
        ]

        highlights = highlight_attack_events(detections, [self.segment], self.frames, match_id="clip")
        decoded = decode_highlight_attacks(highlights, match_id="clip", fps=25.0)

        self.assertEqual(highlights[0].team, "team_a")
        self.assertEqual(highlights[0].terminal.timestamp_sec, 8.0)
        self.assertEqual(len(decoded.attacks), 1)
        attack = decoded.attacks[0]
        self.assertEqual(attack.team_id, "team_a")
        self.assertEqual(attack.terminal.terminal_type.value, "shot_attempt")
        self.assertEqual(attack.possession.start_sec, 0.0)
        self.assertEqual(attack.possession.end_sec, 8.0)
        self.assertNotIn(9.0, [action.timestamp_sec for action in attack.possession.actions])

    def test_attacking_block_after_last_shot_is_the_finish(self) -> None:
        detections = [
            _detection(1.0, "drive", "team_b"),
            _detection(4.0, "shot", "team_a", score=0.5),
            _detection(6.0, "block", "team_b", score=0.6),
        ]

        highlights = highlight_attack_events(detections, [self.segment], self.frames, match_id="clip")

        self.assertEqual(highlights[0].team, "team_b")
        self.assertEqual(highlights[0].terminal.timestamp_sec, 6.0)

    def test_keeper_action_right_after_the_shot_is_not_the_finish(self) -> None:
        detections = [
            _detection(2.0, "drive", "team_a"),
            _detection(8.0, "shot", "team_a", score=0.5),
            _detection(8.4, "shot", "team_b", score=0.7),
        ]

        highlights = highlight_attack_events(detections, [self.segment], self.frames, match_id="clip")

        self.assertEqual(highlights[0].terminal.timestamp_sec, 8.0)
        self.assertEqual(highlights[0].events[-1].timestamp, 8.0)

    def test_track_mostly_seen_as_goalkeeper_is_not_the_shooter(self) -> None:
        frames = [
            FrameObservation.from_dict(
                {
                    "frame_index": index,
                    "timestamp_sec": index / 25.0,
                    "objects": [
                        *(_player(f"p{slot}", 90.0) for slot in range(9)),
                        {**_player("keeper", 90.0), "object_type": "goalkeeper"},
                    ],
                }
            )
            for index in range(400)
        ]
        shot = _detection(8.0, "shot", "team_a", score=0.5)
        keeper_block = replace(_detection(8.3, "block", "team_a", score=0.8), track_id="keeper")

        highlights = highlight_attack_events(
            [_detection(2.0, "drive", "team_a"), shot, keeper_block],
            [self.segment],
            frames,
            match_id="clip",
        )

        self.assertEqual(highlights[0].terminal, shot)

    def test_header_terminal_is_a_headed_attempt(self) -> None:
        detections = [_detection(1.0, "cross", "team_a"), _detection(2.0, "header", "team_a")]

        highlights = highlight_attack_events(detections, [self.segment], self.frames, match_id="clip")
        decoded = decode_highlight_attacks(highlights, match_id="clip", fps=25.0)

        self.assertEqual(decoded.attacks[0].terminal.terminal_type.value, "headed_attempt")

    def test_segment_without_attempt_yields_no_attack(self) -> None:
        detections = [_detection(1.0, "drive", "team_a"), _detection(2.0, "pass", "team_a")]

        highlights = highlight_attack_events(detections, [self.segment], self.frames, match_id="clip")
        decoded = decode_highlight_attacks(highlights, match_id="clip", fps=25.0)

        self.assertEqual(highlights[0].reason, "no_shot_or_header")
        self.assertEqual(decoded.attacks, ())

    def test_attack_on_left_goal_is_mirrored_like_footpass(self) -> None:
        detections = [
            replace(_detection(1.0, "drive", "team_a"), pitch_x=40.0),
            replace(_detection(3.0, "tackle", "team_b"), pitch_x=20.0),
            replace(_detection(5.0, "shot", "team_a"), pitch_x=12.0, pitch_y=30.0),
        ]

        highlights = highlight_attack_events(detections, [self.segment], self.frames, match_id="clip")
        decoded = decode_highlight_attacks(highlights, match_id="clip", fps=25.0)

        self.assertIs(highlights[0].left_to_right, False)
        self.assertEqual(highlights[0].team_directions(), {"team_a": False, "team_b": True})
        by_type = {event.event_type: event for event in highlights[0].events}
        self.assertEqual(by_type["shot"].x, 93.0)
        self.assertEqual(by_type["shot"].y, 30.0)
        self.assertEqual(by_type["drive"].x, 65.0)
        self.assertEqual(by_type["tackle"].x, 20.0)
        terminal = decoded.attacks[0].possession.actions[-1]
        self.assertIs(terminal.raw_left_to_right, False)
        self.assertEqual(terminal.x, 93.0)


class WeakFinishFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        def frame(index: int) -> FrameObservation:
            objects: list[dict[str, object]] = [_player(f"p{slot}", 90.0) for slot in range(8)]
            objects.append(
                {
                    **_player("striker", 90.0),
                    "team": "team_a",
                    "pitch_x": 96.0,
                    "pitch_y": 29.0,
                    "bbox": {"x1": 500, "y1": 400, "x2": 530, "y2": 490},
                }
            )
            objects.append(
                {
                    **_player("keeper", 90.0),
                    "team": "team_b",
                    "pitch_x": 99.0,
                    "pitch_y": 31.0,
                    "bbox": {"x1": 700, "y1": 380, "x2": 730, "y2": 470},
                }
            )
            if index == 245:
                objects.append(
                    {
                        "object_type": "ball",
                        "confidence": 0.6,
                        "bbox": {"x1": 520, "y1": 470, "x2": 530, "y2": 480},
                    }
                )
            return FrameObservation.from_dict(
                {"frame_index": index, "timestamp_sec": index / 25.0, "objects": objects}
            )

        self.frames = [frame(index) for index in range(400)]
        self.segment = WideSegment(start_index=0, end_index=399, start_sec=0.0, end_sec=399 / 25.0)
        self.detections = [
            _detection(1.0, "drive", "team_a"),
            replace(_detection(5.0, "header", "team_a"), pitch_x=60.0),
        ]
        self.weak = replace(
            _detection(10.0, "shot", None, score=0.2),
            track_id="keeper",
            pitch_x=99.0,
            pitch_y=31.0,
        )

    def test_weak_shot_replaces_a_midfield_terminal_and_goes_to_the_ball_carrier(self) -> None:
        highlights = highlight_attack_events(
            self.detections, [self.segment], self.frames, match_id="clip", weak_shots=[self.weak]
        )

        self.assertEqual(highlights[0].reason, "weak_finish_after_far_terminal")
        self.assertEqual(highlights[0].terminal.track_id, "striker")
        self.assertEqual(highlights[0].terminal.frame_index, 245)
        self.assertEqual(highlights[0].terminal.pitch_x, 96.0)

    def test_terminal_near_goal_ignores_weak_shots(self) -> None:
        detections = [self.detections[0], replace(self.detections[1], pitch_x=90.0)]

        highlights = highlight_attack_events(
            detections, [self.segment], self.frames, match_id="clip", weak_shots=[self.weak]
        )

        self.assertEqual(highlights[0].reason, "terminal_found")
        self.assertEqual(highlights[0].terminal.timestamp_sec, 5.0)

    def test_weak_shot_far_from_goal_is_not_used(self) -> None:
        weak = replace(self.weak, pitch_x=70.0)

        highlights = highlight_attack_events(
            self.detections, [self.segment], self.frames, match_id="clip", weak_shots=[weak]
        )

        self.assertEqual(highlights[0].reason, "terminal_found")


if __name__ == "__main__":
    unittest.main()
