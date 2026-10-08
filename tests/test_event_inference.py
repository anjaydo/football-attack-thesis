import unittest

from attack_analysis.event_inference import (
    EventInferenceConfig,
    enrich_shots_with_trained_xg,
    infer_events_from_observations,
)
from attack_analysis.models import Event
from attack_analysis.observations import FrameObservation
from attack_analysis.segmentation import segment_attacks


def tracked_object(
    track_id: str,
    object_type: str,
    x: float,
    y: float,
    team: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "track_id": track_id,
        "object_type": object_type,
        "confidence": 0.9,
        "bbox": {"x1": 10, "y1": 10, "x2": 20, "y2": 30},
        "pitch_x": x,
        "pitch_y": y,
    }
    if team:
        result["team"] = team
    return result


def _video_frame(
    index: int,
    owner_id: str | None,
    team: str | None,
    ball_x: float,
    timestamp: float | None = None,
) -> FrameObservation:
    objects: list[dict[str, object]] = [tracked_object("ball-1", "ball", ball_x, 34)]
    if owner_id is not None and team is not None:
        objects.insert(0, tracked_object(owner_id, "player", ball_x + 0.4, 34, team))
    return FrameObservation.from_dict(
        {
            "frame_index": index,
            "timestamp_sec": index / 25.0 if timestamp is None else timestamp,
            "objects": objects,
        }
    )


def _video_touch(indexes: range, owner_id: str, team: str, ball_x: float) -> list[FrameObservation]:
    return [_video_frame(index, owner_id, team, ball_x) for index in indexes]


def frame(
    index: int,
    owner_id: str,
    team: str,
    ball_x: float,
    scene_cut: bool = False,
) -> FrameObservation:
    return FrameObservation.from_dict(
        {
            "frame_index": index,
            "timestamp_sec": float(index),
            "scene_cut": scene_cut,
            "objects": [
                tracked_object(owner_id, "player", ball_x + 0.5, 10, team),
                tracked_object("ball-1", "ball", ball_x, 10),
            ],
        }
    )


class EventInferenceTests(unittest.TestCase):
    def test_infers_carry_pass_and_turnover(self) -> None:
        frames = [
            frame(0, "a1", "team_a", 10),
            frame(1, "a1", "team_a", 14),
            frame(2, "a2", "team_a", 30),
            frame(3, "b1", "team_b", 35),
        ]
        events = infer_events_from_observations(frames)
        event_types = [event.event_type for event in events]

        self.assertEqual(
            event_types,
            ["possession_start", "carry", "pass", "turnover", "possession_start"],
        )
        self.assertEqual(events[-1].team, "team_b")
        self.assertEqual(events[-1].x, 70.0)
        attacks = segment_attacks(events)
        self.assertEqual(len(attacks), 2)
        self.assertEqual(attacks[0].team, "team_a")
        self.assertEqual(attacks[1].team, "team_b")

    def test_requires_calibrated_team_aware_tracks(self) -> None:
        uncalibrated = FrameObservation.from_dict(
            {
                "frame_index": 0,
                "timestamp_sec": 0,
                "objects": [
                    {
                        "track_id": "1",
                        "object_type": "ball",
                        "confidence": 0.9,
                        "bbox": {"x1": 1, "y1": 1, "x2": 2, "y2": 2},
                    }
                ],
            }
        )
        with self.assertRaises(ValueError):
            infer_events_from_observations([uncalibrated])

    def test_validates_team_direction(self) -> None:
        with self.assertRaises(ValueError):
            EventInferenceConfig(attack_directions={"team_a": "north"})

    def test_does_not_infer_turnover_across_scene_cut(self) -> None:
        events = infer_events_from_observations(
            [
                frame(0, "a1", "team_a", 20),
                frame(1, "b1", "team_b", 80, scene_cut=True),
            ]
        )
        self.assertNotIn("turnover", [event.event_type for event in events])
        self.assertEqual(
            [event.event_type for event in events],
            ["possession_start", "possession_start"],
        )

    def test_rejects_physically_impossible_transfer(self) -> None:
        with self.assertRaises(ValueError):
            infer_events_from_observations(
                [
                    frame(0, "a1", "team_a", 5),
                    FrameObservation.from_dict(
                        {
                            **frame(1, "a2", "team_a", 100).to_dict(),
                            "timestamp_sec": 0.1,
                        }
                    ),
                ]
            )

    def test_forward_ball_trajectory_stays_unverified_and_has_no_xg(self) -> None:
        owner_frame = frame(0, "a1", "team_a", 80)
        flight_frames = [
            FrameObservation.from_dict(
                {
                    "frame_index": 1,
                    "timestamp_sec": 0.4,
                    "objects": [tracked_object("ball-1", "ball", 92, 10)],
                }
            ),
            FrameObservation.from_dict(
                {
                    "frame_index": 2,
                    "timestamp_sec": 0.8,
                    "objects": [tracked_object("ball-1", "ball", 104, 10)],
                }
            ),
        ]
        events = infer_events_from_observations([owner_frame, *flight_frames])
        candidates = [event for event in events if event.event_type == "shot_candidate"]
        self.assertEqual(len(candidates), 1)
        self.assertIsNone(candidates[0].xg)
        self.assertIsNone(candidates[0].xg_source)
        self.assertEqual(candidates[0].outcome, "unverified")

    def test_video_rate_teleport_does_not_become_a_shot(self) -> None:
        frames = [
            *_video_touch(range(8), "a1", "team_a", 82.0),
            _video_frame(8, None, None, 104.0),
        ]
        events = infer_events_from_observations(frames)
        self.assertNotIn("shot_candidate", [event.event_type for event in events])

    def test_video_rate_owner_flicker_is_not_a_turnover(self) -> None:
        frames = [
            *_video_touch(range(8), "a1", "team_a", 40.0),
            _video_frame(8, "b1", "team_b", 40.2),
            *_video_touch(range(9, 17), "a1", "team_a", 41.0),
        ]
        events = infer_events_from_observations(frames)
        self.assertNotIn("turnover", [event.event_type for event in events])
        self.assertNotIn("pass", [event.event_type for event in events])
        self.assertTrue(all(event.team == "team_a" for event in events))

    def test_video_rate_ball_reaching_the_goal_is_a_shot_candidate(self) -> None:
        frames = [
            *_video_touch(range(6), "a1", "team_a", 80.0),
            _video_frame(6, None, None, 86.0, timestamp=0.45),
            _video_frame(7, None, None, 94.0, timestamp=0.75),
            _video_frame(8, None, None, 102.0, timestamp=1.05),
            _video_frame(9, None, None, 104.0, timestamp=1.30),
        ]
        events = infer_events_from_observations(frames)
        candidates = [event for event in events if event.event_type == "shot_candidate"]
        self.assertEqual(len(candidates), 1)
        self.assertGreaterEqual(candidates[0].end_x or 0, 96.0)

    def test_trained_xg_receives_calibrated_video_context(self) -> None:
        class FakeModel:
            source = "trained_xg:test:xg-v1"

            def predict(self, features):
                self.features = features
                return 0.42

        observation = FrameObservation.from_dict(
            {
                "frame_index": 25,
                "timestamp_sec": 1.0,
                "objects": [
                    tracked_object("a1", "player", 90, 34, "team_a"),
                    tracked_object("b1", "player", 91, 34, "team_b"),
                    tracked_object("gk", "goalkeeper", 103, 34, "team_b"),
                ],
            }
        )
        model = FakeModel()
        enriched = enrich_shots_with_trained_xg(
            (Event(1.0, "shot", "team_a", x=90, y=34, xg=0.1),),
            (observation,),
            model,
        )

        self.assertEqual(enriched[0].xg, 0.42)
        self.assertEqual(enriched[0].xg_source, "trained_xg:test:xg-v1")
        self.assertEqual(enriched[0].xg_inputs["foe_players"], 2)
        self.assertEqual(enriched[0].xg_inputs["defending_players"], 1)
        self.assertEqual(model.features["x_goalkeeper"], 103)
