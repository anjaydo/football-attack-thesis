import unittest

from attack_analysis.event_inference import (
    EventInferenceConfig,
    infer_events_from_observations,
)
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
        events = infer_events_from_observations(
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
        self.assertNotIn("pass", [event.event_type for event in events])

    def test_infers_shot_and_geometry_xg_from_ball_trajectory(self) -> None:
        owner_frame = frame(0, "a1", "team_a", 80)
        flight_frames = [
            FrameObservation.from_dict(
                {
                    "frame_index": 1,
                    "timestamp_sec": 0.5,
                    "objects": [tracked_object("ball-1", "ball", 90, 34)],
                }
            ),
            FrameObservation.from_dict(
                {
                    "frame_index": 2,
                    "timestamp_sec": 1.0,
                    "objects": [tracked_object("ball-1", "ball", 104, 34)],
                }
            ),
        ]
        events = infer_events_from_observations([owner_frame, *flight_frames])
        shots = [event for event in events if event.event_type == "shot"]
        self.assertEqual(len(shots), 1)
        self.assertIsNotNone(shots[0].xg)
        self.assertGreater(shots[0].xg, 0)
        self.assertEqual(shots[0].xg_source, "heuristic_geometry_v1")
