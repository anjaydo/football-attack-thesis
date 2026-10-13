import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

from attack_analysis.footpass_dst import (
    MISSING_VALUE,
    assign_role_slots,
    build_dst_inputs,
    dst_segment_detections,
    footpass_direction_flag,
    slot_index,
)
from attack_analysis.footpass_video_events import SegmentTaadOutput
from attack_analysis.observations import FrameObservation


def _object(track: str, team: str | None, x: float, y: float, kind: str = "player") -> dict[str, object]:
    return {
        "track_id": track,
        "object_type": kind,
        "confidence": 0.9,
        "team": team,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 30},
        "pitch_x": x,
        "pitch_y": y,
    }


def _frames(count: int = 40) -> list[FrameObservation]:
    frames = []
    for index in range(count):
        drift = index * 0.1
        frames.append(
            FrameObservation.from_dict(
                {
                    "frame_index": 500 + index,
                    "timestamp_sec": (500 + index) / 25.0,
                    "objects": [
                        # team_a attacks the left goal (x=0): striker deep, full back wide.
                        _object("striker", "team_a", 15.0 - drift, 34.0),
                        _object("left_back", "team_a", 55.0, 60.0),
                        _object("right_back", "team_a", 55.0, 8.0),
                        # team_b defends the left goal.
                        _object("keeper", "team_b", 3.0, 34.0, kind="goalkeeper"),
                        _object("centre_back", "team_b", 12.0, 30.0),
                    ],
                }
            )
        )
    return frames


class RoleSlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = _frames()
        self.directions = {"team_a": False, "team_b": True}
        self.assignments = {
            item.track_id: item
            for item in assign_role_slots(
                self.frames,
                self.directions,
                goalkeeper_share={"keeper": 1.0},
            )
        }

    def test_footpass_flag_zero_attacks_the_right_goal(self) -> None:
        self.assertEqual(footpass_direction_flag(True), 0)
        self.assertEqual(footpass_direction_flag(False), 1)
        self.assertEqual(slot_index(1, 1), 13)

    def test_keeper_takes_the_goalkeeper_slot_of_the_side_defending_that_goal(self) -> None:
        self.assertEqual(self.assignments["keeper"].slot, slot_index(0, 1))

    def test_outfield_roles_follow_attack_view_positions(self) -> None:
        self.assertEqual(self.assignments["striker"].slot, slot_index(1, 12))
        self.assertEqual(self.assignments["left_back"].flag, 1)
        self.assertIn(self.assignments["left_back"].role_id, {2, 10})
        self.assertIn(self.assignments["right_back"].role_id, {11, 13})

    @unittest.skipIf(np is None, "numpy is not installed")
    def test_team_vector_uses_footpass_units_and_missing_value(self) -> None:
        taad = SegmentTaadOutput(
            probabilities={},
            logits={"striker": np.full((9, 40), 2.0)},
            present={"striker": np.ones(40, dtype=bool)},
            empty_logits=np.arange(9, dtype=float),
        )
        teamvec, logits = build_dst_inputs(
            self.frames, tuple(self.assignments.values()), taad
        )

        striker = self.assignments["striker"].slot
        self.assertEqual(teamvec.shape, (5, 26, 40))
        self.assertAlmostEqual(float(teamvec[0, striker, 0]), 15.0 / 105.0, places=5)
        self.assertAlmostEqual(float(teamvec[1, striker, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(teamvec[2, striker, 20]), -0.1 / 105.0 * 223.0, places=3)
        self.assertEqual(float(teamvec[4, striker, 5]), 1.0)
        unused = slot_index(1, 3)
        self.assertEqual(float(teamvec[0, unused, 0]), MISSING_VALUE)
        self.assertEqual(float(teamvec[4, unused, 0]), MISSING_VALUE)
        self.assertEqual(float(logits[5, striker, 10]), 2.0)
        self.assertEqual(float(logits[5, unused, 10]), 5.0)

    def test_dst_event_maps_back_to_track_and_team(self) -> None:
        striker = self.assignments["striker"]
        detections = dst_segment_detections(
            [(30, striker.slot, 5, 0.8)],
            self.frames,
            tuple(self.assignments.values()),
            team_by_flag={1: "team_a", 0: "team_b"},
            segment_index=2,
        )

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.action, "shot")
        self.assertEqual(detection.track_id, "striker")
        self.assertEqual(detection.team, "team_a")
        self.assertEqual(detection.frame_index, 530)
        self.assertAlmostEqual(detection.pitch_x, 12.0, places=3)
        self.assertEqual(detection.segment_index, 2)


if __name__ == "__main__":
    unittest.main()
