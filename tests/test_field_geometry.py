import unittest

from attack_analysis.field_geometry import (
    FieldGeometry,
    FieldLineConfig,
    FieldLineSegment,
    _find_intersections,
)
from attack_analysis.observations import FrameObservation


class FieldGeometryTests(unittest.TestCase):
    def test_finds_cross_intersection(self) -> None:
        horizontal = FieldLineSegment.from_endpoints(10, 50, 90, 50, 100)
        vertical = FieldLineSegment.from_endpoints(50, 10, 50, 90, 100)

        points = _find_intersections((horizontal, vertical), 100, 100)

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].kind, "cross")
        self.assertEqual((points[0].x, points[0].y), (50.0, 50.0))

    def test_frame_observation_round_trips_field_geometry(self) -> None:
        segment = FieldLineSegment.from_endpoints(1, 2, 30, 2, 100)
        geometry = FieldGeometry(
            method="test",
            frame_width=100,
            frame_height=50,
            pitch_mask_coverage=0.8,
            line_mask_coverage=0.02,
            segments=(segment,),
            intersections=(),
        )
        frame = FrameObservation.from_dict(
            {
                "frame_index": 0,
                "timestamp_sec": 0,
                "objects": [],
                "field_geometry": geometry.to_dict(),
            }
        )

        self.assertIsNotNone(frame.field_geometry)
        self.assertEqual(frame.field_geometry.segments[0].length_px, 29.0)
        self.assertEqual(frame.to_dict()["field_geometry"]["method"], "test")

    def test_rejects_invalid_line_configuration(self) -> None:
        with self.assertRaises(ValueError):
            FieldLineConfig(min_length_ratio=0).validate()


if __name__ == "__main__":
    unittest.main()
