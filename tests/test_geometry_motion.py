import unittest

from attack_analysis.geometry import PitchProjector, compute_homography
from attack_analysis.motion import SpeedEstimator
from attack_analysis.vision_pipeline import VisionConfig


class GeometryAndMotionTests(unittest.TestCase):
    def test_maps_image_rectangle_to_pitch(self) -> None:
        projector = PitchProjector.from_points(
            [(0, 0), (100, 0), (100, 50), (0, 50)],
            [(0, 0), (105, 0), (105, 68), (0, 68)],
        )
        x, y = projector.project((50, 25))
        self.assertAlmostEqual(x, 52.5)
        self.assertAlmostEqual(y, 34.0)

    def test_rejects_degenerate_homography(self) -> None:
        with self.assertRaises(ValueError):
            compute_homography(
                [(0, 0), (1, 0), (2, 0), (3, 0)],
                [(0, 0), (1, 0), (2, 0), (3, 0)],
            )

    def test_estimates_track_speed(self) -> None:
        estimator = SpeedEstimator(smoothing=0.5)
        self.assertEqual(estimator.update("player:1", (0, 0), 0), 0.0)
        self.assertAlmostEqual(estimator.update("player:1", (6, 8), 2), 5.0)
        self.assertAlmostEqual(estimator.update("player:1", (12, 16), 4), 5.0)

    def test_projects_speed_to_pitch_units(self) -> None:
        projector = PitchProjector.from_points(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            [(0, 0), (10, 0), (10, 10), (0, 10)],
        )
        estimator = SpeedEstimator()
        estimator.update("ball:1", (0, 0), 0, projector.project)
        speed_mps = estimator.update("ball:1", (100, 0), 2, projector.project)
        self.assertAlmostEqual(speed_mps, 5.0)

    def test_validates_vision_config(self) -> None:
        with self.assertRaises(ValueError):
            VisionConfig(ball_confidence=1.5).validate()

