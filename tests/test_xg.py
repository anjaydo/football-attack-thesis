import unittest

from attack_analysis.xg import heuristic_xg, shot_angle, shot_distance


class ExpectedGoalsTests(unittest.TestCase):
    def test_close_central_shot_has_more_xg_than_far_shot(self) -> None:
        self.assertGreater(heuristic_xg((94, 34)), heuristic_xg((75, 34)))

    def test_central_shot_has_more_xg_than_wide_shot(self) -> None:
        self.assertGreater(heuristic_xg((90, 34)), heuristic_xg((90, 5)))

    def test_geometry_is_positive(self) -> None:
        self.assertGreater(shot_distance((90, 34)), 0)
        self.assertGreater(shot_angle((90, 34)), 0)
        self.assertTrue(0 < heuristic_xg((90, 34)) < 1)

