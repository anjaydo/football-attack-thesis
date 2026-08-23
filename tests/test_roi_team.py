import unittest

import numpy as np

from attack_analysis.roi import PolygonROI
from attack_analysis.team import OnlineTeamClassifier


class RoiTests(unittest.TestCase):
    def test_polygon_contains_points_and_boundary(self) -> None:
        roi = PolygonROI(((0, 0), (10, 0), (10, 10), (0, 10)))
        self.assertTrue(roi.contains((5, 5)))
        self.assertTrue(roi.contains((0, 5)))
        self.assertFalse(roi.contains((11, 5)))

    def test_rejects_degenerate_polygon(self) -> None:
        with self.assertRaises(ValueError):
            PolygonROI(((0, 0), (1, 0), (2, 0)))


class TeamClassifierTests(unittest.TestCase):
    def test_clusters_two_jersey_color_groups(self) -> None:
        classifier = OnlineTeamClassifier(min_tracks=4)
        labels = classifier.update_frame(
            {
                "1": np.asarray([0.10, 0.12, 0.11]),
                "2": np.asarray([0.12, 0.11, 0.10]),
                "3": np.asarray([0.88, 0.90, 0.89]),
                "4": np.asarray([0.90, 0.88, 0.91]),
            }
        )
        self.assertEqual(labels["1"], labels["2"])
        self.assertEqual(labels["3"], labels["4"])
        self.assertNotEqual(labels["1"], labels["3"])

    def test_waits_for_enough_tracks(self) -> None:
        classifier = OnlineTeamClassifier(min_tracks=4)
        labels = classifier.update_frame(
            {"1": np.asarray([0.1, 0.2, 0.3]), "2": np.asarray([0.9, 0.8, 0.7])}
        )
        self.assertIsNone(labels["1"])
        self.assertIsNone(labels["2"])

