import unittest

import numpy as np

from attack_analysis.scene import SceneCutDetector


class SceneCutTests(unittest.TestCase):
    def test_detects_abrupt_change(self) -> None:
        detector = SceneCutDetector(threshold=0.3, sample_step=1)
        dark = np.zeros((10, 10, 3), dtype=np.uint8)
        bright = np.full((10, 10, 3), 255, dtype=np.uint8)
        self.assertFalse(detector.update(dark))
        self.assertFalse(detector.update(dark))
        self.assertTrue(detector.update(bright))

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaises(ValueError):
            SceneCutDetector(threshold=0)

