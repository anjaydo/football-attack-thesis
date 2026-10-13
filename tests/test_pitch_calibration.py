import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from attack_analysis.pitch_calibration import (
    CalibrationFitConfig,
    FrameFit,
    KeypointPitchProjector,
    PITCH_KEYPOINTS,
    PitchCalibrationTrack,
    fit_frame_homography,
    smooth_calibrations,
)

FRAME = (1920.0, 1080.0)


def _broadcast_view() -> np.ndarray:
    """Pitch metres -> pixels for a main-camera view of the right third."""

    pitch = np.float32([[60, -5], [110, -5], [110, 73], [60, 73]])
    image = np.float32([[100, 150], [1800, 150], [2400, 1100], [-500, 1100]])
    if cv2 is None:
        return np.eye(3)
    return cv2.getPerspectiveTransform(pitch, image)


PITCH_TO_IMAGE = _broadcast_view()


def _image_points(matrix: np.ndarray = PITCH_TO_IMAGE) -> np.ndarray:
    pitch = np.asarray(PITCH_KEYPOINTS, dtype=float)
    projected = np.hstack([pitch, np.ones((len(pitch), 1))]) @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


def _visible(ids: list[int]) -> np.ndarray:
    confidence = np.zeros(len(PITCH_KEYPOINTS))
    confidence[[item - 1 for item in ids]] = 0.9
    return confidence


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class FitFrameHomographyTests(unittest.TestCase):
    def test_right_penalty_area_landmarks_recover_pitch_positions(self) -> None:
        fit = fit_frame_homography(
            _image_points(),
            _visible([18, 19, 20, 21, 22, 23, 24, 26, 27]),
            frame_size=FRAME,
        )

        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertGreaterEqual(fit.inliers, 8)
        spot = _image_points()[21]
        projected = fit.as_array() @ np.array([spot[0], spot[1], 1.0])
        self.assertAlmostEqual(projected[0] / projected[2], 94.0, places=3)
        self.assertAlmostEqual(projected[1] / projected[2], 34.0, places=3)

    def test_noisy_fit_is_repeatable(self) -> None:
        rng = np.random.default_rng(3)
        points = _image_points() + rng.normal(0.0, 2.0, size=(len(PITCH_KEYPOINTS), 2))
        visible = _visible([18, 19, 20, 21, 22, 23, 24, 26, 27])

        first = fit_frame_homography(points, visible, frame_size=FRAME)
        second = fit_frame_homography(points, visible, frame_size=FRAME)

        assert first is not None and second is not None
        self.assertEqual(first.matrix, second.matrix)

    def test_four_landmarks_are_not_enough(self) -> None:
        fit = fit_frame_homography(
            _image_points(),
            _visible([18, 21, 23, 24]),
            frame_size=FRAME,
        )

        self.assertIsNone(fit)

    def test_one_mislabelled_landmark_is_rejected_as_outlier(self) -> None:
        points = _image_points().copy()
        points[25] = points[0]
        fit = fit_frame_homography(
            points,
            _visible([18, 19, 20, 21, 22, 23, 24, 26, 27]),
            frame_size=FRAME,
        )

        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertNotIn(26, fit.keypoint_ids)

    def test_collinear_landmarks_are_refused(self) -> None:
        fit = fit_frame_homography(
            _image_points(),
            _visible([25, 26, 27, 28, 29, 30]),
            frame_size=FRAME,
            config=CalibrationFitConfig(min_inliers=5),
        )

        self.assertIsNone(fit)


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class SmoothCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        fit = fit_frame_homography(
            _image_points(),
            _visible([18, 19, 20, 21, 22, 23, 24, 26, 27]),
            frame_size=FRAME,
        )
        assert fit is not None
        self.fit = fit

    def test_short_gap_is_bridged_and_long_gap_is_left_empty(self) -> None:
        fits: list[FrameFit | None] = [self.fit] * 5 + [None] * 10 + [self.fit] * 5 + [None] * 60

        frames = smooth_calibrations(fits, frame_size=FRAME, window=2, max_gap=25)

        self.assertEqual(frames[0].source, "keypoints")
        self.assertEqual(frames[10].source, "keypoints_bridged")
        self.assertLess(frames[10].confidence, frames[0].confidence)
        self.assertIsNone(frames[79])

    def test_scene_cut_stops_bridging(self) -> None:
        fits: list[FrameFit | None] = [self.fit] * 5 + [None] * 10

        frames = smooth_calibrations(fits, frame_size=FRAME, cut_frames={8}, window=1, max_gap=25)

        self.assertIsNotNone(frames[7])
        self.assertIsNone(frames[8])

    def test_projector_reads_frames_relative_to_start(self) -> None:
        frames = smooth_calibrations([self.fit] * 3, frame_size=FRAME)
        track = PitchCalibrationTrack(frame_size=FRAME, start_frame=100, frames=frames)
        restored = PitchCalibrationTrack.from_dict(track.to_dict())
        projector = KeypointPitchProjector(restored)

        spot = _image_points()[21]
        x, y = projector.for_frame(101).project((float(spot[0]), float(spot[1])))

        self.assertAlmostEqual(x, 94.0, places=2)
        self.assertAlmostEqual(y, 34.0, places=2)
        self.assertIsNone(projector.for_frame(5))
        self.assertIsNone(getattr(projector, "report", None))


if __name__ == "__main__":
    unittest.main()
