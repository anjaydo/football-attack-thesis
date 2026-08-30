import tempfile
import unittest
from pathlib import Path

import numpy as np

from attack_analysis.dynamic_calibration import (
    DynamicCalibrationConfig,
    build_footpass_dynamic_projector,
)


class DynamicCalibrationTests(unittest.TestCase):
    def test_assisted_defaults_match_locked_acceptance_gate(self) -> None:
        config = DynamicCalibrationConfig()
        self.assertEqual(config.max_median_error_m, 0.5)
        self.assertEqual(config.max_p90_error_m, 1.5)
        self.assertEqual(config.min_frame_coverage, 0.9)
        self.assertEqual(config.min_correspondences, 6)

    def test_builds_frame_varying_projector_with_holdout_metrics(self) -> None:
        try:
            import h5py
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("dynamic calibration extras are not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tactical.h5"
            rows = []
            pitch_points = [
                (10, 8),
                (25, 10),
                (40, 15),
                (60, 12),
                (85, 9),
                (15, 30),
                (35, 35),
                (55, 28),
                (75, 38),
                (95, 32),
                (25, 58),
                (80, 55),
            ]
            for frame in range(100, 111):
                pan = (frame - 100) * 0.4
                for player_id, (pitch_x, pitch_y) in enumerate(pitch_points, start=1):
                    image_x = 80.0 + 4.2 * pitch_x + pan
                    image_bottom = 35.0 + 4.0 * pitch_y
                    row = np.zeros(14, dtype=np.float32)
                    row[0] = frame
                    row[1] = player_id
                    row[5] = pitch_x / 105.0
                    row[6] = pitch_y / 68.0
                    row[9] = image_x * 3.0 - 15.0
                    row[10] = image_bottom * 1080.0 / 352.0 - 60.0
                    row[11] = 30.0
                    row[12] = 60.0
                    rows.append(row)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("game_1_H1", data=np.asarray(rows))

            dynamic = build_footpass_dynamic_projector(
                path,
                "game_1_H1",
                start_frame=100,
                end_frame=110,
                video_width=640,
                video_height=352,
                config=DynamicCalibrationConfig(sample_stride=1),
            )

            self.assertEqual(dynamic.report.status, "pass")
            self.assertEqual(dynamic.report.projected_frame_coverage, 1.0)
            self.assertIsNotNone(dynamic.report.median_holdout_error_m)
            self.assertLess(float(dynamic.report.median_holdout_error_m), 0.01)
            projected = dynamic.for_frame(105).project((80.0 + 4.2 * 60 + 2.0, 35.0 + 4.0 * 12))
            self.assertAlmostEqual(projected[0], 60.0, places=2)
            self.assertAlmostEqual(projected[1], 12.0, places=2)

    def test_bridges_only_bounded_gaps_and_reports_temporal_support(self) -> None:
        try:
            import h5py
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("dynamic calibration extras are not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tactical.h5"
            pitch_points = [
                (12, 8),
                (28, 15),
                (48, 10),
                (72, 22),
                (90, 14),
                (30, 52),
                (78, 55),
            ]
            rows = []
            for frame in range(100, 111):
                for player_id, (pitch_x, pitch_y) in enumerate(pitch_points, start=1):
                    row = np.zeros(14, dtype=np.float32)
                    row[0] = frame
                    row[1] = player_id
                    row[5] = pitch_x / 105.0
                    row[6] = pitch_y / 68.0
                    if 104 <= frame <= 107:
                        row[9:13] = np.nan
                    else:
                        image_x = 70.0 + 4.3 * pitch_x + (frame - 100) * 0.3
                        image_bottom = 30.0 + 4.1 * pitch_y
                        row[9] = image_x * 3.0 - 15.0
                        row[10] = image_bottom * 1080.0 / 352.0 - 60.0
                        row[11] = 30.0
                        row[12] = 60.0
                    rows.append(row)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("game_1_H1", data=np.asarray(rows))

            dynamic = build_footpass_dynamic_projector(
                path,
                "game_1_H1",
                start_frame=100,
                end_frame=110,
                video_width=640,
                video_height=352,
                config=DynamicCalibrationConfig(
                    min_correspondences=7,
                    max_interpolation_gap_frames=4,
                    max_projector_gap_frames=0,
                ),
            )

            self.assertEqual(dynamic.report.status, "pass")
            self.assertEqual(dynamic.report.projected_frame_coverage, 1.0)
            self.assertEqual(dynamic.report.direct_frame_coverage, 0.6364)
            self.assertEqual(dynamic.report.interpolated_frame_coverage, 0.3636)
            self.assertEqual(dynamic.report.extrapolated_frame_coverage, 0.0)
            self.assertEqual(dynamic.report.unsupported_frames, 0)
            self.assertEqual(dynamic.support_for_frame(103), "direct")
            self.assertEqual(dynamic.support_for_frame(105), "interpolated")

            strict = build_footpass_dynamic_projector(
                path,
                "game_1_H1",
                start_frame=100,
                end_frame=110,
                video_width=640,
                video_height=352,
                config=DynamicCalibrationConfig(
                    min_correspondences=7,
                    max_interpolation_gap_frames=3,
                    max_projector_gap_frames=0,
                ),
            )
            self.assertEqual(strict.report.status, "pass")
            self.assertEqual(strict.report.ineligible_frames, 4)
            self.assertEqual(strict.support_for_frame(105), "unsupported")
            self.assertIsNone(strict.for_frame(105))

    def test_replay_holes_are_excluded_from_coverage_gate(self) -> None:
        try:
            import h5py
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("dynamic calibration extras are not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tactical.h5"
            pitch_points = [
                (12, 8),
                (28, 15),
                (48, 10),
                (72, 22),
                (90, 14),
                (30, 52),
                (78, 55),
            ]
            rows = []
            for frame in range(100, 151):
                for player_id, (pitch_x, pitch_y) in enumerate(pitch_points, start=1):
                    row = np.zeros(14, dtype=np.float32)
                    row[0] = frame
                    row[1] = player_id
                    row[5] = pitch_x / 105.0
                    row[6] = pitch_y / 68.0
                    if 110 <= frame <= 139:
                        row[9:13] = np.nan
                    else:
                        image_x = 70.0 + 4.3 * pitch_x
                        image_bottom = 30.0 + 4.1 * pitch_y
                        row[9] = image_x * 3.0 - 15.0
                        row[10] = image_bottom * 1080.0 / 352.0 - 60.0
                        row[11] = 30.0
                        row[12] = 60.0
                    rows.append(row)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("game_1_H1", data=np.asarray(rows))

            dynamic = build_footpass_dynamic_projector(
                path,
                "game_1_H1",
                start_frame=100,
                end_frame=150,
                video_width=640,
                video_height=352,
            )
            self.assertEqual(dynamic.report.status, "pass")
            self.assertGreaterEqual(dynamic.report.ineligible_frames, 10)
            self.assertGreaterEqual(dynamic.report.projected_frame_coverage, 0.9)
            self.assertIsNone(dynamic.for_frame(125))
            self.assertIsNotNone(dynamic.for_frame(105))
            self.assertIsNotNone(dynamic.for_frame(145))

    def test_rejects_invalid_frame_window(self) -> None:
        with self.assertRaises(ValueError):
            build_footpass_dynamic_projector(
                Path("missing.h5"),
                "game_1_H1",
                start_frame=10,
                end_frame=9,
                video_width=640,
                video_height=352,
            )
