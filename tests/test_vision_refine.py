import unittest

import numpy as np

from attack_analysis.vision_pipeline import (
    TileBallAssociator,
    VisionConfig,
    _pose_on_player_crops,
    _tiled_ball_detections,
    iter_tiles,
    merge_ball_detections,
)


class _Boxes:
    def __init__(self, xyxy, confidence) -> None:
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(confidence, dtype=float)
        self.id = None

    def __len__(self) -> int:
        return len(self.xyxy)


class _Keypoints:
    def __init__(self, points) -> None:
        self.data = np.asarray([points], dtype=float)


class _Result:
    def __init__(self, xyxy, confidence, keypoints=None) -> None:
        self.boxes = _Boxes(xyxy, confidence)
        self.keypoints = None if keypoints is None else _Keypoints(keypoints)


class _ListModel:
    def __init__(self, result_factory) -> None:
        self._factory = result_factory
        self.calls = 0

    def predict(self, source, **kwargs):
        self.calls += 1
        images = source if isinstance(source, list) else [source]
        return [self._factory(image) for image in images]


class VisionRefineTests(unittest.TestCase):
    def test_tiles_cover_full_hd_without_a_dense_grid(self) -> None:
        windows = iter_tiles(1920, 1080, 480, 0.1)
        self.assertLessEqual(len(windows), 12)
        self.assertGreaterEqual(len(windows), 4)
        self.assertTrue(any(window[0] == 0 and window[1] == 0 for window in windows))
        self.assertTrue(any(window[2] == 1920 for window in windows))
        self.assertTrue(any(window[3] == 1080 for window in windows))

    def test_small_frame_skips_tiling(self) -> None:
        self.assertEqual(iter_tiles(128, 72, 480, 0.1), [])

    def test_merge_keeps_track_id_and_drops_duplicate_or_huge_tile_boxes(self) -> None:
        merged = merge_ball_detections(
            [((10.0, 10.0, 16.0, 16.0), 0.9, 3)],
            [
                ((10.0, 10.0, 16.0, 16.0), 0.4, None),
                ((40.0, 40.0, 48.0, 48.0), 0.55, None),
                ((0.0, 0.0, 200.0, 90.0), 0.99, None),
                ((70.0, 10.0, 100.0, 14.0), 0.8, None),
            ],
            max_side=80.0,
        )
        self.assertEqual(
            [(row[0][0], row[2]) for row in merged],
            [(10.0, 3), (40.0, None)],
        )

    def test_tile_associator_reuses_id_and_does_not_invent_boxes(self) -> None:
        associator = TileBallAssociator()
        first = associator.assign([(0.0, 0.0, 10.0, 10.0)])
        second = associator.assign([(2.0, 2.0, 12.0, 12.0)])
        self.assertEqual(first, second)
        self.assertEqual(associator.assign([]), [])
        third = associator.assign([(2.0, 2.0, 12.0, 12.0)])
        self.assertNotEqual(first[0], third[0])

    def test_tiled_detections_are_shifted_into_frame_coordinates(self) -> None:
        def factory(_image):
            return _Result([[4, 6, 12, 14]], [0.6])

        rows = _tiled_ball_detections(
            _ListModel(factory),
            np.zeros((120, 200, 3), dtype=np.uint8),
            class_id=32,
            confidence=0.15,
            image_size=960,
            device="cpu",
            tile_size=80,
            overlap=0.0,
            bounds=None,
        )
        origins = {(row[0][0] - 4.0, row[0][1] - 6.0) for row in rows}
        self.assertIn((0.0, 0.0), origins)
        self.assertTrue(any(origin[0] > 0 or origin[1] > 0 for origin in origins))

    def test_pose_crop_shifts_keypoints_back_to_the_frame(self) -> None:
        points = [[8, 9, 0.95] for _ in range(17)]

        def factory(_image):
            return _Result([[16, 20, 70, 120]], [0.8], points)

        refined = _pose_on_player_crops(
            _ListModel(factory),
            np.zeros((200, 200, 3), dtype=np.uint8),
            [(40.0, 30.0, 90.0, 150.0)],
            confidence=0.12,
            image_size=320,
            device="cpu",
            keypoint_confidence=0.25,
            pad_ratio=0.35,
        )
        self.assertIsNotNone(refined[0])
        assert refined[0] is not None
        self.assertEqual(len(refined[0]), 17)
        self.assertGreater(refined[0][0][0], 8.0)
        self.assertGreaterEqual(refined[0][0][2], 0.95)

    def test_pose_crop_skips_boxes_too_small_for_a_skeleton(self) -> None:
        model = _ListModel(lambda _image: _Result([[0, 0, 10, 10]], [0.9], [[1, 1, 0.9]] * 17))
        refined = _pose_on_player_crops(
            model,
            np.zeros((80, 80, 3), dtype=np.uint8),
            [(4.0, 4.0, 16.0, 20.0)],
            confidence=0.12,
            image_size=320,
            device="cpu",
            keypoint_confidence=0.25,
            pad_ratio=0.35,
        )
        self.assertEqual(refined, [None])
        self.assertEqual(model.calls, 0)

    def test_ball_tiles_are_off_by_default(self) -> None:
        self.assertEqual(VisionConfig().ball_tile_size, 0)

    def test_config_rejects_invalid_tile_overlap(self) -> None:
        with self.assertRaises(ValueError):
            VisionConfig(ball_tile_overlap=0.9).validate()
