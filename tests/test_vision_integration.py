import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from attack_analysis.quality import read_observations_jsonl
from attack_analysis.vision_pipeline import VisionConfig, annotate_video


class FakeBoxes:
    def __init__(self, xyxy, confidence, track_ids) -> None:
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(confidence, dtype=float)
        self.id = np.asarray(track_ids, dtype=float)

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeKeypoints:
    def __init__(self, count: int) -> None:
        person = [[20 + index, 20 + index, 0.9] for index in range(17)]
        self.data = np.asarray([person] * count, dtype=float)


class FakeResult:
    def __init__(self, pose: bool) -> None:
        if pose:
            self.boxes = FakeBoxes([[10, 10, 30, 50]], [0.9], [1])
            self.keypoints = FakeKeypoints(1)
        else:
            self.boxes = FakeBoxes([[40, 40, 46, 46]], [0.8], [1])
            self.keypoints = None


class FakeYOLO:
    def __init__(self, model_name: str) -> None:
        self.pose = "pose" in model_name

    def predict(self, frame, **kwargs):
        return [FakeResult(True)]

    def track(self, frame, **kwargs):
        return [FakeResult(kwargs.get("classes") == [0])]


class FakeCapture:
    def __init__(self) -> None:
        self.frames = [np.zeros((72, 128, 3), dtype=np.uint8) for _ in range(2)]
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {1: 25.0, 2: 128.0, 3: 72.0}[property_id]

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self) -> None:
        pass

    def set(self, property_id: int, value: float) -> bool:
        return True


class FakeWriter:
    def __init__(self) -> None:
        self.frames_written = 0

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        self.frames_written += 1

    def release(self) -> None:
        pass


class FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_POS_FRAMES = 4
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0
    COLOR_BGR2LAB = 0

    def VideoCapture(self, path):
        return FakeCapture()

    def VideoWriter(self, *args):
        return FakeWriter()

    @staticmethod
    def VideoWriter_fourcc(*args):
        return 0

    @staticmethod
    def getTextSize(text, font, scale, thickness):
        return (len(text) * 5, 10), 2

    @staticmethod
    def rectangle(*args, **kwargs):
        pass

    @staticmethod
    def putText(*args, **kwargs):
        pass

    @staticmethod
    def line(*args, **kwargs):
        pass

    @staticmethod
    def circle(*args, **kwargs):
        pass

    @staticmethod
    def cvtColor(crop, code):
        return crop


class VisionIntegrationTests(unittest.TestCase):
    def test_video_loop_writes_schema_compatible_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.touch()
            observations_path = root / "observations.jsonl"

            with patch(
                "attack_analysis.vision_pipeline._load_dependencies",
                return_value=(FakeCV2(), FakeYOLO),
            ):
                summary = annotate_video(
                    input_path,
                    root / "output.mp4",
                    VisionConfig(team_classification=False),
                    observations_path=observations_path,
                )

            frames = read_observations_jsonl(observations_path)
            self.assertEqual(summary.frames_processed, 2)
            self.assertEqual(summary.unique_player_tracks, 1)
            self.assertEqual(summary.unique_ball_tracks, 1)
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].objects[0].object_type, "player")
            self.assertEqual(frames[0].objects[1].object_type, "ball")
            self.assertEqual(frames[0].objects[0].bbox.x1, 10.0)

    def test_time_window_preserves_global_frame_and_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.touch()
            observations_path = root / "observations.jsonl"

            with patch(
                "attack_analysis.vision_pipeline._load_dependencies",
                return_value=(FakeCV2(), FakeYOLO),
            ):
                summary = annotate_video(
                    input_path,
                    root / "output.mp4",
                    VisionConfig(
                        team_classification=False,
                        start_frame=250,
                        max_frames=1,
                    ),
                    observations_path=observations_path,
                )

            frames = read_observations_jsonl(observations_path)
            self.assertEqual(summary.start_frame, 250)
            self.assertEqual(summary.end_frame, 250)
            self.assertEqual(summary.frames_processed, 1)
            self.assertEqual(frames[0].frame_index, 250)
            self.assertEqual(frames[0].timestamp_sec, 10.0)
