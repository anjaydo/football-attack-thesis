import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.yolo_dataset import export_soccernet_ball_dataset


def make_sequence(root: Path, name: str) -> Path:
    sequence = root / name
    images = sequence / "img1"
    images.mkdir(parents=True)
    (images / "000001.jpg").write_bytes(b"image")
    payload = {
        "categories": [{"id": 4, "name": "ball"}],
        "images": [
            {
                "image_id": "1",
                "file_name": "000001.jpg",
                "width": 100,
                "height": 50,
            }
        ],
        "annotations": [
            {
                "image_id": "1",
                "category_id": 4,
                "bbox_image": {
                    "x": 40,
                    "y": 20,
                    "x_center": 45,
                    "y_center": 25,
                    "w": 10,
                    "h": 10,
                },
            }
        ],
    }
    (sequence / "Labels-GameState.json").write_text(json.dumps(payload), encoding="utf-8")
    return sequence


class YoloDatasetTests(unittest.TestCase):
    def test_exports_sequence_split_without_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = make_sequence(root, "SNGS-TRAIN")
            val = make_sequence(root, "SNGS-VAL")
            report = export_soccernet_ball_dataset(
                root / "dataset", [train], [val], link_mode="copy"
            )

            label = root / "dataset/labels/train/SNGS-TRAIN_000001.txt"
            self.assertEqual(label.read_text(encoding="ascii"), "0 0.45000000 0.50000000 0.10000000 0.20000000\n")
            self.assertEqual(report["train"]["ball_boxes"], 1)
            self.assertTrue((root / "dataset/data.yaml").exists())

    def test_rejects_same_sequence_in_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sequence = make_sequence(root, "SNGS-SAME")
            with self.assertRaises(ValueError):
                export_soccernet_ball_dataset(root / "dataset", [sequence], [sequence])


if __name__ == "__main__":
    unittest.main()
