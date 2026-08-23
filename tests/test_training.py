import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attack_analysis.training import train_ball_model


class FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.arguments = None

    def train(self, **kwargs) -> None:
        self.arguments = kwargs


class TrainingTests(unittest.TestCase):
    def test_builds_reproducible_ball_training_run(self) -> None:
        models = []

        def factory(name: str) -> FakeModel:
            model = FakeModel(name)
            models.append(model)
            return model

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names:\n  0: ball\n", encoding="utf-8")
            with patch("attack_analysis.training._load_yolo", return_value=factory):
                report = train_ball_model(
                    data_yaml, root / "runs", epochs=2, image_size=640, batch_size=2
                )

        self.assertEqual(models[0].name, "yolo26s.pt")
        self.assertEqual(models[0].arguments["classes"], [0])
        self.assertEqual(models[0].arguments["single_cls"], True)
        self.assertTrue(str(report["best_weights"]).endswith("weights\\best.pt"))

    def test_requires_dataset_yaml(self) -> None:
        with self.assertRaises(FileNotFoundError):
            train_ball_model(Path("missing.yaml"), Path("runs"))


if __name__ == "__main__":
    unittest.main()
