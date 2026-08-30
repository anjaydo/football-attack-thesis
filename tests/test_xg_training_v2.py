from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from attack_analysis.xg_training_v2 import XGTrainingV2Config, train_xg_v2_models
from attack_analysis.xg_v2 import (
    XG_V2_FEATURE_COLUMNS,
    load_trained_xg_v2_model,
    normalize_xg_v2_features,
)


class XGTrainingV2Tests(unittest.TestCase):
    def test_grouped_nested_training_keeps_outer_test_out_of_selection(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        rows = []
        for match in range(10):
            for index in range(12):
                goal = int(index % 6 == 0)
                x = 96.0 if goal else 72.0 + (index % 4)
                features = normalize_xg_v2_features(
                    {
                        "x_shooter": x,
                        "y_shooter": 34.0,
                        "body_part": "Right Foot",
                        "under_pressure": False,
                        "play_pattern": "Regular Play",
                        "shot_type": "Open Play",
                        "goalkeeper_visible": True,
                        "x_goalkeeper": 104.0,
                        "y_goalkeeper": 34.0,
                        "defenders_visible": True,
                        "observed_players": 10,
                        "foe_players": 5,
                        "ally_players": 4,
                        "defending_players": 1,
                    }
                )
                rows.append({"match_id": f"m{match}", **features, "goal": goal})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "xg.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["match_id", *XG_V2_FEATURE_COLUMNS, "goal"],
                )
                writer.writeheader()
                writer.writerows(rows)
            output = root / "xg-v2.joblib"
            report = train_xg_v2_models(
                csv_path,
                output,
                config=XGTrainingV2Config(inner_folds=2, bootstrap_samples=50),
            )
            output_exists = output.is_file()
            with self.assertRaisesRegex(PermissionError, "not domain-validated"):
                load_trained_xg_v2_model(output)
            trained = load_trained_xg_v2_model(output, allow_research_only=True)
            probability = trained.predict(
                {"x_shooter": 92.0, "y_shooter": 34.0, "body_part": "Right Foot"}
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_trained_xg_v2_model(
                    output,
                    expected_sha256="0" * 64,
                    allow_research_only=True,
                )
        self.assertEqual(report["schema_version"], "xg-training-v2")
        self.assertTrue(output_exists)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        split = report["split"]
        self.assertFalse(set(split["train_groups"]) & set(split["test_groups"]))
        self.assertEqual(report["domain_validation"]["footpass_frozen_test"], "not_run")

    def test_rejects_missing_explicit_feature_schema(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("match_id,goal\nm1,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                train_xg_v2_models(path, Path(directory) / "model.joblib")


if __name__ == "__main__":
    unittest.main()
