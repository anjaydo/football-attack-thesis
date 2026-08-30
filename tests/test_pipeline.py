import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attack_analysis.geometry import PitchProjector
from attack_analysis.pipeline import run_video_pipeline
from attack_analysis.vision_pipeline import VisionRunSummary


ROOT = Path(__file__).resolve().parents[1]


def fake_summary(input_path: Path, output_path: Path, observations_path: Path) -> VisionRunSummary:
    return VisionRunSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        observations_path=str(observations_path),
        frames_processed=4,
        fps=25,
        width=1280,
        height=720,
        unique_player_tracks=3,
        unique_ball_tracks=1,
        speed_unit="km/h",
        pose_model="fake-pose",
        detection_model="fake-detect",
        roi_enabled=False,
        team_classification_enabled=True,
        scene_cuts_detected=0,
    )


class PipelineTests(unittest.TestCase):
    def test_pipeline_skips_events_without_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.touch()

            def fake_annotate(input_file, output_file, config, projector, observations, roi):
                output_file.touch()
                observations.write_text(
                    (ROOT / "examples" / "observations_quality_sample.jsonl").read_text(
                        encoding="utf-8"
                    ),
                    encoding="utf-8",
                )
                return fake_summary(input_file, output_file, observations)

            with patch("attack_analysis.pipeline.annotate_video", side_effect=fake_annotate):
                manifest = run_video_pipeline(input_path, root / "run")

            self.assertEqual(manifest["status"], "vision_complete")
            self.assertEqual(manifest["analysis_status"], "skipped")
            self.assertEqual(
                manifest["analysis_skip_reason"],
                "calibration_required_for_event_inference",
            )
            self.assertTrue((root / "run" / "manifest.json").is_file())

    def test_pipeline_emits_v2_possessions_without_promoting_trajectory_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.touch()

            def fake_annotate(input_file, output_file, config, projector, observations, roi):
                output_file.touch()
                observations.write_text(
                    (ROOT / "examples" / "observations_calibrated_sample.jsonl").read_text(
                        encoding="utf-8"
                    ),
                    encoding="utf-8",
                )
                return fake_summary(input_file, output_file, observations)

            projector = PitchProjector.from_points(
                [(0, 0), (1, 0), (1, 1), (0, 1)],
                [(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            with patch("attack_analysis.pipeline.annotate_video", side_effect=fake_annotate):
                manifest = run_video_pipeline(
                    input_path,
                    root / "run",
                    projector=projector,
                )

            self.assertEqual(manifest["status"], "analysis_complete")
            self.assertEqual(manifest["analysis_status"], "complete")
            self.assertGreater(manifest["event_count"], 0)
            self.assertGreater(manifest["possession_count"], 0)
            self.assertEqual(manifest["attack_count"], 0)
            self.assertEqual(manifest["release_status"], "research_only")
            self.assertIn("content_tree_sha256", manifest["provenance"])
            self.assertTrue((root / "run" / "events.json").is_file())
            self.assertTrue((root / "run" / "sequences-v2.json").is_file())
            self.assertTrue((root / "run" / "possession-candidates-v1.json").is_file())
            sequences = json.loads(
                (root / "run" / "sequences-v2.json").read_text(encoding="utf-8")
            )
            self.assertTrue(sequences["possession_values"])
            self.assertTrue(
                all("peak_progression_m" not in item for item in sequences["possession_values"])
            )
            self.assertTrue(
                all("box_entry" not in item for item in sequences["possession_values"])
            )

    def test_assisted_source_tier_and_dataset_hash_propagate_to_every_v2_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.write_bytes(b"video")
            playbyplay = root / "playbyplay.json"
            tactical = root / "tactical.h5"
            playbyplay.write_text('{"events": {}}', encoding="utf-8")
            tactical.write_bytes(b"tactical")

            def fake_annotate(input_file, output_file, config, projector, observations, roi):
                output_file.touch()
                observations.write_text(
                    (ROOT / "examples" / "observations_calibrated_sample.jsonl").read_text(
                        encoding="utf-8"
                    ),
                    encoding="utf-8",
                )
                return fake_summary(input_file, output_file, observations)

            projector = PitchProjector.from_points(
                [(0, 0), (1, 0), (1, 1), (0, 1)],
                [(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            with patch("attack_analysis.pipeline.annotate_video", side_effect=fake_annotate):
                manifest = run_video_pipeline(
                    input_path,
                    root / "run",
                    projector=projector,
                    source_tier="predicted_actions_oracle_tactical",
                    dataset_paths=(playbyplay, tactical),
                )

            self.assertEqual(manifest["source_tier"], "predicted_actions_oracle_tactical")
            self.assertNotEqual(manifest["provenance"]["dataset_sha256"], "")
            events = json.loads((root / "run" / "events.json").read_text(encoding="utf-8"))
            sequences = json.loads(
                (root / "run" / "sequences-v2.json").read_text(encoding="utf-8")
            )
            self.assertEqual(events["source_tier"], "predicted_actions_oracle_tactical")
            self.assertTrue(events["events"])
            self.assertTrue(
                all(
                    item["source_tier"] == "predicted_actions_oracle_tactical"
                    for item in events["events"]
                )
            )
            self.assertEqual(sequences["source_tier"], "predicted_actions_oracle_tactical")
            self.assertTrue(sequences["observed_actions"])
            self.assertTrue(
                all(
                    item["source_tier"] == "predicted_actions_oracle_tactical"
                    for item in sequences["observed_actions"]
                )
            )
            self.assertTrue(
                all(
                    item["source_tier"] == "predicted_actions_oracle_tactical"
                    for item in sequences["possessions"]
                )
            )

    def test_assisted_projector_cannot_be_rendered_as_video_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.touch()
            tactical = root / "tactical.h5"
            tactical.touch()
            assisted_projector = SimpleNamespace(
                report=SimpleNamespace(source=str(tactical.resolve()))
            )
            with self.assertRaisesRegex(ValueError, "cannot be rendered as video_only"):
                run_video_pipeline(
                    input_path,
                    root / "run",
                    projector=assisted_projector,
                )

    def test_ball_id_churn_blocks_event_inference_even_when_continuity_is_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            input_path.touch()

            def fake_annotate(input_file, output_file, config, projector, observations, roi):
                output_file.touch()
                rows = []
                source_rows = (
                    ROOT / "examples" / "observations_calibrated_sample.jsonl"
                ).read_text(encoding="utf-8").splitlines()
                for index, line in enumerate(source_rows):
                    payload = json.loads(line)
                    for item in payload["objects"]:
                        if item["object_type"] == "ball":
                            item["track_id"] = f"ball-{index}"
                    rows.append(json.dumps(payload))
                observations.write_text("\n".join(rows) + "\n", encoding="utf-8")
                return fake_summary(input_file, output_file, observations)

            projector = PitchProjector.from_points(
                [(0, 0), (1, 0), (1, 1), (0, 1)],
                [(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            with patch("attack_analysis.pipeline.annotate_video", side_effect=fake_annotate):
                manifest = run_video_pipeline(input_path, root / "run", projector=projector)

            self.assertEqual(manifest["quality"]["ball_track_continuity"], 1.0)
            self.assertFalse(manifest["event_quality_gates"]["ball_tracking"])
            self.assertEqual(
                manifest["analysis_skip_reason"],
                "ball_track_fragmentation_proxy_above_0.3",
            )
