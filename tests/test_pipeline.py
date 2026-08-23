import tempfile
import unittest
from pathlib import Path
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

    def test_pipeline_runs_events_and_attacks_with_calibration(self) -> None:
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
            self.assertGreater(manifest["attack_count"], 0)
            self.assertTrue((root / "run" / "events.json").is_file())
            self.assertTrue((root / "run" / "attacks.json").is_file())

