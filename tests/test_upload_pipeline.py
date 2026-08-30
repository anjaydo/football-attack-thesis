from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.geometry import PitchProjector
from attack_analysis.upload_pipeline import run_cached_video_pipeline, vision_cache_key
from attack_analysis.vision_pipeline import VisionConfig


class UploadPipelineTests(unittest.TestCase):
    def test_cache_key_changes_with_model_or_frame_limit(self) -> None:
        first = vision_cache_key("abc", VisionConfig(max_frames=10))
        second = vision_cache_key("abc", VisionConfig(max_frames=20))
        third = vision_cache_key("abc", VisionConfig(max_frames=10, detection_model="other.pt"))
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_cache_key_never_depends_on_secret_values(self) -> None:
        config = VisionConfig(max_frames=10)
        first = vision_cache_key("abc", config, {"api_key": "first-secret", "split": "dev"})
        second = vision_cache_key("abc", config, {"api_key": "second-secret", "split": "dev"})
        changed_public_value = vision_cache_key(
            "abc", config, {"api_key": "first-secret", "split": "test"}
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed_public_value)

    def test_reuses_complete_manifest(self) -> None:
        calls = 0

        def runner(input_path: Path, run_directory: Path, config: VisionConfig) -> dict[str, object]:
            nonlocal calls
            calls += 1
            run_directory.mkdir(parents=True)
            annotated = run_directory / "annotated.mp4"
            annotated.touch()
            payload = {
                "status": "vision_complete",
                "outputs": {
                    "annotated_video": str(annotated.resolve()),
                    "manifest": str((run_directory / "manifest.json").resolve()),
                },
            }
            (run_directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            return payload

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.touch()
            config = VisionConfig(max_frames=10)
            first = run_cached_video_pipeline(
                video, root / "cache", "abc", config, runner=runner, browser_encoder=lambda path: path
            )
            second = run_cached_video_pipeline(
                video, root / "cache", "abc", config, runner=runner, browser_encoder=lambda path: path
            )
        self.assertEqual(calls, 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["manifest_version"], "attack-analysis-manifest-v2")
        self.assertEqual(first["source_tier"], "video_only")
        self.assertEqual(first["provenance_status"], "validated")
        self.assertIn("content_tree_sha256", first["provenance"])
        self.assertIn("output_sha256", first)

    def test_exposes_browser_encoded_video_and_preserves_raw_artifact(self) -> None:
        def runner(input_path: Path, run_directory: Path, config: VisionConfig) -> dict[str, object]:
            run_directory.mkdir(parents=True)
            raw = run_directory / "annotated.mp4"
            raw.touch()
            return {"outputs": {"annotated_video": str(raw.resolve())}}

        def browser_encoder(raw: Path) -> Path:
            encoded = raw.with_name("annotated-browser.mp4")
            encoded.touch()
            return encoded

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.touch()
            manifest = run_cached_video_pipeline(
                video,
                root / "cache",
                "abc",
                VisionConfig(max_frames=10),
                runner=runner,
                browser_encoder=browser_encoder,
            )
        self.assertTrue(str(manifest["outputs"]["annotated_video"]).endswith("annotated-browser.mp4"))
        self.assertTrue(str(manifest["outputs"]["annotated_video_raw"]).endswith("annotated.mp4"))

    def test_rejects_tampered_cached_output_and_reruns(self) -> None:
        calls = 0

        def runner(input_path: Path, run_directory: Path, config: VisionConfig) -> dict[str, object]:
            nonlocal calls
            calls += 1
            run_directory.mkdir(parents=True, exist_ok=True)
            annotated = run_directory / "annotated.mp4"
            annotated.write_bytes(f"run-{calls}".encode())
            return {"outputs": {"annotated_video": str(annotated.resolve())}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            content_root = root / "source"
            content_root.mkdir()
            (content_root / "pipeline.py").write_text("VERSION = 1\n", encoding="utf-8")
            first = run_cached_video_pipeline(
                video,
                root / "cache",
                "caller-id",
                VisionConfig(max_frames=10),
                runner=runner,
                browser_encoder=lambda path: path,
                content_root=content_root,
            )
            Path(first["outputs"]["annotated_video"]).write_bytes(b"tampered")
            second = run_cached_video_pipeline(
                video,
                root / "cache",
                "caller-id",
                VisionConfig(max_frames=10),
                runner=runner,
                browser_encoder=lambda path: path,
                content_root=content_root,
            )

        self.assertEqual(calls, 2)
        self.assertFalse(second["cache_hit"])

    def test_rejects_explicitly_legacy_cached_manifest_even_when_hashes_match(self) -> None:
        calls = 0

        def runner(input_path: Path, run_directory: Path, config: VisionConfig) -> dict[str, object]:
            nonlocal calls
            calls += 1
            run_directory.mkdir(parents=True, exist_ok=True)
            annotated = run_directory / "annotated.mp4"
            annotated.write_bytes(f"run-{calls}".encode())
            return {"outputs": {"annotated_video": str(annotated.resolve())}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            cache_root = root / "cache"
            content_root = root / "source"
            content_root.mkdir()
            (content_root / "pipeline.py").write_text("VERSION = 1\n", encoding="utf-8")
            config = VisionConfig(max_frames=10)
            first = run_cached_video_pipeline(
                video,
                cache_root,
                "caller-id",
                config,
                runner=runner,
                browser_encoder=lambda path: path,
                content_root=content_root,
            )
            manifest_path = Path(first["outputs"]["manifest"])
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            persisted["provenance_status"] = "legacy_unvalidated"
            persisted["legacy_unvalidated"] = True
            manifest_path.write_text(json.dumps(persisted), encoding="utf-8")

            second = run_cached_video_pipeline(
                video,
                cache_root,
                "caller-id",
                config,
                runner=runner,
                browser_encoder=lambda path: path,
                content_root=content_root,
            )

        self.assertEqual(calls, 2)
        self.assertFalse(second["cache_hit"])
        self.assertEqual(second["provenance_status"], "validated")

    def test_forwards_assisted_source_tier_and_dataset_paths_to_real_runner_contract(self) -> None:
        captured: dict[str, object] = {}

        def runner(
            input_path: Path,
            run_directory: Path,
            config: VisionConfig,
            *,
            projector: object,
            source_tier: str,
            dataset_paths: tuple[Path, ...],
        ) -> dict[str, object]:
            captured["source_tier"] = source_tier
            captured["dataset_paths"] = dataset_paths
            captured["projector"] = projector
            run_directory.mkdir(parents=True, exist_ok=True)
            annotated = run_directory / "annotated.mp4"
            annotated.touch()
            return {"outputs": {"annotated_video": str(annotated.resolve())}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            playbyplay = root / "playbyplay.json"
            tactical = root / "tactical.h5"
            playbyplay.write_text("{}", encoding="utf-8")
            tactical.write_bytes(b"tactical")
            projector = PitchProjector.from_points(
                [(0, 0), (1, 0), (1, 1), (0, 1)],
                [(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            manifest = run_cached_video_pipeline(
                video,
                root / "cache",
                "caller-id",
                VisionConfig(max_frames=10),
                projector=projector,
                source_tier="predicted_actions_oracle_tactical",
                dataset_paths=(playbyplay, tactical),
                runner=runner,
                browser_encoder=lambda path: path,
            )

        self.assertEqual(captured["source_tier"], "predicted_actions_oracle_tactical")
        self.assertEqual(captured["dataset_paths"], (playbyplay, tactical))
        self.assertIs(captured["projector"], projector)
        self.assertEqual(manifest["source_tier"], "predicted_actions_oracle_tactical")

    def test_cached_pipeline_rejects_assisted_projector_as_video_only(self) -> None:
        class AssistedProjector:
            report = {"source": "tactical.h5"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.touch()
            with self.assertRaisesRegex(ValueError, "cannot be rendered as video_only"):
                run_cached_video_pipeline(
                    video,
                    root / "cache",
                    "caller-id",
                    VisionConfig(max_frames=10),
                    projector=AssistedProjector(),
                )

    def test_assisted_cache_is_invalidated_when_playbyplay_content_changes(self) -> None:
        calls = 0

        def runner(
            input_path: Path,
            run_directory: Path,
            config: VisionConfig,
            **options: object,
        ) -> dict[str, object]:
            nonlocal calls
            calls += 1
            run_directory.mkdir(parents=True, exist_ok=True)
            annotated = run_directory / "annotated.mp4"
            annotated.write_bytes(f"run-{calls}".encode())
            return {"outputs": {"annotated_video": str(annotated.resolve())}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            playbyplay = root / "playbyplay.json"
            tactical = root / "tactical.h5"
            playbyplay.write_text('{"version": 1}', encoding="utf-8")
            tactical.write_bytes(b"tactical")
            projector = PitchProjector.from_points(
                [(0, 0), (1, 0), (1, 1), (0, 1)],
                [(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            first = run_cached_video_pipeline(
                video,
                root / "cache",
                "caller-id",
                VisionConfig(max_frames=10),
                projector=projector,
                source_tier="predicted_actions_oracle_tactical",
                dataset_paths=(playbyplay, tactical),
                runner=runner,
                browser_encoder=lambda path: path,
            )
            playbyplay.write_text('{"version": 2}', encoding="utf-8")
            second = run_cached_video_pipeline(
                video,
                root / "cache",
                "caller-id",
                VisionConfig(max_frames=10),
                projector=projector,
                source_tier="predicted_actions_oracle_tactical",
                dataset_paths=(playbyplay, tactical),
                runner=runner,
                browser_encoder=lambda path: path,
            )

        self.assertEqual(calls, 2)
        self.assertNotEqual(first["cache_key"], second["cache_key"])
        self.assertNotEqual(
            first["provenance"]["dataset_sha256"],
            second["provenance"]["dataset_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
