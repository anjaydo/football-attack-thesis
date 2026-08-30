from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.provenance import (
    MANIFEST_VERSION,
    StaleArtifactError,
    build_provenance,
    canonical_json,
    compare_manifest_provenance,
    content_sha256,
    file_sha256,
    invalidate_artifact_tree,
    mark_legacy_manifest,
    require_current_manifest,
    tree_sha256,
)


class ProvenanceTests(unittest.TestCase):
    def test_canonical_hash_is_order_independent_and_omits_secrets(self) -> None:
        first = {"split": "dev", "nested": {"b": 2, "a": 1}, "api_key": "secret-one"}
        second = {"api_key": "secret-two", "nested": {"a": 1, "b": 2}, "split": "dev"}
        self.assertEqual(content_sha256(first), content_sha256(second))
        self.assertNotIn("secret", canonical_json(first))

    def test_file_and_tree_hashes_change_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            first_file = file_sha256(source)
            first_tree = tree_sha256(root, suffixes={".py"})
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second_file = file_sha256(source)
            second_tree = tree_sha256(root, suffixes={".py"})
        self.assertNotEqual(first_file, second_file)
        self.assertNotEqual(first_tree, second_tree)

    def test_legacy_and_stale_manifests_never_compare_current(self) -> None:
        expected = build_provenance(
            source_tier="video_only",
            video_sha256="video",
            config={"frames": 10},
            model_fingerprints={"detector": "model"},
            schema_versions={"events": "v2"},
        )
        legacy = mark_legacy_manifest({"status": "complete"})
        legacy_comparison = compare_manifest_provenance(
            legacy,
            expected_provenance=expected,
            source_tier="video_only",
        )
        self.assertEqual(legacy["provenance_status"], "legacy_unvalidated")
        self.assertEqual(legacy_comparison.status, "legacy_unvalidated")

        explicitly_legacy = {
            "manifest_version": MANIFEST_VERSION,
            "source_tier": "video_only",
            "provenance": expected,
            "provenance_status": "legacy_unvalidated",
            "legacy_unvalidated": True,
        }
        explicitly_legacy_comparison = compare_manifest_provenance(
            explicitly_legacy,
            expected_provenance=expected,
            source_tier="video_only",
        )
        self.assertEqual(explicitly_legacy_comparison.status, "legacy_unvalidated")
        self.assertFalse(explicitly_legacy_comparison.is_current)

        explicitly_stale = {
            "manifest_version": MANIFEST_VERSION,
            "source_tier": "video_only",
            "provenance": expected,
            "provenance_status": "stale",
        }
        explicitly_stale_comparison = compare_manifest_provenance(
            explicitly_stale,
            expected_provenance=expected,
            source_tier="video_only",
        )
        self.assertEqual(explicitly_stale_comparison.status, "stale")
        self.assertFalse(explicitly_stale_comparison.is_current)

        current = {
            "manifest_version": MANIFEST_VERSION,
            "source_tier": "video_only",
            "provenance": expected,
        }
        self.assertTrue(
            compare_manifest_provenance(
                current,
                expected_provenance=expected,
                source_tier="video_only",
            ).is_current
        )
        stale = {**current, "provenance": {**expected, "config_sha256": "changed"}}
        with self.assertRaises(StaleArtifactError):
            require_current_manifest(
                stale,
                expected_provenance=expected,
                source_tier="video_only",
            )

    def test_invalidation_marks_manifests_and_preserves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "old-cache"
            run = root / "run-1"
            run.mkdir(parents=True)
            artifact = run / "observations.jsonl"
            artifact.write_text("{}\n", encoding="utf-8")
            manifest = run / "manifest.json"
            manifest.write_text(
                '{"provenance_status":"validated","stale_fields":[]}',
                encoding="utf-8",
            )

            report = invalidate_artifact_tree(
                root,
                reason="geometry contract changed",
                replacement="runs/verification-v2",
            )
            updated = json.loads(manifest.read_text(encoding="utf-8"))

            self.assertEqual(updated["provenance_status"], "stale")
            self.assertIn("content_tree_sha256", updated["stale_fields"])
            self.assertTrue(artifact.is_file())
            self.assertTrue((root / "STALE.json").is_file())
            self.assertEqual(report["manifests_updated"], ["run-1/manifest.json"])


if __name__ == "__main__":
    unittest.main()
