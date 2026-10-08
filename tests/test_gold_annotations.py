from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.gold_annotations import (
    FROZEN_TEST_MATCH_IDS,
    GOLD_SCHEMA_VERSION,
    GoldPeriodIntervalV2,
    GoldSequenceAnnotationV2,
    append_sealed_test_annotation_jsonl,
    annotation_content_hash,
    build_gold_manifest,
    continuous_full_match_coverage_report,
    deterministic_reannotation_blocks,
    freeze_sealed_test_annotations,
    gold_progress_report,
    gold_split,
    intra_rater_report,
    period_interval_content_sha256,
    read_gold_annotations_jsonl,
    verify_sealed_test_annotations,
)
from attack_analysis.models import ATTACK_V2_RUBRIC_VERSION


def annotation(**overrides: object) -> GoldSequenceAnnotationV2:
    values: dict[str, object] = {
        "match_id": "game_6",
        "period": 1,
        "stable_team_id": "home",
        "start_frame": 100,
        "end_frame": 200,
        "start_sec": 4.0,
        "end_sec": 8.0,
        "label": "attack",
        "terminal_type": "shot_attempt",
        "terminal_frame": 200,
        "terminal_sec": 8.0,
        "terminal_cause": None,
        "shot_body_part": "right_foot",
        "shot_outcome": "on_target",
        "beneficiary_team_id": None,
        "last_touch_team_id": None,
        "replay": False,
        "completeness": "complete",
        "confidence": 0.9,
        "reason_codes": ("clear_attempt",),
        "notes": "",
        "annotator_id": "expert-1",
        "annotated_at": "2026-08-27T00:00:00Z",
    }
    values.update(overrides)
    return GoldSequenceAnnotationV2(**values).with_stable_id()


def frozen_test_fixture() -> tuple[
    tuple[GoldSequenceAnnotationV2, ...], tuple[GoldPeriodIntervalV2, ...]
]:
    labels: list[GoldSequenceAnnotationV2] = []
    intervals: list[GoldPeriodIntervalV2] = []
    for match_id in FROZEN_TEST_MATCH_IDS:
        for period in (1, 2):
            intervals.append(
                GoldPeriodIntervalV2(
                    match_id=match_id,
                    period=period,
                    start_frame=0,
                    end_frame=9,
                    start_sec=0.0,
                    end_sec=0.36,
                    fps=25.0,
                )
            )
            labels.append(
                annotation(
                    match_id=match_id,
                    period=period,
                    stable_team_id="",
                    start_frame=0,
                    end_frame=9,
                    start_sec=0.0,
                    end_sec=0.36,
                    label="dead_ball",
                    terminal_type="period_end",
                    terminal_frame=9,
                    terminal_sec=0.36,
                    shot_body_part=None,
                    shot_outcome=None,
                    reason_codes=("synthetic_period_coverage",),
                )
            )
    return tuple(labels), tuple(intervals)


class GoldAnnotationTests(unittest.TestCase):
    def test_split_is_frozen(self) -> None:
        self.assertEqual(gold_split("game_6_H1"), "train")
        self.assertEqual(gold_split(18), "development")
        self.assertEqual(gold_split("game_50"), "test")
        with self.assertRaises(ValueError):
            gold_split("game_1")

    def test_round_trip_has_stable_id_and_schema(self) -> None:
        item = annotation()
        payload = item.to_dict()
        self.assertEqual(payload["schema_version"], GOLD_SCHEMA_VERSION)
        self.assertEqual(payload["rubric_version"], ATTACK_V2_RUBRIC_VERSION)
        self.assertEqual(payload["split"], "train")
        self.assertEqual(GoldSequenceAnnotationV2.from_dict(payload), item)
        self.assertEqual(annotation_content_hash((item,)), annotation_content_hash((item,)))
        self.assertEqual(len(annotation_content_hash((item,))), 64)
        with self.assertRaisesRegex(ValueError, "rubric_version"):
            annotation(rubric_version="attack-v2-rubric-1.0").validate()

    def test_tampered_stable_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stable identity"):
            annotation(annotation_id="gold-sequence-tampered").validate()

    def test_stable_id_normalizes_half_suffix_and_uses_single_rubric_contract(self) -> None:
        plain = annotation(match_id="game_6")
        half_suffixed = annotation(match_id="game_6_H1")
        self.assertEqual(plain.annotation_id, half_suffixed.annotation_id)
        self.assertEqual(plain.rubric_version, ATTACK_V2_RUBRIC_VERSION)

    def test_strict_attack_terminal_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            annotation(terminal_type="turnover", shot_outcome=None).validate()
        with self.assertRaisesRegex(ValueError, "complete"):
            annotation(completeness="right_censored").validate()
        with self.assertRaisesRegex(ValueError, "headed_attempt"):
            annotation(terminal_type="headed_attempt", shot_body_part="right_foot").validate()

    def test_forced_out_requires_defensive_cause_and_attacking_restart(self) -> None:
        item = annotation(
            terminal_type="forced_goal_line_out",
            terminal_cause="defender_deflection_out",
            shot_body_part=None,
            shot_outcome=None,
            beneficiary_team_id="home",
            last_touch_team_id="away",
        )
        item.validate()
        with self.assertRaisesRegex(ValueError, "approved defensive cause"):
            annotation(
                terminal_type="forced_goal_line_out",
                terminal_cause="attacker_missed_cross",
                shot_body_part=None,
                shot_outcome=None,
                beneficiary_team_id="home",
                last_touch_team_id="away",
            ).validate()

    def test_manifest_discovers_exact_twelve_match_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "game_6.mp4").write_bytes(b"video-content")
            manifest = build_gold_manifest(root)
        self.assertEqual(len(manifest["matches"]), 12)
        self.assertFalse(manifest["test_labels_sealed"])
        self.assertIsNone(manifest["test_label_seal"])
        game_six = next(item for item in manifest["matches"] if item["match_id"] == "game_6")
        self.assertTrue(game_six["pilot"])
        self.assertTrue(game_six["video_exists"])
        self.assertEqual(
            game_six["video_sha256"], hashlib.sha256(b"video-content").hexdigest()
        )
        self.assertNotIn("video_size", game_six)
        self.assertNotIn("video_mtime_ns", game_six)

    def test_reannotation_blocks_are_deterministic_and_cover_target(self) -> None:
        first = deterministic_reannotation_blocks({"game_6": 1000.0})
        second = deterministic_reannotation_blocks({"game_6": 1000.0})
        self.assertEqual(first, second)
        self.assertGreaterEqual(sum(item.end_sec - item.start_sec for item in first), 200.0)

    def test_frozen_test_labels_require_final_evaluator_flag(self) -> None:
        test_item = annotation(match_id="game_0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(test_item.to_dict()) + "\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                read_gold_annotations_jsonl(path)
            self.assertEqual(len(read_gold_annotations_jsonl(path, allow_frozen_test=True)), 1)

    def test_test_annotations_use_isolated_sealed_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = append_sealed_test_annotation_jsonl(root, annotation(match_id="game_0"))
            self.assertTrue((root / ".attack-v2-sealed").exists())
            with self.assertRaises(PermissionError):
                read_gold_annotations_jsonl(path)
            self.assertEqual(len(read_gold_annotations_jsonl(path, allow_frozen_test=True)), 1)

    def test_frozen_test_hash_seal_blocks_append_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items, intervals = frozen_test_fixture()
            labels = None
            for item in items:
                labels = append_sealed_test_annotation_jsonl(root, item)
            assert labels is not None
            seal = freeze_sealed_test_annotations(root, intervals)
            self.assertTrue(seal.is_file())
            verified = verify_sealed_test_annotations(root)
            self.assertEqual(verified["state"], "frozen")
            self.assertEqual(verified["matches"], list(FROZEN_TEST_MATCH_IDS))
            self.assertEqual(len(str(verified["annotation_content_sha256"])), 64)
            self.assertEqual(len(str(verified["file_sha256"])), 64)
            self.assertEqual(
                verified["period_intervals_content_sha256"],
                period_interval_content_sha256(intervals),
            )
            with self.assertRaisesRegex(PermissionError, "already been sealed"):
                append_sealed_test_annotation_jsonl(root, annotation(match_id="game_36"))
            labels.write_text(labels.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "verification failed"):
                verify_sealed_test_annotations(root)

    def test_seal_rejects_missing_frozen_matches_and_coverage_declarations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            append_sealed_test_annotation_jsonl(root, annotation(match_id="game_0"))
            with self.assertRaisesRegex(ValueError, "exactly game_0"):
                freeze_sealed_test_annotations(root)

    def test_verified_seal_is_the_only_way_manifest_claims_test_is_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            sealed = root / "sealed"
            videos.mkdir()
            items, intervals = frozen_test_fixture()
            for item in items:
                append_sealed_test_annotation_jsonl(sealed, item)
            freeze_sealed_test_annotations(sealed, intervals)
            manifest = build_gold_manifest(videos, sealed_directory=sealed)
            self.assertTrue(manifest["test_labels_sealed"])
            self.assertEqual(manifest["test_label_seal"]["state"], "frozen")

    def test_continuous_validator_accepts_explicit_states_without_inventing_labels(self) -> None:
        declaration = GoldPeriodIntervalV2(
            match_id="game_6",
            period=1,
            start_frame=0,
            end_frame=39,
            start_sec=0.0,
            end_sec=3.9,
            fps=10.0,
        )
        labels = (
            annotation(
                start_frame=0,
                end_frame=9,
                start_sec=0.0,
                end_sec=0.9,
                label="possession",
                terminal_type="turnover",
                terminal_frame=9,
                terminal_sec=0.9,
                shot_body_part=None,
                shot_outcome=None,
            ),
            annotation(
                stable_team_id="",
                start_frame=10,
                end_frame=19,
                start_sec=1.0,
                end_sec=1.9,
                label="dead_ball",
                terminal_type="unknown",
                terminal_frame=None,
                terminal_sec=None,
                shot_body_part=None,
                shot_outcome=None,
            ),
            annotation(
                stable_team_id="",
                start_frame=20,
                end_frame=29,
                start_sec=2.0,
                end_sec=2.9,
                label="uncertain",
                terminal_type="unknown",
                terminal_frame=None,
                terminal_sec=None,
                shot_body_part=None,
                shot_outcome=None,
                completeness="uncertain",
            ),
            annotation(
                start_frame=30,
                end_frame=39,
                start_sec=3.0,
                end_sec=3.9,
                terminal_frame=39,
                terminal_sec=3.9,
            ),
        )
        report = continuous_full_match_coverage_report(labels, (declaration,))
        self.assertTrue(report["complete"])
        self.assertEqual(report["annotation_count"], 4)
        self.assertEqual(report["label_counts"]["dead_ball"], 1)

        gap_report = continuous_full_match_coverage_report(
            (labels[0], labels[2], labels[3]), (declaration,)
        )
        self.assertFalse(gap_report["complete"])
        self.assertTrue(any("gap frames 10-19" in value for value in gap_report["errors"]))

        overlapping_uncertain = annotation(
            stable_team_id="",
            start_frame=19,
            end_frame=29,
            start_sec=1.9,
            end_sec=2.9,
            label="uncertain",
            terminal_type="unknown",
            terminal_frame=None,
            terminal_sec=None,
            shot_body_part=None,
            shot_outcome=None,
            completeness="uncertain",
        )
        overlap_report = continuous_full_match_coverage_report(
            (labels[0], labels[1], overlapping_uncertain, labels[3]),
            (declaration,),
        )
        self.assertFalse(overlap_report["complete"])
        self.assertTrue(any("overlap frames 19-19" in value for value in overlap_report["errors"]))

    def test_intra_rater_report_uses_stable_sequences(self) -> None:
        first = annotation(annotation_round=1)
        second = annotation(annotation_round=2, confidence=0.8)
        report = intra_rater_report((first,), (second,))
        self.assertEqual(report["attack_label_kappa"], 1.0)
        self.assertTrue(report["meets_annotation_gate"])

    def test_intra_rater_matches_shifted_boundaries_not_only_exact_ids(self) -> None:
        first = annotation(annotation_round=1)
        second = annotation(annotation_round=2, start_frame=105, start_sec=4.2)
        self.assertNotEqual(first.annotation_id, second.annotation_id)
        report = intra_rater_report((first,), (second,))
        self.assertEqual(report["matched_sequences"], 1)
        self.assertEqual(report["median_boundary_disagreement_sec"], 0.1)

    def test_gold_progress_starts_on_development_and_keeps_p11_open(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = gold_progress_report(root)
            self.assertEqual(report["p11_status"], "open")
            self.assertFalse(report["closable_by_code"])
            self.assertEqual(report["next_step"], "annotate_development")
            self.assertEqual(report["splits"]["development"]["matches_missing"], ["game_18", "game_24", "game_47"])
            self.assertEqual(report["splits"]["test"]["state"], "not_started")

            labels = root / "data" / "attack-v2-development.jsonl"
            labels.parent.mkdir(parents=True)
            labels.write_text(
                json.dumps(annotation(match_id="game_18").to_dict(), ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            started = gold_progress_report(root)
            self.assertIn("game_18", started["splits"]["development"]["matches_with_labels"])
            self.assertEqual(
                started["splits"]["development"]["matches_missing"],
                ["game_24", "game_47"],
            )
            self.assertEqual(started["next_step"], "annotate_development")


if __name__ == "__main__":
    unittest.main()
