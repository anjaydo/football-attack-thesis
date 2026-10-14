from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.action_spotting_v2 import OperatingPointMeasurement
from attack_analysis.cli import _sequence_records, main as cli_main
from attack_analysis.gold_annotations import (
    FROZEN_TEST_MATCHES,
    GoldPeriodIntervalV2,
    GoldSequenceAnnotationV2,
    append_sealed_test_annotation_jsonl,
    freeze_sealed_test_annotations,
)
from attack_analysis.models import (
    AttackSequenceV2,
    ObservedActionV2,
    PossessionSequenceV2,
    SequenceStatus,
    SourceTier,
    TerminalEventV2,
    TerminalType,
)
from attack_analysis.provenance import content_sha256
from attack_analysis.release_control import (
    FinalTestAlreadyClaimedError,
    ReleaseComponentHashes,
    ReleaseIntegrityError,
    claim_final_test_once,
    complete_final_test_claim,
    final_test_evaluation_protocol_sha256,
    final_test_ledger_status,
    freeze_attack_v2_release,
    load_frozen_release_manifest,
    path_sha256,
    write_frozen_release_manifest,
)


def component_hashes(
    seed: str = "base",
    frozen_baseline_sha256: str | None = None,
) -> ReleaseComponentHashes:
    return ReleaseComponentHashes(
        config_sha256=content_sha256({"kind": seed, "component": "config"}),
        model_sha256=content_sha256({"kind": seed, "component": "model"}),
        schema_sha256=content_sha256({"kind": seed, "component": "schema"}),
        rubric_sha256=content_sha256({"kind": seed, "component": "rubric"}),
        development_dataset_sha256=content_sha256(
            {"kind": seed, "component": "development"}
        ),
        frozen_baseline_sha256=(
            frozen_baseline_sha256
            or content_sha256({"kind": seed, "component": "frozen_baseline"})
        ),
    )


def frozen_release(seed: str = "base", frozen_baseline_sha256: str | None = None):
    return freeze_attack_v2_release(
        release_id="attack-v2-r1",
        component_hashes=component_hashes(seed, frozen_baseline_sha256),
        measurements=(
            OperatingPointMeasurement(0.30, 1.0, 2.0 / 3.0, "imbalanced"),
            OperatingPointMeasurement(0.40, 0.8, 0.8, "balanced-low"),
            OperatingPointMeasurement(0.50, 0.8, 0.8, "balanced-high"),
        ),
    )


def strict_attack_prediction(gold: GoldSequenceAnnotationV2) -> AttackSequenceV2:
    action = ObservedActionV2(
        match_id=gold.match_id,
        period=gold.period,
        frame=gold.end_frame,
        timestamp_sec=gold.end_sec,
        action_type="shot",
        source_tier=SourceTier.VIDEO_ONLY,
        actor_team_id=gold.stable_team_id,
        possession_team_id=gold.stable_team_id,
        evidence_ids=(f"video:{gold.match_id}:P{gold.period}:shot",),
    )
    terminal = TerminalEventV2(
        match_id=gold.match_id,
        period=gold.period,
        frame=gold.end_frame,
        timestamp_sec=gold.end_sec,
        terminal_type=TerminalType.SHOT_ATTEMPT,
        possession_team_id=gold.stable_team_id,
        source_tier=SourceTier.VIDEO_ONLY,
        body_part="right_foot",
        outcome="on_target",
        action_id=action.action_id,
        evidence_ids=(action.action_id,),
    )
    possession = PossessionSequenceV2(
        match_id=gold.match_id,
        period=gold.period,
        possession_team_id=gold.stable_team_id,
        start_frame=gold.start_frame,
        end_frame=gold.end_frame,
        start_sec=gold.start_sec,
        end_sec=gold.end_sec,
        actions=(action,),
        status=SequenceStatus.COMPLETE,
        source_tier=SourceTier.VIDEO_ONLY,
        terminal=terminal,
    )
    return AttackSequenceV2(possession=possession)


class FrozenReleaseTests(unittest.TestCase):
    def test_strict_evaluator_rejects_arbitrary_legacy_and_censored_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arbitrary = root / "arbitrary.json"
            arbitrary.write_text(
                json.dumps([{"schema_version": "2.0", "attack_id": "made-up"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "AttackSequenceV2"):
                _sequence_records(arbitrary, human_gold=False)

            gold = GoldSequenceAnnotationV2(
                match_id="game_18",
                period=1,
                stable_team_id="home",
                start_frame=100,
                end_frame=200,
                start_sec=4.0,
                end_sec=8.0,
                label="attack",
                terminal_type="shot_attempt",
                terminal_frame=200,
                terminal_sec=8.0,
                terminal_cause=None,
                shot_body_part="right_foot",
                shot_outcome="on_target",
                beneficiary_team_id=None,
                last_touch_team_id=None,
                replay=False,
                completeness="complete",
                confidence=1.0,
                reason_codes=("clear_attempt",),
                notes="",
                annotator_id="expert-1",
                annotated_at="2026-08-27T00:00:00Z",
            ).with_stable_id()
            valid = strict_attack_prediction(gold).to_dict()

            legacy = root / "legacy.json"
            legacy.write_text(
                json.dumps([{"schema_version": "PossessionCandidateV1", "attack_id": "x"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema 2.0"):
                _sequence_records(legacy, human_gold=False)

            censored = json.loads(json.dumps(valid))
            censored["possession"]["status"] = "right_censored"
            censored_path = root / "censored.json"
            censored_path.write_text(json.dumps([censored]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observed end"):
                _sequence_records(censored_path, human_gold=False)

            non_allowlisted = json.loads(json.dumps(valid))
            non_allowlisted["possession"]["terminal"]["terminal_type"] = "turnover"
            non_allowlisted["possession"]["terminal"]["terminal_id"] = ""
            non_allowlisted["possession"]["possession_id"] = ""
            non_allowlisted["attack_id"] = ""
            non_allowlisted_path = root / "turnover.json"
            non_allowlisted_path.write_text(
                json.dumps([non_allowlisted]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "not permitted"):
                _sequence_records(non_allowlisted_path, human_gold=False)

    def test_human_gold_is_validated_and_test_json_cannot_bypass_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = GoldSequenceAnnotationV2(
                match_id="game_18",
                period=1,
                stable_team_id="home",
                start_frame=100,
                end_frame=200,
                start_sec=4.0,
                end_sec=8.0,
                label="attack",
                terminal_type="shot_attempt",
                terminal_frame=200,
                terminal_sec=8.0,
                terminal_cause=None,
                shot_body_part="right_foot",
                shot_outcome="on_target",
                beneficiary_team_id=None,
                last_touch_team_id=None,
                replay=False,
                completeness="complete",
                confidence=1.0,
                reason_codes=("clear_attempt",),
                notes="",
                annotator_id="expert-1",
                annotated_at="2026-08-27T00:00:00Z",
            ).with_stable_id()
            development_path = root / "development.json"
            development_path.write_text(
                json.dumps([development.to_dict()]), encoding="utf-8"
            )
            parsed = _sequence_records(development_path, human_gold=True)
            self.assertEqual(parsed, (development,))

            test_payload = {
                **development.to_dict(),
                "match_id": "game_0",
                "annotation_id": None,
            }
            test_path = root / "alternate-test.json"
            test_path.write_text(json.dumps([test_payload]), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "sealed JSONL"):
                _sequence_records(
                    test_path,
                    human_gold=True,
                    allow_frozen_test=True,
                )

    def test_selects_balanced_development_point_and_binds_every_hash(self) -> None:
        record = frozen_release()

        selected = record.operating_point["selected"]
        self.assertEqual(selected["label"], "balanced-high")
        self.assertEqual(selected["threshold"], 0.5)
        self.assertEqual(record.operating_point["development_dataset_hash"], record.component_hashes.development_dataset_sha256)
        self.assertEqual(len(record.release_sha256), 64)
        self.assertEqual(len(record.development_sweep_sha256), 64)
        record.verify()

    def test_release_selection_is_development_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "development"):
            freeze_attack_v2_release(
                release_id="attack-v2-r1",
                component_hashes=component_hashes(),
                measurements=(OperatingPointMeasurement(0.5, 0.8, 0.8),),
                split="test",
            )

    def test_manifest_is_idempotent_but_cannot_be_replaced_or_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.json"
            first = frozen_release()
            write_frozen_release_manifest(first, path)
            write_frozen_release_manifest(first, path)
            self.assertEqual(load_frozen_release_manifest(path).release_sha256, first.release_sha256)

            with self.assertRaises(FileExistsError):
                write_frozen_release_manifest(frozen_release("changed"), path)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["component_hashes"]["rubric_sha256"] = "f" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ReleaseIntegrityError, "manifest hash mismatch"):
                load_frozen_release_manifest(path)

    def test_operating_point_tampering_is_detected_independently(self) -> None:
        payload = frozen_release().to_dict()
        payload["development_selection"]["operating_point"]["selected"]["threshold"] = 0.1
        # Recomputing only the outer hash cannot disguise an altered frozen selection.
        identity = {key: value for key, value in payload.items() if key != "release_sha256"}
        payload["release_sha256"] = content_sha256(identity)
        with self.assertRaisesRegex(ReleaseIntegrityError, "operating-point hash mismatch"):
            type(frozen_release()).from_dict(payload)

    def test_manifest_rejects_unhashed_extra_claim_fields(self) -> None:
        payload = frozen_release().to_dict()
        payload["validated"] = True
        with self.assertRaisesRegex(ReleaseIntegrityError, "unexpected"):
            type(frozen_release()).from_dict(payload)

    def test_hash_contract_rejects_labels_and_path_digest_supports_bundles(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ReleaseComponentHashes("not-a-hash", *("a" * 64 for _ in range(5)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "model.bin"
            file_path.write_bytes(b"model")
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "weights.bin").write_bytes(b"weights")
            self.assertEqual(len(path_sha256(file_path)), 64)
            self.assertEqual(len(path_sha256(bundle)), 64)

    def test_cli_freezes_and_verifies_release_from_content_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {"threshold": 0.2, "precision": 0.9, "recall": 0.6},
                            {"threshold": 0.4, "precision": 0.8, "recall": 0.8},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            inputs = []
            for name in (
                "config.json",
                "model.bin",
                "schema.json",
                "rubric.md",
                "dev.jsonl",
                "frozen-baseline.json",
            ):
                path = root / name
                path.write_text(name, encoding="utf-8")
                inputs.append(path)
            manifest = root / "release.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli_main(
                    [
                        "freeze-attack-v2-release",
                        "attack-v2-cli-r1",
                        str(candidates),
                        *(str(path) for path in inputs),
                        str(manifest),
                    ]
                )
            self.assertTrue(manifest.is_file())
            self.assertEqual(
                load_frozen_release_manifest(manifest).operating_point["selected"]["threshold"],
                0.4,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(["verify-attack-v2-release", str(manifest)])

    def test_cli_cannot_open_frozen_test_without_manifest_and_ledger(self) -> None:
        with self.assertRaisesRegex(PermissionError, "release-manifest"):
            cli_main(
                [
                    "evaluate-attack-v2",
                    "attack-v2-frozen-test.jsonl",
                    "predictions.json",
                    "--allow-frozen-test",
                ]
            )

    def test_cli_final_evaluation_consumes_release_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_records = tuple(
                GoldSequenceAnnotationV2(
                    match_id=f"game_{match_number}",
                    period=period,
                    stable_team_id="home",
                    start_frame=100,
                    end_frame=200,
                    start_sec=4.0,
                    end_sec=8.0,
                    label="attack",
                    terminal_type="shot_attempt",
                    terminal_frame=200,
                    terminal_sec=8.0,
                    terminal_cause=None,
                    shot_body_part="right_foot",
                    shot_outcome="on_target",
                    beneficiary_team_id=None,
                    last_touch_team_id=None,
                    replay=False,
                    completeness="complete",
                    confidence=1.0,
                    reason_codes=("clear_attempt",),
                    notes="",
                    annotator_id="expert-1",
                    annotated_at="2026-08-27T00:00:00Z",
                ).with_stable_id()
                for match_number in FROZEN_TEST_MATCHES
                for period in (1, 2)
            )
            labels = root / "attack-v2-frozen-test.jsonl"
            for gold in gold_records:
                labels = append_sealed_test_annotation_jsonl(root, gold)
            intervals = tuple(
                GoldPeriodIntervalV2(
                    match_id=gold.match_id,
                    period=gold.period,
                    start_frame=gold.start_frame,
                    end_frame=gold.end_frame,
                    start_sec=gold.start_sec,
                    end_sec=gold.end_sec,
                )
                for gold in gold_records
            )
            freeze_sealed_test_annotations(root, intervals)
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps([strict_attack_prediction(item).to_dict() for item in gold_records]),
                encoding="utf-8",
            )
            baseline = root / "frozen-baseline.json"
            baseline.write_text("[]", encoding="utf-8")
            manifest = root / "release.json"
            release = frozen_release(frozen_baseline_sha256=path_sha256(baseline))
            write_frozen_release_manifest(release, manifest)
            ledger = root / "ledger"
            output = root / "result.json"
            arguments = [
                "evaluate-attack-v2",
                str(labels),
                str(predictions),
                "--allow-frozen-test",
                "--release-manifest",
                str(manifest),
                "--final-test-ledger",
                str(ledger),
                "--frozen-baseline",
                str(baseline),
                "--output",
                str(output),
            ]
            wrong_baseline = root / "wrong-frozen-baseline.json"
            wrong_baseline.write_text("[\n]", encoding="utf-8")
            wrong_arguments = list(arguments)
            wrong_arguments[wrong_arguments.index(str(baseline))] = str(wrong_baseline)
            with self.assertRaisesRegex(ValueError, "does not match the release manifest"):
                with contextlib.redirect_stdout(io.StringIO()):
                    cli_main(wrong_arguments)
            self.assertEqual(final_test_ledger_status(release, ledger)["state"], "available")

            with contextlib.redirect_stdout(io.StringIO()):
                cli_main(arguments)
            self.assertTrue(output.is_file())
            self.assertEqual(final_test_ledger_status(release, ledger)["state"], "completed")
            with self.assertRaises(FinalTestAlreadyClaimedError):
                with contextlib.redirect_stdout(io.StringIO()):
                    cli_main(arguments)


class FinalTestLedgerTests(unittest.TestCase):
    def _claim(self, release, ledger: Path):
        return claim_final_test_once(
            release,
            ledger,
            final_test_labels_sha256=content_sha256("sealed-labels"),
            predictions_sha256=content_sha256("predictions"),
            evaluation_protocol_sha256=content_sha256(
                {"tiou": [0.25, 0.5, 0.75], "bootstrap": 2000, "seed": 42}
            ),
        )

    def test_claim_is_one_shot_and_completion_is_append_only(self) -> None:
        release = frozen_release()
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory)
            self.assertEqual(final_test_ledger_status(release, ledger)["state"], "available")

            claim = self._claim(release, ledger)
            self.assertEqual(final_test_ledger_status(release, ledger)["state"], "claimed")
            with self.assertRaises(FinalTestAlreadyClaimedError):
                self._claim(release, ledger)

            completion = complete_final_test_claim(
                release,
                ledger,
                claim_sha256=claim["claim_sha256"],
                result_sha256=content_sha256({"attack_f1": 0.81}),
            )
            self.assertEqual(completion["state"], "completed")
            self.assertEqual(final_test_ledger_status(release, ledger)["state"], "completed")
            with self.assertRaises(FinalTestAlreadyClaimedError):
                complete_final_test_claim(
                    release,
                    ledger,
                    claim_sha256=claim["claim_sha256"],
                    result_sha256=content_sha256({"attack_f1": 0.81}),
                )

    def test_final_protocol_is_frozen_to_2000_resamples_and_seed_42(self) -> None:
        self.assertEqual(len(final_test_evaluation_protocol_sha256()), 64)
        with self.assertRaisesRegex(ValueError, "2,000"):
            final_test_evaluation_protocol_sha256(bootstrap_samples=100)
        with self.assertRaisesRegex(ValueError, "seed 42"):
            final_test_evaluation_protocol_sha256(bootstrap_seed=7)

    def test_same_release_id_cannot_evade_ledger_with_changed_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory)
            self._claim(frozen_release("first"), ledger)
            with self.assertRaises(FinalTestAlreadyClaimedError):
                self._claim(frozen_release("second"), ledger)

    def test_corrupt_existing_claim_blocks_rerun(self) -> None:
        release = frozen_release()
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory)
            claim = self._claim(release, ledger)
            claim_path = Path(claim["claim_path"])
            claim_path.write_text("{corrupt", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseIntegrityError, "remains consumed"):
                self._claim(release, ledger)


if __name__ == "__main__":
    unittest.main()
