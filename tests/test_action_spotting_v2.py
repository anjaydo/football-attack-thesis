import tempfile
import unittest
from pathlib import Path

from attack_analysis.action_spotting_v2 import (
    OFFICIAL_TOLERANCE_FRAMES,
    ActionCandidate,
    ActionSpottingEvaluationProtocol,
    ActionSpottingResourceContract,
    ActionSpottingSearchPlan,
    ActionSpottingSearchTrial,
    ActionTrainingRecord,
    DecodeConstraints,
    MatchClassAwareSampler,
    OperatingPointMeasurement,
    SoccerNetAuxiliaryAblation,
    SoccerNetAuxiliaryFeatureContract,
    TemporalWindow,
    WindowLogitChunk,
    build_overlapping_windows,
    compute_match_class_sampling_weights,
    global_constrained_decode,
    merge_center_weighted_logits,
    mine_duplicate_conflict_hard_negatives,
    select_and_freeze_attack_operating_point,
    write_frozen_operating_point,
)


def training_record(
    index: int,
    class_name: str,
    match_id: str,
    target: int = 1,
) -> ActionTrainingRecord:
    return ActionTrainingRecord(
        record_id=f"r-{index}",
        match_id=match_id,
        period=1,
        frame=index,
        class_name=class_name,
        target=target,
    )


def candidate(
    candidate_id: str,
    frame: int,
    class_name: str,
    score: float,
    *,
    team_id: str | None = None,
    conflict_group: str | None = None,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=candidate_id,
        match_id="game_18",
        period=1,
        frame=frame,
        class_name=class_name,
        score=score,
        team_id=team_id,
        conflict_group=conflict_group,
    )


class MatchClassAwareSamplerTests(unittest.TestCase):
    def test_full_dataset_is_weighted_without_cap_500(self) -> None:
        records = tuple(
            training_record(
                index,
                "pass" if index < 1_100 else "shot",
                f"game_{index % 4}",
            )
            for index in range(1_200)
        )
        sampler = MatchClassAwareSampler(
            records,
            weighting="inverse_frequency",
            seed=17,
        )

        epoch = sampler.epoch(0)
        self.assertEqual(len(epoch), 1_200)
        self.assertEqual({item.record_id for item in epoch}, {item.record_id for item in records})
        self.assertEqual(epoch, sampler.epoch(0))
        self.assertNotEqual(epoch, sampler.epoch(1))

        weights = {item.record.record_id: item.weight for item in sampler.weighted_records}
        self.assertGreater(weights["r-1150"], weights["r-100"])

    def test_equalises_match_mass_inside_each_class(self) -> None:
        records = tuple(
            [training_record(index, "shot", "long") for index in range(8)]
            + [training_record(100 + index, "shot", "short") for index in range(2)]
            + [training_record(200 + index, "pass", "long") for index in range(20)]
        )
        weighted = compute_match_class_sampling_weights(
            records,
            weighting="inverse_frequency",
        )
        mass = {}
        for item in weighted:
            key = (item.record.class_name, item.record.match_id)
            mass[key] = mass.get(key, 0.0) + item.weight

        self.assertAlmostEqual(mass[("shot", "long")], mass[("shot", "short")])
        self.assertAlmostEqual(
            mass[("shot", "long")] + mass[("shot", "short")],
            mass[("pass", "long")],
        )

    def test_sampler_rejects_duplicate_record_ids(self) -> None:
        item = training_record(1, "shot", "game_1")
        with self.assertRaises(ValueError):
            compute_match_class_sampling_weights((item, item))


class OverlappingWindowTests(unittest.TestCase):
    def test_30_second_half_overlap_covers_every_boundary_frame(self) -> None:
        windows = build_overlapping_windows(
            match_id="game_18",
            period=1,
            total_frames=2_000,
            fps=25.0,
        )

        self.assertEqual(windows[0].start_frame, 0)
        self.assertEqual(windows[0].end_frame_exclusive, 750)
        self.assertEqual(windows[1].start_frame, 375)
        self.assertEqual(windows[-1].end_frame_exclusive, 2_000)
        self.assertEqual(windows[-1].start_frame, 1_250)
        for event_frame in (0, 374, 375, 749, 750, 1_999):
            self.assertTrue(any(window.contains(event_frame) for window in windows))
        self.assertTrue(
            all(any(window.contains(frame) for window in windows) for frame in range(2_000))
        )

    def test_short_period_is_one_padded_window(self) -> None:
        windows = build_overlapping_windows(
            match_id="game_18",
            period=2,
            total_frames=125,
            fps=25.0,
            start_frame=10,
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].start_frame, 10)
        self.assertEqual(windows[0].end_frame_exclusive, 135)
        self.assertEqual(windows[0].padding_frames, 625)

    def test_center_weighted_merge_prefers_nearest_window_center(self) -> None:
        left = TemporalWindow("game_18", 1, 0, 5, 5)
        right = TemporalWindow("game_18", 1, 2, 7, 5)
        merged = merge_center_weighted_logits(
            (
                WindowLogitChunk(left, ("shot",), ((0.0,),) * 5),
                WindowLogitChunk(right, ("shot",), ((10.0,),) * 5),
            )
        )
        by_frame = {item.frame: item for item in merged}

        self.assertLess(by_frame[2].logits[0], 5.0)
        self.assertAlmostEqual(by_frame[3].logits[0], 5.0)
        self.assertGreater(by_frame[4].logits[0], 5.0)
        self.assertEqual(by_frame[3].contributing_windows, 2)
        self.assertEqual(by_frame[0].logits, (0.0,))
        self.assertEqual(by_frame[6].logits, (10.0,))

    def test_merge_rejects_feature_order_mismatch(self) -> None:
        window = TemporalWindow("game_18", 1, 0, 2, 2)
        with self.assertRaises(ValueError):
            merge_center_weighted_logits(
                (
                    WindowLogitChunk(window, ("shot",), ((1.0,), (1.0,))),
                    WindowLogitChunk(window, ("header",), ((1.0,), (1.0,))),
                )
            )


class ConstrainedDecodeTests(unittest.TestCase):
    def test_global_decode_is_order_independent_and_tracks_suppression(self) -> None:
        items = (
            candidate("shot-best", 100, "shot", 0.9, team_id="A"),
            candidate("shot-duplicate", 106, "shot", 0.8, team_id="B"),
            candidate("pass-best", 200, "pass", 0.95, conflict_group="control"),
            candidate("drive-conflict", 200, "drive", 0.85, conflict_group="control"),
            # Shot and block may co-exist unless an explicit ontology constraint says otherwise.
            candidate("block", 100, "block", 0.7, team_id="B"),
        )
        constraints = DecodeConstraints(conflict_radius_frames=0)

        forward = global_constrained_decode(items, constraints=constraints)
        reverse = global_constrained_decode(reversed(items), constraints=constraints)

        self.assertEqual(forward, reverse)
        self.assertEqual(
            {item.candidate_id for item in forward.selected},
            {"shot-best", "pass-best", "block"},
        )
        self.assertEqual(
            {(item.candidate.candidate_id, item.reason) for item in forward.suppressed},
            {("shot-duplicate", "duplicate"), ("drive-conflict", "conflict")},
        )

    def test_hard_negatives_cover_context_around_duplicates_and_conflicts(self) -> None:
        items = (
            candidate("shot-best", 100, "shot", 0.9),
            candidate("shot-duplicate", 105, "shot", 0.8),
            candidate("pass-best", 200, "pass", 0.95, conflict_group="owner"),
            candidate("drive-conflict", 200, "drive", 0.85, conflict_group="owner"),
        )
        negatives = mine_duplicate_conflict_hard_negatives(
            items,
            constraints=DecodeConstraints(conflict_radius_frames=0),
            context_radius_frames=6,
            context_stride_frames=3,
        )

        self.assertEqual({item.reason for item in negatives}, {"duplicate", "conflict"})
        duplicate_frames = {
            item.frame for item in negatives if item.source_prediction_id == "shot-duplicate"
        }
        self.assertIn(105, duplicate_frames)
        self.assertLess(min(duplicate_frames), 105)
        self.assertGreater(max(duplicate_frames), 105)
        self.assertTrue(all(item.as_training_record().target == 0 for item in negatives))

    def test_cross_team_conflicts_are_explicit_not_arbitrary(self) -> None:
        constraints = DecodeConstraints(
            conflict_radius_frames=1,
            team_exclusive_classes=frozenset({"pass", "drive"}),
        )
        result = global_constrained_decode(
            (
                candidate("a", 10, "pass", 0.8, team_id="A"),
                candidate("b", 10, "drive", 0.7, team_id="B"),
            ),
            constraints=constraints,
        )
        self.assertEqual(len(result.conflicts), 1)


class OperatingPointTests(unittest.TestCase):
    def test_official_protocol_persists_plus_minus_12_frames(self) -> None:
        protocol = ActionSpottingEvaluationProtocol()
        self.assertEqual(protocol.tolerance_frames, OFFICIAL_TOLERANCE_FRAMES)
        with self.assertRaises(ValueError):
            ActionSpottingEvaluationProtocol(tolerance_frames=10)

    def test_selects_balanced_attack_f1_and_freezes_dev_record(self) -> None:
        record = select_and_freeze_attack_operating_point(
            (
                OperatingPointMeasurement(0.3, 1.0, 2.0 / 3.0, "imbalanced"),
                OperatingPointMeasurement(0.4, 0.8, 0.8, "balanced-low"),
                OperatingPointMeasurement(0.5, 0.8, 0.8, "balanced-high"),
            ),
            split="development",
            release_id="attack-v2-r1",
            development_dataset_hash="sha256:dev",
        )

        self.assertEqual(record.selected.label, "balanced-high")
        self.assertEqual(record.selected.threshold, 0.5)
        self.assertTrue(record.frozen)
        self.assertEqual(len(record.freeze_hash), 64)
        self.assertEqual(record.to_dict()["protocol"]["tolerance_frames"], 12)

    def test_operating_point_cannot_tune_on_test(self) -> None:
        with self.assertRaises(ValueError):
            select_and_freeze_attack_operating_point(
                (OperatingPointMeasurement(0.5, 0.8, 0.8),),
                split="test",
                release_id="r1",
                development_dataset_hash="hash",
            )

    def test_frozen_record_refuses_different_overwrite(self) -> None:
        first = select_and_freeze_attack_operating_point(
            (OperatingPointMeasurement(0.5, 0.8, 0.8),),
            split="development",
            release_id="r1",
            development_dataset_hash="hash",
        )
        second = select_and_freeze_attack_operating_point(
            (OperatingPointMeasurement(0.6, 0.8, 0.8),),
            split="development",
            release_id="r2",
            development_dataset_hash="hash",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "frozen.json"
            write_frozen_operating_point(first, path)
            write_frozen_operating_point(first, path)
            with self.assertRaises(FileExistsError):
                write_frozen_operating_point(second, path)


class SearchContractTests(unittest.TestCase):
    def test_rtx_4050_resource_contract_is_enforced(self) -> None:
        resource = ActionSpottingResourceContract()
        self.assertTrue(resource.cached_embeddings)
        self.assertTrue(resource.amp)
        self.assertEqual(resource.batch_size, 1)
        self.assertEqual(resource.gradient_accumulation_steps, 8)
        self.assertLessEqual(resource.max_vram_gb, 5.5)
        with self.assertRaises(ValueError):
            ActionSpottingResourceContract(max_vram_gb=5.6)
        with self.assertRaises(ValueError):
            ActionSpottingResourceContract(batch_size=2)
        with self.assertRaises(ValueError):
            ActionSpottingResourceContract(window_seconds=20.0)

    def test_search_is_sequential_max_12_and_final_three_seeds(self) -> None:
        trials = tuple(
            ActionSpottingSearchTrial(f"config-{index}", {"learning_rate": index + 1})
            for index in range(12)
        )
        plan = ActionSpottingSearchPlan(trials)
        self.assertEqual(len(plan.trials), 12)
        self.assertEqual(plan.final_seeds, (42, 43, 44))
        self.assertEqual(plan.to_dict()["execution"], "sequential")

        with self.assertRaises(ValueError):
            ActionSpottingSearchPlan(
                trials
                + (ActionSpottingSearchTrial("config-12", {"learning_rate": 13}),)
            )
        with self.assertRaises(ValueError):
            ActionSpottingSearchPlan(trials[:1], final_seeds=(42, 43))
        with self.assertRaises(ValueError):
            ActionSpottingSearchPlan(trials[:1], sequential=False)

    def test_auxiliary_checkpoint_rejects_mismatch_and_requires_ablation(self) -> None:
        with self.assertRaises(ValueError):
            SoccerNetAuxiliaryFeatureContract(
                enabled=True,
                expected_architecture="tcn-v2",
                expected_features=("spotting-embedding",),
                checkpoint_architecture="transformer-v1",
                checkpoint_features=("spotting-embedding",),
                checkpoint_checksum="sha256:checkpoint",
            )

        enabled = SoccerNetAuxiliaryFeatureContract(
            enabled=True,
            expected_architecture="tcn-v2",
            expected_features=("spotting-embedding",),
            checkpoint_architecture="tcn-v2",
            checkpoint_features=("spotting-embedding",),
            checkpoint_checksum="sha256:checkpoint",
        )
        disabled = SoccerNetAuxiliaryFeatureContract(
            enabled=False,
            expected_architecture="tcn-v2",
            expected_features=("spotting-embedding",),
        )
        SoccerNetAuxiliaryAblation(enabled, disabled)

        enabled_trial = ActionSpottingSearchTrial(
            "with-aux",
            {"learning_rate": 1e-4},
            enabled,
            "soccernet-frozen",
        )
        without_trial = ActionSpottingSearchTrial(
            "without-aux",
            {"learning_rate": 1e-4},
            disabled,
            "soccernet-frozen",
        )
        plan = ActionSpottingSearchPlan((enabled_trial, without_trial))
        self.assertEqual(len(plan.trials), 2)

        with self.assertRaises(ValueError):
            ActionSpottingSearchPlan((enabled_trial,))


if __name__ == "__main__":
    unittest.main()
