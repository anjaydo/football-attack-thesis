import unittest

import numpy as np

from attack_analysis.temporal_model import (
    MODEL_VALIDATION_STATUS,
    CheckpointCandidate,
    DilatedTCNConfig,
    MultiTaskLossSpec,
    TemporalDecodeEvidence,
    TemporalFeatureBatch,
    TemporalFeatureCacheConfig,
    TemporalState,
    class_balanced_focal_loss_numpy,
    decode_period_semi_markov,
    select_checkpoint_by_human_dev_attack_f1,
)


STATE_ORDER = tuple(TemporalState)
STATE_INDEX = {state: index for index, state in enumerate(STATE_ORDER)}


def emissions(preferred: list[TemporalState], high: float = 6.0) -> np.ndarray:
    scores = np.full((len(preferred), len(STATE_ORDER)), -4.0, dtype=np.float64)
    for index, state in enumerate(preferred):
        scores[index, STATE_INDEX[state]] = high
    return scores


class TemporalFeatureContractTests(unittest.TestCase):
    def test_cache_schema_is_fixed_at_five_hz(self) -> None:
        config = TemporalFeatureCacheConfig()
        self.assertEqual(config.sample_rate_hz, 5.0)
        self.assertEqual(config.sample_period_sec, 0.2)
        self.assertEqual(config.chunk_seconds, 30.0)
        self.assertGreater(config.feature_width, 0)
        with self.assertRaises(TypeError):
            TemporalFeatureCacheConfig(sample_rate_hz=10.0)  # type: ignore[call-arg]

    def test_feature_batch_rejects_non_five_hz_grid(self) -> None:
        config = TemporalFeatureCacheConfig()
        feature_values = np.zeros((3, config.feature_width), dtype=np.float32)
        masks = np.zeros((3, config.missingness_width), dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "fixed 5 Hz grid"):
            TemporalFeatureBatch(
                match_id="game_18",
                period=1,
                timestamps_sec=np.asarray([0.0, 0.2, 0.5]),
                source_frames=np.asarray([0, 5, 12]),
                features=feature_values,
                missingness=masks,
            )

    def test_tcn_config_has_thirty_second_context_and_gpu_safe_defaults(self) -> None:
        config = DilatedTCNConfig()
        self.assertEqual(
            tuple(state.value for state in TemporalState),
            (
                "dead_ball",
                "control_team_A",
                "control_team_B",
                "contested",
                "ball_flight",
                "unknown",
            ),
        )
        self.assertEqual(config.context_samples, 150)
        self.assertEqual(config.effective_receptive_field_seconds, 30.0)
        self.assertGreaterEqual(config.theoretical_receptive_field_seconds, 30.0)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertTrue(config.use_amp)
        self.assertLessEqual(config.peak_vram_budget_gb, 5.5)
        self.assertEqual(config.validation_status, MODEL_VALIDATION_STATUS)
        self.assertEqual(len(MultiTaskLossSpec().heads), 9)
        with self.assertRaisesRegex(ValueError, "5.5 GB"):
            DilatedTCNConfig(peak_vram_budget_gb=5.6)


class SemiMarkovDecoderTests(unittest.TestCase):
    def test_failed_contested_touch_does_not_switch_owner(self) -> None:
        preferred = [
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTESTED,
            TemporalState.CONTROL_TEAM_B,
            TemporalState.CONTROL_TEAM_B,
        ]
        scores = emissions(preferred)
        # Remaining with A is less locally attractive, but B has no established
        # control evidence and therefore cannot take ownership after the touch.
        scores[3:, STATE_INDEX[TemporalState.CONTROL_TEAM_A]] = 0.0
        result = decode_period_semi_markov(
            scores,
            evidence=TemporalDecodeEvidence(
                contested_touch=(False, False, True, False, False),
            ),
        )
        self.assertEqual(result.states[2], TemporalState.CONTESTED)
        self.assertEqual(result.owner_context[2], "A")
        self.assertEqual(result.states[-1], TemporalState.CONTROL_TEAM_A)
        self.assertNotIn(TemporalState.CONTROL_TEAM_B, result.states)

    def test_established_opponent_control_allows_switch(self) -> None:
        preferred = [
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTESTED,
            TemporalState.CONTROL_TEAM_B,
            TemporalState.CONTROL_TEAM_B,
        ]
        result = decode_period_semi_markov(
            emissions(preferred),
            evidence=TemporalDecodeEvidence(
                established_control=(None, None, None, "B", None),
                contested_touch=(False, False, True, False, False),
            ),
        )
        self.assertEqual(result.states[-1], TemporalState.CONTROL_TEAM_B)

    def test_terminal_is_absorbing_and_reset_opens_new_sequence(self) -> None:
        preferred = [
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTROL_TEAM_A,
            TemporalState.BALL_FLIGHT,
            TemporalState.CONTROL_TEAM_B,
            TemporalState.CONTROL_TEAM_B,
        ]
        result = decode_period_semi_markov(
            emissions(preferred),
            evidence=TemporalDecodeEvidence(
                terminal_types=(None, None, "shot_attempt", None, None),
                terminal_teams=(None, None, "A", None, None),
                reset_teams=(None, None, None, "B", None),
            ),
        )
        self.assertEqual(result.terminal_indices, (2,))
        self.assertEqual(len(result.sequences), 2)
        self.assertEqual(result.sequences[0].end_index, 2)
        self.assertEqual(result.sequences[0].status, "complete")
        self.assertEqual(result.sequences[1].start_index, 3)
        self.assertEqual(result.states[3], TemporalState.CONTROL_TEAM_B)
        self.assertTrue(
            all(not (segment.start_index <= 2 < segment.end_index) for segment in result.segments)
        )

    def test_single_sample_terminal_block_is_boundary_duration_censored(self) -> None:
        result = decode_period_semi_markov(
            emissions([TemporalState.CONTROL_TEAM_A]),
            evidence=TemporalDecodeEvidence(
                terminal_types=("shot_attempt",),
                terminal_teams=("A",),
            ),
        )
        self.assertEqual(result.states, (TemporalState.CONTROL_TEAM_A,))
        self.assertEqual(result.sequences[0].status, "complete")

    def test_observation_gap_is_unknown_and_marks_two_sided_censoring(self) -> None:
        preferred = [
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTROL_TEAM_A,
            TemporalState.CONTROL_TEAM_B,
            TemporalState.CONTROL_TEAM_B,
            TemporalState.CONTROL_TEAM_B,
        ]
        result = decode_period_semi_markov(
            emissions(preferred),
            evidence=TemporalDecodeEvidence(
                observation_gap=(False, False, True, True, False, False),
            ),
        )
        self.assertEqual(result.states[2:4], (TemporalState.UNKNOWN, TemporalState.UNKNOWN))
        self.assertEqual(len(result.censoring_markers), 1)
        marker = result.censoring_markers[0]
        self.assertTrue(marker.left_sequence_right_censored)
        self.assertTrue(marker.right_sequence_left_censored)
        self.assertTrue(result.sequences[0].right_censored)
        self.assertTrue(result.sequences[1].left_censored)

    def test_global_decode_is_deterministic(self) -> None:
        scores = np.zeros((12, len(STATE_ORDER)), dtype=np.float64)
        evidence = TemporalDecodeEvidence(
            contested_touch=(False,) * 5 + (True,) + (False,) * 6,
            terminal_types=(None,) * 9 + ("period_end",) + (None,) * 2,
        )
        first = decode_period_semi_markov(scores, evidence=evidence)
        second = decode_period_semi_markov(scores.copy(), evidence=evidence)
        self.assertEqual(first, second)


class LossAndCheckpointTests(unittest.TestCase):
    def test_class_balanced_focal_loss_is_finite_and_improves_with_logits(self) -> None:
        targets = np.asarray([0, 1])
        counts = np.asarray([100, 5])
        weak = class_balanced_focal_loss_numpy(np.zeros((2, 2)), targets, counts)
        strong = class_balanced_focal_loss_numpy(
            np.asarray([[5.0, -5.0], [-5.0, 5.0]]),
            targets,
            counts,
        )
        self.assertTrue(np.isfinite(strong))
        self.assertLess(strong, weak)

    def test_checkpoint_selection_uses_human_dev_attack_f1_not_action_or_loss(self) -> None:
        action_winner = CheckpointCandidate(
            "best-action",
            human_dev_attack_f1=0.70,
            human_dev_attack_precision=0.70,
            human_dev_attack_recall=0.70,
            micro_action_f1=0.99,
            validation_loss=0.01,
            epoch=2,
            seed=1,
        )
        attack_winner = CheckpointCandidate(
            "best-attack",
            human_dev_attack_f1=0.80,
            human_dev_attack_precision=0.78,
            human_dev_attack_recall=0.82,
            micro_action_f1=0.60,
            validation_loss=5.0,
            epoch=8,
            seed=1,
        )
        selected = select_checkpoint_by_human_dev_attack_f1(
            [action_winner, attack_winner]
        )
        self.assertEqual(selected.checkpoint_id, "best-attack")

    def test_checkpoint_tie_break_prefers_balanced_precision_and_recall(self) -> None:
        imbalanced = CheckpointCandidate("p-heavy", 0.80, 0.95, 0.65, 1, 1)
        balanced = CheckpointCandidate("balanced", 0.80, 0.80, 0.80, 3, 2)
        self.assertEqual(
            select_checkpoint_by_human_dev_attack_f1([imbalanced, balanced]).checkpoint_id,
            "balanced",
        )


if __name__ == "__main__":
    unittest.main()
