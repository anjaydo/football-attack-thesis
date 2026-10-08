from __future__ import annotations

import unittest

import numpy as np

from attack_analysis.attack_v2 import decode_footpass_attack_v2
from attack_analysis.fusion import score_attack_with_observations
from attack_analysis.models import ObservedActionV2, SourceTier
from attack_analysis.shot_xg import ShotXGRequiredError, score_analysis_shots
from attack_analysis.xg_v2 import XG_V2_FEATURE_COLUMNS, TrainedXGModelV2


SOURCE = SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL


def _action(
    second: float,
    action_type: str,
    team: str,
    *,
    x: float | None = 90.0,
    y: float | None = 34.0,
) -> ObservedActionV2:
    return ObservedActionV2(
        match_id="game_18",
        period=1,
        frame=round(second * 25),
        timestamp_sec=second,
        action_type=action_type,
        source_tier=SOURCE,
        actor_team_id=team,
        possession_team_id=team,
        x=x,
        y=y,
        evidence_ids=(f"ev-{second}-{action_type}",),
    )


class _ConstantProbability:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame: object) -> np.ndarray:
        del frame
        return np.array([[1.0 - self.probability, self.probability]])


def _model(probability: float = 0.27, *, status: str = "statsbomb_gate_passed_domain_unvalidated") -> TrainedXGModelV2:
    return TrainedXGModelV2(
        model=_ConstantProbability(probability),
        calibrator=None,
        calibration_input="none",
        model_sha256="abc123",
        feature_columns=XG_V2_FEATURE_COLUMNS,
        report={
            "task": "xg-v2",
            "status": status,
            "dataset": {"rows": 87738, "matches": 3461},
        },
    )


class ShotXGTests(unittest.TestCase):
    def test_shot_terminal_receives_statsbomb_xg(self) -> None:
        decoded = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        )
        payload = score_analysis_shots(
            decoded.attacks,
            decoded.possessions,
            model=_model(),
        )
        self.assertEqual(payload["status"], "scored")
        self.assertEqual(payload["shot_terminals_scored"], 1)
        record = payload["records"][0]
        self.assertEqual(record["event_type"], "shot")
        self.assertEqual(record["xg_probability"], 0.27)
        self.assertTrue(record["confirmed_attack"])
        self.assertEqual(payload["training_data"]["source"], "statsbomb_open_data")
        self.assertEqual(payload["training_data"]["rows"], 87738)

    def test_sequence_that_does_not_end_on_shot_has_no_xg_requirement(self) -> None:
        decoded = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=40.0), _action(2.0, "pass", "A", x=55.0)]
        )
        payload = score_analysis_shots(decoded.attacks, decoded.possessions, model=None)
        self.assertEqual(payload["status"], "no_shot_terminals")
        self.assertEqual(payload["records"], [])

    def test_shot_without_model_is_rejected(self) -> None:
        decoded = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        )
        with self.assertRaises(ShotXGRequiredError):
            score_analysis_shots(decoded.attacks, decoded.possessions, model=None)

    def test_non_statsbomb_model_is_rejected(self) -> None:
        decoded = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        )
        with self.assertRaises(ShotXGRequiredError):
            score_analysis_shots(
                decoded.attacks,
                decoded.possessions,
                model=_model(status="research_only"),
            )

    def test_censored_shot_still_receives_probability(self) -> None:
        decoded = decode_footpass_attack_v2(
            [
                _action(1.0, "pass", "A", x=40.0),
                _action(20.0, "pass", "A", x=70.0),
                _action(21.0, "shot", "A", x=92.0),
            ]
        )
        censored = [
            item
            for item in decoded.possessions
            if item.status.value != "complete"
            and item.actions
            and item.actions[-1].action_type == "shot"
        ]
        self.assertTrue(censored)
        payload = score_analysis_shots(decoded.attacks, decoded.possessions, model=_model(0.11))
        censored_records = [item for item in payload["records"] if not item["confirmed_attack"]]
        self.assertEqual(len(censored_records), 1)
        self.assertEqual(censored_records[0]["event_type"], "shot")
        self.assertEqual(censored_records[0]["xg_probability"], 0.11)

    def test_missing_location_does_not_drop_shot_xg(self) -> None:
        decoded = decode_footpass_attack_v2(
            [
                _action(1.0, "pass", "A", x=70.0),
                _action(2.0, "shot", "A", x=None, y=None),
            ]
        )
        result = score_attack_with_observations(
            decoded.attacks[0],
            None,
            model=_model(0.05),
            ball_f1=0.1,
        )
        self.assertIsNotNone(result.xg_record)
        self.assertEqual(result.xg_record.xg_probability, 0.05)
        self.assertIsNone(result.withheld_reason)


if __name__ == "__main__":
    unittest.main()
