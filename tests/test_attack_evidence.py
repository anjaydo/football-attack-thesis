from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attack_analysis.attack_evidence import build_attack_evidence
from attack_analysis.attack_v2 import decode_footpass_attack_v2
from attack_analysis.fusion import score_attack_with_observations
from attack_analysis.llm import DeterministicEvidenceClient
from attack_analysis.models import ObservedActionV2, SourceTier
from attack_analysis.observations import BoundingBox, FrameObservation, TrackedObject
from attack_analysis.agent_tools import EvidenceToolRegistry


SOURCE = SourceTier.PREDICTED_ACTIONS_ORACLE_TACTICAL


def _action(second: float, action_type: str, team: str, *, x: float = 90.0, y: float = 34.0) -> ObservedActionV2:
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


def _player(track_id: str, team: str, x: float, y: float, *, goalkeeper: bool = False) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        object_type="goalkeeper" if goalkeeper else "player",
        confidence=0.9,
        bbox=BoundingBox(10, 10, 30, 50),
        team=team,
        pitch_x=x,
        pitch_y=y,
    )


class AttackEvidenceTests(unittest.TestCase):
    def test_builds_fusion_without_model_and_legacy_bundle_has_evidence(self) -> None:
        attack = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        ).attacks[0]
        frame = FrameObservation(
            frame_index=50,
            timestamp_sec=2.0,
            objects=(
                _player("a1", "team_a", 90.0, 34.0),
                _player("b1", "team_b", 100.0, 34.0, goalkeeper=True),
            ),
        )
        evidence = build_attack_evidence(attack, frames=(frame,))
        self.assertIsNotNone(evidence.fusion)
        self.assertIsNone(evidence.xg_record)
        self.assertEqual(evidence.withheld_reason, "no_xg_v2_model")
        bundle = evidence.as_legacy_bundle()
        self.assertIn(attack.attack_id, bundle.evidence_ids())
        client = DeterministicEvidenceClient()
        tools = EvidenceToolRegistry({attack.attack_id: bundle})
        conclusion = client.evaluate_v2(bundle, tools)
        self.assertTrue(conclusion.abstained_effectiveness)
        self.assertEqual(conclusion.chance_quality, "unknown")

    def test_legacy_bundle_carries_ordered_attack_view_timeline(self) -> None:
        attack = decode_footpass_attack_v2(
            [
                _action(1.0, "pass", "A", x=40.0, y=10.0),
                _action(2.0, "tackle", "B", x=30.0, y=12.0),
                _action(3.0, "cross", "A", x=95.0, y=5.0),
                _action(4.0, "shot", "A", x=96.0, y=34.0),
            ]
        ).attacks[0]
        bundle = build_attack_evidence(attack).as_legacy_bundle(
            shot_outcome="goal",
            outcome_source="video_scoreboard",
        )
        context = bundle.to_dict()["sequence_context"]
        events = context["events"]

        self.assertEqual([item["step"] for item in events], [1, 2, 3, 4])
        self.assertEqual(events[1]["team"], "opponent")
        self.assertEqual(events[1]["x"], 75.0)
        self.assertEqual(events[0]["zone"], "middle_third")
        self.assertEqual(events[2]["channel"], "wide")
        self.assertEqual(events[2]["forward_gain_m"], 55.0)
        self.assertEqual(events[3]["zone"], "penalty_box")
        self.assertEqual(events[3]["action"], "shot_attempt")
        self.assertTrue(events[3]["terminal"])
        self.assertEqual(context["penalty_box_entry"]["step"], 4)
        self.assertEqual(context["opponent_touches"], 1)
        self.assertEqual(context["shot_outcome"], "goal")
        self.assertEqual(context["attacking_action_counts"], {"pass": 1, "cross": 1, "shot_attempt": 1})

        tools = EvidenceToolRegistry({attack.attack_id: bundle})
        read = DeterministicEvidenceClient().evaluate_v2(bundle, tools).tactical_read
        self.assertEqual(
            [(item["step"], item["moment"]) for item in context["key_moments"]],
            [
                (1, "first_observed_event") if context["sequence_status"] == "left_censored" else (1, "possession_start"),
                (2, "opponent_touch"),
                (3, "first_entry_final_third"),
                (3, "largest_forward_gain"),
                (3, "cross"),
                (4, "first_entry_penalty_box"),
                (4, "terminal"),
            ],
        )
        self.assertTrue(read.startswith("Bàn thắng đến từ một đợt lên bóng"), read)
        self.assertIn("tạt bóng từ hành lang biên đưa bóng lên 1/3 cuối sân", read)
        self.assertIn("đối phương có một pha tắc bóng", read)
        self.assertIn("ngay trong lần đầu bóng vào vòng cấm", read)
        self.assertIn("Bóng đi vào lưới", read)
        self.assertNotIn("bước", read)

    def test_polish_translates_leaked_tokens_and_rounds_decimals(self) -> None:
        from attack_analysis.llm import _polish_text

        text = _polish_text(
            "Bước 2 shot_attempt ở final_third, central (opponent_touches=1), xG 0.10350537629252776.",
            "Vietnamese",
        )
        self.assertEqual(
            text,
            "Bước 2 dứt điểm ở 1/3 cuối sân, trung lộ (số lần đối phương chạm bóng: 1), xG 0.104.",
        )
        self.assertEqual(_polish_text("xG 0.12345 middle_third", "English"), "xG 0.123 middle_third")
        self.assertEqual(
            _polish_text("sequence_status=left_censored (left_censored), half-space, kết quả: goal", "Vietnamese"),
            "trạng thái chuỗi: không quan sát được đoạn đầu, hành lang trong, kết quả: bàn thắng",
        )
        self.assertEqual(
            _polish_text("dẫn bóng ở defensive_third (1/3 sân nhà của đội tấn công), xG: 0.134 (xG: 0.134)", "Vietnamese"),
            "dẫn bóng ở 1/3 sân nhà của đội tấn công, xG: 0.134",
        )
        self.assertEqual(
            _polish_text(
                "Ở giây 11.4, số 7 dẫn bóng (evidence_ids: actv2_07340289502728e0136db79a). "
                "Thủ môn đứng gần (goalkeeper visible=true), sút ở t=0.8 giây, XG-v2 0.37.",
                "Vietnamese",
            ),
            "Ở giây thứ 11, số 7 dẫn bóng. Thủ môn đứng gần, sút ở giây thứ 1, xG 0.37.",
        )
        self.assertEqual(
            _polish_text("Ở giây thứ 0,8 số 11 tiến 17.9 m (bước tiến lớn nhất); bàn thắng (kết quả: bàn thắng).", "Vietnamese"),
            "Ở giây thứ 1 số 11 tiến 18 m; bàn thắng.",
        )

    def test_reads_observations_jsonl_from_manifest(self) -> None:
        attack = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        ).attacks[0]
        frame = FrameObservation(
            frame_index=50,
            timestamp_sec=2.0,
            objects=(_player("a1", "team_a", 90.0, 34.0),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            path.write_text(json.dumps(frame.to_dict()) + "\n", encoding="utf-8")
            evidence = build_attack_evidence(
                attack,
                window_manifest={"outputs": {"observations": str(path)}},
            )
        self.assertTrue(evidence.quality_gates["yolo_observations"])
        self.assertEqual(score_attack_with_observations(attack, frame).withheld_reason, "no_xg_v2_model")

    def test_maps_match_frame_to_fullhd_observation_index(self) -> None:
        from attack_analysis.attack_evidence import video_frame_for_match_frame

        self.assertEqual(
            video_frame_for_match_frame(
                250,
                {"video_clock": {"preview_fps": 50.0, "match_fps": 25.0, "preview_offset_sec": 1.5}},
            ),
            575,
        )

    def test_freeze_frame_without_yolo_uses_tactical_counts(self) -> None:
        attack = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        ).attacks[0]
        freeze = {
            "x_shooter": 90.0,
            "y_shooter": 34.0,
            "x_goalkeeper": 102.0,
            "y_goalkeeper": 34.0,
            "foe_players": 8,
            "ally_players": 9,
            "defending_players": 1,
            "under_pressure": 1,
            "observed_players": 18,
            "gk_distance_m": 12.0,
            "evidence_id": "footpass-hdf5:game_18_H1:50:freeze",
        }
        result = score_attack_with_observations(attack, None, footpass_freeze=freeze)
        self.assertEqual(result.association.method, "footpass_tactical")
        self.assertEqual(result.association.xg_raw["foe_players"], 8)
        self.assertEqual(result.association.spatial_context["provenance"]["x_goalkeeper"], "footpass_tactical")
        evidence = build_attack_evidence(
            attack,
            frames=(
                FrameObservation(
                    frame_index=50,
                    timestamp_sec=2.0,
                    objects=(_player("a1", "team_a", 90.0, 34.0),),
                ),
            ),
        )
        legacy = evidence.as_legacy_bundle()
        self.assertIsNotNone(legacy.xg_raw)
        self.assertIn("play_pattern", legacy.xg_raw)

    def test_scores_location_only_xg_without_yolo_or_freeze(self) -> None:
        attack = decode_footpass_attack_v2(
            [_action(1.0, "pass", "A", x=70.0), _action(2.0, "shot", "A")]
        ).attacks[0]
        evidence = build_attack_evidence(attack)
        self.assertIsNotNone(evidence.fusion)
        self.assertEqual(evidence.fusion.association.method, "event_location")
        self.assertEqual(evidence.fusion.association.xg_raw["x_shooter"], 90.0)
        self.assertEqual(evidence.withheld_reason, "no_xg_v2_model")
        self.assertNotEqual(evidence.withheld_reason, "no_yolo_observations")
        self.assertTrue(
            any("location" in item.lower() or "missingness" in item.lower() for item in evidence.limitations)
        )


if __name__ == "__main__":
    unittest.main()
