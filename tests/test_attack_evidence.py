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


if __name__ == "__main__":
    unittest.main()
