from __future__ import annotations

import unittest

from attack_analysis.agent_tools import EvidenceToolRegistry, verify_verdict
from attack_analysis.attack_features import build_attack_features
from attack_analysis.llm import DeterministicEvidenceClient
from attack_analysis.models import AttackSequence, Event
from attack_analysis.research import EvidenceBundle, EvidenceClaim, EvaluationVerdict


def _sequence() -> AttackSequence:
    return AttackSequence(
        attack_id="game_1_H1:attack-0001",
        match_id="game_1_H1",
        team="0",
        start_sec=1.0,
        end_sec=3.0,
        start_frame=25,
        end_frame=75,
        termination_reason="shot",
        source="footpass_oracle",
        events=(
            Event(
                timestamp=1.0,
                event_type="pass",
                team="0",
                x=40.0,
                y=30.0,
                frame=25,
                jersey=9,
                confidence=1.0,
                source="footpass_oracle",
                evidence_id="event:1",
            ),
            Event(
                timestamp=3.0,
                event_type="shot",
                team="0",
                x=94.0,
                y=34.0,
                frame=75,
                jersey=9,
                xg=0.35,
                xg_source="test",
                source="footpass_oracle",
                evidence_id="event:2",
            ),
        ),
    )


class ResearchContractTests(unittest.TestCase):
    def test_extended_attack_round_trip(self) -> None:
        sequence = _sequence()
        self.assertEqual(AttackSequence.from_dict(sequence.to_dict()), sequence)

    def test_features_round_trip_and_evidence_ids(self) -> None:
        features = build_attack_features(_sequence())
        self.assertEqual(features.xg_max, 0.35)
        self.assertEqual(features.evidence_ids, ("event:1", "event:2"))
        self.assertEqual(type(features).from_dict(features.to_dict()), features)

    def test_failed_quality_gate_forces_abstention(self) -> None:
        features = build_attack_features(_sequence())
        bundle = EvidenceBundle(
            attack=features,
            quality_gates={"events": True, "tracking": False, "calibration": False},
            quality_metrics={},
            limitations=("tracking unavailable",),
        )
        verdict = DeterministicEvidenceClient().evaluate(
            bundle, EvidenceToolRegistry({features.attack_id: bundle})
        )
        self.assertTrue(verdict.abstained)
        self.assertEqual(verdict.label, "insufficient_evidence")

    def test_verdict_rejects_unknown_evidence(self) -> None:
        features = build_attack_features(_sequence())
        bundle = EvidenceBundle(
            attack=features,
            quality_gates={"events": True, "tracking": True, "calibration": True},
            quality_metrics={},
        )
        verdict = EvaluationVerdict(
            attack_id=features.attack_id,
            label="effective",
            score=4,
            confidence=0.8,
            claims=(EvidenceClaim("unsupported", ("invented",)),),
        )
        with self.assertRaisesRegex(ValueError, "unsupported evidence"):
            verify_verdict(verdict, bundle)


if __name__ == "__main__":
    unittest.main()
