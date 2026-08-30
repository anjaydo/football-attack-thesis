"""Possession-first football sequence analysis with a strict Attack-v2 contract."""

from .attack_v2 import (
    AttackV2DecodeResult,
    FootpassAttackV2Config,
    decode_footpass_attack_v2,
    legacy_attack_to_possession_v2,
    qualify_attack_v2,
)
from .models import (
    ATTACK_TERMINAL_ALLOWLIST,
    AttackSequence,
    AttackSequenceV2,
    Event,
    FieldProvenanceV2,
    ObservedActionV2,
    PossessionSequenceV2,
    PossessionState,
    PossessionValueV2,
    SequenceStatus,
    SourceTier,
    TerminalEventV2,
    TerminalType,
)
from .research import AttackFeatures, BenchmarkRecord, EvidenceBundle, EvaluationVerdict
from .scoring import Evaluation, evaluate_attack
from .segmentation import SegmentationConfig, segment_attacks

__all__ = [
    "AttackSequence",
    "AttackSequenceV2",
    "AttackV2DecodeResult",
    "AttackFeatures",
    "BenchmarkRecord",
    "EvidenceBundle",
    "Event",
    "FieldProvenanceV2",
    "FootpassAttackV2Config",
    "ObservedActionV2",
    "PossessionSequenceV2",
    "PossessionState",
    "PossessionValueV2",
    "SequenceStatus",
    "SourceTier",
    "TerminalEventV2",
    "TerminalType",
    "ATTACK_TERMINAL_ALLOWLIST",
    "Evaluation",
    "EvaluationVerdict",
    "SegmentationConfig",
    "evaluate_attack",
    "decode_footpass_attack_v2",
    "legacy_attack_to_possession_v2",
    "qualify_attack_v2",
    "segment_attacks",
]
