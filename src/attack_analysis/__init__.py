"""Football attack effectiveness research baseline."""

from .models import AttackSequence, Event
from .research import AttackFeatures, BenchmarkRecord, EvidenceBundle, EvaluationVerdict
from .scoring import Evaluation, evaluate_attack
from .segmentation import SegmentationConfig, segment_attacks

__all__ = [
    "AttackSequence",
    "AttackFeatures",
    "BenchmarkRecord",
    "EvidenceBundle",
    "Event",
    "Evaluation",
    "EvaluationVerdict",
    "SegmentationConfig",
    "evaluate_attack",
    "segment_attacks",
]
