"""Football attack effectiveness research baseline."""

from .models import AttackSequence, Event
from .scoring import Evaluation, evaluate_attack
from .segmentation import SegmentationConfig, segment_attacks

__all__ = [
    "AttackSequence",
    "Event",
    "Evaluation",
    "SegmentationConfig",
    "evaluate_attack",
    "segment_attacks",
]
