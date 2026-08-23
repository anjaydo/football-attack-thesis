from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .attack_features import build_attack_features
from .models import AttackSequence


LABELS = ("ineffective", "promising", "effective")


@dataclass(frozen=True)
class AttackAnnotation:
    attack_id: str
    match_id: str
    label: str
    score: int
    confidence: float
    reason: str
    annotated_at: str
    annotation_round: int = 1

    def validate(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if not 1 <= self.score <= 5:
            raise ValueError("score must be between 1 and 5")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.annotation_round < 1:
            raise ValueError("annotation_round must be positive")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "attack_id": self.attack_id,
            "match_id": self.match_id,
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
            "annotated_at": self.annotated_at,
            "annotation_round": self.annotation_round,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AttackAnnotation":
        item = cls(
            attack_id=str(data["attack_id"]),
            match_id=str(data["match_id"]),
            label=str(data["label"]),
            score=int(data["score"]),
            confidence=float(data["confidence"]),
            reason=str(data["reason"]),
            annotated_at=str(data["annotated_at"]),
            annotation_round=int(data.get("annotation_round", 1)),
        )
        item.validate()
        return item


def _stratum(sequence: AttackSequence) -> tuple[str, str, str]:
    features = build_attack_features(sequence)
    shot = "shot" if features.shots else "no_shot"
    progression = (
        "high" if features.progression_ratio >= 0.5 else "medium" if features.progression_ratio >= 0.2 else "low"
    )
    duration = "long" if features.duration_sec >= 15 else "medium" if features.duration_sec >= 6 else "short"
    return shot, progression, duration


def stratified_annotation_sample(
    attacks: Iterable[AttackSequence],
    *,
    sample_size: int = 300,
    seed: int = 42,
) -> tuple[AttackSequence, ...]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    groups: dict[tuple[str, str, str], list[AttackSequence]] = defaultdict(list)
    for sequence in attacks:
        groups[_stratum(sequence)].append(sequence)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[AttackSequence] = []
    while len(selected) < sample_size and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < sample_size:
                selected.append(groups[key].pop())
    return tuple(selected)


def build_annotation_manifest(attacks: Iterable[AttackSequence]) -> dict[str, object]:
    items = tuple(attacks)
    return {
        "schema_version": "1.0",
        "count": len(items),
        "match_count": len({item.match_id for item in items}),
        "strata": {
            "|".join(key): value
            for key, value in Counter(_stratum(item) for item in items).items()
        },
        "attacks": [
            {
                "sequence": item.to_dict(),
                "features": build_attack_features(item).to_dict(),
                "annotation": {
                    "label": None,
                    "score": None,
                    "confidence": None,
                    "reason": None,
                },
            }
            for item in items
        ],
    }


def read_annotations_jsonl(path: Path) -> tuple[AttackAnnotation, ...]:
    result: list[AttackAnnotation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    result.append(AttackAnnotation.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid annotation at line {line_number}: {error}") from error
    return tuple(result)


def weighted_kappa(first: Iterable[int], second: Iterable[int]) -> float:
    left, right = tuple(first), tuple(second)
    if len(left) != len(right) or not left:
        raise ValueError("rating vectors must be non-empty and have equal length")
    if any(value < 1 or value > 5 for value in left + right):
        raise ValueError("ratings must be between 1 and 5")
    categories = range(1, 6)
    observed = sum(((a - b) / 4.0) ** 2 for a, b in zip(left, right)) / len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    expected = sum(
        ((a - b) / 4.0) ** 2
        * (left_counts[a] / len(left))
        * (right_counts[b] / len(right))
        for a in categories
        for b in categories
    )
    return 1.0 if expected == 0 and observed == 0 else 0.0 if expected == 0 else round(1 - observed / expected, 4)
