from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .metrics import FINAL_THIRD_X, PENALTY_BOX_X, PENALTY_BOX_Y_MAX, PENALTY_BOX_Y_MIN
from .models import AttackSequence, Event


PossessionLabel = Literal[
    "normal_possession",
    "attack",
    "dangerous_attack",
    "insufficient_evidence",
]

ATTACK_SCORE_THRESHOLD = 30.0
DANGEROUS_ATTACK_SCORE_THRESHOLD = 60.0
MIN_POSITION_COVERAGE = 0.5
RUBRIC_VERSION = "possession-attack-qualification-v1"


@dataclass(frozen=True)
class PossessionAssessment:
    sequence_id: str
    label: PossessionLabel
    is_attack: bool
    score: float
    confidence: float
    start_x: float | None
    furthest_x: float | None
    peak_progression_m: float
    directness_m_per_sec: float
    position_coverage: float
    final_third_touches: int
    penalty_box_touches: int
    shots: int
    xg_sum: float
    attacking_actions: tuple[str, ...]
    components: dict[str, float]
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    rubric_version: str = RUBRIC_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _event_points(event: Event) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    if event.x is not None and event.y is not None:
        points.append((event.x, event.y))
    if event.end_x is not None and event.end_y is not None:
        points.append((event.end_x, event.end_y))
    return tuple(points)


def assess_possession(sequence: AttackSequence) -> PossessionAssessment:
    """Qualify a possession before downstream attack-effectiveness evaluation.

    The score is a transparent baseline, not a calibrated probability. It rewards
    chance creation, penetration and peak progression; possession duration or raw
    event count alone never makes a sequence an attack.
    """

    positioned = tuple(event for event in sequence.events if _event_points(event))
    position_coverage = len(positioned) / len(sequence.events)
    all_points = tuple(point for event in sequence.events for point in _event_points(event))
    start_x = next((event.x for event in sequence.events if event.x is not None), None)
    furthest_x = max((point[0] for point in all_points), default=None)
    peak_progression = (
        max(0.0, furthest_x - start_x)
        if start_x is not None and furthest_x is not None
        else 0.0
    )
    duration = max(0.0, sequence.end_sec - sequence.start_sec)
    directness = peak_progression / max(1.0, duration)
    final_third_touches = sum(
        any(x >= FINAL_THIRD_X for x, _ in _event_points(event))
        for event in sequence.events
    )
    penalty_box_touches = sum(
        any(
            x >= PENALTY_BOX_X and PENALTY_BOX_Y_MIN <= y <= PENALTY_BOX_Y_MAX
            for x, y in _event_points(event)
        )
        for event in sequence.events
    )
    shot_events = tuple(event for event in sequence.events if event.event_type == "shot")
    xg_sum = sum(event.xg or 0.0 for event in shot_events)
    attacking_actions = tuple(
        event.event_type
        for event in sequence.events
        if event.event_type in {"cross", "header", "shot"}
    )

    chance_score = 0.0
    if shot_events:
        chance_score = 30.0 + min(15.0, xg_sum / 0.3 * 15.0)
    zone_score = 25.0 if penalty_box_touches else 15.0 if final_third_touches else 0.0
    progression_score = min(25.0, peak_progression / 40.0 * 25.0)
    action_score = 8.0 if any(action in {"cross", "header"} for action in attacking_actions) else 0.0
    tempo_score = 7.0 if directness >= 2.0 else 3.0 if directness >= 1.0 else 0.0
    xt_gain = sum(
        max(0.0, event.xt_after - event.xt_before)
        for event in sequence.events
        if event.xt_before is not None and event.xt_after is not None
    )
    value_score = min(10.0, xt_gain / 0.1 * 10.0)
    components = {
        "chance": round(chance_score, 3),
        "zone_penetration": round(zone_score, 3),
        "peak_progression": round(progression_score, 3),
        "attacking_action": round(action_score, 3),
        "directness": round(tempo_score, 3),
        "possession_value": round(value_score, 3),
    }
    score = round(min(100.0, sum(components.values())), 1)

    has_position_evidence = position_coverage >= MIN_POSITION_COVERAGE
    has_explicit_chance = bool(shot_events)
    insufficient = not has_position_evidence and not has_explicit_chance
    if insufficient:
        label: PossessionLabel = "insufficient_evidence"
    elif score >= DANGEROUS_ATTACK_SCORE_THRESHOLD:
        label = "dangerous_attack"
    elif score >= ATTACK_SCORE_THRESHOLD or has_explicit_chance:
        label = "attack"
    else:
        label = "normal_possession"

    reasons: list[str] = []
    if shot_events:
        reasons.append(f"Tạo {len(shot_events)} cú sút với tổng xG {xg_sum:.3f}.")
    if penalty_box_touches:
        reasons.append(f"Có {penalty_box_touches} action chạm vùng cấm.")
    elif final_third_touches:
        reasons.append(f"Có {final_third_touches} action trong 1/3 cuối sân.")
    if peak_progression >= 5.0:
        reasons.append(
            f"Peak progression {peak_progression:.1f} m, đạt xa nhất x={furthest_x:.1f} m."
        )
    if any(action == "cross" for action in attacking_actions):
        reasons.append("Chuỗi có hành động tạt bóng.")
    elif any(action == "header" for action in attacking_actions):
        reasons.append("Chuỗi có hành động đánh đầu.")
    if not reasons:
        reasons.append("Không có đủ tín hiệu penetration hoặc chance creation để gọi là attack.")

    limitations: list[str] = []
    if not shot_events:
        limitations.append("Chuỗi không tạo cú sút; không có xG outcome để đánh giá.")
    if position_coverage < 1.0:
        limitations.append(
            f"Chỉ {position_coverage:.0%} event có tọa độ mặt sân."
        )
    if insufficient:
        limitations.append("Thiếu tọa độ và không có cú sút để phân loại possession đáng tin cậy.")

    mean_event_confidence = sum(event.confidence for event in sequence.events) / len(
        sequence.events
    )
    confidence = mean_event_confidence * (
        0.55 + 0.45 * position_coverage if has_explicit_chance else position_coverage
    )
    return PossessionAssessment(
        sequence_id=sequence.attack_id,
        label=label,
        is_attack=label in {"attack", "dangerous_attack"},
        score=score,
        confidence=round(min(1.0, confidence), 4),
        start_x=round(start_x, 3) if start_x is not None else None,
        furthest_x=round(furthest_x, 3) if furthest_x is not None else None,
        peak_progression_m=round(peak_progression, 3),
        directness_m_per_sec=round(directness, 3),
        position_coverage=round(position_coverage, 4),
        final_third_touches=final_third_touches,
        penalty_box_touches=penalty_box_touches,
        shots=len(shot_events),
        xg_sum=round(xg_sum, 4),
        attacking_actions=attacking_actions,
        components=components,
        reasons=tuple(reasons),
        limitations=tuple(limitations),
    )
