from __future__ import annotations

from collections import Counter
from math import prod

from .metrics import compute_metrics
from .models import AttackSequence
from .research import AttackFeatures


def _coordinate_span(values: list[float]) -> float:
    return max(values) - min(values) if len(values) >= 2 else 0.0


def build_attack_features(
    sequence: AttackSequence,
    *,
    quality_metrics: dict[str, float] | None = None,
) -> AttackFeatures:
    metrics = compute_metrics(sequence)
    action_counts = dict(Counter(event.event_type for event in sequence.events))
    x_values = [
        value
        for event in sequence.events
        for value in (event.x, event.end_x)
        if value is not None
    ]
    y_values = [
        value
        for event in sequence.events
        for value in (event.y, event.end_y)
        if value is not None
    ]
    xg_values = [
        event.xg
        for event in sequence.events
        if event.event_type == "shot" and event.xg is not None
    ]
    evidence_ids = tuple(
        event.evidence_id or f"event:{sequence.attack_id}:{index}"
        for index, event in enumerate(sequence.events)
    )
    duration = max(0.0, sequence.end_sec - sequence.start_sec)
    uncertainty = {
        "event": round(1.0 - sum(item.confidence for item in sequence.events) / len(sequence.events), 4)
    }
    for key, value in (quality_metrics or {}).items():
        uncertainty[key] = round(1.0 - min(1.0, max(0.0, float(value))), 4)

    result = AttackFeatures(
        attack_id=sequence.attack_id,
        match_id=sequence.match_id,
        team=sequence.team,
        duration_sec=round(duration, 3),
        event_count=len(sequence.events),
        action_counts=action_counts,
        progression_m=metrics.progression_m,
        progression_ratio=metrics.progression_ratio,
        width_m=round(_coordinate_span(y_values), 3),
        depth_m=round(_coordinate_span(x_values), 3),
        entered_final_third=metrics.entered_final_third,
        entered_penalty_box=metrics.entered_penalty_box,
        shots=metrics.shots,
        shots_on_target=metrics.shots_on_target,
        goals=metrics.goals,
        xg_sum=metrics.xg_sum,
        xg_max=round(max(xg_values, default=0.0), 4),
        attack_goal_probability=round(1.0 - prod(1.0 - value for value in xg_values), 4),
        positive_xt_gain=metrics.positive_xt_gain,
        tempo_events_per_sec=round(len(sequence.events) / max(duration, 1.0), 4),
        nearby_defenders_mean=None,
        ended_in_turnover=metrics.ended_in_turnover,
        source=sequence.source,
        uncertainty=uncertainty,
        evidence_ids=evidence_ids,
    )
    result.validate()
    return result
