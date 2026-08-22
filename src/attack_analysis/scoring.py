from __future__ import annotations

from dataclasses import asdict, dataclass

from .metrics import AttackMetrics, compute_metrics
from .models import AttackSequence


@dataclass(frozen=True)
class ScoreComponents:
    outcome: float
    danger: float
    progression: float
    penetration: float
    control: float


@dataclass(frozen=True)
class Evaluation:
    attack_id: str
    score: float
    is_effective: bool
    label: str
    evidence_coverage: float
    components: ScoreComponents
    metrics: AttackMetrics
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "attack_id": self.attack_id,
            "score": self.score,
            "is_effective": self.is_effective,
            "label": self.label,
            "evidence_coverage": self.evidence_coverage,
            "components": asdict(self.components),
            "metrics": self.metrics.to_dict(),
            "explanation": self.explanation,
        }


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _components(metrics: AttackMetrics) -> ScoreComponents:
    if metrics.goals:
        outcome = 1.0
    elif metrics.shots:
        outcome = _clip(metrics.attack_goal_probability / 0.5)
    else:
        outcome = 0.0

    xg_danger = _clip(metrics.xg_sum / 0.5)
    xt_danger = _clip(metrics.positive_xt_gain / 0.15)
    danger = 0.7 * xg_danger + 0.3 * xt_danger

    if metrics.entered_penalty_box:
        penetration = 1.0
    elif metrics.entered_final_third:
        penetration = 0.45
    else:
        penetration = 0.0

    if metrics.ended_in_turnover:
        control = 0.0
    elif metrics.shots:
        control = 0.6
    else:
        control = 1.0

    return ScoreComponents(
        outcome=round(outcome, 4),
        danger=round(danger, 4),
        progression=round(_clip(metrics.progression_ratio / 0.8), 4),
        penetration=round(penetration, 4),
        control=round(control, 4),
    )


def _explain(metrics: AttackMetrics, score: float, label: str) -> str:
    evidence: list[str] = []
    if metrics.goals:
        evidence.append(f"ghi {metrics.goals} bàn")
    elif metrics.shots:
        evidence.append(
            f"tạo {metrics.shots} cú sút với tổng xG {metrics.xg_sum:.2f}"
        )
    else:
        evidence.append("không tạo được cú sút")

    if metrics.entered_penalty_box:
        evidence.append("đưa bóng vào vòng cấm")
    elif metrics.entered_final_third:
        evidence.append("vào được 1/3 sân cuối")
    else:
        evidence.append("không vào được 1/3 sân cuối")

    evidence.append(f"tịnh tiến {metrics.progression_m:.1f} m")
    if metrics.ended_in_turnover:
        evidence.append("kết thúc bằng mất bóng")

    return f"{label} ({score:.1f}/100): " + "; ".join(evidence) + "."


def evaluate_attack(sequence: AttackSequence) -> Evaluation:
    metrics = compute_metrics(sequence)
    components = _components(metrics)
    score = 100.0 * (
        0.35 * components.outcome
        + 0.30 * components.danger
        + 0.20 * components.progression
        + 0.10 * components.penetration
        + 0.05 * components.control
    )
    score = round(score, 1)

    if score >= 60.0:
        label = "effective"
    elif score >= 45.0:
        label = "promising"
    else:
        label = "ineffective"

    # This measures feature coverage, not calibrated predictive confidence.
    evidence_count = int(metrics.shots > 0) + int(metrics.xg_sum > 0) + int(
        metrics.positive_xt_gain > 0
    ) + int(metrics.entered_final_third)
    evidence_coverage = round(evidence_count / 4.0, 2)

    return Evaluation(
        attack_id=sequence.attack_id,
        score=score,
        is_effective=score >= 60.0,
        label=label,
        evidence_coverage=evidence_coverage,
        components=components,
        metrics=metrics,
        explanation=_explain(metrics, score, label),
    )
