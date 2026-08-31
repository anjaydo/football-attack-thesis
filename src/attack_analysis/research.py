from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal


VerdictLabel = Literal["effective", "promising", "ineffective", "insufficient_evidence"]


@dataclass(frozen=True)
class AttackFeatures:
    attack_id: str
    match_id: str | None
    team: str
    duration_sec: float
    event_count: int
    action_counts: dict[str, int]
    progression_m: float
    progression_ratio: float
    width_m: float
    depth_m: float
    entered_final_third: bool
    entered_penalty_box: bool
    shots: int
    shots_on_target: int
    goals: int
    xg_sum: float
    xg_max: float
    attack_goal_probability: float
    positive_xt_gain: float
    tempo_events_per_sec: float
    nearby_defenders_mean: float | None
    ended_in_turnover: bool
    source: str
    uncertainty: dict[str, float]
    evidence_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.duration_sec < 0:
            raise ValueError("duration_sec must be non-negative")
        if self.event_count < 1:
            raise ValueError("event_count must be positive")
        if not 0.0 <= self.progression_ratio <= 1.0:
            raise ValueError("progression_ratio must be between 0 and 1")
        for name, value in self.uncertainty.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"uncertainty {name} must be between 0 and 1")
        if not self.evidence_ids:
            raise ValueError("at least one evidence_id is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackFeatures":
        item = cls(
            attack_id=str(data["attack_id"]),
            match_id=(str(data["match_id"]) if data.get("match_id") else None),
            team=str(data["team"]),
            duration_sec=float(data["duration_sec"]),
            event_count=int(data["event_count"]),
            action_counts={str(k): int(v) for k, v in data.get("action_counts", {}).items()},
            progression_m=float(data["progression_m"]),
            progression_ratio=float(data["progression_ratio"]),
            width_m=float(data.get("width_m", 0.0)),
            depth_m=float(data.get("depth_m", 0.0)),
            entered_final_third=bool(data["entered_final_third"]),
            entered_penalty_box=bool(data["entered_penalty_box"]),
            shots=int(data["shots"]),
            shots_on_target=int(data.get("shots_on_target", 0)),
            goals=int(data.get("goals", 0)),
            xg_sum=float(data.get("xg_sum", 0.0)),
            xg_max=float(data.get("xg_max", 0.0)),
            attack_goal_probability=float(data.get("attack_goal_probability", 0.0)),
            positive_xt_gain=float(data.get("positive_xt_gain", 0.0)),
            tempo_events_per_sec=float(data.get("tempo_events_per_sec", 0.0)),
            nearby_defenders_mean=(
                float(data["nearby_defenders_mean"])
                if data.get("nearby_defenders_mean") is not None
                else None
            ),
            ended_in_turnover=bool(data.get("ended_in_turnover", False)),
            source=str(data.get("source", "unknown")),
            uncertainty={str(k): float(v) for k, v in data.get("uncertainty", {}).items()},
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", ())),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_id: str
    role: str
    metric: str
    value_per90: float
    percentile: float
    season: str
    sample_size: int
    source_url: str
    snapshot_date: str
    player_name: str | None = None
    team_name: str | None = None

    def validate(self) -> None:
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError("percentile must be between 0 and 100")
        if self.sample_size < 1:
            raise ValueError("sample_size must be positive")
        date.fromisoformat(self.snapshot_date)
        if not self.source_url:
            raise ValueError("source_url must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRecord":
        item = cls(
            benchmark_id=str(data["benchmark_id"]),
            role=str(data["role"]),
            metric=str(data["metric"]),
            value_per90=float(data["value_per90"]),
            percentile=float(data["percentile"]),
            season=str(data["season"]),
            sample_size=int(data["sample_size"]),
            source_url=str(data["source_url"]),
            snapshot_date=str(data["snapshot_date"]),
            player_name=(str(data["player_name"]) if data.get("player_name") else None),
            team_name=(str(data["team_name"]) if data.get("team_name") else None),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class EvidenceBundle:
    attack: AttackFeatures
    quality_gates: dict[str, bool]
    quality_metrics: dict[str, float]
    benchmarks: tuple[BenchmarkRecord, ...] = ()
    source_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    spatial_context: dict[str, Any] | None = None
    xg_raw: dict[str, Any] | None = None

    @property
    def downstream_ready(self) -> bool:
        required = ["events", "calibration", "tracking"]
        if "attack_qualified" in self.quality_gates:
            required.append("attack_qualified")
        return all(self.quality_gates.get(key, False) for key in required)

    def evidence_ids(self) -> frozenset[str]:
        ids = set(self.attack.evidence_ids)
        ids.update(record.benchmark_id for record in self.benchmarks)
        ids.update(self.source_ids)
        if self.spatial_context:
            freeze_id = self.spatial_context.get("footpass_evidence_id")
            if freeze_id:
                ids.add(str(freeze_id))
        return frozenset(ids)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "attack": self.attack.to_dict(),
            "quality_gates": dict(self.quality_gates),
            "quality_metrics": dict(self.quality_metrics),
            "benchmarks": [item.to_dict() for item in self.benchmarks],
            "source_ids": list(self.source_ids),
            "limitations": list(self.limitations),
            "downstream_ready": self.downstream_ready,
        }
        if self.spatial_context:
            payload["spatial_context"] = dict(self.spatial_context)
        if self.xg_raw:
            payload["xg_raw"] = dict(self.xg_raw)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceBundle":
        return cls(
            attack=AttackFeatures.from_dict(data["attack"]),
            quality_gates={str(k): bool(v) for k, v in data.get("quality_gates", {}).items()},
            quality_metrics={str(k): float(v) for k, v in data.get("quality_metrics", {}).items()},
            benchmarks=tuple(BenchmarkRecord.from_dict(item) for item in data.get("benchmarks", ())),
            source_ids=tuple(str(item) for item in data.get("source_ids", ())),
            limitations=tuple(str(item) for item in data.get("limitations", ())),
            spatial_context=(
                dict(data["spatial_context"]) if data.get("spatial_context") else None
            ),
            xg_raw=dict(data["xg_raw"]) if data.get("xg_raw") else None,
        )


@dataclass(frozen=True)
class EvidenceClaim:
    text: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class EvaluationVerdict:
    attack_id: str
    label: VerdictLabel
    score: int
    confidence: float
    claims: tuple[EvidenceClaim, ...]
    limitations: tuple[str, ...] = ()
    abstained: bool = False
    model: str | None = None
    prompt_version: str | None = None
    tool_trace: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    usage: dict[str, int] = field(default_factory=dict)

    def validate(self, available_evidence: frozenset[str] | None = None) -> None:
        if not 1 <= self.score <= 5:
            raise ValueError("score must be between 1 and 5")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.abstained != (self.label == "insufficient_evidence"):
            raise ValueError("abstained must match insufficient_evidence label")
        if not self.abstained and not self.claims:
            raise ValueError("a non-abstained verdict requires evidence claims")
        if available_evidence is not None:
            unsupported = {
                evidence_id
                for claim in self.claims
                for evidence_id in claim.evidence_ids
                if evidence_id not in available_evidence
            }
            if unsupported:
                raise ValueError(f"unsupported evidence ids: {sorted(unsupported)}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "attack_id": self.attack_id,
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "claims": [item.to_dict() for item in self.claims],
            "limitations": list(self.limitations),
            "abstained": self.abstained,
            "tool_trace": list(self.tool_trace),
            "usage": dict(self.usage),
        }
        if self.model is not None:
            result["model"] = self.model
        if self.prompt_version is not None:
            result["prompt_version"] = self.prompt_version
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationVerdict":
        label = str(data["label"])
        if label not in {"effective", "promising", "ineffective", "insufficient_evidence"}:
            raise ValueError(f"unsupported verdict label: {label}")
        item = cls(
            attack_id=str(data["attack_id"]),
            label=label,  # type: ignore[arg-type]
            score=int(data["score"]),
            confidence=float(data["confidence"]),
            claims=tuple(
                EvidenceClaim(
                    text=str(claim["text"]),
                    evidence_ids=tuple(str(value) for value in claim.get("evidence_ids", ())),
                )
                for claim in data.get("claims", ())
            ),
            limitations=tuple(str(value) for value in data.get("limitations", ())),
            abstained=bool(data.get("abstained", False)),
            model=(str(data["model"]) if data.get("model") else None),
            prompt_version=(str(data["prompt_version"]) if data.get("prompt_version") else None),
            tool_trace=tuple(dict(value) for value in data.get("tool_trace", ())),
            usage={str(k): int(v) for k, v in data.get("usage", {}).items()},
        )
        item.validate()
        return item


CHANCE_QUALITY_VALUES = ("high", "medium", "low", "unknown")


@dataclass(frozen=True)
class AttackConclusionV2:
    attack_id: str
    chance_quality: str
    xg_probability: float | None
    tactical_read: str
    structural_features: tuple[str, ...]
    comparison_to_role_benchmark: str | None
    claims: tuple[EvidenceClaim, ...]
    limitations: tuple[str, ...]
    abstained_effectiveness: bool = True
    effectiveness_verdict: None = None
    model: str | None = None
    prompt_version: str | None = None
    tool_trace: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def validate(self, available_evidence: frozenset[str] | None = None) -> None:
        if self.chance_quality not in CHANCE_QUALITY_VALUES:
            raise ValueError(f"unsupported chance_quality: {self.chance_quality}")
        if self.effectiveness_verdict is not None:
            raise ValueError("categorical effectiveness is withheld for Attack-v2 conclusions")
        if not self.abstained_effectiveness:
            raise ValueError("Attack-v2 conclusions must abstain from effectiveness")
        if not self.tactical_read.strip():
            raise ValueError("tactical_read is required")
        if available_evidence is not None:
            unsupported = {
                evidence_id
                for claim in self.claims
                for evidence_id in claim.evidence_ids
                if evidence_id not in available_evidence
            }
            if unsupported:
                raise ValueError(f"unsupported evidence ids: {sorted(unsupported)}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = {
            "schema_version": "attack-conclusion-v2/1.0",
            "attack_id": self.attack_id,
            "chance_quality": self.chance_quality,
            "xg_probability": self.xg_probability,
            "tactical_read": self.tactical_read,
            "structural_features": list(self.structural_features),
            "comparison_to_role_benchmark": self.comparison_to_role_benchmark,
            "claims": [item.to_dict() for item in self.claims],
            "limitations": list(self.limitations),
            "abstained_effectiveness": self.abstained_effectiveness,
            "effectiveness_verdict": self.effectiveness_verdict,
            "tool_trace": list(self.tool_trace),
        }
        if self.model is not None:
            result["model"] = self.model
        if self.prompt_version is not None:
            result["prompt_version"] = self.prompt_version
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackConclusionV2":
        item = cls(
            attack_id=str(data["attack_id"]),
            chance_quality=str(data.get("chance_quality") or "unknown"),
            xg_probability=(
                float(data["xg_probability"]) if data.get("xg_probability") is not None else None
            ),
            tactical_read=str(data.get("tactical_read") or ""),
            structural_features=tuple(str(value) for value in data.get("structural_features", ())),
            comparison_to_role_benchmark=(
                str(data["comparison_to_role_benchmark"])
                if data.get("comparison_to_role_benchmark")
                else None
            ),
            claims=tuple(
                EvidenceClaim(
                    text=str(claim["text"]),
                    evidence_ids=tuple(str(value) for value in claim.get("evidence_ids", ())),
                )
                for claim in data.get("claims", ())
            ),
            limitations=tuple(str(value) for value in data.get("limitations", ())),
            abstained_effectiveness=bool(data.get("abstained_effectiveness", True)),
            effectiveness_verdict=None,
            model=(str(data["model"]) if data.get("model") else None),
            prompt_version=(str(data["prompt_version"]) if data.get("prompt_version") else None),
            tool_trace=tuple(dict(value) for value in data.get("tool_trace", ())),
        )
        item.validate()
        return item


def insufficient_evidence_verdict(
    attack_id: str,
    limitations: tuple[str, ...],
    *,
    model: str | None = None,
) -> EvaluationVerdict:
    return EvaluationVerdict(
        attack_id=attack_id,
        label="insufficient_evidence",
        score=1,
        confidence=1.0,
        claims=(),
        limitations=limitations or ("Required quality gates did not pass.",),
        abstained=True,
        model=model,
    )
