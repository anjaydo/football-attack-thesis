from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, TypeVar


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _optional_number(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


@dataclass(frozen=True)
class Event:
    timestamp: float
    event_type: str
    team: str
    x: float | None = None
    y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    outcome: str | None = None
    xg: float | None = None
    xg_source: str | None = None
    xg_inputs: dict[str, float | int | None] | None = None
    xt_before: float | None = None
    xt_after: float | None = None
    frame: int | None = None
    player_id: str | None = None
    jersey: int | None = None
    role: str | None = None
    confidence: float = 1.0
    source: str = "structured"
    evidence_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        event = cls(
            timestamp=float(data["timestamp"]),
            event_type=str(data["event_type"]).lower(),
            team=str(data["team"]),
            x=_optional_number(data, "x"),
            y=_optional_number(data, "y"),
            end_x=_optional_number(data, "end_x"),
            end_y=_optional_number(data, "end_y"),
            outcome=(str(data["outcome"]).lower() if data.get("outcome") else None),
            xg=_optional_number(data, "xg"),
            xg_source=(str(data["xg_source"]) if data.get("xg_source") else None),
            xg_inputs=(
                {
                    str(key): (None if value is None else float(value))
                    for key, value in data["xg_inputs"].items()
                }
                if data.get("xg_inputs") is not None
                else None
            ),
            xt_before=_optional_number(data, "xt_before"),
            xt_after=_optional_number(data, "xt_after"),
            frame=(int(data["frame"]) if data.get("frame") is not None else None),
            player_id=(str(data["player_id"]) if data.get("player_id") is not None else None),
            jersey=(int(data["jersey"]) if data.get("jersey") is not None else None),
            role=(str(data["role"]) if data.get("role") is not None else None),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "structured")),
            evidence_id=(str(data["evidence_id"]) if data.get("evidence_id") else None),
        )
        event.validate()
        return event

    def validate(self) -> None:
        for key, value, upper in (
            ("x", self.x, PITCH_LENGTH),
            ("end_x", self.end_x, PITCH_LENGTH),
            ("y", self.y, PITCH_WIDTH),
            ("end_y", self.end_y, PITCH_WIDTH),
        ):
            if value is not None and not 0.0 <= value <= upper:
                raise ValueError(f"{key} must be between 0 and {upper}")
        if self.xg is not None and not 0.0 <= self.xg <= 1.0:
            raise ValueError("xg must be between 0 and 1")
        if self.xg_source is not None and self.xg is None:
            raise ValueError("xg_source requires xg")
        if self.xg_inputs is not None:
            if self.xg is None:
                raise ValueError("xg_inputs require xg")
            forbidden = {"goal", "statsbomb_xg", "target", "outcome"}
            if forbidden.intersection(key.casefold() for key in self.xg_inputs):
                raise ValueError("xg_inputs contain target leakage")
            if any(
                value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
                for value in self.xg_inputs.values()
            ):
                raise ValueError("xg_inputs must contain only numeric values or null")
        if self.frame is not None and self.frame < 0:
            raise ValueError("frame must be non-negative")
        if self.jersey is not None and self.jersey < 0:
            raise ValueError("jersey must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.source:
            raise ValueError("source must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class AttackSequence:
    attack_id: str
    team: str
    start_sec: float
    end_sec: float
    events: tuple[Event, ...]
    match_id: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    termination_reason: str | None = None
    source: str = "structured"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackSequence":
        sequence = cls(
            attack_id=str(data["attack_id"]),
            team=str(data["team"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            events=tuple(Event.from_dict(item) for item in data["events"]),
            match_id=(str(data["match_id"]) if data.get("match_id") else None),
            start_frame=(int(data["start_frame"]) if data.get("start_frame") is not None else None),
            end_frame=(int(data["end_frame"]) if data.get("end_frame") is not None else None),
            termination_reason=(
                str(data["termination_reason"]) if data.get("termination_reason") else None
            ),
            source=str(data.get("source", "structured")),
        )
        sequence.validate()
        return sequence

    def validate(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be greater than or equal to start_sec")
        if not self.events:
            raise ValueError("an attack must contain at least one event")
        if any(event.team != self.team for event in self.events):
            raise ValueError("all events must belong to the attacking team")
        timestamps = [event.timestamp for event in self.events]
        if timestamps != sorted(timestamps):
            raise ValueError("events must be ordered by timestamp")
        if timestamps[0] < self.start_sec or timestamps[-1] > self.end_sec:
            raise ValueError("event timestamps must fall within the attack interval")
        if (self.start_frame is None) != (self.end_frame is None):
            raise ValueError("start_frame and end_frame must be provided together")
        if self.start_frame is not None and self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")
        if not self.source:
            raise ValueError("source must not be empty")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "attack_id": self.attack_id,
            "team": self.team,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "events": [event.to_dict() for event in self.events],
        }
        for key in (
            "match_id",
            "start_frame",
            "end_frame",
            "termination_reason",
            "source",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


# Attack-v2 is intentionally additive.  The legacy Event/AttackSequence
# contracts above remain available while callers migrate to the stricter
# possession-first domain model below.
ATTACK_V2_SCHEMA_VERSION = "2.0"
ATTACK_V2_RUBRIC_VERSION = "attack-v2-rubric-1"


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SourceTier(_StringEnum):
    ORACLE_ACTIONS_ORACLE_TACTICAL = "oracle_actions_oracle_tactical"
    PREDICTED_ACTIONS_ORACLE_TACTICAL = "predicted_actions_oracle_tactical"
    VIDEO_ONLY = "video_only"
    FUSED = "fused"
    LEGACY_UNVALIDATED = "legacy_unvalidated"


class SequenceStatus(_StringEnum):
    COMPLETE = "complete"
    LEFT_CENSORED = "left_censored"
    RIGHT_CENSORED = "right_censored"
    UNCERTAIN = "uncertain"


class TerminalType(_StringEnum):
    SHOT_ATTEMPT = "shot_attempt"
    HEADED_ATTEMPT = "headed_attempt"
    FORCED_GOAL_LINE_OUT = "forced_goal_line_out"
    TURNOVER = "turnover"
    UNFORCED_OUT = "unforced_out"
    FOUL = "foul"
    OFFSIDE = "offside"
    PERIOD_END = "period_end"
    UNKNOWN = "unknown"


ATTACK_TERMINAL_ALLOWLIST = frozenset(
    {
        TerminalType.SHOT_ATTEMPT,
        TerminalType.HEADED_ATTEMPT,
        TerminalType.FORCED_GOAL_LINE_OUT,
    }
)

FORCED_GOAL_LINE_OUT_CAUSES = frozenset(
    {
        "keeper_save_out",
        "defender_block_out",
        "defender_deflection_out",
        "defender_clearance_out",
    }
)


class PossessionState(_StringEnum):
    DEAD_BALL = "dead_ball"
    CONTROLLED = "controlled"
    CONTESTED = "contested"
    BALL_FLIGHT = "ball_flight"
    UNKNOWN = "unknown"


EnumType = TypeVar("EnumType", bound=Enum)


def _coerce_enum(value: object, enum_type: type[EnumType], field_name: str) -> EnumType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as error:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {allowed}") from error


def _validate_probability(value: float | None, field_name: str) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _normalise_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _stable_content_id(prefix: str, payload: dict[str, object]) -> str:
    serialised = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(serialised).hexdigest()[:24]}"


@dataclass(frozen=True)
class FieldProvenanceV2:
    field_name: str
    source_tier: SourceTier
    source_record_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_tier",
            _coerce_enum(self.source_tier, SourceTier, "source_tier"),
        )
        object.__setattr__(self, "field_name", self.field_name.strip())
        object.__setattr__(self, "evidence_ids", _normalise_strings(self.evidence_ids))
        if not self.field_name:
            raise ValueError("field_name must not be empty")
        _validate_probability(self.confidence, "confidence")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldProvenanceV2":
        return cls(
            field_name=str(data["field_name"]),
            source_tier=_coerce_enum(data["source_tier"], SourceTier, "source_tier"),
            source_record_id=(
                str(data["source_record_id"])
                if data.get("source_record_id") is not None
                else None
            ),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
            confidence=_optional_number(data, "confidence"),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "field_name": self.field_name,
            "source_tier": self.source_tier.value,
            "evidence_ids": list(self.evidence_ids),
        }
        if self.source_record_id is not None:
            result["source_record_id"] = self.source_record_id
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass(frozen=True)
class ObservedActionV2:
    match_id: str
    period: int
    frame: int
    timestamp_sec: float
    action_type: str
    source_tier: SourceTier
    actor_team_id: str | None = None
    possession_team_id: str | None = None
    beneficiary_team_id: str | None = None
    raw_left_to_right: bool | None = None
    x: float | None = None
    y: float | None = None
    actor_id: str | None = None
    jersey: int | None = None
    role: str | None = None
    bbox_visible: bool | None = None
    replay: bool = False
    body_part: str | None = None
    outcome: str | None = None
    qualifiers: tuple[str, ...] = ()
    action_probability: float | None = None
    actor_probability: float | None = None
    timestamp_probability: float | None = None
    field_provenance: tuple[FieldProvenanceV2, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    association_id: str | None = None
    action_id: str = ""
    schema_version: str = ATTACK_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_tier",
            _coerce_enum(self.source_tier, SourceTier, "source_tier"),
        )
        object.__setattr__(self, "match_id", self.match_id.strip())
        object.__setattr__(self, "action_type", self.action_type.strip().lower())
        object.__setattr__(
            self,
            "body_part",
            self.body_part.strip().lower() if self.body_part is not None else None,
        )
        object.__setattr__(
            self,
            "outcome",
            self.outcome.strip().lower() if self.outcome is not None else None,
        )
        object.__setattr__(
            self,
            "qualifiers",
            _normalise_strings(value.lower() for value in self.qualifiers),
        )
        object.__setattr__(self, "evidence_ids", _normalise_strings(self.evidence_ids))
        object.__setattr__(
            self,
            "association_id",
            self.association_id.strip() if self.association_id is not None else None,
        )
        provenance = tuple(
            item if isinstance(item, FieldProvenanceV2) else FieldProvenanceV2.from_dict(item)
            for item in self.field_provenance
        )
        object.__setattr__(self, "field_provenance", provenance)
        expected_id = self.expected_action_id()
        if not self.action_id:
            object.__setattr__(self, "action_id", expected_id)
        self.validate(expected_id=expected_id)

    def expected_action_id(self) -> str:
        # Inferred possession, probabilities, and geometry deliberately do not
        # participate: enriching an observation must not change its join key.
        return _stable_content_id(
            "actv2",
            {
                "match_id": self.match_id,
                "period": self.period,
                "frame": self.frame,
                "timestamp_sec": round(float(self.timestamp_sec), 6),
                "action_type": self.action_type,
                "actor_team_id": self.actor_team_id,
                "actor_id": self.actor_id,
                "jersey": self.jersey,
                "raw_left_to_right": self.raw_left_to_right,
            },
        )

    def validate(self, *, expected_id: str | None = None) -> None:
        if self.schema_version != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ATTACK_V2_SCHEMA_VERSION}")
        if not self.match_id:
            raise ValueError("match_id must not be empty")
        if self.period < 1:
            raise ValueError("period must be at least 1")
        if self.frame < 0:
            raise ValueError("frame must be non-negative")
        if self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be non-negative")
        if not self.action_type:
            raise ValueError("action_type must not be empty")
        if self.x is not None and not 0.0 <= self.x <= PITCH_LENGTH:
            raise ValueError(f"x must be between 0 and {PITCH_LENGTH}")
        if self.y is not None and not 0.0 <= self.y <= PITCH_WIDTH:
            raise ValueError(f"y must be between 0 and {PITCH_WIDTH}")
        if self.jersey is not None and self.jersey < 0:
            raise ValueError("jersey must be non-negative")
        for name in ("action_probability", "actor_probability", "timestamp_probability"):
            _validate_probability(getattr(self, name), name)
        expected = expected_id or self.expected_action_id()
        if self.action_id != expected:
            raise ValueError("action_id does not match the stable content ID")
        field_names = [item.field_name for item in self.field_provenance]
        if len(field_names) != len(set(field_names)):
            raise ValueError("field_provenance must contain at most one record per field")
        if self.source_tier is SourceTier.FUSED:
            required = {
                "frame",
                "timestamp_sec",
                "action_type",
                "replay",
            }
            for field_name in (
                "actor_team_id",
                "possession_team_id",
                "beneficiary_team_id",
                "raw_left_to_right",
                "x",
                "y",
                "actor_id",
                "jersey",
                "role",
                "bbox_visible",
                "body_part",
                "outcome",
            ):
                if getattr(self, field_name) is not None:
                    required.add(field_name)
            missing = required.difference(field_names)
            if missing:
                raise ValueError(
                    "fused actions require field provenance for: " + ", ".join(sorted(missing))
                )
            native_sources = {item.source_tier for item in self.field_provenance}
            if SourceTier.FUSED in native_sources or SourceTier.LEGACY_UNVALIDATED in native_sources:
                raise ValueError(
                    "fused field provenance must identify native non-fused source tiers"
                )
            if len(native_sources) < 2:
                raise ValueError("fused actions require at least two independently sourced fields")
            if not self.association_id or self.association_id not in self.evidence_ids:
                raise ValueError(
                    "fused actions require an association_id included in evidence_ids"
                )
        elif self.association_id is not None:
            raise ValueError("association_id is reserved for genuinely fused actions")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservedActionV2":
        return cls(
            match_id=str(data["match_id"]),
            period=int(data["period"]),
            frame=int(data["frame"]),
            timestamp_sec=float(data["timestamp_sec"]),
            action_type=str(data["action_type"]),
            source_tier=_coerce_enum(data["source_tier"], SourceTier, "source_tier"),
            actor_team_id=(
                str(data["actor_team_id"]) if data.get("actor_team_id") is not None else None
            ),
            possession_team_id=(
                str(data["possession_team_id"])
                if data.get("possession_team_id") is not None
                else None
            ),
            beneficiary_team_id=(
                str(data["beneficiary_team_id"])
                if data.get("beneficiary_team_id") is not None
                else None
            ),
            raw_left_to_right=(
                bool(data["raw_left_to_right"])
                if data.get("raw_left_to_right") is not None
                else None
            ),
            x=_optional_number(data, "x"),
            y=_optional_number(data, "y"),
            actor_id=(str(data["actor_id"]) if data.get("actor_id") is not None else None),
            jersey=(int(data["jersey"]) if data.get("jersey") is not None else None),
            role=(str(data["role"]) if data.get("role") is not None else None),
            bbox_visible=(
                bool(data["bbox_visible"]) if data.get("bbox_visible") is not None else None
            ),
            replay=bool(data.get("replay", False)),
            body_part=(str(data["body_part"]) if data.get("body_part") is not None else None),
            outcome=(str(data["outcome"]) if data.get("outcome") is not None else None),
            qualifiers=tuple(str(value) for value in data.get("qualifiers", ())),
            action_probability=_optional_number(data, "action_probability"),
            actor_probability=_optional_number(data, "actor_probability"),
            timestamp_probability=_optional_number(data, "timestamp_probability"),
            field_provenance=tuple(
                FieldProvenanceV2.from_dict(item) for item in data.get("field_provenance", ())
            ),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
            association_id=(
                str(data["association_id"])
                if data.get("association_id") is not None
                else None
            ),
            action_id=str(data.get("action_id", "")),
            schema_version=str(data.get("schema_version", ATTACK_V2_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "match_id": self.match_id,
            "period": self.period,
            "frame": self.frame,
            "timestamp_sec": self.timestamp_sec,
            "action_type": self.action_type,
            "source_tier": self.source_tier.value,
            "replay": self.replay,
            "qualifiers": list(self.qualifiers),
            "field_provenance": [item.to_dict() for item in self.field_provenance],
            "evidence_ids": list(self.evidence_ids),
        }
        for field_name in (
            "actor_team_id",
            "possession_team_id",
            "beneficiary_team_id",
            "raw_left_to_right",
            "x",
            "y",
            "actor_id",
            "jersey",
            "role",
            "bbox_visible",
            "body_part",
            "outcome",
            "action_probability",
            "actor_probability",
            "timestamp_probability",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        if self.association_id is not None:
            result["association_id"] = self.association_id
        return result


@dataclass(frozen=True)
class TerminalEventV2:
    match_id: str
    period: int
    frame: int
    timestamp_sec: float
    terminal_type: TerminalType
    possession_team_id: str
    source_tier: SourceTier
    beneficiary_team_id: str | None = None
    last_touch_team_id: str | None = None
    cause: str | None = None
    body_part: str | None = None
    outcome: str | None = None
    confidence: float | None = None
    approximate_time: bool = False
    action_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    terminal_id: str = ""
    schema_version: str = ATTACK_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_type",
            _coerce_enum(self.terminal_type, TerminalType, "terminal_type"),
        )
        object.__setattr__(
            self,
            "source_tier",
            _coerce_enum(self.source_tier, SourceTier, "source_tier"),
        )
        object.__setattr__(self, "match_id", self.match_id.strip())
        object.__setattr__(self, "possession_team_id", self.possession_team_id.strip())
        object.__setattr__(
            self,
            "beneficiary_team_id",
            self.beneficiary_team_id.strip()
            if self.beneficiary_team_id is not None
            else None,
        )
        object.__setattr__(
            self,
            "last_touch_team_id",
            self.last_touch_team_id.strip()
            if self.last_touch_team_id is not None
            else None,
        )
        object.__setattr__(
            self,
            "cause",
            self.cause.strip().lower() if self.cause is not None else None,
        )
        object.__setattr__(
            self,
            "body_part",
            self.body_part.strip().lower() if self.body_part is not None else None,
        )
        object.__setattr__(
            self,
            "outcome",
            self.outcome.strip().lower() if self.outcome is not None else None,
        )
        object.__setattr__(self, "evidence_ids", _normalise_strings(self.evidence_ids))
        expected_id = self.expected_terminal_id()
        if not self.terminal_id:
            object.__setattr__(self, "terminal_id", expected_id)
        self.validate(expected_id=expected_id)

    @property
    def qualifies_as_attack(self) -> bool:
        return self.terminal_type in ATTACK_TERMINAL_ALLOWLIST

    def expected_terminal_id(self) -> str:
        return _stable_content_id(
            "termv2",
            {
                "match_id": self.match_id,
                "period": self.period,
                "frame": self.frame,
                "timestamp_sec": round(float(self.timestamp_sec), 6),
                "terminal_type": self.terminal_type.value,
                "possession_team_id": self.possession_team_id,
                "beneficiary_team_id": self.beneficiary_team_id,
                "last_touch_team_id": self.last_touch_team_id,
                "cause": self.cause,
                "action_id": self.action_id,
            },
        )

    def validate(self, *, expected_id: str | None = None) -> None:
        if self.schema_version != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ATTACK_V2_SCHEMA_VERSION}")
        if not self.match_id or not self.possession_team_id:
            raise ValueError("match_id and possession_team_id must not be empty")
        if self.period < 1 or self.frame < 0 or self.timestamp_sec < 0:
            raise ValueError("period, frame, and timestamp_sec must be non-negative")
        _validate_probability(self.confidence, "confidence")
        if self.terminal_type is TerminalType.FORCED_GOAL_LINE_OUT:
            if self.beneficiary_team_id != self.possession_team_id:
                raise ValueError(
                    "forced_goal_line_out requires the attacking possession team to receive the restart"
                )
            if (
                not self.last_touch_team_id
                or self.last_touch_team_id == self.possession_team_id
            ):
                raise ValueError(
                    "forced_goal_line_out requires an opponent goalkeeper/defender last touch"
                )
            if not self.cause:
                raise ValueError(
                    "forced_goal_line_out requires an observed save, block, or deflection cause"
                )
            if self.cause not in FORCED_GOAL_LINE_OUT_CAUSES:
                raise ValueError(
                    "forced_goal_line_out cause must be a verified save/block/deflection/clearance"
                )
        if self.terminal_id != (expected_id or self.expected_terminal_id()):
            raise ValueError("terminal_id does not match the stable content ID")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerminalEventV2":
        return cls(
            match_id=str(data["match_id"]),
            period=int(data["period"]),
            frame=int(data["frame"]),
            timestamp_sec=float(data["timestamp_sec"]),
            terminal_type=_coerce_enum(data["terminal_type"], TerminalType, "terminal_type"),
            possession_team_id=str(data["possession_team_id"]),
            source_tier=_coerce_enum(data["source_tier"], SourceTier, "source_tier"),
            beneficiary_team_id=(
                str(data["beneficiary_team_id"])
                if data.get("beneficiary_team_id") is not None
                else None
            ),
            last_touch_team_id=(
                str(data["last_touch_team_id"])
                if data.get("last_touch_team_id") is not None
                else None
            ),
            cause=(str(data["cause"]) if data.get("cause") is not None else None),
            body_part=(str(data["body_part"]) if data.get("body_part") is not None else None),
            outcome=(str(data["outcome"]) if data.get("outcome") is not None else None),
            confidence=_optional_number(data, "confidence"),
            approximate_time=bool(data.get("approximate_time", False)),
            action_id=(str(data["action_id"]) if data.get("action_id") is not None else None),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
            terminal_id=str(data.get("terminal_id", "")),
            schema_version=str(data.get("schema_version", ATTACK_V2_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "terminal_id": self.terminal_id,
            "match_id": self.match_id,
            "period": self.period,
            "frame": self.frame,
            "timestamp_sec": self.timestamp_sec,
            "terminal_type": self.terminal_type.value,
            "possession_team_id": self.possession_team_id,
            "source_tier": self.source_tier.value,
            "approximate_time": self.approximate_time,
            "evidence_ids": list(self.evidence_ids),
        }
        for field_name in (
            "beneficiary_team_id",
            "last_touch_team_id",
            "cause",
            "body_part",
            "outcome",
            "confidence",
            "action_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


@dataclass(frozen=True)
class PossessionStateTransitionV2:
    frame: int
    timestamp_sec: float
    state: PossessionState
    team_id: str | None = None
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state",
            _coerce_enum(self.state, PossessionState, "state"),
        )
        object.__setattr__(self, "evidence_ids", _normalise_strings(self.evidence_ids))
        if self.frame < 0 or self.timestamp_sec < 0:
            raise ValueError("transition frame and timestamp_sec must be non-negative")
        if self.state is PossessionState.CONTROLLED and not self.team_id:
            raise ValueError("controlled transitions require team_id")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PossessionStateTransitionV2":
        return cls(
            frame=int(data["frame"]),
            timestamp_sec=float(data["timestamp_sec"]),
            state=_coerce_enum(data["state"], PossessionState, "state"),
            team_id=(str(data["team_id"]) if data.get("team_id") is not None else None),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "frame": self.frame,
            "timestamp_sec": self.timestamp_sec,
            "state": self.state.value,
            "evidence_ids": list(self.evidence_ids),
        }
        if self.team_id is not None:
            result["team_id"] = self.team_id
        return result


@dataclass(frozen=True)
class PossessionSequenceV2:
    match_id: str
    period: int
    possession_team_id: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    actions: tuple[ObservedActionV2, ...]
    status: SequenceStatus
    source_tier: SourceTier
    terminal: TerminalEventV2 | None = None
    state_transitions: tuple[PossessionStateTransitionV2, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    rubric_version: str = ATTACK_V2_RUBRIC_VERSION
    possession_id: str = ""
    schema_version: str = ATTACK_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_enum(self.status, SequenceStatus, "status"))
        object.__setattr__(
            self,
            "source_tier",
            _coerce_enum(self.source_tier, SourceTier, "source_tier"),
        )
        object.__setattr__(self, "match_id", self.match_id.strip())
        object.__setattr__(self, "possession_team_id", self.possession_team_id.strip())
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "state_transitions", tuple(self.state_transitions))
        object.__setattr__(self, "evidence_ids", _normalise_strings(self.evidence_ids))
        expected_id = self.expected_possession_id()
        if not self.possession_id:
            object.__setattr__(self, "possession_id", expected_id)
        self.validate(expected_id=expected_id)

    def expected_possession_id(self) -> str:
        terminal_frame = self.terminal.frame if self.terminal is not None else self.end_frame
        terminal_type = (
            self.terminal.terminal_type.value
            if self.terminal is not None
            else TerminalType.UNKNOWN.value
        )
        return _stable_content_id(
            "posv2",
            {
                "match_id": self.match_id,
                "period": self.period,
                "possession_team_id": self.possession_team_id,
                "start_frame": self.start_frame,
                "terminal_frame": terminal_frame,
                "terminal_type": terminal_type,
                "rubric_version": self.rubric_version,
            },
        )

    def validate(self, *, expected_id: str | None = None) -> None:
        if self.schema_version != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ATTACK_V2_SCHEMA_VERSION}")
        if not self.match_id or not self.possession_team_id or not self.rubric_version:
            raise ValueError("match_id, possession_team_id, and rubric_version must not be empty")
        if self.period < 1 or self.start_frame < 0:
            raise ValueError("period must be positive and frames must be non-negative")
        if self.end_frame < self.start_frame or self.end_sec < self.start_sec:
            raise ValueError("possession end must not precede its start")
        if not self.actions:
            raise ValueError("a possession must contain at least one observed action")
        ordering = [(item.frame, item.timestamp_sec, item.action_id) for item in self.actions]
        if ordering != sorted(ordering):
            raise ValueError("actions must be ordered by frame, time, and stable ID")
        if len({item.action_id for item in self.actions}) != len(self.actions):
            raise ValueError("a possession cannot contain duplicate actions")
        for action in self.actions:
            if action.match_id != self.match_id or action.period != self.period:
                raise ValueError("all actions must belong to the possession match and period")
            if not self.start_frame <= action.frame <= self.end_frame:
                raise ValueError("action frame must fall within the possession interval")
            if not self.start_sec <= action.timestamp_sec <= self.end_sec:
                raise ValueError("action time must fall within the possession interval")
            if (
                action.possession_team_id is not None
                and action.possession_team_id != self.possession_team_id
            ):
                raise ValueError("action possession_team_id conflicts with its possession")
        transition_order = [
            (item.frame, item.timestamp_sec) for item in self.state_transitions
        ]
        if transition_order != sorted(transition_order):
            raise ValueError("state transitions must be ordered")
        if self.terminal is not None:
            if self.terminal.match_id != self.match_id or self.terminal.period != self.period:
                raise ValueError("terminal must belong to the possession match and period")
            if self.terminal.possession_team_id != self.possession_team_id:
                raise ValueError("terminal possession team must match the possession")
            if not self.start_frame <= self.terminal.frame <= self.end_frame:
                raise ValueError("terminal frame must fall within the possession interval")
            if not self.start_sec <= self.terminal.timestamp_sec <= self.end_sec:
                raise ValueError("terminal time must fall within the possession interval")
        if self.status is SequenceStatus.COMPLETE and (
            self.terminal is None or self.terminal.terminal_type is TerminalType.UNKNOWN
        ):
            raise ValueError("complete possessions require a known terminal")
        if self.source_tier is SourceTier.LEGACY_UNVALIDATED and self.status is not SequenceStatus.UNCERTAIN:
            raise ValueError("legacy_unvalidated possessions must remain uncertain")
        if self.possession_id != (expected_id or self.expected_possession_id()):
            raise ValueError("possession_id does not match the stable content ID")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PossessionSequenceV2":
        return cls(
            match_id=str(data["match_id"]),
            period=int(data["period"]),
            possession_team_id=str(data["possession_team_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            actions=tuple(ObservedActionV2.from_dict(item) for item in data["actions"]),
            status=_coerce_enum(data["status"], SequenceStatus, "status"),
            source_tier=_coerce_enum(data["source_tier"], SourceTier, "source_tier"),
            terminal=(
                TerminalEventV2.from_dict(data["terminal"])
                if data.get("terminal") is not None
                else None
            ),
            state_transitions=tuple(
                PossessionStateTransitionV2.from_dict(item)
                for item in data.get("state_transitions", ())
            ),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
            rubric_version=str(data.get("rubric_version", ATTACK_V2_RUBRIC_VERSION)),
            possession_id=str(data.get("possession_id", "")),
            schema_version=str(data.get("schema_version", ATTACK_V2_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rubric_version": self.rubric_version,
            "possession_id": self.possession_id,
            "match_id": self.match_id,
            "period": self.period,
            "possession_team_id": self.possession_team_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "status": self.status.value,
            "source_tier": self.source_tier.value,
            "actions": [item.to_dict() for item in self.actions],
            "terminal": self.terminal.to_dict() if self.terminal is not None else None,
            "state_transitions": [item.to_dict() for item in self.state_transitions],
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class AttackSequenceV2:
    possession: PossessionSequenceV2
    calibrated_confidence: float | None = None
    calibration_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    attack_id: str = ""
    schema_version: str = ATTACK_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        evidence_ids = self.evidence_ids
        if not evidence_ids and self.possession.terminal is not None:
            evidence_ids = (
                self.possession.terminal.terminal_id,
                *self.possession.terminal.evidence_ids,
            )
        object.__setattr__(self, "evidence_ids", _normalise_strings(evidence_ids))
        expected_id = self.expected_attack_id()
        if not self.attack_id:
            object.__setattr__(self, "attack_id", expected_id)
        self.validate(expected_id=expected_id)

    @property
    def match_id(self) -> str:
        return self.possession.match_id

    @property
    def period(self) -> int:
        return self.possession.period

    @property
    def team_id(self) -> str:
        return self.possession.possession_team_id

    @property
    def terminal(self) -> TerminalEventV2:
        if self.possession.terminal is None:  # guarded by validate; keeps typing precise
            raise RuntimeError("validated attacks always have a terminal")
        return self.possession.terminal

    def expected_attack_id(self) -> str:
        terminal = self.possession.terminal
        return _stable_content_id(
            "atkv2",
            {
                "match_id": self.possession.match_id,
                "period": self.possession.period,
                "possession_team_id": self.possession.possession_team_id,
                "start_frame": self.possession.start_frame,
                "terminal_frame": terminal.frame if terminal is not None else None,
                "terminal_type": terminal.terminal_type.value if terminal is not None else None,
                "rubric_version": self.possession.rubric_version,
            },
        )

    def validate(self, *, expected_id: str | None = None) -> None:
        if self.schema_version != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ATTACK_V2_SCHEMA_VERSION}")
        if self.possession.status is not SequenceStatus.COMPLETE:
            raise ValueError("only complete possessions can become attacks")
        if self.possession.terminal is None:
            raise ValueError("an attack requires a terminal")
        if self.possession.terminal.terminal_type not in ATTACK_TERMINAL_ALLOWLIST:
            raise ValueError("terminal is not permitted by the Attack-v2 rubric")
        _validate_probability(self.calibrated_confidence, "calibrated_confidence")
        if self.calibrated_confidence is not None and not self.calibration_id:
            raise ValueError("calibrated_confidence requires calibration_id")
        if self.attack_id != (expected_id or self.expected_attack_id()):
            raise ValueError("attack_id does not match the stable content ID")
        if not self.evidence_ids:
            raise ValueError("an attack requires terminal evidence")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttackSequenceV2":
        return cls(
            possession=PossessionSequenceV2.from_dict(data["possession"]),
            calibrated_confidence=_optional_number(data, "calibrated_confidence"),
            calibration_id=(
                str(data["calibration_id"]) if data.get("calibration_id") is not None else None
            ),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
            attack_id=str(data.get("attack_id", "")),
            schema_version=str(data.get("schema_version", ATTACK_V2_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "attack_id": self.attack_id,
            "possession": self.possession.to_dict(),
            "evidence_ids": list(self.evidence_ids),
        }
        if self.calibrated_confidence is not None:
            result["calibrated_confidence"] = self.calibrated_confidence
        if self.calibration_id is not None:
            result["calibration_id"] = self.calibration_id
        return result


@dataclass(frozen=True)
class PossessionValueV2:
    possession_id: str
    source_tier: SourceTier
    peak_progression_m: float | None = None
    box_entry: bool | None = None
    xt: float | None = None
    epv: float | None = None
    p_score_10s: float | None = None
    p_concede_10s: float | None = None
    evidence_ids: tuple[str, ...] = ()
    schema_version: str = ATTACK_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_tier",
            _coerce_enum(self.source_tier, SourceTier, "source_tier"),
        )
        object.__setattr__(self, "possession_id", self.possession_id.strip())
        object.__setattr__(self, "evidence_ids", _normalise_strings(self.evidence_ids))
        if self.schema_version != ATTACK_V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ATTACK_V2_SCHEMA_VERSION}")
        if not self.possession_id:
            raise ValueError("possession_id must not be empty")
        _validate_probability(self.p_score_10s, "p_score_10s")
        _validate_probability(self.p_concede_10s, "p_concede_10s")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PossessionValueV2":
        return cls(
            possession_id=str(data["possession_id"]),
            source_tier=_coerce_enum(data["source_tier"], SourceTier, "source_tier"),
            peak_progression_m=_optional_number(data, "peak_progression_m"),
            box_entry=(bool(data["box_entry"]) if data.get("box_entry") is not None else None),
            xt=_optional_number(data, "xt"),
            epv=_optional_number(data, "epv"),
            p_score_10s=_optional_number(data, "p_score_10s"),
            p_concede_10s=_optional_number(data, "p_concede_10s"),
            evidence_ids=tuple(str(value) for value in data.get("evidence_ids", ())),
            schema_version=str(data.get("schema_version", ATTACK_V2_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "possession_id": self.possession_id,
            "source_tier": self.source_tier.value,
            "evidence_ids": list(self.evidence_ids),
        }
        for field_name in (
            "peak_progression_m",
            "box_entry",
            "xt",
            "epv",
            "p_score_10s",
            "p_concede_10s",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result
