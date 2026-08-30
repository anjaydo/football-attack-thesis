from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping

from .models import ATTACK_V2_RUBRIC_VERSION
from .provenance import file_sha256


GOLD_SCHEMA_VERSION = "attack-gold-v2.0"

HUMAN_TRAIN_MATCHES = (6, 8, 16, 21, 33, 48)
DEVELOPMENT_MATCHES = (18, 24, 47)
FROZEN_TEST_MATCHES = (0, 36, 50)

ANNOTATION_LABELS = ("possession", "attack", "uncertain", "dead_ball")
COMPLETENESS_VALUES = ("complete", "left_censored", "right_censored", "uncertain")
TERMINAL_TYPES = (
    "shot_attempt",
    "headed_attempt",
    "forced_goal_line_out",
    "turnover",
    "unforced_out",
    "foul",
    "offside",
    "period_end",
    "unknown",
)
ATTACK_TERMINALS = frozenset(
    {"shot_attempt", "headed_attempt", "forced_goal_line_out"}
)
FORCED_OUT_CAUSES = frozenset(
    {
        "keeper_save_out",
        "defender_block_out",
        "defender_deflection_out",
        "defender_clearance_out",
    }
)
SHOT_OUTCOMES = (
    "goal",
    "on_target",
    "blocked",
    "off_target",
    "woodwork",
    "unknown",
)

FROZEN_TEST_MATCH_IDS = tuple(f"game_{number}" for number in FROZEN_TEST_MATCHES)
FROZEN_TEST_PERIOD_KEYS = frozenset(
    (match_id, period)
    for match_id in FROZEN_TEST_MATCH_IDS
    for period in (1, 2)
)
FROZEN_TEST_LABEL_FILE = "attack-v2-frozen-test.jsonl"
FROZEN_TEST_SEAL_FILE = "attack-v2-frozen-test.seal.json"
FROZEN_TEST_INTERVAL_FILE = "attack-v2-frozen-test.periods.json"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: object, *, prefix: str) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def match_number(match_id: str | int) -> int:
    if isinstance(match_id, int):
        return match_id
    normalized = str(match_id).split("_H", 1)[0]
    if normalized.startswith("game_"):
        normalized = normalized[5:]
    return int(normalized)


def gold_split(match_id: str | int) -> str:
    number = match_number(match_id)
    if number in HUMAN_TRAIN_MATCHES:
        return "train"
    if number in DEVELOPMENT_MATCHES:
        return "development"
    if number in FROZEN_TEST_MATCHES:
        return "test"
    raise ValueError(f"match {match_id!r} is not part of the frozen Attack-v2 gold split")


def canonical_match_id(match_id: str | int) -> str:
    """Return the split-level match identifier without a half suffix."""

    return f"game_{match_number(match_id)}"


@dataclass(frozen=True)
class GoldPeriodIntervalV2:
    """Declared inclusive frame range for one continuously labelled period.

    Annotation frame ranges are inclusive. Consequently two adjacent labels
    must satisfy ``next.start_frame == previous.end_frame + 1``. Times are
    checked against the declared FPS rather than used as an ambiguous second
    interval convention.
    """

    match_id: str
    period: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    fps: float = 25.0

    @property
    def key(self) -> tuple[str, int]:
        return canonical_match_id(self.match_id), self.period

    def validate(self) -> None:
        gold_split(self.match_id)
        if self.period not in (1, 2):
            raise ValueError("period must be 1 or 2")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("declared period frame bounds are invalid")
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError("declared period time bounds are invalid")
        if self.fps <= 0:
            raise ValueError("declared period FPS must be positive")
        expected_end_sec = self.start_sec + (
            (self.end_frame - self.start_frame) / self.fps
        )
        tolerance = max(1e-3, 0.51 / self.fps)
        if abs(self.end_sec - expected_end_sec) > tolerance:
            raise ValueError(
                "declared period time/frame bounds disagree with the declared FPS"
            )

    def expected_sec(self, frame: int) -> float:
        return self.start_sec + ((frame - self.start_frame) / self.fps)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "match_id": canonical_match_id(self.match_id),
            "period": self.period,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "fps": self.fps,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "GoldPeriodIntervalV2":
        item = cls(
            match_id=str(data["match_id"]),
            period=int(data["period"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            fps=float(data.get("fps", 25.0)),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class GoldSequenceAnnotationV2:
    match_id: str
    period: int
    stable_team_id: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    label: str
    terminal_type: str
    terminal_frame: int | None
    terminal_sec: float | None
    terminal_cause: str | None
    shot_body_part: str | None
    shot_outcome: str | None
    beneficiary_team_id: str | None
    last_touch_team_id: str | None
    replay: bool
    completeness: str
    confidence: float
    reason_codes: tuple[str, ...]
    notes: str
    annotator_id: str
    annotated_at: str
    annotation_round: int = 1
    rubric_version: str = ATTACK_V2_RUBRIC_VERSION
    annotation_id: str | None = None

    def expected_stable_id(self) -> str:
        """Return the identity hash independently of any supplied annotation ID."""

        identity = {
            "match_id": canonical_match_id(self.match_id),
            "period": self.period,
            "stable_team_id": self.stable_team_id,
            "start_frame": self.start_frame,
            "terminal_frame": self.terminal_frame,
            "terminal_type": self.terminal_type,
            "rubric_version": self.rubric_version,
        }
        return _stable_hash(identity, prefix="gold-sequence")

    def with_stable_id(self) -> "GoldSequenceAnnotationV2":
        if self.annotation_id:
            return self
        return replace(self, annotation_id=self.expected_stable_id())

    def validate(self) -> None:
        gold_split(self.match_id)
        if self.period not in (1, 2):
            raise ValueError("period must be 1 or 2")
        if self.label in {"possession", "attack"} and not self.stable_team_id.strip():
            raise ValueError("possession/attack labels require a stable_team_id")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("frame bounds are invalid")
        if self.start_sec < 0 or self.end_sec < self.start_sec:
            raise ValueError("time bounds are invalid")
        if self.label not in ANNOTATION_LABELS:
            raise ValueError(f"label must be one of {ANNOTATION_LABELS}")
        if self.terminal_type not in TERMINAL_TYPES:
            raise ValueError(f"terminal_type must be one of {TERMINAL_TYPES}")
        if self.completeness not in COMPLETENESS_VALUES:
            raise ValueError(f"completeness must be one of {COMPLETENESS_VALUES}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.annotation_round < 1:
            raise ValueError("annotation_round must be positive")
        if self.rubric_version != ATTACK_V2_RUBRIC_VERSION:
            raise ValueError(
                f"rubric_version must be {ATTACK_V2_RUBRIC_VERSION}"
            )
        if not self.annotator_id.strip():
            raise ValueError("annotator_id must not be empty")
        if not self.reason_codes:
            raise ValueError("at least one reason_code is required")
        if any(not item.strip() for item in self.reason_codes):
            raise ValueError("reason_codes must not contain empty values")
        try:
            datetime.fromisoformat(self.annotated_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("annotated_at must be an ISO-8601 timestamp") from error

        is_attack_terminal = self.terminal_type in ATTACK_TERMINALS
        if self.label == "attack":
            if self.completeness != "complete":
                raise ValueError("an attack annotation must be complete")
            if not is_attack_terminal:
                raise ValueError("an attack must end with an allowlisted terminal")
            if self.replay:
                raise ValueError("a replay cannot be annotated as a live attack")
        elif self.label == "possession" and is_attack_terminal:
            raise ValueError("a complete allowlisted terminal must be labelled attack")
        elif self.label == "dead_ball" and is_attack_terminal:
            raise ValueError("dead_ball cannot carry an allowlisted attack terminal")

        if self.completeness == "complete" and self.terminal_type != "unknown":
            if self.terminal_frame is None or self.terminal_sec is None:
                raise ValueError("a complete known terminal requires frame and timestamp")
        if self.terminal_frame is not None and not self.start_frame <= self.terminal_frame <= self.end_frame:
            raise ValueError("terminal_frame must be inside the annotated sequence")
        if self.terminal_sec is not None and not self.start_sec <= self.terminal_sec <= self.end_sec:
            raise ValueError("terminal_sec must be inside the annotated sequence")

        if self.terminal_type == "headed_attempt" and self.shot_body_part != "head":
            raise ValueError("headed_attempt requires shot_body_part='head'")
        if self.terminal_type == "forced_goal_line_out":
            if self.terminal_cause not in FORCED_OUT_CAUSES:
                raise ValueError("forced_goal_line_out requires an approved defensive cause")
            if not self.beneficiary_team_id or self.beneficiary_team_id != self.stable_team_id:
                raise ValueError("forced_goal_line_out must award the restart to the attacking team")
            if not self.last_touch_team_id or self.last_touch_team_id == self.stable_team_id:
                raise ValueError("forced_goal_line_out requires an opponent last touch")
        if self.terminal_type in {"shot_attempt", "headed_attempt"}:
            if self.shot_outcome not in SHOT_OUTCOMES:
                raise ValueError(f"shot_outcome must be one of {SHOT_OUTCOMES}")

        stable = self.expected_stable_id()
        if self.annotation_id is not None and self.annotation_id != stable:
            raise ValueError("annotation_id does not match the stable identity fields")

    def to_dict(self) -> dict[str, object]:
        item = self.with_stable_id()
        item.validate()
        payload = dict(item.__dict__)
        payload["reason_codes"] = list(item.reason_codes)
        payload["schema_version"] = GOLD_SCHEMA_VERSION
        payload["split"] = gold_split(item.match_id)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "GoldSequenceAnnotationV2":
        if data.get("schema_version", GOLD_SCHEMA_VERSION) != GOLD_SCHEMA_VERSION:
            raise ValueError(f"unsupported gold schema: {data.get('schema_version')}")
        item = cls(
            match_id=str(data["match_id"]),
            period=int(data["period"]),
            stable_team_id=str(data["stable_team_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            label=str(data["label"]),
            terminal_type=str(data["terminal_type"]),
            terminal_frame=(None if data.get("terminal_frame") is None else int(data["terminal_frame"])),
            terminal_sec=(None if data.get("terminal_sec") is None else float(data["terminal_sec"])),
            terminal_cause=(None if data.get("terminal_cause") is None else str(data["terminal_cause"])),
            shot_body_part=(None if data.get("shot_body_part") is None else str(data["shot_body_part"])),
            shot_outcome=(None if data.get("shot_outcome") is None else str(data["shot_outcome"])),
            beneficiary_team_id=(
                None if data.get("beneficiary_team_id") is None else str(data["beneficiary_team_id"])
            ),
            last_touch_team_id=(
                None if data.get("last_touch_team_id") is None else str(data["last_touch_team_id"])
            ),
            replay=bool(data.get("replay", False)),
            completeness=str(data["completeness"]),
            confidence=float(data["confidence"]),
            reason_codes=tuple(str(value) for value in data.get("reason_codes", ())),
            notes=str(data.get("notes", "")),
            annotator_id=str(data["annotator_id"]),
            annotated_at=str(data["annotated_at"]),
            annotation_round=int(data.get("annotation_round", 1)),
            rubric_version=str(
                data.get("rubric_version", ATTACK_V2_RUBRIC_VERSION)
            ),
            annotation_id=(None if data.get("annotation_id") is None else str(data["annotation_id"])),
        ).with_stable_id()
        item.validate()
        return item


def _period_interval_payload(
    intervals: Iterable[GoldPeriodIntervalV2],
) -> dict[str, object]:
    values = tuple(intervals)
    for item in values:
        item.validate()
    ordered = sorted(values, key=lambda item: item.key)
    keys = [item.key for item in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("declared period intervals contain duplicate match/period keys")
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "rubric_version": ATTACK_V2_RUBRIC_VERSION,
        "frame_interval_convention": "inclusive",
        "period_intervals": [item.to_dict() for item in ordered],
    }


def period_interval_content_sha256(
    intervals: Iterable[GoldPeriodIntervalV2],
) -> str:
    payload = _period_interval_payload(intervals)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def read_gold_period_intervals(path: Path) -> tuple[GoldPeriodIntervalV2, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        if payload.get("schema_version", GOLD_SCHEMA_VERSION) != GOLD_SCHEMA_VERSION:
            raise ValueError("period interval schema_version does not match gold schema")
        if (
            payload.get("rubric_version", ATTACK_V2_RUBRIC_VERSION)
            != ATTACK_V2_RUBRIC_VERSION
        ):
            raise ValueError("period interval rubric_version does not match Attack-v2")
        if payload.get("frame_interval_convention", "inclusive") != "inclusive":
            raise ValueError("period intervals must use inclusive frame bounds")
        values = payload.get("period_intervals")
    else:
        values = None
    if not isinstance(values, list) or not values:
        raise ValueError("period interval file must contain a non-empty period_intervals array")
    intervals = tuple(GoldPeriodIntervalV2.from_dict(value) for value in values)
    _period_interval_payload(intervals)
    return intervals


def continuous_full_match_coverage_report(
    annotations: Iterable[GoldSequenceAnnotationV2],
    declared_intervals: Iterable[GoldPeriodIntervalV2],
    *,
    annotation_round: int | None = None,
) -> dict[str, object]:
    """Audit exact, non-overlapping coverage of every declared match period.

    This function only validates human-provided intervals. It never inserts or
    infers labels for missing frames. Gaps must be explicitly annotated as
    ``uncertain`` or ``dead_ball`` (or a genuine possession/attack) before the
    coverage gate can pass.
    """

    declarations = tuple(declared_intervals)
    declaration_payload = _period_interval_payload(declarations)
    declaration_map = {item.key: item for item in declarations}
    items = tuple(annotations)
    errors: list[str] = []

    rounds = sorted({item.annotation_round for item in items})
    if annotation_round is None:
        if len(rounds) > 1:
            errors.append(
                "multiple annotation rounds are present; select exactly one round for coverage"
            )
        selected_round = rounds[0] if len(rounds) == 1 else None
    else:
        if annotation_round < 1:
            raise ValueError("annotation_round must be positive")
        selected_round = annotation_round
    selected = tuple(
        item for item in items
        if selected_round is None or item.annotation_round == selected_round
    )

    annotation_map: dict[tuple[str, int], list[GoldSequenceAnnotationV2]] = {}
    seen_ids: set[str] = set()
    for item in selected:
        try:
            normalized = item.with_stable_id()
            normalized.validate()
        except ValueError as error:
            errors.append(f"invalid annotation: {error}")
            continue
        annotation_id = normalized.annotation_id or normalized.expected_stable_id()
        if annotation_id in seen_ids:
            errors.append(f"duplicate annotation_id: {annotation_id}")
        seen_ids.add(annotation_id)
        key = (canonical_match_id(normalized.match_id), normalized.period)
        if key not in declaration_map:
            errors.append(f"annotation belongs to undeclared interval {key[0]} period {key[1]}")
            continue
        annotation_map.setdefault(key, []).append(normalized)

    label_counts = {label: 0 for label in ANNOTATION_LABELS}
    complete_intervals = 0
    interval_reports: list[dict[str, object]] = []
    for key in sorted(declaration_map):
        declaration = declaration_map[key]
        values = sorted(
            annotation_map.get(key, ()),
            key=lambda item: (item.start_frame, item.end_frame, item.annotation_id or ""),
        )
        interval_errors: list[str] = []
        if not values:
            interval_errors.append("no annotations")
        else:
            if values[0].start_frame != declaration.start_frame:
                if values[0].start_frame > declaration.start_frame:
                    interval_errors.append(
                        f"leading gap frames {declaration.start_frame}-{values[0].start_frame - 1}"
                    )
                else:
                    interval_errors.append(
                        f"annotation starts before declared frame {declaration.start_frame}"
                    )
            previous: GoldSequenceAnnotationV2 | None = None
            tolerance = max(1e-3, 0.51 / declaration.fps)
            for item in values:
                label_counts[item.label] += 1
                if item.start_frame < declaration.start_frame or item.end_frame > declaration.end_frame:
                    interval_errors.append(
                        f"{item.annotation_id} lies outside declared frame bounds"
                    )
                expected_start_sec = declaration.expected_sec(item.start_frame)
                expected_end_sec = declaration.expected_sec(item.end_frame)
                if (
                    abs(item.start_sec - expected_start_sec) > tolerance
                    or abs(item.end_sec - expected_end_sec) > tolerance
                ):
                    interval_errors.append(
                        f"{item.annotation_id} time bounds disagree with frame/FPS declaration"
                    )
                if previous is not None:
                    expected_start = previous.end_frame + 1
                    if item.start_frame > expected_start:
                        interval_errors.append(
                            f"gap frames {expected_start}-{item.start_frame - 1}"
                        )
                    elif item.start_frame < expected_start:
                        interval_errors.append(
                            f"overlap frames {item.start_frame}-{previous.end_frame}"
                        )
                previous = item
            if values[-1].end_frame != declaration.end_frame:
                if values[-1].end_frame < declaration.end_frame:
                    interval_errors.append(
                        f"trailing gap frames {values[-1].end_frame + 1}-{declaration.end_frame}"
                    )
                else:
                    interval_errors.append(
                        f"annotation ends after declared frame {declaration.end_frame}"
                    )
        if not interval_errors:
            complete_intervals += 1
        errors.extend(
            f"{key[0]} period {key[1]}: {message}" for message in interval_errors
        )
        interval_reports.append(
            {
                "match_id": key[0],
                "period": key[1],
                "annotation_count": len(values),
                "complete": not interval_errors,
                "errors": interval_errors,
            }
        )

    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "rubric_version": ATTACK_V2_RUBRIC_VERSION,
        "frame_interval_convention": "inclusive",
        "annotation_round": selected_round,
        "annotation_count": len(selected),
        "declared_interval_count": len(declarations),
        "complete_interval_count": complete_intervals,
        "declaration_content_sha256": hashlib.sha256(
            _canonical_json(declaration_payload).encode("utf-8")
        ).hexdigest(),
        "label_counts": label_counts,
        "intervals": interval_reports,
        "complete": not errors and complete_intervals == len(declarations),
        "errors": errors,
    }


def validate_continuous_full_match_coverage(
    annotations: Iterable[GoldSequenceAnnotationV2],
    declared_intervals: Iterable[GoldPeriodIntervalV2],
    *,
    annotation_round: int | None = None,
) -> dict[str, object]:
    report = continuous_full_match_coverage_report(
        annotations,
        declared_intervals,
        annotation_round=annotation_round,
    )
    if not report["complete"]:
        messages = tuple(str(value) for value in report["errors"])
        preview = "; ".join(messages[:5])
        if len(messages) > 5:
            preview += f"; ... ({len(messages) - 5} more)"
        raise ValueError(f"continuous full-match coverage validation failed: {preview}")
    return report


@dataclass(frozen=True)
class ReannotationBlock:
    match_id: str
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, object]:
        return {
            "match_id": self.match_id,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
        }


def build_gold_manifest(
    video_root: Path,
    *,
    sealed_directory: Path | None = None,
) -> dict[str, object]:
    matches = HUMAN_TRAIN_MATCHES + DEVELOPMENT_MATCHES + FROZEN_TEST_MATCHES
    entries: list[dict[str, object]] = []
    for number in matches:
        path = video_root / f"game_{number}.mp4"
        exists = path.is_file()
        entries.append(
            {
                "match_id": f"game_{number}",
                "split": gold_split(number),
                "pilot": number == 6,
                "video_path": str(path),
                "video_exists": exists,
                "video_sha256": file_sha256(path) if exists else None,
            }
        )
    verified_seal = (
        None
        if sealed_directory is None
        else verify_sealed_test_annotations(sealed_directory)
    )
    seal_identity = (
        None
        if verified_seal is None
        else {
            key: verified_seal[key]
            for key in (
                "state",
                "file_sha256",
                "annotation_content_sha256",
                "period_intervals_file_sha256",
                "period_intervals_content_sha256",
                "seal_file_sha256",
            )
        }
    )
    identity = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "rubric_version": ATTACK_V2_RUBRIC_VERSION,
        "matches": [
            {
                key: entry[key]
                for key in ("match_id", "split", "video_exists", "video_sha256")
            }
            for entry in entries
        ],
        "test_label_seal": seal_identity,
    }
    return {
        **identity,
        "manifest_id": _stable_hash(identity, prefix="gold-manifest"),
        "continuous_timeline_annotation": True,
        "candidate_blinding_required": True,
        "test_labels_sealed": verified_seal is not None,
        "reannotation_fraction": 0.2,
        "reannotation_washout_days": 14,
        "matches": entries,
    }


def deterministic_reannotation_blocks(
    durations_sec: Mapping[str, float],
    *,
    fraction: float = 0.2,
    block_sec: float = 300.0,
    seed: int = 42,
) -> tuple[ReannotationBlock, ...]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if block_sec <= 0:
        raise ValueError("block_sec must be positive")
    rng = random.Random(seed)
    result: list[ReannotationBlock] = []
    for match_id in sorted(durations_sec):
        duration = float(durations_sec[match_id])
        if duration <= 0:
            raise ValueError("durations must be positive")
        target = duration * fraction
        possible = list(range(0, max(1, int(duration)), max(1, int(block_sec))))
        rng.shuffle(possible)
        covered = 0.0
        chosen: list[ReannotationBlock] = []
        for start in possible:
            end = min(duration, start + block_sec)
            chosen.append(ReannotationBlock(match_id, float(start), float(end)))
            covered += end - start
            if covered >= target:
                break
        result.extend(sorted(chosen, key=lambda value: value.start_sec))
    return tuple(result)


def read_gold_annotations_jsonl(
    path: Path,
    *,
    allow_frozen_test: bool = False,
) -> tuple[GoldSequenceAnnotationV2, ...]:
    result: list[GoldSequenceAnnotationV2] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = GoldSequenceAnnotationV2.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid gold annotation at line {line_number}: {error}") from error
            if gold_split(item.match_id) == "test" and not allow_frozen_test:
                raise PermissionError("frozen test labels may only be opened by the final evaluator")
            result.append(item)
    return tuple(result)


def append_gold_annotation_jsonl(path: Path, annotation: GoldSequenceAnnotationV2) -> None:
    item = annotation.with_stable_id()
    item.validate()
    if gold_split(item.match_id) == "test":
        raise PermissionError("write frozen test annotations through the sealed annotation workflow")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def append_sealed_test_annotation_jsonl(
    sealed_directory: Path,
    annotation: GoldSequenceAnnotationV2,
) -> Path:
    """Write final-test labels to an isolated file rejected by normal readers.

    This is a process seal, not encryption: training and tuning APIs refuse test
    labels unless the final evaluator explicitly opts in.
    """

    item = annotation.with_stable_id()
    item.validate()
    if gold_split(item.match_id) != "test":
        raise ValueError("the sealed workflow accepts only frozen test matches")
    sealed_directory.mkdir(parents=True, exist_ok=True)
    marker = sealed_directory / ".attack-v2-sealed"
    if not marker.exists():
        marker.write_text(
            json.dumps(
                {
                    "schema_version": GOLD_SCHEMA_VERSION,
                    "rubric_version": ATTACK_V2_RUBRIC_VERSION,
                    "purpose": "final_evaluation_only",
                    "matches": list(FROZEN_TEST_MATCH_IDS),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    path = sealed_directory / FROZEN_TEST_LABEL_FILE
    seal_path = sealed_directory / FROZEN_TEST_SEAL_FILE
    if seal_path.exists():
        raise PermissionError("frozen test labels have already been sealed and cannot be appended")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def freeze_sealed_test_annotations(
    sealed_directory: Path,
    declared_intervals: Iterable[GoldPeriodIntervalV2] | None = None,
) -> Path:
    """Hash/freeze the isolated final-test labels exactly once.

    The seal is tamper-evident, not encryption. Once created, the annotation
    append workflow refuses all further writes.
    """

    labels_path = sealed_directory / FROZEN_TEST_LABEL_FILE
    seal_path = sealed_directory / FROZEN_TEST_SEAL_FILE
    if seal_path.exists():
        raise PermissionError("frozen test labels have already been sealed")
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    annotations = read_gold_annotations_jsonl(labels_path, allow_frozen_test=True)
    if not annotations or any(gold_split(item.match_id) != "test" for item in annotations):
        raise ValueError("sealed label file must contain only frozen-test annotations")
    actual_matches = sorted({canonical_match_id(item.match_id) for item in annotations})
    if actual_matches != sorted(FROZEN_TEST_MATCH_IDS):
        raise ValueError(
            "sealed label file must contain exactly game_0, game_36 and game_50"
        )
    if {item.annotation_round for item in annotations} != {1}:
        raise ValueError("the frozen consensus file must contain annotation_round=1 only")

    interval_path = sealed_directory / FROZEN_TEST_INTERVAL_FILE
    if declared_intervals is None:
        if not interval_path.is_file():
            raise ValueError(
                "sealing requires declared match/period intervals for continuous coverage"
            )
        intervals = read_gold_period_intervals(interval_path)
    else:
        intervals = tuple(declared_intervals)
        interval_payload = _period_interval_payload(intervals)
        serialized = json.dumps(
            interval_payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        if interval_path.exists():
            existing = read_gold_period_intervals(interval_path)
            if period_interval_content_sha256(existing) != period_interval_content_sha256(
                intervals
            ):
                raise ValueError("persisted frozen-test period intervals do not match")
        else:
            interval_path.write_text(serialized, encoding="utf-8")

    if {item.key for item in intervals} != FROZEN_TEST_PERIOD_KEYS:
        raise ValueError(
            "frozen-test intervals must declare periods 1 and 2 for exactly game_0, game_36 and game_50"
        )
    coverage = validate_continuous_full_match_coverage(
        annotations,
        intervals,
        annotation_round=1,
    )
    marker_path = sealed_directory / ".attack-v2-sealed"
    if not marker_path.is_file():
        raise ValueError("isolated frozen-test marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "rubric_version": ATTACK_V2_RUBRIC_VERSION,
        "purpose": "final_evaluation_only",
        "matches": list(FROZEN_TEST_MATCH_IDS),
    }
    if marker != expected_marker:
        raise ValueError("isolated frozen-test marker does not match the v2 contract")

    seal_identity = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "rubric_version": ATTACK_V2_RUBRIC_VERSION,
        "state": "frozen",
        "label_file": labels_path.name,
        "period_intervals_file": interval_path.name,
        "annotation_count": len(annotations),
        "annotation_content_sha256": annotation_content_hash(annotations),
        "file_sha256": file_sha256(labels_path),
        "period_intervals_content_sha256": period_interval_content_sha256(intervals),
        "period_intervals_file_sha256": file_sha256(interval_path),
        "marker_file_sha256": file_sha256(marker_path),
        "matches": list(FROZEN_TEST_MATCH_IDS),
        "period_keys": [
            f"{match_id}:P{period}"
            for match_id, period in sorted(FROZEN_TEST_PERIOD_KEYS)
        ],
        "continuous_coverage": {
            "complete": True,
            "annotation_round": 1,
            "annotation_count": coverage["annotation_count"],
            "declared_interval_count": coverage["declared_interval_count"],
            "label_counts": coverage["label_counts"],
        },
    }
    seal = {
        **seal_identity,
        "seal_identity_sha256": hashlib.sha256(
            _canonical_json(seal_identity).encode("utf-8")
        ).hexdigest(),
    }
    try:
        with seal_path.open("x", encoding="utf-8") as handle:
            json.dump(seal, handle, ensure_ascii=False, sort_keys=True, indent=2)
    except FileExistsError as error:
        raise PermissionError("frozen test labels have already been sealed") from error
    return seal_path


def verify_sealed_test_annotations(sealed_directory: Path) -> dict[str, object]:
    labels_path = sealed_directory / FROZEN_TEST_LABEL_FILE
    seal_path = sealed_directory / FROZEN_TEST_SEAL_FILE
    interval_path = sealed_directory / FROZEN_TEST_INTERVAL_FILE
    marker_path = sealed_directory / ".attack-v2-sealed"
    if not seal_path.is_file():
        raise PermissionError("frozen test labels have not been hash-sealed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, dict):
        raise ValueError("frozen test label seal verification failed")
    annotations = read_gold_annotations_jsonl(labels_path, allow_frozen_test=True)
    intervals = read_gold_period_intervals(interval_path)
    coverage = validate_continuous_full_match_coverage(
        annotations,
        intervals,
        annotation_round=1,
    )
    actual_matches = sorted({canonical_match_id(item.match_id) for item in annotations})
    seal_identity = {
        key: value for key, value in seal.items() if key != "seal_identity_sha256"
    }
    expected_identity_hash = hashlib.sha256(
        _canonical_json(seal_identity).encode("utf-8")
    ).hexdigest()
    current_content_hash = annotation_content_hash(annotations)
    if (
        seal.get("schema_version") != GOLD_SCHEMA_VERSION
        or seal.get("rubric_version") != ATTACK_V2_RUBRIC_VERSION
        or seal.get("state") != "frozen"
        or seal.get("label_file") != FROZEN_TEST_LABEL_FILE
        or seal.get("period_intervals_file") != FROZEN_TEST_INTERVAL_FILE
        or seal.get("matches") != list(FROZEN_TEST_MATCH_IDS)
        or actual_matches != sorted(FROZEN_TEST_MATCH_IDS)
        or {item.key for item in intervals} != FROZEN_TEST_PERIOD_KEYS
        or seal.get("file_sha256") != file_sha256(labels_path)
        or seal.get("annotation_content_sha256") != current_content_hash
        or seal.get("period_intervals_file_sha256") != file_sha256(interval_path)
        or seal.get("period_intervals_content_sha256")
        != period_interval_content_sha256(intervals)
        or seal.get("marker_file_sha256") != file_sha256(marker_path)
        or seal.get("annotation_count") != len(annotations)
        or seal.get("seal_identity_sha256") != expected_identity_hash
        or not coverage["complete"]
    ):
        raise ValueError("frozen test label seal verification failed")
    return {**seal, "seal_file_sha256": file_sha256(seal_path)}


def annotation_content_hash(annotations: Iterable[GoldSequenceAnnotationV2]) -> str:
    payload = [item.with_stable_id().to_dict() for item in annotations]
    payload.sort(key=lambda item: str(item["annotation_id"]))
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _cohen_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("label vectors must be non-empty and have equal length")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    categories = sorted(set(left) | set(right))
    expected = sum(
        (left.count(value) / len(left)) * (right.count(value) / len(right))
        for value in categories
    )
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)


def intra_rater_report(
    first_round: Iterable[GoldSequenceAnnotationV2],
    second_round: Iterable[GoldSequenceAnnotationV2],
) -> dict[str, object]:
    from .sequence_evaluation import match_sequence_spans

    first_items = tuple(item.with_stable_id() for item in first_round)
    second_items = tuple(item.with_stable_id() for item in second_round)
    matching = match_sequence_spans(first_items, second_items, tiou_threshold=0.1)
    if not matching.matches:
        raise ValueError("annotation rounds have no temporally matching sequences")
    first = {item.annotation_id: item for item in first_items}
    second = {item.annotation_id: item for item in second_items}
    pairs = [
        (first[item.ground_truth_id], second[item.prediction_id])
        for item in matching.matches
    ]
    labels_first = [left.label for left, _ in pairs]
    labels_second = [right.label for _, right in pairs]
    terminal_matches = [left.terminal_type == right.terminal_type for left, right in pairs]
    start_errors = [abs(left.start_sec - right.start_sec) for left, right in pairs]
    end_errors = [abs(left.end_sec - right.end_sec) for left, right in pairs]
    all_errors = start_errors + end_errors
    ordered = sorted(all_errors)
    p90_index = max(0, int(0.9 * len(ordered) + 0.999999) - 1)
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "matched_sequences": len(pairs),
        "first_round_sequences": len(first_items),
        "second_round_sequences": len(second_items),
        "unmatched_first_round": len(matching.unmatched_ground_truth_ids),
        "unmatched_second_round": len(matching.unmatched_prediction_ids),
        "sequence_match_rate": round(
            2 * len(pairs) / (len(first_items) + len(second_items)), 4
        ),
        "attack_label_kappa": round(_cohen_kappa(labels_first, labels_second), 4),
        "terminal_agreement": round(sum(terminal_matches) / len(terminal_matches), 4),
        "median_boundary_disagreement_sec": round(median(all_errors), 4),
        "p90_boundary_disagreement_sec": round(ordered[p90_index], 4),
        "meets_annotation_gate": (
            _cohen_kappa(labels_first, labels_second) >= 0.8
            and sum(terminal_matches) / len(terminal_matches) >= 0.85
            and median(all_errors) <= 0.5
            and ordered[p90_index] <= 1.0
            and 2 * len(pairs) / (len(first_items) + len(second_items)) >= 0.9
        ),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
