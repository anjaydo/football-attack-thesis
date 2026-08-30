"""Reproducible infrastructure for the Attack-v2 action spotter.

This module deliberately contains *contracts and deterministic post-processing*,
not a claim that a model has reached any of the thesis acceptance gates.  The
training runner can consume these contracts without silently falling back to the
old 500-example cap, non-overlapping chunks, or a mismatched auxiliary model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


OFFICIAL_TOLERANCE_FRAMES = 12
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_WINDOW_OVERLAP = 0.5
MAX_SEARCH_CONFIGURATIONS = 12
FINAL_SEED_COUNT = 3
MAX_VRAM_GB = 5.5


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ActionTrainingRecord:
    """One labelled action instant used by the class-aware sampler."""

    record_id: str
    match_id: str
    period: int
    frame: int
    class_name: str
    target: int = 1
    source: str = "human_oracle"
    hard_negative_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id or not self.match_id or not self.class_name:
            raise ValueError("record_id, match_id and class_name are required")
        if self.period <= 0 or self.frame < 0:
            raise ValueError("period must be positive and frame must be non-negative")
        if self.target not in {0, 1}:
            raise ValueError("target must be 0 or 1")
        if self.hard_negative_reason not in {None, "duplicate", "conflict"}:
            raise ValueError("unsupported hard-negative reason")
        if self.hard_negative_reason is not None and self.target != 0:
            raise ValueError("a hard-negative record must have target=0")

    @property
    def stratum(self) -> tuple[str, int]:
        return self.class_name, self.target


@dataclass(frozen=True)
class WeightedActionTrainingRecord:
    record: ActionTrainingRecord
    weight: float
    class_weight: float
    match_balance: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.weight, self.class_weight, self.match_balance)
        ):
            raise ValueError("sampling weights must be finite and positive")


def compute_match_class_sampling_weights(
    records: Iterable[ActionTrainingRecord],
    *,
    weighting: str = "effective_number",
    beta: float = 0.9999,
) -> tuple[WeightedActionTrainingRecord, ...]:
    """Return a weight for every record; no per-class truncation is performed.

    Class imbalance is handled with inverse frequency or effective-number
    weighting.  Within each ``(class, target)`` stratum, every represented match
    receives equal total mass, so a long match cannot dominate a short match.
    The final weights are normalised to mean one.
    """

    sequence = tuple(records)
    if not sequence:
        raise ValueError("at least one training record is required")
    if len({item.record_id for item in sequence}) != len(sequence):
        raise ValueError("training record IDs must be unique")
    if weighting not in {"inverse_frequency", "effective_number"}:
        raise ValueError("weighting must be inverse_frequency or effective_number")
    if weighting == "effective_number" and not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")

    stratum_counts = Counter(item.stratum for item in sequence)
    match_stratum_counts = Counter(
        (item.stratum, item.match_id) for item in sequence
    )
    matches_per_stratum = Counter()
    for stratum, _match_id in match_stratum_counts:
        matches_per_stratum[stratum] += 1

    raw_class_weights: dict[tuple[str, int], float] = {}
    for stratum, count in stratum_counts.items():
        if weighting == "inverse_frequency" or beta == 0.0:
            raw_class_weights[stratum] = 1.0 / count
        else:
            effective_count = (1.0 - beta**count) / (1.0 - beta)
            raw_class_weights[stratum] = 1.0 / effective_count

    unnormalised: list[tuple[ActionTrainingRecord, float, float, float]] = []
    for item in sequence:
        class_weight = raw_class_weights[item.stratum]
        count_in_match = match_stratum_counts[(item.stratum, item.match_id)]
        represented_matches = matches_per_stratum[item.stratum]
        # This factor has a mean of one inside the stratum and assigns the
        # same aggregate mass to each match represented in that stratum.
        match_balance = stratum_counts[item.stratum] / (
            represented_matches * count_in_match
        )
        weight = class_weight * match_balance
        unnormalised.append((item, weight, class_weight, match_balance))

    mean_weight = sum(item[1] for item in unnormalised) / len(unnormalised)
    return tuple(
        WeightedActionTrainingRecord(
            record=item,
            weight=weight / mean_weight,
            class_weight=class_weight,
            match_balance=match_balance,
        )
        for item, weight, class_weight, match_balance in unnormalised
    )


class MatchClassAwareSampler:
    """Deterministic all-record epoch ordering plus framework-ready weights."""

    def __init__(
        self,
        records: Iterable[ActionTrainingRecord],
        *,
        weighting: str = "effective_number",
        beta: float = 0.9999,
        seed: int = 42,
    ) -> None:
        self.weighted_records = compute_match_class_sampling_weights(
            records,
            weighting=weighting,
            beta=beta,
        )
        self.seed = int(seed)

    def epoch(self, epoch_index: int = 0) -> tuple[ActionTrainingRecord, ...]:
        """Return every record once, ordered by a weighted random permutation."""

        if epoch_index < 0:
            raise ValueError("epoch_index must be non-negative")
        generator = random.Random(self.seed + epoch_index)
        ranked: list[tuple[float, str, ActionTrainingRecord]] = []
        for item in self.weighted_records:
            # Efraimidis-Spirakis exponential keys: lower values are sampled
            # earlier, but no example is discarded or capped.
            draw = max(generator.random(), 1e-15)
            key = -math.log(draw) / item.weight
            ranked.append((key, item.record.record_id, item.record))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked)

    def weights(self) -> tuple[float, ...]:
        return tuple(item.weight for item in self.weighted_records)


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    match_id: str
    period: int
    frame: int
    class_name: str
    score: float
    team_id: str | None = None
    actor_id: str | None = None
    conflict_group: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.match_id or not self.class_name:
            raise ValueError("candidate_id, match_id and class_name are required")
        if self.period <= 0 or self.frame < 0:
            raise ValueError("period must be positive and frame must be non-negative")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate score must be in [0, 1]")


@dataclass(frozen=True)
class DecodeConstraints:
    """Constraints applied once globally after overlapping-window merging."""

    nms_radius_frames: int = OFFICIAL_TOLERANCE_FRAMES
    per_class_nms_radius: Mapping[str, int] = field(default_factory=dict)
    conflict_radius_frames: int = 0
    mutually_exclusive_class_groups: tuple[frozenset[str], ...] = ()
    team_exclusive_classes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.nms_radius_frames < 0 or self.conflict_radius_frames < 0:
            raise ValueError("decode radii must be non-negative")
        if any(value < 0 for value in self.per_class_nms_radius.values()):
            raise ValueError("per-class NMS radii must be non-negative")
        if any(len(group) < 2 for group in self.mutually_exclusive_class_groups):
            raise ValueError("a mutually exclusive class group needs at least two classes")

    def nms_radius_for(self, class_name: str) -> int:
        return int(self.per_class_nms_radius.get(class_name, self.nms_radius_frames))


@dataclass(frozen=True)
class SuppressedActionCandidate:
    candidate: ActionCandidate
    reason: str
    winner_id: str | None

    def __post_init__(self) -> None:
        if self.reason not in {"below_threshold", "duplicate", "conflict"}:
            raise ValueError("unsupported suppression reason")
        if self.reason != "below_threshold" and not self.winner_id:
            raise ValueError("duplicate/conflict suppression requires a winner")


@dataclass(frozen=True)
class DecodeResult:
    selected: tuple[ActionCandidate, ...]
    suppressed: tuple[SuppressedActionCandidate, ...]

    @property
    def duplicates(self) -> tuple[SuppressedActionCandidate, ...]:
        return tuple(item for item in self.suppressed if item.reason == "duplicate")

    @property
    def conflicts(self) -> tuple[SuppressedActionCandidate, ...]:
        return tuple(item for item in self.suppressed if item.reason == "conflict")


def _is_mutually_exclusive(
    first: ActionCandidate,
    second: ActionCandidate,
    constraints: DecodeConstraints,
) -> bool:
    if (
        first.conflict_group
        and second.conflict_group
        and first.conflict_group == second.conflict_group
    ):
        return True
    if any(
        first.class_name in group and second.class_name in group
        for group in constraints.mutually_exclusive_class_groups
    ):
        return True
    if (
        first.team_id is not None
        and second.team_id is not None
        and first.team_id != second.team_id
        and first.class_name in constraints.team_exclusive_classes
        and second.class_name in constraints.team_exclusive_classes
    ):
        return True
    return False


def global_constrained_decode(
    candidates: Iterable[ActionCandidate],
    *,
    thresholds: Mapping[str, float] | None = None,
    constraints: DecodeConstraints | None = None,
) -> DecodeResult:
    """Apply deterministic global NMS and explicit simultaneous constraints.

    Input is expected to contain candidates from the complete period after
    overlapping-window logits have been merged.  Sorting by score with fully
    specified tie-breakers prevents chunk order from changing the output.
    """

    settings = constraints or DecodeConstraints()
    cutoffs = dict(thresholds or {})
    if any(not 0.0 <= float(value) <= 1.0 for value in cutoffs.values()):
        raise ValueError("all class thresholds must be in [0, 1]")
    sequence = tuple(candidates)
    if len({item.candidate_id for item in sequence}) != len(sequence):
        raise ValueError("candidate IDs must be unique")

    ranked = sorted(
        sequence,
        key=lambda item: (
            -item.score,
            item.match_id,
            item.period,
            item.frame,
            item.class_name,
            item.team_id or "",
            item.candidate_id,
        ),
    )
    selected: list[ActionCandidate] = []
    suppressed: list[SuppressedActionCandidate] = []
    for candidate in ranked:
        if candidate.score < float(cutoffs.get(candidate.class_name, 0.0)):
            suppressed.append(
                SuppressedActionCandidate(candidate, "below_threshold", None)
            )
            continue

        competitors = [
            item
            for item in selected
            if item.match_id == candidate.match_id and item.period == candidate.period
        ]
        duplicate = next(
            (
                item
                for item in competitors
                if item.class_name == candidate.class_name
                and abs(item.frame - candidate.frame)
                <= max(
                    settings.nms_radius_for(item.class_name),
                    settings.nms_radius_for(candidate.class_name),
                )
            ),
            None,
        )
        if duplicate is not None:
            suppressed.append(
                SuppressedActionCandidate(candidate, "duplicate", duplicate.candidate_id)
            )
            continue

        conflict = next(
            (
                item
                for item in competitors
                if abs(item.frame - candidate.frame) <= settings.conflict_radius_frames
                and _is_mutually_exclusive(item, candidate, settings)
            ),
            None,
        )
        if conflict is not None:
            suppressed.append(
                SuppressedActionCandidate(candidate, "conflict", conflict.candidate_id)
            )
            continue
        selected.append(candidate)

    selected.sort(
        key=lambda item: (
            item.match_id,
            item.period,
            item.frame,
            item.class_name,
            item.candidate_id,
        )
    )
    suppressed.sort(
        key=lambda item: (
            item.candidate.match_id,
            item.candidate.period,
            item.candidate.frame,
            item.candidate.class_name,
            item.candidate.candidate_id,
        )
    )
    return DecodeResult(tuple(selected), tuple(suppressed))


@dataclass(frozen=True)
class HardNegativeRecord:
    record_id: str
    match_id: str
    period: int
    frame: int
    class_name: str
    reason: str
    source_prediction_id: str
    winner_prediction_id: str
    anchor_frame: int

    def as_training_record(self) -> ActionTrainingRecord:
        return ActionTrainingRecord(
            record_id=self.record_id,
            match_id=self.match_id,
            period=self.period,
            frame=self.frame,
            class_name=self.class_name,
            target=0,
            source="decoded_hard_negative_v2",
            hard_negative_reason=self.reason,
        )


def mine_duplicate_conflict_hard_negatives(
    candidates: Iterable[ActionCandidate],
    *,
    constraints: DecodeConstraints | None = None,
    context_radius_frames: int = OFFICIAL_TOLERANCE_FRAMES,
    context_stride_frames: int = 6,
) -> tuple[HardNegativeRecord, ...]:
    """Mine temporal hard negatives around NMS duplicates and conflicts."""

    if context_radius_frames < 0 or context_stride_frames <= 0:
        raise ValueError("hard-negative context radius/stride is invalid")
    decoded = global_constrained_decode(candidates, constraints=constraints)
    records: dict[str, HardNegativeRecord] = {}
    for item in decoded.suppressed:
        if item.reason not in {"duplicate", "conflict"}:
            continue
        loser = item.candidate
        start = max(0, loser.frame - context_radius_frames)
        stop = loser.frame + context_radius_frames
        frames = list(range(start, stop + 1, context_stride_frames))
        if loser.frame not in frames:
            frames.append(loser.frame)
        for frame in sorted(set(frames)):
            payload = {
                "match_id": loser.match_id,
                "period": loser.period,
                "frame": frame,
                "class_name": loser.class_name,
                "reason": item.reason,
                "source_prediction_id": loser.candidate_id,
                "winner_prediction_id": item.winner_id,
            }
            record_id = f"hard-negative-v2:{_canonical_hash(payload)[:24]}"
            records[record_id] = HardNegativeRecord(
                record_id=record_id,
                match_id=loser.match_id,
                period=loser.period,
                frame=frame,
                class_name=loser.class_name,
                reason=item.reason,
                source_prediction_id=loser.candidate_id,
                winner_prediction_id=str(item.winner_id),
                anchor_frame=loser.frame,
            )
    return tuple(
        sorted(
            records.values(),
            key=lambda item: (
                item.match_id,
                item.period,
                item.frame,
                item.class_name,
                item.record_id,
            ),
        )
    )


@dataclass(frozen=True)
class TemporalWindow:
    match_id: str
    period: int
    start_frame: int
    end_frame_exclusive: int
    nominal_frame_count: int

    def __post_init__(self) -> None:
        if not self.match_id or self.period <= 0:
            raise ValueError("match_id and a positive period are required")
        if self.start_frame < 0 or self.end_frame_exclusive <= self.start_frame:
            raise ValueError("window frame bounds are invalid")
        if self.nominal_frame_count <= 0:
            raise ValueError("nominal_frame_count must be positive")
        if self.frame_count > self.nominal_frame_count:
            raise ValueError("actual window cannot exceed nominal window length")

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    @property
    def padding_frames(self) -> int:
        return self.nominal_frame_count - self.frame_count

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame_exclusive


def build_overlapping_windows(
    *,
    match_id: str,
    period: int,
    total_frames: int,
    fps: float,
    start_frame: int = 0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_WINDOW_OVERLAP,
) -> tuple[TemporalWindow, ...]:
    """Build 30-second, 50%-overlap windows with complete boundary coverage."""

    if total_frames <= 0 or fps <= 0.0 or start_frame < 0:
        raise ValueError("total_frames/fps must be positive and start_frame non-negative")
    if window_seconds <= 0.0 or not 0.0 <= overlap < 1.0:
        raise ValueError("window duration/overlap is invalid")
    nominal = max(1, int(round(window_seconds * fps)))
    stride = max(1, int(round(nominal * (1.0 - overlap))))
    last_relative_start = max(0, total_frames - nominal)
    relative_starts = list(range(0, last_relative_start + 1, stride))
    if relative_starts[-1] != last_relative_start:
        relative_starts.append(last_relative_start)
    relative_starts = sorted(set(relative_starts))
    absolute_end = start_frame + total_frames
    return tuple(
        TemporalWindow(
            match_id=match_id,
            period=period,
            start_frame=start_frame + relative_start,
            end_frame_exclusive=min(
                start_frame + relative_start + nominal,
                absolute_end,
            ),
            nominal_frame_count=nominal,
        )
        for relative_start in relative_starts
    )


@dataclass(frozen=True)
class WindowLogitChunk:
    window: TemporalWindow
    class_names: tuple[str, ...]
    logits: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be unique and non-empty")
        if len(self.logits) != self.window.frame_count:
            raise ValueError("one logit row is required for every actual window frame")
        if any(len(row) != len(self.class_names) for row in self.logits):
            raise ValueError("every logit row must match class_names")
        if any(not math.isfinite(value) for row in self.logits for value in row):
            raise ValueError("logits must be finite")


@dataclass(frozen=True)
class MergedFrameLogits:
    match_id: str
    period: int
    frame: int
    class_names: tuple[str, ...]
    logits: tuple[float, ...]
    contributing_windows: int
    total_weight: float


def _center_weight(window: TemporalWindow, frame: int, edge_weight: float) -> float:
    relative = frame - window.start_frame
    # Half-sample sine weights are strictly positive, symmetric, and maximal
    # at the centre.  The floor keeps very short/padded edges numerically safe.
    weight = math.sin(math.pi * (relative + 0.5) / window.frame_count)
    return max(edge_weight, weight)


def merge_center_weighted_logits(
    chunks: Iterable[WindowLogitChunk],
    *,
    edge_weight: float = 1e-3,
) -> tuple[MergedFrameLogits, ...]:
    """Merge overlapping logits by distance to each window centre."""

    if not 0.0 < edge_weight <= 1.0:
        raise ValueError("edge_weight must be in (0, 1]")
    sequence = tuple(chunks)
    if not sequence:
        return ()
    class_names_by_period: dict[tuple[str, int], tuple[str, ...]] = {}
    sums: dict[tuple[str, int, int], list[float]] = {}
    weights: defaultdict[tuple[str, int, int], float] = defaultdict(float)
    contributors: defaultdict[tuple[str, int, int], int] = defaultdict(int)
    for chunk in sequence:
        period_key = (chunk.window.match_id, chunk.window.period)
        expected_names = class_names_by_period.setdefault(period_key, chunk.class_names)
        if expected_names != chunk.class_names:
            raise ValueError("class name/order mismatch between overlapping chunks")
        for offset, row in enumerate(chunk.logits):
            frame = chunk.window.start_frame + offset
            key = (*period_key, frame)
            weight = _center_weight(chunk.window, frame, edge_weight)
            if key not in sums:
                sums[key] = [0.0] * len(row)
            for index, value in enumerate(row):
                sums[key][index] += value * weight
            weights[key] += weight
            contributors[key] += 1
    return tuple(
        MergedFrameLogits(
            match_id=match_id,
            period=period,
            frame=frame,
            class_names=class_names_by_period[(match_id, period)],
            logits=tuple(value / weights[(match_id, period, frame)] for value in values),
            contributing_windows=contributors[(match_id, period, frame)],
            total_weight=weights[(match_id, period, frame)],
        )
        for (match_id, period, frame), values in sorted(sums.items())
    )


@dataclass(frozen=True)
class ActionSpottingEvaluationProtocol:
    name: str = "footpass-official-v2"
    tolerance_frames: int = OFFICIAL_TOLERANCE_FRAMES
    tuned_split: str = "development"
    primary_selection_metric: str = "attack_f1_tiou_0.5"

    def __post_init__(self) -> None:
        if self.tolerance_frames != OFFICIAL_TOLERANCE_FRAMES:
            raise ValueError("the official action tolerance is frozen at ±12 frames")
        if self.tuned_split != "development":
            raise ValueError("operating-point tuning is development-only")


@dataclass(frozen=True)
class OperatingPointMeasurement:
    threshold: float
    attack_precision: float
    attack_recall: float
    label: str = ""

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (self.threshold, self.attack_precision, self.attack_recall)
        ):
            raise ValueError("operating-point values must be finite and in [0, 1]")

    @property
    def attack_f1(self) -> float:
        denominator = self.attack_precision + self.attack_recall
        return (
            2.0 * self.attack_precision * self.attack_recall / denominator
            if denominator
            else 0.0
        )


@dataclass(frozen=True)
class FrozenOperatingPoint:
    release_id: str
    development_dataset_hash: str
    selected: OperatingPointMeasurement
    evaluated_count: int
    protocol: ActionSpottingEvaluationProtocol
    selection_rule: str
    freeze_hash: str
    frozen: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "action-operating-point-v2.0",
            "release_id": self.release_id,
            "development_dataset_hash": self.development_dataset_hash,
            "selected": {
                **asdict(self.selected),
                "attack_f1": self.selected.attack_f1,
            },
            "evaluated_count": self.evaluated_count,
            "protocol": asdict(self.protocol),
            "selection_rule": self.selection_rule,
            "freeze_hash": self.freeze_hash,
            "frozen": self.frozen,
        }


def select_and_freeze_attack_operating_point(
    measurements: Iterable[OperatingPointMeasurement],
    *,
    split: str,
    release_id: str,
    development_dataset_hash: str,
    protocol: ActionSpottingEvaluationProtocol | None = None,
) -> FrozenOperatingPoint:
    """Select by attack F1, then balance P/R, then maximise the weaker side."""

    if split != "development":
        raise ValueError("threshold/NMS selection is permitted only on development")
    if not release_id or not development_dataset_hash:
        raise ValueError("release_id and development_dataset_hash are required")
    sequence = tuple(measurements)
    if not sequence:
        raise ValueError("at least one operating point is required")
    selected = max(
        sequence,
        key=lambda item: (
            item.attack_f1,
            -abs(item.attack_precision - item.attack_recall),
            min(item.attack_precision, item.attack_recall),
            item.threshold,
            item.label,
        ),
    )
    evaluation_protocol = protocol or ActionSpottingEvaluationProtocol()
    selection_rule = (
        "max attack-level F1; tie min |precision-recall|; "
        "tie max min(precision,recall); deterministic max threshold"
    )
    payload = {
        "release_id": release_id,
        "development_dataset_hash": development_dataset_hash,
        "selected": {
            **asdict(selected),
            "attack_f1": selected.attack_f1,
        },
        "evaluated_count": len(sequence),
        "protocol": asdict(evaluation_protocol),
        "selection_rule": selection_rule,
    }
    return FrozenOperatingPoint(
        release_id=release_id,
        development_dataset_hash=development_dataset_hash,
        selected=selected,
        evaluated_count=len(sequence),
        protocol=evaluation_protocol,
        selection_rule=selection_rule,
        freeze_hash=_canonical_hash(payload),
    )


def write_frozen_operating_point(
    record: FrozenOperatingPoint,
    path: Path,
) -> None:
    """Persist once; refuse to replace a different frozen operating point."""

    payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing.strip() != payload.strip():
            raise FileExistsError(f"refusing to replace frozen operating point: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SoccerNetAuxiliaryFeatureContract:
    """Frozen auxiliary-feature checkpoint and architecture compatibility."""

    enabled: bool = False
    frozen: bool = True
    expected_architecture: str | None = None
    expected_features: tuple[str, ...] = ()
    checkpoint_architecture: str | None = None
    checkpoint_features: tuple[str, ...] = ()
    checkpoint_checksum: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and not self.frozen:
            raise ValueError("SoccerNet auxiliary features must remain frozen")
        if self.enabled:
            if not self.expected_architecture or not self.checkpoint_architecture:
                raise ValueError("auxiliary architecture signatures are required")
            if not self.expected_features or not self.checkpoint_features:
                raise ValueError("auxiliary feature signatures are required")
            if not self.checkpoint_checksum:
                raise ValueError("a frozen auxiliary checkpoint checksum is required")
            if self.expected_architecture != self.checkpoint_architecture:
                raise ValueError("SoccerNet auxiliary checkpoint architecture mismatch")
            if self.expected_features != self.checkpoint_features:
                raise ValueError("SoccerNet auxiliary checkpoint feature mismatch")


@dataclass(frozen=True)
class SoccerNetAuxiliaryAblation:
    enabled: SoccerNetAuxiliaryFeatureContract
    disabled: SoccerNetAuxiliaryFeatureContract

    def __post_init__(self) -> None:
        if not self.enabled.enabled or self.disabled.enabled:
            raise ValueError("ablation requires one enabled and one disabled contract")
        if (
            self.enabled.expected_architecture
            != self.disabled.expected_architecture
            or self.enabled.expected_features != self.disabled.expected_features
        ):
            raise ValueError("auxiliary ablation must hold architecture/features fixed")


@dataclass(frozen=True)
class ActionSpottingResourceContract:
    cached_embeddings: bool = True
    amp: bool = True
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_vram_gb: float = MAX_VRAM_GB
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    window_overlap: float = DEFAULT_WINDOW_OVERLAP

    def __post_init__(self) -> None:
        if not self.cached_embeddings:
            raise ValueError("Attack-v2 search requires cached embeddings")
        if not self.amp:
            raise ValueError("Attack-v2 search requires AMP on the RTX 4050")
        if self.batch_size != 1:
            raise ValueError("Attack-v2 GPU contract requires batch_size=1")
        if self.gradient_accumulation_steps != 8:
            raise ValueError("Attack-v2 GPU contract requires gradient accumulation=8")
        if self.max_vram_gb <= 0.0 or self.max_vram_gb > MAX_VRAM_GB:
            raise ValueError("peak VRAM gate cannot exceed 5.5 GB")
        if self.window_seconds != DEFAULT_WINDOW_SECONDS:
            raise ValueError("Attack-v2 spotting windows are fixed at 30 seconds")
        if self.window_overlap != DEFAULT_WINDOW_OVERLAP:
            raise ValueError("Attack-v2 spotting overlap is fixed at 50%")


@dataclass(frozen=True)
class ActionSpottingSearchTrial:
    configuration_id: str
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    auxiliary: SoccerNetAuxiliaryFeatureContract = field(
        default_factory=SoccerNetAuxiliaryFeatureContract
    )
    ablation_group: str | None = None

    def __post_init__(self) -> None:
        if not self.configuration_id:
            raise ValueError("configuration_id is required")
        if any(not str(key) for key in self.hyperparameters):
            raise ValueError("hyperparameter names must be non-empty")
        if self.auxiliary.enabled and not self.ablation_group:
            raise ValueError("enabled auxiliary feature requires an ablation_group")

    def canonical_hyperparameters(self) -> str:
        return json.dumps(
            dict(self.hyperparameters),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


@dataclass(frozen=True)
class ActionSpottingSearchPlan:
    trials: tuple[ActionSpottingSearchTrial, ...]
    final_seeds: tuple[int, ...] = (42, 43, 44)
    resources: ActionSpottingResourceContract = field(
        default_factory=ActionSpottingResourceContract
    )
    sequential: bool = True
    max_configurations: int = MAX_SEARCH_CONFIGURATIONS

    def __post_init__(self) -> None:
        if not self.trials:
            raise ValueError("the search plan requires at least one trial")
        if self.max_configurations != MAX_SEARCH_CONFIGURATIONS:
            raise ValueError("the search ceiling is frozen at 12 configurations")
        if len(self.trials) > self.max_configurations:
            raise ValueError("at most 12 spotting configurations may be searched")
        identifiers = [item.configuration_id for item in self.trials]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("search configuration IDs must be unique")
        if not self.sequential:
            raise ValueError("RTX 4050 action search must execute sequentially")
        if len(self.final_seeds) != FINAL_SEED_COUNT or len(set(self.final_seeds)) != 3:
            raise ValueError("the frozen final candidate must run exactly three seeds")

        for enabled_trial in (item for item in self.trials if item.auxiliary.enabled):
            ablations = [
                item
                for item in self.trials
                if item.ablation_group == enabled_trial.ablation_group
                and not item.auxiliary.enabled
                and item.canonical_hyperparameters()
                == enabled_trial.canonical_hyperparameters()
            ]
            if not ablations:
                raise ValueError(
                    "every enabled SoccerNet auxiliary trial needs a matched frozen ablation"
                )
            SoccerNetAuxiliaryAblation(enabled_trial.auxiliary, ablations[0].auxiliary)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "action-spotting-search-v2.0",
            "execution": "sequential",
            "configuration_count": len(self.trials),
            "max_configurations": self.max_configurations,
            "final_seeds": list(self.final_seeds),
            "resources": asdict(self.resources),
            "official_tolerance_frames": OFFICIAL_TOLERANCE_FRAMES,
            "trials": [
                {
                    "configuration_id": item.configuration_id,
                    "hyperparameters": dict(item.hyperparameters),
                    "auxiliary": asdict(item.auxiliary),
                    "ablation_group": item.ablation_group,
                }
                for item in self.trials
            ],
        }


__all__ = [
    "OFFICIAL_TOLERANCE_FRAMES",
    "DEFAULT_WINDOW_SECONDS",
    "DEFAULT_WINDOW_OVERLAP",
    "MAX_SEARCH_CONFIGURATIONS",
    "FINAL_SEED_COUNT",
    "MAX_VRAM_GB",
    "ActionTrainingRecord",
    "WeightedActionTrainingRecord",
    "compute_match_class_sampling_weights",
    "MatchClassAwareSampler",
    "ActionCandidate",
    "DecodeConstraints",
    "SuppressedActionCandidate",
    "DecodeResult",
    "global_constrained_decode",
    "HardNegativeRecord",
    "mine_duplicate_conflict_hard_negatives",
    "TemporalWindow",
    "build_overlapping_windows",
    "WindowLogitChunk",
    "MergedFrameLogits",
    "merge_center_weighted_logits",
    "ActionSpottingEvaluationProtocol",
    "OperatingPointMeasurement",
    "FrozenOperatingPoint",
    "select_and_freeze_attack_operating_point",
    "write_frozen_operating_point",
    "SoccerNetAuxiliaryFeatureContract",
    "SoccerNetAuxiliaryAblation",
    "ActionSpottingResourceContract",
    "ActionSpottingSearchTrial",
    "ActionSpottingSearchPlan",
]

