"""Untrained temporal-model infrastructure for Attack-v2.

This module defines the reproducible 5 Hz feature contract, a memory-conscious
dilated-TCN specification, the multi-task loss contract, and a deterministic
semi-Markov decoder.  It deliberately contains no fitted weights and must not
be presented as a validated model.  Human-development attack F1 is the only
primary checkpoint-selection metric exposed here.

The NumPy decoder is usable without PyTorch.  PyTorch is imported lazily only
when :func:`build_torch_temporal_model` is called.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import ceil, isfinite
from typing import Mapping, Sequence

import numpy as np


TEMPORAL_FEATURE_SCHEMA_VERSION = "attack-temporal-features-v2.0"
TEMPORAL_MODEL_CONFIG_VERSION = "attack-temporal-model-v2.0"
MODEL_VALIDATION_STATUS = "infrastructure_untrained"
FIXED_SAMPLE_RATE_HZ = 5.0


class TemporalState(str, Enum):
    """The exact public state space used by the learned possession decoder."""

    DEAD_BALL = "dead_ball"
    CONTROL_TEAM_A = "control_team_A"
    CONTROL_TEAM_B = "control_team_B"
    CONTESTED = "contested"
    BALL_FLIGHT = "ball_flight"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


TEMPORAL_STATES: tuple[TemporalState, ...] = tuple(TemporalState)
_STATE_INDEX = {state: index for index, state in enumerate(TEMPORAL_STATES)}
TEAM_A = "A"
TEAM_B = "B"
_TEAMS = frozenset({TEAM_A, TEAM_B})


@dataclass(frozen=True)
class FeatureGroup:
    """One fixed-width group in the cached temporal feature vector."""

    name: str
    width: int
    semantics: str
    has_missingness_mask: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("feature-group name must not be empty")
        if self.width <= 0:
            raise ValueError("feature-group width must be positive")
        if not self.semantics.strip():
            raise ValueError("feature-group semantics must not be empty")


FIXED_FEATURE_GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup("visual_embedding", 128, "cached frozen/fine-tuned frame embedding"),
    FeatureGroup("ball_track", 12, "position, velocity, visibility and occlusion support"),
    FeatureGroup("team_A_tracks", 16, "stable-team-A aggregate tracking features"),
    FeatureGroup("team_B_tracks", 16, "stable-team-B aggregate tracking features"),
    FeatureGroup("pitch_calibration", 8, "field projection, validity and uncertainty"),
    FeatureGroup("action_logits", 8, "drive/pass/cross/throw-in/shot/header/tackle/block logits"),
    FeatureGroup("event_context", 12, "contact, restart, boundary and temporal context"),
    FeatureGroup("source_quality", 8, "visibility, provenance and association quality"),
)


@dataclass(frozen=True)
class TemporalFeatureCacheConfig:
    """Immutable schema for frame features cached at exactly 5 Hz."""

    schema_version: str = field(default=TEMPORAL_FEATURE_SCHEMA_VERSION, init=False)
    sample_rate_hz: float = field(default=FIXED_SAMPLE_RATE_HZ, init=False)
    dtype: str = field(default="float32", init=False)
    timebase: str = field(default="period_seconds", init=False)
    chunk_seconds: float = 30.0
    compression: str = "zstd"
    feature_groups: tuple[FeatureGroup, ...] = field(
        default=FIXED_FEATURE_GROUPS,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.chunk_seconds != 30.0:
            raise ValueError("feature-cache chunks are fixed at 30 seconds")
        if self.compression not in {"zstd", "none"}:
            raise ValueError("compression must be 'zstd' or 'none'")

    @property
    def sample_period_sec(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def feature_width(self) -> int:
        return sum(group.width for group in self.feature_groups)

    @property
    def missingness_width(self) -> int:
        return sum(group.has_missingness_mask for group in self.feature_groups)

    def schema_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "validation_status": MODEL_VALIDATION_STATUS,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_period_sec": self.sample_period_sec,
            "timebase": self.timebase,
            "dtype": self.dtype,
            "chunk_seconds": self.chunk_seconds,
            "compression": self.compression,
            "feature_width": self.feature_width,
            "missingness_width": self.missingness_width,
            "required_index_columns": [
                "match_id",
                "period",
                "sample_index",
                "timestamp_sec",
                "source_frame",
            ],
            "feature_groups": [asdict(group) for group in self.feature_groups],
        }


@dataclass(frozen=True)
class TemporalFeatureBatch:
    """Validated in-memory representation of one cached period."""

    match_id: str
    period: int
    timestamps_sec: np.ndarray
    source_frames: np.ndarray
    features: np.ndarray
    missingness: np.ndarray
    config: TemporalFeatureCacheConfig = field(default_factory=TemporalFeatureCacheConfig)

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_sec, dtype=np.float64)
        source_frames = np.asarray(self.source_frames, dtype=np.int64)
        features = np.asarray(self.features, dtype=np.float32)
        missingness = np.asarray(self.missingness, dtype=np.bool_)
        if not self.match_id.strip():
            raise ValueError("match_id must not be empty")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("timestamps_sec must be a non-empty vector")
        if source_frames.shape != timestamps.shape:
            raise ValueError("source_frames must align with timestamps_sec")
        if features.shape != (timestamps.size, self.config.feature_width):
            raise ValueError("features do not match the fixed feature-cache width")
        expected_missingness = (timestamps.size, self.config.missingness_width)
        if missingness.shape != expected_missingness:
            raise ValueError("missingness does not match the fixed group-mask width")
        if np.any(~np.isfinite(timestamps)) or np.any(timestamps < 0):
            raise ValueError("timestamps_sec must be finite and non-negative")
        if np.any(source_frames < 0):
            raise ValueError("source_frames must be non-negative")
        if np.any(~np.isfinite(features)):
            raise ValueError("features must be finite; use missingness masks for unknown data")
        if timestamps.size > 1:
            expected_step = self.config.sample_period_sec
            if not np.allclose(np.diff(timestamps), expected_step, atol=1e-6, rtol=0.0):
                raise ValueError("timestamps_sec must lie on the fixed 5 Hz grid")
            if np.any(np.diff(source_frames) < 0):
                raise ValueError("source_frames must be monotonic")
        object.__setattr__(self, "timestamps_sec", timestamps)
        object.__setattr__(self, "source_frames", source_frames)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "missingness", missingness)


@dataclass(frozen=True)
class TemporalHeadSpec:
    name: str
    output_width: int
    target: str
    loss: str
    weight: float


MULTITASK_HEADS: tuple[TemporalHeadSpec, ...] = (
    TemporalHeadSpec(
        "possession_state",
        len(TEMPORAL_STATES),
        "one of the six exact temporal states",
        "semi_markov_structured_negative_log_likelihood",
        1.0,
    ),
    TemporalHeadSpec("start_boundary", 1, "possession start", "class_balanced_focal", 0.5),
    TemporalHeadSpec("end_boundary", 1, "possession end", "class_balanced_focal", 0.5),
    TemporalHeadSpec("shot", 1, "shot attempt", "class_balanced_focal", 1.0),
    TemporalHeadSpec(
        "headed_at_goal",
        1,
        "intentional headed attempt at goal",
        "class_balanced_focal",
        1.0,
    ),
    TemporalHeadSpec(
        "forced_goal_line_out",
        1,
        "defensive save/block/deflection leading to attacking corner",
        "class_balanced_focal",
        1.0,
    ),
    TemporalHeadSpec(
        "forced_out_cause",
        4,
        "save/block/deflection/unknown",
        "class_balanced_focal",
        0.5,
    ),
    TemporalHeadSpec("replay_live", 2, "replay versus live play", "cross_entropy", 0.5),
    TemporalHeadSpec(
        "uncertainty",
        1,
        "probability that evidence is insufficient",
        "binary_cross_entropy",
        0.25,
    ),
)


@dataclass(frozen=True)
class MultiTaskLossSpec:
    """Frozen training-loss semantics; this is a specification, not fitted loss."""

    heads: tuple[TemporalHeadSpec, ...] = MULTITASK_HEADS
    class_balance_beta: float = 0.999
    focal_gamma: float = 2.0
    structured_nll_weight: float = 1.0
    mask_replay_from_live_possession_loss: bool = True
    ignore_unknown_targets: bool = True
    validation_status: str = field(default=MODEL_VALIDATION_STATUS, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.class_balance_beta < 1.0:
            raise ValueError("class_balance_beta must be in [0, 1)")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if self.structured_nll_weight <= 0:
            raise ValueError("structured_nll_weight must be positive")
        if {head.name for head in self.heads} != {
            "possession_state",
            "start_boundary",
            "end_boundary",
            "shot",
            "headed_at_goal",
            "forced_goal_line_out",
            "forced_out_cause",
            "replay_live",
            "uncertainty",
        }:
            raise ValueError("multi-task heads must match the frozen Attack-v2 contract")


def class_balanced_focal_loss_numpy(
    logits: np.ndarray,
    targets: Sequence[int] | np.ndarray,
    class_counts: Sequence[int] | np.ndarray,
    *,
    beta: float = 0.999,
    gamma: float = 2.0,
) -> float:
    """Return deterministic mean class-balanced focal loss for multiclass logits."""

    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    counts = np.asarray(class_counts, dtype=np.float64)
    if scores.ndim != 2 or labels.shape != (scores.shape[0],):
        raise ValueError("logits must be N x C and targets must have length N")
    if counts.shape != (scores.shape[1],) or np.any(counts <= 0):
        raise ValueError("class_counts must contain one positive count per class")
    if np.any(labels < 0) or np.any(labels >= scores.shape[1]):
        raise ValueError("target class index is out of range")
    if not 0.0 <= beta < 1.0 or gamma < 0:
        raise ValueError("invalid focal-loss beta or gamma")
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    selected = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
    if beta == 0.0:
        weights = np.ones_like(counts)
    else:
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights *= counts.size / weights.sum()
    losses = -weights[labels] * np.power(1.0 - selected, gamma) * np.log(selected)
    return float(np.mean(losses))


@dataclass(frozen=True)
class DilatedTCNConfig:
    """30-second effective-context TCN with RTX-4050-friendly training defaults."""

    config_version: str = field(default=TEMPORAL_MODEL_CONFIG_VERSION, init=False)
    validation_status: str = field(default=MODEL_VALIDATION_STATUS, init=False)
    sample_rate_hz: float = field(default=FIXED_SAMPLE_RATE_HZ, init=False)
    context_seconds: float = field(default=30.0, init=False)
    input_width: int = field(
        default_factory=lambda: TemporalFeatureCacheConfig().feature_width,
        init=False,
    )
    hidden_channels: int = 96
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    convolutions_per_level: int = 2
    dropout: float = 0.2
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_amp: bool = True
    cache_embeddings: bool = True
    peak_vram_budget_gb: float = 5.5
    maximum_sequential_configs: int = 12
    final_seed_count: int = 3

    def __post_init__(self) -> None:
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least three")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must be positive")
        if self.convolutions_per_level <= 0:
            raise ValueError("convolutions_per_level must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size != 1 or self.gradient_accumulation_steps != 8:
            raise ValueError("local training is fixed to batch size 1 and accumulation 8")
        if not self.use_amp or not self.cache_embeddings:
            raise ValueError("local training requires AMP and cached embeddings")
        if self.peak_vram_budget_gb > 5.5:
            raise ValueError("peak VRAM budget must not exceed 5.5 GB")
        if self.maximum_sequential_configs > 12 or self.final_seed_count != 3:
            raise ValueError("search is capped at 12 configurations and three final seeds")
        if self.theoretical_receptive_field_seconds < self.context_seconds:
            raise ValueError("TCN theoretical receptive field must cover the 30-second window")

    @property
    def context_samples(self) -> int:
        return round(self.context_seconds * self.sample_rate_hz)

    @property
    def theoretical_receptive_field_samples(self) -> int:
        return 1 + (
            self.convolutions_per_level
            * (self.kernel_size - 1)
            * sum(self.dilations)
        )

    @property
    def theoretical_receptive_field_seconds(self) -> float:
        return self.theoretical_receptive_field_samples / self.sample_rate_hz

    @property
    def effective_receptive_field_seconds(self) -> float:
        return min(self.context_seconds, self.theoretical_receptive_field_seconds)

    def training_metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "context_samples": self.context_samples,
            "theoretical_receptive_field_samples": self.theoretical_receptive_field_samples,
            "theoretical_receptive_field_seconds": self.theoretical_receptive_field_seconds,
            "effective_receptive_field_seconds": self.effective_receptive_field_seconds,
            "checkpoint_metric": "human_development_attack_f1_tiou_0.5",
            "checkpoint_metric_forbidden_substitutes": [
                "micro_action_f1",
                "validation_loss",
            ],
        }


def build_torch_temporal_model(config: DilatedTCNConfig | None = None):
    """Build the optional TCN and heads, importing PyTorch only on demand.

    Returned weights are randomly initialized.  Calling this function does not
    train, calibrate, or validate the model.
    """

    settings = config or DilatedTCNConfig()
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Install PyTorch separately to build the optional temporal model") from error

    class _ResidualDilatedBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (settings.kernel_size - 1) // 2
            layers: list[nn.Module] = []
            for _ in range(settings.convolutions_per_level):
                layers.extend(
                    (
                        nn.Conv1d(
                            channels,
                            channels,
                            settings.kernel_size,
                            padding=padding,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                        nn.Dropout(settings.dropout),
                    )
                )
            self.network = nn.Sequential(*layers)
            self.normalization = nn.GroupNorm(1, channels)

        def forward(self, values):
            return self.normalization(values + self.network(values))

    class _AttackTemporalTCN(nn.Module):
        validation_status = MODEL_VALIDATION_STATUS

        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(settings.input_width, settings.hidden_channels, 1)
            self.encoder = nn.Sequential(
                *(
                    _ResidualDilatedBlock(settings.hidden_channels, dilation)
                    for dilation in settings.dilations
                )
            )
            self.heads = nn.ModuleDict(
                {
                    head.name: nn.Conv1d(settings.hidden_channels, head.output_width, 1)
                    for head in MULTITASK_HEADS
                }
            )

        def forward(self, values):
            if values.ndim != 3 or values.shape[-1] != settings.input_width:
                raise ValueError("expected input shaped batch x time x fixed feature width")
            encoded = self.encoder(self.input_projection(values.transpose(1, 2)))
            return {
                name: head(encoded).transpose(1, 2)
                for name, head in self.heads.items()
            }

    model = _AttackTemporalTCN()
    model.attack_v2_config = settings.training_metadata()
    model.attack_v2_loss_spec = asdict(MultiTaskLossSpec())
    model.torch_version = torch.__version__
    return model


@dataclass(frozen=True)
class StateDurationConstraint:
    minimum_sec: float
    maximum_sec: float

    def __post_init__(self) -> None:
        if self.minimum_sec <= 0 or self.maximum_sec < self.minimum_sec:
            raise ValueError("duration constraint must have 0 < minimum <= maximum")

    def sample_bounds(self, sample_rate_hz: float = FIXED_SAMPLE_RATE_HZ) -> tuple[int, int]:
        return (
            max(1, ceil(self.minimum_sec * sample_rate_hz - 1e-12)),
            max(1, ceil(self.maximum_sec * sample_rate_hz - 1e-12)),
        )


DEFAULT_DURATION_CONSTRAINTS: tuple[tuple[TemporalState, StateDurationConstraint], ...] = (
    (TemporalState.DEAD_BALL, StateDurationConstraint(0.2, 30.0)),
    (TemporalState.CONTROL_TEAM_A, StateDurationConstraint(0.4, 30.0)),
    (TemporalState.CONTROL_TEAM_B, StateDurationConstraint(0.4, 30.0)),
    (TemporalState.CONTESTED, StateDurationConstraint(0.2, 5.0)),
    (TemporalState.BALL_FLIGHT, StateDurationConstraint(0.2, 10.0)),
    (TemporalState.UNKNOWN, StateDurationConstraint(0.2, 30.0)),
)


@dataclass(frozen=True)
class SemiMarkovDecoderConfig:
    """Hard constraints and scores for deterministic period-level decoding."""

    config_version: str = field(default=TEMPORAL_MODEL_CONFIG_VERSION, init=False)
    validation_status: str = field(default=MODEL_VALIDATION_STATUS, init=False)
    sample_rate_hz: float = field(default=FIXED_SAMPLE_RATE_HZ, init=False)
    duration_constraints: tuple[
        tuple[TemporalState, StateDurationConstraint], ...
    ] = DEFAULT_DURATION_CONSTRAINTS
    state_change_penalty: float = -0.25
    owner_change_penalty: float = -0.50
    latent_same_state_chunk_penalty: float = -1e-6

    def __post_init__(self) -> None:
        states = tuple(state for state, _ in self.duration_constraints)
        if states != TEMPORAL_STATES:
            raise ValueError("duration constraints must cover the six exact states in order")
        for value in (
            self.state_change_penalty,
            self.owner_change_penalty,
            self.latent_same_state_chunk_penalty,
        ):
            if not isfinite(value) or value > 0:
                raise ValueError("transition penalties must be finite and non-positive")

    def sample_duration(self, state: TemporalState) -> tuple[int, int]:
        return dict(self.duration_constraints)[state].sample_bounds(self.sample_rate_hz)


_TERMINAL_TYPES = frozenset(
    {
        "shot_attempt",
        "headed_attempt",
        "forced_goal_line_out",
        "turnover",
        "unforced_out",
        "foul",
        "offside",
        "period_end",
        "unknown",
    }
)


def _team(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized not in _TEAMS:
        raise ValueError(f"{field_name} values must be 'A', 'B', or None")
    return normalized


@dataclass(frozen=True)
class TemporalDecodeEvidence:
    """Hard evidence aligned to 5 Hz samples; empty vectors mean all unknown."""

    established_control: tuple[str | None, ...] = ()
    contested_touch: tuple[bool, ...] = ()
    observation_gap: tuple[bool, ...] = ()
    terminal_types: tuple[str | None, ...] = ()
    terminal_teams: tuple[str | None, ...] = ()
    reset_teams: tuple[str | None, ...] = ()
    left_censored: bool = False
    right_censored: bool = False

    def normalized(self, sample_count: int) -> "TemporalDecodeEvidence":
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

        def optional_team_vector(values: tuple[str | None, ...], name: str) -> tuple[str | None, ...]:
            if not values:
                return (None,) * sample_count
            if len(values) != sample_count:
                raise ValueError(f"{name} must align with emission samples")
            return tuple(_team(value, name) for value in values)

        def bool_vector(values: tuple[bool, ...], name: str) -> tuple[bool, ...]:
            if not values:
                return (False,) * sample_count
            if len(values) != sample_count:
                raise ValueError(f"{name} must align with emission samples")
            return tuple(bool(value) for value in values)

        if not self.terminal_types:
            terminal_types: tuple[str | None, ...] = (None,) * sample_count
        else:
            if len(self.terminal_types) != sample_count:
                raise ValueError("terminal_types must align with emission samples")
            terminal_types = tuple(
                None if value is None else str(value).strip().lower()
                for value in self.terminal_types
            )
            invalid = {value for value in terminal_types if value not in _TERMINAL_TYPES | {None}}
            if invalid:
                raise ValueError(f"unsupported terminal types: {sorted(invalid)}")

        normalized = TemporalDecodeEvidence(
            established_control=optional_team_vector(
                self.established_control,
                "established_control",
            ),
            contested_touch=bool_vector(self.contested_touch, "contested_touch"),
            observation_gap=bool_vector(self.observation_gap, "observation_gap"),
            terminal_types=terminal_types,
            terminal_teams=optional_team_vector(self.terminal_teams, "terminal_teams"),
            reset_teams=optional_team_vector(self.reset_teams, "reset_teams"),
            left_censored=self.left_censored,
            right_censored=self.right_censored,
        )
        for index, terminal_type in enumerate(normalized.terminal_types):
            if terminal_type is not None and normalized.observation_gap[index]:
                raise ValueError("a terminal cannot be asserted inside an observation gap")
            terminal_team = normalized.terminal_teams[index]
            if terminal_team is not None and terminal_type is None:
                raise ValueError("terminal_team requires a terminal_type at the same sample")
            reset_team = normalized.reset_teams[index]
            if reset_team is not None and index > 0:
                previous_is_boundary = (
                    normalized.terminal_types[index - 1] is not None
                    or normalized.observation_gap[index - 1]
                )
                if not previous_is_boundary:
                    raise ValueError("reset_team is only valid after a terminal/gap or at period start")
        return normalized


@dataclass(frozen=True)
class DecodedTemporalSegment:
    state: TemporalState
    owner_context: str | None
    start_index: int
    end_index: int
    sequence_index: int | None

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index < self.start_index:
            raise ValueError("invalid decoded segment bounds")


@dataclass(frozen=True)
class DecodedSequenceRange:
    sequence_index: int
    start_index: int
    end_index: int
    terminal_index: int | None
    terminal_type: str | None
    left_censored: bool
    right_censored: bool
    status: str


@dataclass(frozen=True)
class GapCensoringMarker:
    start_index: int
    end_index: int
    left_sequence_right_censored: bool
    right_sequence_left_censored: bool
    reason: str = "observation_gap"


@dataclass(frozen=True)
class TemporalDecodeResult:
    states: tuple[TemporalState, ...]
    owner_context: tuple[str | None, ...]
    segments: tuple[DecodedTemporalSegment, ...]
    sequences: tuple[DecodedSequenceRange, ...]
    censoring_markers: tuple[GapCensoringMarker, ...]
    terminal_indices: tuple[int, ...]
    global_score: float
    validation_status: str = MODEL_VALIDATION_STATUS


@dataclass(frozen=True)
class _Node:
    state: TemporalState
    owner: str | None


_NODES: tuple[_Node, ...] = (
    _Node(TemporalState.DEAD_BALL, None),
    _Node(TemporalState.DEAD_BALL, TEAM_A),
    _Node(TemporalState.DEAD_BALL, TEAM_B),
    _Node(TemporalState.CONTROL_TEAM_A, TEAM_A),
    _Node(TemporalState.CONTROL_TEAM_B, TEAM_B),
    _Node(TemporalState.CONTESTED, None),
    _Node(TemporalState.CONTESTED, TEAM_A),
    _Node(TemporalState.CONTESTED, TEAM_B),
    _Node(TemporalState.BALL_FLIGHT, None),
    _Node(TemporalState.BALL_FLIGHT, TEAM_A),
    _Node(TemporalState.BALL_FLIGHT, TEAM_B),
    _Node(TemporalState.UNKNOWN, None),
    _Node(TemporalState.UNKNOWN, TEAM_A),
    _Node(TemporalState.UNKNOWN, TEAM_B),
)


def _control_team(state: TemporalState) -> str | None:
    if state is TemporalState.CONTROL_TEAM_A:
        return TEAM_A
    if state is TemporalState.CONTROL_TEAM_B:
        return TEAM_B
    return None


def _sample_allows_node(evidence: TemporalDecodeEvidence, index: int, node: _Node) -> bool:
    reset_team = evidence.reset_teams[index]
    if reset_team is not None:
        return _control_team(node.state) == reset_team
    established = evidence.established_control[index]
    if established is not None:
        return _control_team(node.state) == established
    if evidence.contested_touch[index]:
        return node.state is TemporalState.CONTESTED
    return True


def _initial_node_allowed(evidence: TemporalDecodeEvidence, index: int, node: _Node) -> bool:
    if not _sample_allows_node(evidence, index, node):
        return False
    control_team = _control_team(node.state)
    if control_team is not None:
        return True
    return node.owner is None


def _transition_score(
    previous: _Node,
    target: _Node,
    absolute_start: int,
    evidence: TemporalDecodeEvidence,
    config: SemiMarkovDecoderConfig,
) -> float:
    target_control = _control_team(target.state)
    if target_control is None:
        if target.owner != previous.owner:
            return -np.inf
    else:
        if target.owner != target_control:
            return -np.inf
        if previous.owner is not None and previous.owner != target_control:
            established = evidence.established_control[absolute_start]
            reset = evidence.reset_teams[absolute_start]
            if established != target_control and reset != target_control:
                return -np.inf
    if previous.state is target.state and previous.owner == target.owner:
        return config.latent_same_state_chunk_penalty
    score = config.state_change_penalty
    if previous.owner is not None and target.owner != previous.owner:
        score += config.owner_change_penalty
    return score


def _decode_observable_block(
    scores: np.ndarray,
    absolute_start: int,
    evidence: TemporalDecodeEvidence,
    config: SemiMarkovDecoderConfig,
    terminal_team: str | None,
) -> tuple[tuple[TemporalState, ...], tuple[str | None, ...], float]:
    """Segmental Viterbi for one block delimited by absorbing constraints."""

    length = scores.shape[0]
    node_count = len(_NODES)
    prefix = np.zeros((length + 1, len(TEMPORAL_STATES)), dtype=np.float64)
    prefix[1:] = np.cumsum(scores, axis=0)
    allowed = np.zeros((length, node_count), dtype=np.bool_)
    for local_index in range(length):
        absolute_index = absolute_start + local_index
        for node_index, node in enumerate(_NODES):
            allowed[local_index, node_index] = _sample_allows_node(
                evidence,
                absolute_index,
                node,
            )
    invalid_prefix = np.zeros((length + 1, node_count), dtype=np.int32)
    invalid_prefix[1:] = np.cumsum(~allowed, axis=0)

    dp = np.full((length + 1, node_count), -np.inf, dtype=np.float64)
    trace_previous = np.full((length + 1, node_count), -1, dtype=np.int16)
    trace_duration = np.zeros((length + 1, node_count), dtype=np.int16)
    # A transition into a segment depends only on the already-computed DP row
    # at that boundary.  Cache its best predecessor so a full half does not pay
    # an extra node_count factor for every candidate duration.
    boundary_score = np.full((length + 1, node_count), -np.inf, dtype=np.float64)
    boundary_previous = np.full((length + 1, node_count), -1, dtype=np.int16)
    epsilon = 1e-12

    for end in range(1, length + 1):
        for target_index, target in enumerate(_NODES):
            minimum, maximum = config.sample_duration(target.state)
            maximum = min(maximum, end)
            state_index = _STATE_INDEX[target.state]
            best_score = -np.inf
            best_previous = -1
            best_duration = 0
            # First and last segments are duration-censored by a hard block
            # boundary (period edge, terminal, or gap).  Interior segments keep
            # the full minimum-duration constraint.
            for duration in range(1, maximum + 1):
                start = end - duration
                if duration < minimum and start != 0 and end != length:
                    continue
                if invalid_prefix[end, target_index] != invalid_prefix[start, target_index]:
                    continue
                segment_score = prefix[end, state_index] - prefix[start, state_index]
                absolute_segment_start = absolute_start + start
                if start == 0:
                    if not _initial_node_allowed(evidence, absolute_segment_start, target):
                        continue
                    candidate = segment_score
                    previous_index = -1
                else:
                    candidate = boundary_score[start, target_index] + segment_score
                    previous_index = int(boundary_previous[start, target_index])
                if candidate > best_score + epsilon:
                    best_score = candidate
                    best_previous = previous_index
                    best_duration = duration
            dp[end, target_index] = best_score
            trace_previous[end, target_index] = best_previous
            trace_duration[end, target_index] = best_duration

        if end < length:
            absolute_boundary = absolute_start + end
            for target_index, target in enumerate(_NODES):
                best_transition_score = -np.inf
                best_source_index = -1
                for source_index, source in enumerate(_NODES):
                    previous_score = dp[end, source_index]
                    if not np.isfinite(previous_score):
                        continue
                    value = previous_score + _transition_score(
                        source,
                        target,
                        absolute_boundary,
                        evidence,
                        config,
                    )
                    if value > best_transition_score + epsilon:
                        best_transition_score = value
                        best_source_index = source_index
                boundary_score[end, target_index] = best_transition_score
                boundary_previous[end, target_index] = best_source_index

    viable = [
        index
        for index, node in enumerate(_NODES)
        if terminal_team is None or node.owner == terminal_team
    ]
    final_index = max(viable, key=lambda index: (dp[length, index], -index))
    if not np.isfinite(dp[length, final_index]):
        raise ValueError("hard evidence and duration constraints admit no temporal path")

    decoded_states: list[TemporalState | None] = [None] * length
    decoded_owners: list[str | None] = [None] * length
    end = length
    node_index = final_index
    while end > 0:
        duration = int(trace_duration[end, node_index])
        if duration <= 0:
            raise RuntimeError("invalid semi-Markov traceback")
        start = end - duration
        node = _NODES[node_index]
        for index in range(start, end):
            decoded_states[index] = node.state
            decoded_owners[index] = node.owner
        previous_index = int(trace_previous[end, node_index])
        end = start
        if end > 0:
            if previous_index < 0:
                raise RuntimeError("semi-Markov traceback ended before the block start")
            node_index = previous_index

    return (
        tuple(state for state in decoded_states if state is not None),
        tuple(decoded_owners),
        float(dp[length, final_index]),
    )


def _sequence_status(left_censored: bool, right_censored: bool, terminal: str | None) -> str:
    if left_censored and right_censored:
        return "uncertain"
    if left_censored:
        return "left_censored"
    if right_censored:
        return "right_censored"
    return "complete" if terminal is not None else "uncertain"


def decode_period_semi_markov(
    emission_scores: Sequence[Sequence[float]] | np.ndarray,
    *,
    evidence: TemporalDecodeEvidence | None = None,
    config: SemiMarkovDecoderConfig | None = None,
) -> TemporalDecodeResult:
    """Decode an entire period with global segmental Viterbi constraints.

    ``emission_scores`` must be ``T x 6`` in :data:`TEMPORAL_STATES` order.
    Terminal samples are absorbing ends for their sequence; decoding restarts at
    the next sample.  Observation gaps are forced to ``unknown`` and create
    two-sided censoring boundaries.  These hard boundaries factor the global
    objective into independent blocks without allowing a path to cross them.
    """

    scores = np.asarray(emission_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(TEMPORAL_STATES) or scores.shape[0] == 0:
        raise ValueError("emission_scores must be a non-empty T x 6 matrix")
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("emission_scores may contain finite values or negative infinity")
    if np.any(np.all(np.isneginf(scores), axis=1)):
        raise ValueError("every sample needs at least one finite emission score")
    settings = config or SemiMarkovDecoderConfig()
    aligned = (evidence or TemporalDecodeEvidence()).normalized(scores.shape[0])
    sample_count = scores.shape[0]
    states: list[TemporalState] = [TemporalState.UNKNOWN] * sample_count
    owners: list[str | None] = [None] * sample_count
    sequences: list[DecodedSequenceRange] = []
    block_ranges: list[tuple[int, int, int]] = []
    total_score = 0.0
    sequence_index = 0

    index = 0
    while index < sample_count:
        if aligned.observation_gap[index]:
            index += 1
            continue
        observable_end = index
        while observable_end + 1 < sample_count and not aligned.observation_gap[observable_end + 1]:
            observable_end += 1
        block_start = index
        while block_start <= observable_end:
            terminal_index = next(
                (
                    candidate
                    for candidate in range(block_start, observable_end + 1)
                    if aligned.terminal_types[candidate] is not None
                ),
                None,
            )
            block_end = terminal_index if terminal_index is not None else observable_end
            terminal_team = (
                aligned.terminal_teams[block_end]
                if terminal_index is not None
                else None
            )
            block_states, block_owners, block_score = _decode_observable_block(
                scores[block_start : block_end + 1],
                block_start,
                aligned,
                settings,
                terminal_team,
            )
            states[block_start : block_end + 1] = block_states
            owners[block_start : block_end + 1] = block_owners
            total_score += block_score
            previous_gap = block_start > 0 and aligned.observation_gap[block_start - 1]
            following_gap = block_end + 1 < sample_count and aligned.observation_gap[block_end + 1]
            left_censored = previous_gap or (block_start == 0 and aligned.left_censored)
            right_censored = (
                terminal_index is None
                and (following_gap or (block_end == sample_count - 1 and aligned.right_censored))
            )
            terminal_type = (
                aligned.terminal_types[terminal_index]
                if terminal_index is not None
                else None
            )
            sequences.append(
                DecodedSequenceRange(
                    sequence_index=sequence_index,
                    start_index=block_start,
                    end_index=block_end,
                    terminal_index=terminal_index,
                    terminal_type=terminal_type,
                    left_censored=left_censored,
                    right_censored=right_censored,
                    status=_sequence_status(left_censored, right_censored, terminal_type),
                )
            )
            block_ranges.append((block_start, block_end, sequence_index))
            sequence_index += 1
            block_start = block_end + 1
        index = observable_end + 1

    markers: list[GapCensoringMarker] = []
    index = 0
    while index < sample_count:
        if not aligned.observation_gap[index]:
            index += 1
            continue
        gap_start = index
        while index + 1 < sample_count and aligned.observation_gap[index + 1]:
            index += 1
        gap_end = index
        markers.append(
            GapCensoringMarker(
                start_index=gap_start,
                end_index=gap_end,
                left_sequence_right_censored=gap_start > 0,
                right_sequence_left_censored=gap_end + 1 < sample_count,
            )
        )
        index += 1

    segment_items: list[DecodedTemporalSegment] = []
    for block_start, block_end, block_sequence_index in block_ranges:
        segment_start = block_start
        for current in range(block_start + 1, block_end + 2):
            boundary = current > block_end or (
                states[current] is not states[segment_start]
                or owners[current] != owners[segment_start]
            )
            if boundary:
                segment_items.append(
                    DecodedTemporalSegment(
                        state=states[segment_start],
                        owner_context=owners[segment_start],
                        start_index=segment_start,
                        end_index=current - 1,
                        sequence_index=block_sequence_index,
                    )
                )
                segment_start = current
    for marker in markers:
        segment_items.append(
            DecodedTemporalSegment(
                state=TemporalState.UNKNOWN,
                owner_context=None,
                start_index=marker.start_index,
                end_index=marker.end_index,
                sequence_index=None,
            )
        )
    segment_items.sort(key=lambda item: (item.start_index, item.end_index))
    terminal_indices = tuple(
        index for index, value in enumerate(aligned.terminal_types) if value is not None
    )
    return TemporalDecodeResult(
        states=tuple(states),
        owner_context=tuple(owners),
        segments=tuple(segment_items),
        sequences=tuple(sequences),
        censoring_markers=tuple(markers),
        terminal_indices=terminal_indices,
        global_score=float(total_score),
    )


@dataclass(frozen=True)
class CheckpointCandidate:
    checkpoint_id: str
    human_dev_attack_f1: float
    human_dev_attack_precision: float
    human_dev_attack_recall: float
    epoch: int
    seed: int
    micro_action_f1: float | None = None
    validation_loss: float | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must not be empty")
        for name in (
            "human_dev_attack_f1",
            "human_dev_attack_precision",
            "human_dev_attack_recall",
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a finite probability")
        for name in ("micro_action_f1", "validation_loss"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if self.epoch < 0 or self.seed < 0:
            raise ValueError("epoch and seed must be non-negative")


def select_checkpoint_by_human_dev_attack_f1(
    candidates: Sequence[CheckpointCandidate],
) -> CheckpointCandidate:
    """Select only by human-development attack F1 and balanced-F1 tie-breaks.

    ``micro_action_f1`` and ``validation_loss`` are intentionally absent from
    the ordering key, so they can never substitute for the human sequence
    metric.
    """

    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.human_dev_attack_f1,
            abs(candidate.human_dev_attack_precision - candidate.human_dev_attack_recall),
            -min(candidate.human_dev_attack_precision, candidate.human_dev_attack_recall),
            candidate.seed,
            candidate.epoch,
            candidate.checkpoint_id,
        ),
    )


def checkpoint_selection_manifest(candidate: CheckpointCandidate) -> dict[str, object]:
    """Describe the selection decision without claiming that gates were met."""

    return {
        "selected_checkpoint": candidate.checkpoint_id,
        "selection_metric": "human_development_attack_f1_tiou_0.5",
        "selection_value": candidate.human_dev_attack_f1,
        "tie_breaks": ["minimum_precision_recall_gap", "maximum_min_precision_recall"],
        "ignored_for_selection": ["micro_action_f1", "validation_loss"],
        "validation_status": MODEL_VALIDATION_STATUS,
    }
