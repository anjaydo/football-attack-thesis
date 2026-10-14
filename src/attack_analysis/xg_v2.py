from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import exp, hypot, isfinite, log, pi
from numbers import Real
from pathlib import Path
import random
from typing import Mapping, Sequence

from .models import (
    ATTACK_QUALIFYING_STATUSES,
    AttackSequenceV2,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SequenceStatus,
    SourceTier,
    TerminalType,
)
from .xg import shot_angle, shot_distance


XG_V2_SCHEMA_VERSION = "xg-v2.0"
MAX_OBSERVED_PLAYERS = 22
MAX_SHOT_DISTANCE = hypot(PITCH_LENGTH, PITCH_WIDTH / 2.0)

# All model inputs are numeric. A zero fill never carries missing-value
# semantics alone: continuous/boolean fields have masks, categoricals have an
# explicit unknown level.
XG_V2_FEATURE_COLUMNS = (
    "x_shooter_norm",
    "y_shooter_norm",
    "shooter_location_missing",
    "shot_angle_norm",
    "shot_angle_missing",
    "distance_to_goal_norm",
    "distance_to_goal_missing",
    "body_part_right_foot",
    "body_part_left_foot",
    "body_part_head",
    "body_part_other",
    "body_part_unknown",
    "under_pressure",
    "under_pressure_missing",
    "goalkeeper_visible",
    "goalkeeper_visibility_missing",
    "x_goalkeeper_norm",
    "y_goalkeeper_norm",
    "goalkeeper_location_missing",
    "defenders_visible",
    "defenders_visibility_missing",
    "play_pattern_regular_play",
    "play_pattern_counter",
    "play_pattern_corner",
    "play_pattern_free_kick",
    "play_pattern_goal_kick",
    "play_pattern_kick_off",
    "play_pattern_throw_in",
    "play_pattern_other",
    "play_pattern_unknown",
    "shot_type_open_play",
    "shot_type_penalty",
    "shot_type_free_kick",
    "shot_type_other",
    "shot_type_unknown",
    "observed_players_norm",
    "observed_players_missing",
    "foe_players_norm",
    "foe_players_missing",
    "ally_players_norm",
    "ally_players_missing",
    "defending_players_norm",
    "defending_players_missing",
)

_BODY_PART_COLUMNS = {
    "right_foot": "body_part_right_foot",
    "left_foot": "body_part_left_foot",
    "head": "body_part_head",
    "other": "body_part_other",
    "unknown": "body_part_unknown",
}
_PLAY_PATTERN_COLUMNS = {
    "regular_play": "play_pattern_regular_play",
    "counter": "play_pattern_counter",
    "corner": "play_pattern_corner",
    "free_kick": "play_pattern_free_kick",
    "goal_kick": "play_pattern_goal_kick",
    "kick_off": "play_pattern_kick_off",
    "throw_in": "play_pattern_throw_in",
    "other": "play_pattern_other",
    "unknown": "play_pattern_unknown",
}
_SHOT_TYPE_COLUMNS = {
    "open_play": "shot_type_open_play",
    "penalty": "shot_type_penalty",
    "free_kick": "shot_type_free_kick",
    "other": "shot_type_other",
    "unknown": "shot_type_unknown",
}


def _raw_value(raw: Mapping[str, object], *names: str) -> object | None:
    return next((raw[name] for name in names if name in raw), None)


def _optional_number(raw: Mapping[str, object], *names: str) -> float | None:
    value = _raw_value(raw, *names)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{names[0]} must be numeric, not boolean")
    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError(f"{names[0]} must be numeric or null") from error
    else:
        raise ValueError(f"{names[0]} must be numeric or null")
    if number != number:  # tabular NaN means unknown
        return None
    if not isfinite(number):
        raise ValueError(f"{names[0]} must be finite")
    return number


def _optional_bool(raw: Mapping[str, object], *names: str) -> bool | None:
    value = _raw_value(raw, *names)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and float(value) in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        label = value.strip().casefold()
        if label in {"true", "yes", "y", "1"}:
            return True
        if label in {"false", "no", "n", "0"}:
            return False
    raise ValueError(f"{names[0]} must be boolean or null")


def _optional_count(raw: Mapping[str, object], name: str) -> int | None:
    value = _optional_number(raw, name)
    if value is None:
        return None
    if not value.is_integer() or not 0 <= value <= MAX_OBSERVED_PLAYERS:
        raise ValueError(f"{name} must be an integer between 0 and {MAX_OBSERVED_PLAYERS}")
    return int(value)


def _normalise_label(value: object | None) -> str | None:
    if value is None:
        return None
    label = str(value).strip().casefold()
    if not label:
        return None
    for character in ("-", "/", "."):
        label = label.replace(character, " ")
    return "_".join(label.split())


def _body_part_category(value: object | None) -> str:
    label = _normalise_label(value)
    if label is None or label in {"unknown", "not_available", "na", "n_a"}:
        return "unknown"
    if label in {"right_foot", "rightfoot", "foot_right"}:
        return "right_foot"
    if label in {"left_foot", "leftfoot", "foot_left"}:
        return "left_foot"
    if label in {"head", "headed", "header"}:
        return "head"
    if label in {"foot", "feet", "boot"}:
        return "right_foot"
    return "other"


def _play_pattern_category(value: object | None) -> str:
    label = _normalise_label(value)
    if label is None or label in {"unknown", "not_available", "na", "n_a"}:
        return "unknown"
    aliases = {
        "regular_play": "regular_play",
        "from_counter": "counter",
        "counter": "counter",
        "from_corner": "corner",
        "corner": "corner",
        "from_free_kick": "free_kick",
        "free_kick": "free_kick",
        "from_goal_kick": "goal_kick",
        "goal_kick": "goal_kick",
        "from_kick_off": "kick_off",
        "kick_off": "kick_off",
        "from_throw_in": "throw_in",
        "throw_in": "throw_in",
    }
    return aliases.get(label, "other")


def _shot_type_category(value: object | None) -> str:
    label = _normalise_label(value)
    if label is None or label in {"unknown", "not_available", "na", "n_a"}:
        return "unknown"
    if label in {"open_play", "regular_play"}:
        return "open_play"
    if label in {"penalty", "penalty_kick"}:
        return "penalty"
    if label in {"free_kick", "direct_free_kick"}:
        return "free_kick"
    return "other"


def _set_one_hot(
    features: dict[str, float], columns: Mapping[str, str], selected: str
) -> None:
    for category, column in columns.items():
        features[column] = float(category == selected)


def _normalised_count(
    count: int | None, observed_players: int | None
) -> tuple[float, float]:
    if count is None or observed_players is None or observed_players <= 0:
        return 0.0, 1.0
    return count / observed_players, 0.0


def normalize_xg_v2_features(raw: Mapping[str, object]) -> dict[str, float]:
    """Build the canonical numeric feature row without hiding missingness.

    Input coordinates are metres on a 105 x 68 pitch with the attacking goal
    at x=105. ``ally_players`` excludes the shooter. Counts are divided by the
    actual observed-player count; if that count is absent but ally and foe
    counts exist, it is derived as ``1 + ally + foe``.
    """

    if not isinstance(raw, Mapping):
        raise TypeError("raw xG features must be a mapping")
    features = {column: 0.0 for column in XG_V2_FEATURE_COLUMNS}

    x_shooter = _optional_number(raw, "x_shooter")
    y_shooter = _optional_number(raw, "y_shooter")
    if x_shooter is not None and not 0 <= x_shooter <= PITCH_LENGTH:
        raise ValueError(f"x_shooter must be between 0 and {PITCH_LENGTH}")
    if y_shooter is not None and not 0 <= y_shooter <= PITCH_WIDTH:
        raise ValueError(f"y_shooter must be between 0 and {PITCH_WIDTH}")
    location_known = x_shooter is not None and y_shooter is not None
    if location_known:
        features["x_shooter_norm"] = x_shooter / PITCH_LENGTH
        features["y_shooter_norm"] = y_shooter / PITCH_WIDTH
    features["shooter_location_missing"] = float(not location_known)

    angle = _optional_number(raw, "shot_angle", "shot_angle_rad")
    distance = _optional_number(raw, "distance_to_goal", "shot_distance")
    if angle is None and location_known:
        angle = shot_angle((x_shooter, y_shooter))
    if distance is None and location_known:
        distance = shot_distance((x_shooter, y_shooter))
    if angle is not None and not 0 <= angle <= pi:
        raise ValueError("shot_angle must be in radians between 0 and pi")
    if distance is not None and not 0 <= distance <= MAX_SHOT_DISTANCE + 1e-9:
        raise ValueError("distance_to_goal is outside the canonical pitch range")
    if angle is None:
        features["shot_angle_missing"] = 1.0
    else:
        features["shot_angle_norm"] = angle / pi
    if distance is None:
        features["distance_to_goal_missing"] = 1.0
    else:
        features["distance_to_goal_norm"] = distance / MAX_SHOT_DISTANCE

    _set_one_hot(features, _BODY_PART_COLUMNS, _body_part_category(raw.get("body_part")))
    pressure = _optional_bool(raw, "under_pressure")
    features["under_pressure"] = float(pressure is True)
    features["under_pressure_missing"] = float(pressure is None)

    x_goalkeeper = _optional_number(raw, "x_goalkeeper")
    y_goalkeeper = _optional_number(raw, "y_goalkeeper")
    if x_goalkeeper is not None and not 0 <= x_goalkeeper <= PITCH_LENGTH:
        raise ValueError(f"x_goalkeeper must be between 0 and {PITCH_LENGTH}")
    if y_goalkeeper is not None and not 0 <= y_goalkeeper <= PITCH_WIDTH:
        raise ValueError(f"y_goalkeeper must be between 0 and {PITCH_WIDTH}")
    goalkeeper_location_known = x_goalkeeper is not None and y_goalkeeper is not None
    goalkeeper_visible = _optional_bool(raw, "goalkeeper_visible")
    if goalkeeper_location_known and goalkeeper_visible is False:
        raise ValueError("goalkeeper coordinates conflict with goalkeeper_visible=False")
    if goalkeeper_visible is None and goalkeeper_location_known:
        goalkeeper_visible = True
    features["goalkeeper_visible"] = float(goalkeeper_visible is True)
    features["goalkeeper_visibility_missing"] = float(goalkeeper_visible is None)
    if goalkeeper_location_known:
        features["x_goalkeeper_norm"] = x_goalkeeper / PITCH_LENGTH
        features["y_goalkeeper_norm"] = y_goalkeeper / PITCH_WIDTH
    features["goalkeeper_location_missing"] = float(not goalkeeper_location_known)

    foe_players = _optional_count(raw, "foe_players")
    ally_players = _optional_count(raw, "ally_players")
    defending_players = _optional_count(raw, "defending_players")
    observed_players = _optional_count(raw, "observed_players")
    if observed_players is None and foe_players is not None and ally_players is not None:
        observed_players = 1 + foe_players + ally_players
        if observed_players > MAX_OBSERVED_PLAYERS:
            raise ValueError("derived observed_players exceeds 22")
    for name, count in (
        ("foe_players", foe_players),
        ("ally_players", ally_players),
        ("defending_players", defending_players),
    ):
        if count is not None and observed_players is not None and count > observed_players:
            raise ValueError(f"{name} cannot exceed observed_players")
    if (
        defending_players is not None
        and foe_players is not None
        and defending_players > foe_players
    ):
        raise ValueError("defending_players cannot exceed foe_players")

    defenders_visible = _optional_bool(raw, "defenders_visible", "defender_visible")
    if defenders_visible is False and any(
        count is not None and count > 0 for count in (foe_players, defending_players)
    ):
        raise ValueError("positive defender counts conflict with defenders_visible=False")
    if defenders_visible is None and (foe_players is not None or defending_players is not None):
        defenders_visible = True
    features["defenders_visible"] = float(defenders_visible is True)
    features["defenders_visibility_missing"] = float(defenders_visible is None)

    _set_one_hot(
        features, _PLAY_PATTERN_COLUMNS, _play_pattern_category(raw.get("play_pattern"))
    )
    _set_one_hot(features, _SHOT_TYPE_COLUMNS, _shot_type_category(raw.get("shot_type")))

    if observed_players is None:
        features["observed_players_missing"] = 1.0
    else:
        features["observed_players_norm"] = observed_players / MAX_OBSERVED_PLAYERS
    for name, count in (
        ("foe_players", foe_players),
        ("ally_players", ally_players),
        ("defending_players", defending_players),
    ):
        value, missing = _normalised_count(count, observed_players)
        features[f"{name}_norm"] = value
        features[f"{name}_missing"] = missing

    if tuple(features) != XG_V2_FEATURE_COLUMNS:
        raise RuntimeError("internal xG-v2 feature schema drift")
    if any(not isfinite(value) for value in features.values()):
        raise ValueError("normalised xG-v2 features must all be finite")
    return features


@dataclass(frozen=True)
class XGFeatureVectorV2:
    values: tuple[float, ...]
    schema_version: str = XG_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        if self.schema_version != XG_V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {XG_V2_SCHEMA_VERSION}")
        if len(self.values) != len(XG_V2_FEATURE_COLUMNS):
            raise ValueError("feature vector length does not match XG_V2_FEATURE_COLUMNS")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("feature vector values must be finite")

    @classmethod
    def from_raw(cls, raw: Mapping[str, object]) -> "XGFeatureVectorV2":
        values = normalize_xg_v2_features(raw)
        return cls(tuple(values[column] for column in XG_V2_FEATURE_COLUMNS))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "XGFeatureVectorV2":
        missing = set(XG_V2_FEATURE_COLUMNS).difference(values)
        extra = set(values).difference(XG_V2_FEATURE_COLUMNS)
        if missing or extra:
            raise ValueError(
                f"feature mapping schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return cls(tuple(float(values[column]) for column in XG_V2_FEATURE_COLUMNS))

    def to_mapping(self) -> dict[str, float]:
        return dict(zip(XG_V2_FEATURE_COLUMNS, self.values, strict=True))


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a numeric probability")
    probability = float(value)
    if not isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return probability


_POSITIVE_OUTCOMES = {"goal", "scored", "1", "true"}
_NEGATIVE_OUTCOMES = {
    "blocked",
    "off_target",
    "on_target",
    "post",
    "saved",
    "saved_off_target",
    "saved_to_post",
    "wayward",
    "no_goal",
    "0",
    "false",
}


def _binary_outcome(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and float(value) in (0.0, 1.0):
        return int(value)
    label = _normalise_label(value)
    if label is None or label in {
        "unknown",
        "unlabelled",
        "unlabeled",
        "unverified",
        "na",
        "n_a",
    }:
        return None
    if label in _POSITIVE_OUTCOMES:
        return 1
    if label in _NEGATIVE_OUTCOMES:
        return 0
    raise ValueError(f"unsupported shot outcome: {value!r}")


def calibration_slope_intercept(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> tuple[float | None, float | None]:
    """Fit outcome ~ intercept + slope*logit(p), or None if unidentified."""

    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    checked_probabilities = [
        _probability(value, "probability") for value in probabilities
    ]
    checked_outcomes: list[int] = []
    for value in outcomes:
        parsed = _binary_outcome(value)
        if parsed is None:
            raise ValueError("calibration requires observed binary outcomes")
        checked_outcomes.append(parsed)
    if len(checked_probabilities) < 3 or len(set(checked_outcomes)) < 2:
        return None, None
    logits = [
        log(min(1 - 1e-6, max(1e-6, value)) / (1 - min(1 - 1e-6, max(1e-6, value))))
        for value in checked_probabilities
    ]
    if max(logits) - min(logits) <= 1e-12:
        return None, None
    # Identity calibration (intercept=0, slope=1) is the stable starting
    # point.  Initialising the intercept at the outcome log-odds while also
    # keeping slope=1 double-counts the base rate and can make Newton steps
    # diverge on otherwise well-behaved, imbalanced football data.
    intercept = 0.0
    slope = 1.0
    converged = False
    for _ in range(max_iterations):
        fitted = [_sigmoid(intercept + slope * value) for value in logits]
        weights = [value * (1 - value) for value in fitted]
        h00 = sum(weights)
        h01 = sum(w * x for w, x in zip(weights, logits, strict=True))
        h11 = sum(w * x * x for w, x in zip(weights, logits, strict=True))
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-14 or not isfinite(determinant):
            return None, None
        gradient0 = sum(y - p for y, p in zip(checked_outcomes, fitted, strict=True))
        gradient1 = sum(
            (y - p) * x
            for y, p, x in zip(checked_outcomes, fitted, logits, strict=True)
        )
        delta_intercept = (gradient0 * h11 - gradient1 * h01) / determinant
        delta_slope = (gradient1 * h00 - gradient0 * h01) / determinant
        intercept += delta_intercept
        slope += delta_slope
        if not isfinite(intercept) or not isfinite(slope) or max(abs(intercept), abs(slope)) > 50:
            return None, None
        if max(abs(delta_intercept), abs(delta_slope)) <= tolerance:
            converged = True
            break
    return (slope, intercept) if converged else (None, None)


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = exp(-value)
        return 1 / (1 + inverse)
    exponential = exp(value)
    return exponential / (1 + exponential)


@dataclass(frozen=True)
class XGMetricsV2:
    observations: int
    labelled_outcomes: int
    scored_labelled: int
    missing_outcomes: int
    missing_predictions: int
    brier: float | None
    ece_10: float | None
    calibration_slope: float | None
    calibration_intercept: float | None

    @property
    def coverage(self) -> float | None:
        return (
            self.scored_labelled / self.labelled_outcomes
            if self.labelled_outcomes
            else None
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "observations": self.observations,
            "labelled_outcomes": self.labelled_outcomes,
            "scored_labelled": self.scored_labelled,
            "missing_outcomes": self.missing_outcomes,
            "missing_predictions": self.missing_predictions,
            "coverage": self.coverage,
            "brier": self.brier,
            "ece_10": self.ece_10,
            "calibration_slope": self.calibration_slope,
            "calibration_intercept": self.calibration_intercept,
        }


def evaluate_xg_v2(
    probabilities: Sequence[float | None],
    outcomes: Sequence[int | bool | str | None],
    *,
    bins: int = 10,
) -> XGMetricsV2:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    labelled = missing_outcomes = missing_predictions = 0
    paired_probabilities: list[float] = []
    paired_outcomes: list[int] = []
    for raw_probability, raw_outcome in zip(probabilities, outcomes, strict=True):
        outcome = _binary_outcome(raw_outcome)
        if outcome is None:
            missing_outcomes += 1
            if raw_probability is not None:
                _probability(raw_probability, "probability")
            continue
        labelled += 1
        if raw_probability is None:
            missing_predictions += 1
            continue
        paired_probabilities.append(_probability(raw_probability, "probability"))
        paired_outcomes.append(outcome)
    if not paired_probabilities:
        return XGMetricsV2(
            len(probabilities), labelled, 0, missing_outcomes, missing_predictions,
            None, None, None, None,
        )
    brier = sum(
        (p - y) ** 2 for p, y in zip(paired_probabilities, paired_outcomes, strict=True)
    ) / len(paired_probabilities)
    members: list[list[int]] = [[] for _ in range(bins)]
    for index, probability in enumerate(paired_probabilities):
        members[min(int(probability * bins), bins - 1)].append(index)
    ece = 0.0
    for indices in members:
        if indices:
            mean_p = sum(paired_probabilities[index] for index in indices) / len(indices)
            mean_y = sum(paired_outcomes[index] for index in indices) / len(indices)
            ece += len(indices) / len(paired_probabilities) * abs(mean_p - mean_y)
    slope, intercept = calibration_slope_intercept(paired_probabilities, paired_outcomes)
    return XGMetricsV2(
        len(probabilities), labelled, len(paired_probabilities), missing_outcomes,
        missing_predictions, brier, ece, slope, intercept,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class PairedBrierBootstrapV2:
    delta_candidate_minus_baseline: float
    lower_95: float
    upper_95: float
    confidence: float
    bootstrap_samples: int
    seed: int
    matches: int
    observations: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "delta_candidate_minus_baseline": self.delta_candidate_minus_baseline,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "confidence": self.confidence,
            "bootstrap_samples": self.bootstrap_samples,
            "seed": self.seed,
            "matches": self.matches,
            "observations": self.observations,
        }


def paired_match_cluster_bootstrap_brier_delta(
    match_ids: Sequence[str],
    outcomes: Sequence[int | bool | str],
    candidate_probabilities: Sequence[float],
    baseline_probabilities: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> PairedBrierBootstrapV2:
    lengths = {
        len(match_ids), len(outcomes), len(candidate_probabilities), len(baseline_probabilities)
    }
    if len(lengths) != 1 or not match_ids:
        raise ValueError("paired bootstrap inputs must be non-empty and have equal length")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    row_deltas: list[float] = []
    cluster_indices: dict[str, list[int]] = {}
    for index, (match_id, raw_y, raw_candidate, raw_baseline) in enumerate(
        zip(match_ids, outcomes, candidate_probabilities, baseline_probabilities, strict=True)
    ):
        match = str(match_id).strip()
        if not match:
            raise ValueError("match IDs must not be empty")
        outcome = _binary_outcome(raw_y)
        if outcome is None:
            raise ValueError("paired bootstrap requires observed binary outcomes")
        candidate = _probability(raw_candidate, "candidate probability")
        baseline = _probability(raw_baseline, "baseline probability")
        row_deltas.append((candidate - outcome) ** 2 - (baseline - outcome) ** 2)
        cluster_indices.setdefault(match, []).append(index)
    clusters = sorted(cluster_indices)
    if len(clusters) < 2:
        raise ValueError("match-cluster bootstrap requires at least two matches")
    point_delta = sum(row_deltas) / len(row_deltas)
    generator = random.Random(seed)
    sampled_deltas = []
    for _ in range(samples):
        sampled_rows = []
        for _ in clusters:
            sampled_rows.extend(cluster_indices[generator.choice(clusters)])
        sampled_deltas.append(
            sum(row_deltas[index] for index in sampled_rows) / len(sampled_rows)
        )
    alpha = (1 - confidence) / 2
    return PairedBrierBootstrapV2(
        point_delta,
        _percentile(sampled_deltas, alpha),
        _percentile(sampled_deltas, 1 - alpha),
        confidence,
        samples,
        seed,
        len(clusters),
        len(row_deltas),
    )


@dataclass(frozen=True)
class XGAcceptanceGatesV2:
    complete_coverage: bool
    brier: bool
    ece_10: bool
    calibration_slope: bool
    calibration_intercept: bool
    paired_brier_delta_upper_ci: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.complete_coverage,
                self.brier,
                self.ece_10,
                self.calibration_slope,
                self.calibration_intercept,
                self.paired_brier_delta_upper_ci,
            )
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "passed": self.passed,
            "complete_coverage": self.complete_coverage,
            "brier_le_0_09": self.brier,
            "ece_10_le_0_03": self.ece_10,
            "calibration_slope_0_8_to_1_2": self.calibration_slope,
            "absolute_calibration_intercept_le_0_10": self.calibration_intercept,
            "paired_brier_delta_upper_95_lt_0": self.paired_brier_delta_upper_ci,
        }


def xg_v2_acceptance_gates(
    metrics: XGMetricsV2, paired_delta: PairedBrierBootstrapV2 | None
) -> XGAcceptanceGatesV2:
    """Apply the locked release gates; absent evidence always fails."""

    return XGAcceptanceGatesV2(
        complete_coverage=(
            metrics.coverage == 1.0
            and metrics.scored_labelled > 0
            and metrics.missing_outcomes == 0
        ),
        brier=metrics.brier is not None and metrics.brier <= 0.09,
        ece_10=metrics.ece_10 is not None and metrics.ece_10 <= 0.03,
        calibration_slope=(
            metrics.calibration_slope is not None
            and 0.8 <= metrics.calibration_slope <= 1.2
        ),
        calibration_intercept=(
            metrics.calibration_intercept is not None
            and abs(metrics.calibration_intercept) <= 0.10
        ),
        paired_brier_delta_upper_ci=(
            paired_delta is not None and paired_delta.upper_95 < 0
        ),
    )


_TERMINAL_OUTCOME = object()


def _stable_record_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"xgv2_{hashlib.sha256(encoded).hexdigest()[:24]}"


@dataclass(frozen=True)
class ConfirmedXGRecordV2:
    attack_id: str
    possession_id: str
    match_id: str
    period: int
    team_id: str
    terminal_id: str
    terminal_type: TerminalType
    source_tier: SourceTier
    xg_probability: float
    model_id: str
    outcome: int | None = None
    features: XGFeatureVectorV2 | None = None
    record_id: str = ""
    schema_version: str = XG_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        terminal_type = (
            self.terminal_type
            if isinstance(self.terminal_type, TerminalType)
            else TerminalType(str(self.terminal_type))
        )
        source_tier = (
            self.source_tier
            if isinstance(self.source_tier, SourceTier)
            else SourceTier(str(self.source_tier))
        )
        object.__setattr__(self, "terminal_type", terminal_type)
        object.__setattr__(self, "source_tier", source_tier)
        object.__setattr__(self, "xg_probability", _probability(self.xg_probability, "xg"))
        object.__setattr__(self, "outcome", _binary_outcome(self.outcome))
        if self.schema_version != XG_V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {XG_V2_SCHEMA_VERSION}")
        if terminal_type not in {TerminalType.SHOT_ATTEMPT, TerminalType.HEADED_ATTEMPT}:
            raise ValueError("xG-v2 records require a shot_attempt or headed_attempt terminal")
        if source_tier is SourceTier.LEGACY_UNVALIDATED:
            raise ValueError("legacy_unvalidated sequences cannot receive xG-v2")
        if self.period < 1:
            raise ValueError("period must be positive")
        for name in (
            "attack_id", "possession_id", "match_id", "team_id", "terminal_id", "model_id"
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        expected = self.expected_record_id()
        if not self.record_id:
            object.__setattr__(self, "record_id", expected)
        elif self.record_id != expected:
            raise ValueError("record_id does not match the stable xG-v2 content ID")

    def expected_record_id(self) -> str:
        return _stable_record_id(
            {
                "attack_id": self.attack_id,
                "terminal_id": self.terminal_id,
                "model_id": self.model_id,
                "schema_version": self.schema_version,
            }
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "attack_id": self.attack_id,
            "possession_id": self.possession_id,
            "match_id": self.match_id,
            "period": self.period,
            "team_id": self.team_id,
            "terminal_id": self.terminal_id,
            "terminal_type": self.terminal_type.value,
            "source_tier": self.source_tier.value,
            "xg_probability": self.xg_probability,
            "model_id": self.model_id,
            "outcome": self.outcome,
        }
        if self.features is not None:
            payload["features"] = self.features.to_mapping()
            payload["feature_schema_version"] = self.features.schema_version
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConfirmedXGRecordV2":
        raw_features = payload.get("features")
        if raw_features is not None and not isinstance(raw_features, Mapping):
            raise ValueError("features must be a mapping or null")
        return cls(
            attack_id=str(payload["attack_id"]),
            possession_id=str(payload["possession_id"]),
            match_id=str(payload["match_id"]),
            period=int(payload["period"]),
            team_id=str(payload["team_id"]),
            terminal_id=str(payload["terminal_id"]),
            terminal_type=TerminalType(str(payload["terminal_type"])),
            source_tier=SourceTier(str(payload["source_tier"])),
            xg_probability=float(payload["xg_probability"]),
            model_id=str(payload["model_id"]),
            outcome=_binary_outcome(payload.get("outcome")),
            features=(
                XGFeatureVectorV2.from_mapping(raw_features)
                if raw_features is not None
                else None
            ),
            record_id=str(payload.get("record_id", "")),
            schema_version=str(payload.get("schema_version", XG_V2_SCHEMA_VERSION)),
        )


def score_confirmed_attack_xg_v2(
    attack: object,
    xg_probability: float,
    *,
    model_id: str,
    outcome: int | bool | str | None | object = _TERMINAL_OUTCOME,
    features: XGFeatureVectorV2 | Mapping[str, object] | None = None,
) -> ConfirmedXGRecordV2:
    """Gate xG behind a complete independently-confirmed shot terminal."""

    if not isinstance(attack, AttackSequenceV2):
        raise TypeError("xG-v2 scoring requires an AttackSequenceV2")
    possession = attack.possession
    if possession.status not in ATTACK_QUALIFYING_STATUSES:
        raise ValueError("right-censored or uncertain sequences cannot receive xG-v2")
    if possession.source_tier is SourceTier.LEGACY_UNVALIDATED:
        raise ValueError("legacy_unvalidated sequences cannot receive xG-v2")
    terminal = attack.terminal
    if terminal.terminal_type not in {
        TerminalType.SHOT_ATTEMPT, TerminalType.HEADED_ATTEMPT
    }:
        raise ValueError("xG-v2 requires a shot_attempt or headed_attempt terminal")
    resolved_outcome = (
        _binary_outcome(terminal.outcome)
        if outcome is _TERMINAL_OUTCOME
        else _binary_outcome(outcome)
    )
    if features is not None and not isinstance(features, XGFeatureVectorV2):
        features = (
            XGFeatureVectorV2.from_mapping(features)
            if set(features) == set(XG_V2_FEATURE_COLUMNS)
            else XGFeatureVectorV2.from_raw(features)
        )
    return ConfirmedXGRecordV2(
        attack_id=attack.attack_id,
        possession_id=possession.possession_id,
        match_id=attack.match_id,
        period=attack.period,
        team_id=attack.team_id,
        terminal_id=terminal.terminal_id,
        terminal_type=terminal.terminal_type,
        source_tier=possession.source_tier,
        xg_probability=xg_probability,
        model_id=model_id,
        outcome=resolved_outcome,
        features=features,
    )


@dataclass(frozen=True)
class TrainedXGModelV2:
    """Checksum-addressed xG-v2 inference artifact.

    The artifact accepts only the exact feature schema used during training;
    callers may provide either raw semantic inputs or a canonical feature
    vector.  Loading performs no V1 imputation or schema coercion.
    """

    model: object
    calibrator: object | None
    calibration_input: str
    model_sha256: str
    feature_columns: tuple[str, ...]
    report: Mapping[str, object]

    @property
    def model_id(self) -> str:
        return f"xg-v2:sha256:{self.model_sha256}"

    def predict(
        self,
        features: XGFeatureVectorV2 | Mapping[str, object],
    ) -> float:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'ml' extra to run xG-v2 inference") from error
        if isinstance(features, XGFeatureVectorV2):
            vector = features
        elif set(features) == set(XG_V2_FEATURE_COLUMNS):
            vector = XGFeatureVectorV2.from_mapping(features)
        else:
            vector = XGFeatureVectorV2.from_raw(features)
        row = pd.DataFrame([vector.to_mapping()], columns=self.feature_columns)
        probability = float(self.model.predict_proba(row)[:, 1][0])
        if self.calibrator is not None:
            if self.calibration_input != "logit_probability":
                raise ValueError("unsupported xG-v2 calibration input contract")
            clipped = min(1.0 - 1e-6, max(1e-6, probability))
            calibration_value = log(clipped / (1.0 - clipped))
            probability = float(
                self.calibrator.predict_proba(
                    np.asarray([[calibration_value]], dtype=float)
                )[:, 1][0]
            )
        return _probability(probability, "predicted xG")

    def score_attack(
        self,
        attack: object,
        features: XGFeatureVectorV2 | Mapping[str, object],
        *,
        outcome: int | bool | str | None | object = _TERMINAL_OUTCOME,
    ) -> ConfirmedXGRecordV2:
        vector = (
            features
            if isinstance(features, XGFeatureVectorV2)
            else XGFeatureVectorV2.from_mapping(features)
            if set(features) == set(XG_V2_FEATURE_COLUMNS)
            else XGFeatureVectorV2.from_raw(features)
        )
        return score_confirmed_attack_xg_v2(
            attack,
            self.predict(vector),
            model_id=self.model_id,
            outcome=outcome,
            features=vector,
        )


def load_trained_xg_v2_model(
    artifact_path: str | Path,
    *,
    expected_sha256: str | None = None,
    allow_research_only: bool = False,
) -> TrainedXGModelV2:
    """Load an exact-schema xG-v2 artifact and optionally pin its checksum."""

    try:
        import joblib
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'ml' extra to load xG-v2") from error
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"trained xG-v2 artifact not found: {path}")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256.casefold():
        raise ValueError("xG-v2 model checksum does not match the frozen release")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("xG-v2 artifact payload must be a mapping")
    if payload.get("schema_version") != XG_V2_SCHEMA_VERSION:
        raise ValueError(
            f"xG-v2 artifact schema must be {XG_V2_SCHEMA_VERSION}"
        )
    feature_columns = tuple(payload.get("feature_columns") or ())
    if feature_columns != XG_V2_FEATURE_COLUMNS:
        raise ValueError("xG-v2 artifact feature schema does not match the runtime")
    if "model" not in payload:
        raise ValueError("xG-v2 artifact does not contain a trained model")
    calibration_input = str(payload.get("calibration_input") or "none")
    calibrator = payload.get("calibrator")
    if calibrator is not None and calibration_input != "logit_probability":
        raise ValueError("xG-v2 calibrated artifact must declare logit_probability input")
    report = payload.get("report") or {}
    if not isinstance(report, Mapping):
        raise ValueError("xG-v2 artifact report must be a mapping")
    release_status = str(report.get("status") or "")
    if release_status != "validated" and not allow_research_only:
        raise PermissionError(
            "xG-v2 artifact is not domain-validated; pass allow_research_only=True "
            "only for explicitly labelled research output"
        )
    return TrainedXGModelV2(
        model=payload["model"],
        calibrator=calibrator,
        calibration_input=calibration_input,
        model_sha256=checksum,
        feature_columns=feature_columns,
        report=report,
    )
