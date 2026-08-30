from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


XG_FEATURE_COLUMNS = (
    "x_shooter",
    "y_shooter",
    "shot_angle",
    "distance_to_goal",
    "x_goalkeeper",
    "y_goalkeeper",
    "foe_players",
    "ally_players",
    "defending_players",
    "under_pressure",
    "is_regular_play",
    "is_counter",
    "is_dead_ball",
    "is_penalty",
    "minute",
    "second",
)

EFFECTIVENESS_FEATURE_COLUMNS = (
    "duration_sec",
    "event_count",
    "progression_m",
    "progression_ratio",
    "width_m",
    "depth_m",
    "entered_final_third",
    "entered_penalty_box",
    "shots",
    "shots_on_target",
    "xg_sum",
    "xg_max",
    "attack_goal_probability",
    "positive_xt_gain",
    "tempo_events_per_sec",
    "ended_in_turnover",
    "event_uncertainty",
)


@dataclass(frozen=True)
class GroupSplit:
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train_groups": list(self.train_groups),
            "validation_groups": list(self.validation_groups),
            "test_groups": list(self.test_groups),
        }


def make_group_split(
    groups: Iterable[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> GroupSplit:
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError("at least three matches are required for grouped train/validation/test splits")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and leave room for a test set")
    random.Random(seed).shuffle(unique)
    train_end = max(1, round(len(unique) * train_fraction))
    validation_end = max(train_end + 1, round(len(unique) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(unique) - 1)
    return GroupSplit(
        train_groups=tuple(sorted(unique[:train_end])),
        validation_groups=tuple(sorted(unique[train_end:validation_end])),
        test_groups=tuple(sorted(unique[validation_end:])),
    )


def validate_no_target_leakage(feature_columns: Sequence[str], target: str) -> None:
    forbidden = {target, "goal", "statsbomb_xg", "label", "score", "outcome"}
    leaked = sorted(forbidden.intersection(feature_columns))
    if leaked:
        raise ValueError(f"target leakage detected in feature columns: {leaked}")


def _dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'ml' extra to train research models") from error
    return joblib, np, pd, HistGradientBoostingClassifier, LogisticRegression, (make_pipeline, StandardScaler)


def _classification_metrics(y_true: Any, probabilities: Any, *, multiclass: bool) -> dict[str, Any]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        log_loss,
        precision_recall_curve,
        roc_auc_score,
        f1_score,
        auc,
    )

    predictions = probabilities.argmax(axis=1) if multiclass else (probabilities >= 0.5).astype(int)
    result: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    if not multiclass:
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        result.update(
            {
                "roc_auc": float(roc_auc_score(y_true, probabilities)),
                "pr_auc": float(auc(recall, precision)),
                "brier": float(brier_score_loss(y_true, probabilities)),
            }
        )
    return result


def _fit_candidates(x_train: Any, y_train: Any, seed: int) -> dict[str, Any]:
    _, _, _, HistGradientBoostingClassifier, LogisticRegression, helpers = _dependencies()
    make_pipeline, StandardScaler = helpers
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ).fit(x_train, y_train),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        ).fit(x_train, y_train),
    }


def train_xg_models(
    csv_path: Path,
    output_path: Path,
    *,
    match_column: str = "match_id",
    target: str = "goal",
    seed: int = 42,
) -> dict[str, Any]:
    joblib, np, pd, _, LogisticRegression, _ = _dependencies()
    frame = pd.read_csv(csv_path)
    required = set(XG_FEATURE_COLUMNS) | {match_column, target}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"xG training data is missing columns: {missing}")
    validate_no_target_leakage(XG_FEATURE_COLUMNS, target)
    if not set(frame[target].dropna().unique()).issubset({0, 1}):
        raise ValueError("xG target must be binary goal outcome")
    split = make_group_split(frame[match_column].astype(str), seed=seed)
    masks = {
        "train": frame[match_column].astype(str).isin(split.train_groups),
        "validation": frame[match_column].astype(str).isin(split.validation_groups),
        "test": frame[match_column].astype(str).isin(split.test_groups),
    }
    medians = frame.loc[masks["train"], XG_FEATURE_COLUMNS].median(numeric_only=True)
    x = frame.loc[:, XG_FEATURE_COLUMNS].fillna(medians).fillna(0.0)
    y = frame[target].astype(int)
    candidates = _fit_candidates(x.loc[masks["train"]], y.loc[masks["train"]], seed)
    validation: dict[str, Any] = {}
    for name, model in candidates.items():
        probabilities = model.predict_proba(x.loc[masks["validation"]])[:, 1]
        validation[name] = _classification_metrics(y.loc[masks["validation"]], probabilities, multiclass=False)
    selected_name = min(validation, key=lambda name: validation[name]["brier"])
    selected = candidates[selected_name]
    validation_probabilities = selected.predict_proba(x.loc[masks["validation"]])[:, 1]
    calibrator = LogisticRegression(max_iter=1000, random_state=seed).fit(
        np.asarray(validation_probabilities).reshape(-1, 1),
        y.loc[masks["validation"]],
    )
    calibrated_validation = calibrator.predict_proba(
        np.asarray(validation_probabilities).reshape(-1, 1)
    )[:, 1]
    calibration_enabled = (
        _classification_metrics(
            y.loc[masks["validation"]], calibrated_validation, multiclass=False
        )["brier"]
        <= validation[selected_name]["brier"]
    )
    if not calibration_enabled:
        calibrator = None
    raw_test_probabilities = selected.predict_proba(x.loc[masks["test"]])[:, 1]
    test_probabilities = (
        calibrator.predict_proba(np.asarray(raw_test_probabilities).reshape(-1, 1))[:, 1]
        if calibrator is not None
        else raw_test_probabilities
    )
    from .xg import heuristic_xg

    test_rows = frame.loc[masks["test"]]
    heuristic_probabilities = np.asarray(
        [
            heuristic_xg((float(row.x_shooter), float(row.y_shooter)))
            for row in test_rows.itertuples(index=False)
        ],
        dtype=float,
    )
    report = {
        "task": "xg",
        "selected_model": selected_name,
        "feature_columns": list(XG_FEATURE_COLUMNS),
        "target": target,
        "split": split.to_dict(),
        "validation": validation,
        "calibration": {
            "method": "platt_logistic" if calibration_enabled else "none",
            "enabled": calibration_enabled,
        },
        "dataset": {
            "rows": int(len(frame)),
            "matches": int(frame[match_column].astype(str).nunique()),
            "goals": int(y.sum()),
            "goal_rate": float(y.mean()),
            "split_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        },
        "heuristic_test": _classification_metrics(
            y.loc[masks["test"]], heuristic_probabilities, multiclass=False
        ),
        "test": _classification_metrics(y.loc[masks["test"]], test_probabilities, multiclass=False),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": "xg-v1",
            "model": selected,
            "calibrator": calibrator,
            "medians": medians,
            "feature_columns": list(XG_FEATURE_COLUMNS),
            "report": report,
        },
        output_path,
    )
    output_path.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _flatten_effectiveness_row(payload: dict[str, Any]) -> dict[str, Any]:
    features = payload.get("features", payload)
    uncertainty = features.get("uncertainty", {})
    result = dict(features)
    result["event_uncertainty"] = float(uncertainty.get("event", 0.0))
    return result


def train_effectiveness_models(
    jsonl_path: Path,
    output_path: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    joblib, _, pd, _, _, _ = _dependencies()
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("effectiveness training set is empty")
    frame = pd.DataFrame(_flatten_effectiveness_row(row) for row in rows)
    labels = pd.Series([str(row["label"]) for row in rows])
    matches = pd.Series([str(row["match_id"]) for row in rows])
    allowed = {"ineffective": 0, "promising": 1, "effective": 2}
    if not set(labels).issubset(allowed):
        raise ValueError("effectiveness labels must be ineffective/promising/effective")
    validate_no_target_leakage(EFFECTIVENESS_FEATURE_COLUMNS, "label")
    missing = sorted(set(EFFECTIVENESS_FEATURE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"effectiveness training data is missing columns: {missing}")
    split = make_group_split(matches, seed=seed)
    masks = {
        "train": matches.isin(split.train_groups),
        "validation": matches.isin(split.validation_groups),
        "test": matches.isin(split.test_groups),
    }
    x = frame.loc[:, EFFECTIVENESS_FEATURE_COLUMNS].astype(float)
    medians = x.loc[masks["train"]].median(numeric_only=True)
    x = x.fillna(medians).fillna(0.0)
    y = labels.map(allowed).astype(int)
    candidates = _fit_candidates(x.loc[masks["train"]], y.loc[masks["train"]], seed)
    validation = {
        name: _classification_metrics(
            y.loc[masks["validation"]], model.predict_proba(x.loc[masks["validation"]]), multiclass=True
        )
        for name, model in candidates.items()
    }
    selected_name = max(validation, key=lambda name: validation[name]["macro_f1"])
    selected = candidates[selected_name]
    test_probabilities = selected.predict_proba(x.loc[masks["test"]])
    report = {
        "task": "attack_effectiveness",
        "selected_model": selected_name,
        "feature_columns": list(EFFECTIVENESS_FEATURE_COLUMNS),
        "class_mapping": allowed,
        "split": split.to_dict(),
        "validation": validation,
        "test": _classification_metrics(y.loc[masks["test"]], test_probabilities, multiclass=True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": selected, "medians": medians, "report": report}, output_path)
    output_path.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
