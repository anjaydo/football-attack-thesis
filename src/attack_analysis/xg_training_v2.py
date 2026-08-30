from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .ml import make_group_split, validate_no_target_leakage
from .xg_v2 import (
    XG_V2_SCHEMA_VERSION,
    XG_V2_FEATURE_COLUMNS,
    evaluate_xg_v2,
    paired_match_cluster_bootstrap_brier_delta,
    xg_v2_acceptance_gates,
)


GEOMETRY_LOGISTIC_COLUMNS = (
    "x_shooter_norm",
    "y_shooter_norm",
    "shot_angle_norm",
    "distance_to_goal_norm",
)


@dataclass(frozen=True)
class XGTrainingV2Config:
    seed: int = 42
    inner_folds: int = 3
    bootstrap_samples: int = 2000

    def __post_init__(self) -> None:
        if self.inner_folds < 2:
            raise ValueError("inner_folds must be at least 2")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")


def _dependencies() -> tuple[Any, ...]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'ml' extra to train xG-v2") from error
    return (
        joblib,
        np,
        pd,
        HistGradientBoostingClassifier,
        LogisticRegression,
        GroupKFold,
        make_pipeline,
        StandardScaler,
    )


def _new_candidate(name: str, seed: int) -> Any:
    (
        _,
        _,
        _,
        HistGradientBoostingClassifier,
        LogisticRegression,
        _,
        make_pipeline,
        StandardScaler,
    ) = _dependencies()
    if name == "geometry_logistic":
        from sklearn.compose import ColumnTransformer

        return make_pipeline(
            ColumnTransformer(
                [("geometry", "passthrough", list(GEOMETRY_LOGISTIC_COLUMNS))],
                remainder="drop",
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                random_state=seed,
            ),
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"unknown xG-v2 candidate: {name}")


def _brier(outcomes: Any, probabilities: Any) -> float:
    return float(((probabilities - outcomes) ** 2).mean())


def _probability_logits(np: Any, probabilities: Any) -> Any:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _validate_frame(frame: Any, match_column: str, target: str) -> None:
    required = set(XG_V2_FEATURE_COLUMNS) | {match_column, target}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"xG-v2 training data is missing columns: {missing}")
    validate_no_target_leakage(XG_V2_FEATURE_COLUMNS, target)
    if not set(frame[target].dropna().unique()).issubset({0, 1}):
        raise ValueError("xG-v2 target must be binary")
    if frame[target].isna().any():
        raise ValueError("StatsBomb xG-v2 training requires observed outcomes")
    if frame[match_column].astype(str).str.strip().eq("").any():
        raise ValueError("match IDs must not be empty")


def train_xg_v2_models(
    csv_path: Path,
    output_path: Path,
    *,
    match_column: str = "match_id",
    target: str = "goal",
    config: XGTrainingV2Config | None = None,
) -> dict[str, object]:
    """Train/select/calibrate xG-v2 without touching the outer test fold.

    Candidate selection uses grouped inner CV only on the outer training split.
    Platt calibration is fit on the disjoint outer calibration split; the outer
    test split is evaluated once after all choices are fixed.
    """

    settings = config or XGTrainingV2Config()
    (
        joblib,
        np,
        pd,
        _,
        LogisticRegression,
        GroupKFold,
        _,
        _,
    ) = _dependencies()
    frame = pd.read_csv(csv_path)
    _validate_frame(frame, match_column, target)
    groups = frame[match_column].astype(str)
    split = make_group_split(groups, seed=settings.seed)
    masks = {
        "train": groups.isin(split.train_groups),
        "calibration": groups.isin(split.validation_groups),
        "test": groups.isin(split.test_groups),
    }
    x = frame.loc[:, XG_V2_FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("xG-v2 numeric features must be finite; use explicit missingness masks")
    y = frame[target].astype(int)
    train_groups = groups.loc[masks["train"]]
    unique_train_groups = int(train_groups.nunique())
    folds = min(settings.inner_folds, unique_train_groups)
    if folds < 2:
        raise ValueError("outer training split needs at least two matches for grouped inner CV")

    candidate_names = ("geometry_logistic", "hist_gradient_boosting")
    inner_scores: dict[str, list[float]] = {name: [] for name in candidate_names}
    group_kfold = GroupKFold(n_splits=folds)
    train_indices = np.flatnonzero(masks["train"].to_numpy())
    for fold_index, (inner_train, inner_valid) in enumerate(
        group_kfold.split(
            x.iloc[train_indices],
            y.iloc[train_indices],
            groups.iloc[train_indices],
        )
    ):
        fit_indices = train_indices[inner_train]
        valid_indices = train_indices[inner_valid]
        if y.iloc[fit_indices].nunique() < 2 or y.iloc[valid_indices].nunique() < 2:
            raise ValueError(f"inner fold {fold_index} does not contain both outcome classes")
        for name in candidate_names:
            model = _new_candidate(name, settings.seed + fold_index)
            model.fit(x.iloc[fit_indices], y.iloc[fit_indices])
            probabilities = model.predict_proba(x.iloc[valid_indices])[:, 1]
            inner_scores[name].append(_brier(y.iloc[valid_indices].to_numpy(), probabilities))

    mean_inner_brier = {
        name: sum(scores) / len(scores) for name, scores in inner_scores.items()
    }
    selected_name = min(candidate_names, key=lambda name: (mean_inner_brier[name], name))
    selected = _new_candidate(selected_name, settings.seed)
    selected.fit(x.loc[masks["train"]], y.loc[masks["train"]])

    raw_calibration = selected.predict_proba(x.loc[masks["calibration"]])[:, 1]
    calibration_y = y.loc[masks["calibration"]]
    calibrator = None
    if calibration_y.nunique() == 2:
        # Platt scaling is linear in the candidate's log-odds, not in its raw
        # probability.  A near-unregularised fit keeps calibration semantics
        # separate from model selection while remaining numerically stable.
        calibrator = LogisticRegression(
            max_iter=2000,
            C=1e6,
            random_state=settings.seed,
        )
        calibrator.fit(_probability_logits(np, raw_calibration), calibration_y)
    calibrated_calibration = (
        calibrator.predict_proba(_probability_logits(np, raw_calibration))[:, 1]
        if calibrator is not None
        else raw_calibration
    )

    raw_test = selected.predict_proba(x.loc[masks["test"]])[:, 1]
    test_probabilities = (
        calibrator.predict_proba(_probability_logits(np, raw_test))[:, 1]
        if calibrator is not None
        else raw_test
    )
    # The paired baseline is a fitted geometry-only logistic model using the
    # same outer train/calibration/test split.  This is deliberately stronger
    # and fairer than the old fixed heuristic curve.
    geometry_baseline = _new_candidate("geometry_logistic", settings.seed)
    geometry_baseline.fit(x.loc[masks["train"]], y.loc[masks["train"]])
    baseline_raw_calibration = geometry_baseline.predict_proba(
        x.loc[masks["calibration"]]
    )[:, 1]
    baseline_calibrator = None
    if calibration_y.nunique() == 2:
        baseline_calibrator = LogisticRegression(
            max_iter=2000,
            C=1e6,
            random_state=settings.seed,
        )
        baseline_calibrator.fit(
            _probability_logits(np, baseline_raw_calibration),
            calibration_y,
        )
    baseline_raw_test = geometry_baseline.predict_proba(x.loc[masks["test"]])[:, 1]
    baseline_probabilities = (
        baseline_calibrator.predict_proba(
            _probability_logits(np, baseline_raw_test)
        )[:, 1]
        if baseline_calibrator is not None
        else baseline_raw_test
    )
    test_outcomes = y.loc[masks["test"]].tolist()
    test_matches = groups.loc[masks["test"]].tolist()
    test_metrics = evaluate_xg_v2(test_probabilities.tolist(), test_outcomes)
    baseline_metrics = evaluate_xg_v2(baseline_probabilities.tolist(), test_outcomes)
    paired_delta = paired_match_cluster_bootstrap_brier_delta(
        test_matches,
        test_outcomes,
        test_probabilities.tolist(),
        baseline_probabilities.tolist(),
        samples=settings.bootstrap_samples,
        seed=settings.seed,
    )
    gates = xg_v2_acceptance_gates(test_metrics, paired_delta)
    calibration_metrics = evaluate_xg_v2(
        calibrated_calibration.tolist(), calibration_y.tolist()
    )
    report: dict[str, object] = {
        "schema_version": "xg-training-v2",
        "task": "xg-v2",
        "status": "research_only" if not gates.passed else "statsbomb_gate_passed_domain_unvalidated",
        "selected_model": selected_name,
        "selection_metric": "mean_grouped_inner_cv_brier",
        "feature_columns": list(XG_V2_FEATURE_COLUMNS),
        "target": target,
        "split": {
            "train_groups": list(split.train_groups),
            "calibration_groups": list(split.validation_groups),
            "test_groups": list(split.test_groups),
        },
        "inner_cv": {
            "folds": folds,
            "brier_by_candidate": inner_scores,
            "mean_brier_by_candidate": mean_inner_brier,
        },
        "calibration": {
            "method": "platt_logit_disjoint_group_fold" if calibrator is not None else "none",
            "input": "logit_probability" if calibrator is not None else "none",
            "metrics": calibration_metrics.to_dict(),
        },
        "test": test_metrics.to_dict(),
        "geometry_logistic_baseline_test": baseline_metrics.to_dict(),
        "paired_brier_delta": paired_delta.to_dict(),
        "acceptance": gates.to_dict(),
        "dataset": {
            "rows": len(frame),
            "matches": int(groups.nunique()),
            "goals": int(y.sum()),
            "split_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        },
        "domain_validation": {
            "footpass_human_dev": "not_run",
            "footpass_frozen_test": "not_run",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": XG_V2_SCHEMA_VERSION,
            "model": selected,
            "calibrator": calibrator,
            "calibration_input": (
                "logit_probability" if calibrator is not None else "none"
            ),
            "geometry_logistic_baseline": geometry_baseline,
            "geometry_logistic_baseline_calibrator": baseline_calibrator,
            "geometry_logistic_columns": list(GEOMETRY_LOGISTIC_COLUMNS),
            "medians": {name: 0.0 for name in XG_V2_FEATURE_COLUMNS},
            "feature_columns": list(XG_V2_FEATURE_COLUMNS),
            "report": report,
        },
        output_path,
    )
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


__all__ = ["XGTrainingV2Config", "train_xg_v2_models"]
