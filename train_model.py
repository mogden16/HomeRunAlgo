"""Time-aware backtest for real MLB home run prediction from batter-game data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FINAL_DATA_PATH, RANDOM_STATE, TRAIN_FRACTION, TSCV_N_SPLITS

DATE_COL = "game_date"
TARGET_COL = "hit_hr"
FEATURE_COLUMNS = [
    "hr_per_pa_last_30d",
    "barrel_rate_last_50_bbe",
    "hard_hit_rate_last_50_bbe",
    "avg_exit_velocity_last_50_bbe",
    "avg_launch_angle_last_50_bbe",
    "fly_ball_rate_last_50_bbe",
    "pitcher_hr9_season_to_date",
    "pitcher_barrel_rate_allowed_last_50_bbe",
    "pitcher_hard_hit_rate_allowed_last_50_bbe",
    "temperature_f",
    "wind_speed_mph",
    "humidity_pct",
    "expected_pa_proxy",
    "days_since_last_game",
    "platoon_advantage",
]
OPTIONAL_SECONDARY_FEATURES = [
    "hr_rate_season_to_date",
    "pull_air_rate_last_50_bbe",
    "batter_k_rate_season_to_date",
    "batter_bb_rate_season_to_date",
    "pitcher_fb_rate_allowed_last_50_bbe",
    "pitcher_k_rate_season_to_date",
    "pitcher_bb_rate_season_to_date",
    "pressure_hpa",
    "wind_direction_deg",
    "recent_form_hr_last_7d",
    "recent_form_barrels_last_14d",
]
BASELINE_FEATURES = [
    "hr_per_pa_last_30d",
    "hr_rate_season_to_date",
    "barrel_rate_last_50_bbe",
]
THRESHOLD_GRID = np.arange(0.05, 0.51, 0.01)
HOLDOUT_SUMMARY_THRESHOLDS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
THRESHOLD_OBJECTIVES = ["f1", "f0.5", "precision_at_min_recall", "balanced_accuracy"]
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "f0.5",
    "balanced_accuracy",
    "positive_prediction_rate",
]
CONSERVATIVE_SORT_COLUMNS = ["precision", "positive_prediction_rate", "threshold"]
CONSERVATIVE_SORT_ASC = [False, True, False]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    missing_columns = [col for col in [DATE_COL, TARGET_COL] + FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return df.sort_values([DATE_COL, "game_pk", "player_id"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, fraction: float = TRAIN_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(DATE_COL).reset_index(drop=True)
    approx_index = min(max(int(len(df_sorted) * fraction), 1), len(df_sorted) - 1)
    cutoff_date = df_sorted.iloc[approx_index][DATE_COL]
    train_df = df_sorted[df_sorted[DATE_COL] < cutoff_date].copy()
    test_df = df_sorted[df_sorted[DATE_COL] >= cutoff_date].copy()
    return train_df, test_df


def validate_temporal_integrity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    max_train_date = train_df[DATE_COL].max()
    min_test_date = test_df[DATE_COL].min()
    if max_train_date >= min_test_date:
        raise ValueError(
            f"Temporal leakage detected: max train date ({max_train_date.date()}) >= min test date ({min_test_date.date()})."
        )
    print(f"[OK] Temporal integrity check passed: max(train_date)={max_train_date.date()} < min(test_date)={min_test_date.date()}")


def run_dataset_sanity_checks(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if df.duplicated(["game_pk", "player_id"]).any():
        raise ValueError("Duplicate batter-game rows found.")
    if df[[DATE_COL, "player_id", TARGET_COL]].isna().any().any():
        raise ValueError("Missing game_date, player_id, or hit_hr values found.")
    validate_temporal_integrity(train_df, test_df)


def available_feature_columns(df: pd.DataFrame) -> list[str]:
    real_features = [column for column in FEATURE_COLUMNS if column in df.columns]
    extra_features = [column for column in OPTIONAL_SECONDARY_FEATURES if column in df.columns]
    return real_features + extra_features


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def tune_logistic_pipeline(X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
    splitter = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)
    grid = GridSearchCV(
        estimator=build_logistic_pipeline(),
        param_grid={"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
        cv=splitter,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        error_score=0.0,
    )
    grid.fit(X_train, y_train)
    return grid


def build_xgboost_pipeline() -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def maybe_calibrate_logistic(
    estimator: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    calibrate_logistic: bool,
    model_name: str,
) -> tuple[Pipeline | CalibratedClassifierCV, str]:
    if model_name != "logistic":
        return estimator, "not_applicable"
    if not calibrate_logistic:
        return estimator, "disabled"

    # Use a time-aware CV wrapper so the sigmoid calibration is learned only from
    # historical training folds, preserving chronology as much as sklearn allows.
    calibrator = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=TSCV_N_SPLITS),
    )
    calibrator.fit(X_train, y_train)
    return calibrator, "enabled"


def get_oof_probabilities_time_series(
    estimator: Pipeline | CalibratedClassifierCV,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = TSCV_N_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate time-ordered OOF probabilities using only past data in each fold."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof_prob = np.full(shape=len(y_train), fill_value=np.nan, dtype=float)

    for fold_number, (fit_idx, valid_idx) in enumerate(splitter.split(X_train), start=1):
        if np.unique(y_train[fit_idx]).size < 2:
            print(
                f"  OOF fold {fold_number}/{n_splits}: skipped because the training window "
                "contains only one class."
            )
            continue

        fold_model = clone(estimator)
        fold_model.fit(X_train[fit_idx], y_train[fit_idx])
        oof_prob[valid_idx] = fold_model.predict_proba(X_train[valid_idx])[:, 1]
        print(
            f"  OOF fold {fold_number}/{n_splits}: train_rows={len(fit_idx):,}, "
            f"valid_rows={len(valid_idx):,}, valid_hr_rate={y_train[valid_idx].mean():.4f}"
        )

    valid_mask = ~np.isnan(oof_prob)
    if not valid_mask.any():
        raise ValueError("No OOF probabilities were generated from TimeSeriesSplit.")
    return y_train[valid_mask], oof_prob[valid_mask]


def safe_probability_metric(metric_fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, y_prob))
    except ValueError:
        return float("nan")


def compute_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f0.5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "positive_prediction_rate": float(np.mean(y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def summarize_thresholds(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray | list[float]) -> pd.DataFrame:
    rows = [compute_threshold_metrics(y_true, y_score, float(threshold)) for threshold in thresholds]
    return pd.DataFrame(rows)


def objective_value(row: pd.Series | dict[str, float], objective: str) -> float:
    if objective in {"f1", "f0.5", "balanced_accuracy"}:
        return float(row[objective])
    if objective == "precision_at_min_recall":
        return float(row["precision"])
    raise ValueError(f"Unsupported threshold objective: {objective}")


def add_objective_score(summary_df: pd.DataFrame, objective: str) -> pd.DataFrame:
    return summary_df.assign(objective_score=summary_df.apply(lambda row: objective_value(row, objective), axis=1))


def conservative_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["objective_score", *CONSERVATIVE_SORT_COLUMNS],
        ascending=[False, *CONSERVATIVE_SORT_ASC],
        kind="mergesort",
    )


def select_threshold_with_tolerance(
    candidate_df: pd.DataFrame,
    objective: str,
    threshold_tolerance: float,
) -> pd.Series:
    ranked_df = conservative_rank(add_objective_score(candidate_df, objective))
    best_score = float(ranked_df.iloc[0]["objective_score"])
    near_best_df = ranked_df[ranked_df["objective_score"] >= best_score - threshold_tolerance].copy()
    return conservative_rank(near_best_df).iloc[0]


def apply_threshold_constraints(summary_df: pd.DataFrame, min_recall: float, max_positive_rate: float) -> pd.DataFrame:
    return summary_df.loc[
        (summary_df["recall"] >= min_recall) & (summary_df["positive_prediction_rate"] <= max_positive_rate)
    ].copy()


def find_best_threshold(
    summary_df: pd.DataFrame,
    objective: str,
    min_recall: float,
    max_positive_rate: float,
    threshold_tolerance: float,
) -> dict[str, float | str | bool | pd.Series]:
    if summary_df.empty:
        raise ValueError("Threshold search failed because no threshold candidates were evaluated.")

    stages: list[tuple[str, float, float]] = [("requested_constraints", min_recall, max_positive_rate)]
    relaxed_positive_rate = max_positive_rate
    while relaxed_positive_rate < 0.25 - 1e-12:
        relaxed_positive_rate = round(min(relaxed_positive_rate + 0.02, 0.25), 4)
        stages.append((f"relaxed_max_positive_rate_to_{relaxed_positive_rate:.2f}", min_recall, relaxed_positive_rate))

    relaxed_min_recall = min_recall
    while relaxed_min_recall > 0.05 + 1e-12:
        relaxed_min_recall = round(max(relaxed_min_recall - 0.02, 0.05), 4)
        stages.append((f"relaxed_min_recall_to_{relaxed_min_recall:.2f}", relaxed_min_recall, relaxed_positive_rate))

    selected_row: pd.Series | None = None
    stage_name = "unconstrained_best"
    used_fallback = False
    constraint_min_recall = min_recall
    constraint_max_positive_rate = max_positive_rate
    for current_stage, stage_min_recall, stage_max_positive_rate in stages:
        candidate_df = apply_threshold_constraints(summary_df, stage_min_recall, stage_max_positive_rate)
        if candidate_df.empty:
            continue
        selected_row = select_threshold_with_tolerance(candidate_df, objective, threshold_tolerance)
        stage_name = current_stage
        used_fallback = current_stage != "requested_constraints"
        constraint_min_recall = stage_min_recall
        constraint_max_positive_rate = stage_max_positive_rate
        break

    warning = ""
    if selected_row is None:
        selected_row = select_threshold_with_tolerance(summary_df, objective, threshold_tolerance)
        used_fallback = True
        warning = (
            "WARNING: No threshold satisfied the requested or staged fallback constraints; "
            "using the unconstrained near-best threshold."
        )

    return {
        "selected_row": selected_row,
        "threshold": float(selected_row["threshold"]),
        "objective_score": float(objective_value(selected_row, objective)),
        "used_fallback": bool(used_fallback),
        "fallback_stage": stage_name,
        "applied_min_recall": float(constraint_min_recall),
        "applied_max_positive_rate": float(constraint_max_positive_rate),
        "threshold_tolerance": float(threshold_tolerance),
        "warning": warning,
    }


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    metrics = compute_threshold_metrics(y_true, y_prob, threshold)
    metrics.update(
        {
            "accuracy": float(accuracy_score(y_true, (y_prob >= threshold).astype(int))),
            "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "roc_auc": safe_probability_metric(roc_auc_score, y_true, y_prob),
            "pr_auc": safe_probability_metric(average_precision_score, y_true, y_prob),
        }
    )
    return metrics


def prediction_rate_ratio(predicted_rate: float, actual_rate: float) -> float:
    if actual_rate <= 0:
        return float("inf") if predicted_rate > 0 else 1.0
    return float(predicted_rate / actual_rate)


def is_operationally_usable(metrics_dict: dict[str, float], actual_rate: float) -> bool:
    return (
        metrics_dict["positive_prediction_rate"] <= 0.20
        and metrics_dict["precision"] > actual_rate
        and metrics_dict["recall"] >= 0.05
    )


def print_prevalence_summary(actual_rate: float, positive_prediction_rate: float, label: str) -> None:
    ratio = prediction_rate_ratio(positive_prediction_rate, actual_rate)
    print(f"{label} actual HR rate              : {actual_rate:.4f}")
    print(f"{label} positive prediction rate   : {positive_prediction_rate:.4f}")
    print(f"{label} prediction/actual rate ratio: {ratio:.4f}")


def print_metric_block(
    metrics: dict[str, float],
    actual_rate: float,
    label: str,
    include_probability_metrics: bool = False,
) -> None:
    print(f"Precision                    : {metrics['precision']:.4f}")
    print(f"Recall                       : {metrics['recall']:.4f}")
    print(f"F1                           : {metrics['f1']:.4f}")
    print(f"F0.5                         : {metrics['f0.5']:.4f}")
    print(f"Balanced Accuracy            : {metrics['balanced_accuracy']:.4f}")
    print(f"Positive prediction rate     : {metrics['positive_prediction_rate']:.4f}")
    print_prevalence_summary(actual_rate, metrics["positive_prediction_rate"], label)
    if include_probability_metrics:
        print(f"Accuracy                     : {metrics['accuracy']:.4f}")
        print(f"Log loss                     : {metrics['log_loss']:.4f}")
        print(f"Brier score                  : {metrics['brier_score']:.4f}")
        print(f"ROC-AUC                      : {metrics['roc_auc']:.4f}")
        print(f"PR-AUC                       : {metrics['pr_auc']:.4f}")


def print_threshold_table(summary_df: pd.DataFrame, title: str, objective: str, limit: int | None = None) -> None:
    display_df = conservative_rank(add_objective_score(summary_df.copy(), objective))
    if limit is not None:
        display_df = display_df.head(limit)

    print(f"\n{title}")
    print("-" * len(title))
    header = (
        f"{'thr':>6} {'obj':>7} {'prec':>7} {'rec':>7} {'f1':>7} {'f0.5':>7} "
        f"{'bal_acc':>8} {'pos_rate':>9} {'tp':>6} {'fp':>6} {'fn':>6} {'tn':>6}"
    )
    print(header)
    for _, row in display_df.iterrows():
        print(
            f"{row['threshold']:6.2f} {row['objective_score']:7.4f} {row['precision']:7.4f} "
            f"{row['recall']:7.4f} {row['f1']:7.4f} {row['f0.5']:7.4f} {row['balanced_accuracy']:8.4f} "
            f"{row['positive_prediction_rate']:9.4f} {int(row['tp']):6d} {int(row['fp']):6d} "
            f"{int(row['fn']):6d} {int(row['tn']):6d}"
        )


def print_confusion_matrix(metrics: dict[str, float]) -> None:
    print("\nConfusion matrix")
    print("-" * 16)
    print(f"TN={metrics['tn']}")
    print(f"FP={metrics['fp']}")
    print(f"FN={metrics['fn']}")
    print(f"TP={metrics['tp']}")


def print_calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> None:
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    print("\nCalibration summary (predicted probability -> actual HR rate):")
    for predicted, observed in zip(mean_pred, fraction_pos):
        print(f"  {predicted:0.3f} -> {observed:0.3f}")


def fit_selected_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    calibrate_logistic: bool,
) -> tuple[Pipeline | CalibratedClassifierCV, str, dict[str, float], str]:
    calibration_status = "not_applicable"
    if model_name == "xgboost":
        model = build_xgboost_pipeline()
        if model is None:
            print("XGBoost is not installed; falling back to logistic regression.")
            grid = tune_logistic_pipeline(X_train, y_train)
            model, calibration_status = maybe_calibrate_logistic(grid.best_estimator_, X_train, y_train, calibrate_logistic, "logistic")
            return model, "logistic", grid.best_params_, calibration_status
        model.fit(X_train, y_train)
        return model, "xgboost", {}, calibration_status

    grid = tune_logistic_pipeline(X_train, y_train)
    model, calibration_status = maybe_calibrate_logistic(
        grid.best_estimator_, X_train, y_train, calibrate_logistic, "logistic"
    )
    return model, "logistic", grid.best_params_, calibration_status


def evaluate_baseline_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_name: str,
    objective: str,
    min_recall: float,
    max_positive_rate: float,
    threshold_tolerance: float,
) -> dict[str, float | str | bool]:
    if feature_name not in train_df.columns or feature_name not in test_df.columns:
        return {"model": feature_name, "warning": "missing_feature_column"}

    train_subset = train_df[[feature_name, TARGET_COL]].dropna()
    test_subset = test_df[[feature_name, TARGET_COL]].dropna()
    if train_subset.empty or test_subset.empty:
        return {"model": feature_name, "warning": "insufficient_non_missing_rows"}
    if train_subset[feature_name].nunique(dropna=True) < 2:
        return {"model": feature_name, "warning": "no_threshold_variation_in_train"}

    train_scores = train_subset[feature_name].to_numpy(dtype=float)
    train_y = train_subset[TARGET_COL].to_numpy()
    test_scores = test_subset[feature_name].to_numpy(dtype=float)
    test_y = test_subset[TARGET_COL].to_numpy()

    baseline_thresholds = np.unique(train_scores[np.isfinite(train_scores)])
    baseline_summary = summarize_thresholds(train_y, train_scores, baseline_thresholds)
    threshold_info = find_best_threshold(
        baseline_summary,
        objective,
        min_recall,
        max_positive_rate,
        threshold_tolerance,
    )
    holdout_metrics = compute_threshold_metrics(test_y, test_scores, float(threshold_info["threshold"]))
    actual_rate = float(test_y.mean())
    operationally_usable = is_operationally_usable(holdout_metrics, actual_rate)
    if holdout_metrics["positive_prediction_rate"] > 0.50:
        operationally_usable = False

    return {
        "model": feature_name,
        "threshold": float(threshold_info["threshold"]),
        "used_fallback": bool(threshold_info["used_fallback"]),
        "fallback_stage": str(threshold_info["fallback_stage"]),
        "objective_score": float(threshold_info["objective_score"]),
        "operationally_usable": "yes" if operationally_usable else "no",
        "not_operationally_usable_reason": "holdout_positive_rate_above_0.50"
        if holdout_metrics["positive_prediction_rate"] > 0.50
        else "",
        **{metric: float(holdout_metrics[metric]) for metric in [*METRIC_COLUMNS, "tp", "fp", "fn", "tn"]},
        "actual_hr_rate": actual_rate,
        "prediction_to_actual_rate_ratio": prediction_rate_ratio(
            holdout_metrics["positive_prediction_rate"], actual_rate
        ),
        "train_rows": int(len(train_subset)),
        "test_rows": int(len(test_subset)),
    }


def print_comparison_table(rows: list[dict[str, float | str]], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    header = (
        f"{'name':<28} {'thr':>6} {'prec':>7} {'rec':>7} {'f1':>7} {'f0.5':>7} {'bal_acc':>8} "
        f"{'pos_rate':>9} {'act_rate':>9} {'ratio':>8} {'usable':>8}"
    )
    print(header)
    for row in rows:
        if "warning" in row:
            print(
                f"{str(row['model']):<28} {'n/a':>6} {'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>7} "
                f"{'n/a':>8} {'n/a':>9} {'n/a':>9} {'n/a':>8} {'no':>8}  ({row['warning']})"
            )
            continue
        print(
            f"{str(row['model']):<28} {row['threshold']:6.2f} {row['precision']:7.4f} {row['recall']:7.4f} "
            f"{row['f1']:7.4f} {row['f0.5']:7.4f} {row['balanced_accuracy']:8.4f} "
            f"{row['positive_prediction_rate']:9.4f} {row['actual_hr_rate']:9.4f} "
            f"{row['prediction_to_actual_rate_ratio']:8.4f} {str(row['operationally_usable']):>8}"
        )


def holdout_commentary(
    model_row: dict[str, float | str],
    baseline_rows: list[dict[str, float | str]],
) -> list[str]:
    actual_rate = float(model_row["actual_hr_rate"])
    pred_rate = float(model_row["positive_prediction_rate"])
    ratio = float(model_row["prediction_to_actual_rate_ratio"])
    overpredicting = ratio > 1.0 + 1e-9
    beat_baselines = True
    evaluated_baselines = [row for row in baseline_rows if "warning" not in row]
    if evaluated_baselines:
        best_baseline = max(evaluated_baselines, key=lambda row: (row["f0.5"], row["precision"]))
        beat_baselines = (float(model_row["f0.5"]) > float(best_baseline["f0.5"])) or (
            np.isclose(float(model_row["f0.5"]), float(best_baseline["f0.5"]))
            and float(model_row["precision"]) > float(best_baseline["precision"])
        )

    return [
        (
            f"Overprediction check: {'yes' if overpredicting else 'no'} "
            f"(predicted positive rate {pred_rate:.4f} vs actual HR rate {actual_rate:.4f}, ratio {ratio:.4f})."
        ),
        f"Operational usability: {model_row['operationally_usable']}.",
        (
            "Full model vs baselines: "
            + (
                "full model beat the evaluated single-feature baselines on the decision-usefulness ranking."
                if beat_baselines
                else "at least one evaluated single-feature baseline matched or beat the full model on F0.5/precision."
            )
        ),
    ]


def run_backtest(
    data_path: str,
    model_name: str = "logistic",
    threshold_objective: str = "f0.5",
    min_recall: float = 0.10,
    max_positive_rate: float = 0.15,
    threshold_tolerance: float = 0.002,
    calibrate_logistic: bool = False,
) -> None:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run python generate_data.py first.")

    df = load_data(data_path)
    train_df, test_df = chronological_split(df)
    run_dataset_sanity_checks(df, train_df, test_df)

    feature_columns = available_feature_columns(df)
    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print("=" * 60)
    print("TIME-AWARE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Rows: total={len(df):,}, train={len(train_df):,}, test={len(test_df):,}")
    print(f"Train date range: {train_df[DATE_COL].min().date()} -> {train_df[DATE_COL].max().date()}")
    print(f"Test date range : {test_df[DATE_COL].min().date()} -> {test_df[DATE_COL].max().date()}")
    print(f"Base HR rate train/test: {train_df[TARGET_COL].mean():.4f} / {test_df[TARGET_COL].mean():.4f}")
    print(f"Features used ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f"Model family requested: {model_name}")
    print(f"Threshold objective   : {threshold_objective}")
    print(f"Threshold constraints : min_recall={min_recall:.2f}, max_positive_rate={max_positive_rate:.2f}")
    print(f"Threshold tolerance   : {threshold_tolerance:.4f}")
    print(f"Logistic calibration  : {'enabled' if calibrate_logistic else 'disabled'}")
    print("\nWhy this workflow is classification-focused:")
    print("- Time-aware validation keeps every fold trained only on past games.")
    print("- Threshold tuning uses training-period OOF probabilities only, so the holdout set stays untouched until the end.")
    print("- This matters for imbalanced HR events because the best yes/no threshold is rarely 0.50.")

    model, resolved_model_name, best_params, calibration_status = fit_selected_model(
        X_train, y_train, model_name, calibrate_logistic
    )
    print(f"\nModel family used: {resolved_model_name}")
    if best_params:
        print(f"Best hyperparameters: {best_params}")
        if "clf__C" in best_params:
            print(f"Best logistic C: {best_params['clf__C']}")
    else:
        print("Best hyperparameters: fixed default configuration")
    print(f"Calibration applied   : {calibration_status}")

    print("\nGenerating training OOF probabilities for threshold tuning...")
    y_train_oof, y_prob_oof = get_oof_probabilities_time_series(model, X_train, y_train)
    train_threshold_summary = summarize_thresholds(y_train_oof, y_prob_oof, THRESHOLD_GRID)
    threshold_info = find_best_threshold(
        train_threshold_summary,
        threshold_objective,
        min_recall,
        max_positive_rate,
        threshold_tolerance,
    )
    threshold_result = threshold_info["selected_row"]
    chosen_threshold = float(threshold_info["threshold"])

    print("\nThreshold tuning summary (training CV only)")
    print("-" * 60)
    print(f"Chosen threshold               : {chosen_threshold:.2f}")
    print(f"Objective used                 : {threshold_objective}")
    print(f"Objective score                : {threshold_info['objective_score']:.4f}")
    print(f"Threshold tolerance            : {threshold_tolerance:.4f}")
    print(f"Fallback used                  : {'yes' if threshold_info['used_fallback'] else 'no'}")
    print(f"Fallback stage                 : {threshold_info['fallback_stage']}")
    print(f"Applied min_recall             : {threshold_info['applied_min_recall']:.2f}")
    print(f"Applied max_positive_rate      : {threshold_info['applied_max_positive_rate']:.2f}")
    print(f"Precision                      : {threshold_result['precision']:.4f}")
    print(f"Recall                         : {threshold_result['recall']:.4f}")
    print(f"F1                             : {threshold_result['f1']:.4f}")
    print(f"F0.5                           : {threshold_result['f0.5']:.4f}")
    print(f"Balanced accuracy              : {threshold_result['balanced_accuracy']:.4f}")
    print(f"Positive prediction rate       : {threshold_result['positive_prediction_rate']:.4f}")
    print_prevalence_summary(float(y_train_oof.mean()), float(threshold_result['positive_prediction_rate']), "Training OOF")
    if threshold_info["warning"]:
        print(str(threshold_info["warning"]))
    else:
        print(
            "Selection rationale             : chose the most conservative near-best threshold within tolerance "
            "after applying the staged constraint search."
        )

    print_threshold_table(
        train_threshold_summary,
        title=f"Top 10 training OOF thresholds ranked by {threshold_objective}",
        objective=threshold_objective,
        limit=10,
    )

    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob_test, chosen_threshold)
    actual_holdout_rate = float(y_test.mean())
    operationally_usable = is_operationally_usable(metrics, actual_holdout_rate)

    print("\nHoldout evaluation")
    print("-" * 60)
    print(f"Threshold used                 : {chosen_threshold:.2f}")
    print_metric_block(metrics, actual_holdout_rate, label="Holdout", include_probability_metrics=True)
    print(f"Operationally usable           : {'yes' if operationally_usable else 'no'}")
    print_confusion_matrix(metrics)
    print_calibration_summary(y_test, y_prob_test)

    holdout_threshold_summary = summarize_thresholds(y_test, y_prob_test, HOLDOUT_SUMMARY_THRESHOLDS)
    print_threshold_table(
        holdout_threshold_summary,
        title="Holdout threshold tradeoff summary (key thresholds)",
        objective=threshold_objective,
        limit=None,
    )

    baseline_rows: list[dict[str, float | str | bool]] = []
    print("\nSingle-feature baseline holdout comparison")
    print("-" * 60)
    for baseline_feature in BASELINE_FEATURES:
        row = evaluate_baseline_feature(
            train_df=train_df,
            test_df=test_df,
            feature_name=baseline_feature,
            objective=threshold_objective,
            min_recall=min_recall,
            max_positive_rate=max_positive_rate,
            threshold_tolerance=threshold_tolerance,
        )
        baseline_rows.append(row)
        if "warning" in row:
            print(f"{baseline_feature}: {row['warning']}")
            continue
        unusable_note = ""
        if row.get("not_operationally_usable_reason"):
            unusable_note = f"; not operationally usable ({row['not_operationally_usable_reason']})"
        print(
            f"{baseline_feature}: threshold={row['threshold']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, f0.5={row['f0.5']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, pos_rate={row['positive_prediction_rate']:.4f}, "
            f"actual_rate={row['actual_hr_rate']:.4f}, ratio={row['prediction_to_actual_rate_ratio']:.4f}, "
            f"fallback_used={'yes' if row['used_fallback'] else 'no'} ({row['fallback_stage']}), "
            f"operationally_usable={row['operationally_usable']}{unusable_note}"
        )

    comparison_rows: list[dict[str, float | str]] = [
        {
            "model": f"{resolved_model_name} full model",
            "threshold": chosen_threshold,
            **{metric: float(metrics[metric]) for metric in METRIC_COLUMNS},
            "actual_hr_rate": actual_holdout_rate,
            "prediction_to_actual_rate_ratio": prediction_rate_ratio(
                metrics["positive_prediction_rate"], actual_holdout_rate
            ),
            "operationally_usable": "yes" if operationally_usable else "no",
        }
    ]
    comparison_rows.extend(baseline_rows)
    comparison_rows_sorted = sorted(
        comparison_rows,
        key=lambda row: (
            0 if row.get("warning") else (1 if row.get("operationally_usable") == "yes" else 0),
            float(row.get("f0.5", -1.0)) if "warning" not in row else -1.0,
            float(row.get("precision", -1.0)) if "warning" not in row else -1.0,
        ),
        reverse=True,
    )
    print_comparison_table(comparison_rows_sorted, title="Model comparison summary (ranked by decision usefulness)")

    print("\nHoldout text summary")
    print("-" * 60)
    for line in holdout_commentary(comparison_rows[0], baseline_rows):
        print(f"- {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", default=str(FINAL_DATA_PATH), help="Path to batter-game dataset CSV.")
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic", help="Model family to train.")
    parser.add_argument(
        "--threshold-objective",
        choices=THRESHOLD_OBJECTIVES,
        default="f0.5",
        help="Objective used to rank threshold candidates.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.10,
        help="Minimum recall required during threshold search before falling back.",
    )
    parser.add_argument(
        "--max-positive-rate",
        type=float,
        default=0.15,
        help="Maximum predicted positive rate allowed during threshold search before falling back.",
    )
    parser.add_argument(
        "--threshold-tolerance",
        type=float,
        default=0.002,
        help="Allow thresholds within this objective tolerance of the best score, then pick the most conservative option.",
    )
    parser.add_argument(
        "--calibrate-logistic",
        action="store_true",
        help="Apply sigmoid probability calibration for logistic models using time-aware CV on training data only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(
        args.data_path,
        model_name=args.model,
        threshold_objective=args.threshold_objective,
        min_recall=args.min_recall,
        max_positive_rate=args.max_positive_rate,
        threshold_tolerance=args.threshold_tolerance,
        calibrate_logistic=args.calibrate_logistic,
    )
