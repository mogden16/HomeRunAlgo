"""Time-aware backtest for real MLB home run prediction from batter-game data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
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


def get_oof_probabilities_time_series(
    estimator: Pipeline,
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


def objective_value(row: pd.Series, objective: str) -> float:
    if objective in {"f1", "f0.5", "balanced_accuracy"}:
        return float(row[objective])
    if objective == "precision_at_min_recall":
        return float(row["precision"])
    raise ValueError(f"Unsupported threshold objective: {objective}")


def find_best_threshold(
    summary_df: pd.DataFrame,
    objective: str,
    min_recall: float,
    max_positive_rate: float,
) -> tuple[pd.Series, bool]:
    if summary_df.empty:
        raise ValueError("Threshold search failed because no threshold candidates were evaluated.")

    constrained_mask = (
        (summary_df["recall"] >= min_recall)
        & (summary_df["positive_prediction_rate"] <= max_positive_rate)
    )
    candidate_df = summary_df.loc[constrained_mask].copy()
    used_fallback = False
    if candidate_df.empty:
        candidate_df = summary_df.copy()
        used_fallback = True

    ranked_df = candidate_df.assign(
        objective_score=lambda df: df.apply(lambda row: objective_value(row, objective), axis=1)
    ).sort_values(
        by=["objective_score", "precision", "positive_prediction_rate", "threshold"],
        ascending=[False, False, True, False],
        kind="mergesort",
    )
    return ranked_df.iloc[0], used_fallback


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


def print_metric_block(metrics: dict[str, float], include_probability_metrics: bool = False) -> None:
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"Recall             : {metrics['recall']:.4f}")
    print(f"F1                 : {metrics['f1']:.4f}")
    print(f"F0.5               : {metrics['f0.5']:.4f}")
    print(f"Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"Positive pred rate : {metrics['positive_prediction_rate']:.4f}")
    if include_probability_metrics:
        print(f"Accuracy           : {metrics['accuracy']:.4f}")
        print(f"Log loss           : {metrics['log_loss']:.4f}")
        print(f"Brier score        : {metrics['brier_score']:.4f}")
        print(f"ROC-AUC            : {metrics['roc_auc']:.4f}")
        print(f"PR-AUC             : {metrics['pr_auc']:.4f}")


def print_threshold_table(summary_df: pd.DataFrame, title: str, objective: str, limit: int | None = None) -> None:
    display_df = summary_df.copy()
    display_df = display_df.assign(
        objective_score=display_df.apply(lambda row: objective_value(row, objective), axis=1)
    ).sort_values(
        by=["objective_score", "precision", "positive_prediction_rate", "threshold"],
        ascending=[False, False, True, False],
        kind="mergesort",
    )
    if limit is not None:
        display_df = display_df.head(limit)

    print(f"\n{title}")
    print("-" * len(title))
    header = (
        f"{'thr':>6} {'obj':>7} {'prec':>7} {'rec':>7} {'f1':>7} "
        f"{'f0.5':>7} {'bal_acc':>8} {'pos_rate':>9}"
    )
    print(header)
    for _, row in display_df.iterrows():
        print(
            f"{row['threshold']:6.2f} {objective_value(row, objective):7.4f} {row['precision']:7.4f} "
            f"{row['recall']:7.4f} {row['f1']:7.4f} {row['f0.5']:7.4f} "
            f"{row['balanced_accuracy']:8.4f} {row['positive_prediction_rate']:9.4f}"
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


def fit_selected_model(X_train: np.ndarray, y_train: np.ndarray, model_name: str) -> tuple[Pipeline, str, dict[str, float]]:
    if model_name == "xgboost":
        model = build_xgboost_pipeline()
        if model is None:
            print("XGBoost is not installed; falling back to logistic regression.")
            grid = tune_logistic_pipeline(X_train, y_train)
            return grid.best_estimator_, "logistic", grid.best_params_
        model.fit(X_train, y_train)
        return model, "xgboost", {}

    grid = tune_logistic_pipeline(X_train, y_train)
    return grid.best_estimator_, "logistic", grid.best_params_


def evaluate_baseline_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_name: str,
    objective: str,
    min_recall: float,
    max_positive_rate: float,
) -> dict[str, float | str]:
    train_subset = train_df[[feature_name, TARGET_COL]].dropna()
    test_subset = test_df[[feature_name, TARGET_COL]].dropna()
    if train_subset.empty or test_subset.empty:
        return {"model": feature_name, "warning": "insufficient_non_missing_rows"}

    train_scores = train_subset[feature_name].to_numpy(dtype=float)
    train_y = train_subset[TARGET_COL].to_numpy()
    test_scores = test_subset[feature_name].to_numpy(dtype=float)
    test_y = test_subset[TARGET_COL].to_numpy()

    baseline_thresholds = np.unique(train_scores[np.isfinite(train_scores)])
    baseline_summary = summarize_thresholds(train_y, train_scores, baseline_thresholds)
    best_row, used_fallback = find_best_threshold(baseline_summary, objective, min_recall, max_positive_rate)
    holdout_metrics = compute_threshold_metrics(test_y, test_scores, float(best_row["threshold"]))
    return {
        "model": feature_name,
        "threshold": float(best_row["threshold"]),
        "used_fallback": bool(used_fallback),
        **{metric: float(holdout_metrics[metric]) for metric in METRIC_COLUMNS},
        "train_rows": int(len(train_subset)),
        "test_rows": int(len(test_subset)),
    }


def print_comparison_table(rows: list[dict[str, float | str]], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'model':<28} {'prec':>7} {'rec':>7} {'f1':>7} {'f0.5':>7} {'bal_acc':>8} {'pos_rate':>9}"
    print(header)
    for row in rows:
        if "warning" in row:
            print(f"{str(row['model']):<28} {'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>8} {'n/a':>9}  ({row['warning']})")
            continue
        print(
            f"{str(row['model']):<28} {row['precision']:7.4f} {row['recall']:7.4f} {row['f1']:7.4f} "
            f"{row['f0.5']:7.4f} {row['balanced_accuracy']:8.4f} {row['positive_prediction_rate']:9.4f}"
        )


def run_backtest(
    data_path: str,
    model_name: str = "logistic",
    threshold_objective: str = "f0.5",
    min_recall: float = 0.15,
    max_positive_rate: float = 0.20,
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
    print("\nWhy this workflow is classification-focused:")
    print("- Time-aware validation keeps every fold trained only on past games.")
    print("- Threshold tuning uses training-period OOF probabilities only, so the holdout set stays untouched until the end.")
    print("- This matters for imbalanced HR events because the best yes/no threshold is rarely 0.50.")

    model, resolved_model_name, best_params = fit_selected_model(X_train, y_train, model_name)
    print(f"\nModel family used: {resolved_model_name}")
    if best_params:
        print(f"Best hyperparameters: {best_params}")
        if "clf__C" in best_params:
            print(f"Best logistic C: {best_params['clf__C']}")
    else:
        print("Best hyperparameters: fixed default configuration")

    print("\nGenerating training OOF probabilities for threshold tuning...")
    y_train_oof, y_prob_oof = get_oof_probabilities_time_series(model, X_train, y_train)
    train_threshold_summary = summarize_thresholds(y_train_oof, y_prob_oof, THRESHOLD_GRID)
    threshold_result, used_fallback = find_best_threshold(
        train_threshold_summary,
        threshold_objective,
        min_recall,
        max_positive_rate,
    )
    chosen_threshold = float(threshold_result["threshold"])

    print("\nThreshold tuning summary (training CV only)")
    print("-" * 60)
    print(f"Chosen threshold         : {chosen_threshold:.2f}")
    print(f"Objective used           : {threshold_objective}")
    print(f"Objective score          : {objective_value(threshold_result, threshold_objective):.4f}")
    print(f"Precision                : {threshold_result['precision']:.4f}")
    print(f"Recall                   : {threshold_result['recall']:.4f}")
    print(f"F1                       : {threshold_result['f1']:.4f}")
    print(f"F0.5                     : {threshold_result['f0.5']:.4f}")
    print(f"Balanced accuracy        : {threshold_result['balanced_accuracy']:.4f}")
    print(f"Positive prediction rate : {threshold_result['positive_prediction_rate']:.4f}")
    if used_fallback:
        print(
            "WARNING: No threshold satisfied the configured min_recall/max_positive_rate constraints; "
            "fell back to the unconstrained best training threshold."
        )

    print_threshold_table(
        train_threshold_summary,
        title=f"Top 10 training OOF thresholds ranked by {threshold_objective}",
        objective=threshold_objective,
        limit=10,
    )

    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob_test, chosen_threshold)

    print("\nHoldout evaluation")
    print("-" * 60)
    print(f"Threshold used     : {chosen_threshold:.2f}")
    print_metric_block(metrics, include_probability_metrics=True)
    print_confusion_matrix(metrics)
    print_calibration_summary(y_test, y_prob_test)

    holdout_threshold_summary = summarize_thresholds(y_test, y_prob_test, HOLDOUT_SUMMARY_THRESHOLDS)
    print_threshold_table(
        holdout_threshold_summary,
        title="Holdout threshold tradeoff summary (key thresholds)",
        objective=threshold_objective,
        limit=None,
    )

    baseline_rows: list[dict[str, float | str]] = []
    print("\nSingle-feature baseline holdout comparison")
    print("-" * 60)
    for baseline_feature in BASELINE_FEATURES:
        if baseline_feature not in df.columns:
            row = {"model": baseline_feature, "warning": "missing_feature"}
        else:
            row = evaluate_baseline_feature(
                train_df=train_df,
                test_df=test_df,
                feature_name=baseline_feature,
                objective=threshold_objective,
                min_recall=min_recall,
                max_positive_rate=max_positive_rate,
            )
        baseline_rows.append(row)
        if "warning" in row:
            print(f"{baseline_feature}: {row['warning']}")
            continue
        fallback_note = " (fallback to unconstrained threshold)" if row.get("used_fallback") else ""
        print(
            f"{baseline_feature}: threshold={row['threshold']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, f0.5={row['f0.5']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, pos_rate={row['positive_prediction_rate']:.4f}"
            f"{fallback_note}"
        )

    comparison_rows: list[dict[str, float | str]] = [
        {
            "model": f"{resolved_model_name} full model",
            **{metric: float(metrics[metric]) for metric in METRIC_COLUMNS},
        }
    ]
    comparison_rows.extend(baseline_rows)
    print_comparison_table(comparison_rows, title="Model comparison summary (holdout set)")


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
        default=0.15,
        help="Minimum recall required during threshold search before falling back.",
    )
    parser.add_argument(
        "--max-positive-rate",
        type=float,
        default=0.20,
        help="Maximum predicted positive rate allowed during threshold search before falling back.",
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
    )
