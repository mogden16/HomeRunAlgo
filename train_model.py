"""Time-aware backtest for real MLB home run prediction from batter-game data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FINAL_DATA_PATH, RANDOM_STATE, TRAIN_FRACTION, TSCV_N_SPLITS

DATE_COL = "game_date"
TARGET_COL = "hit_hr"
MAX_MODEL_FEATURE_MISSINGNESS = 0.50
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


def prune_sparse_features(train_df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    kept: list[str] = []
    for feature_name in feature_columns:
        missing_pct = float(train_df[feature_name].isna().mean())
        keep_for_model = missing_pct <= MAX_MODEL_FEATURE_MISSINGNESS
        reason = "kept: train missingness <= threshold" if keep_for_model else "excluded: train missingness > threshold"
        if keep_for_model:
            kept.append(feature_name)
        rows.append(
            {
                "feature_name": feature_name,
                "train_missing_pct": missing_pct,
                "keep_for_model": "yes" if keep_for_model else "no",
                "reason": reason,
            }
        )
    audit_df = pd.DataFrame(rows).sort_values(["keep_for_model", "train_missing_pct"], ascending=[False, True])
    return kept, audit_df


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
        ]
    )


def tune_logistic_pipeline(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    splitter = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)
    grid = GridSearchCV(
        estimator=build_logistic_pipeline(),
        param_grid={"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0]},
        cv=splitter,
        scoring="neg_log_loss",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"Best logistic C: {grid.best_params_['clf__C']} (CV log loss {-grid.best_score_:.4f})")
    return grid.best_estimator_


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


def print_calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> None:
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    print("\nCalibration summary (predicted probability -> actual HR rate):")
    for predicted, observed in zip(mean_pred, fraction_pos):
        print(f"  {predicted:0.3f} -> {observed:0.3f}")


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    return metrics


def compare_metric_direction(before: float, after: float, tolerance: float = 1e-4) -> str:
    if after > before + tolerance:
        return "improved"
    if after < before - tolerance:
        return "worsened"
    return "stayed similar"


def run_backtest(data_path: str, model_name: str = "logistic") -> None:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run python generate_data.py first.")

    df = load_data(data_path)
    train_df, test_df = chronological_split(df)
    run_dataset_sanity_checks(df, train_df, test_df)

    feature_columns = available_feature_columns(df)
    pruned_feature_columns, pruning_audit_df = prune_sparse_features(train_df, feature_columns)
    if not pruned_feature_columns:
        raise RuntimeError("All candidate features were excluded by missingness threshold.")

    print("\nFeature-pruning audit")
    print("-" * 60)
    print(pruning_audit_df.to_string(index=False))
    excluded_features = pruning_audit_df.loc[pruning_audit_df["keep_for_model"] == "no", "feature_name"].tolist()
    print(f"\nFinal modeled feature count: {len(pruned_feature_columns)}")
    print(f"Excluded features ({len(excluded_features)}): {excluded_features if excluded_features else 'None'}")

    X_train = train_df[pruned_feature_columns].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df[pruned_feature_columns].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print("=" * 60)
    print("TIME-AWARE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Rows: total={len(df):,}, train={len(train_df):,}, test={len(test_df):,}")
    print(f"Train date range: {train_df[DATE_COL].min().date()} -> {train_df[DATE_COL].max().date()}")
    print(f"Test date range : {test_df[DATE_COL].min().date()} -> {test_df[DATE_COL].max().date()}")
    print(f"Base HR rate train/test: {train_df[TARGET_COL].mean():.4f} / {test_df[TARGET_COL].mean():.4f}")
    print(f"Features used ({len(pruned_feature_columns)}): {', '.join(pruned_feature_columns)}")

    if model_name == "xgboost":
        model = build_xgboost_pipeline()
        if model is None:
            print("XGBoost is not installed; falling back to logistic regression.")
            model = tune_logistic_pipeline(X_train, y_train)
        else:
            model.fit(X_train, y_train)
    else:
        model = tune_logistic_pipeline(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob)
    baseline_metrics = metrics
    if excluded_features:
        X_train_unpruned = train_df[feature_columns].to_numpy()
        X_test_unpruned = test_df[feature_columns].to_numpy()
        if model_name == "xgboost":
            baseline_model = build_xgboost_pipeline()
            if baseline_model is None:
                baseline_model = tune_logistic_pipeline(X_train_unpruned, y_train)
            else:
                baseline_model.fit(X_train_unpruned, y_train)
        else:
            baseline_model = tune_logistic_pipeline(X_train_unpruned, y_train)
        baseline_prob = baseline_model.predict_proba(X_test_unpruned)[:, 1]
        baseline_metrics = evaluate_predictions(y_test, baseline_prob)

    print("\nHoldout evaluation")
    print("-" * 60)
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Log loss   : {metrics['log_loss']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC     : {metrics['pr_auc']:.4f}")
    print(
        "Metric direction vs unpruned feature set: "
        f"ROC-AUC {compare_metric_direction(baseline_metrics['roc_auc'], metrics['roc_auc'])}, "
        f"PR-AUC {compare_metric_direction(baseline_metrics['pr_auc'], metrics['pr_auc'])}"
    )
    print_calibration_summary(y_test, y_prob)

    ranking_df = test_df.copy()
    ranking_df["score"] = y_prob
    ranking_df = ranking_df.sort_values("score", ascending=False).reset_index(drop=True)
    batter_name_col = "batter_name" if "batter_name" in ranking_df.columns else "player_name"
    pitcher_name_col = "pitcher_name" if "pitcher_name" in ranking_df.columns else "opp_pitcher_name"
    ranking_df["tier"] = pd.qcut(ranking_df["score"], q=3, labels=["Tier 3", "Tier 2", "Tier 1"], duplicates="drop")
    ranking_preview = ranking_df.rename(
        columns={
            "player_id": "batter_id",
            batter_name_col: "batter_name",
            "opp_pitcher_id": "pitcher_id",
            pitcher_name_col: "pitcher_name",
        }
    )[["batter_id", "batter_name", "pitcher_id", "pitcher_name", "score", "tier"]].head(15)
    print("\nTop-ranked candidates preview")
    print("-" * 60)
    print(ranking_preview.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", default=str(FINAL_DATA_PATH), help="Path to batter-game dataset CSV.")
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic", help="Model family to train.")
    args = parser.parse_args()
    run_backtest(args.data_path, model_name=args.model)
