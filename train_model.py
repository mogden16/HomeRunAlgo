"""Time-aware backtest for real MLB home run prediction from batter-game data."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
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
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FINAL_DATA_PATH, RANDOM_STATE, TRAIN_FRACTION, TSCV_N_SPLITS

DATE_COL = "game_date"
TARGET_COL = "hit_hr"
STABLE_FEATURE_COLUMNS = [
    "hr_rate_season_to_date",
    "hr_per_pa_last_30d",
    "barrel_rate_last_50_bbe",
    "hard_hit_rate_last_50_bbe",
    "avg_launch_angle_last_50_bbe",
    "avg_exit_velocity_last_50_bbe",
    "fly_ball_rate_last_50_bbe",
    "pull_air_rate_last_50_bbe",
    "batter_k_rate_season_to_date",
    "batter_bb_rate_season_to_date",
    "recent_form_hr_last_7d",
    "recent_form_barrels_last_14d",
    "expected_pa_proxy",
    "days_since_last_game",
    "pitch_hand_primary",
    "pitcher_hr9_season_to_date",
    "pitcher_barrel_rate_allowed_last_50_bbe",
    "pitcher_hard_hit_rate_allowed_last_50_bbe",
    "pitcher_fb_rate_allowed_last_50_bbe",
    "pitcher_k_rate_season_to_date",
    "pitcher_bb_rate_season_to_date",
    "starter_or_bullpen_proxy",
    "temperature_f",
    "humidity_pct",
    "wind_speed_mph",
    "wind_direction_deg",
    "pressure_hpa",
    "platoon_advantage",
]
STABLE_AUDIT_COLUMNS = [
    "hr_rate_season_to_date",
    "hr_per_pa_last_30d",
    "barrel_rate_last_50_bbe",
    "hard_hit_rate_last_50_bbe",
    "avg_launch_angle_last_50_bbe",
    "avg_exit_velocity_last_50_bbe",
    "fly_ball_rate_last_50_bbe",
    "pull_air_rate_last_50_bbe",
    "batter_k_rate_season_to_date",
    "batter_bb_rate_season_to_date",
    "recent_form_hr_last_7d",
    "recent_form_barrels_last_14d",
    "expected_pa_proxy",
    "days_since_last_game",
    "opp_pitcher_id",
    "pitch_hand_primary",
    "pitcher_hr9_season_to_date",
    "pitcher_barrel_rate_allowed_last_50_bbe",
    "pitcher_hard_hit_rate_allowed_last_50_bbe",
    "pitcher_fb_rate_allowed_last_50_bbe",
    "pitcher_k_rate_season_to_date",
    "pitcher_bb_rate_season_to_date",
    "starter_or_bullpen_proxy",
    "temperature_f",
    "humidity_pct",
    "wind_speed_mph",
    "wind_direction_deg",
    "pressure_hpa",
    "platoon_advantage",
]
LIVE_PRODUCTION_FEATURE_COLUMNS = [
    "hr_per_pa_last_30d",
    "hr_per_pa_last_10d",
    "barrels_per_pa_last_30d",
    "barrels_per_pa_last_10d",
    "hard_hit_rate_last_30d",
    "hard_hit_rate_last_10d",
    "bbe_95plus_ev_rate_last_30d",
    "bbe_95plus_ev_rate_last_10d",
    "avg_exit_velocity_last_10d",
    "max_exit_velocity_last_10d",
    "pitcher_hr_allowed_per_pa_last_30d",
    "pitcher_barrels_allowed_per_bbe_last_30d",
    "pitcher_hard_hit_allowed_rate_last_30d",
    "pitcher_avg_ev_allowed_last_30d",
    "pitcher_95plus_ev_allowed_rate_last_30d",
    "temperature_f",
    "wind_speed_mph",
    "humidity_pct",
    "platoon_advantage",
]
LIVE_PLUS_FEATURE_COLUMNS = [
    *LIVE_PRODUCTION_FEATURE_COLUMNS,
    "park_factor_hr_vs_batter_hand",
    "batter_hr_per_pa_vs_pitcher_hand",
    "batter_barrels_per_pa_vs_pitcher_hand",
    "pitcher_hr_allowed_per_pa_vs_batter_hand",
    "pitcher_barrels_allowed_per_bbe_vs_batter_hand",
    "split_matchup_hr",
    "split_matchup_barrel",
    "split_matchup_hard_hit",
]
EXPERIMENTAL_FEATURE_COLUMNS = [
    *LIVE_PLUS_FEATURE_COLUMNS,
    "hr_count_last_30d",
    "hr_count_last_10d",
    "pa_last_30d",
    "pa_last_10d",
    "barrels_last_30d",
    "barrels_last_10d",
    "pitcher_hr_allowed_last_30d",
    "pitcher_barrels_allowed_last_30d",
    "pitcher_hard_hit_allowed_last_30d",
    "pitcher_bbe_allowed_last_30d",
    "pitcher_fastball_pct",
    "pitcher_breaking_ball_pct",
    "pitcher_offspeed_pct",
    "batter_hard_hit_rate_vs_fastballs",
    "batter_barrel_rate_vs_fastballs",
    "batter_contact_rate_vs_fastballs",
    "fastball_matchup_hard_hit",
    "fastball_matchup_barrel",
    "park_factor_hr",
    "park_factor_hr_vs_lhb",
    "park_factor_hr_vs_rhb",
    "park_factor_hr_vs_batter_hand",
    "batter_hr_per_pa_vs_rhp",
    "batter_hr_per_pa_vs_lhp",
    "batter_barrels_per_pa_vs_rhp",
    "batter_barrels_per_pa_vs_lhp",
    "batter_hard_hit_rate_vs_rhp",
    "batter_hard_hit_rate_vs_lhp",
    "batter_95plus_ev_rate_vs_rhp",
    "batter_95plus_ev_rate_vs_lhp",
    "batter_hr_per_pa_vs_pitcher_hand",
    "batter_barrels_per_pa_vs_pitcher_hand",
    "batter_hard_hit_rate_vs_pitcher_hand",
    "batter_95plus_ev_rate_vs_pitcher_hand",
    "pitcher_hr_allowed_per_pa_vs_rhb",
    "pitcher_hr_allowed_per_pa_vs_lhb",
    "pitcher_barrels_allowed_per_bbe_vs_rhb",
    "pitcher_barrels_allowed_per_bbe_vs_lhb",
    "pitcher_hard_hit_allowed_rate_vs_rhb",
    "pitcher_hard_hit_allowed_rate_vs_lhb",
    "pitcher_95plus_ev_allowed_rate_vs_rhb",
    "pitcher_95plus_ev_allowed_rate_vs_lhb",
    "pitcher_hr_allowed_per_pa_vs_batter_hand",
    "pitcher_barrels_allowed_per_bbe_vs_batter_hand",
    "pitcher_hard_hit_allowed_rate_vs_batter_hand",
    "pitcher_95plus_ev_allowed_rate_vs_batter_hand",
    "split_matchup_hr",
    "split_matchup_barrel",
    "split_matchup_hard_hit",
    "pitcher_four_seam_pct",
    "pitcher_sinker_pct",
    "pitcher_cutter_pct",
    "pitcher_slider_pct",
    "pitcher_curveball_pct",
    "pitcher_changeup_pct",
    "batter_hard_hit_rate_vs_breaking",
    "batter_barrel_rate_vs_breaking",
    "batter_contact_rate_vs_breaking",
    "batter_hard_hit_rate_vs_offspeed",
    "batter_barrel_rate_vs_offspeed",
    "batter_contact_rate_vs_offspeed",
    "batter_hard_hit_rate_vs_slider",
    "batter_barrel_rate_vs_slider",
    "batter_hard_hit_rate_vs_changeup",
    "batter_barrel_rate_vs_changeup",
    "batter_hard_hit_rate_vs_curveball",
    "batter_barrel_rate_vs_curveball",
    "breaking_matchup_hard_hit",
    "breaking_matchup_barrel",
    "offspeed_matchup_hard_hit",
    "offspeed_matchup_barrel",
    "slider_matchup_barrel",
    "changeup_matchup_barrel",
    "pressure_hpa",
    "wind_direction_deg",
]
FEATURE_COLUMNS = list(STABLE_FEATURE_COLUMNS)
OPTIONAL_SECONDARY_FEATURES: list[str] = []
FEATURE_PROFILE_CHOICES = ["stable", "live", "live_plus", "expanded", "all"]
BASELINE_FEATURES = [
    "hr_per_pa_last_30d",
    "hr_per_pa_last_10d",
    "barrels_per_pa_last_30d",
]
THRESHOLD_GRID = np.arange(0.05, 0.51, 0.01)
HOLDOUT_SUMMARY_THRESHOLDS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
THRESHOLD_OBJECTIVES = ["f1", "f0.5", "precision_at_min_recall", "balanced_accuracy"]
CALIBRATION_CHOICES = ["disabled", "sigmoid", "isotonic"]
MODEL_CHOICES = ["logistic", "histgb", "xgboost", "all", "both"]
SELECTION_METRIC_CHOICES = ["pr_auc", "roc_auc", "neg_log_loss", "neg_brier"]
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "f0.5",
    "balanced_accuracy",
    "positive_prediction_rate",
]
TOP_MISSING_FEATURES_TO_PRINT = 8
TOP_BUCKET_PERCENTS = [0.01, 0.02, 0.05, 0.10, 0.20]
DEFAULT_TOP_CANDIDATES_TO_PRINT = 20
MAX_MODEL_FEATURE_MISSINGNESS = 0.50
DEFAULT_MISSINGNESS_THRESHOLDS = [0.35, 0.50, 0.65]
MODEL_FAMILY_PRIORITY = {"logistic": 0, "histgb": 1, "xgboost": 2}
HISTGB_RANDOM_SEARCH_ITERATIONS = 18
XGBOOST_RANDOM_SEARCH_ITERATIONS = 18
RANKING_EXPORT_CORE_COLUMNS = [
    "game_date",
    "game_pk",
    "batter_id",
    "batter_name",
    "team",
    "opponent_team",
    "pitcher_id",
    "pitcher_name",
]
REASON_TEXT_BY_FEATURE = {
    "hr_per_pa_last_10d": "recent_hr_rate_10d",
    "hr_per_pa_last_30d": "recent_hr_rate_30d",
    "barrels_per_pa_last_10d": "recent_barrel_rate_10d",
    "barrels_per_pa_last_30d": "recent_barrel_rate_30d",
    "hard_hit_rate_last_10d": "recent_hard_hit_rate_10d",
    "hard_hit_rate_last_30d": "recent_hard_hit_rate_30d",
    "bbe_95plus_ev_rate_last_10d": "recent_95plus_ev_rate_10d",
    "bbe_95plus_ev_rate_last_30d": "recent_95plus_ev_rate_30d",
    "avg_exit_velocity_last_10d": "recent_avg_ev_10d",
    "max_exit_velocity_last_10d": "recent_max_ev_10d",
    "pitcher_hr_allowed_per_pa_last_30d": "pitcher_hr_allowed_30d",
    "pitcher_barrels_allowed_per_bbe_last_30d": "pitcher_barrels_allowed_30d",
    "pitcher_hard_hit_allowed_rate_last_30d": "pitcher_hard_hit_allowed_30d",
    "pitcher_avg_ev_allowed_last_30d": "pitcher_avg_ev_allowed_30d",
    "pitcher_95plus_ev_allowed_rate_last_30d": "pitcher_95plus_ev_allowed_30d",
    "temperature_f": "temperature",
    "wind_speed_mph": "wind_speed",
    "humidity_pct": "humidity",
    "platoon_advantage": "platoon_advantage",
    "park_factor_hr_vs_batter_hand": "park_factor_hr_vs_batter_hand",
    "batter_hr_per_pa_vs_pitcher_hand": "batter_hr_per_pa_vs_pitcher_hand",
    "batter_barrels_per_pa_vs_pitcher_hand": "batter_barrels_per_pa_vs_pitcher_hand",
    "pitcher_hr_allowed_per_pa_vs_batter_hand": "pitcher_hr_allowed_per_pa_vs_batter_hand",
    "pitcher_barrels_allowed_per_bbe_vs_batter_hand": "pitcher_barrels_allowed_per_bbe_vs_batter_hand",
    "split_matchup_hr": "split_matchup_hr",
    "split_matchup_barrel": "split_matchup_barrel",
    "split_matchup_hard_hit": "split_matchup_hard_hit",
}

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values:",
    category=UserWarning,
)


def fit_safely_with_imputer_warning_suppressed(estimator, X, y):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values:",
            category=UserWarning,
        )
        estimator.fit(X, y)
    return estimator


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    required_columns = [DATE_COL, TARGET_COL, "game_pk", "player_id"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")
    missing_model_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_model_features:
        print(f"WARNING: dataset is missing configured model features and will train with the remaining available features only: {missing_model_features}")
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


def feature_columns_for_profile(profile: str) -> list[str]:
    if profile == "stable":
        return list(STABLE_FEATURE_COLUMNS)
    if profile == "live":
        return list(LIVE_PRODUCTION_FEATURE_COLUMNS)
    if profile == "live_plus":
        return list(LIVE_PLUS_FEATURE_COLUMNS)
    if profile == "expanded":
        return list(EXPERIMENTAL_FEATURE_COLUMNS)
    raise ValueError(f"Unknown feature profile: {profile}")


def available_feature_columns(df: pd.DataFrame, feature_profile: str = "stable") -> list[str]:
    return [column for column in feature_columns_for_profile(feature_profile) if column in df.columns]


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns].copy()
    binary_maps = {
        "pitch_hand_primary": {"L": 0.0, "R": 1.0},
        "starter_or_bullpen_proxy": {"bullpen_like": 0.0, "starter_like": 1.0},
    }
    for column, value_map in binary_maps.items():
        if column in X.columns:
            X[column] = X[column].map(value_map).astype(float)
    return X


def build_stable_feature_audit(df: pd.DataFrame, modeled_features: list[str]) -> pd.DataFrame:
    modeled_feature_set = set(modeled_features)
    rows: list[dict[str, object]] = []
    for feature_name in STABLE_AUDIT_COLUMNS:
        is_present = feature_name in df.columns
        missing_pct = float(df[feature_name].isna().mean() * 100.0) if is_present else 100.0
        included_in_model = feature_name in modeled_feature_set
        reason = ""
        if not is_present:
            reason = "not present in engineered dataset"
        elif feature_name == "opp_pitcher_id":
            reason = "identifier retained for context/ranked output; not used as numeric model input"
        elif not included_in_model:
            reason = "excluded from modeled feature profile"
        rows.append(
            {
                "feature_name": feature_name,
                "present_in_engineered_dataset": "yes" if is_present else "no",
                "missing_pct": missing_pct,
                "included_in_model": "yes" if included_in_model else "no",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def print_stable_feature_audit(audit_df: pd.DataFrame) -> None:
    print("\nRestored stable feature audit")
    print("-" * 100)
    print(f"{'feature_name':<46} {'present':<8} {'missing_pct':>11} {'included':<10} reason")
    for _, row in audit_df.iterrows():
        print(
            f"{row['feature_name']:<46} {row['present_in_engineered_dataset']:<8} "
            f"{row['missing_pct']:11.2f} {row['included_in_model']:<10} {row['reason']}"
        )


def prune_model_features_by_training_missingness(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = MAX_MODEL_FEATURE_MISSINGNESS,
) -> tuple[list[str], pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    kept: list[str] = []
    excluded: list[str] = []
    for feature in feature_columns:
        missing_pct = float(train_df[feature].isna().mean()) if feature in train_df.columns else 1.0
        keep_for_model = missing_pct <= threshold
        reason = "kept: training missingness within threshold" if keep_for_model else f"excluded: training missingness > {threshold:.0%}"
        rows.append(
            {
                "feature_name": feature,
                "training_missing_pct": missing_pct * 100.0,
                "keep_for_model": "yes" if keep_for_model else "no",
                "reason": reason,
            }
        )
        if keep_for_model:
            kept.append(feature)
        else:
            excluded.append(feature)
    audit_df = pd.DataFrame(rows).sort_values(["keep_for_model", "training_missing_pct", "feature_name"], ascending=[True, False, True])
    print("\nFeature-pruning audit (training missingness)")
    print("-" * 60)
    print(audit_df.to_string(index=False, formatters={"training_missing_pct": lambda v: f"{v:.2f}%"}))
    print(f"Configured MAX_MODEL_FEATURE_MISSINGNESS: {threshold:.0%}")
    print(f"Model features kept: {len(kept)} | excluded: {len(excluded)}")
    if excluded:
        print(f"Excluded feature list: {excluded}")
    return kept, audit_df, excluded


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )


def build_histgb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )


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
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


def get_threshold_classification_scorer():
    return make_scorer(fbeta_score, beta=0.5, zero_division=0)


def get_selection_scorer(selection_metric: str):
    if selection_metric == "pr_auc":
        return make_scorer(average_precision_score, response_method="predict_proba")
    if selection_metric == "roc_auc":
        return make_scorer(roc_auc_score, response_method="predict_proba")
    if selection_metric == "neg_log_loss":
        return make_scorer(log_loss, response_method="predict_proba", greater_is_better=False, labels=[0, 1])
    if selection_metric == "neg_brier":
        return make_scorer(brier_score_loss, response_method="predict_proba", greater_is_better=False)
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def selection_score_from_cv_summary(summary: dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "pr_auc":
        return float(summary["mean_cv_pr_auc"])
    if selection_metric == "roc_auc":
        return float(summary["mean_cv_roc_auc"])
    if selection_metric == "neg_log_loss":
        return -float(summary["mean_cv_log_loss"])
    if selection_metric == "neg_brier":
        return -float(summary["mean_cv_brier_score"])
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def resolve_model_families(model_name: str) -> list[str]:
    if model_name == "all":
        return ["logistic", "histgb", "xgboost"]
    if model_name == "both":
        return ["logistic", "xgboost"]
    return [model_name]


def resolve_feature_profiles(feature_profile: str, compare_against: str | None = None) -> list[str]:
    if feature_profile == "all":
        return ["stable", "live", "live_plus", "expanded"]
    profiles = [feature_profile]
    if compare_against and compare_against not in {"none", feature_profile, "all"}:
        profiles.append(compare_against)
    return profiles


def resolve_missingness_thresholds(missingness_threshold: float | None) -> list[float]:
    if missingness_threshold is None:
        return list(DEFAULT_MISSINGNESS_THRESHOLDS)
    return [float(missingness_threshold)]


def tune_logistic_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    selection_metric: str = "pr_auc",
) -> GridSearchCV:
    splitter = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)
    grid = GridSearchCV(
        estimator=build_logistic_pipeline(),
        param_grid=[
            {
                "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "clf__solver": ["lbfgs"],
                "clf__class_weight": [None, "balanced"],
            },
            {
                "clf__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                "clf__solver": ["saga"],
                "clf__l1_ratio": [1.0],
                "clf__class_weight": [None, "balanced"],
            },
            {
                "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                "clf__solver": ["saga"],
                "clf__l1_ratio": [0.1, 0.5, 0.9],
                "clf__class_weight": [None, "balanced"],
            },
        ],
        cv=splitter,
        scoring=get_selection_scorer(selection_metric),
        n_jobs=-1,
        refit=True,
        error_score=0.0,
    )
    fit_safely_with_imputer_warning_suppressed(grid, X_train, y_train)
    return grid


def tune_histgb_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    selection_metric: str = "pr_auc",
) -> RandomizedSearchCV:
    splitter = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)
    search = RandomizedSearchCV(
        estimator=build_histgb_pipeline(),
        param_distributions={
            "clf__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
            "clf__max_iter": [100, 150, 200, 300, 400],
            "clf__max_depth": [None, 3, 5, 7],
            "clf__max_leaf_nodes": [15, 31, 63, 127],
            "clf__min_samples_leaf": [10, 20, 50, 100],
            "clf__l2_regularization": [0.0, 0.01, 0.1, 1.0],
        },
        n_iter=HISTGB_RANDOM_SEARCH_ITERATIONS,
        cv=splitter,
        scoring=get_selection_scorer(selection_metric),
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        error_score=0.0,
    )
    fit_safely_with_imputer_warning_suppressed(search, X_train, y_train)
    return search


def tune_xgboost_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    selection_metric: str = "pr_auc",
) -> RandomizedSearchCV | None:
    pipeline = build_xgboost_pipeline()
    if pipeline is None:
        return None
    splitter = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "clf__n_estimators": [100, 150, 200, 300, 400, 500],
            "clf__max_depth": [2, 3, 4, 5, 6],
            "clf__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
            "clf__min_child_weight": [1, 2, 3, 5, 8],
            "clf__subsample": [0.65, 0.8, 1.0],
            "clf__colsample_bytree": [0.65, 0.8, 1.0],
            "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "clf__gamma": [0.0, 0.1, 0.3, 0.5],
        },
        n_iter=XGBOOST_RANDOM_SEARCH_ITERATIONS,
        cv=splitter,
        scoring=get_selection_scorer(selection_metric),
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        error_score=0.0,
    )
    fit_safely_with_imputer_warning_suppressed(search, X_train, y_train)
    return search


def calibration_cv_supported(y_train: np.ndarray, n_splits: int) -> tuple[bool, str]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_number, (fit_idx, _) in enumerate(splitter.split(np.arange(len(y_train))), start=1):
        unique_classes = np.unique(y_train[fit_idx])
        if unique_classes.size < 2:
            return False, f"fold {fold_number} training window has only one class"
    return True, ""


def missingness_percentages(X_df: pd.DataFrame) -> pd.Series:
    return X_df.isna().mean().mul(100.0)


def fully_missing_feature_names(X_df: pd.DataFrame) -> list[str]:
    return [str(column) for column in X_df.columns[X_df.isna().all()]]


def fold_missingness_records(X_train_df: pd.DataFrame, n_splits: int = TSCV_N_SPLITS) -> list[dict[str, object]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    records: list[dict[str, object]] = []
    for fold_number, (fit_idx, valid_idx) in enumerate(splitter.split(X_train_df), start=1):
        fit_df = X_train_df.iloc[fit_idx]
        missing_pct = missingness_percentages(fit_df).sort_values(ascending=False)
        all_missing = fully_missing_feature_names(fit_df)
        records.append(
            {
                "fold": fold_number,
                "train_rows": len(fit_idx),
                "valid_rows": len(valid_idx),
                "all_missing_features": all_missing,
                "missing_pct": missing_pct,
                "all_missing_count": len(all_missing),
                "imputer_skips_features": len(all_missing) > 0,
            }
        )
    return records


def print_missingness_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    fold_records: list[dict[str, object]],
) -> pd.DataFrame:
    fold_missing_features = {
        feature
        for record in fold_records
        for feature in record["all_missing_features"]
    }
    summary_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "train_missing_pct": train_df[feature_columns].isna().mean().mul(100.0).reindex(feature_columns).values,
            "test_missing_pct": test_df[feature_columns].isna().mean().mul(100.0).reindex(feature_columns).values,
            "ever_fully_missing_in_train_fold": [feature in fold_missing_features for feature in feature_columns],
        }
    ).sort_values(
        by=["train_missing_pct", "test_missing_pct", "ever_fully_missing_in_train_fold", "feature"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )

    print("\nDataset-level missingness summary (selected model features)")
    print("-" * 60)
    print(f"{'feature':<36} {'train_%':>9} {'test_%':>9} {'full_miss_any_fold':>18}")
    for _, row in summary_df.iterrows():
        print(
            f"{row['feature']:<36} {row['train_missing_pct']:9.2f} {row['test_missing_pct']:9.2f} "
            f"{str(bool(row['ever_fully_missing_in_train_fold'])):>18}"
        )
    return summary_df


def print_fold_missingness_diagnostics(fold_records: list[dict[str, object]]) -> list[str]:
    print("\nTraining OOF fold missing-feature diagnostics")
    print("-" * 60)
    fully_missing_union: list[str] = []
    for record in fold_records:
        all_missing = list(record["all_missing_features"])
        fully_missing_union.extend(all_missing)
        top_missing = record["missing_pct"].head(TOP_MISSING_FEATURES_TO_PRINT)
        top_missing_text = ", ".join(f"{feature}={pct:.1f}%" for feature, pct in top_missing.items())
        all_missing_text = ", ".join(all_missing) if all_missing else "none"
        print(
            f"Fold {record['fold']}: train_rows={record['train_rows']:,}, valid_rows={record['valid_rows']:,}, "
            f"100%-missing features={record['all_missing_count']}"
        )
        print(f"  Fully missing feature names : {all_missing_text}")
        print(f"  Worst missingness in train  : {top_missing_text}")
        print(
            "  Imputer behavior            : "
            + (
                "SimpleImputer(strategy='median') will skip/drop these fully missing columns in this fold's transformed matrix."
                if record["imputer_skips_features"]
                else "No features are fully missing in this fold, so median imputation can operate on every feature."
            )
        )
    unique_fully_missing = sorted(set(fully_missing_union))
    if unique_fully_missing:
        print(
            "\nFeatures that are fully missing in at least one training fold: "
            + ", ".join(unique_fully_missing)
        )
    else:
        print("\nNo selected model feature is fully missing in any TimeSeriesSplit training fold.")
    return unique_fully_missing


def maybe_calibrate_logistic(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    calibration_mode: str,
    model_name: str,
) -> tuple[Pipeline | CalibratedClassifierCV, dict[str, str]]:
    status = {
        "requested": calibration_mode,
        "used": "not_applicable" if model_name != "logistic" else "disabled",
        "status": "not_applicable" if model_name != "logistic" else "skipped",
        "message": "Calibration only applies to logistic models." if model_name != "logistic" else "Calibration disabled by request.",
    }
    if model_name != "logistic":
        return estimator, status
    if calibration_mode == "disabled":
        return estimator, status

    supported, reason = calibration_cv_supported(y_train, TSCV_N_SPLITS)
    if not supported:
        status.update(
            {
                "used": "disabled",
                "status": "skipped",
                "message": f"Calibration skipped to avoid leakage/instability because {reason}.",
            }
        )
        return estimator, status

    methods_to_try = [calibration_mode]
    if calibration_mode == "isotonic":
        methods_to_try.append("sigmoid")

    # Safest reasonable implementation: calibrated CV refits the tuned estimator
    # inside each chronological split using only earlier training rows, then learns
    # the calibrator on that fold's validation rows. This preserves time ordering
    # and avoids holdout leakage, even though it is not a custom rolling calibrator.
    last_error = ""
    for method in methods_to_try:
        try:
            calibrator = CalibratedClassifierCV(
                estimator=clone(estimator),
                method=method,
                cv=TimeSeriesSplit(n_splits=TSCV_N_SPLITS),
            )
            fit_safely_with_imputer_warning_suppressed(calibrator, X_train, y_train)
            if method == calibration_mode:
                status.update(
                    {
                        "used": method,
                        "status": "applied",
                        "message": f"Calibration applied with {method} using time-aware CV on training data only.",
                    }
                )
            else:
                status.update(
                    {
                        "used": method,
                        "status": "downgraded",
                        "message": (
                            f"Requested {calibration_mode} calibration, but downgraded to {method} for stability after: {last_error}"
                        ),
                    }
                )
            return calibrator, status
        except ValueError as exc:
            last_error = str(exc)
            continue

    status.update(
        {
            "used": "disabled",
            "status": "skipped",
            "message": f"Calibration skipped after failures: {last_error}",
        }
    )
    return estimator, status


def get_oof_probabilities_time_series(
    estimator: Pipeline | CalibratedClassifierCV,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_columns: list[str],
    n_splits: int = TSCV_N_SPLITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate time-ordered OOF probabilities using only past data in each fold."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof_prob = np.full(shape=len(y_train), fill_value=np.nan, dtype=float)

    for fold_number, (fit_idx, valid_idx) in enumerate(splitter.split(X_train), start=1):
        fit_df = X_train.iloc[fit_idx]
        valid_df = X_train.iloc[valid_idx]
        if np.unique(y_train[fit_idx]).size < 2:
            print(
                f"  OOF fold {fold_number}/{n_splits}: skipped because the training window contains only one class."
            )
            continue

        all_missing = fully_missing_feature_names(fit_df)
        if all_missing:
            feature_index_map = ", ".join(
                f"{feature_columns.index(feature)}={feature}" for feature in all_missing
            )
            print(
                f"  OOF fold {fold_number}/{n_splits}: fully-missing training features -> {feature_index_map}. "
                "Median imputation will drop these columns for this fold."
            )

        fold_model = clone(estimator)
        fit_safely_with_imputer_warning_suppressed(fold_model, fit_df, y_train[fit_idx])
        oof_prob[valid_idx] = fold_model.predict_proba(valid_df)[:, 1]
        print(
            f"  OOF fold {fold_number}/{n_splits}: train_rows={len(fit_idx):,}, "
            f"valid_rows={len(valid_idx):,}, valid_hr_rate={y_train[valid_idx].mean():.4f}"
        )

    valid_mask = ~np.isnan(oof_prob)
    if not valid_mask.any():
        raise ValueError("No OOF probabilities were generated from TimeSeriesSplit.")
    return y_train[valid_mask], oof_prob[valid_mask]


def cross_validate_probability_metrics_time_series(
    estimator: Pipeline | CalibratedClassifierCV,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_columns: list[str],
    n_splits: int = TSCV_N_SPLITS,
) -> dict[str, Any]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof_prob = np.full(shape=len(y_train), fill_value=np.nan, dtype=float)
    fold_metrics: list[dict[str, float | int]] = []

    for fold_number, (fit_idx, valid_idx) in enumerate(splitter.split(X_train), start=1):
        fit_df = X_train.iloc[fit_idx]
        valid_df = X_train.iloc[valid_idx]
        y_fit = y_train[fit_idx]
        y_valid = y_train[valid_idx]
        if np.unique(y_fit).size < 2:
            continue

        all_missing = fully_missing_feature_names(fit_df)
        if all_missing:
            feature_index_map = ", ".join(
                f"{feature_columns.index(feature)}={feature}" for feature in all_missing
            )
            print(
                f"  CV fold {fold_number}/{n_splits}: fully-missing training features -> {feature_index_map}. "
                "Median imputation will drop these columns for this fold."
            )

        fold_model = clone(estimator)
        fit_safely_with_imputer_warning_suppressed(fold_model, fit_df, y_fit)
        fold_prob = fold_model.predict_proba(valid_df)[:, 1]
        oof_prob[valid_idx] = fold_prob
        fold_metrics.append(
            {
                "fold": int(fold_number),
                "train_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "valid_hr_rate": float(y_valid.mean()),
                "pr_auc": safe_probability_metric(average_precision_score, y_valid, fold_prob),
                "roc_auc": safe_probability_metric(roc_auc_score, y_valid, fold_prob),
                "log_loss": float(log_loss(y_valid, fold_prob, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y_valid, fold_prob)),
            }
        )

    valid_mask = ~np.isnan(oof_prob)
    if not valid_mask.any():
        raise ValueError("No OOF probabilities were generated from TimeSeriesSplit.")

    fold_df = pd.DataFrame(fold_metrics)
    return {
        "fold_metrics": fold_metrics,
        "oof_y_true": y_train[valid_mask],
        "oof_y_prob": oof_prob[valid_mask],
        "mean_cv_pr_auc": float(fold_df["pr_auc"].mean()) if not fold_df.empty else float("nan"),
        "mean_cv_roc_auc": float(fold_df["roc_auc"].mean()) if not fold_df.empty else float("nan"),
        "mean_cv_log_loss": float(fold_df["log_loss"].mean()) if not fold_df.empty else float("nan"),
        "mean_cv_brier_score": float(fold_df["brier_score"].mean()) if not fold_df.empty else float("nan"),
    }


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


def select_threshold_with_tolerance(
    candidate_df: pd.DataFrame,
    objective: str,
    threshold_tolerance: float,
) -> pd.Series:
    scored_df = add_objective_score(candidate_df.copy(), objective)
    best_score = float(scored_df["objective_score"].max())
    near_best_df = scored_df[scored_df["objective_score"] >= best_score - threshold_tolerance].copy()
    ranked_df = near_best_df.sort_values(
        by=["precision", "positive_prediction_rate", "threshold"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    return ranked_df.iloc[0]


def apply_threshold_constraints(summary_df: pd.DataFrame, min_recall: float, max_positive_rate: float) -> pd.DataFrame:
    return summary_df.loc[
        (summary_df["recall"] >= min_recall) & (summary_df["positive_prediction_rate"] <= max_positive_rate)
    ].copy()


def select_best_by_metric(
    summary_df: pd.DataFrame,
    metric: str,
    min_recall: float,
    max_positive_rate: float,
) -> tuple[pd.Series, bool]:
    constrained_df = apply_threshold_constraints(summary_df, min_recall, max_positive_rate)
    source_df = constrained_df if not constrained_df.empty else summary_df
    used_fallback = constrained_df.empty
    ranked_df = source_df.sort_values(
        by=[metric, "precision", "positive_prediction_rate", "threshold"],
        ascending=[False, False, True, False],
        kind="mergesort",
    )
    return ranked_df.iloc[0], used_fallback


def select_best_precision_subject_to_recall_floor(
    summary_df: pd.DataFrame,
    min_recall: float,
    max_positive_rate: float,
) -> tuple[pd.Series, str]:
    constrained_df = apply_threshold_constraints(summary_df, min_recall, max_positive_rate)
    candidate_df = constrained_df
    fallback_reason = ""
    if candidate_df.empty:
        candidate_df = summary_df.loc[summary_df["recall"] >= min_recall].copy()
        fallback_reason = (
            "WARNING: No thresholds met recall and positive-rate constraints; "
            "falling back to thresholds meeting recall only."
        )
    if candidate_df.empty:
        candidate_df = summary_df
        fallback_reason = (
            "WARNING: No thresholds met recall floor; falling back to the full OOF threshold table."
        )

    ranked_df = candidate_df.sort_values(
        by=["precision", "f0.5", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return ranked_df.iloc[0], fallback_reason


def get_recommended_threshold_band(
    summary_df: pd.DataFrame,
    min_recall: float,
    max_positive_rate: float,
    f0_5_tolerance: float = 0.005,
) -> pd.DataFrame:
    constrained_df = apply_threshold_constraints(summary_df, min_recall, max_positive_rate)
    if constrained_df.empty:
        return constrained_df
    best_f0_5 = float(constrained_df["f0.5"].max())
    return constrained_df.loc[constrained_df["f0.5"] >= best_f0_5 - f0_5_tolerance].copy()


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


def summarize_top_bucket_lift(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_percents: list[float] | None = None,
) -> pd.DataFrame:
    if top_percents is None:
        top_percents = TOP_BUCKET_PERCENTS
    ranked = pd.DataFrame({"actual": y_true, "score": y_prob}).sort_values("score", ascending=False).reset_index(drop=True)
    baseline_rate = float(ranked["actual"].mean())
    total_hr = int(ranked["actual"].sum())
    rows: list[dict[str, float | int | str]] = []
    for pct in top_percents:
        n_rows = max(1, int(np.ceil(len(ranked) * pct)))
        bucket = ranked.head(n_rows)
        hr_rate = float(bucket["actual"].mean()) if n_rows > 0 else 0.0
        cumulative_hrs = int(bucket["actual"].sum())
        rows.append(
            {
                "bucket": f"top_{int(pct * 100)}%",
                "top_percent": float(pct),
                "rows": int(n_rows),
                "bucket_hr_rate": hr_rate,
                "baseline_hr_rate": baseline_rate,
                "lift_vs_baseline": (hr_rate / baseline_rate) if baseline_rate > 0 else float("nan"),
                "cumulative_hrs_captured": cumulative_hrs,
                "cumulative_hr_capture_pct": (cumulative_hrs / total_hr) if total_hr > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_prediction_deciles(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    ranked = pd.DataFrame({"actual": y_true, "score": y_prob}).sort_values("score", ascending=False).reset_index(drop=True)
    ranked["bucket"] = pd.qcut(
        ranked["score"].rank(method="first", ascending=False),
        q=min(bins, len(ranked)),
        labels=False,
        duplicates="drop",
    )
    grouped = (
        ranked.groupby("bucket", as_index=False)
        .agg(rows=("actual", "size"), avg_predicted_score=("score", "mean"), actual_hr_rate=("actual", "mean"), hrs=("actual", "sum"))
        .sort_values("bucket")
    )
    total_hrs = float(grouped["hrs"].sum())
    grouped["bucket"] = grouped["bucket"].astype(int) + 1
    grouped["bucket_label"] = grouped["bucket"].apply(lambda x: f"{bins - x + 1}/{bins} (high-to-low)")
    grouped["cumulative_hrs"] = grouped["hrs"].cumsum()
    grouped["cumulative_hr_capture_pct"] = grouped["cumulative_hrs"] / total_hrs if total_hrs > 0 else np.nan
    return grouped


def assign_confidence_tiers(predicted_percentile: pd.Series) -> pd.Series:
    """
    Assign adaptive tiers using percentile ranks within the current slate:
    A=top 1%, B=next 4%, C=next 15%, D=remaining candidates.
    """
    return np.select(
        [predicted_percentile >= 0.99, predicted_percentile >= 0.95, predicted_percentile >= 0.80],
        ["A", "B", "C"],
        default="D",
    )


def extract_logistic_coefficient_map(
    fitted_model: Pipeline | CalibratedClassifierCV,
    feature_columns: list[str],
) -> dict[str, float]:
    pipeline: Pipeline | None = None
    if isinstance(fitted_model, Pipeline):
        pipeline = fitted_model
    elif isinstance(fitted_model, CalibratedClassifierCV) and hasattr(fitted_model, "calibrated_classifiers_"):
        if fitted_model.calibrated_classifiers_:
            pipeline = fitted_model.calibrated_classifiers_[0].estimator
    if pipeline is None or not hasattr(pipeline.named_steps.get("clf"), "coef_"):
        return {}

    imputer = pipeline.named_steps.get("imputer")
    clf = pipeline.named_steps.get("clf")
    if imputer is None or clf is None:
        return {}
    coef = clf.coef_.ravel()
    stats = getattr(imputer, "statistics_", None)
    if stats is None:
        return {}
    kept_columns = [feature for feature, stat in zip(feature_columns, stats, strict=False) if not np.isnan(stat)]
    if len(kept_columns) != len(coef):
        return {}
    return {feature: float(value) for feature, value in zip(kept_columns, coef, strict=False)}


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_mph(value: float) -> str:
    return f"{value:.1f} mph"


def _format_temp(value: float) -> str:
    return f"{value:.0f}F"


def _strength_label(percentile: float) -> str:
    if percentile >= 0.90:
        return "top-decile"
    if percentile >= 0.75:
        return "top-quartile"
    return "above-baseline"


def _feature_reason_bucket(feature: str) -> str:
    if feature.startswith("pitcher_"):
        return "pitcher"
    if feature in {
        "temperature_f",
        "wind_speed_mph",
        "humidity_pct",
        "platoon_advantage",
        "park_factor_hr_vs_batter_hand",
    }:
        return "context"
    return "batter"


def _build_feature_reason(feature: str, value: float, percentile: float, row: pd.Series) -> str:
    pitcher_name = str(row.get("pitcher_name") or "the opposing pitcher")
    strength = _strength_label(percentile)
    if feature == "hr_per_pa_last_10d":
        return f"HR rate over the last 10 days is {_format_pct(value)}, a {strength} live-model signal."
    if feature == "hr_per_pa_last_30d":
        return f"HR rate over the last 30 days is {_format_pct(value)}, which keeps the recent power trend elevated."
    if feature == "barrels_per_pa_last_10d":
        return f"Barrel rate over the last 10 days is {_format_pct(value)}, a {strength} indicator of current lift-and-damage contact."
    if feature == "barrels_per_pa_last_30d":
        return f"Barrel rate over the last 30 days is {_format_pct(value)}, which the model treats as sustained power quality."
    if feature == "hard_hit_rate_last_10d":
        return f"Hard-hit rate over the last 10 days is {_format_pct(value)}, a {strength} recent-contact signal."
    if feature == "hard_hit_rate_last_30d":
        return f"Hard-hit rate over the last 30 days is {_format_pct(value)}, showing consistent quality contact."
    if feature == "bbe_95plus_ev_rate_last_10d":
        return f"95+ EV contact rate over the last 10 days is {_format_pct(value)}, which supports near-term HR upside."
    if feature == "bbe_95plus_ev_rate_last_30d":
        return f"95+ EV contact rate over the last 30 days is {_format_pct(value)}, a strong loud-contact input for the model."
    if feature == "avg_exit_velocity_last_10d":
        return f"Average exit velocity over the last 10 days is {value:.1f} mph, a {strength} recent batted-ball quality marker."
    if feature == "max_exit_velocity_last_10d":
        return f"Peak exit velocity over the last 10 days is {value:.1f} mph, which points to top-end HR juice."
    if feature == "pitcher_hr_allowed_per_pa_last_30d":
        return f"{pitcher_name} has allowed homers in {_format_pct(value)} of plate appearances over the last 30 days."
    if feature == "pitcher_barrels_allowed_per_bbe_last_30d":
        return f"{pitcher_name} has allowed barrels on {_format_pct(value)} of recent batted balls."
    if feature == "pitcher_hard_hit_allowed_rate_last_30d":
        return f"{pitcher_name} has allowed a {_format_pct(value)} hard-hit rate over the last 30 days."
    if feature == "pitcher_avg_ev_allowed_last_30d":
        return f"{pitcher_name} has allowed {value:.1f} mph average exit velocity recently, which the model flags as hittable contact."
    if feature == "pitcher_95plus_ev_allowed_rate_last_30d":
        return f"{pitcher_name} has allowed {_format_pct(value)} 95+ EV contact over the last 30 days."
    if feature == "temperature_f":
        return f"Forecast temperature is {_format_temp(value)}, a positive hitting-context input in the live model."
    if feature == "wind_speed_mph":
        return f"Projected wind speed is {_format_mph(value)}, which the live model treats as favorable weather context."
    if feature == "humidity_pct":
        return f"Projected humidity is {value:.0f}%, a modest positive weather input in the live model."
    if feature == "platoon_advantage":
        return "The hitter has the platoon advantage in this matchup."
    if feature == "park_factor_hr_vs_batter_hand":
        return f"Park factor for the hitter's handedness is {value:.1f}, which adds venue lift to the matchup."
    if feature == "batter_hr_per_pa_vs_pitcher_hand":
        return f"Against this pitcher handedness, the hitter owns a {_format_pct(value)} HR-per-PA split."
    if feature == "batter_barrels_per_pa_vs_pitcher_hand":
        return f"Against this pitcher handedness, the hitter has a {_format_pct(value)} barrel-per-PA split."
    if feature == "pitcher_hr_allowed_per_pa_vs_batter_hand":
        return f"{pitcher_name} has allowed homers in {_format_pct(value)} of plate appearances to this batter side."
    if feature == "pitcher_barrels_allowed_per_bbe_vs_batter_hand":
        return f"{pitcher_name} has allowed barrels on {_format_pct(value)} of batted balls to this batter side."
    if feature == "split_matchup_hr":
        return f"The handedness-specific HR matchup score is {value:.4f}, combining the hitter split and pitcher split."
    if feature == "split_matchup_barrel":
        return f"The handedness-specific barrel matchup score is {value:.4f}, reinforcing the power-contact fit."
    if feature == "split_matchup_hard_hit":
        return f"The handedness-specific hard-hit matchup score is {value:.4f}, supporting loud-contact upside."
    return ""


def generate_reason_strings(
    row: pd.Series,
    reference_df: pd.DataFrame,
    positive_coef_map: dict[str, float],
    max_reasons: int = 3,
) -> list[str]:
    contributions: list[dict[str, object]] = []
    for feature in REASON_TEXT_BY_FEATURE:
        if feature not in row.index or feature not in reference_df.columns:
            continue
        value = row.get(feature)
        if pd.isna(value):
            continue
        series = reference_df[feature].dropna()
        if series.empty:
            continue
        q60 = float(series.quantile(0.60))
        q75 = float(series.quantile(0.75))
        q90 = float(series.quantile(0.90))
        q85 = float(series.quantile(0.85))
        coef_weight = max(positive_coef_map.get(feature, 0.0), 0.0)
        if coef_weight <= 0:
            continue
        bucket = _feature_reason_bucket(feature)
        is_platoon = feature == "platoon_advantage"
        percentile = float((series <= value).mean()) if not is_platoon else 1.0
        if bucket == "context":
            strong_signal = bool(value >= q90) or (is_platoon and value >= 1)
            medium_signal = bool(value >= q85) or (is_platoon and value >= 1)
            supporting_signal = False
        else:
            strong_signal = bool(value >= q90) or (is_platoon and value >= 1)
            medium_signal = bool(value >= q75) or (is_platoon and value >= 1)
            supporting_signal = bool(value >= q60) or (is_platoon and value >= 1)
        reason = _build_feature_reason(feature, float(value), percentile, row)
        if not reason:
            continue
        score = None
        if strong_signal:
            score = 2.0 + coef_weight + percentile
        elif medium_signal:
            score = 1.0 + coef_weight + percentile
        elif supporting_signal:
            score = 0.5 + coef_weight + percentile
        if score is None:
            continue
        bucket_priority = {"batter": 3.0, "pitcher": 2.5, "context": 1.0}.get(bucket, 0.0)
        contributions.append(
            {
                "feature": feature,
                "bucket": bucket,
                "score": float(score) + bucket_priority,
                "percentile": percentile,
                "reason": reason,
            }
        )
    if not contributions:
        return ["The live model favors the overall recent-form and matchup profile, but no single feature cleared the explanation threshold."][:max_reasons]
    ordered = sorted(contributions, key=lambda item: float(item["score"]), reverse=True)
    selected: list[str] = []
    seen_reasons: set[str] = set()
    for bucket in ["batter", "pitcher", "context"]:
        for item in ordered:
            if item["bucket"] != bucket:
                continue
            reason = str(item["reason"])
            if reason in seen_reasons:
                continue
            selected.append(reason)
            seen_reasons.add(reason)
            break
        if len(selected) >= max_reasons:
            return selected[:max_reasons]
    for item in ordered:
        reason = str(item["reason"])
        if reason in seen_reasons:
            continue
        selected.append(reason)
        seen_reasons.add(reason)
        if len(selected) >= max_reasons:
            break
    return selected[:max_reasons]


def build_ranked_predictions_output(
    source_df: pd.DataFrame,
    y_prob: np.ndarray,
    reference_df: pd.DataFrame,
    fitted_model: Pipeline | CalibratedClassifierCV,
    feature_columns: list[str],
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    ranked = source_df.copy().reset_index(drop=True)
    identity_columns = [col for col in ["batter_id", "batter_name", "pitcher_id", "pitcher_name", "opp_pitcher_name"] if col in ranked.columns]
    print(f"Ranked-output identity columns available: {identity_columns}")
    ranked["predicted_hr_probability"] = y_prob
    ranked["predicted_hr_percentile"] = ranked["predicted_hr_probability"].rank(method="first", pct=True)
    ranked["predicted_hr_score"] = (ranked["predicted_hr_percentile"] * 100).round(1)
    ranked["rank"] = ranked["predicted_hr_probability"].rank(method="first", ascending=False).astype(int)
    ranked["confidence_tier"] = assign_confidence_tiers(ranked["predicted_hr_percentile"])
    if y_true is not None:
        ranked["actual_hit_hr"] = y_true

    coef_map = extract_logistic_coefficient_map(fitted_model, feature_columns)
    positive_coef_map = {k: v for k, v in coef_map.items() if v > 0}
    reasons = ranked.apply(
        lambda row: generate_reason_strings(row, reference_df=reference_df, positive_coef_map=positive_coef_map, max_reasons=3),
        axis=1,
    )
    ranked["top_reason_1"] = reasons.apply(lambda items: items[0] if len(items) > 0 else "")
    ranked["top_reason_2"] = reasons.apply(lambda items: items[1] if len(items) > 1 else "")
    ranked["top_reason_3"] = reasons.apply(lambda items: items[2] if len(items) > 2 else "")

    rename_candidates = {
        "player_id": "batter_id",
        "player_name": "batter_name",
        "opp_pitcher_id": "pitcher_id",
        "opp_pitcher_name": "pitcher_name",
        "batting_team": "team",
        "home_team": "team",
        "away_team": "opponent_team",
        "opposing_team": "opponent_team",
        "opponent": "opponent_team",
    }
    ranked = ranked.rename(columns={src: dst for src, dst in rename_candidates.items() if src in ranked.columns and dst not in ranked.columns})
    ranked = validate_ranked_output_identity(ranked)
    dedupe_keys = [col for col in ["game_pk", "batter_id"] if col in ranked.columns]
    if dedupe_keys:
        before = len(ranked)
        ranked = ranked.drop_duplicates(subset=dedupe_keys, keep="first").copy()
        dropped = before - len(ranked)
        if dropped > 0:
            print(f"WARNING: dropped {dropped} duplicate ranked rows on keys {dedupe_keys}.")
    export_columns = [col for col in RANKING_EXPORT_CORE_COLUMNS if col in ranked.columns]
    export_columns += [
        "predicted_hr_probability",
        "predicted_hr_score",
        "predicted_hr_percentile",
        "rank",
        "confidence_tier",
    ]
    if "actual_hit_hr" in ranked.columns:
        export_columns.append("actual_hit_hr")
    export_columns += ["top_reason_1", "top_reason_2", "top_reason_3"]
    return ranked.sort_values("rank").reset_index(drop=True)[export_columns]


def validate_ranked_output_identity(ranked: pd.DataFrame) -> pd.DataFrame:
    required = ["batter_id", "batter_name"]
    missing = [col for col in required if col not in ranked.columns]
    if missing:
        raise ValueError(f"Ranked output missing hitter identity columns: {missing}")

    batter_null = int(ranked["batter_id"].isna().sum())
    if batter_null > 0:
        raise ValueError(f"Ranked output has {batter_null} rows with missing batter_id.")
    name_null = int(ranked["batter_name"].isna().sum())
    if name_null > 0:
        raise ValueError(f"Ranked output has {name_null} rows with missing batter_name.")
    if "player_name" in ranked.columns:
        disagree = int((ranked["batter_name"].fillna("") != ranked["player_name"].fillna("")).sum())
        print(f"ranked_output_identity_check(batter_name vs player_name disagreements): {disagree:,}")
    candidate_identity_from_hitter = "batter_name" in ranked.columns and "batter_id" in ranked.columns
    print(f"ranked_output_candidate_fields_use_hitter_identity: {candidate_identity_from_hitter}")

    if "pitcher_name" in ranked.columns:
        compare = ranked[["batter_name", "pitcher_name"]].dropna()
        mismatch_share = float((compare["batter_name"] == compare["pitcher_name"]).mean()) if len(compare) else 0.0
        print(f"ranked_output_identity_mismatch_share(batter_name==pitcher_name): {mismatch_share:.3f}")
        display_cols = [col for col in ["batter_id", "batter_name", "pitcher_id", "pitcher_name", "predicted_hr_score"] if col in ranked.columns]
        print("ranked_output_identity_preview_top20:")
        print(ranked[display_cols].head(20).to_string(index=False))
        if mismatch_share > 0.25:
            raise ValueError("Ranked output identity risk: batter_name overlaps pitcher_name too frequently in top rows.")
    if "pitcher_id" in ranked.columns:
        id_overlap = float((ranked["batter_id"] == ranked["pitcher_id"]).mean())
        print(f"ranked_output_identity_id_overlap_share(batter_id==pitcher_id): {id_overlap:.3f}")
        if id_overlap > 0.10:
            raise ValueError("Ranked output identity risk: displayed candidate ids appear to come from pitcher_id.")
    return ranked


def print_top_bucket_lift_table(lift_df: pd.DataFrame) -> None:
    print("\nHoldout ranking quality: top-bucket lift")
    print("-" * 60)
    print(
        f"{'bucket':<8} {'rows':>7} {'bucket_hr':>10} {'baseline':>10} {'lift':>8} {'cum_HR':>8} {'cum_capture':>12}"
    )
    for _, row in lift_df.iterrows():
        print(
            f"{row['bucket']:<8} {int(row['rows']):7d} {row['bucket_hr_rate']:10.4f} {row['baseline_hr_rate']:10.4f} "
            f"{row['lift_vs_baseline']:8.2f} {int(row['cumulative_hrs_captured']):8d} {row['cumulative_hr_capture_pct']:12.4f}"
        )


def print_prediction_bucket_summary(decile_df: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print("-" * 60)
    print(f"{'bucket':<14} {'rows':>7} {'avg_score':>10} {'actual_hr':>10} {'cum_capture':>12}")
    for _, row in decile_df.iterrows():
        print(
            f"{row['bucket_label']:<14} {int(row['rows']):7d} {row['avg_predicted_score']:10.4f} "
            f"{row['actual_hr_rate']:10.4f} {row['cumulative_hr_capture_pct']:12.4f}"
        )


def print_top_candidates_summary(ranked_df: pd.DataFrame, top_n: int = DEFAULT_TOP_CANDIDATES_TO_PRINT) -> None:
    print(f"\nTop {top_n} ranked HR candidates (ranking tool output)")
    print("-" * 60)
    for _, row in ranked_df.head(top_n).iterrows():
        batter_id = row.get("batter_id", "unknown")
        name = row["batter_name"] if "batter_name" in ranked_df.columns else batter_id
        pitcher_name = row.get("pitcher_name", "unknown")
        actual_txt = f" | actual_hr={int(row['actual_hit_hr'])}" if "actual_hit_hr" in ranked_df.columns else ""
        print(
            f"{int(row['rank'])}. batter_id={batter_id} | batter_name={name} | pitcher_name={pitcher_name} | score={row['predicted_hr_score']:.1f} | "
            f"tier={row['confidence_tier']}{actual_txt}"
        )
        reasons = [row.get("top_reason_1", ""), row.get("top_reason_2", ""), row.get("top_reason_3", "")]
        for reason in [reason for reason in reasons if reason]:
            print(f"   - {reason}")


def prediction_rate_ratio(predicted_rate: float, actual_rate: float) -> float:
    if actual_rate <= 0:
        return float("inf") if predicted_rate > 0 else 1.0
    return float(predicted_rate / actual_rate)


def is_operationally_usable(metrics_dict: dict[str, float], actual_rate: float) -> bool:
    return (
        metrics_dict["precision"] >= 0.18
        and metrics_dict["positive_prediction_rate"] <= (1.25 * actual_rate)
        and metrics_dict["recall"] >= 0.10
    )


def usability_reason(metrics_dict: dict[str, float], actual_rate: float) -> str:
    failures: list[str] = []
    if metrics_dict["precision"] < 0.18:
        failures.append(f"precision {metrics_dict['precision']:.4f} < 0.18")
    if metrics_dict["positive_prediction_rate"] > 1.25 * actual_rate:
        failures.append(
            f"positive_prediction_rate {metrics_dict['positive_prediction_rate']:.4f} > 1.25 * holdout_hr_rate {(1.25 * actual_rate):.4f}"
        )
    if metrics_dict["recall"] < 0.10:
        failures.append(f"recall {metrics_dict['recall']:.4f} < 0.10")
    return "; ".join(failures) if failures else "passes all operational usability checks"


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
    display_df = add_objective_score(summary_df.copy(), objective).sort_values(
        by=["objective_score", "precision", "positive_prediction_rate", "threshold"],
        ascending=[False, False, True, False],
        kind="mergesort",
    )
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


def print_oof_threshold_diagnostics(
    train_threshold_summary: pd.DataFrame,
    min_recall: float,
    max_positive_rate: float,
) -> None:
    best_f0_5_row, f0_5_used_fallback = select_best_by_metric(
        train_threshold_summary, metric="f0.5", min_recall=min_recall, max_positive_rate=max_positive_rate
    )
    best_bal_acc_row, bal_acc_used_fallback = select_best_by_metric(
        train_threshold_summary, metric="balanced_accuracy", min_recall=min_recall, max_positive_rate=max_positive_rate
    )
    best_prec_recall_row, precision_fallback_warning = select_best_precision_subject_to_recall_floor(
        train_threshold_summary, min_recall=min_recall, max_positive_rate=max_positive_rate
    )
    recommended_band = get_recommended_threshold_band(
        train_threshold_summary,
        min_recall=min_recall,
        max_positive_rate=max_positive_rate,
        f0_5_tolerance=0.005,
    )

    constrained_df = apply_threshold_constraints(train_threshold_summary, min_recall, max_positive_rate)
    top5_source_df = constrained_df
    top5_warning = ""
    if top5_source_df.empty:
        top5_source_df = train_threshold_summary
        top5_warning = "WARNING: No thresholds met constraints; showing top 5 from full OOF threshold table."
    top5_df = top5_source_df.sort_values(
        by=["f0.5", "precision"],
        ascending=[False, False],
        kind="mergesort",
    ).head(5)

    print("\nOOF threshold diagnostics")
    print("-" * 60)
    if f0_5_used_fallback:
        print("WARNING: Best OOF F0.5 used fallback to full threshold table (no constrained candidates).")
    print(
        "Best threshold by OOF F0.5                 : "
        f"thr={best_f0_5_row['threshold']:.4f}, f0.5={best_f0_5_row['f0.5']:.4f}, "
        f"precision={best_f0_5_row['precision']:.4f}, recall={best_f0_5_row['recall']:.4f}, "
        f"pos_rate={best_f0_5_row['positive_prediction_rate']:.4f}"
    )
    if bal_acc_used_fallback:
        print("WARNING: Best OOF balanced accuracy used fallback to full threshold table (no constrained candidates).")
    print(
        "Best threshold by OOF balanced accuracy    : "
        f"thr={best_bal_acc_row['threshold']:.4f}, bal_acc={best_bal_acc_row['balanced_accuracy']:.4f}, "
        f"precision={best_bal_acc_row['precision']:.4f}, recall={best_bal_acc_row['recall']:.4f}, "
        f"pos_rate={best_bal_acc_row['positive_prediction_rate']:.4f}"
    )
    if precision_fallback_warning:
        print(precision_fallback_warning)
    print(
        "Best threshold by precision @ recall floor : "
        f"thr={best_prec_recall_row['threshold']:.4f}, precision={best_prec_recall_row['precision']:.4f}, "
        f"recall={best_prec_recall_row['recall']:.4f}, f0.5={best_prec_recall_row['f0.5']:.4f}, "
        f"pos_rate={best_prec_recall_row['positive_prediction_rate']:.4f}"
    )

    print(
        "Recommended threshold band                 : "
        "thresholds within 0.005 of best constrained OOF F0.5"
    )
    if recommended_band.empty:
        print("  No recommended band found under constraints.")
    else:
        print(
            f"  lower={recommended_band['threshold'].min():.4f}, "
            f"upper={recommended_band['threshold'].max():.4f}, "
            f"count={len(recommended_band)}"
        )

    if top5_warning:
        print(top5_warning)
    print("Top 5 OOF thresholds under constraints")
    header = (
        f"{'threshold':>9} {'precision':>9} {'recall':>8} {'f1':>7} {'f0.5':>7} "
        f"{'bal_acc':>8} {'pos_rate':>9}"
    )
    print(header)
    for _, row in top5_df.iterrows():
        print(
            f"{row['threshold']:9.4f} {row['precision']:9.4f} {row['recall']:8.4f} "
            f"{row['f1']:7.4f} {row['f0.5']:7.4f} {row['balanced_accuracy']:8.4f} "
            f"{row['positive_prediction_rate']:9.4f}"
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


def tune_model_family(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    selection_metric: str = "pr_auc",
) -> GridSearchCV | RandomizedSearchCV | None:
    if model_name == "logistic":
        return tune_logistic_pipeline(X_train, y_train, selection_metric=selection_metric)
    if model_name == "histgb":
        return tune_histgb_pipeline(X_train, y_train, selection_metric=selection_metric)
    if model_name == "xgboost":
        return tune_xgboost_pipeline(X_train, y_train, selection_metric=selection_metric)
    raise ValueError(f"Unsupported model family: {model_name}")


def evaluate_training_candidate(
    train_df: pd.DataFrame,
    *,
    feature_profile: str,
    model_name: str,
    missingness_threshold: float,
    selection_metric: str = "pr_auc",
) -> dict[str, Any] | None:
    initial_feature_columns = available_feature_columns(train_df, feature_profile=feature_profile)
    feature_columns, pruning_audit_df, excluded_for_missingness = prune_model_features_by_training_missingness(
        train_df,
        initial_feature_columns,
        threshold=missingness_threshold,
    )
    if not feature_columns:
        return None

    X_train = prepare_feature_matrix(train_df, feature_columns)
    y_train = train_df[TARGET_COL].to_numpy()
    search = tune_model_family(X_train, y_train, model_name, selection_metric=selection_metric)
    if search is None:
        return None

    cv_summary = cross_validate_probability_metrics_time_series(
        search.best_estimator_,
        X_train,
        y_train,
        feature_columns=feature_columns,
    )
    summary_row = {
        "model_family": model_name,
        "feature_profile": feature_profile,
        "missingness_threshold": float(missingness_threshold),
        "feature_count": int(len(feature_columns)),
        "excluded_feature_count": int(len(excluded_for_missingness)),
        "selection_metric": selection_metric,
        "selection_score": float(selection_score_from_cv_summary(cv_summary, selection_metric)),
        "mean_cv_pr_auc": float(cv_summary["mean_cv_pr_auc"]),
        "mean_cv_roc_auc": float(cv_summary["mean_cv_roc_auc"]),
        "mean_cv_log_loss": float(cv_summary["mean_cv_log_loss"]),
        "mean_cv_brier_score": float(cv_summary["mean_cv_brier_score"]),
        "search_best_score": float(search.best_score_) if np.isfinite(search.best_score_) else float("nan"),
        "model_priority": int(MODEL_FAMILY_PRIORITY[model_name]),
        "best_params": dict(search.best_params_),
    }
    return {
        "summary_row": summary_row,
        "feature_columns": feature_columns,
        "excluded_features": excluded_for_missingness,
        "pruning_audit_df": pruning_audit_df,
        "search": search,
        "cv_summary": cv_summary,
    }


def rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["selection_score"]),
            float(row["mean_cv_log_loss"]),
            float(row["mean_cv_brier_score"]),
            int(row["model_priority"]),
        ),
    )


def print_candidate_search_summary(candidate_rows: list[dict[str, Any]]) -> None:
    print("\nTraining-only candidate search leaderboard")
    print("-" * 96)
    header = (
        f"{'model':<10} {'profile':<9} {'miss':>6} {'feat':>5} {'pr_auc':>8} "
        f"{'roc_auc':>8} {'logloss':>9} {'brier':>8}"
    )
    print(header)
    for row in candidate_rows:
        print(
            f"{row['model_family']:<10} {row['feature_profile']:<9} {row['missingness_threshold']:6.2f} "
            f"{int(row['feature_count']):5d} {row['mean_cv_pr_auc']:8.4f} "
            f"{row['mean_cv_roc_auc']:8.4f} {row['mean_cv_log_loss']:9.4f} "
            f"{row['mean_cv_brier_score']:8.4f}"
        )


def search_training_candidates(
    train_df: pd.DataFrame,
    *,
    model_name: str,
    feature_profile: str,
    compare_against: str | None,
    missingness_threshold: float | None,
    selection_metric: str = "pr_auc",
) -> list[dict[str, Any]]:
    candidate_results: list[dict[str, Any]] = []
    for requested_profile in resolve_feature_profiles(feature_profile, compare_against):
        for requested_model in resolve_model_families(model_name):
            for requested_threshold in resolve_missingness_thresholds(missingness_threshold):
                print("\n" + "=" * 60)
                print(
                    "TRAINING SEARCH CANDIDATE: "
                    f"model={requested_model}, profile={requested_profile}, missingness={requested_threshold:.2f}"
                )
                print("=" * 60)
                candidate_result = evaluate_training_candidate(
                    train_df,
                    feature_profile=requested_profile,
                    model_name=requested_model,
                    missingness_threshold=requested_threshold,
                    selection_metric=selection_metric,
                )
                if candidate_result is None:
                    print("Candidate skipped: no usable features or model unavailable.")
                    continue
                candidate_results.append(candidate_result)
    return candidate_results


def choose_logistic_calibration(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_columns: list[str],
    requested_mode: str,
) -> tuple[Pipeline | CalibratedClassifierCV, dict[str, str], list[dict[str, Any]]]:
    mode_candidates = ["disabled"] if requested_mode == "disabled" else ["disabled", "sigmoid", "isotonic"]
    evaluations: list[dict[str, Any]] = []

    for mode in mode_candidates:
        if mode == "disabled":
            model = clone(estimator)
            fit_safely_with_imputer_warning_suppressed(model, X_train, y_train)
            status = {
                "requested": requested_mode,
                "used": "disabled",
                "status": "skipped",
                "message": "Calibration disabled for final model fit.",
            }
        else:
            model, status = maybe_calibrate_logistic(clone(estimator), X_train, y_train, mode, "logistic")

        cv_summary = cross_validate_probability_metrics_time_series(
            model,
            X_train,
            y_train,
            feature_columns=feature_columns,
        )
        evaluations.append(
            {
                "mode": mode,
                "status": status,
                "model": model,
                "mean_cv_pr_auc": float(cv_summary["mean_cv_pr_auc"]),
                "mean_cv_roc_auc": float(cv_summary["mean_cv_roc_auc"]),
                "mean_cv_log_loss": float(cv_summary["mean_cv_log_loss"]),
                "mean_cv_brier_score": float(cv_summary["mean_cv_brier_score"]),
            }
        )

    baseline_pr_auc = next(item["mean_cv_pr_auc"] for item in evaluations if item["mode"] == "disabled")
    viable = [item for item in evaluations if item["mean_cv_pr_auc"] >= baseline_pr_auc - 1e-12]
    chosen = max(
        viable,
        key=lambda item: (
            float(item["mean_cv_pr_auc"]),
            -float(item["mean_cv_log_loss"]),
            -float(item["mean_cv_brier_score"]),
        ),
    )
    chosen_status = dict(chosen["status"])
    chosen_status["requested"] = requested_mode
    if chosen["mode"] == "disabled" and requested_mode != "disabled":
        chosen_status["message"] = (
            "Calibration search kept disabled because sigmoid/isotonic did not improve training CV PR-AUC."
        )
    return chosen["model"], chosen_status, evaluations


def fit_selected_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    calibration_mode: str,
) -> tuple[Pipeline | CalibratedClassifierCV | None, str, dict[str, object], dict[str, str]]:
    calibration_status = {
        "requested": calibration_mode,
        "used": "not_applicable",
        "status": "not_applicable",
        "message": "Calibration not applicable.",
    }
    search = tune_model_family(X_train, y_train, model_name, selection_metric="pr_auc")
    if search is None:
        calibration_status.update(
            {
                "requested": calibration_mode,
                "used": "not_applicable",
                "status": "skipped",
                "message": f"{model_name} is unavailable in this environment.",
            }
        )
        return None, model_name, {}, calibration_status

    if model_name != "logistic":
        calibration_status.update(
            {
                "requested": calibration_mode,
                "used": "not_applicable",
                "status": "not_applicable",
                "message": f"Calibration skipped because {model_name} was used.",
            }
        )
        return search.best_estimator_, model_name, dict(search.best_params_), calibration_status

    model, calibration_status, _ = choose_logistic_calibration(
        search.best_estimator_,
        X_train,
        y_train,
        list(X_train.columns),
        calibration_mode,
    )
    return model, "logistic", dict(search.best_params_), calibration_status


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

    return {
        "model": feature_name,
        "threshold": float(threshold_info["threshold"]),
        "used_fallback": bool(threshold_info["used_fallback"]),
        "fallback_stage": str(threshold_info["fallback_stage"]),
        "objective_score": float(threshold_info["objective_score"]),
        "operationally_usable": "yes" if operationally_usable else "no",
        "operational_usability_reason": usability_reason(holdout_metrics, actual_rate),
        **{metric: float(holdout_metrics[metric]) for metric in [*METRIC_COLUMNS, "tp", "fp", "fn", "tn"]},
        "actual_hr_rate": actual_rate,
        "prediction_to_actual_rate_ratio": prediction_rate_ratio(
            holdout_metrics["positive_prediction_rate"], actual_rate
        ),
        "train_rows": int(len(train_subset)),
        "test_rows": int(len(test_subset)),
    }


def print_baseline_results(rows: list[dict[str, float | str | bool]]) -> None:
    header = (
        f"{'baseline':<28} {'thr':>8} {'prec':>7} {'rec':>7} {'f1':>7} {'f0.5':>7} {'bal_acc':>8} {'pos_rate':>9} {'usable':>8}"
    )
    print(header)
    for row in rows:
        if "warning" in row:
            print(f"{str(row['model']):<28} {'n/a':>8} {'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>8} {'n/a':>9} {'no':>8}  ({row['warning']})")
            continue
        print(
            f"{str(row['model']):<28} {row['threshold']:8.4f} {row['precision']:7.4f} {row['recall']:7.4f} "
            f"{row['f1']:7.4f} {row['f0.5']:7.4f} {row['balanced_accuracy']:8.4f} {row['positive_prediction_rate']:9.4f} {str(row['operationally_usable']):>8}"
        )


def print_comparison_table(rows: list[dict[str, float | str]], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    header = (
        f"{'name':<28} {'thr':>8} {'prec':>7} {'rec':>7} {'f1':>7} {'f0.5':>7} {'bal_acc':>8} {'pos_rate':>9} {'usable':>8}"
    )
    print(header)
    for row in rows:
        if "warning" in row:
            print(
                f"{str(row['model']):<28} {'n/a':>8} {'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>7} {'n/a':>8} {'n/a':>9} {'no':>8}  ({row['warning']})"
            )
            continue
        print(
            f"{str(row['model']):<28} {row['threshold']:8.4f} {row['precision']:7.4f} {row['recall']:7.4f} "
            f"{row['f1']:7.4f} {row['f0.5']:7.4f} {row['balanced_accuracy']:8.4f} {row['positive_prediction_rate']:9.4f} {str(row['operationally_usable']):>8}"
        )


def print_model_family_comparison(rows: list[dict[str, float | str]]) -> None:
    print("\nModel-family comparison summary")
    print("-" * 60)
    header = (
        f"{'model_family':<14} {'thr':>8} {'prec':>7} {'rec':>7} {'f1':>7} {'f0.5':>7} {'bal_acc':>8} "
        f"{'pos_rate':>9} {'roc_auc':>8} {'pr_auc':>8} {'usable':>8}"
    )
    print(header)
    for row in rows:
        print(
            f"{str(row['model_family']):<14} {row['threshold']:8.4f} {row['precision']:7.4f} {row['recall']:7.4f} "
            f"{row['f1']:7.4f} {row['f0.5']:7.4f} {row['balanced_accuracy']:8.4f} {row['positive_prediction_rate']:9.4f} "
            f"{row['roc_auc']:8.4f} {row['pr_auc']:8.4f} {str(row['operationally_usable']):>8}"
        )


def best_evaluated_baseline(baseline_rows: list[dict[str, float | str]]) -> dict[str, float | str] | None:
    evaluated_baselines = [row for row in baseline_rows if "warning" not in row]
    if not evaluated_baselines:
        return None
    return max(evaluated_baselines, key=lambda row: (float(row["f0.5"]), float(row["precision"])))


def holdout_commentary(model_row: dict[str, float | str], baseline_rows: list[dict[str, float | str]]) -> list[str]:
    actual_rate = float(model_row["actual_hr_rate"])
    pred_rate = float(model_row["positive_prediction_rate"])
    ratio = float(model_row["prediction_to_actual_rate_ratio"])
    overpredicting = pred_rate > (1.25 * actual_rate)
    best_baseline = best_evaluated_baseline(baseline_rows)
    if best_baseline is None:
        baseline_line = "Full model vs baselines: no baseline could be fully evaluated."
    else:
        beat_baselines = (float(model_row["f0.5"]) > float(best_baseline["f0.5"])) or (
            np.isclose(float(model_row["f0.5"]), float(best_baseline["f0.5"]))
            and float(model_row["precision"]) > float(best_baseline["precision"])
        )
        baseline_line = (
            f"Full model vs baselines: {'beats' if beat_baselines else 'does not beat'} the best single-feature baseline ({best_baseline['model']}) on F0.5/precision."
        )

    return [
        (
            f"Overprediction check: {'yes' if overpredicting else 'no'} "
            f"(predicted positive rate {pred_rate:.4f} vs operational cap {1.25 * actual_rate:.4f}; actual HR rate {actual_rate:.4f}, ratio {ratio:.4f})."
        ),
        f"Operational usability: {model_row['operationally_usable']} ({model_row['operational_usability_reason']}).",
        baseline_line,
    ]


def compare_feature_profile_rows(
    primary_row: dict[str, float | str],
    comparison_row: dict[str, float | str],
    *,
    primary_profile: str,
    comparison_profile: str,
) -> str:
    primary_f05 = float(primary_row["f0.5"])
    comparison_f05 = float(comparison_row["f0.5"])
    primary_precision = float(primary_row["precision"])
    comparison_precision = float(comparison_row["precision"])
    if primary_f05 > comparison_f05 + 1e-6:
        verdict = "improved"
    elif comparison_f05 > primary_f05 + 1e-6:
        verdict = "worsened"
    elif primary_precision > comparison_precision + 1e-6:
        verdict = "improved"
    elif comparison_precision > primary_precision + 1e-6:
        verdict = "worsened"
    else:
        verdict = "stayed similar"
    return (
        f"{primary_profile} vs {comparison_profile}: {verdict} "
        f"(F0.5 {primary_f05:.4f} vs {comparison_f05:.4f}, "
        f"precision {primary_precision:.4f} vs {comparison_precision:.4f})."
    )


def serialize_report_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): serialize_report_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_report_value(item) for item in value]
    return value


def print_candidate_leaderboard(candidate_rows: list[dict[str, Any]]) -> None:
    print("\nTraining-only candidate search leaderboard")
    print("-" * 96)
    header = (
        f"{'model':<10} {'profile':<9} {'miss':>6} {'feat':>5} {'pr_auc':>8} "
        f"{'roc_auc':>8} {'logloss':>9} {'brier':>8}"
    )
    print(header)
    for row in candidate_rows:
        print(
            f"{row['model_family']:<10} {row['feature_profile']:<9} {row['missingness_threshold']:6.2f} "
            f"{int(row['feature_count']):5d} {row['mean_cv_pr_auc']:8.4f} "
            f"{row['mean_cv_roc_auc']:8.4f} {row['mean_cv_log_loss']:9.4f} "
            f"{row['mean_cv_brier_score']:8.4f}"
        )


def evaluate_model_run(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_result: dict[str, Any],
    threshold_objective: str,
    min_recall: float,
    max_positive_rate: float,
    threshold_tolerance: float,
    calibration: str,
    save_ranked_output: bool = False,
    ranked_output_path: str | None = None,
) -> dict[str, Any]:
    summary_row = dict(candidate_result["summary_row"])
    feature_columns = list(candidate_result["feature_columns"])
    X_train = prepare_feature_matrix(train_df, feature_columns)
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = prepare_feature_matrix(test_df, feature_columns)
    y_test = test_df[TARGET_COL].to_numpy()
    base_estimator = clone(candidate_result["search"].best_estimator_)

    if summary_row["model_family"] == "logistic":
        model, calibration_status, calibration_search_rows = choose_logistic_calibration(
            base_estimator,
            X_train,
            y_train,
            feature_columns,
            calibration,
        )
    else:
        model = clone(base_estimator)
        fit_safely_with_imputer_warning_suppressed(model, X_train, y_train)
        calibration_status = {
            "requested": calibration,
            "used": "not_applicable",
            "status": "not_applicable",
            "message": f"Calibration skipped because {summary_row['model_family']} was used.",
        }
        calibration_search_rows = []

    print("\n" + "=" * 60)
    print("SELECTED MODEL EVALUATION")
    print("=" * 60)
    print(f"Selected model family        : {summary_row['model_family']}")
    print(f"Selected feature profile     : {summary_row['feature_profile']}")
    print(f"Selected missingness cutoff  : {summary_row['missingness_threshold']:.2f}")
    print(f"Selected feature count       : {len(feature_columns)}")
    print(f"Best hyperparameters         : {summary_row['best_params'] if summary_row['best_params'] else 'fixed default configuration'}")
    print(f"Selection metric             : {summary_row['selection_metric']}")
    print(f"Training CV PR-AUC           : {summary_row['mean_cv_pr_auc']:.4f}")
    print(f"Training CV log loss         : {summary_row['mean_cv_log_loss']:.4f}")
    print(f"Training CV Brier            : {summary_row['mean_cv_brier_score']:.4f}")
    print(f"Calibration mode requested   : {calibration_status['requested']}")
    print(f"Calibration mode actually used: {calibration_status['used']}")
    print(f"Calibration status           : {calibration_status['status']}")
    print(f"Calibration note             : {calibration_status['message']}")
    if calibration_search_rows:
        print("\nPost-selection calibration search")
        print("-" * 60)
        header = f"{'mode':<10} {'pr_auc':>8} {'roc_auc':>8} {'logloss':>9} {'brier':>8}"
        print(header)
        for row in calibration_search_rows:
            print(
                f"{row['mode']:<10} {row['mean_cv_pr_auc']:8.4f} {row['mean_cv_roc_auc']:8.4f} "
                f"{row['mean_cv_log_loss']:9.4f} {row['mean_cv_brier_score']:8.4f}"
            )

    print("\nGenerating training OOF probabilities for threshold tuning...")
    train_cv_summary = cross_validate_probability_metrics_time_series(
        model,
        X_train,
        y_train,
        feature_columns=feature_columns,
    )
    y_train_oof = train_cv_summary["oof_y_true"]
    y_prob_oof = train_cv_summary["oof_y_prob"]
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

    print("\nThreshold tuning summary (training OOF only)")
    print("-" * 60)
    print(f"Selected threshold           : {chosen_threshold:.4f}")
    print(f"Objective used               : {threshold_objective}")
    print(f"Objective score              : {threshold_info['objective_score']:.4f}")
    print(f"Max positive rate used       : {max_positive_rate:.4f}")
    print(f"Threshold tolerance used     : {threshold_tolerance:.4f}")
    print(f"Fallback used                : {'yes' if threshold_info['used_fallback'] else 'no'}")
    print(f"Fallback stage               : {threshold_info['fallback_stage']}")
    print(f"Applied min_recall           : {threshold_info['applied_min_recall']:.2f}")
    print(f"Applied max_positive_rate    : {threshold_info['applied_max_positive_rate']:.2f}")
    print(f"Precision                    : {threshold_result['precision']:.4f}")
    print(f"Recall                       : {threshold_result['recall']:.4f}")
    print(f"F1                           : {threshold_result['f1']:.4f}")
    print(f"F0.5                         : {threshold_result['f0.5']:.4f}")
    print(f"Balanced accuracy            : {threshold_result['balanced_accuracy']:.4f}")
    print(f"Positive prediction rate     : {threshold_result['positive_prediction_rate']:.4f}")
    print_prevalence_summary(float(y_train_oof.mean()), float(threshold_result['positive_prediction_rate']), "Training OOF")
    if threshold_info["warning"]:
        print(str(threshold_info["warning"]))
    else:
        print(
            "Selection rationale          : objective ranked exactly as before; within tolerance, ties were broken by higher precision, then lower positive rate, then higher threshold."
        )
    print_threshold_table(
        train_threshold_summary,
        title=f"Top 10 training OOF thresholds ranked by {threshold_objective}",
        objective=threshold_objective,
        limit=10,
    )
    print_oof_threshold_diagnostics(
        train_threshold_summary,
        min_recall=min_recall,
        max_positive_rate=max_positive_rate,
    )

    y_prob_test = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob_test, chosen_threshold)
    actual_holdout_rate = float(y_test.mean())
    operationally_usable = is_operationally_usable(metrics, actual_holdout_rate)
    model_row: dict[str, float | str] = {
        **summary_row,
        "model": f"{summary_row['model_family']}::{summary_row['feature_profile']}",
        "threshold": chosen_threshold,
        **{metric: float(metrics[metric]) for metric in METRIC_COLUMNS},
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "log_loss": float(metrics["log_loss"]),
        "brier_score": float(metrics["brier_score"]),
        "calibration_used": calibration_status["used"],
        "actual_hr_rate": actual_holdout_rate,
        "prediction_to_actual_rate_ratio": prediction_rate_ratio(
            metrics["positive_prediction_rate"], actual_holdout_rate
        ),
        "operationally_usable": "yes" if operationally_usable else "no",
        "operational_usability_reason": usability_reason(metrics, actual_holdout_rate),
    }

    print("\nHoldout evaluation")
    print("-" * 60)
    print(f"Selected threshold           : {chosen_threshold:.4f}")
    print(f"Actual holdout HR rate       : {actual_holdout_rate:.4f}")
    print_metric_block(metrics, actual_holdout_rate, label="Holdout", include_probability_metrics=True)
    print(f"Operationally usable         : {model_row['operationally_usable']}")
    print(f"Operational usability rule   : {model_row['operational_usability_reason']}")
    print_confusion_matrix(metrics)
    print_calibration_summary(y_test, y_prob_test)
    print_threshold_table(
        summarize_thresholds(y_test, y_prob_test, HOLDOUT_SUMMARY_THRESHOLDS),
        title="Holdout threshold tradeoff summary (key thresholds)",
        objective=threshold_objective,
        limit=None,
    )
    print("\nRanking-first note")
    print("-" * 60)
    print("This model output is intended primarily for ranking/surfacing HR candidates, not as a strict yes/no oracle.")

    lift_df = summarize_top_bucket_lift(y_test, y_prob_test, top_percents=TOP_BUCKET_PERCENTS)
    decile_df = summarize_prediction_deciles(y_test, y_prob_test, bins=10)
    ventile_df = summarize_prediction_deciles(y_test, y_prob_test, bins=20)
    print_top_bucket_lift_table(lift_df)
    print_prediction_bucket_summary(decile_df, label="Holdout prediction deciles (high-to-low score buckets)")
    print_prediction_bucket_summary(ventile_df, label="Holdout prediction ventiles (high-to-low score buckets)")

    ranked_output = build_ranked_predictions_output(
        source_df=test_df,
        y_prob=y_prob_test,
        reference_df=train_df,
        fitted_model=model,
        feature_columns=feature_columns,
        y_true=y_test,
    )
    print_top_candidates_summary(ranked_output, top_n=min(DEFAULT_TOP_CANDIDATES_TO_PRINT, len(ranked_output)))

    output_path: str | None = None
    if save_ranked_output:
        if ranked_output_path:
            output_path = ranked_output_path
        else:
            output_path = f"ranked_predictions_{summary_row['model_family']}_{summary_row['feature_profile']}.csv"
        ranked_output.to_csv(output_path, index=False)
        print(f"\nRanked output CSV saved       : {output_path}")

    return {
        "model_name": summary_row["model_family"],
        "fitted_model": model,
        "best_params": summary_row["best_params"],
        "calibration_status": calibration_status,
        "calibration_search_rows": calibration_search_rows,
        "threshold_info": threshold_info,
        "threshold_result": threshold_result,
        "holdout_metrics": metrics,
        "summary_row": model_row,
        "lift_df": lift_df,
        "decile_df": decile_df,
        "ventile_df": ventile_df,
        "ranked_output": ranked_output,
        "ranked_output_path": output_path,
        "feature_columns": feature_columns,
        "cv_summary": train_cv_summary,
        "excluded_features": list(candidate_result["excluded_features"]),
    }


def run_backtest(
    data_path: str,
    model_name: str = "all",
    feature_profile: str = "all",
    compare_against: str | None = None,
    threshold_objective: str = "f0.5",
    min_recall: float = 0.15,
    max_positive_rate: float = 0.14,
    threshold_tolerance: float = 0.001,
    calibration: str = "sigmoid",
    selection_metric: str = "pr_auc",
    missingness_threshold: float | None = None,
    save_ranked_output: bool = False,
    ranked_output_path: str | None = None,
    report_path: str | None = None,
) -> dict[str, Any]:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run python generate_data.py first.")

    df = load_data(data_path)
    train_df, test_df = chronological_split(df)
    run_dataset_sanity_checks(df, train_df, test_df)

    print("=" * 60)
    print("TIME-AWARE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Rows: total={len(df):,}, train={len(train_df):,}, test={len(test_df):,}")
    print(f"Train date range: {train_df[DATE_COL].min().date()} -> {train_df[DATE_COL].max().date()}")
    print(f"Test date range : {test_df[DATE_COL].min().date()} -> {test_df[DATE_COL].max().date()}")
    print(f"Base HR rate train/test: {train_df[TARGET_COL].mean():.4f} / {test_df[TARGET_COL].mean():.4f}")
    print(f"Model family requested       : {model_name}")
    print(f"Feature profile requested    : {feature_profile}")
    print(f"Selection metric used        : {selection_metric}")
    print(
        "Missingness thresholds used : "
        f"{', '.join(f'{threshold:.2f}' for threshold in resolve_missingness_thresholds(missingness_threshold))}"
    )
    print(f"Threshold objective used     : {threshold_objective}")
    print(f"Max positive rate used       : {max_positive_rate:.2f}")
    print(f"Threshold tolerance used     : {threshold_tolerance:.4f}")
    print(f"Calibration mode requested   : {calibration}")

    baseline_rows: list[dict[str, float | str | bool]] = []
    print("\nSingle-feature baseline holdout comparison")
    print("-" * 60)
    for baseline_feature in BASELINE_FEATURES:
        baseline_rows.append(
            evaluate_baseline_feature(
                train_df=train_df,
                test_df=test_df,
                feature_name=baseline_feature,
                objective=threshold_objective,
                min_recall=min_recall,
                max_positive_rate=max_positive_rate,
                threshold_tolerance=threshold_tolerance,
            )
        )
    print_baseline_results(baseline_rows)
    for row in baseline_rows:
        if "warning" not in row:
            print(f"  - {row['model']}: operationally_usable={row['operationally_usable']} ({row['operational_usability_reason']})")

    candidate_results = search_training_candidates(
        train_df,
        model_name=model_name,
        feature_profile=feature_profile,
        compare_against=compare_against,
        missingness_threshold=missingness_threshold,
        selection_metric=selection_metric,
    )
    if not candidate_results:
        raise ValueError("No model candidates completed successfully.")

    candidate_rows = rank_candidate_rows([dict(result["summary_row"]) for result in candidate_results])
    print_candidate_leaderboard(candidate_rows)
    selected_row = candidate_rows[0]
    selected_candidate = next(
        result
        for result in candidate_results
        if result["summary_row"]["model_family"] == selected_row["model_family"]
        and result["summary_row"]["feature_profile"] == selected_row["feature_profile"]
        and np.isclose(result["summary_row"]["missingness_threshold"], selected_row["missingness_threshold"])
    )
    final_result = evaluate_model_run(
        train_df=train_df,
        test_df=test_df,
        candidate_result=selected_candidate,
        threshold_objective=threshold_objective,
        min_recall=min_recall,
        max_positive_rate=max_positive_rate,
        threshold_tolerance=threshold_tolerance,
        calibration=calibration,
        save_ranked_output=save_ranked_output,
        ranked_output_path=ranked_output_path,
    )

    comparison_rows: list[dict[str, float | str]] = [final_result["summary_row"]]
    comparison_rows.extend(baseline_rows)
    print_comparison_table(comparison_rows, title="Compact model comparison summary (selected winner)")
    print("\nHoldout text summary")
    print("-" * 60)
    for line in holdout_commentary(final_result["summary_row"], baseline_rows):
        print(f"- {line}")

    report = {
        "selection_metric": selection_metric,
        "candidate_leaderboard": candidate_rows,
        "selected_candidate": selected_row,
        "final_holdout_summary": final_result["summary_row"],
        "baseline_rows": baseline_rows,
    }
    if report_path is not None:
        output_path = Path(report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            pd.DataFrame(report["candidate_leaderboard"]).to_csv(output_path, index=False)
        else:
            output_path.write_text(json.dumps(serialize_report_value(report), indent=2), encoding="utf-8")
        print(f"\nSearch report written         : {report_path}")

    print("\nBacktest complete.")
    print("-" * 60)
    print(
        "Selected final model         : "
        f"{final_result['summary_row']['model_family']} / {final_result['summary_row']['feature_profile']}"
    )
    print(f"Final holdout PR-AUC         : {final_result['summary_row']['pr_auc']:.4f}")
    print(f"Final holdout ROC-AUC        : {final_result['summary_row']['roc_auc']:.4f}")
    print(f"Final holdout log loss       : {final_result['summary_row']['log_loss']:.4f}")
    print(f"Final holdout Brier          : {final_result['summary_row']['brier_score']:.4f}")
    return {
        "train_df": train_df,
        "test_df": test_df,
        "candidate_results": candidate_results,
        "candidate_rows": candidate_rows,
        "selected_candidate": selected_candidate,
        "final_result": final_result,
        "baseline_rows": baseline_rows,
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", default=str(FINAL_DATA_PATH), help="Path to batter-game dataset CSV.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="all", help="Model family to search.")
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILE_CHOICES,
        default="all",
        help="Feature profile or search space for the backtest.",
    )
    parser.add_argument(
        "--compare-against",
        choices=[*FEATURE_PROFILE_CHOICES, "none"],
        default="none",
        help="Optional second feature profile to include in the training-only search.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRIC_CHOICES,
        default="pr_auc",
        help="Primary training-only CV metric used to rank candidate configurations.",
    )
    parser.add_argument(
        "--missingness-threshold",
        type=float,
        default=None,
        help="Optional fixed feature-missingness pruning threshold. Defaults to searching 0.35, 0.50, and 0.65.",
    )
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
        default=0.14,
        help="Maximum predicted positive rate allowed during threshold search before falling back.",
    )
    parser.add_argument(
        "--threshold-tolerance",
        type=float,
        default=0.001,
        help="Allow thresholds within this objective tolerance of the best score, then pick the more conservative option.",
    )
    parser.add_argument(
        "--calibration",
        choices=CALIBRATION_CHOICES,
        default="sigmoid",
        help="Probability calibration mode for logistic models, fit on training data only with time-aware CV where possible.",
    )
    parser.add_argument(
        "--calibrate-logistic",
        action="store_true",
        help="Deprecated alias for --calibration sigmoid to preserve existing CLI behavior.",
    )
    parser.add_argument(
        "--save-ranked-output",
        action="store_true",
        help="Save holdout ranked candidates to CSV for downstream inspection/use.",
    )
    parser.add_argument(
        "--ranked-output-path",
        default=None,
        help="Optional output path for ranked prediction CSV.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path to write the candidate-search report as JSON or CSV.",
    )
    args = parser.parse_args()
    if args.calibrate_logistic and args.calibration == "disabled":
        args.calibration = "sigmoid"
    return args


if __name__ == "__main__":
    args = parse_args()
    run_backtest(
        args.data_path,
        model_name=args.model,
        feature_profile=args.feature_profile,
        compare_against=None if args.compare_against == "none" else args.compare_against,
        selection_metric=args.selection_metric,
        missingness_threshold=args.missingness_threshold,
        threshold_objective=args.threshold_objective,
        min_recall=args.min_recall,
        max_positive_rate=args.max_positive_rate,
        threshold_tolerance=args.threshold_tolerance,
        calibration=args.calibration,
        save_ranked_output=args.save_ranked_output,
        ranked_output_path=args.ranked_output_path,
        report_path=args.report_path,
    )
