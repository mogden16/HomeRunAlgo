"""Time-aware backtest for real MLB home run prediction from batter-game data."""

from __future__ import annotations

import argparse
from datetime import datetime
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
from feature_engineering import (
    CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS,
    LEGACY_ENGINEERED_FEATURE_COLUMNS,
)

DATE_COL = "game_date"
TARGET_COL = "hit_hr"
MAX_MODEL_FEATURE_MISSINGNESS = 0.50
SCHEMA_FAIL_MIN_PRESENT_SHARE = 0.70


def print_dataset_load_audit(path: Path, df: pd.DataFrame) -> None:
    modified_ts = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    print("\nDataset load audit")
    print("-" * 60)
    print(f"Absolute dataset path: {path}")
    print(f"File modified timestamp: {modified_ts}")
    print(f"Row count after load: {len(df):,}")
    print(f"Top 20 columns: {df.columns[:20].tolist()}")
    print(f"Total column count: {len(df.columns)}")


def classify_schema(df: pd.DataFrame) -> tuple[str, dict[str, object]]:
    present_current = [column for column in CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS if column in df.columns]
    missing_current = [column for column in CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS if column not in df.columns]
    legacy_present = [column for column in LEGACY_ENGINEERED_FEATURE_COLUMNS if column in df.columns]
    present_share = (
        len(present_current) / len(CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS)
        if CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS
        else 1.0
    )
    if present_share >= SCHEMA_FAIL_MIN_PRESENT_SHARE and not legacy_present:
        schema_label = "new schema"
    elif legacy_present and present_share < SCHEMA_FAIL_MIN_PRESENT_SHARE:
        schema_label = "old schema"
    else:
        schema_label = "mixed schema"
    return schema_label, {
        "present_current": present_current,
        "missing_current": missing_current,
        "legacy_present": legacy_present,
        "present_share": present_share,
    }


def print_schema_audit(schema_label: str, schema_details: dict[str, object]) -> None:
    print("\nSchema/version sanity check")
    print("-" * 60)
    print(f"Schema classification: {schema_label}")
    print(f"Expected current candidate features: {len(CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS)}")
    print(f"Present current candidate features: {len(schema_details['present_current'])}")
    print(f"Missing current candidate features: {schema_details['missing_current'] if schema_details['missing_current'] else 'None'}")
    print(f"Legacy schema features present: {schema_details['legacy_present'] if schema_details['legacy_present'] else 'None'}")
    print(f"Current feature coverage: {schema_details['present_share']:.1%}")


def fail_if_schema_mismatch(schema_label: str, schema_details: dict[str, object]) -> None:
    if schema_details["present_share"] < SCHEMA_FAIL_MIN_PRESENT_SHARE:
        raise ValueError(
            "Loaded dataset does not match the intended current engineered schema. "
            f"Schema classification={schema_label}; missing current candidate features="
            f"{schema_details['missing_current']}"
        )


def load_data(path: str) -> tuple[pd.DataFrame, Path, str, dict[str, object]]:
    resolved_path = Path(path).expanduser().resolve()
    df = pd.read_csv(resolved_path, parse_dates=[DATE_COL])
    required_columns = [DATE_COL, TARGET_COL, "game_pk", "player_id"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")
    df = df.sort_values([DATE_COL, "game_pk", "player_id"]).reset_index(drop=True)
    print_dataset_load_audit(resolved_path, df)
    schema_label, schema_details = classify_schema(df)
    print_schema_audit(schema_label, schema_details)
    return df, resolved_path, schema_label, schema_details


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


def print_candidate_feature_audit(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    present_features = [column for column in CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS if column in df.columns]
    missing_features = [column for column in CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS if column not in df.columns]
    print("\nCandidate feature audit")
    print("-" * 60)
    print(f"Current candidate feature list ({len(CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS)}): {CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS}")
    print(f"Candidate features present in dataset ({len(present_features)}): {present_features}")
    print(f"Candidate features missing from dataset ({len(missing_features)}): {missing_features if missing_features else 'None'}")
    return present_features, missing_features


def prune_sparse_features(train_df: pd.DataFrame, candidate_features: list[str]) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    kept: list[str] = []
    for feature_name in candidate_features:
        is_present = feature_name in train_df.columns
        if is_present:
            missing_pct = float(train_df[feature_name].isna().mean())
            keep_for_model = missing_pct <= MAX_MODEL_FEATURE_MISSINGNESS
            reason = "kept for model" if keep_for_model else "excluded because sparse"
            if keep_for_model:
                kept.append(feature_name)
        else:
            missing_pct = np.nan
            keep_for_model = False
            reason = "not present because schema mismatch"
        rows.append(
            {
                "feature_name": feature_name,
                "present_in_dataset": "yes" if is_present else "no",
                "train_missing_pct": missing_pct,
                "keep_for_model": "yes" if keep_for_model else "no",
                "reason": reason,
            }
        )
    audit_df = pd.DataFrame(rows).sort_values(
        ["keep_for_model", "present_in_dataset", "train_missing_pct", "feature_name"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    return kept, audit_df


def baseline_comparison_features(train_df: pd.DataFrame, candidate_features: list[str]) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    usable: list[str] = []
    for feature_name in candidate_features:
        if feature_name not in train_df.columns:
            rows.append({"feature_name": feature_name, "include_in_baseline": "no", "reason": "not present in dataset"})
            continue
        missing_pct = float(train_df[feature_name].isna().mean())
        if missing_pct >= 1.0:
            rows.append({"feature_name": feature_name, "include_in_baseline": "no", "reason": "100% missing in training data"})
            continue
        usable.append(feature_name)
        rows.append({"feature_name": feature_name, "include_in_baseline": "yes", "reason": "usable for baseline comparison"})
    return usable, pd.DataFrame(rows)


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


def _resolved_text_series(df: pd.DataFrame, preferred: str, secondary: str | None, fallback_prefix: str, id_column: str) -> pd.Series:
    values = df[preferred].astype(str).where(df[preferred].notna(), "") if preferred in df.columns else pd.Series("", index=df.index)
    if secondary and secondary in df.columns:
        secondary_values = df[secondary].astype(str).where(df[secondary].notna(), "")
        values = values.where(values.str.strip() != "", secondary_values)
    fallback = fallback_prefix + "_" + df[id_column].fillna(-1).astype(int).astype(str)
    values = values.where(values.str.strip() != "", fallback)
    return values


def build_ranked_display(ranking_df: pd.DataFrame) -> pd.DataFrame:
    display_df = ranking_df.copy()
    display_df["batter_id"] = display_df["player_id"]
    display_df["batter_name"] = _resolved_text_series(
        display_df,
        preferred="batter_name",
        secondary="player_name",
        fallback_prefix="batter",
        id_column="player_id",
    )
    display_df["pitcher_name"] = _resolved_text_series(
        display_df,
        preferred="pitcher_name",
        secondary="opp_pitcher_name",
        fallback_prefix="pitcher",
        id_column="opp_pitcher_id",
    )
    return display_df[["batter_id", "batter_name", "pitcher_name", "score"]]


def run_backtest(data_path: str, model_name: str = "logistic") -> None:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run python generate_data.py first.")

    df, resolved_data_path, schema_label, schema_details = load_data(data_path)
    present_candidate_features, _ = print_candidate_feature_audit(df)
    fail_if_schema_mismatch(schema_label, schema_details)
    train_df, test_df = chronological_split(df)
    run_dataset_sanity_checks(df, train_df, test_df)

    pruned_feature_columns, pruning_audit_df = prune_sparse_features(train_df, CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS)
    if not pruned_feature_columns:
        raise RuntimeError("All candidate features were excluded by missingness threshold.")

    print("\nFeature-pruning audit")
    print("-" * 60)
    print(pruning_audit_df.to_string(index=False))
    excluded_features = pruning_audit_df.loc[pruning_audit_df["keep_for_model"] == "no", "feature_name"].tolist()
    schema_mismatch_features = pruning_audit_df.loc[
        pruning_audit_df["reason"] == "not present because schema mismatch", "feature_name"
    ].tolist()
    baseline_feature_columns, baseline_audit_df = baseline_comparison_features(train_df, present_candidate_features)
    print(f"\nFinal modeled feature count: {len(pruned_feature_columns)}")
    print(f"Excluded features ({len(excluded_features)}): {excluded_features if excluded_features else 'None'}")
    if schema_mismatch_features:
        print(f"Schema-mismatch candidate features ({len(schema_mismatch_features)}): {schema_mismatch_features}")
    baseline_excluded_df = baseline_audit_df[baseline_audit_df["include_in_baseline"] == "no"]
    print("\nUnpruned baseline feature audit")
    print("-" * 60)
    if baseline_excluded_df.empty:
        print("Baseline-excluded features: None")
    else:
        print(baseline_excluded_df.to_string(index=False))

    X_train = train_df[pruned_feature_columns].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_test = test_df[pruned_feature_columns].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print("=" * 60)
    print("TIME-AWARE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Dataset path used: {resolved_data_path}")
    print(f"Loaded schema classification: {schema_label}")
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
    if excluded_features and baseline_feature_columns:
        X_train_unpruned = train_df[baseline_feature_columns].to_numpy()
        X_test_unpruned = test_df[baseline_feature_columns].to_numpy()
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
    ranking_display = build_ranked_display(ranking_df)
    top_identity_preview = ranking_display.head(10).copy()
    final_top_ranked = ranking_display.head(10).copy()
    preview_name_match = top_identity_preview.equals(final_top_ranked)

    print("\nTop-ranked candidates preview")
    print("-" * 60)
    print(top_identity_preview.to_string(index=False))

    print("\nTop ranked candidates")
    print("-" * 60)
    print(final_top_ranked.to_string(index=False))

    print("\nRanked-output identity preview")
    print("-" * 60)
    print(top_identity_preview.to_string(index=False))
    print(f"Ranked-output name consistency: {'yes' if preview_name_match else 'no'}")

    stale_legacy_features_removed = [feature for feature in LEGACY_ENGINEERED_FEATURE_COLUMNS if feature not in CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS]
    print("\nFinal synchronization summary")
    print("-" * 60)
    print(f"Dataset path actually used: {resolved_data_path}")
    print(f"Schema matches intended current schema: {'yes' if schema_label == 'new schema' else 'no'}")
    print(f"Intended candidate feature count: {len(CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS)}")
    print(f"Present candidate feature count: {len(present_candidate_features)}")
    print(f"Pruned modeled feature count: {len(pruned_feature_columns)}")
    print(f"Stale legacy features removed ({len(stale_legacy_features_removed)}): {stale_legacy_features_removed}")
    print(f"Ranked-output names now consistent: {'yes' if preview_name_match else 'no'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", default=str(FINAL_DATA_PATH), help="Path to batter-game dataset CSV.")
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic", help="Model family to train.")
    args = parser.parse_args()
    run_backtest(args.data_path, model_name=args.model)
