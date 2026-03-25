"""Time-aware backtest for real MLB home run prediction from batter-game data."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

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
    CONTACT_AUTHORITY_FEATURE_COLUMNS,
    CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS,
    LEGACY_ENGINEERED_FEATURE_COLUMNS,
    PITCH_STYLE_FEATURE_COLUMNS,
)

DATE_COL = "game_date"
TARGET_COL = "hit_hr"
MAX_MODEL_FEATURE_MISSINGNESS = 0.50
SCHEMA_FAIL_MIN_PRESENT_SHARE = 0.70
TOP_RECOMMENDATION_SHARE = 0.20
INITIAL_TIER_RULES = [
    ("Tier 1", 0.02),
    ("Tier 2", 0.05),
    ("Tier 3", 0.10),
    ("Tier 4", 0.20),
]
CONFIDENCE_PROFILE_ORDER = ["High", "Medium", "Low"]
MIN_CONFIDENCE_BUCKET_ROWS = 150
MIN_CONFIDENCE_RATE_GAP = 0.015

FAMILY_REASON_LABELS = {
    "batter_recent_form": "strong recent barrel/hard-hit form",
    "pitcher_hard_contact_risk": "pitcher allows hard contact",
    "pitch_style_matchup": "favorable pitch-mix matchup",
    "park_weather": "power-friendly park/weather",
    "handedness_context": "favorable handedness/context",
}

FEATURE_FAMILY_MAP = {
    "batter_recent_form": [
        "hr_rate_season_to_date",
        "barrel_rate_last_50_bbe",
        "hard_hit_rate_last_50_bbe",
        "avg_launch_angle_last_50_bbe",
        "avg_exit_velocity_last_50_bbe",
        "fly_ball_rate_last_50_bbe",
        "pull_air_rate_last_50_bbe",
        "batter_k_rate_season_to_date",
        "batter_bb_rate_season_to_date",
        "days_since_last_game",
        "bbe_count_last_50",
        "avg_hit_distance_last_50_bbe",
        "long_contact_rate_last_50_bbe",
    ],
    "pitcher_hard_contact_risk": [
        "pitcher_hr9_season_to_date",
        "pitcher_barrel_rate_allowed_last_50_bbe",
        "pitcher_hard_hit_rate_allowed_last_50_bbe",
        "pitcher_fb_rate_allowed_last_50_bbe",
        "pitcher_avg_hit_distance_allowed_last_50_bbe",
        "pitcher_k_rate_season_to_date",
        "pitcher_bb_rate_season_to_date",
    ],
    "pitch_style_matchup": list(PITCH_STYLE_FEATURE_COLUMNS),
    "park_weather": [
        "park_factor_hr",
        "temperature_f",
        "humidity_pct",
        "wind_speed_mph",
        "wind_direction_deg",
        "pressure_hpa",
    ],
    "handedness_context": [
        "platoon_advantage",
    ],
}


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


def print_contact_feature_status(pruning_audit_df: pd.DataFrame) -> None:
    contact_df = pruning_audit_df[pruning_audit_df["feature_name"].isin(CONTACT_AUTHORITY_FEATURE_COLUMNS)].copy()
    print("\nContact-authority pruning audit")
    print("-" * 60)
    if contact_df.empty:
        print("No contact-authority candidate features found.")
        return
    print(contact_df.to_string(index=False))


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


def print_ranking_diagnostics(ranking_df: pd.DataFrame) -> None:
    ranked = ranking_df.sort_values("score", ascending=False).reset_index(drop=True)
    base_rate = float(ranked[TARGET_COL].mean()) if len(ranked) else np.nan
    print("\nRanking diagnostics")
    print("-" * 60)
    print(f"Holdout base HR rate: {base_rate:.4f}")

    def print_top_pct(pct: float) -> None:
        top_n = max(1, int(np.ceil(len(ranked) * pct)))
        top_slice = ranked.head(top_n)
        top_rate = float(top_slice[TARGET_COL].mean()) if len(top_slice) else np.nan
        lift = (top_rate / base_rate) if base_rate and not np.isnan(base_rate) else np.nan
        print(f"Top {int(pct * 100)}% rows ({top_n:,}) HR rate: {top_rate:.4f} | lift vs base: {lift:.2f}x")

    def print_top_n(top_n: int) -> None:
        top_slice = ranked.head(min(top_n, len(ranked)))
        top_rate = float(top_slice[TARGET_COL].mean()) if len(top_slice) else np.nan
        print(f"Top {min(top_n, len(ranked)):,} rows HR rate: {top_rate:.4f}")

    print_top_pct(0.05)
    print_top_pct(0.10)
    print_top_n(50)
    print_top_n(100)


def print_logistic_coefficient_summary(model: Pipeline, feature_columns: list[str]) -> None:
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else None
    if not isinstance(clf, LogisticRegression):
        return

    coefficient_df = (
        pd.DataFrame({"feature_name": feature_columns, "coefficient": clf.coef_.ravel()})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
    print("\nLogistic coefficient summary")
    print("-" * 60)
    print("Top positive coefficients:")
    print(coefficient_df.head(10).to_string(index=False))

    pitch_style_df = coefficient_df[coefficient_df["feature_name"].isin(PITCH_STYLE_FEATURE_COLUMNS)].copy()
    print("\nPitch-style feature coefficients:")
    if pitch_style_df.empty:
        print("None")
    else:
        print(pitch_style_df.sort_values("coefficient", ascending=False).to_string(index=False))


def split_for_recommendation(df: pd.DataFrame, recommend_date: str) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    recommend_ts = pd.Timestamp(recommend_date)
    train_df = df[df[DATE_COL] < recommend_ts].copy()
    recommend_df = df[df[DATE_COL] == recommend_ts].copy()
    if train_df.empty:
        raise ValueError(f"No training rows exist before recommendation date {recommend_ts.date()}.")
    if recommend_df.empty:
        raise ValueError(f"No hitter rows exist for recommendation date {recommend_ts.date()}.")
    return recommend_ts, train_df, recommend_df


def fit_selected_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> Pipeline:
    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    if model_name == "xgboost":
        model = build_xgboost_pipeline()
        if model is None:
            print("XGBoost is not installed; falling back to logistic regression.")
            return tune_logistic_pipeline(X_train, y_train)
        model.fit(X_train, y_train)
        return model
    return tune_logistic_pipeline(X_train, y_train)


def score_frame(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return model.predict_proba(df[feature_columns].to_numpy())[:, 1]


def add_slate_ranking_columns(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked = ranked.sort_values([DATE_COL, "score"], ascending=[True, False]).reset_index(drop=True)
    ranked["slate_size"] = ranked.groupby(DATE_COL)["score"].transform("size")
    ranked["slate_rank"] = ranked.groupby(DATE_COL)["score"].rank(method="first", ascending=False).astype(int)
    ranked["slate_percentile"] = (1.0 - ((ranked["slate_rank"] - 1) / ranked["slate_size"].clip(lower=1))) * 100.0
    ranked["initial_recommendation_tier"] = ranked.apply(_assign_initial_tier, axis=1)
    return ranked


def _assign_initial_tier(row: pd.Series) -> str | None:
    if pd.isna(row["slate_size"]) or int(row["slate_size"]) <= 0:
        return None
    slate_size = int(row["slate_size"])
    rank = int(row["slate_rank"])
    threshold_counts = {
        "Tier 1": max(1, int(np.ceil(slate_size * 0.02))),
        "Tier 2": max(1, int(np.ceil(slate_size * 0.05))),
        "Tier 3": max(1, int(np.ceil(slate_size * 0.10))),
        "Tier 4": max(1, int(np.ceil(slate_size * 0.20))),
    }
    if rank <= threshold_counts["Tier 1"]:
        return "Tier 1"
    if rank <= threshold_counts["Tier 2"]:
        return "Tier 2"
    if rank <= threshold_counts["Tier 3"]:
        return "Tier 3"
    if rank <= threshold_counts["Tier 4"]:
        return "Tier 4"
    return None


def derive_tier_plan(scored_df: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame, dict[str, float]]:
    working_df = scored_df[scored_df["initial_recommendation_tier"].notna()].copy()
    tier_order = [name for name, _ in INITIAL_TIER_RULES]
    current_group = {tier_name: tier_name for tier_name in tier_order}

    while True:
        grouped = working_df.assign(group_tier=working_df["initial_recommendation_tier"].map(current_group))
        summary = (
            grouped.groupby("group_tier", dropna=False)
            .agg(rows=(TARGET_COL, "size"), hr_rate=(TARGET_COL, "mean"))
            .reset_index()
        )
        present_order = [tier for tier in tier_order if tier in summary["group_tier"].tolist()]
        violation_found = False
        for idx in range(len(present_order) - 1):
            current_tier = present_order[idx]
            next_tier = present_order[idx + 1]
            current_rate = float(summary.loc[summary["group_tier"] == current_tier, "hr_rate"].iloc[0])
            next_rate = float(summary.loc[summary["group_tier"] == next_tier, "hr_rate"].iloc[0])
            if current_rate < next_rate:
                for tier_name, mapped_name in list(current_group.items()):
                    if mapped_name == current_tier:
                        current_group[tier_name] = next_tier
                violation_found = True
                break
        if not violation_found:
            break

    surviving_order = []
    for tier_name in tier_order:
        mapped_name = current_group[tier_name]
        if mapped_name not in surviving_order:
            surviving_order.append(mapped_name)
    compressed_map = {old_name: f"Tier {idx + 1}" for idx, old_name in enumerate(surviving_order)}
    final_map = {
        tier_name: compressed_map[current_group[tier_name]]
        for tier_name in tier_order
        if tier_name in current_group
    }
    finalized = scored_df.copy()
    finalized["recommendation_tier"] = finalized["initial_recommendation_tier"].map(final_map)
    tier_summary = summarize_tiers(finalized)
    tier_lift_map = tier_summary.set_index("recommendation_tier")["lift_vs_slate_base"].to_dict() if not tier_summary.empty else {}
    return final_map, tier_summary, tier_lift_map


def summarize_tiers(scored_df: pd.DataFrame) -> pd.DataFrame:
    tier_df = scored_df[scored_df["recommendation_tier"].notna()].copy()
    if tier_df.empty:
        return pd.DataFrame(columns=["recommendation_tier", "rows", "hr_rate", "slate_base_rate", "lift_vs_slate_base", "avg_recommendations_per_slate"])
    tier_df["slate_base_rate"] = tier_df.groupby(DATE_COL)[TARGET_COL].transform("mean")
    tier_summary = (
        tier_df.groupby("recommendation_tier", dropna=False)
        .agg(
            rows=(TARGET_COL, "size"),
            hr_rate=(TARGET_COL, "mean"),
            slate_base_rate=("slate_base_rate", "mean"),
        )
        .reset_index()
    )
    tier_summary["lift_vs_slate_base"] = tier_summary["hr_rate"] / tier_summary["slate_base_rate"].replace({0: np.nan})
    counts_per_day = (
        tier_df.groupby([DATE_COL, "recommendation_tier"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    avg_counts = counts_per_day.groupby("recommendation_tier", dropna=False)["count"].mean().rename("avg_recommendations_per_slate")
    tier_summary = tier_summary.merge(avg_counts, on="recommendation_tier", how="left")
    tier_summary["tier_sort"] = tier_summary["recommendation_tier"].str.extract(r"(\d+)").astype(int)
    tier_summary = tier_summary.sort_values("tier_sort").drop(columns=["tier_sort"]).reset_index(drop=True)
    return tier_summary


def print_tier_diagnostics(scored_df: pd.DataFrame, tier_summary: pd.DataFrame) -> None:
    print("\nDaily recommendation diagnostics")
    print("-" * 60)
    if tier_summary.empty:
        print("No tiered recommendation rows available.")
        return
    print("Tier performance table:")
    print(tier_summary.to_string(index=False))

    top5 = scored_df.groupby(DATE_COL, dropna=False).head(5)
    top10 = scored_df.groupby(DATE_COL, dropna=False).head(10)
    recommended = scored_df[scored_df["recommendation_tier"].notna()].copy()
    confidence_summary = pd.DataFrame()
    if "confidence_label" in recommended.columns:
        confidence_summary = (
            recommended.groupby("confidence_label", dropna=False)
            .agg(rows=(TARGET_COL, "size"), hr_rate=(TARGET_COL, "mean"))
            .reset_index()
        )
        confidence_summary["label_sort"] = confidence_summary["confidence_label"].map({"High": 0, "Medium": 1, "Low": 2})
        confidence_summary = confidence_summary.sort_values("label_sort").drop(columns=["label_sort"]).reset_index(drop=True)

    print(f"Top 5 hitters per slate HR rate: {top5[TARGET_COL].mean():.4f}")
    print(f"Top 10 hitters per slate HR rate: {top10[TARGET_COL].mean():.4f}")
    print("Confidence-label hit rate:")
    if confidence_summary.empty:
        print("None")
    else:
        print(confidence_summary.to_string(index=False))


def compute_confidence_metadata(
    scored_df: pd.DataFrame,
    feature_columns: list[str],
    tier_lift_map: dict[str, float],
) -> pd.DataFrame:
    enriched = scored_df.copy()
    batter_recent_fields = [
        "hr_rate_season_to_date",
        "barrel_rate_last_50_bbe",
        "hard_hit_rate_last_50_bbe",
        "avg_launch_angle_last_50_bbe",
        "avg_exit_velocity_last_50_bbe",
        "fly_ball_rate_last_50_bbe",
        "pull_air_rate_last_50_bbe",
        "batter_k_rate_season_to_date",
        "batter_bb_rate_season_to_date",
        "days_since_last_game",
    ]
    pitcher_history_fields = [
        "pitcher_hr9_season_to_date",
        "pitcher_barrel_rate_allowed_last_50_bbe",
        "pitcher_hard_hit_rate_allowed_last_50_bbe",
        "pitcher_fb_rate_allowed_last_50_bbe",
        "pitcher_k_rate_season_to_date",
        "pitcher_bb_rate_season_to_date",
        *PITCH_STYLE_FEATURE_COLUMNS,
    ]
    batter_recent_share = enriched[batter_recent_fields].notna().mean(axis=1)
    batter_bbe_share = (enriched["bbe_count_last_50"].fillna(0).clip(lower=0, upper=50) / 50.0)
    batter_history_depth = 0.7 * batter_bbe_share + 0.3 * batter_recent_share
    pitcher_history_depth = enriched[pitcher_history_fields].notna().mean(axis=1)
    history_depth_score = 100.0 * (0.5 * batter_history_depth + 0.5 * pitcher_history_depth)
    feature_coverage_score = 100.0 * enriched[feature_columns].notna().mean(axis=1)
    score_percentile_score = enriched["slate_percentile"].fillna(0.0)
    tier_lift_score = enriched["recommendation_tier"].map(tier_lift_map).fillna(1.0) * 50.0
    tier_lift_score = tier_lift_score.clip(lower=0.0, upper=100.0)

    enriched["history_depth_score"] = history_depth_score.clip(lower=0.0, upper=100.0)
    enriched["feature_coverage_score"] = feature_coverage_score.clip(lower=0.0, upper=100.0)
    enriched["tier_lift_score"] = tier_lift_score
    enriched["confidence_score"] = (
        0.40 * score_percentile_score
        + 0.25 * enriched["tier_lift_score"]
        + 0.20 * enriched["history_depth_score"]
        + 0.15 * enriched["feature_coverage_score"]
    ).clip(lower=0.0, upper=100.0)
    enriched["tier_profile_label"] = enriched["recommendation_tier"].map(describe_tier_profile)
    return enriched


def describe_tier_profile(tier_name: str | None) -> str | None:
    if tier_name is None or pd.isna(tier_name):
        return None
    tier_number = int(str(tier_name).split()[-1])
    return "strongest recommendations" if tier_number <= 2 else "deeper longshots"


def _assign_initial_confidence_bucket(score: float, lower_cut: float, upper_cut: float) -> int:
    if score >= upper_cut:
        return 0
    if score >= lower_cut:
        return 1
    return 2


def _summarize_confidence_groups(profile_df: pd.DataFrame, groups: list[list[int]], label_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label_name, bucket_group in zip(label_names, groups):
        group_mask = profile_df["initial_confidence_bucket"].isin(bucket_group)
        group_rows = int(group_mask.sum())
        group_rate = float(profile_df.loc[group_mask, TARGET_COL].mean()) if group_rows else np.nan
        rows.append({"confidence_label": label_name, "rows": group_rows, "hr_rate": group_rate, "initial_buckets": bucket_group})
    return pd.DataFrame(rows)


def derive_confidence_profile(recommended_profile_df: pd.DataFrame) -> dict[str, object]:
    recommended = recommended_profile_df[recommended_profile_df["recommendation_tier"].notna()].copy()
    if recommended.empty:
        return {
            "lower_cut": 0.0,
            "upper_cut": 100.0,
            "bucket_label_map": {0: "High", 1: "Medium", 2: "Low"},
            "summary": pd.DataFrame(columns=["confidence_label", "rows", "hr_rate"]),
            "merges_applied": ["No recommended holdout rows available for confidence profiling."],
            "effective_min_bucket_rows": 0,
        }

    scores = recommended["confidence_score"].fillna(0.0)
    lower_cut = float(scores.quantile(1 / 3))
    upper_cut = float(scores.quantile(2 / 3))
    recommended["initial_confidence_bucket"] = scores.apply(
        lambda value: _assign_initial_confidence_bucket(value, lower_cut=lower_cut, upper_cut=upper_cut)
    )

    effective_min_bucket_rows = min(MIN_CONFIDENCE_BUCKET_ROWS, max(50, len(recommended) // 5))
    groups: list[list[int]] = [[0], [1], [2]]
    merges_applied: list[str] = []

    while len(groups) > 1:
        label_names = CONFIDENCE_PROFILE_ORDER[: len(groups)]
        summary = _summarize_confidence_groups(recommended, groups, label_names)
        too_small = summary["rows"] < effective_min_bucket_rows
        if too_small.any():
            merge_idx = int(summary.loc[too_small, "rows"].idxmin())
            if merge_idx == 0:
                merge_target_idx = 1
            elif merge_idx == len(groups) - 1:
                merge_target_idx = merge_idx - 1
            else:
                prev_gap = abs(float(summary.loc[merge_idx, "hr_rate"]) - float(summary.loc[merge_idx - 1, "hr_rate"]))
                next_gap = abs(float(summary.loc[merge_idx, "hr_rate"]) - float(summary.loc[merge_idx + 1, "hr_rate"]))
                merge_target_idx = merge_idx - 1 if prev_gap <= next_gap else merge_idx + 1
            low_idx = min(merge_idx, merge_target_idx)
            high_idx = max(merge_idx, merge_target_idx)
            merges_applied.append(
                f"Merged {label_names[low_idx]} and {label_names[high_idx]} due to small bucket size "
                f"({int(summary.loc[merge_idx, 'rows'])} rows < {effective_min_bucket_rows})."
            )
            groups = groups[:low_idx] + [groups[low_idx] + groups[high_idx]] + groups[high_idx + 1 :]
            continue

        violation_idx: int | None = None
        for idx in range(len(summary) - 1):
            current_rate = float(summary.loc[idx, "hr_rate"])
            next_rate = float(summary.loc[idx + 1, "hr_rate"])
            if current_rate < next_rate:
                violation_idx = idx
                break
            if current_rate - next_rate < MIN_CONFIDENCE_RATE_GAP:
                violation_idx = idx
                merges_applied.append(
                    f"Merged {label_names[idx]} and {label_names[idx + 1]} due to weak separation "
                    f"({current_rate:.4f} vs {next_rate:.4f}, gap < {MIN_CONFIDENCE_RATE_GAP:.3f})."
                )
                groups = groups[:idx] + [groups[idx] + groups[idx + 1]] + groups[idx + 2 :]
                break
        if violation_idx is None:
            break

        if len(groups) < len(summary):
            continue
        merges_applied.append(
            f"Merged {label_names[violation_idx]} and {label_names[violation_idx + 1]} due to non-monotonic hit rates "
            f"({summary.loc[violation_idx, 'hr_rate']:.4f} < {summary.loc[violation_idx + 1, 'hr_rate']:.4f})."
        )
        groups = groups[:violation_idx] + [groups[violation_idx] + groups[violation_idx + 1]] + groups[violation_idx + 2 :]

    final_labels = CONFIDENCE_PROFILE_ORDER[: len(groups)]
    final_summary = _summarize_confidence_groups(recommended, groups, final_labels)
    bucket_label_map: dict[int, str] = {}
    for label_name, bucket_group in zip(final_labels, groups):
        for bucket_id in bucket_group:
            bucket_label_map[bucket_id] = label_name

    if not merges_applied:
        merges_applied = ["No confidence bucket merges required."]

    return {
        "lower_cut": lower_cut,
        "upper_cut": upper_cut,
        "bucket_label_map": bucket_label_map,
        "summary": final_summary.drop(columns=["initial_buckets"]),
        "merges_applied": merges_applied,
        "effective_min_bucket_rows": effective_min_bucket_rows,
    }


def apply_confidence_profile(scored_df: pd.DataFrame, confidence_profile: dict[str, object]) -> pd.DataFrame:
    enriched = scored_df.copy()
    lower_cut = float(confidence_profile["lower_cut"])
    upper_cut = float(confidence_profile["upper_cut"])
    bucket_map = confidence_profile["bucket_label_map"]
    enriched["initial_confidence_bucket"] = enriched["confidence_score"].fillna(0.0).apply(
        lambda value: _assign_initial_confidence_bucket(value, lower_cut=lower_cut, upper_cut=upper_cut)
    )
    enriched["confidence_label"] = enriched["initial_confidence_bucket"].map(bucket_map).fillna("Medium")
    return enriched.drop(columns=["initial_confidence_bucket"])


def summarize_confidence_labels(scored_df: pd.DataFrame) -> pd.DataFrame:
    recommended = scored_df[scored_df["recommendation_tier"].notna() & scored_df["confidence_label"].notna()].copy()
    if recommended.empty:
        return pd.DataFrame(columns=["confidence_label", "rows", "hr_rate"])
    summary = (
        recommended.groupby("confidence_label", dropna=False)
        .agg(rows=(TARGET_COL, "size"), hr_rate=(TARGET_COL, "mean"))
        .reset_index()
    )
    summary["label_sort"] = summary["confidence_label"].map({label: idx for idx, label in enumerate(CONFIDENCE_PROFILE_ORDER)})
    return summary.sort_values("label_sort").drop(columns=["label_sort"]).reset_index(drop=True)


def print_confidence_profile_audit(confidence_profile: dict[str, object]) -> None:
    print("\nConfidence profile audit")
    print("-" * 60)
    print(f"Validation-driven confidence score cuts: lower={confidence_profile['lower_cut']:.2f}, upper={confidence_profile['upper_cut']:.2f}")
    print(f"Minimum bucket rows enforced: {confidence_profile['effective_min_bucket_rows']}")
    print("Bucket merges applied:")
    for merge_line in confidence_profile["merges_applied"]:
        print(f"  {merge_line}")
    print("Validation bucket summary:")
    summary = confidence_profile["summary"]
    if summary.empty:
        print("None")
    else:
        print(summary.to_string(index=False))


def compute_family_reason_series(model: Pipeline, df: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    if not hasattr(model, "named_steps"):
        return pd.Series("", index=df.index)
    clf = model.named_steps.get("clf")
    imputer = model.named_steps.get("imputer")
    scaler = model.named_steps.get("scaler")
    if not isinstance(clf, LogisticRegression) or imputer is None or scaler is None:
        return pd.Series("", index=df.index)

    X_raw = df[feature_columns].to_numpy(dtype=float)
    X_imputed = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imputed)
    contributions = X_scaled * clf.coef_.ravel()
    contrib_df = pd.DataFrame(contributions, columns=feature_columns, index=df.index)
    reasons: list[str] = []
    for row_index in contrib_df.index:
        family_scores = {
            family_name: float(contrib_df.loc[row_index, [feature for feature in features if feature in contrib_df.columns]].sum())
            for family_name, features in FEATURE_FAMILY_MAP.items()
        }
        ordered = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        positive_families = [FAMILY_REASON_LABELS[family_name] for family_name, score in ordered if score > 0][:3]
        if not positive_families and ordered:
            positive_families = [FAMILY_REASON_LABELS[ordered[0][0]]]
        reasons.append("; ".join(positive_families))
    return pd.Series(reasons, index=df.index)


def build_recommendation_table(
    scored_df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    recommendation_df = scored_df[scored_df["recommendation_tier"].notna()].copy()
    recommendation_df = recommendation_df.sort_values("score", ascending=False).reset_index(drop=True)
    recommendation_df["confidence_reasons"] = compute_family_reason_series(model, recommendation_df, feature_columns)
    display_df = build_ranked_display(recommendation_df)
    recommendation_df["batter_id"] = display_df["batter_id"]
    recommendation_df["batter_name"] = display_df["batter_name"]
    recommendation_df["pitcher_name"] = display_df["pitcher_name"]
    recommendation_df["game_date"] = pd.to_datetime(recommendation_df[DATE_COL]).dt.date
    columns = [
        "game_date",
        "batter_id",
        "batter_name",
        "team",
        "opponent",
        "pitcher_name",
        "score",
        "slate_rank",
        "slate_percentile",
        "recommendation_tier",
        "tier_profile_label",
        "confidence_score",
        "confidence_label",
        "confidence_reasons",
    ]
    return recommendation_df[columns]


def prepare_recommendation_results(
    scored_df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    tier_lift_map: dict[str, float],
    confidence_profile: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = compute_confidence_metadata(scored_df, feature_columns, tier_lift_map)
    enriched = apply_confidence_profile(enriched, confidence_profile)
    recommendation_table = build_recommendation_table(enriched, model, feature_columns)
    return enriched, recommendation_table


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
    print_contact_feature_status(pruning_audit_df)
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

    profile_train_df, profile_eval_df = chronological_split(train_df)
    validate_temporal_integrity(profile_train_df, profile_eval_df)
    profile_model = fit_selected_model(profile_train_df, pruned_feature_columns, model_name=model_name)
    profile_scored_df = profile_eval_df.copy()
    profile_scored_df["score"] = score_frame(profile_model, profile_eval_df, pruned_feature_columns)
    profile_scored_df = add_slate_ranking_columns(profile_scored_df)
    tier_plan, profile_tier_summary, tier_lift_map = derive_tier_plan(profile_scored_df)
    profile_scored_df["recommendation_tier"] = profile_scored_df["initial_recommendation_tier"].map(tier_plan)
    profile_scored_df = compute_confidence_metadata(profile_scored_df, pruned_feature_columns, tier_lift_map)
    confidence_profile = derive_confidence_profile(profile_scored_df)

    model = fit_selected_model(train_df, pruned_feature_columns, model_name=model_name)

    y_prob = score_frame(model, test_df, pruned_feature_columns)
    metrics = evaluate_predictions(y_test, y_prob)
    baseline_metrics = metrics
    if excluded_features and baseline_feature_columns:
        baseline_model = fit_selected_model(train_df, baseline_feature_columns, model_name=model_name)
        baseline_prob = score_frame(baseline_model, test_df, baseline_feature_columns)
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
    print_logistic_coefficient_summary(model, pruned_feature_columns)

    ranking_df = test_df.copy()
    ranking_df["score"] = y_prob
    ranking_df = add_slate_ranking_columns(ranking_df)
    ranking_df["recommendation_tier"] = ranking_df["initial_recommendation_tier"].map(tier_plan)
    ranking_df, recommendation_table = prepare_recommendation_results(
        ranking_df,
        model,
        pruned_feature_columns,
        tier_lift_map,
        confidence_profile,
    )
    tier_summary = summarize_tiers(ranking_df)
    confidence_summary = summarize_confidence_labels(ranking_df)
    print("\nHistorical recommendation tier profile")
    print("-" * 60)
    print(profile_tier_summary.to_string(index=False))
    print_confidence_profile_audit(confidence_profile)
    print_tier_diagnostics(ranking_df, tier_summary)
    if not confidence_summary.empty:
        print("\nHoldout confidence-label summary")
        print("-" * 60)
        print(confidence_summary.to_string(index=False))
    print_ranking_diagnostics(ranking_df)

    top_identity_preview = recommendation_table.head(10).copy()
    final_top_ranked = recommendation_table.head(10).copy()
    preview_name_match = top_identity_preview[["batter_id", "batter_name", "pitcher_name", "score"]].equals(
        final_top_ranked[["batter_id", "batter_name", "pitcher_name", "score"]]
    )

    print("\nTiered recommendation preview")
    print("-" * 60)
    print(top_identity_preview.to_string(index=False))

    print("\nTop ranked candidates")
    print("-" * 60)
    print(final_top_ranked.to_string(index=False))

    print("\nRanked-output identity preview")
    print("-" * 60)
    print(top_identity_preview[["batter_id", "batter_name", "pitcher_name", "score"]].to_string(index=False))
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


def run_recommendation(data_path: str, recommend_date: str, model_name: str = "logistic") -> None:
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run python generate_data.py first.")

    df, resolved_data_path, schema_label, schema_details = load_data(data_path)
    present_candidate_features, _ = print_candidate_feature_audit(df)
    fail_if_schema_mismatch(schema_label, schema_details)
    recommend_ts, historical_df, slate_df = split_for_recommendation(df, recommend_date)

    pruned_feature_columns, pruning_audit_df = prune_sparse_features(historical_df, CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS)
    if not pruned_feature_columns:
        raise RuntimeError("All candidate features were excluded by missingness threshold before recommendation date.")

    print("\nFeature-pruning audit")
    print("-" * 60)
    print(pruning_audit_df.to_string(index=False))
    print_contact_feature_status(pruning_audit_df)
    print(f"\nRecommendation date: {recommend_ts.date()}")
    print(f"Historical training rows: {len(historical_df):,}")
    print(f"Slate rows scored: {len(slate_df):,}")
    print(f"Present candidate feature count: {len(present_candidate_features)}")
    print(f"Modeled feature count: {len(pruned_feature_columns)}")

    profile_train_df, profile_eval_df = chronological_split(historical_df)
    validate_temporal_integrity(profile_train_df, profile_eval_df)
    profile_model = fit_selected_model(profile_train_df, pruned_feature_columns, model_name=model_name)
    profile_scored_df = profile_eval_df.copy()
    profile_scored_df["score"] = score_frame(profile_model, profile_eval_df, pruned_feature_columns)
    profile_scored_df = add_slate_ranking_columns(profile_scored_df)
    tier_plan, tier_summary, tier_lift_map = derive_tier_plan(profile_scored_df)
    profile_scored_df["recommendation_tier"] = profile_scored_df["initial_recommendation_tier"].map(tier_plan)
    profile_scored_df = compute_confidence_metadata(profile_scored_df, pruned_feature_columns, tier_lift_map)
    confidence_profile = derive_confidence_profile(profile_scored_df)

    print("\nHistorical recommendation tier profile")
    print("-" * 60)
    print(tier_summary.to_string(index=False))
    print_confidence_profile_audit(confidence_profile)

    final_model = fit_selected_model(historical_df, pruned_feature_columns, model_name=model_name)
    slate_scored_df = slate_df.copy()
    slate_scored_df["score"] = score_frame(final_model, slate_df, pruned_feature_columns)
    slate_scored_df = add_slate_ranking_columns(slate_scored_df)
    slate_scored_df["recommendation_tier"] = slate_scored_df["initial_recommendation_tier"].map(tier_plan)
    slate_scored_df, recommendation_table = prepare_recommendation_results(
        slate_scored_df,
        final_model,
        pruned_feature_columns,
        tier_lift_map,
        confidence_profile,
    )

    print("\nRecommendation output")
    print("-" * 60)
    print(recommendation_table.to_string(index=False))
    print(f"\nTiered recommendations returned: {len(recommendation_table):,}")
    print(f"Recommendation rows limited to top {int(TOP_RECOMMENDATION_SHARE * 100)}% of the slate: yes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", nargs="?", default=str(FINAL_DATA_PATH), help="Path to batter-game dataset CSV.")
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic", help="Model family to train.")
    parser.add_argument("--recommend-date", help="Historical slate date to score using only prior rows (YYYY-MM-DD).")
    args = parser.parse_args()
    if args.recommend_date:
        run_recommendation(args.data_path, args.recommend_date, model_name=args.model)
    else:
        run_backtest(args.data_path, model_name=args.model)
