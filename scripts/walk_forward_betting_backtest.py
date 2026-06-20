#!/usr/bin/env python
"""Walk-forward MLB HR prediction and betting audit.

This script intentionally does not invent historical odds. Without an odds CSV it
reports prediction-quality metrics only. With real odds it adds flat-bet edge and
fractional-Kelly betting results.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_model import (  # noqa: E402
    DATE_COL,
    DEFAULT_CONFIDENCE_POLICY,
    LIVE_PRODUCTION_FEATURE_COLUMNS,
    LIVE_USABLE_CANDIDATE_V2_PROFILE,
    LIVE_USABLE_CANDIDATE_V3_PROFILE,
    PREGAME_SAFE_PROFILE,
    TARGET_COL,
    apply_confidence_policy_to_frame,
    feature_columns_for_profile,
    load_data,
    prepare_feature_matrix,
)

DEFAULT_DATASET_PATH = Path("data/live/model_training_dataset.csv")
DEFAULT_OUTPUT_PATH = Path("data/live/walk_forward_betting_audit.json")
DEFAULT_PROFILES = [PREGAME_SAFE_PROFILE, "live"]
TOP_K_VALUES = [1, 3, 5, 10]
PROBABILITY_THRESHOLDS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
TIER_STRATEGIES = ["elite", "strong_or_better", "watch_or_better"]
MIN_EDGE_VALUES = [0.00, 0.02, 0.05, 0.08]
TIER_ORDER = {"longshot": 0, "watch": 1, "strong": 2, "elite": 3}
DEFAULT_POSITIVE_CALL_MIN_BETS = 25
DEFAULT_POSITIVE_CALL_MIN_DAYS = 10
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SEED = 42
HISTORICAL_CONTEXT_FEATURES = {
    "temperature_f",
    "wind_speed_mph",
    "humidity_pct",
    "wind_direction_deg",
    "pressure_hpa",
    "roofed_park",
    "platoon_advantage",
    "expected_pa_today",
    "batting_order_slot",
    "lineup_confirmation_score",
    "projected_lineup_rank",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--odds-path", type=Path, default=None, help="Optional real historical odds CSV.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--models", nargs="+", default=["logistic", "histgb"], choices=["logistic", "histgb"])
    parser.add_argument("--min-train-days", type=int, default=45)
    parser.add_argument("--max-test-days", type=int, default=None)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--kelly-max-stake", type=float, default=0.02)
    parser.add_argument("--calibration-buckets", type=int, default=10)
    parser.add_argument("--train-years", nargs="+", type=int, default=None)
    parser.add_argument("--selection-years", nargs="+", type=int, default=None)
    parser.add_argument("--test-years", nargs="+", type=int, default=None)
    parser.add_argument("--train-start-date", type=str, default=None)
    parser.add_argument("--train-end-date", type=str, default=None)
    parser.add_argument("--test-start-date", type=str, default=None)
    parser.add_argument("--test-end-date", type=str, default=None)
    parser.add_argument("--positive-call-min-bets", type=int, default=DEFAULT_POSITIVE_CALL_MIN_BETS)
    parser.add_argument("--positive-call-min-days", type=int, default=DEFAULT_POSITIVE_CALL_MIN_DAYS)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def model_specs(model_names: Iterable[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name in model_names:
        if name == "logistic":
            specs.append(
                ModelSpec(
                    name="logistic",
                    estimator=Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            (
                                "clf",
                                LogisticRegression(
                                    C=0.1,
                                    class_weight=None,
                                    max_iter=2000,
                                    solver="lbfgs",
                                    random_state=42,
                                ),
                            ),
                        ]
                    ),
                )
            )
        elif name == "histgb":
            specs.append(
                ModelSpec(
                    name="histgb",
                    estimator=Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "clf",
                                HistGradientBoostingClassifier(
                                    learning_rate=0.03,
                                    max_iter=150,
                                    max_depth=3,
                                    max_leaf_nodes=31,
                                    min_samples_leaf=50,
                                    l2_regularization=1.0,
                                    random_state=42,
                                ),
                            ),
                        ]
                    ),
                )
            )
    return specs


def available_profile_features(df: pd.DataFrame, profile: str) -> list[str]:
    return [column for column in feature_columns_for_profile(profile) if column in df.columns]


def audit_profile_as_of_safety(df: pd.DataFrame, profile: str) -> dict[str, Any]:
    feature_columns = available_profile_features(df, profile)
    findings: list[dict[str, str]] = []
    pitcher_features = [column for column in feature_columns if column.startswith("pitcher_")]
    context_features = [column for column in feature_columns if column in HISTORICAL_CONTEXT_FEATURES]
    if pitcher_features and not any(column in df.columns for column in ["pitcher_snapshot_at", "probable_pitcher_as_of"]):
        findings.append(
            {
                "severity": "high",
                "code": "historical_actual_pitcher_without_as_of_snapshot",
                "detail": "Historical matchup features use the realized primary pitcher and lack a pregame probable-pitcher timestamp.",
            }
        )
    weather_features = [column for column in context_features if column in {"temperature_f", "wind_speed_mph", "humidity_pct", "wind_direction_deg", "pressure_hpa", "roofed_park"}]
    if weather_features and not any(column in df.columns for column in ["weather_forecast_as_of", "weather_snapshot_at"]):
        findings.append(
            {
                "severity": "high",
                "code": "historical_observed_weather_without_forecast_snapshot",
                "detail": "Historical weather features are observations and lack the pregame forecast snapshot available to live scoring.",
            }
        )
    opportunity_features = [column for column in context_features if column in {"expected_pa_today", "batting_order_slot", "lineup_confirmation_score", "projected_lineup_rank"}]
    if opportunity_features and not any(column in df.columns for column in ["lineup_snapshot_at", "lineup_as_of"]):
        findings.append(
            {
                "severity": "high",
                "code": "historical_lineup_without_as_of_snapshot",
                "detail": "Historical lineup/opportunity features lack an as-of timestamp proving pregame availability.",
            }
        )
    if "platoon_advantage" in feature_columns and not any(column in df.columns for column in ["pitcher_snapshot_at", "probable_pitcher_as_of"]):
        findings.append(
            {
                "severity": "high",
                "code": "platoon_feature_uses_unversioned_pitcher_identity",
                "detail": "Platoon advantage inherits the historical realized-pitcher risk.",
            }
        )
    return {
        "profile": profile,
        "feature_columns": feature_columns,
        "release_eligible": not any(row["severity"] == "high" for row in findings),
        "findings": findings,
        "safe_feature_evidence": "Batter rolling features are closed-left/shifted and covered by leakage tests." if profile == PREGAME_SAFE_PROFILE else None,
    }


def american_odds_to_implied_probability(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    raise ValueError("American odds cannot be zero.")


def american_odds_profit_per_unit(odds: float) -> float:
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    raise ValueError("American odds cannot be zero.")


def load_odds_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    odds = pd.read_csv(path)
    required = {"game_date", "batter_id", "american_odds"}
    missing = sorted(required - set(odds.columns))
    if missing:
        raise ValueError(f"Odds CSV is missing required columns: {missing}")
    odds = odds.copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"], errors="coerce").dt.date.astype(str)
    odds["batter_id"] = pd.to_numeric(odds["batter_id"], errors="coerce").astype("Int64")
    odds["american_odds"] = pd.to_numeric(odds["american_odds"], errors="coerce")
    odds = odds.dropna(subset=["game_date", "batter_id", "american_odds"])
    if "game_pk" in odds.columns:
        odds["game_pk"] = pd.to_numeric(odds["game_pk"], errors="coerce").astype("Int64")
        key_cols = ["game_date", "game_pk", "batter_id"]
    else:
        key_cols = ["game_date", "batter_id"]
    odds = odds.drop_duplicates(key_cols, keep="last")
    odds["implied_probability"] = odds["american_odds"].map(american_odds_to_implied_probability)
    odds["profit_per_unit"] = odds["american_odds"].map(american_odds_profit_per_unit)
    keep_cols = [*key_cols, "american_odds", "implied_probability", "profit_per_unit"]
    optional_cols = [column for column in ["book", "sportsbook", "odds_timestamp"] if column in odds.columns]
    return odds[keep_cols + optional_cols].copy()


def join_odds(predictions: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    frame = predictions.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.date.astype(str)
    frame["batter_id"] = pd.to_numeric(frame["batter_id"], errors="coerce").astype("Int64")
    if odds is None:
        frame["american_odds"] = np.nan
        frame["implied_probability"] = np.nan
        frame["profit_per_unit"] = np.nan
        return frame
    if "game_pk" in odds.columns and "game_pk" in frame.columns:
        frame["game_pk"] = pd.to_numeric(frame["game_pk"], errors="coerce").astype("Int64")
        return frame.merge(odds, on=["game_date", "game_pk", "batter_id"], how="left", validate="many_to_one")
    return frame.merge(odds, on=["game_date", "batter_id"], how="left", validate="many_to_one")


def safe_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "average_precision": None,
        "roc_auc": None,
        "log_loss": None,
        "brier_score": None,
    }
    if len(y_true) == 0:
        return metrics
    labels = [0, 1]
    metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    if len(np.unique(y_true)) >= 2:
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def calibration_table(frame: pd.DataFrame, probability_col: str, bucket_count: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    if bucket_count < 2:
        raise ValueError("--calibration-buckets must be >= 2.")
    source = frame[[probability_col, "actual_hit_hr"]].dropna().copy()
    if source.empty:
        return []
    q = min(bucket_count, len(source))
    source["bucket"] = pd.qcut(source[probability_col], q=q, labels=False, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for bucket, group in source.groupby("bucket", sort=True):
        rows.append(
            {
                "bucket": int(bucket),
                "count": int(len(group)),
                "probability_min": float(group[probability_col].min()),
                "probability_max": float(group[probability_col].max()),
                "predicted_mean": float(group[probability_col].mean()),
                "actual_rate": float(group["actual_hit_hr"].mean()),
                "calibration_error": float(group[probability_col].mean() - group["actual_hit_hr"].mean()),
                "absolute_calibration_error": float(abs(group[probability_col].mean() - group["actual_hit_hr"].mean())),
            }
        )
    return rows


def month_key(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m")


def _normalize_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Invalid date value: {value}")
    return pd.Timestamp(timestamp).normalize()


def _year_mask(series: pd.Series, years: list[int] | None) -> pd.Series:
    if not years:
        return pd.Series(True, index=series.index)
    return pd.to_datetime(series, errors="coerce").dt.year.isin(sorted(set(int(year) for year in years)))


def _date_mask(
    series: pd.Series,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce").dt.normalize()
    mask = pd.Series(True, index=series.index)
    if start is not None:
        mask &= timestamps >= start
    if end is not None:
        mask &= timestamps <= end
    return mask


def filter_source_for_walk_forward(
    df: pd.DataFrame,
    *,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp], dict[str, Any]]:
    source = df.sort_values([DATE_COL, "game_pk", "batter_id"]).reset_index(drop=True).copy()
    source_dates = pd.to_datetime(source[DATE_COL], errors="coerce").dt.normalize()
    available_years = sorted(source_dates.dropna().dt.year.unique().tolist())

    explicit_window = any(
        value is not None
        for value in [train_years, test_years, train_start_date, train_end_date, test_start_date, test_end_date]
    )
    default_latest_year_window = False
    if not explicit_window and len(available_years) >= 2:
        train_years = available_years[:-1]
        test_years = [available_years[-1]]
        explicit_window = True
        default_latest_year_window = True

    train_start = _normalize_timestamp(train_start_date)
    train_end = _normalize_timestamp(train_end_date)
    test_start = _normalize_timestamp(test_start_date)
    test_end = _normalize_timestamp(test_end_date)
    if train_start is not None and train_end is not None and train_start > train_end:
        raise ValueError("train_start_date cannot be after train_end_date.")
    if test_start is not None and test_end is not None and test_start > test_end:
        raise ValueError("test_start_date cannot be after test_end_date.")

    if explicit_window:
        train_mask = _year_mask(source[DATE_COL], train_years) & _date_mask(source[DATE_COL], start=train_start, end=train_end)
        test_mask = _year_mask(source[DATE_COL], test_years) & _date_mask(source[DATE_COL], start=test_start, end=test_end)
        if not test_mask.any():
            raise RuntimeError("No rows matched the requested walk-forward test window.")
        union_mask = train_mask | test_mask
        source = source.loc[union_mask].copy()
        source_dates = pd.to_datetime(source[DATE_COL], errors="coerce").dt.normalize()
        test_dates = sorted(source_dates.loc[test_mask.loc[source.index]].dropna().unique().tolist())
        window_summary = {
            "train_years": sorted(set(train_years or [])),
            "test_years": sorted(set(test_years or [])),
            "train_start_date": str(train_start.date()) if train_start is not None else None,
            "train_end_date": str(train_end.date()) if train_end is not None else None,
            "test_start_date": str(test_start.date()) if test_start is not None else None,
            "test_end_date": str(test_end.date()) if test_end is not None else None,
            "available_years": available_years,
            "window_mode": "default_latest_year" if default_latest_year_window else "explicit",
        }
        return source.reset_index(drop=True), test_dates, window_summary

    return source, [], {
        "train_years": [],
        "test_years": [],
        "train_start_date": None,
        "train_end_date": None,
        "test_start_date": None,
        "test_end_date": None,
        "available_years": available_years,
        "window_mode": "rolling_min_train_days",
    }


def max_drawdown(profits: Iterable[float]) -> float:
    peak = 0.0
    equity = 0.0
    worst = 0.0
    for profit in profits:
        equity += float(profit)
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return float(worst)


def daily_cluster_bootstrap_hit_rate(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    source = frame[["game_date", "actual_hit_hr"]].dropna().copy()
    source["game_date"] = pd.to_datetime(source["game_date"], errors="coerce").dt.date.astype(str)
    grouped = {date: group["actual_hit_hr"].to_numpy(dtype=float) for date, group in source.groupby("game_date")}
    dates = sorted(grouped)
    if not dates or iterations <= 0:
        return {"method": "daily_cluster_bootstrap", "confidence": confidence, "iterations": 0, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        sampled_values = np.concatenate([grouped[str(date)] for date in sampled_dates])
        if len(sampled_values):
            estimates.append(float(sampled_values.mean()))
    if not estimates:
        return {"method": "daily_cluster_bootstrap", "confidence": confidence, "iterations": 0, "lower": None, "upper": None}
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "daily_cluster_bootstrap",
        "confidence": confidence,
        "iterations": int(len(estimates)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
    }


def flat_profit(row: pd.Series) -> float | None:
    if pd.isna(row.get("profit_per_unit")):
        return None
    return float(row["profit_per_unit"]) if int(row["actual_hit_hr"]) == 1 else -1.0


def kelly_stake_fraction(row: pd.Series, *, fraction: float, max_stake: float) -> float | None:
    if pd.isna(row.get("profit_per_unit")) or pd.isna(row.get("implied_probability")):
        return None
    p = float(row["predicted_hr_probability"])
    b = float(row["profit_per_unit"])
    q = 1.0 - p
    raw = (b * p - q) / b
    return float(max(0.0, min(max_stake, fraction * raw)))


def summarize_selection(
    frame: pd.DataFrame,
    *,
    strategy_name: str,
    odds_available: bool,
    kelly_fraction: float,
    kelly_max_stake: float,
    total_actual_positives: int,
    total_population: int,
) -> dict[str, Any]:
    selected = frame.copy()
    selected = selected.sort_values(["game_date", "slate_rank", "batter_name"]).reset_index(drop=True)
    bets = int(len(selected))
    wins = int(selected["actual_hit_hr"].sum()) if bets else 0
    false_positives = int(bets - wins)
    slate_days = int(pd.to_datetime(selected["game_date"], errors="coerce").dt.normalize().nunique()) if bets else 0
    result: dict[str, Any] = {
        "strategy": strategy_name,
        "bets": bets,
        "wins": wins,
        "losses": int(bets - wins),
        "false_positives": false_positives,
        "slate_days": slate_days,
        "precision": float(wins / bets) if bets else None,
        "hit_rate": float(wins / bets) if bets else None,
        "recall": float(wins / total_actual_positives) if total_actual_positives > 0 else None,
        "positive_prediction_rate": float(bets / total_population) if total_population > 0 else None,
        "avg_predicted_probability": float(selected["predicted_hr_probability"].mean()) if bets else None,
        "average_precision": safe_probability_metrics(
            selected["actual_hit_hr"].to_numpy(dtype=int),
            selected["predicted_hr_probability"].to_numpy(dtype=float),
        )["average_precision"] if bets else None,
        "roi": None,
        "pnl_units": None,
        "max_drawdown_units": None,
        "kelly_roi": None,
        "kelly_pnl_units": None,
        "kelly_max_drawdown_units": None,
        "performance_by_month": [],
    }
    if not odds_available or selected["american_odds"].notna().sum() == 0:
        return result

    selected = selected[selected["american_odds"].notna()].copy()
    if selected.empty:
        return result
    selected["flat_profit"] = selected.apply(flat_profit, axis=1)
    selected["kelly_stake"] = selected.apply(
        lambda row: kelly_stake_fraction(row, fraction=kelly_fraction, max_stake=kelly_max_stake),
        axis=1,
    )
    selected["kelly_profit"] = np.where(
        selected["actual_hit_hr"].eq(1),
        selected["kelly_stake"] * selected["profit_per_unit"],
        -selected["kelly_stake"],
    )
    flat_bets = int(len(selected))
    result["bets_with_odds"] = flat_bets
    result["roi"] = float(selected["flat_profit"].sum() / flat_bets) if flat_bets else None
    result["pnl_units"] = float(selected["flat_profit"].sum())
    result["max_drawdown_units"] = max_drawdown(selected["flat_profit"].fillna(0.0))
    total_kelly_staked = float(selected["kelly_stake"].sum())
    result["kelly_roi"] = float(selected["kelly_profit"].sum() / total_kelly_staked) if total_kelly_staked > 0 else None
    result["kelly_pnl_units"] = float(selected["kelly_profit"].sum())
    result["kelly_max_drawdown_units"] = max_drawdown(selected["kelly_profit"].fillna(0.0))
    month_rows = []
    selected["month"] = selected["game_date"].map(month_key)
    for month, group in selected.groupby("month", sort=True):
        if not month:
            continue
        month_rows.append(
            {
                "month": month,
                "bets": int(len(group)),
                "wins": int(group["actual_hit_hr"].sum()),
                "hit_rate": float(group["actual_hit_hr"].mean()),
                "pnl_units": float(group["flat_profit"].sum()),
                "roi": float(group["flat_profit"].sum() / len(group)),
                "max_drawdown_units": max_drawdown(group["flat_profit"].fillna(0.0)),
            }
        )
    result["performance_by_month"] = month_rows
    return result


def strategy_frames(predictions: pd.DataFrame, odds_available: bool) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    ranked = predictions.sort_values(["game_date", "slate_rank"]).copy()
    for k in TOP_K_VALUES:
        frames.append((f"top_{k}_daily", ranked[ranked["slate_rank"] <= k].copy()))
    for threshold in PROBABILITY_THRESHOLDS:
        frames.append((f"probability_at_least_{threshold:.2f}", ranked[ranked["predicted_hr_probability"] >= threshold].copy()))
    for tier_name in TIER_STRATEGIES:
        if tier_name == "elite":
            min_value = TIER_ORDER["elite"]
        elif tier_name == "strong_or_better":
            min_value = TIER_ORDER["strong"]
        else:
            min_value = TIER_ORDER["watch"]
        frames.append(
            (
                f"tier_{tier_name}",
                ranked[ranked["confidence_tier"].map(lambda item: TIER_ORDER.get(str(item), -1)) >= min_value].copy(),
            )
        )
    if odds_available and "implied_probability" in ranked.columns:
        ranked = ranked.copy()
        ranked["model_edge"] = ranked["predicted_hr_probability"] - ranked["implied_probability"]
        for edge in MIN_EDGE_VALUES:
            frames.append((f"min_edge_{edge:.2f}", ranked[ranked["model_edge"] >= edge].copy()))
    return frames


def predict_walk_forward(
    df: pd.DataFrame,
    *,
    profile: str,
    spec: ModelSpec,
    min_train_days: int,
    max_test_days: int | None,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    test_start_date: str | None = None,
    test_end_date: str | None = None,
) -> pd.DataFrame:
    feature_columns = available_profile_features(df, profile)
    if not feature_columns:
        raise RuntimeError(f"No available features for profile {profile}.")
    source, explicit_test_dates, _window_summary = filter_source_for_walk_forward(
        df,
        train_years=train_years,
        test_years=test_years,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
    )
    unique_dates = sorted(pd.to_datetime(source[DATE_COL]).dt.normalize().dropna().unique())
    if explicit_test_dates:
        test_dates = explicit_test_dates
    else:
        if len(unique_dates) <= min_train_days:
            raise RuntimeError(
                f"Dataset has {len(unique_dates)} dates, fewer than required min_train_days={min_train_days}."
            )
        test_dates = unique_dates[min_train_days:]
    if max_test_days is not None:
        test_dates = test_dates[:max_test_days]
    rows: list[pd.DataFrame] = []
    for test_date in test_dates:
        train_df = source[pd.to_datetime(source[DATE_COL]).dt.normalize() < test_date].copy()
        test_df = source[pd.to_datetime(source[DATE_COL]).dt.normalize() == test_date].copy()
        if train_df.empty or test_df.empty or train_df[TARGET_COL].nunique() < 2:
            continue
        if explicit_test_dates:
            prior_train_days = pd.to_datetime(train_df[DATE_COL]).dt.normalize().nunique()
            if prior_train_days < min_train_days:
                continue
        model = clone(spec.estimator)
        X_train = prepare_feature_matrix(train_df, feature_columns)
        y_train = train_df[TARGET_COL].to_numpy(dtype=int)
        X_test = prepare_feature_matrix(test_df, feature_columns)
        model.fit(X_train, y_train)
        scored = test_df.copy()
        scored["predicted_hr_probability"] = model.predict_proba(X_test)[:, 1]
        scored["model_name"] = spec.name
        scored["feature_profile"] = profile
        scored["feature_count"] = len(feature_columns)
        scored["trained_through"] = str((pd.Timestamp(test_date) - pd.Timedelta(days=1)).date())
        scored["actual_hit_hr"] = scored[TARGET_COL].astype(int)
        scored = scored.sort_values(["predicted_hr_probability", "batter_name"], ascending=[False, True]).reset_index(drop=True)
        scored["slate_rank"] = np.arange(1, len(scored) + 1)
        scored = apply_confidence_policy_to_frame(
            scored,
            probability_col="predicted_hr_probability",
            date_col=DATE_COL,
            policy=DEFAULT_CONFIDENCE_POLICY,
            percentile_col="slate_percentile",
            rank_col="policy_rank",
            tier_col="confidence_tier",
        )
        rows.append(
            scored[
                [
                    DATE_COL,
                    "game_pk",
                    "batter_id",
                    "batter_name",
                    "team",
                    "opponent",
                    "pitcher_id",
                    "opp_pitcher_id",
                    "predicted_hr_probability",
                    "actual_hit_hr",
                    "slate_rank",
                    "confidence_tier",
                    "model_name",
                    "feature_profile",
                    "feature_count",
                    "trained_through",
                ]
            ].rename(columns={DATE_COL: "game_date"})
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    odds_available: bool,
    kelly_fraction: float,
    kelly_max_stake: float,
    calibration_buckets: int,
    positive_call_min_bets: int,
    positive_call_min_days: int,
    fixed_positive_call_strategy: str | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    y_true = predictions["actual_hit_hr"].to_numpy(dtype=int)
    y_prob = predictions["predicted_hr_probability"].to_numpy(dtype=float)
    total_actual_positives = int(np.sum(y_true))
    total_population = int(len(predictions))
    strategy_frame_rows = strategy_frames(predictions, odds_available)
    strategy_frame_map = {name: frame for name, frame in strategy_frame_rows}
    strategy_results = [
        summarize_selection(
            frame,
            strategy_name=name,
            odds_available=odds_available,
            kelly_fraction=kelly_fraction,
            kelly_max_stake=kelly_max_stake,
            total_actual_positives=total_actual_positives,
            total_population=total_population,
        )
        for name, frame in strategy_frame_rows
    ]
    threshold_rows = [
        row for row in strategy_results if str(row["strategy"]).startswith("probability_at_least_")
    ]
    eligible_threshold_rows = [
        row
        for row in threshold_rows
        if int(row.get("bets") or 0) >= positive_call_min_bets
        and int(row.get("slate_days") or 0) >= positive_call_min_days
    ]
    if fixed_positive_call_strategy is not None:
        positive_call_choice = next(
            (row for row in threshold_rows if row["strategy"] == fixed_positive_call_strategy),
            None,
        )
        if positive_call_choice is None:
            raise ValueError(f"Frozen positive-call strategy was not evaluated: {fixed_positive_call_strategy}")
        threshold_source = "frozen_from_selection_period"
    elif eligible_threshold_rows:
        positive_call_choice = sorted(
            eligible_threshold_rows,
            key=lambda row: (
                -float(row.get("precision") or 0.0),
                -float(row.get("recall") or 0.0),
                -int(row.get("bets") or 0),
                -float(str(row["strategy"]).replace("probability_at_least_", "")),
            ),
        )[0]
        threshold_source = "selected_on_selection_period"
    else:
        positive_call_choice = None
        threshold_source = "rejected_insufficient_support"
    top3 = next((row for row in strategy_results if row["strategy"] == "top_3_daily"), {})
    top10 = next((row for row in strategy_results if row["strategy"] == "top_10_daily"), {})
    top_k_confidence_intervals = {
        str(k): daily_cluster_bootstrap_hit_rate(
            strategy_frame_map[f"top_{k}_daily"],
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + k,
        )
        for k in TOP_K_VALUES
    }
    positive_call_interval = (
        daily_cluster_bootstrap_hit_rate(
            strategy_frame_map[str(positive_call_choice["strategy"])],
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        if positive_call_choice is not None
        else {"method": "daily_cluster_bootstrap", "confidence": 0.95, "iterations": 0, "lower": None, "upper": None}
    )
    monthly_rows: list[dict[str, Any]] = []
    month_frame = predictions.copy()
    month_frame["month"] = month_frame["game_date"].map(month_key)
    for month, group in month_frame.groupby("month", sort=True):
        if not month:
            continue
        top3_month = group[group["slate_rank"] <= 3]
        metrics = safe_probability_metrics(
            group["actual_hit_hr"].to_numpy(dtype=int),
            group["predicted_hr_probability"].to_numpy(dtype=float),
        )
        monthly_rows.append(
            {
                "month": month,
                "rows": int(len(group)),
                "base_hr_rate": float(group["actual_hit_hr"].mean()),
                "average_precision": metrics.get("average_precision"),
                "log_loss": metrics.get("log_loss"),
                "brier_score": metrics.get("brier_score"),
                "top_3_daily_hit_rate": float(top3_month["actual_hit_hr"].mean()) if len(top3_month) else None,
                "top_3_daily_bets": int(len(top3_month)),
            }
        )
    return {
        "row_count": int(len(predictions)),
        "date_min": str(pd.to_datetime(predictions["game_date"]).min().date()) if len(predictions) else None,
        "date_max": str(pd.to_datetime(predictions["game_date"]).max().date()) if len(predictions) else None,
        "base_hr_rate": float(np.mean(y_true)) if len(y_true) else None,
        "probability_metrics": safe_probability_metrics(y_true, y_prob),
        "calibration_buckets": calibration_table(predictions, "predicted_hr_probability", calibration_buckets),
        "probability_bucket_performance": calibration_table(predictions, "predicted_hr_probability", calibration_buckets),
        "strategy_results": strategy_results,
        "positive_call_summary": {
            "metric": "precision_first",
            "minimum_bets_required": int(positive_call_min_bets),
            "minimum_slate_days_required": int(positive_call_min_days),
            "threshold_source": threshold_source,
            "selected_threshold_strategy": None if positive_call_choice is None else positive_call_choice["strategy"],
            "precision": None if positive_call_choice is None else positive_call_choice.get("precision"),
            "precision_confidence_interval": positive_call_interval,
            "recall": None if positive_call_choice is None else positive_call_choice.get("recall"),
            "bets": None if positive_call_choice is None else positive_call_choice.get("bets"),
            "slate_days": None if positive_call_choice is None else positive_call_choice.get("slate_days"),
            "false_positives": None if positive_call_choice is None else positive_call_choice.get("false_positives"),
            "positive_prediction_rate": None if positive_call_choice is None else positive_call_choice.get("positive_prediction_rate"),
        },
        "monthly_prediction_metrics": monthly_rows,
        "ranking_summary": {
            "top_k_hit_rate_confidence_intervals": top_k_confidence_intervals,
            "top_3_daily_hit_rate": top3.get("hit_rate"),
            "top_3_daily_bets": top3.get("bets"),
            "top_10_daily_hit_rate": top10.get("hit_rate"),
            "top_10_daily_bets": top10.get("bets"),
        },
    }


def rank_model_reports(reports: list[dict[str, Any]], odds_available: bool) -> list[dict[str, Any]]:
    ranked = sorted(
        reports,
        key=lambda row: (
            -int(bool(row.get("leakage_audit", {}).get("release_eligible", False))),
            -float(row.get("selection_evaluation", row["evaluation"])["positive_call_summary"].get("precision") or 0.0),
            -float(row.get("selection_evaluation", row["evaluation"])["ranking_summary"].get("top_3_daily_hit_rate") or 0.0),
            -float(row.get("selection_evaluation", row["evaluation"])["probability_metrics"].get("average_precision") or 0.0),
            float(row.get("selection_evaluation", row["evaluation"])["probability_metrics"].get("brier_score") or 999.0),
            float(row.get("selection_evaluation", row["evaluation"])["probability_metrics"].get("log_loss") or 999.0),
        ),
    )
    output = []
    for rank, row in enumerate(ranked, start=1):
        evaluation = row.get("selection_evaluation", row["evaluation"])
        output.append(
            {
                "rank": rank,
                "model_name": row["model_name"],
                "feature_profile": row["feature_profile"],
                "release_eligible": bool(row.get("leakage_audit", {}).get("release_eligible", False)),
                "positive_call_threshold": evaluation["positive_call_summary"].get("selected_threshold_strategy"),
                "positive_call_precision": evaluation["positive_call_summary"].get("precision"),
                "positive_call_bets": evaluation["positive_call_summary"].get("bets"),
                "average_precision": evaluation["probability_metrics"].get("average_precision"),
                "log_loss": evaluation["probability_metrics"].get("log_loss"),
                "brier_score": evaluation["probability_metrics"].get("brier_score"),
                "top_3_daily_hit_rate": evaluation["ranking_summary"].get("top_3_daily_hit_rate"),
                "top_3_daily_bets": evaluation["ranking_summary"].get("top_3_daily_bets"),
                "ranking_period": "selection",
            }
        )
    return output


def resolve_selection_and_final_years(
    df: pd.DataFrame,
    *,
    train_years: list[int] | None,
    selection_years: list[int] | None,
    test_years: list[int] | None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    available_years = sorted(pd.to_datetime(df[DATE_COL], errors="coerce").dt.year.dropna().astype(int).unique().tolist())
    if len(available_years) < 3 and (selection_years is None or train_years is None or test_years is None):
        raise ValueError("A leakage-safe selection/final audit requires at least three seasons or explicit train, selection, and test years.")
    resolved_test_years = sorted(set(test_years or [available_years[-1]]))
    resolved_history_years = sorted(set(train_years or [year for year in available_years if year < min(resolved_test_years)]))
    resolved_selection_years = sorted(set(selection_years or [resolved_history_years[-1]]))
    initial_train_years = [year for year in resolved_history_years if year < min(resolved_selection_years)]
    if not initial_train_years:
        raise ValueError("Selection walk-forward has no earlier training season. Provide at least one train year before the selection year.")
    if set(resolved_selection_years) & set(resolved_test_years):
        raise ValueError("Selection years and final test years must not overlap.")
    if max(resolved_history_years) >= min(resolved_test_years):
        raise ValueError("All historical train years must precede the final test years.")
    return initial_train_years, resolved_selection_years, resolved_history_years, resolved_test_years


def main() -> None:
    args = parse_args()
    df = load_data(str(args.dataset_path))
    odds = load_odds_csv(args.odds_path)
    odds_available = odds is not None

    initial_train_years, selection_years, final_train_years, final_test_years = resolve_selection_and_final_years(
        df,
        train_years=args.train_years,
        selection_years=args.selection_years,
        test_years=args.test_years,
    )
    reports: list[dict[str, Any]] = []
    for profile in args.profiles:
        for spec in model_specs(args.models):
            selection_predictions = predict_walk_forward(
                df,
                profile=profile,
                spec=spec,
                min_train_days=args.min_train_days,
                max_test_days=args.max_test_days,
                train_years=initial_train_years,
                test_years=selection_years,
                train_start_date=args.train_start_date,
                train_end_date=args.train_end_date,
            )
            if selection_predictions.empty:
                continue
            selection_predictions = join_odds(selection_predictions, odds)
            leakage_audit = audit_profile_as_of_safety(df, profile)
            selection_evaluation = evaluate_predictions(
                selection_predictions,
                odds_available=odds_available,
                kelly_fraction=args.kelly_fraction,
                kelly_max_stake=args.kelly_max_stake,
                calibration_buckets=args.calibration_buckets,
                positive_call_min_bets=args.positive_call_min_bets,
                positive_call_min_days=args.positive_call_min_days,
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            )
            frozen_strategy = selection_evaluation["positive_call_summary"].get("selected_threshold_strategy")
            if frozen_strategy is None:
                continue
            final_predictions = predict_walk_forward(
                df,
                profile=profile,
                spec=spec,
                min_train_days=args.min_train_days,
                max_test_days=args.max_test_days,
                train_years=final_train_years,
                test_years=final_test_years,
                train_start_date=args.train_start_date,
                train_end_date=args.train_end_date,
                test_start_date=args.test_start_date,
                test_end_date=args.test_end_date,
            )
            if final_predictions.empty:
                continue
            final_predictions = join_odds(final_predictions, odds)
            reports.append(
                {
                    "model_name": spec.name,
                    "feature_profile": profile,
                    "feature_count": int(final_predictions["feature_count"].iloc[0]),
                    "predictions_with_odds": int(final_predictions["american_odds"].notna().sum()),
                    "leakage_audit": leakage_audit,
                    "selection_evaluation": selection_evaluation,
                    "evaluation": evaluate_predictions(
                        final_predictions,
                        odds_available=odds_available,
                        kelly_fraction=args.kelly_fraction,
                        kelly_max_stake=args.kelly_max_stake,
                        calibration_buckets=args.calibration_buckets,
                        positive_call_min_bets=args.positive_call_min_bets,
                        positive_call_min_days=args.positive_call_min_days,
                        fixed_positive_call_strategy=str(frozen_strategy),
                        bootstrap_iterations=args.bootstrap_iterations,
                        bootstrap_seed=args.bootstrap_seed,
                    ),
                }
            )

    if not reports:
        raise RuntimeError("No walk-forward model reports were produced.")

    rankings = rank_model_reports(reports, odds_available)
    selected_ranking = rankings[0]
    selected_report = next(
        row
        for row in reports
        if row["model_name"] == selected_ranking["model_name"]
        and row["feature_profile"] == selected_ranking["feature_profile"]
    )
    report = {
        "inputs": {
            "dataset_path": str(args.dataset_path),
            "odds_path": str(args.odds_path) if args.odds_path else None,
            "output_path": str(args.output_path),
            "profiles": list(args.profiles),
            "models": list(args.models),
            "min_train_days": int(args.min_train_days),
            "max_test_days": args.max_test_days,
            "initial_train_years": initial_train_years,
            "selection_years": selection_years,
            "final_train_years": final_train_years,
            "final_test_years": final_test_years,
            "train_start_date": args.train_start_date,
            "train_end_date": args.train_end_date,
            "test_start_date": args.test_start_date,
            "test_end_date": args.test_end_date,
            "calibration_buckets": int(args.calibration_buckets),
            "positive_call_min_bets": int(args.positive_call_min_bets),
            "positive_call_min_days": int(args.positive_call_min_days),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "bootstrap_seed": int(args.bootstrap_seed),
            "top_k_values": TOP_K_VALUES,
            "probability_thresholds": PROBABILITY_THRESHOLDS,
            "tier_strategies": TIER_STRATEGIES,
            "min_edge_values": MIN_EDGE_VALUES if odds_available else [],
            "kelly_fraction": float(args.kelly_fraction),
            "kelly_max_stake": float(args.kelly_max_stake),
        },
        "odds_status": {
            "real_odds_provided": odds_available,
            "odds_side_analysis_available": odds_available,
            "message": (
                "Real odds CSV joined; optional ROI, edge, and Kelly side-analysis use those odds, but model ranking remains prediction-centric."
                if odds_available
                else "No odds CSV provided. This report remains complete because the primary objective is prediction quality, not betting profitability."
            ),
            "required_odds_csv_columns": ["game_date", "batter_id", "american_odds"],
            "optional_odds_csv_columns": ["game_pk", "book", "sportsbook", "odds_timestamp"],
        },
        "temporal_contract": {
            "model_and_threshold_selection_period": selection_years,
            "untouched_final_evaluation_period": final_test_years,
            "selection_uses_final_evaluation": False,
            "threshold_is_frozen_before_final_evaluation": True,
        },
        "leakage_audits": [row["leakage_audit"] for row in reports],
        "selection_policy": {
            "release_eligible_profiles_rank_before_profiles_with_high_as_of_risk": True,
            "ranking_metrics": ["positive_call_precision", "top_3_daily_hit_rate", "average_precision", "brier_score", "log_loss"],
            "final_evaluation_metrics_do_not_affect_ranking": True,
        },
        "model_rankings": rankings,
        "selected_model": {
            **selected_ranking,
            "final_evaluation": selected_report["evaluation"],
        },
        "model_reports": reports,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Walk-forward prediction audit complete.")
    print(f"Model/profile reports: {len(reports)}")
    print(f"Odds side-analysis available: {odds_available}")
    print(f"Report written: {args.output_path}")


if __name__ == "__main__":
    main()
