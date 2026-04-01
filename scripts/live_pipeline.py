"""Shared helpers for forward-only live pick generation and settlement."""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.base import clone

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DEFAULT_GAME_HOUR_LOCAL,
    LIVE_CURRENT_PICKS_PATH,
    LIVE_MODEL_BUNDLE_PATH,
    LIVE_MODEL_DATA_PATH,
    LIVE_MODEL_METADATA_PATH,
    LIVE_MODEL_START_DATE,
    LIVE_PICK_HISTORY_PATH,
    LIVE_TRACKING_START_DATE,
    PARKS,
)
from data_sources import ensure_directories
from feature_engineering import (
    build_matchup_selected_handedness_features,
    compute_batter_handedness_split_features,
    compute_batter_trailing_features,
    compute_pitcher_handedness_split_features,
    compute_pitcher_trailing_features,
)
from generate_data import generate_mlb_dataset
from train_model import (
    LIVE_PLUS_FEATURE_COLUMNS,
    LIVE_PRODUCTION_FEATURE_COLUMNS,
    LIVE_SHRUNK_FEATURE_COLUMNS,
    LIVE_SHRUNK_PRECISE_FEATURE_COLUMNS,
    MAX_MODEL_FEATURE_MISSINGNESS,
    REASON_TEXT_BY_FEATURE,
    build_histgb_pipeline,
    build_logistic_pipeline,
    build_xgboost_pipeline,
    choose_logistic_calibration,
    evaluate_model_run,
    evaluate_training_candidate,
    extract_logistic_coefficient_map,
    fit_safely_with_imputer_warning_suppressed,
    generate_reason_strings,
    maybe_calibrate_logistic,
    prepare_feature_matrix,
    prune_model_features_by_training_missingness,
    run_backtest,
)
from weather_audit import (
    PRIMARY_WEATHER_FEATURE_COLUMNS,
    audit_weather_feature_coverage,
    print_weather_join_contract,
    weather_join_contract,
)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PERSON_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/people/{player_id}"
MLB_TEAM_ROSTER_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST_TIMEOUT_SECONDS = 20
OPEN_METEO_FORECAST_MAX_ATTEMPTS = 3
OPEN_METEO_FORECAST_RETRY_BACKOFF_SECONDS = 2
LIVE_PUBLISH_FRESHNESS_TOLERANCE_DAYS = 7
OFFSEASON_LINEUP_FALLBACK_DAYS = 210
TERMINAL_GAME_STATUS_TOKENS = (
    "final",
    "game over",
    "completed early",
    "cancelled",
)
LIVE_GAME_STATUS_TOKENS = (
    "in progress",
    "manager challenge",
    "review",
    "warmup",
    "mid",
    "delayed start",
    "delayed",
    "suspended",
)
TEAM_CODE_ALIASES = {
    "ARI": "AZ",
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
    "ATH": "OAK",
}
LIVE_PLUS_ONLY_FEATURE_COLUMNS = [
    feature for feature in LIVE_PLUS_FEATURE_COLUMNS if feature not in LIVE_PRODUCTION_FEATURE_COLUMNS
]
LIVE_SHRUNK_ONLY_FEATURE_COLUMNS = [
    feature for feature in LIVE_SHRUNK_FEATURE_COLUMNS if feature not in LIVE_PLUS_FEATURE_COLUMNS
]
LIVE_SHRUNK_PRECISE_ONLY_FEATURE_COLUMNS = [
    feature for feature in LIVE_SHRUNK_PRECISE_FEATURE_COLUMNS if feature not in LIVE_PLUS_FEATURE_COLUMNS
]
LIVE_COMPATIBLE_FEATURE_COLUMNS = list(
    dict.fromkeys([*LIVE_PLUS_FEATURE_COLUMNS, *LIVE_SHRUNK_FEATURE_COLUMNS, *LIVE_SHRUNK_PRECISE_FEATURE_COLUMNS])
)
LIVE_BATTER_SPLIT_SOURCE_COLUMNS = [
    "batter_hr_per_pa_vs_rhp",
    "batter_hr_per_pa_vs_lhp",
    "batter_barrels_per_pa_vs_rhp",
    "batter_barrels_per_pa_vs_lhp",
    "batter_hard_hit_rate_vs_rhp",
    "batter_hard_hit_rate_vs_lhp",
]
LIVE_PITCHER_SPLIT_SOURCE_COLUMNS = [
    "pitcher_hr_allowed_per_pa_vs_rhb",
    "pitcher_hr_allowed_per_pa_vs_lhb",
    "pitcher_barrels_allowed_per_bbe_vs_rhb",
    "pitcher_barrels_allowed_per_bbe_vs_lhb",
    "pitcher_hard_hit_allowed_rate_vs_rhb",
    "pitcher_hard_hit_allowed_rate_vs_lhb",
]
LIVE_PARK_FACTOR_SOURCE_COLUMNS = [
    "park_factor_hr_vs_lhb",
    "park_factor_hr_vs_rhb",
]
LIVE_SHRUNK_SNAPSHOT_COLUMNS = [
    "hr_rate_season_to_date_shrunk",
    "avg_launch_angle_last_50_bbe",
    "batter_pa_total_to_date",
    "batter_pa_vs_rhp_to_date",
    "batter_pa_vs_lhp_to_date",
]


def eastern_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").normalize()


def default_training_end_date() -> str:
    return (eastern_today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def default_publish_date() -> str:
    return eastern_today().strftime("%Y-%m-%d")


def parse_game_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def classify_game_state(game_like: dict[str, Any], reference_time: datetime | None = None) -> str:
    status_token = str(game_like.get("status") or "").strip().lower()
    if any(token in status_token for token in TERMINAL_GAME_STATUS_TOKENS):
        return "final"
    if any(token in status_token for token in LIVE_GAME_STATUS_TOKENS):
        return "live"
    reference = reference_time or datetime.now(timezone.utc)
    game_datetime = parse_game_datetime(game_like.get("game_datetime"))
    if game_datetime is not None and game_datetime <= reference:
        return "live"
    return "pregame"


def build_slate_state(
    schedule_games: list[dict[str, Any]],
    *,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    reference = reference_time or datetime.now(timezone.utc)
    games: list[dict[str, Any]] = []
    first_game_datetime: datetime | None = None
    for game in schedule_games:
        game_state = classify_game_state(game, reference)
        game_datetime = parse_game_datetime(game.get("game_datetime"))
        if game_datetime is not None and (first_game_datetime is None or game_datetime < first_game_datetime):
            first_game_datetime = game_datetime
        games.append(
            {
                **dict(game),
                "game_state": game_state,
                "is_locked": game_state != "pregame",
                "is_final": game_state == "final",
            }
        )
    return {
        "reference_time": reference,
        "first_game_datetime": first_game_datetime.isoformat() if first_game_datetime is not None else None,
        "games": games,
        "games_by_pk": {
            int(game["game_pk"]): game
            for game in games
            if game.get("game_pk") is not None
        },
        "all_final": bool(games) and all(bool(game["is_final"]) for game in games),
        "has_live_games": any(bool(game["game_state"] == "live") for game in games),
    }


def normalize_team_code(team_code: object) -> str:
    if pd.isna(team_code):
        return ""
    code = str(team_code).strip().upper()
    return TEAM_CODE_ALIASES.get(code, code)


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return [row for row in payload if isinstance(row, dict)]


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def serialize_for_json(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return round(float(value), 6)
    return value


def normalize_game_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)
    return str(timestamp.date())


def _coerce_int(value: Any) -> int | None:
    if value in (None, "") or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value in (None, "") or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolved_result_label(row: dict[str, Any]) -> str:
    token = str(row.get("result_label") or row.get("result") or "Pending").strip()
    if token in {"HR", "No HR", "Pending"}:
        return token
    normalized = token.lower()
    if normalized in {"1", "hr", "hit", "home_run", "home run", "success"}:
        return "HR"
    if normalized in {"0", "miss", "no hr", "no_hr", "failed", "failure"}:
        return "No HR"
    return "Pending"


def weather_code_label(value: Any) -> str:
    code = _coerce_int(value)
    if code is None:
        return "Unknown"
    if code in {0, 1}:
        return "Clear"
    if code in {2, 3}:
        return "Cloudy"
    if code in {45, 48}:
        return "Fog"
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    if code in {95, 96, 99}:
        return "Storm"
    return "Unknown"


def park_game_meta(home_team: Any) -> dict[str, Any]:
    park = PARKS.get(normalize_team_code(home_team))
    if not park:
        return {
            "ballpark_name": "",
            "ballpark_region_abbr": "",
            "field_bearing_deg": None,
        }
    return {
        "ballpark_name": str(park.get("ballpark") or ""),
        "ballpark_region_abbr": str(park.get("region_abbr") or ""),
        "field_bearing_deg": _coerce_float(park.get("field_bearing_deg")),
    }


def _build_pick_record_base(row: dict[str, Any]) -> dict[str, Any]:
    game_date = normalize_game_date(row.get("game_date"))
    batter_id = _coerce_int(row.get("batter_id"))
    pitcher_id = _coerce_int(row.get("pitcher_id"))
    batter_name = str(row.get("batter_name") or "Unknown hitter")
    pitcher_name = str(row.get("pitcher_name") or "")
    game_pk = _coerce_int(row.get("game_pk"))
    return {
        "pick_id": str(row.get("pick_id") or build_pick_id(game_date, game_pk, batter_id, batter_name, pitcher_id, pitcher_name)),
        "published_at": str(row.get("published_at") or datetime.now(timezone.utc).isoformat()),
        "game_pk": game_pk,
        "game_date": game_date,
        "game_datetime": str(row.get("game_datetime") or ""),
        "game_status": str(row.get("game_status") or row.get("status") or ""),
        "game_state": str(row.get("game_state") or classify_game_state(row)),
        "rank": _coerce_int(row.get("rank")) or 999,
        "batter_id": batter_id,
        "batter_name": batter_name,
        "team": str(row.get("team") or ""),
        "opponent_team": str(row.get("opponent_team") or row.get("opponent") or ""),
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_name,
        "confidence_tier": str(row.get("confidence_tier") or "watch"),
        "predicted_hr_probability": _coerce_float(row.get("predicted_hr_probability")),
        "predicted_hr_score": _coerce_float(row.get("predicted_hr_score")),
        "top_reason_1": str(row.get("top_reason_1") or ""),
        "top_reason_2": str(row.get("top_reason_2") or ""),
        "top_reason_3": str(row.get("top_reason_3") or ""),
        "lineup_source": str(row.get("lineup_source") or "projected"),
        "batting_order": _coerce_int(row.get("batting_order")),
        "ballpark_name": str(row.get("ballpark_name") or row.get("ballpark") or ""),
        "ballpark_region_abbr": str(row.get("ballpark_region_abbr") or ""),
        "weather_code": _coerce_int(row.get("weather_code")),
        "weather_label": str(row.get("weather_label") or weather_code_label(row.get("weather_code"))),
        "temperature_f": _coerce_float(row.get("temperature_f")),
        "wind_speed_mph": _coerce_float(row.get("wind_speed_mph")),
        "wind_direction_deg": _coerce_float(row.get("wind_direction_deg")),
        "field_bearing_deg": _coerce_float(row.get("field_bearing_deg")),
    }


def canonicalize_current_pick_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        base = _build_pick_record_base(row)
        base["result"] = _resolved_result_label(row)
        canonical_rows.append(base)
    return canonical_rows


def canonicalize_history_pick_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        base = _build_pick_record_base(row)
        result_label = _resolved_result_label(row)
        actual_hit_hr = row.get("actual_hit_hr")
        if actual_hit_hr in (None, "") or pd.isna(actual_hit_hr):
            actual_hit_hr = 1 if result_label == "HR" else 0 if result_label == "No HR" else None
        else:
            actual_hit_hr = _coerce_int(actual_hit_hr)
        base["result_label"] = result_label
        base["actual_hit_hr"] = actual_hit_hr
        canonical_rows.append(base)
    return canonical_rows


def build_pick_id(game_date: str, game_pk: int | None, batter_id: int | None, batter_name: str, pitcher_id: int | None, pitcher_name: str) -> str:
    hitter = batter_id if batter_id is not None else batter_name.lower().replace(" ", "-")
    pitcher = pitcher_id if pitcher_id is not None else pitcher_name.lower().replace(" ", "-")
    if game_pk is not None:
        return f"{game_date}:{game_pk}:{hitter}:{pitcher}"
    return f"{game_date}:{hitter}:{pitcher}"


def confidence_tier_from_percentile(percentile: float) -> str:
    if percentile >= 0.99:
        return "elite"
    if percentile >= 0.95:
        return "strong"
    if percentile >= 0.80:
        return "watch"
    return "longshot"


CONFIDENCE_TIER_ORDER = {
    "longshot": 0,
    "watch": 1,
    "strong": 2,
    "elite": 3,
}


def _confidence_tier_value(tier: str | None) -> int:
    return CONFIDENCE_TIER_ORDER.get(str(tier or "").strip().lower(), -1)


def latest_non_null(series: pd.Series, default: Any = None) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return default
    return non_null.iloc[-1]


def refresh_live_dataset(
    *,
    output_path: Path = LIVE_MODEL_DATA_PATH,
    start_date: str = LIVE_MODEL_START_DATE,
    end_date: str | None = None,
    force_refresh: bool = False,
) -> Path:
    ensure_directories()
    resolved_end_date = end_date or default_training_end_date()
    generate_mlb_dataset(
        output_path=str(output_path),
        start_date=start_date,
        end_date=resolved_end_date,
        force_refresh=force_refresh,
    )
    return output_path


def _candidate_is_live_compatible(feature_columns: list[str]) -> bool:
    return set(feature_columns).issubset(set(LIVE_PLUS_FEATURE_COLUMNS))


def _live_publish_weather_contract_label() -> str:
    contract = weather_join_contract()
    return (
        f"{' + '.join(contract['join_keys'])}; "
        f"{contract['park_lookup']}; "
        f"{contract['selection_rule']}"
    )


def _weather_coverage_summary_payload(audit_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "warning_null_rate_threshold": float(audit_payload["warning_null_rate_threshold"]),
        "high_missing_features": list(audit_payload["high_missing_features"]),
        "missing_columns": list(audit_payload["missing_columns"]),
        "all_null_features": list(audit_payload["all_null_features"]),
        "summary": {
            feature: {
                "row_count": int(stats["row_count"]),
                "non_null_count": int(stats["non_null_count"]),
                "null_rate": serialize_for_json(float(stats["null_rate"])) if stats["null_rate"] is not None else None,
                "present": bool(stats["present"]),
            }
            for feature, stats in audit_payload["summary"].items()
        },
        "contract": dict(audit_payload["contract"]),
    }


def _audit_feature_frame_columns(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    context: str,
) -> dict[str, list[str]]:
    missing_columns = [feature for feature in feature_columns if feature not in frame.columns]
    all_null_columns = [
        feature
        for feature in feature_columns
        if feature in frame.columns and int(frame[feature].notna().sum()) == 0
    ]
    print(f"\nFeature-frame readiness ({context})")
    print("-" * 60)
    print(f"Required feature count        : {len(feature_columns)}")
    print(f"Missing feature columns       : {missing_columns if missing_columns else 'none'}")
    print(f"All-null feature columns      : {all_null_columns if all_null_columns else 'none'}")
    return {
        "missing_columns": missing_columns,
        "all_null_columns": all_null_columns,
    }


def _build_pipeline_for_model_family(model_family: str):
    if model_family == "logistic":
        return build_logistic_pipeline()
    if model_family == "histgb":
        return build_histgb_pipeline()
    if model_family == "xgboost":
        pipeline = build_xgboost_pipeline()
        if pipeline is None:
            raise RuntimeError("xgboost is not installed, so the stored live configuration cannot be refit.")
        return pipeline
    raise ValueError(f"Unsupported model family for live refit: {model_family}")


def _filtered_best_params(estimator, best_params: dict[str, Any] | None) -> dict[str, Any]:
    if not best_params:
        return {}
    valid_keys = set(estimator.get_params().keys())
    return {key: value for key, value in best_params.items() if key in valid_keys}


def _persist_live_bundle(bundle_path: Path, metadata_path: Path, bundle: dict[str, Any], metadata: dict[str, Any]) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _fit_full_dataset_bundle_candidate(
    df: pd.DataFrame,
    *,
    model_family: str,
    feature_profile: str,
    missingness_threshold: float,
    selection_metric: str,
    calibration: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    full_candidate = evaluate_training_candidate(
        df,
        feature_profile=feature_profile,
        model_name=model_family,
        missingness_threshold=missingness_threshold,
        selection_metric=selection_metric,
    )
    if full_candidate is None:
        raise RuntimeError("Unable to fit the promoted live candidate on the full training dataset.")

    feature_columns = list(full_candidate["feature_columns"])
    X_full = prepare_feature_matrix(df, feature_columns)
    y_full = df["hit_hr"].to_numpy()
    base_estimator = full_candidate["search"].best_estimator_
    if model_family == "logistic":
        model, calibration_status, calibration_search_rows = choose_logistic_calibration(
            base_estimator,
            X_full,
            y_full,
            feature_columns,
            calibration,
        )
    else:
        model = clone(base_estimator)
        fit_safely_with_imputer_warning_suppressed(model, X_full, y_full)
        calibration_status = {
            "requested": calibration,
            "used": "not_applicable",
            "status": "not_applicable",
            "message": f"Calibration skipped because {model_family} was used.",
        }
        calibration_search_rows = []
    calibration_search_summary = [
        {key: value for key, value in row.items() if key != "model"}
        for row in calibration_search_rows
    ]

    reference_columns = sorted(set(feature_columns) | {feature for feature in REASON_TEXT_BY_FEATURE if feature in df.columns})
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "reference_df": df[reference_columns].copy(),
        "trained_through": str(df["game_date"].max().date()),
        "dataset_max_game_date": str(df["game_date"].max().date()),
        "model_family": model_family,
        "feature_profile": feature_profile,
        "missingness_threshold": float(missingness_threshold),
        "selection_metric": selection_metric,
        "best_params": dict(full_candidate["summary_row"]["best_params"]),
        "calibration_status": calibration_status,
        "calibration_search_rows": calibration_search_summary,
        "excluded_features": list(full_candidate["excluded_features"]),
        "training_cv_summary": {
            "mean_cv_pr_auc": float(full_candidate["summary_row"]["mean_cv_pr_auc"]),
            "mean_cv_roc_auc": float(full_candidate["summary_row"]["mean_cv_roc_auc"]),
            "mean_cv_log_loss": float(full_candidate["summary_row"]["mean_cv_log_loss"]),
            "mean_cv_brier_score": float(full_candidate["summary_row"]["mean_cv_brier_score"]),
        },
    }
    return bundle, full_candidate, calibration_status, calibration_search_summary


def _fit_live_bundle_fast_refit(
    df: pd.DataFrame,
    *,
    model_family: str,
    feature_profile: str,
    missingness_threshold: float,
    selection_metric: str,
    calibration: str,
    best_params: dict[str, Any] | None,
    existing_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    configured_feature_map = {
        "live": LIVE_PRODUCTION_FEATURE_COLUMNS,
        "live_plus": LIVE_PLUS_FEATURE_COLUMNS,
        "live_shrunk": LIVE_SHRUNK_FEATURE_COLUMNS,
        "live_shrunk_precise": LIVE_SHRUNK_PRECISE_FEATURE_COLUMNS,
    }
    configured_features = [
        column for column in configured_feature_map.get(feature_profile, []) if column in df.columns
    ]
    if not configured_features:
        metadata_feature_columns = existing_metadata.get("feature_columns")
        if isinstance(metadata_feature_columns, list):
            configured_features = [
                str(column)
                for column in metadata_feature_columns
                if isinstance(column, str) and column in df.columns
            ]
    if not configured_features:
        from train_model import available_feature_columns

        configured_features = available_feature_columns(df, feature_profile=feature_profile)
    feature_columns, _, excluded_features = prune_model_features_by_training_missingness(
        df,
        configured_features,
        threshold=missingness_threshold,
    )
    if not feature_columns:
        raise RuntimeError(f"No model features remain after applying missingness threshold {missingness_threshold:.2f}.")

    estimator = _build_pipeline_for_model_family(model_family)
    resolved_best_params = _filtered_best_params(estimator, best_params)
    if resolved_best_params:
        estimator.set_params(**resolved_best_params)

    X_full = prepare_feature_matrix(df, feature_columns)
    y_full = df["hit_hr"].to_numpy()
    if model_family == "logistic":
        if calibration == "disabled":
            model = clone(estimator)
            fit_safely_with_imputer_warning_suppressed(model, X_full, y_full)
            calibration_status = {
                "requested": calibration,
                "used": "disabled",
                "status": "skipped",
                "message": "Calibration disabled for fast live refit.",
            }
        else:
            model, calibration_status = maybe_calibrate_logistic(
                clone(estimator),
                X_full,
                y_full,
                calibration,
                model_family,
            )
            if calibration_status.get("used") == "disabled":
                fit_safely_with_imputer_warning_suppressed(model, X_full, y_full)
        calibration_search_rows: list[dict[str, Any]] = []
    else:
        model = clone(estimator)
        fit_safely_with_imputer_warning_suppressed(model, X_full, y_full)
        calibration_status = {
            "requested": calibration,
            "used": "not_applicable",
            "status": "not_applicable",
            "message": f"Calibration skipped because {model_family} was used.",
        }
        calibration_search_rows = []

    reference_columns = sorted(set(feature_columns) | {feature for feature in REASON_TEXT_BY_FEATURE if feature in df.columns})
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "reference_df": df[reference_columns].copy(),
        "trained_through": str(df["game_date"].max().date()),
        "dataset_max_game_date": str(df["game_date"].max().date()),
        "model_family": model_family,
        "feature_profile": feature_profile,
        "missingness_threshold": float(missingness_threshold),
        "selection_metric": selection_metric,
        "best_params": dict(resolved_best_params),
        "calibration_status": calibration_status,
        "calibration_search_rows": calibration_search_rows,
        "excluded_features": list(excluded_features),
        "training_cv_summary": dict(existing_metadata.get("training_cv_summary") or {}),
        "final_holdout_summary": dict(existing_metadata.get("final_holdout_summary") or {}),
        "promotion_decision": dict(existing_metadata.get("promotion_decision") or {}),
        "refit_strategy": "fast_refit",
    }
    metadata = {
        "trained_through": bundle["trained_through"],
        "dataset_max_game_date": bundle["dataset_max_game_date"],
        "model_family": bundle["model_family"],
        "feature_profile": bundle["feature_profile"],
        "missingness_threshold": bundle["missingness_threshold"],
        "selection_metric": bundle["selection_metric"],
        "feature_columns": bundle["feature_columns"],
        "best_params": bundle["best_params"],
        "calibration_status": calibration_status,
        "calibration_search_rows": calibration_search_rows,
        "excluded_features": bundle["excluded_features"],
        "training_cv_summary": bundle["training_cv_summary"],
        "final_holdout_summary": bundle["final_holdout_summary"],
        "promotion_decision": bundle["promotion_decision"],
        "row_count": int(len(df)),
        "hr_rate": serialize_for_json(float(df["hit_hr"].mean())),
        "refit_strategy": "fast_refit",
    }
    return bundle, metadata


def train_live_model_bundle(
    dataset_path: Path,
    *,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    model_name: str = "logistic",
    calibration: str = "sigmoid",
    feature_profile: str = "live_shrunk",
    selection_metric: str = "pr_auc",
    missingness_threshold: float | None = None,
    training_mode: str = "search",
) -> dict[str, Any]:
    df = pd.read_csv(dataset_path, parse_dates=["game_date"])
    print_weather_join_contract("live training dataset")
    weather_coverage = audit_weather_feature_coverage(
        df,
        context="live training dataset",
        feature_columns=list(PRIMARY_WEATHER_FEATURE_COLUMNS),
        fail_on_missing_columns=True,
        fail_on_all_null=False,
    )
    if training_mode == "fast_refit":
        existing_metadata = load_model_metadata(metadata_path) if metadata_path.exists() else {}
        resolved_model_family = str(model_name or existing_metadata.get("model_family") or "logistic")
        resolved_feature_profile = str(feature_profile or existing_metadata.get("feature_profile") or "live_shrunk")
        resolved_missingness_threshold = float(
            existing_metadata.get("missingness_threshold")
            if existing_metadata.get("missingness_threshold") is not None
            else (missingness_threshold if missingness_threshold is not None else MAX_MODEL_FEATURE_MISSINGNESS)
        )
        resolved_selection_metric = str(existing_metadata.get("selection_metric") or selection_metric)
        calibration_status = existing_metadata.get("calibration_status") if isinstance(existing_metadata.get("calibration_status"), dict) else {}
        resolved_calibration = str(calibration_status.get("used") or calibration)
        if existing_metadata:
            print("\nFast live refit")
            print("-" * 60)
            print(f"Selection metric snapshot  : {resolved_selection_metric}")
            print(f"Calibration mode reused    : {resolved_calibration}")
        else:
            print("\nFast live refit bootstrap")
            print("-" * 60)
            print("No existing model metadata found; using the requested live configuration for a direct refit.")
            print(f"Selection metric snapshot  : {resolved_selection_metric}")
            print(f"Calibration mode requested : {resolved_calibration}")
        print(f"Model family               : {resolved_model_family}")
        print(f"Feature profile            : {resolved_feature_profile}")
        print(f"Missingness threshold      : {resolved_missingness_threshold:.2f}")
        bundle, metadata = _fit_live_bundle_fast_refit(
            df,
            model_family=resolved_model_family,
            feature_profile=resolved_feature_profile,
            missingness_threshold=resolved_missingness_threshold,
            selection_metric=resolved_selection_metric,
            calibration=resolved_calibration,
            best_params=existing_metadata.get("best_params") if isinstance(existing_metadata.get("best_params"), dict) else {},
            existing_metadata=existing_metadata,
        )
        bundle["weather_feature_coverage"] = _weather_coverage_summary_payload(weather_coverage)
        bundle["weather_join_contract"] = _live_publish_weather_contract_label()
        metadata["weather_feature_coverage"] = _weather_coverage_summary_payload(weather_coverage)
        metadata["weather_join_contract"] = _live_publish_weather_contract_label()
        _persist_live_bundle(bundle_path, metadata_path, bundle, metadata)
        return bundle

    backtest = run_backtest(
        str(dataset_path),
        model_name=model_name,
        feature_profile=feature_profile,
        compare_against=(
            "live_plus"
            if feature_profile in {"live_shrunk", "live_shrunk_precise"}
            else ("live" if feature_profile == "live_plus" else None)
        ),
        selection_metric=selection_metric,
        missingness_threshold=missingness_threshold,
        calibration=calibration,
    )
    selected_candidate = backtest["selected_candidate"]
    winner_holdout_summary = backtest["final_result"]["summary_row"]
    candidate_results = backtest["candidate_results"]
    train_df = backtest["train_df"]
    test_df = backtest["test_df"]

    baseline_candidate = next(
        (
            result
            for result in candidate_results
            if result["summary_row"]["model_family"] == "logistic"
            and result["summary_row"]["feature_profile"] == "live"
            and np.isclose(result["summary_row"]["missingness_threshold"], MAX_MODEL_FEATURE_MISSINGNESS)
        ),
        None,
    )
    if baseline_candidate is None:
        baseline_candidate = evaluate_training_candidate(
            train_df,
            feature_profile="live",
            model_name="logistic",
            missingness_threshold=MAX_MODEL_FEATURE_MISSINGNESS,
            selection_metric=selection_metric,
        )
        if baseline_candidate is None:
            raise RuntimeError("Unable to evaluate the live baseline candidate.")
    if (
        selected_candidate["summary_row"]["model_family"] == "logistic"
        and selected_candidate["summary_row"]["feature_profile"] == "live"
        and np.isclose(selected_candidate["summary_row"]["missingness_threshold"], MAX_MODEL_FEATURE_MISSINGNESS)
    ):
        baseline_holdout = backtest["final_result"]
    else:
        baseline_holdout = evaluate_model_run(
            train_df=train_df,
            test_df=test_df,
            candidate_result=baseline_candidate,
            threshold_objective="f0.5",
            min_recall=0.15,
            max_positive_rate=0.14,
            threshold_tolerance=0.001,
            calibration=calibration,
            save_ranked_output=False,
            ranked_output_path=None,
        )

    winner_is_compatible = _candidate_is_live_compatible(list(selected_candidate["feature_columns"]))
    winner_beats_baseline = float(winner_holdout_summary["pr_auc"]) >= float(baseline_holdout["summary_row"]["pr_auc"])
    if winner_is_compatible and winner_beats_baseline:
        promoted_candidate = selected_candidate
        promoted_holdout = backtest["final_result"]
        promotion_reason = "selected_candidate"
    else:
        promoted_candidate = baseline_candidate
        promoted_holdout = baseline_holdout
        promotion_reason = "baseline_fallback"

    bundle, full_candidate, calibration_status, calibration_search_rows = _fit_full_dataset_bundle_candidate(
        df,
        model_family=str(promoted_candidate["summary_row"]["model_family"]),
        feature_profile=str(promoted_candidate["summary_row"]["feature_profile"]),
        missingness_threshold=float(promoted_candidate["summary_row"]["missingness_threshold"]),
        selection_metric=selection_metric,
        calibration=calibration,
    )
    bundle["final_holdout_summary"] = dict(promoted_holdout["summary_row"])
    bundle["promotion_decision"] = {
        "used": promotion_reason,
        "winner_live_compatible": winner_is_compatible,
        "winner_beats_baseline": winner_beats_baseline,
        "winner_pr_auc": float(winner_holdout_summary["pr_auc"]),
        "baseline_pr_auc": float(baseline_holdout["summary_row"]["pr_auc"]),
    }
    bundle["weather_feature_coverage"] = _weather_coverage_summary_payload(weather_coverage)
    bundle["weather_join_contract"] = _live_publish_weather_contract_label()
    metadata = {
        "trained_through": bundle["trained_through"],
        "dataset_max_game_date": bundle["dataset_max_game_date"],
        "model_family": bundle["model_family"],
        "feature_profile": bundle["feature_profile"],
        "missingness_threshold": bundle["missingness_threshold"],
        "selection_metric": bundle["selection_metric"],
        "feature_columns": bundle["feature_columns"],
        "best_params": bundle["best_params"],
        "calibration_status": calibration_status,
        "calibration_search_rows": calibration_search_rows,
        "excluded_features": bundle["excluded_features"],
        "training_cv_summary": bundle["training_cv_summary"],
        "final_holdout_summary": bundle["final_holdout_summary"],
        "promotion_decision": bundle["promotion_decision"],
        "weather_feature_coverage": bundle["weather_feature_coverage"],
        "weather_join_contract": bundle["weather_join_contract"],
        "row_count": int(len(df)),
        "hr_rate": serialize_for_json(float(df["hit_hr"].mean())),
    }
    _persist_live_bundle(bundle_path, metadata_path, bundle, metadata)
    return bundle


def load_model_bundle(bundle_path: Path = LIVE_MODEL_BUNDLE_PATH) -> dict[str, Any]:
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


def load_model_metadata(metadata_path: Path = LIVE_MODEL_METADATA_PATH) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    return payload


def evaluate_live_publish_freshness(
    *,
    schedule_date: str,
    dataset_df: pd.DataFrame,
    model_metadata: dict[str, Any],
    tolerance_days: int = LIVE_PUBLISH_FRESHNESS_TOLERANCE_DAYS,
) -> dict[str, Any]:
    schedule_timestamp = pd.Timestamp(schedule_date).normalize()
    metadata_trained_through = pd.to_datetime(model_metadata.get("trained_through"), errors="coerce")
    dataset_max_game_date = pd.to_datetime(dataset_df["game_date"].max(), errors="coerce")
    if pd.notna(metadata_trained_through):
        metadata_trained_through = metadata_trained_through.normalize()
    if pd.notna(dataset_max_game_date):
        dataset_max_game_date = dataset_max_game_date.normalize()

    metadata_lag_days = None if pd.isna(metadata_trained_through) else int((schedule_timestamp - metadata_trained_through).days)
    dataset_lag_days = None if pd.isna(dataset_max_game_date) else int((schedule_timestamp - dataset_max_game_date).days)
    metadata_stale = metadata_lag_days is None or metadata_lag_days > tolerance_days
    dataset_stale = dataset_lag_days is None or dataset_lag_days > tolerance_days

    return {
        "schedule_date": str(schedule_timestamp.date()),
        "metadata_trained_through": None if pd.isna(metadata_trained_through) else str(metadata_trained_through.date()),
        "dataset_max_game_date": None if pd.isna(dataset_max_game_date) else str(dataset_max_game_date.date()),
        "metadata_lag_days": metadata_lag_days,
        "dataset_lag_days": dataset_lag_days,
        "tolerance_days": int(tolerance_days),
        "metadata_stale": bool(metadata_stale),
        "dataset_stale": bool(dataset_stale),
        "passed": bool(not metadata_stale and not dataset_stale),
    }


def assert_live_publish_freshness(
    *,
    schedule_date: str,
    dataset_df: pd.DataFrame,
    model_metadata: dict[str, Any],
    tolerance_days: int = LIVE_PUBLISH_FRESHNESS_TOLERANCE_DAYS,
) -> dict[str, Any]:
    diagnostics = evaluate_live_publish_freshness(
        schedule_date=schedule_date,
        dataset_df=dataset_df,
        model_metadata=model_metadata,
        tolerance_days=tolerance_days,
    )
    print("\nLive publish freshness check")
    print("-" * 60)
    print(f"Schedule date                 : {diagnostics['schedule_date']}")
    print(f"Model metadata trained_through: {diagnostics['metadata_trained_through']}")
    print(f"Dataset max game_date         : {diagnostics['dataset_max_game_date']}")
    print(f"Freshness tolerance (days)    : {diagnostics['tolerance_days']}")
    print(f"Metadata lag days             : {diagnostics['metadata_lag_days']}")
    print(f"Dataset lag days              : {diagnostics['dataset_lag_days']}")
    print(f"Freshness status              : {'passed' if diagnostics['passed'] else 'failed'}")
    if not diagnostics["passed"]:
        raise RuntimeError(
            "Live publish aborted because the training data is stale for "
            f"schedule_date={diagnostics['schedule_date']}. "
            f"model_metadata.trained_through={diagnostics['metadata_trained_through']} "
            f"and dataset_max_game_date={diagnostics['dataset_max_game_date']} "
            f"with tolerance_days={diagnostics['tolerance_days']}. "
            "Run scripts/train_live_model.py before publishing."
        )
    return diagnostics


def fetch_schedule_games(target_date: str) -> list[dict[str, Any]]:
    def _projected_lineup_rows(lineups_payload: dict[str, Any], side: str) -> list[dict[str, Any]]:
        side_key = f"{side}Players"
        players = lineups_payload.get(side_key)
        if not isinstance(players, list):
            return []
        rows: list[dict[str, Any]] = []
        for player in players:
            player_id = player.get("id")
            if not player_id:
                continue
            rows.append(
                {
                    "batter_id": int(player_id),
                    "batter_name": str(player.get("fullName") or ""),
                    "batting_order": _coerce_int(player.get("battingOrder")) // 100 if _coerce_int(player.get("battingOrder")) else None,
                }
            )
        return rows

    response = requests.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": target_date, "hydrate": "probablePitcher,team,lineups"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    games: list[dict[str, Any]] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            lineups_payload = game.get("lineups", {}) if isinstance(game.get("lineups"), dict) else {}
            home_lineup = _projected_lineup_rows(lineups_payload, "home")
            away_lineup = _projected_lineup_rows(lineups_payload, "away")
            games.append(
                {
                    "game_pk": int(game["gamePk"]),
                    "game_date": str(game.get("officialDate") or target_date),
                    "game_datetime": str(game.get("gameDate") or ""),
                    "status": str(game.get("status", {}).get("detailedState") or ""),
                    "home_team_id": int(home.get("team", {}).get("id")) if home.get("team", {}).get("id") else None,
                    "away_team_id": int(away.get("team", {}).get("id")) if away.get("team", {}).get("id") else None,
                    "home_team": normalize_team_code(home.get("team", {}).get("abbreviation")),
                    "away_team": normalize_team_code(away.get("team", {}).get("abbreviation")),
                    "home_team_name": str(home.get("team", {}).get("name") or ""),
                    "away_team_name": str(away.get("team", {}).get("name") or ""),
                    "home_pitcher_id": int(home["probablePitcher"]["id"]) if home.get("probablePitcher", {}).get("id") else None,
                    "away_pitcher_id": int(away["probablePitcher"]["id"]) if away.get("probablePitcher", {}).get("id") else None,
                    "home_pitcher_name": str(home.get("probablePitcher", {}).get("fullName") or ""),
                    "away_pitcher_name": str(away.get("probablePitcher", {}).get("fullName") or ""),
                    "home_projected_lineup": home_lineup,
                    "away_projected_lineup": away_lineup,
                    "home_lineup_source": "confirmed" if home_lineup else "projected",
                    "away_lineup_source": "confirmed" if away_lineup else "projected",
                }
            )
    return games


def fetch_player_handedness(player_id: int | None) -> str | None:
    if player_id is None:
        return None
    response = requests.get(MLB_PERSON_URL_TEMPLATE.format(player_id=player_id), timeout=60)
    response.raise_for_status()
    payload = response.json()
    people = payload.get("people", [])
    if not people:
        return None
    hand = people[0].get("pitchHand", {}).get("code")
    if hand in {"L", "R"}:
        return str(hand)
    return None


def fetch_active_roster(team_id: int | None) -> pd.DataFrame:
    if team_id is None:
        return pd.DataFrame(columns=["batter_id", "batter_name"])
    response = requests.get(
        MLB_TEAM_ROSTER_URL_TEMPLATE.format(team_id=team_id),
        params={"rosterType": "active"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    rows: list[dict[str, Any]] = []
    for item in payload.get("roster", []):
        person = item.get("person", {})
        player_id = person.get("id")
        if not player_id:
            continue
        rows.append(
            {
                "batter_id": int(player_id),
                "batter_name": str(person.get("fullName") or ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["batter_id", "batter_name"])
    roster = pd.DataFrame(rows).drop_duplicates(subset=["batter_id"], keep="first")
    roster["batter_id"] = pd.to_numeric(roster["batter_id"], errors="coerce").astype("Int64")
    return roster


def build_active_roster_map(schedule_games: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    roster_map: dict[str, pd.DataFrame] = {}
    team_specs = {}
    for game in schedule_games:
        for team_code_key, team_id_key in [("home_team", "home_team_id"), ("away_team", "away_team_id")]:
            team_code = normalize_team_code(game.get(team_code_key))
            team_id = game.get(team_id_key)
            if team_code and team_code not in team_specs:
                team_specs[team_code] = team_id
    for team_code, team_id in team_specs.items():
        roster_map[team_code] = fetch_active_roster(int(team_id) if team_id is not None else None)
    return roster_map


def load_live_dataset(dataset_path: Path = LIVE_MODEL_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(dataset_path, parse_dates=["game_date"])


def latest_pitcher_hand(dataset_df: pd.DataFrame, pitcher_id: int | None) -> str | None:
    if pitcher_id is None:
        return None
    pitcher_rows = dataset_df[dataset_df["pitcher_id"] == pitcher_id].sort_values(["game_date", "game_pk"])
    hand = latest_non_null(pitcher_rows["pitch_hand_primary"], default=None)
    if hand in {"L", "R"}:
        return str(hand)
    return None


def _summarize_hitter_pool(team_rows: pd.DataFrame) -> pd.DataFrame:
    ordered = team_rows.sort_values(["game_date", "game_pk"])
    summary = (
        ordered.groupby("batter_id", dropna=False)
        .agg(
            batter_name=("batter_name", "last"),
            bat_side=("bat_side", lambda s: latest_non_null(s, default=None)),
            last_game_date=("game_date", "max"),
            games_recent=("game_pk", "nunique"),
            starts_recent=("pa_count", lambda s: int((s >= 3).sum())),
            avg_pa_recent=("pa_count", "mean"),
            latest_hr_form=("hr_per_pa_last_10d", lambda s: float(latest_non_null(s, default=0.0) or 0.0)),
            latest_hr_form_30=("hr_per_pa_last_30d", lambda s: float(latest_non_null(s, default=0.0) or 0.0)),
        )
        .reset_index()
    )
    return summary.sort_values(
        ["starts_recent", "games_recent", "avg_pa_recent", "latest_hr_form", "latest_hr_form_30", "last_game_date", "batter_name"],
        ascending=[False, False, False, False, False, False, True],
    )


def _print_lineup_selection_diagnostics(
    *,
    team_code: str,
    active_roster_count: int | None,
    projected_lineup_count: int | None,
    active_hitter_history_count: int,
    recent_team_games_count: int,
    eligibility_mode: str,
    last_completed_team_game: pd.Timestamp | None,
    days_since_last_team_game: int | None,
    eligible_count: int,
    selected_count: int,
    hitters_per_team: int,
    excluded_rows: pd.DataFrame,
) -> None:
    print(f"\nLive hitter eligibility diagnostics [{team_code}]")
    print("-" * 60)
    if active_roster_count is None:
        print("Active roster count           : n/a")
    else:
        print(f"Active roster count           : {active_roster_count}")
    if projected_lineup_count is None:
        print("Projected lineup count       : n/a")
    else:
        print(f"Projected lineup count       : {projected_lineup_count}")
    print(f"Active roster hitters w/history: {active_hitter_history_count}")
    print(f"Eligibility mode             : {eligibility_mode}")
    if last_completed_team_game is None or pd.isna(last_completed_team_game):
        print("Last completed team game     : n/a")
    else:
        print(f"Last completed team game     : {pd.Timestamp(last_completed_team_game).date()}")
    if days_since_last_team_game is None:
        print("Days since completed game    : n/a")
    else:
        print(f"Days since completed game    : {days_since_last_team_game}")
    print(f"Recent team games considered  : {recent_team_games_count}")
    print(f"Eligible after recent gate    : {eligible_count}")
    print(f"Requested hitters / selected  : {hitters_per_team} / {selected_count}")
    print(f"Underfilled team              : {'yes' if selected_count < hitters_per_team else 'no'}")
    if excluded_rows.empty:
        print("Excluded hitters              : none")
        return
    for row in excluded_rows.sort_values(["last_game_date", "batter_name"], ascending=[False, True]).itertuples(index=False):
        last_game = "" if pd.isna(row.last_game_date) else pd.Timestamp(row.last_game_date).date()
        print(
            f"  excluded {row.batter_name} ({int(row.batter_id)}): "
            f"inactive_recent_games (last_game_date={last_game})"
        )


def select_probable_lineup_hitters(
    dataset_df: pd.DataFrame,
    *,
    team_code: str,
    target_date: str,
    hitters_per_team: int = 9,
    lookback_days: int = 21,
    projected_lineup: pd.DataFrame | None = None,
    active_roster: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    target_timestamp = pd.Timestamp(target_date)
    historical = dataset_df[dataset_df["game_date"] < target_timestamp].copy()
    team_history = historical[historical["team"] == team_code].copy()
    if team_history.empty:
        return []
    last_completed_team_game = pd.Timestamp(team_history["game_date"].max()) if not team_history.empty else None
    days_since_last_team_game = None
    if last_completed_team_game is not None and not pd.isna(last_completed_team_game):
        days_since_last_team_game = int((target_timestamp.normalize() - last_completed_team_game.normalize()).days)
    recent_cutoff = target_timestamp - pd.Timedelta(days=lookback_days)
    recent_team_rows = team_history[team_history["game_date"] >= recent_cutoff].copy()
    recent_team_games = (
        recent_team_rows[["game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk"], ascending=[False, False])
        .head(3)
    )
    eligibility_mode = "recent_team_games"
    if recent_team_games.empty:
        if (
            days_since_last_team_game is not None
            and days_since_last_team_game <= OFFSEASON_LINEUP_FALLBACK_DAYS
        ):
            recent_team_games = (
                team_history[["game_date", "game_pk"]]
                .drop_duplicates()
                .sort_values(["game_date", "game_pk"], ascending=[False, False])
                .head(3)
            )
            eligibility_mode = "season_opening_fallback"
        else:
            eligibility_mode = "stale_team_history"
    active_roster_count = None
    projected_lineup_count = None
    team_history["batter_id"] = pd.to_numeric(team_history["batter_id"], errors="coerce").astype("Int64")
    if projected_lineup is not None:
        projected_lineup = projected_lineup.copy()
        projected_lineup["batter_id"] = pd.to_numeric(projected_lineup["batter_id"], errors="coerce").astype("Int64")
        projected_lineup_count = int(projected_lineup["batter_id"].dropna().nunique())
        if projected_lineup_count > 0:
            team_history = team_history[team_history["batter_id"].isin(projected_lineup["batter_id"])].copy()
            if team_history.empty:
                _print_lineup_selection_diagnostics(
                    team_code=team_code,
                    active_roster_count=active_roster_count,
                    projected_lineup_count=projected_lineup_count,
                    active_hitter_history_count=0,
                    recent_team_games_count=int(len(recent_team_games)),
                    eligibility_mode=eligibility_mode,
                    last_completed_team_game=last_completed_team_game,
                    days_since_last_team_game=days_since_last_team_game,
                    eligible_count=0,
                    selected_count=0,
                    hitters_per_team=hitters_per_team,
                    excluded_rows=pd.DataFrame(columns=["batter_id", "batter_name", "last_game_date"]),
                )
                return []
        else:
            projected_lineup = None
    if active_roster is not None:
        active_roster = active_roster.copy()
        active_roster["batter_id"] = pd.to_numeric(active_roster["batter_id"], errors="coerce").astype("Int64")
        active_roster_count = int(active_roster["batter_id"].dropna().nunique())
        team_history = team_history[team_history["batter_id"].isin(active_roster["batter_id"])].copy()
        if team_history.empty:
            _print_lineup_selection_diagnostics(
                team_code=team_code,
                active_roster_count=active_roster_count,
                projected_lineup_count=projected_lineup_count,
                active_hitter_history_count=0,
                recent_team_games_count=int(len(recent_team_games)),
                eligibility_mode=eligibility_mode,
                last_completed_team_game=last_completed_team_game,
                days_since_last_team_game=days_since_last_team_game,
                eligible_count=0,
                selected_count=0,
                hitters_per_team=hitters_per_team,
                excluded_rows=pd.DataFrame(columns=["batter_id", "batter_name", "last_game_date"]),
            )
            return []

    if recent_team_games.empty:
        eligible_rows = team_history.iloc[0:0].copy()
    else:
        eligible_rows = team_history.merge(
            recent_team_games,
            on=["game_date", "game_pk"],
            how="inner",
        )
    ranked_recent = _summarize_hitter_pool(eligible_rows) if not eligible_rows.empty else pd.DataFrame()
    selected = ranked_recent.head(hitters_per_team).to_dict(orient="records") if not ranked_recent.empty else []

    ranked_full_history = _summarize_hitter_pool(team_history)
    eligible_ids = set(ranked_recent["batter_id"].dropna().astype(int).tolist()) if not ranked_recent.empty else set()
    excluded_rows = ranked_full_history[
        ~ranked_full_history["batter_id"].dropna().astype(int).isin(eligible_ids)
    ].copy()

    _print_lineup_selection_diagnostics(
        team_code=team_code,
        active_roster_count=active_roster_count,
        projected_lineup_count=projected_lineup_count,
        active_hitter_history_count=int(team_history["batter_id"].dropna().nunique()),
        recent_team_games_count=int(len(recent_team_games)),
        eligibility_mode=eligibility_mode,
        last_completed_team_game=last_completed_team_game,
        days_since_last_team_game=days_since_last_team_game,
        eligible_count=int(len(ranked_recent)),
        selected_count=int(len(selected)),
        hitters_per_team=hitters_per_team,
        excluded_rows=excluded_rows,
    )
    selected_name_source = projected_lineup if projected_lineup is not None else active_roster
    lineup_order_lookup: dict[int, int | None] = {}
    if projected_lineup is not None and not projected_lineup.empty and "batting_order" in projected_lineup.columns:
        lineup_order_lookup = {
            int(row.batter_id): _coerce_int(row.batting_order)
            for row in projected_lineup[["batter_id", "batting_order"]].drop_duplicates().itertuples(index=False)
            if pd.notna(row.batter_id)
        }
    if selected_name_source is not None and selected:
        roster_name_lookup = {
            int(row.batter_id): str(row.batter_name)
            for row in selected_name_source[["batter_id", "batter_name"]].drop_duplicates().itertuples(index=False)
            if pd.notna(row.batter_id)
        }
        for row in selected:
            batter_id = int(row["batter_id"])
            if batter_id in roster_name_lookup and roster_name_lookup[batter_id]:
                row["batter_name"] = roster_name_lookup[batter_id]
    for index, row in enumerate(selected, start=1):
        batter_id = int(row["batter_id"])
        row["batting_order"] = lineup_order_lookup.get(batter_id)
        row["lineup_source"] = "confirmed" if projected_lineup is not None and not projected_lineup.empty else "projected"
        row["projected_lineup_rank"] = index
    return selected[:hitters_per_team]


def build_batter_history_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "game_pk",
        "game_date",
        "batter_id",
        "batter_name",
        "team",
        "opponent",
        "is_home",
        "bat_side",
        "pitcher_hand",
        "pitch_hand_primary",
        "pa_count",
        "hr_count",
        "hit_hr",
        "bbe_count",
        "barrel_count",
        "hard_hit_bbe_count",
        "avg_exit_velocity",
        "max_exit_velocity",
        "ev_95plus_bbe_count",
        "fly_ball_bbe_count",
        "pull_air_bbe_count",
        "ballpark",
    ]
    history = dataset_df[[column for column in columns if column in dataset_df.columns]].copy()
    history["game_date"] = pd.to_datetime(history["game_date"])
    history["batter_id"] = pd.to_numeric(history["batter_id"], errors="coerce").astype("Int64")
    return history


def build_pitcher_history_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    base = dataset_df.copy()
    base["game_date"] = pd.to_datetime(base["game_date"])
    base["is_rhb"] = base["bat_side"].eq("R").astype(int)
    base["is_lhb"] = base["bat_side"].eq("L").astype(int)
    base["avg_exit_velocity_num"] = base["avg_exit_velocity"].fillna(0.0) * base["bbe_count"].fillna(0.0)

    for side in ["rhb", "lhb"]:
        mask = base[f"is_{side[0]}{side[1:]}"] if f"is_{side[0]}{side[1:]}" in base.columns else base["bat_side"].eq(side[0].upper()).astype(int)
        base[f"pa_against_{side}"] = base["pa_count"] * mask
        base[f"hr_allowed_{side}"] = base["hr_count"] * mask
        base[f"bbe_allowed_{side}"] = base["bbe_count"] * mask
        base[f"barrels_allowed_{side}"] = base["barrel_count"] * mask
        base[f"hard_hit_bbe_allowed_{side}"] = base["hard_hit_bbe_count"] * mask
        base[f"ev_95plus_bbe_allowed_{side}"] = base["ev_95plus_bbe_count"] * mask

    grouped = (
        base.groupby(["game_pk", "game_date", "pitcher_id"], dropna=False)
        .agg(
            pitcher_name=("opp_pitcher_name", "first"),
            p_throws=("pitch_hand_primary", lambda s: latest_non_null(s, default=None)),
            pa_against=("pa_count", "sum"),
            hr_allowed=("hr_count", "sum"),
            bbe_allowed=("bbe_count", "sum"),
            barrels_allowed=("barrel_count", "sum"),
            hard_hit_bbe_allowed=("hard_hit_bbe_count", "sum"),
            avg_ev_allowed_num=("avg_exit_velocity_num", "sum"),
            max_ev_allowed=("max_exit_velocity", "max"),
            ev_95plus_bbe_allowed=("ev_95plus_bbe_count", "sum"),
            pa_against_rhb=("pa_against_rhb", "sum"),
            pa_against_lhb=("pa_against_lhb", "sum"),
            hr_allowed_rhb=("hr_allowed_rhb", "sum"),
            hr_allowed_lhb=("hr_allowed_lhb", "sum"),
            bbe_allowed_rhb=("bbe_allowed_rhb", "sum"),
            bbe_allowed_lhb=("bbe_allowed_lhb", "sum"),
            barrels_allowed_rhb=("barrels_allowed_rhb", "sum"),
            barrels_allowed_lhb=("barrels_allowed_lhb", "sum"),
            hard_hit_bbe_allowed_rhb=("hard_hit_bbe_allowed_rhb", "sum"),
            hard_hit_bbe_allowed_lhb=("hard_hit_bbe_allowed_lhb", "sum"),
            ev_95plus_bbe_allowed_rhb=("ev_95plus_bbe_allowed_rhb", "sum"),
            ev_95plus_bbe_allowed_lhb=("ev_95plus_bbe_allowed_lhb", "sum"),
        )
        .reset_index()
    )
    grouped["avg_ev_allowed"] = np.where(
        grouped["bbe_allowed"] > 0,
        grouped["avg_ev_allowed_num"] / grouped["bbe_allowed"],
        np.nan,
    )
    grouped = grouped.drop(columns=["avg_ev_allowed_num"])
    grouped["pitcher_id"] = pd.to_numeric(grouped["pitcher_id"], errors="coerce").astype("Int64")
    return grouped.sort_values(["pitcher_id", "game_date", "game_pk"]).reset_index(drop=True)


def fetch_forecast_weather(home_teams: list[str], target_date: str) -> pd.DataFrame:
    def _empty_weather_row(home_team: str) -> dict[str, Any]:
        return {
            "game_date": target_date,
            "home_team": home_team,
            "temperature_f": None,
            "humidity_pct": None,
            "wind_speed_mph": None,
            "wind_direction_deg": None,
            "weather_code": None,
            "weather_label": "Unknown",
            "pressure_hpa": None,
        }

    rows: list[dict[str, Any]] = []
    fallback_home_teams: list[str] = []
    target_timestamp = pd.to_datetime(target_date, errors="coerce")
    historical_target = pd.notna(target_timestamp) and target_timestamp.date() < eastern_today().date()
    for home_team in sorted({normalize_team_code(team) for team in home_teams if team}):
        park = PARKS.get(home_team)
        if park is None:
            rows.append(_empty_weather_row(home_team))
            continue
        if historical_target:
            rows.append(_empty_weather_row(home_team))
            continue
        response = None
        for attempt in range(1, OPEN_METEO_FORECAST_MAX_ATTEMPTS + 1):
            try:
                response = requests.get(
                    OPEN_METEO_FORECAST_URL,
                    params={
                        "latitude": park["lat"],
                        "longitude": park["lon"],
                        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,weather_code,surface_pressure",
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "timezone": park["tz"],
                        "start_date": target_date,
                        "end_date": target_date,
                    },
                    timeout=OPEN_METEO_FORECAST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == OPEN_METEO_FORECAST_MAX_ATTEMPTS:
                    print(
                        "Open-Meteo forecast lookup failed "
                        f"for {home_team} on {target_date} after {attempt} attempts: {exc}. "
                        "Falling back to null weather values."
                    )
                    fallback_home_teams.append(home_team)
                else:
                    time.sleep(OPEN_METEO_FORECAST_RETRY_BACKOFF_SECONDS * attempt)
        if response is None:
            rows.append(_empty_weather_row(home_team))
            continue
        hourly = response.json().get("hourly", {})
        weather = pd.DataFrame(hourly)
        if weather.empty:
            rows.append(_empty_weather_row(home_team))
            continue
        weather["timestamp"] = pd.to_datetime(weather["time"])
        weather["hour_diff"] = (weather["timestamp"].dt.hour - DEFAULT_GAME_HOUR_LOCAL).abs()
        best = weather.sort_values("hour_diff").iloc[0]
        rows.append(
            {
                "game_date": target_date,
                "home_team": home_team,
                "temperature_f": serialize_for_json(float(best.get("temperature_2m"))) if pd.notna(best.get("temperature_2m")) else None,
                "humidity_pct": serialize_for_json(float(best.get("relative_humidity_2m"))) if pd.notna(best.get("relative_humidity_2m")) else None,
                "wind_speed_mph": serialize_for_json(float(best.get("wind_speed_10m"))) if pd.notna(best.get("wind_speed_10m")) else None,
                "wind_direction_deg": serialize_for_json(float(best.get("wind_direction_10m"))) if pd.notna(best.get("wind_direction_10m")) else None,
                "weather_code": _coerce_int(best.get("weather_code")),
                "weather_label": weather_code_label(best.get("weather_code")),
                "pressure_hpa": serialize_for_json(float(best.get("surface_pressure"))) if pd.notna(best.get("surface_pressure")) else None,
            }
        )
    forecast_df = pd.DataFrame(rows)
    if fallback_home_teams:
        forecast_df.attrs["operational_alerts"] = [
            {
                "kind": "warning",
                "code": "weather_forecast_unavailable",
                "title": "Weather data incomplete",
                "message": (
                    f"Weather forecast data was unavailable for {', '.join(sorted(set(fallback_home_teams)))} on {target_date}. "
                    "This slate was generated with null weather inputs for those parks. Rerun later for a weather-refreshed draft."
                ),
                "target_date": target_date,
                "teams": sorted(set(fallback_home_teams)),
            }
        ]
    return forecast_df


def build_live_candidate_frame(
    dataset_df: pd.DataFrame,
    schedule_games: list[dict[str, Any]],
    *,
    target_date: str,
    hitters_per_team: int = 9,
    active_roster_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    print_weather_join_contract("live publish forecast merge")
    slate_state = build_slate_state(schedule_games)
    forecast_weather = fetch_forecast_weather(
        [game["home_team"] for game in schedule_games],
        target_date=target_date,
    )
    audit_weather_feature_coverage(
        forecast_weather,
        context="live forecast lookup table",
        feature_columns=list(PRIMARY_WEATHER_FEATURE_COLUMNS),
        fail_on_missing_columns=False,
        fail_on_all_null=False,
    )
    rows: list[dict[str, Any]] = []
    for game in schedule_games:
        slate_game = slate_state["games_by_pk"].get(int(game["game_pk"])) if game.get("game_pk") is not None else dict(game)
        matchup_specs = [
            {
                "batting_team": game["away_team"],
                "opponent_team": game["home_team"],
                "is_home": 0,
                "pitcher_id": game["home_pitcher_id"],
                "pitcher_name": game["home_pitcher_name"],
                "lineup_source": str(game.get("away_lineup_source") or "projected"),
                "projected_lineup": pd.DataFrame(
                    game.get("away_projected_lineup") or [],
                    columns=["batter_id", "batter_name", "batting_order"],
                )
                if game.get("away_projected_lineup")
                else None,
            },
            {
                "batting_team": game["home_team"],
                "opponent_team": game["away_team"],
                "is_home": 1,
                "pitcher_id": game["away_pitcher_id"],
                "pitcher_name": game["away_pitcher_name"],
                "lineup_source": str(game.get("home_lineup_source") or "projected"),
                "projected_lineup": pd.DataFrame(
                    game.get("home_projected_lineup") or [],
                    columns=["batter_id", "batter_name", "batting_order"],
                )
                if game.get("home_projected_lineup")
                else None,
            },
        ]
        for spec in matchup_specs:
            game_meta = park_game_meta(game.get("home_team"))
            hitters = select_probable_lineup_hitters(
                dataset_df,
                team_code=spec["batting_team"],
                target_date=target_date,
                hitters_per_team=hitters_per_team,
                projected_lineup=spec["projected_lineup"],
                active_roster=active_roster_map.get(spec["batting_team"]) if active_roster_map else None,
            )
            pitcher_hand = latest_pitcher_hand(dataset_df, spec["pitcher_id"]) or fetch_player_handedness(spec["pitcher_id"])
            for hitter in hitters:
                rows.append(
                    {
                        "game_pk": int(game["game_pk"]),
                        "game_date": target_date,
                        "game_datetime": str(game.get("game_datetime") or ""),
                        "game_status": str(game.get("status") or ""),
                        "game_state": str(slate_game.get("game_state") or classify_game_state(game)),
                        "batter_id": int(hitter["batter_id"]),
                        "player_id": int(hitter["batter_id"]),
                        "batter_name": str(hitter["batter_name"]),
                        "player_name": str(hitter["batter_name"]),
                        "pitcher_id": spec["pitcher_id"],
                        "opp_pitcher_id": spec["pitcher_id"],
                        "pitcher_name": spec["pitcher_name"],
                        "opp_pitcher_name": spec["pitcher_name"],
                        "team": spec["batting_team"],
                        "opponent": spec["opponent_team"],
                        "is_home": int(spec["is_home"]),
                        "lineup_source": str(hitter.get("lineup_source") or spec["lineup_source"]),
                        "batting_order": _coerce_int(hitter.get("batting_order")),
                        "bat_side": hitter.get("bat_side"),
                        "pitcher_hand": pitcher_hand,
                        "pitch_hand_primary": pitcher_hand,
                        "home_team": str(game.get("home_team") or ""),
                        "pa_count": 0,
                        "hr_count": 0,
                        "hit_hr": 0,
                        "bbe_count": 0,
                        "barrel_count": 0,
                        "hard_hit_bbe_count": 0,
                        "avg_exit_velocity": np.nan,
                        "max_exit_velocity": np.nan,
                        "ev_95plus_bbe_count": 0,
                        "fly_ball_bbe_count": 0,
                        "pull_air_bbe_count": 0,
                        "ballpark": game_meta["ballpark_name"],
                        "ballpark_name": game_meta["ballpark_name"],
                        "ballpark_region_abbr": game_meta["ballpark_region_abbr"],
                        "field_bearing_deg": game_meta["field_bearing_deg"],
                    }
                )
    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return candidate_df

    candidate_df = candidate_df.merge(
        forecast_weather,
        on=["game_date", "home_team"],
        how="left",
        validate="many_to_one",
    )
    candidate_df["platoon_advantage"] = np.where(
        candidate_df["bat_side"].notna() & candidate_df["pitch_hand_primary"].notna(),
        (candidate_df["bat_side"] != candidate_df["pitch_hand_primary"]).astype(float),
        np.nan,
    )
    candidate_df.attrs["operational_alerts"] = list(forecast_weather.attrs.get("operational_alerts") or [])
    audit_weather_feature_coverage(
        candidate_df,
        context="live candidate frame after forecast merge",
        feature_columns=list(PRIMARY_WEATHER_FEATURE_COLUMNS),
        fail_on_missing_columns=True,
        fail_on_all_null=False,
    )
    return candidate_df


def build_lineup_panels(
    dataset_df: pd.DataFrame,
    schedule_games: list[dict[str, Any]],
    *,
    target_date: str,
    hitters_per_team: int = 9,
    current_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not schedule_games:
        return []
    slate_state = build_slate_state(schedule_games)
    active_roster_map = build_active_roster_map(schedule_games)
    current_rows = list(current_rows or [])
    panels: list[dict[str, Any]] = []
    for game in slate_state["games"]:
        team_panels: list[dict[str, Any]] = []
        for team_key, opponent_key, lineup_key, source_key in [
            ("away_team", "home_team", "away_projected_lineup", "away_lineup_source"),
            ("home_team", "away_team", "home_projected_lineup", "home_lineup_source"),
        ]:
            schedule_lineup_rows = game.get(lineup_key) or []
            schedule_lineup = (
                pd.DataFrame(schedule_lineup_rows, columns=["batter_id", "batter_name", "batting_order"])
                if schedule_lineup_rows
                else None
            )
            team_code = str(game.get(team_key) or "")
            hitters = select_probable_lineup_hitters(
                dataset_df,
                team_code=team_code,
                target_date=target_date,
                hitters_per_team=hitters_per_team,
                projected_lineup=schedule_lineup,
                active_roster=active_roster_map.get(team_code),
            )
            current_team_rows = [
                row
                for row in current_rows
                if _coerce_int(row.get("game_pk")) == _coerce_int(game.get("game_pk")) and str(row.get("team") or "") == team_code
            ]
            selected_batter_ids = {
                _coerce_int(row.get("batter_id"))
                for row in current_team_rows
                if _coerce_int(row.get("batter_id")) is not None
            }
            hitters_payload = sorted(
                [
                    {
                        "batter_id": _coerce_int(row.get("batter_id")),
                        "batter_name": str(row.get("batter_name") or ""),
                        "batting_order": _coerce_int(row.get("batting_order")),
                        "selected_for_pick": _coerce_int(row.get("batter_id")) in selected_batter_ids,
                    }
                    for row in hitters
                ],
                key=lambda row: (
                    row["batting_order"] is None,
                    row["batting_order"] or 999,
                    str(row["batter_name"]),
                ),
            )
            team_panels.append(
                {
                    "team": team_code,
                    "opponent_team": str(game.get(opponent_key) or ""),
                    "lineup_source": str(game.get(source_key) or "projected"),
                    "hitters": hitters_payload,
                }
            )
        panels.append(
            {
                "game_pk": _coerce_int(game.get("game_pk")),
                "game_date": str(game.get("game_date") or target_date),
                "game_datetime": str(game.get("game_datetime") or ""),
                "game_status": str(game.get("status") or ""),
                "game_state": str(game.get("game_state") or "pregame"),
                "matchup": f"{str(game.get('away_team') or '')} @ {str(game.get('home_team') or '')}",
                "teams": team_panels,
            }
        )
    return panels


LIVE_CONTEXT_FEATURES = {"temperature_f", "wind_speed_mph", "humidity_pct", "platoon_advantage"}
LIVE_DERIVED_FEATURES = {
    "park_factor_hr_vs_batter_hand",
    "batter_hr_per_pa_vs_pitcher_hand",
    "batter_barrels_per_pa_vs_pitcher_hand",
    "pitcher_hr_allowed_per_pa_vs_batter_hand",
    "pitcher_barrels_allowed_per_bbe_vs_batter_hand",
    "split_matchup_hr",
    "split_matchup_barrel",
    "split_matchup_hard_hit",
}
LIVE_BATTER_FEATURES = [
    feature
    for feature in LIVE_PRODUCTION_FEATURE_COLUMNS
    if feature not in LIVE_CONTEXT_FEATURES and not feature.startswith("pitcher_")
]
LIVE_PITCHER_FEATURES = [feature for feature in LIVE_PRODUCTION_FEATURE_COLUMNS if feature.startswith("pitcher_")]


def build_latest_feature_snapshot(
    dataset_df: pd.DataFrame,
    *,
    entity_key: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    available_features = [feature for feature in feature_columns if feature in dataset_df.columns]
    if entity_key not in dataset_df.columns or not available_features:
        return pd.DataFrame(columns=[entity_key, *available_features])
    snapshot = dataset_df[[entity_key, "game_date", "game_pk", *available_features]].copy()
    snapshot[entity_key] = pd.to_numeric(snapshot[entity_key], errors="coerce").astype("Int64")
    snapshot["game_date"] = pd.to_datetime(snapshot["game_date"])
    snapshot = snapshot.dropna(subset=[entity_key]).sort_values([entity_key, "game_date", "game_pk"])
    rows: list[dict[str, Any]] = []
    for entity_value, group in snapshot.groupby(entity_key, dropna=True):
        row: dict[str, Any] = {entity_key: int(entity_value)}
        for feature in available_features:
            row[feature] = latest_non_null(group[feature], default=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_latest_ballpark_snapshot(
    dataset_df: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    available_features = [feature for feature in feature_columns if feature in dataset_df.columns]
    if "ballpark" not in dataset_df.columns or not available_features:
        return pd.DataFrame(columns=["ballpark", *available_features])
    snapshot = dataset_df[["ballpark", "game_date", "game_pk", *available_features]].copy()
    snapshot["game_date"] = pd.to_datetime(snapshot["game_date"])
    snapshot = snapshot.dropna(subset=["ballpark"]).sort_values(["ballpark", "game_date", "game_pk"])
    rows: list[dict[str, Any]] = []
    for ballpark, group in snapshot.groupby("ballpark", dropna=True):
        row: dict[str, Any] = {"ballpark": ballpark}
        for feature in available_features:
            row[feature] = latest_non_null(group[feature], default=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def fill_missing_features_from_snapshot(
    frame: pd.DataFrame,
    *,
    snapshot_df: pd.DataFrame,
    entity_key: str,
    feature_columns: list[str],
    suffix: str,
) -> pd.DataFrame:
    available_features = [feature for feature in feature_columns if feature in snapshot_df.columns]
    if frame.empty or snapshot_df.empty or not available_features:
        return frame
    rename_map = {feature: f"{feature}_{suffix}" for feature in available_features}
    merged = frame.merge(
        snapshot_df[[entity_key, *available_features]].rename(columns=rename_map),
        on=entity_key,
        how="left",
        validate="many_to_one",
    )
    for feature in available_features:
        fallback_column = rename_map[feature]
        if feature not in merged.columns:
            merged[feature] = np.nan
        merged[feature] = merged[feature].where(merged[feature].notna(), merged[fallback_column])
    drop_columns = list(rename_map.values())
    return merged.drop(columns=drop_columns)


def build_live_feature_frame(dataset_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()
    candidate_df = candidate_df.copy()
    candidate_df["game_date"] = pd.to_datetime(candidate_df["game_date"])
    batter_history = build_batter_history_table(dataset_df)
    pitcher_history = build_pitcher_history_table(dataset_df)

    batter_augmented = pd.concat([batter_history, candidate_df[batter_history.columns]], ignore_index=True)
    candidate_pitchers = (
        candidate_df[["game_pk", "game_date", "pitcher_id", "pitcher_name", "pitch_hand_primary"]]
        .drop_duplicates()
        .rename(columns={"pitch_hand_primary": "p_throws"})
        .copy()
    )
    for column in [
        "pa_against",
        "hr_allowed",
        "bbe_allowed",
        "barrels_allowed",
        "hard_hit_bbe_allowed",
        "avg_ev_allowed",
        "max_ev_allowed",
        "ev_95plus_bbe_allowed",
        "pa_against_rhb",
        "pa_against_lhb",
        "hr_allowed_rhb",
        "hr_allowed_lhb",
        "bbe_allowed_rhb",
        "bbe_allowed_lhb",
        "barrels_allowed_rhb",
        "barrels_allowed_lhb",
        "hard_hit_bbe_allowed_rhb",
        "hard_hit_bbe_allowed_lhb",
        "ev_95plus_bbe_allowed_rhb",
        "ev_95plus_bbe_allowed_lhb",
    ]:
        candidate_pitchers[column] = 0.0 if column != "avg_ev_allowed" and column != "max_ev_allowed" else np.nan
    pitcher_augmented = pd.concat([pitcher_history, candidate_pitchers[pitcher_history.columns]], ignore_index=True)

    batter_features = compute_batter_trailing_features(batter_augmented)
    pitcher_features = compute_pitcher_trailing_features(pitcher_augmented)
    batter_split_features = compute_batter_handedness_split_features(batter_augmented)
    pitcher_split_features = compute_pitcher_handedness_split_features(pitcher_augmented)
    featured = candidate_df.merge(
        batter_features,
        on=["batter_id", "game_pk"],
        how="left",
        validate="one_to_one",
    )
    featured = featured.merge(
        pitcher_features,
        on=["pitcher_id", "game_pk"],
        how="left",
        validate="many_to_one",
    )
    featured = featured.merge(
        batter_split_features,
        on=["batter_id", "game_pk"],
        how="left",
        validate="one_to_one",
    )
    featured = featured.merge(
        pitcher_split_features,
        on=["pitcher_id", "game_pk"],
        how="left",
        validate="many_to_one",
    )
    batter_snapshot = build_latest_feature_snapshot(
        dataset_df,
        entity_key="batter_id",
        feature_columns=[*LIVE_BATTER_FEATURES, *LIVE_BATTER_SPLIT_SOURCE_COLUMNS, *LIVE_SHRUNK_SNAPSHOT_COLUMNS],
    )
    featured = fill_missing_features_from_snapshot(
        featured,
        snapshot_df=batter_snapshot,
        entity_key="batter_id",
        feature_columns=[*LIVE_BATTER_FEATURES, *LIVE_BATTER_SPLIT_SOURCE_COLUMNS, *LIVE_SHRUNK_SNAPSHOT_COLUMNS],
        suffix="latest_batter",
    )
    pitcher_snapshot = build_latest_feature_snapshot(
        dataset_df,
        entity_key="pitcher_id",
        feature_columns=[*LIVE_PITCHER_FEATURES, *LIVE_PITCHER_SPLIT_SOURCE_COLUMNS],
    )
    featured = fill_missing_features_from_snapshot(
        featured,
        snapshot_df=pitcher_snapshot,
        entity_key="pitcher_id",
        feature_columns=[*LIVE_PITCHER_FEATURES, *LIVE_PITCHER_SPLIT_SOURCE_COLUMNS],
        suffix="latest_pitcher",
    )
    ballpark_snapshot = build_latest_ballpark_snapshot(
        dataset_df,
        feature_columns=list(LIVE_PARK_FACTOR_SOURCE_COLUMNS),
    )
    featured = fill_missing_features_from_snapshot(
        featured,
        snapshot_df=ballpark_snapshot,
        entity_key="ballpark",
        feature_columns=list(LIVE_PARK_FACTOR_SOURCE_COLUMNS),
        suffix="latest_ballpark",
    )
    featured = build_matchup_selected_handedness_features(featured)
    if "park_factor_hr_vs_lhb" not in featured.columns:
        featured["park_factor_hr_vs_lhb"] = np.nan
    if "park_factor_hr_vs_rhb" not in featured.columns:
        featured["park_factor_hr_vs_rhb"] = np.nan
    featured["park_factor_hr_vs_batter_hand"] = np.where(
        featured["bat_side"].eq("L"),
        featured["park_factor_hr_vs_lhb"],
        featured["park_factor_hr_vs_rhb"],
    )
    featured["opponent_team"] = featured["opponent"]
    live_feature_audit = _audit_feature_frame_columns(
        featured,
        feature_columns=list(LIVE_COMPATIBLE_FEATURE_COLUMNS),
        context="live feature frame",
    )
    for feature in LIVE_COMPATIBLE_FEATURE_COLUMNS:
        if feature not in featured.columns:
            featured[feature] = np.nan
    featured.attrs["live_plus_feature_audit"] = live_feature_audit
    return featured


def score_live_candidates(
    candidate_df: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    max_picks: int = 20,
    min_confidence_tier: str | None = None,
    max_picks_per_team: int | None = None,
    max_picks_per_game: int | None = None,
    published_at: str | None = None,
) -> list[dict[str, Any]]:
    if candidate_df.empty:
        return []

    feature_columns = list(bundle["feature_columns"])
    model = bundle["model"]
    reference_df = bundle["reference_df"]

    scored = candidate_df.copy().reset_index(drop=True)
    for feature in feature_columns:
        if feature not in scored.columns:
            scored[feature] = np.nan

    feature_profile = str(bundle.get("feature_profile") or "")
    if feature_profile == "live_plus":
        required_live_features = [
            feature for feature in feature_columns if feature in LIVE_PLUS_ONLY_FEATURE_COLUMNS
        ]
    elif feature_profile == "live_shrunk":
        required_live_features = [
            feature for feature in feature_columns if feature in LIVE_SHRUNK_ONLY_FEATURE_COLUMNS
        ]
    elif feature_profile == "live_shrunk_precise":
        required_live_features = [
            feature for feature in feature_columns if feature in LIVE_SHRUNK_PRECISE_ONLY_FEATURE_COLUMNS
        ]
    else:
        required_live_features = []
    if required_live_features:
        readiness = _audit_feature_frame_columns(
            scored,
            feature_columns=required_live_features,
            context=f"{feature_profile} scoring frame",
        )
        if readiness["missing_columns"] or readiness["all_null_columns"]:
            raise RuntimeError(
                f"{feature_profile} candidate frame is not ready for scoring; "
                f"missing={readiness['missing_columns']}, all_null={readiness['all_null_columns']}"
            )

    probabilities = model.predict_proba(scored[feature_columns])[:, 1]
    scored["predicted_hr_probability"] = probabilities
    scored["predicted_hr_percentile"] = scored["predicted_hr_probability"].rank(method="first", pct=True)
    scored["predicted_hr_score"] = (scored["predicted_hr_percentile"] * 100.0).round(1)
    scored["confidence_tier"] = scored["predicted_hr_percentile"].apply(confidence_tier_from_percentile)

    coef_map = extract_logistic_coefficient_map(model, feature_columns)
    positive_coef_map = {feature: value for feature, value in coef_map.items() if value > 0}
    reasons = scored.apply(
        lambda row: generate_reason_strings(row, reference_df=reference_df, positive_coef_map=positive_coef_map, max_reasons=3),
        axis=1,
    )
    scored["top_reason_1"] = reasons.apply(lambda items: items[0] if len(items) > 0 else "")
    scored["top_reason_2"] = reasons.apply(lambda items: items[1] if len(items) > 1 else "")
    scored["top_reason_3"] = reasons.apply(lambda items: items[2] if len(items) > 2 else "")

    ranked_source = scored.sort_values(
        ["predicted_hr_score", "predicted_hr_probability", "batter_name"],
        ascending=[False, False, True],
    ).copy()

    min_tier_value = _confidence_tier_value(min_confidence_tier) if min_confidence_tier else -1
    selected_indices: list[int] = []
    team_counts: dict[str, int] = {}
    game_counts: dict[int, int] = {}

    for row in ranked_source.itertuples():
        if min_tier_value >= 0 and _confidence_tier_value(getattr(row, "confidence_tier", None)) < min_tier_value:
            continue

        team_key = str(getattr(row, "team", "") or "")
        if max_picks_per_team is not None and team_key:
            if team_counts.get(team_key, 0) >= max_picks_per_team:
                continue

        game_pk = getattr(row, "game_pk", None)
        game_key = int(game_pk) if game_pk is not None and not pd.isna(game_pk) else None
        if max_picks_per_game is not None and game_key is not None:
            if game_counts.get(game_key, 0) >= max_picks_per_game:
                continue

        selected_indices.append(int(row.Index))
        if team_key:
            team_counts[team_key] = team_counts.get(team_key, 0) + 1
        if game_key is not None:
            game_counts[game_key] = game_counts.get(game_key, 0) + 1
        if len(selected_indices) >= max_picks:
            break

    ranked = ranked_source.loc[selected_indices].copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    publish_time = published_at or datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    for row in ranked.to_dict(orient="records"):
        game_date = normalize_game_date(row["game_date"])
        rows.append(
            {
                "pick_id": build_pick_id(
                    game_date=game_date,
                    game_pk=int(row["game_pk"]),
                    batter_id=int(row["batter_id"]),
                    batter_name=str(row["batter_name"]),
                    pitcher_id=int(row["pitcher_id"]) if row.get("pitcher_id") is not None and not pd.isna(row.get("pitcher_id")) else None,
                    pitcher_name=str(row.get("pitcher_name") or ""),
                ),
                "published_at": publish_time,
                "game_pk": int(row["game_pk"]),
                "game_date": game_date,
                "game_datetime": str(row.get("game_datetime") or ""),
                "game_status": str(row.get("game_status") or ""),
                "game_state": str(row.get("game_state") or "pregame"),
                "rank": int(row["rank"]),
                "batter_id": int(row["batter_id"]),
                "batter_name": str(row["batter_name"]),
                "team": str(row["team"]),
                "opponent_team": str(row["opponent_team"]),
                "pitcher_id": int(row["pitcher_id"]) if row.get("pitcher_id") is not None and not pd.isna(row.get("pitcher_id")) else None,
                "pitcher_name": str(row.get("pitcher_name") or ""),
                "confidence_tier": str(row["confidence_tier"]),
                "predicted_hr_probability": serialize_for_json(float(row["predicted_hr_probability"])),
                "predicted_hr_score": serialize_for_json(float(row["predicted_hr_score"])),
                "top_reason_1": str(row["top_reason_1"]),
                "top_reason_2": str(row["top_reason_2"]),
                "top_reason_3": str(row["top_reason_3"]),
                "lineup_source": str(row.get("lineup_source") or "projected"),
                "batting_order": _coerce_int(row.get("batting_order")),
                "ballpark_name": str(row.get("ballpark_name") or row.get("ballpark") or ""),
                "ballpark_region_abbr": str(row.get("ballpark_region_abbr") or ""),
                "weather_code": _coerce_int(row.get("weather_code")),
                "weather_label": str(row.get("weather_label") or weather_code_label(row.get("weather_code"))),
                "temperature_f": _coerce_float(row.get("temperature_f")),
                "wind_speed_mph": _coerce_float(row.get("wind_speed_mph")),
                "wind_direction_deg": _coerce_float(row.get("wind_direction_deg")),
                "field_bearing_deg": _coerce_float(row.get("field_bearing_deg")),
                "result": "Pending",
            }
        )
    return rows


def settle_pick_records(
    records: list[dict[str, Any]],
    dataset_df: pd.DataFrame,
    *,
    resolved_through_date: str,
    schedule_games: list[dict[str, Any]] | None = None,
    reference_time: datetime | None = None,
) -> list[dict[str, Any]]:
    slate_state = build_slate_state(schedule_games or [], reference_time=reference_time)
    games_by_pk = slate_state["games_by_pk"]
    resolved_lookup = {
        (normalize_game_date(row.game_date), int(row.batter_id)): int(row.hit_hr)
        for row in dataset_df[["game_date", "batter_id", "hit_hr"]].drop_duplicates().itertuples(index=False)
        if pd.notna(row.batter_id)
    }
    settled: list[dict[str, Any]] = []
    for row in records:
        game_date = normalize_game_date(row.get("game_date"))
        batter_id = row.get("batter_id")
        current_result = str(row.get("result") or row.get("result_label") or "Pending")
        if current_result in {"HR", "No HR"}:
            updated = dict(row)
            game_pk = _coerce_int(row.get("game_pk"))
            schedule_game = games_by_pk.get(game_pk) if game_pk is not None else None
            if schedule_game:
                updated["game_status"] = str(schedule_game.get("status") or updated.get("game_status") or "")
                updated["game_state"] = str(schedule_game.get("game_state") or updated.get("game_state") or "final")
            settled.append(updated)
            continue
        if not game_date or game_date > resolved_through_date:
            updated = dict(row)
            game_pk = _coerce_int(row.get("game_pk"))
            schedule_game = games_by_pk.get(game_pk) if game_pk is not None else None
            if schedule_game:
                updated["game_status"] = str(schedule_game.get("status") or updated.get("game_status") or "")
                updated["game_state"] = str(schedule_game.get("game_state") or updated.get("game_state") or "pregame")
            settled.append(updated)
            continue

        game_pk = _coerce_int(row.get("game_pk"))
        schedule_game = games_by_pk.get(game_pk) if game_pk is not None else None
        if schedule_game is not None:
            game_state = str(schedule_game.get("game_state") or classify_game_state(schedule_game, reference_time))
        elif row:
            game_state = str(classify_game_state(row, reference_time))
        else:
            game_state = "pregame"
        lookup_value = resolved_lookup.get((game_date, int(batter_id))) if batter_id is not None else None
        if lookup_value is not None and int(lookup_value) == 1:
            resolved_hit = 1
            resolved_label = "HR"
        elif game_state == "final":
            resolved_hit = 0
            resolved_label = "No HR"
        else:
            resolved_hit = None
            resolved_label = "Pending"
        updated = dict(row)
        updated["result"] = resolved_label
        updated["result_label"] = resolved_label
        updated["actual_hit_hr"] = resolved_hit
        updated["game_status"] = str(schedule_game.get("status") or updated.get("game_status") or "") if schedule_game else str(updated.get("game_status") or "")
        updated["game_state"] = game_state
        settled.append(updated)
    return settled


def write_current_picks(rows: list[dict[str, Any]], path: Path = LIVE_CURRENT_PICKS_PATH) -> None:
    canonical_rows = canonicalize_current_pick_rows(rows)
    ordered = sorted(
        canonical_rows,
        key=lambda row: (
            int(row.get("rank") or 999),
            -(float(row.get("predicted_hr_score")) if row.get("predicted_hr_score") is not None else -999.0),
            str(row.get("batter_name") or ""),
        ),
    )
    write_json_array(path, ordered)


def load_pick_history(path: Path = LIVE_PICK_HISTORY_PATH) -> list[dict[str, Any]]:
    return load_json_array(path)


def write_pick_history(rows: list[dict[str, Any]], path: Path = LIVE_PICK_HISTORY_PATH) -> None:
    canonical_rows = canonicalize_history_pick_rows(rows)
    ordered = sorted(
        canonical_rows,
        key=lambda row: (
            str(row.get("game_date") or ""),
            -(float(row.get("predicted_hr_score")) if row.get("predicted_hr_score") is not None else -999.0),
            int(row.get("rank") or 999),
            str(row.get("batter_name") or ""),
        ),
    )
    write_json_array(path, ordered)
