from __future__ import annotations

from typing import Any

import pandas as pd

from config import DEFAULT_GAME_HOUR_LOCAL

WEATHER_JOIN_KEY_COLUMNS = ["game_date", "home_team"]
PRIMARY_WEATHER_FEATURE_COLUMNS = ["temperature_f", "wind_speed_mph", "humidity_pct"]
EXTENDED_WEATHER_FEATURE_COLUMNS = [
    *PRIMARY_WEATHER_FEATURE_COLUMNS,
    "wind_direction_deg",
    "pressure_hpa",
]
WEATHER_WARNING_NULL_RATE = 0.35


def weather_join_contract() -> dict[str, Any]:
    return {
        "join_keys": list(WEATHER_JOIN_KEY_COLUMNS),
        "park_lookup": "PARKS[home_team]",
        "selection_rule": f"nearest available hour to {DEFAULT_GAME_HOUR_LOCAL}:00 local park time",
    }


def print_weather_join_contract(context: str) -> None:
    contract = weather_join_contract()
    print(f"\nWeather join contract ({context})")
    print("-" * 60)
    print(f"Join keys                    : {' + '.join(contract['join_keys'])}")
    print(f"Park lookup                  : {contract['park_lookup']}")
    print(f"Observation selection        : {contract['selection_rule']}")


def summarize_weather_feature_coverage(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    selected_features = feature_columns or list(PRIMARY_WEATHER_FEATURE_COLUMNS)
    row_count = int(len(frame))
    summary: dict[str, dict[str, float | int | None]] = {}
    for feature in selected_features:
        if feature not in frame.columns:
            summary[feature] = {
                "row_count": row_count,
                "non_null_count": 0,
                "null_rate": None,
                "present": 0,
            }
            continue
        non_null_count = int(frame[feature].notna().sum())
        null_rate = float(frame[feature].isna().mean()) if row_count else 0.0
        summary[feature] = {
            "row_count": row_count,
            "non_null_count": non_null_count,
            "null_rate": null_rate,
            "present": 1,
        }
    return summary


def audit_weather_feature_coverage(
    frame: pd.DataFrame,
    *,
    context: str,
    feature_columns: list[str] | None = None,
    warn_threshold: float = WEATHER_WARNING_NULL_RATE,
    fail_on_missing_columns: bool = False,
    fail_on_all_null: bool = False,
) -> dict[str, Any]:
    summary = summarize_weather_feature_coverage(frame, feature_columns=feature_columns)
    missing_columns = [feature for feature, stats in summary.items() if not stats["present"]]
    high_missing_features = [
        feature
        for feature, stats in summary.items()
        if stats["present"] and stats["null_rate"] is not None and float(stats["null_rate"]) > warn_threshold
    ]
    all_null_features = [
        feature
        for feature, stats in summary.items()
        if stats["present"] and int(stats["non_null_count"]) == 0
    ]

    print(f"\nWeather feature coverage ({context})")
    print("-" * 60)
    for feature, stats in summary.items():
        if not stats["present"]:
            print(f"{feature:<28} missing column")
            continue
        null_rate = float(stats["null_rate"]) if stats["null_rate"] is not None else 1.0
        print(
            f"{feature:<28} non_null={int(stats['non_null_count']):>6} / {int(stats['row_count']):<6} "
            f"null_rate={null_rate:>7.2%}"
        )

    if high_missing_features:
        print(
            "WARNING: weather columns exceeded the warning null-rate threshold "
            f"({warn_threshold:.0%}): {high_missing_features}"
        )
    if missing_columns:
        print(f"WARNING: weather columns missing from frame: {missing_columns}")
    if all_null_features:
        print(f"WARNING: weather columns are fully null in this frame: {all_null_features}")

    if fail_on_missing_columns and missing_columns:
        raise RuntimeError(f"{context} is missing required weather columns: {missing_columns}")
    if fail_on_all_null and all_null_features:
        raise RuntimeError(f"{context} contains all-null weather columns: {all_null_features}")

    return {
        "summary": summary,
        "warning_null_rate_threshold": float(warn_threshold),
        "high_missing_features": high_missing_features,
        "missing_columns": missing_columns,
        "all_null_features": all_null_features,
        "contract": weather_join_contract(),
    }
