"""Build Cloudflare Pages dashboard artifacts from published public picks."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LIVE_MODEL_DATA_PATH
from scripts.live_pipeline import build_lineup_panels, fetch_schedule_games
from train_model import extract_logistic_coefficient_map

DEFAULT_TRACKING_START_DATE = "2026-03-25"
DEFAULT_CURRENT_PICKS_PATH = Path("data/live/current_picks.json")
DEFAULT_HISTORY_PATH = Path("data/live/pick_history.json")
DEFAULT_OUTPUT_DIR = Path("cloudflare-app/data")
DEFAULT_MODEL_BUNDLE_PATH = Path("data/live/model_bundle.pkl")
DEFAULT_MODEL_DATA_PATH = LIVE_MODEL_DATA_PATH
DEFAULT_MODEL_METADATA_PATH = Path("data/live/model_metadata.json")
DEFAULT_MODEL_FAMILY = "2024-25 trained"
DEFAULT_FEATURE_PROFILE = "not available"
DEFAULT_DATA_NOTE = "Public dashboard tracking begins on Opening Night, March 25, 2026. Trained on 2024 and 2025 season data."
DEFAULT_REFRESH_SCHEDULE = {
    "timezone": "ET",
    "runs": [
        {
            "time_et": "Early morning ET",
            "type": "prepare",
            "label": "Prepare",
            "description": "Refreshes data, retrains the model once for the day, settles any remaining prior results, and saves a private draft slate.",
        },
        {
            "time_et": "Every 15 minutes pregame",
            "type": "publish",
            "label": "Publish",
            "description": "Re-runs before games lock, using projected lineups early and upgrading to confirmed lineups once MLB posts them.",
        },
        {
            "time_et": "Every 15 minutes in-game",
            "type": "settle",
            "label": "Settle",
            "description": "Checks today’s picks from first pitch through final, marks HR or No HR, and archives the slate when every game is complete.",
        },
    ],
}

DISPLAY_COLUMNS = [
    "pick_id",
    "game_pk",
    "game_date",
    "game_datetime",
    "game_state",
    "rank",
    "batter_name",
    "team",
    "opponent_team",
    "pitcher_name",
    "lineup_source",
    "batting_order",
    "ballpark_name",
    "ballpark_region_abbr",
    "confidence_tier",
    "weather_code",
    "weather_label",
    "wind_speed_mph",
    "wind_direction_deg",
    "field_bearing_deg",
    "predicted_hr_probability",
    "predicted_hr_score",
    "actual_hit_hr",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "result_label",
]
CURRENT_PICK_COLUMNS = [
    "pick_id",
    "published_at",
    "game_pk",
    "game_date",
    "game_datetime",
    "game_status",
    "game_state",
    "rank",
    "batter_id",
    "batter_name",
    "team",
    "opponent_team",
    "pitcher_id",
    "pitcher_name",
    "lineup_source",
    "batting_order",
    "ballpark_name",
    "ballpark_region_abbr",
    "weather_code",
    "weather_label",
    "wind_speed_mph",
    "wind_direction_deg",
    "field_bearing_deg",
    "confidence_tier",
    "predicted_hr_probability",
    "predicted_hr_score",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "result",
]
HISTORY_COLUMNS = [
    "pick_id",
    "published_at",
    "game_pk",
    "game_date",
    "game_datetime",
    "game_status",
    "game_state",
    "rank",
    "batter_id",
    "batter_name",
    "team",
    "opponent_team",
    "pitcher_id",
    "pitcher_name",
    "lineup_source",
    "batting_order",
    "ballpark_name",
    "ballpark_region_abbr",
    "weather_code",
    "weather_label",
    "wind_speed_mph",
    "wind_direction_deg",
    "field_bearing_deg",
    "confidence_tier",
    "predicted_hr_probability",
    "predicted_hr_score",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "result_label",
    "actual_hit_hr",
]
MODEL_FEATURE_DETAILS = {
    "hr_per_pa_last_10d": {
        "label": "HR Rate, Last 10 Days",
        "description": "Recent home-run rate over the last 10 days. This is the shortest-term power trend input.",
    },
    "hr_per_pa_last_30d": {
        "label": "HR Rate, Last 30 Days",
        "description": "Home-run rate over the last 30 days. It captures whether the hitter has been sustaining power recently.",
    },
    "barrels_per_pa_last_10d": {
        "label": "Barrel Rate, Last 10 Days",
        "description": "Recent barrel frequency, measuring how often the hitter is producing ideal home-run quality contact right now.",
    },
    "barrels_per_pa_last_30d": {
        "label": "Barrel Rate, Last 30 Days",
        "description": "Longer recent barrel trend, used as a sustained power-quality signal.",
    },
    "hard_hit_rate_last_10d": {
        "label": "Hard-Hit Rate, Last 10 Days",
        "description": "Share of recent batted balls hit hard. This helps the model separate live contact quality from raw outcomes.",
    },
    "hard_hit_rate_last_30d": {
        "label": "Hard-Hit Rate, Last 30 Days",
        "description": "30-day hard-hit trend, used as a broader measure of how consistently loud the hitter's contact has been.",
    },
    "bbe_95plus_ev_rate_last_10d": {
        "label": "95+ EV Rate, Last 10 Days",
        "description": "How often recent contact has cleared 95 mph. It is a compact proxy for current dangerous contact.",
    },
    "bbe_95plus_ev_rate_last_30d": {
        "label": "95+ EV Rate, Last 30 Days",
        "description": "30-day loud-contact rate, used to stabilize recent quality-of-contact signals.",
    },
    "avg_exit_velocity_last_10d": {
        "label": "Average Exit Velocity, Last 10 Days",
        "description": "Average recent exit velocity. It measures how hard the hitter is striking the ball lately.",
    },
    "max_exit_velocity_last_10d": {
        "label": "Peak Exit Velocity, Last 10 Days",
        "description": "Recent top-end exit velocity. This is used as a ceiling signal for raw home-run juice.",
    },
    "pitcher_hr_allowed_per_pa_last_30d": {
        "label": "Pitcher HR Allowed Rate, Last 30 Days",
        "description": "How often the opposing pitcher has allowed homers recently. It measures matchup vulnerability.",
    },
    "pitcher_barrels_allowed_per_bbe_last_30d": {
        "label": "Pitcher Barrel Rate Allowed, Last 30 Days",
        "description": "How often recent contact against the pitcher has been barreled.",
    },
    "pitcher_hard_hit_allowed_rate_last_30d": {
        "label": "Pitcher Hard-Hit Rate Allowed, Last 30 Days",
        "description": "Recent hard contact allowed by the opposing pitcher.",
    },
    "pitcher_avg_ev_allowed_last_30d": {
        "label": "Pitcher Avg Exit Velocity Allowed, Last 30 Days",
        "description": "Average exit velocity allowed by the opposing pitcher recently.",
    },
    "pitcher_95plus_ev_allowed_rate_last_30d": {
        "label": "Pitcher 95+ EV Rate Allowed, Last 30 Days",
        "description": "Rate of very hard recent contact allowed by the pitcher.",
    },
    "temperature_f": {
        "label": "Temperature",
        "description": "Forecast game-time temperature. The live model uses weather as context, not as a primary driver.",
    },
    "wind_speed_mph": {
        "label": "Wind Speed",
        "description": "Projected wind speed at the park. This gives the model weather context for carry conditions.",
    },
    "humidity_pct": {
        "label": "Humidity",
        "description": "Projected humidity at the park. It is a lighter weather-context feature.",
    },
    "platoon_advantage": {
        "label": "Platoon Advantage",
        "description": "Whether the hitter has the handedness advantage against the pitcher.",
    },
    "park_factor_hr_vs_batter_hand": {
        "label": "Park Factor vs Batter Hand",
        "description": "Home-run environment of the park for the hitter's handedness.",
    },
    "batter_hr_per_pa_vs_pitcher_hand": {
        "label": "Hitter HR Split vs Pitcher Hand",
        "description": "The hitter's historical home-run rate against pitchers of this handedness.",
    },
    "batter_barrels_per_pa_vs_pitcher_hand": {
        "label": "Hitter Barrel Split vs Pitcher Hand",
        "description": "The hitter's barrel rate against pitchers of this handedness.",
    },
    "pitcher_hr_allowed_per_pa_vs_batter_hand": {
        "label": "Pitcher HR Split Allowed vs Batter Hand",
        "description": "The pitcher's home-run susceptibility against hitters of this handedness.",
    },
    "pitcher_barrels_allowed_per_bbe_vs_batter_hand": {
        "label": "Pitcher Barrel Split Allowed vs Batter Hand",
        "description": "The pitcher's barrel susceptibility against hitters of this handedness.",
    },
    "split_matchup_hr": {
        "label": "Handedness HR Matchup Blend",
        "description": "Combined hitter split and pitcher split for home-run rate by handedness.",
    },
    "split_matchup_barrel": {
        "label": "Handedness Barrel Matchup Blend",
        "description": "Combined hitter and pitcher handedness split for barrel quality.",
    },
    "split_matchup_hard_hit": {
        "label": "Handedness Hard-Hit Matchup Blend",
        "description": "Combined hitter and pitcher handedness split for hard-contact tendency.",
    },
    "avg_launch_angle_last_50_bbe": {
        "label": "Avg Launch Angle, Last 50 BBE",
        "description": "Average launch angle over the last 50 batted-ball events. It helps distinguish fly-ball lift from pure hard contact.",
    },
}


def score_sort_value(row: dict[str, Any]) -> float:
    score = row.get("predicted_hr_score")
    if score is None:
        return float("-inf")
    return float(score)


def current_pick_sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return (-score_sort_value(row), int(row.get("rank") or 999), str(row.get("batter_name") or ""))


def history_sort_key(row: dict[str, Any]) -> tuple[str, float, int, str]:
    return (
        str(row.get("game_date") or ""),
        -score_sort_value(row),
        int(row.get("rank") or 999),
        str(row.get("batter_name") or ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-picks-path", default=str(DEFAULT_CURRENT_PICKS_PATH), help="Path to the latest published picks JSON.")
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH), help="Path to the public pick ledger JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where dashboard JSON will be written.")
    parser.add_argument("--tracking-start-date", default=DEFAULT_TRACKING_START_DATE, help="Only picks on or after this date are published.")
    parser.add_argument("--latest-count", type=int, default=12, help="Number of latest picks shown on the landing page.")
    parser.add_argument("--history-per-date", type=int, default=10, help="Number of historical picks preserved per date in the dashboard.")
    parser.add_argument("--min-player-picks", type=int, default=2, help="Minimum settled picks required for a player leaderboard row.")
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return [row for row in data if isinstance(row, dict)]


def serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return value


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_result(value: Any) -> tuple[str, int | None]:
    token = str(value or "").strip().lower()
    if token in {"1", "hr", "hit", "home_run", "home run", "success"}:
        return "HR", 1
    if token in {"0", "miss", "no hr", "no_hr", "failed", "failure"}:
        return "No HR", 0
    return "Pending", None


def build_pick_id(row: dict[str, Any]) -> str:
    game_date = str(row.get("game_date") or row.get("pick_date") or "").strip()
    game_pk = str(row.get("game_pk") or "").strip()
    batter_id = str(row.get("batter_id") or "").strip()
    batter_name = str(row.get("batter_name") or "").strip().lower().replace(" ", "-")
    pitcher_id = str(row.get("pitcher_id") or "").strip()
    pitcher_name = str(row.get("pitcher_name") or "").strip().lower().replace(" ", "-")
    identity = batter_id or batter_name
    matchup = pitcher_id or pitcher_name
    if game_pk:
        return str(row.get("pick_id") or f"{game_date}:{game_pk}:{identity}:{matchup}")
    return str(row.get("pick_id") or f"{game_date}:{identity}:{matchup}")


def normalize_pick(row: dict[str, Any], tracking_start_date: str) -> dict[str, Any] | None:
    game_date = str(row.get("game_date") or row.get("pick_date") or "").strip()
    if not game_date or game_date < tracking_start_date:
        return None

    probability = parse_float(row.get("predicted_hr_probability"))
    score = parse_float(row.get("predicted_hr_score"))
    if score is None and probability is not None:
        score = probability * 100.0

    result_label, actual_hit_hr = normalize_result(row.get("result") or row.get("result_label"))
    normalized = {
        "pick_id": build_pick_id(row),
        "published_at": str(row.get("published_at") or datetime.now(timezone.utc).isoformat()),
        "game_pk": parse_int(row.get("game_pk")),
        "game_date": game_date,
        "game_datetime": str(row.get("game_datetime") or ""),
        "game_status": str(row.get("game_status") or row.get("status") or ""),
        "game_state": str(row.get("game_state") or "pregame"),
        "rank": parse_int(row.get("rank")) or 999,
        "batter_id": parse_int(row.get("batter_id")),
        "batter_name": str(row.get("batter_name") or "Unknown hitter"),
        "team": str(row.get("team") or ""),
        "opponent_team": str(row.get("opponent_team") or row.get("opponent") or ""),
        "pitcher_id": parse_int(row.get("pitcher_id")),
        "pitcher_name": str(row.get("pitcher_name") or ""),
        "lineup_source": str(row.get("lineup_source") or "projected"),
        "batting_order": parse_int(row.get("batting_order")),
        "ballpark_name": str(row.get("ballpark_name") or row.get("ballpark") or ""),
        "ballpark_region_abbr": str(row.get("ballpark_region_abbr") or ""),
        "confidence_tier": str(row.get("confidence_tier") or "watch").lower(),
        "weather_code": parse_int(row.get("weather_code")),
        "weather_label": str(row.get("weather_label") or ""),
        "wind_speed_mph": parse_float(row.get("wind_speed_mph")),
        "wind_direction_deg": parse_float(row.get("wind_direction_deg")),
        "field_bearing_deg": parse_float(row.get("field_bearing_deg")),
        "predicted_hr_probability": probability,
        "predicted_hr_score": score,
        "top_reason_1": str(row.get("top_reason_1") or ""),
        "top_reason_2": str(row.get("top_reason_2") or ""),
        "top_reason_3": str(row.get("top_reason_3") or ""),
        "result_label": result_label,
        "actual_hit_hr": actual_hit_hr,
    }
    return normalized


def clean_current_pick_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in sorted(rows, key=current_pick_sort_key):
        cleaned.append({column: serialize_value(row.get(column)) for column in CURRENT_PICK_COLUMNS[:-1]})
        cleaned[-1]["result"] = str(row.get("result_label") or "Pending")
    return cleaned


def clean_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {column: serialize_value(row.get(column)) for column in HISTORY_COLUMNS}
        for row in sorted(rows, key=history_sort_key)
    ]


def select_active_current_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    latest_current_date = max(str(row["game_date"]) for row in rows if row.get("game_date"))
    return [row for row in rows if str(row["game_date"]) == latest_current_date]


def upsert_history(existing_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_dates = {str(row["game_date"]) for row in current_rows}
    retained_existing = [
        row
        for row in existing_rows
        if not (str(row["game_date"]) in current_dates and row.get("result_label") == "Pending")
    ]
    by_id = {str(row["pick_id"]): dict(row) for row in retained_existing}
    for row in current_rows:
        key = str(row["pick_id"])
        previous = by_id.get(key)
        if previous:
            if previous.get("result_label") in {"HR", "No HR"} and row.get("result_label") == "Pending":
                row["result_label"] = previous["result_label"]
                row["actual_hit_hr"] = previous.get("actual_hit_hr")
            by_id[key].update(row)
        else:
            by_id[key] = dict(row)
    return sorted(by_id.values(), key=history_sort_key)


def recover_pending_history_rows(
    current_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recovered_rows = [
        dict(row)
        for row in history_rows
        if str(row.get("result_label") or row.get("result") or "Pending") == "Pending"
    ]
    if not recovered_rows:
        return current_rows, history_rows

    history_without_pending = [
        dict(row)
        for row in history_rows
        if str(row.get("result_label") or row.get("result") or "Pending") != "Pending"
    ]
    current_by_id = {
        str(row.get("pick_id") or ""): dict(row)
        for row in current_rows
        if str(row.get("pick_id") or "")
    }
    for row in recovered_rows:
        key = str(row.get("pick_id") or "")
        if not key:
            continue
        current_by_id[key] = dict(row)
    repaired_current_rows = sorted(current_by_id.values(), key=current_pick_sort_key)
    return repaired_current_rows, history_without_pending


def top_k_by_date(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=history_sort_key):
        grouped.setdefault(row["game_date"], []).append(row)
    trimmed: list[dict[str, Any]] = []
    for game_date in sorted(grouped):
        trimmed.extend(grouped[game_date][:k])
    return trimmed


def summarize_confidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"elite": 0, "strong": 1, "watch": 2, "longshot": 3}
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        tier = str(row["confidence_tier"] or "watch")
        bucket = buckets.setdefault(tier, {"confidence_tier": tier, "picks": 0, "homers": 0, "prob_total": 0.0, "prob_count": 0})
        bucket["picks"] += 1
        if row["actual_hit_hr"] is not None:
            bucket["homers"] += int(row["actual_hit_hr"])
        if row["predicted_hr_probability"] is not None:
            bucket["prob_total"] += float(row["predicted_hr_probability"])
            bucket["prob_count"] += 1
    rows_out: list[dict[str, Any]] = []
    for tier, bucket in buckets.items():
        picks = int(bucket["picks"])
        rows_out.append(
            {
                "confidence_tier": tier,
                "picks": picks,
                "homers": int(bucket["homers"]),
                "hit_rate": serialize_value((bucket["homers"] / picks) if picks else None),
                "avg_probability": serialize_value((bucket["prob_total"] / bucket["prob_count"]) if bucket["prob_count"] else None),
            }
        )
    return sorted(rows_out, key=lambda row: (order.get(row["confidence_tier"], 99), row["confidence_tier"]))


def eastern_yesterday() -> str:
    return (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date().isoformat()


def build_history_date_options(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["game_date"]) for row in rows if row.get("game_date")}, reverse=True)


def build_player_leaderboard(rows: list[dict[str, Any]], *, min_player_picks: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        batter_name = str(row.get("batter_name") or "")
        team = str(row.get("team") or "")
        bucket = grouped.setdefault(
            (batter_name, team),
            {
                "batter_name": batter_name,
                "team": team,
                "picks": 0,
                "homers": 0,
            },
        )
        bucket["picks"] += 1
        bucket["homers"] += int(row.get("actual_hit_hr") or 0)
    rows_out = [
        {
            **bucket,
            "hit_rate": serialize_value((bucket["homers"] / bucket["picks"]) if bucket["picks"] else None),
        }
        for bucket in grouped.values()
        if int(bucket["picks"]) >= min_player_picks
    ]
    return sorted(
        rows_out,
        key=lambda row: (-int(row["homers"]), -int(row["picks"]), str(row["batter_name"])),
    )


def resolve_default_history_date(history_dates: list[str]) -> str:
    if not history_dates:
        return ""
    yesterday = eastern_yesterday()
    if yesterday in history_dates:
        return yesterday
    return history_dates[0]


def build_season_hr_leaders_2026(
    dataset_path: Path = DEFAULT_MODEL_DATA_PATH,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        return []

    dataset_df = pd.read_csv(dataset_path, parse_dates=["game_date"])
    if dataset_df.empty:
        return []

    player_id_column = "batter_id" if "batter_id" in dataset_df.columns else ("player_id" if "player_id" in dataset_df.columns else None)
    player_name_column = "batter_name" if "batter_name" in dataset_df.columns else ("player_name" if "player_name" in dataset_df.columns else None)
    if player_id_column is None or player_name_column is None:
        return []

    season_df = dataset_df[pd.to_datetime(dataset_df["game_date"], errors="coerce").dt.year.eq(2026)].copy()
    if season_df.empty:
        return []

    season_df[player_id_column] = pd.to_numeric(season_df[player_id_column], errors="coerce")
    sort_columns = ["game_date"]
    if "game_pk" in season_df.columns:
        sort_columns.append("game_pk")
    season_df = season_df.dropna(subset=[player_id_column]).sort_values(sort_columns)
    if season_df.empty:
        return []

    hr_source_column = "hr_count" if "hr_count" in season_df.columns else "hit_hr"
    season_df["season_hr_total"] = pd.to_numeric(season_df.get(hr_source_column), errors="coerce").fillna(0.0)
    season_df["season_pa_total"] = pd.to_numeric(season_df.get("pa_count"), errors="coerce").fillna(0.0)
    agg_kwargs: dict[str, tuple[str, str]] = {
        "batter_name": (player_name_column, "last"),
        "home_runs_2026": ("season_hr_total", "sum"),
        "plate_appearances_2026": ("season_pa_total", "sum"),
    }
    if "team" in season_df.columns:
        agg_kwargs["team"] = ("team", "last")
    if "game_pk" in season_df.columns:
        agg_kwargs["games_2026"] = ("game_pk", "nunique")
    else:
        agg_kwargs["games_2026"] = ("game_date", "nunique")

    grouped = season_df.groupby(player_id_column, dropna=False).agg(**agg_kwargs).reset_index()
    grouped["home_runs_2026"] = grouped["home_runs_2026"].astype(int)
    grouped["plate_appearances_2026"] = grouped["plate_appearances_2026"].astype(int)
    grouped["games_2026"] = grouped["games_2026"].astype(int)
    grouped = grouped.sort_values(
        ["home_runs_2026", "plate_appearances_2026", "games_2026", "batter_name"],
        ascending=[False, False, False, True],
    )
    return [
        {
            "batter_name": str(row["batter_name"] or "Unknown hitter"),
            "team": str(row.get("team") or ""),
            "home_runs_2026": int(row["home_runs_2026"]),
            "plate_appearances_2026": int(row["plate_appearances_2026"]),
            "games_2026": int(row["games_2026"]),
        }
        for row in grouped.head(limit).to_dict(orient="records")
    ]


def to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: serialize_value(row.get(column)) for column in DISPLAY_COLUMNS} for row in rows]


def build_refresh_schedule() -> dict[str, Any]:
    return {
        "timezone": str(DEFAULT_REFRESH_SCHEDULE["timezone"]),
        "runs": [dict(run) for run in DEFAULT_REFRESH_SCHEDULE["runs"]],
    }


def _humanize_feature_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def _feature_detail(feature_name: str) -> dict[str, str]:
    detail = MODEL_FEATURE_DETAILS.get(feature_name)
    if detail:
        return dict(detail)
    return {
        "label": _humanize_feature_name(feature_name),
        "description": "This feature is included in the current model configuration.",
    }


def _coefficient_strength_band(relative_weight: float | None) -> str:
    if relative_weight is None:
        return "Included"
    if relative_weight >= 0.75:
        return "Very strong"
    if relative_weight >= 0.45:
        return "Strong"
    if relative_weight >= 0.2:
        return "Moderate"
    return "Light"


def _coefficient_direction_label(weight: float | None) -> str:
    if weight is None:
        return "No coefficient detail"
    if weight > 0:
        return "Higher values generally boost the score"
    if weight < 0:
        return "Higher values generally lower the score"
    return "Neutral effect in the fitted model"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _model_identity_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "model_family": str(metadata.get("model_family") or DEFAULT_MODEL_FAMILY),
        "feature_profile": str(metadata.get("feature_profile") or DEFAULT_FEATURE_PROFILE),
    }


def _operational_alerts_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    alerts = metadata.get("operational_alerts")
    if not isinstance(alerts, list):
        return []
    return [dict(alert) for alert in alerts if isinstance(alert, dict)]


def build_model_explainer(
    *,
    model_bundle_path: Path = DEFAULT_MODEL_BUNDLE_PATH,
    model_metadata_path: Path = DEFAULT_MODEL_METADATA_PATH,
) -> dict[str, Any]:
    metadata = _load_json_object(model_metadata_path) if model_metadata_path.exists() else {}
    model_identity = _model_identity_from_metadata(metadata)
    feature_columns = [str(feature) for feature in metadata.get("feature_columns", []) if isinstance(feature, str)]
    if not feature_columns:
        return {
            "available": False,
            "title": "Model metric guide",
            "summary": "Model feature details are not available for this build.",
            "model_family": model_identity["model_family"],
            "feature_profile": model_identity["feature_profile"],
            "features": [],
            "strength_source": None,
        }

    coefficient_map: dict[str, float] = {}
    strength_source = "configuration"
    if model_bundle_path.exists():
        try:
            with model_bundle_path.open("rb") as handle:
                bundle = pickle.load(handle)
            coefficient_map = extract_logistic_coefficient_map(bundle.get("model"), feature_columns) if isinstance(bundle, dict) else {}
            if coefficient_map:
                strength_source = "logistic_coefficient"
        except Exception:
            coefficient_map = {}

    max_abs_weight = max((abs(weight) for weight in coefficient_map.values()), default=0.0)
    features: list[dict[str, Any]] = []
    for feature_name in feature_columns:
        detail = _feature_detail(feature_name)
        coefficient_weight = coefficient_map.get(feature_name)
        relative_weight = (abs(coefficient_weight) / max_abs_weight) if coefficient_weight is not None and max_abs_weight > 0 else None
        features.append(
            {
                "feature": feature_name,
                "label": detail["label"],
                "description": detail["description"],
                "strength": _coefficient_strength_band(relative_weight),
                "strength_score": serialize_value(relative_weight),
                "direction": _coefficient_direction_label(coefficient_weight),
                "coefficient_weight": serialize_value(coefficient_weight),
            }
        )

    if strength_source == "logistic_coefficient":
        features.sort(
            key=lambda item: (
                -(float(item["strength_score"]) if item["strength_score"] is not None else -1.0),
                str(item["label"]),
            )
        )

    return {
        "available": True,
        "title": "Model metric guide",
        "summary": f"{len(features)} metrics are active in the current model.",
        "model_family": model_identity["model_family"],
        "feature_profile": model_identity["feature_profile"],
        "trained_through": metadata.get("trained_through"),
        "feature_count": len(features),
        "strength_source": strength_source,
        "features": features,
    }


def build_dashboard_artifacts(
    *,
    current_picks_path: Path = DEFAULT_CURRENT_PICKS_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_bundle_path: Path = DEFAULT_MODEL_BUNDLE_PATH,
    model_data_path: Path = DEFAULT_MODEL_DATA_PATH,
    model_metadata_path: Path = DEFAULT_MODEL_METADATA_PATH,
    tracking_start_date: str = DEFAULT_TRACKING_START_DATE,
    latest_count: int = 12,
    history_per_date: int = 10,
    min_player_picks: int = 2,
    persist_history: bool = True,
    latest_available_date_override: str | None = None,
) -> Path:
    current_picks_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_input = load_json_array(current_picks_path)
    history_input = load_json_array(history_path)

    current_rows = [row for row in (normalize_pick(item, tracking_start_date) for item in current_input) if row is not None]
    history_rows = [row for row in (normalize_pick(item, tracking_start_date) for item in history_input) if row is not None]
    current_rows = sorted(current_rows, key=current_pick_sort_key)
    current_rows, history_rows = recover_pending_history_rows(current_rows, history_rows)
    active_current_rows = sorted(select_active_current_rows(current_rows), key=current_pick_sort_key)
    active_current_dates = {str(row["game_date"]) for row in active_current_rows if row.get("game_date")}
    dashboard_history = sorted(
        [row for row in history_rows if str(row.get("game_date") or "") not in active_current_dates],
        key=lambda row: (-int(str(row["game_date"]).replace("-", "")), -score_sort_value(row), int(row["rank"]), str(row["batter_name"])),
    )

    current_picks_path.write_text(json.dumps(clean_current_pick_rows(active_current_rows), indent=2), encoding="utf-8")
    if persist_history:
        history_path.write_text(json.dumps(clean_history_rows(history_rows), indent=2), encoding="utf-8")

    latest_game_date = latest_available_date_override or max(
        (row["game_date"] for row in ([*active_current_rows, *dashboard_history])),
        default=tracking_start_date,
    )
    latest_picks = list(active_current_rows)
    tracked_rows = [*dashboard_history, *active_current_rows]
    settled_rows = [row for row in tracked_rows if row["actual_hit_hr"] is not None]
    history_dates = build_history_date_options(dashboard_history)
    default_history_date = resolve_default_history_date(history_dates)
    yesterday_value = eastern_yesterday()
    recent_successes = [
        row
        for row in sorted(
            settled_rows,
            key=lambda item: (item["game_date"], score_sort_value(item), -item["rank"]),
            reverse=True,
        )
        if row["actual_hit_hr"] == 1 and str(row["game_date"]) == yesterday_value
    ][:25]
    season_hr_leaders_2026 = build_season_hr_leaders_2026(model_data_path)

    settled_count = len(settled_rows)
    homers = sum(int(row["actual_hit_hr"] or 0) for row in settled_rows)
    hit_rate = (homers / settled_count) if settled_count else None
    metadata = _load_json_object(model_metadata_path) if model_metadata_path.exists() else {}
    operational_alerts = _operational_alerts_from_metadata(metadata)
    player_leaderboard = build_player_leaderboard(settled_rows, min_player_picks=min_player_picks)
    model_explainer = build_model_explainer(
        model_bundle_path=model_bundle_path,
        model_metadata_path=model_metadata_path,
    )
    model_family = str(model_explainer.get("model_family") or DEFAULT_MODEL_FAMILY)
    feature_profile = str(model_explainer.get("feature_profile") or DEFAULT_FEATURE_PROFILE)
    lineup_panels: list[dict[str, Any]] = []
    if active_current_rows and model_data_path.exists():
        try:
            dataset_df = pd.read_csv(model_data_path, parse_dates=["game_date"])
            active_schedule_games = fetch_schedule_games(str(active_current_rows[0]["game_date"]))
            lineup_panels = build_lineup_panels(
                dataset_df,
                active_schedule_games,
                target_date=str(active_current_rows[0]["game_date"]),
                current_rows=active_current_rows,
            )
        except Exception:
            lineup_panels = []

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tracking_start_date": tracking_start_date,
        "model_family": model_family,
        "feature_profile": feature_profile,
        "latest_available_date": latest_game_date,
        "data_note": DEFAULT_DATA_NOTE,
        "operational_alerts": operational_alerts,
        "history_dates": history_dates,
        "history_default_date": default_history_date,
        "yesterday_homer_date": yesterday_value,
        "refresh_schedule": build_refresh_schedule(),
        "overview": {
            "latest_slate_size": len(latest_picks),
            "tracked_dates": len({row["game_date"] for row in tracked_rows}),
            "tracked_picks": len(tracked_rows),
            "settled_picks": settled_count,
            "tracked_homers": homers,
            "tracked_hit_rate": serialize_value(hit_rate),
            "open_picks": len([row for row in tracked_rows if row["actual_hit_hr"] is None]),
        },
        "confidence_summary": summarize_confidence(settled_rows),
        "player_leaderboard": player_leaderboard,
        "model_explainer": model_explainer,
        "latest_picks": to_records(latest_picks),
        "history": to_records(dashboard_history),
        "lineup_panels": lineup_panels,
        "season_hr_leaders_2026": season_hr_leaders_2026,
        "recent_successes": to_records(recent_successes),
    }

    output_path = output_dir / "dashboard.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_path = build_dashboard_artifacts(
        current_picks_path=Path(args.current_picks_path),
        history_path=Path(args.history_path),
        output_dir=Path(args.output_dir),
        model_bundle_path=DEFAULT_MODEL_BUNDLE_PATH,
        model_data_path=DEFAULT_MODEL_DATA_PATH,
        model_metadata_path=DEFAULT_MODEL_METADATA_PATH,
        tracking_start_date=args.tracking_start_date,
        latest_count=args.latest_count,
        history_per_date=args.history_per_date,
        min_player_picks=args.min_player_picks,
    )
    print(f"Wrote dashboard artifact to {output_path}")


if __name__ == "__main__":
    main()
