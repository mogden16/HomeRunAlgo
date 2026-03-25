"""Shared helpers for forward-only live pick generation and settlement."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

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
from feature_engineering import compute_batter_trailing_features, compute_pitcher_trailing_features
from generate_data import generate_mlb_dataset
from train_model import (
    FEATURE_COLUMNS,
    MAX_MODEL_FEATURE_MISSINGNESS,
    REASON_TEXT_BY_FEATURE,
    extract_logistic_coefficient_map,
    fit_selected_model,
    generate_reason_strings,
    prune_model_features_by_training_missingness,
)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PERSON_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/people/{player_id}"
MLB_TEAM_ROSTER_URL_TEMPLATE = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
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


def eastern_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").normalize()


def default_training_end_date() -> str:
    return (eastern_today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def default_publish_date() -> str:
    return eastern_today().strftime("%Y-%m-%d")


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


def train_live_model_bundle(
    dataset_path: Path,
    *,
    bundle_path: Path = LIVE_MODEL_BUNDLE_PATH,
    metadata_path: Path = LIVE_MODEL_METADATA_PATH,
    model_name: str = "logistic",
    calibration: str = "sigmoid",
) -> dict[str, Any]:
    df = pd.read_csv(dataset_path, parse_dates=["game_date"])
    feature_columns, _, excluded = prune_model_features_by_training_missingness(
        df,
        [feature for feature in FEATURE_COLUMNS if feature in df.columns],
        threshold=MAX_MODEL_FEATURE_MISSINGNESS,
    )
    if not feature_columns:
        raise ValueError("No live-model features were available after pruning.")

    X_train = df[feature_columns]
    y_train = df["hit_hr"].to_numpy()
    model, resolved_model_name, best_params, calibration_status = fit_selected_model(X_train, y_train, model_name, calibration)
    if model is None:
        raise RuntimeError(f"Unable to fit live model: {calibration_status['message']}")

    reference_columns = sorted(set(feature_columns) | {feature for feature in REASON_TEXT_BY_FEATURE if feature in df.columns})
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "reference_df": df[reference_columns].copy(),
        "trained_through": str(df["game_date"].max().date()),
        "model_family": resolved_model_name,
        "best_params": best_params,
        "calibration_status": calibration_status,
        "excluded_features": excluded,
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    metadata = {
        "trained_through": bundle["trained_through"],
        "model_family": resolved_model_name,
        "feature_columns": feature_columns,
        "best_params": best_params,
        "calibration_status": calibration_status,
        "excluded_features": excluded,
        "row_count": int(len(df)),
        "hr_rate": serialize_for_json(float(df["hit_hr"].mean())),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bundle


def load_model_bundle(bundle_path: Path = LIVE_MODEL_BUNDLE_PATH) -> dict[str, Any]:
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


def fetch_schedule_games(target_date: str) -> list[dict[str, Any]]:
    response = requests.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": target_date, "hydrate": "probablePitcher,team"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    games: list[dict[str, Any]] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
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


def select_probable_lineup_hitters(
    dataset_df: pd.DataFrame,
    *,
    team_code: str,
    target_date: str,
    hitters_per_team: int = 9,
    lookback_days: int = 21,
    active_roster: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    historical = dataset_df[dataset_df["game_date"] < pd.Timestamp(target_date)].copy()
    team_rows = historical[historical["team"] == team_code].copy()
    if team_rows.empty:
        return []
    if active_roster is not None:
        active_roster = active_roster.copy()
        active_roster["batter_id"] = pd.to_numeric(active_roster["batter_id"], errors="coerce").astype("Int64")
        team_rows["batter_id"] = pd.to_numeric(team_rows["batter_id"], errors="coerce").astype("Int64")
        team_rows = team_rows[team_rows["batter_id"].isin(active_roster["batter_id"])].copy()
        if team_rows.empty:
            return []

    recent_cutoff = pd.Timestamp(target_date) - pd.Timedelta(days=lookback_days)
    recent_rows = team_rows[team_rows["game_date"] >= recent_cutoff].copy()
    ranked_recent = _summarize_hitter_pool(recent_rows) if not recent_rows.empty else pd.DataFrame()
    selected = ranked_recent.head(hitters_per_team).to_dict(orient="records") if not ranked_recent.empty else []
    selected_ids = {int(row["batter_id"]) for row in selected if pd.notna(row.get("batter_id"))}

    if len(selected) < hitters_per_team:
        fallback = _summarize_hitter_pool(team_rows)
        for row in fallback.to_dict(orient="records"):
            batter_id = int(row["batter_id"])
            if batter_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(batter_id)
            if len(selected) >= hitters_per_team:
                break
    if active_roster is not None and selected:
        roster_name_lookup = {
            int(row.batter_id): str(row.batter_name)
            for row in active_roster[["batter_id", "batter_name"]].drop_duplicates().itertuples(index=False)
            if pd.notna(row.batter_id)
        }
        for row in selected:
            batter_id = int(row["batter_id"])
            if batter_id in roster_name_lookup and roster_name_lookup[batter_id]:
                row["batter_name"] = roster_name_lookup[batter_id]
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
    rows: list[dict[str, Any]] = []
    for home_team in sorted({normalize_team_code(team) for team in home_teams if team}):
        park = PARKS.get(home_team)
        if park is None:
            rows.append({"game_date": target_date, "home_team": home_team})
            continue
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": park["lat"],
                "longitude": park["lon"],
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": park["tz"],
                "start_date": target_date,
                "end_date": target_date,
            },
            timeout=60,
        )
        response.raise_for_status()
        hourly = response.json().get("hourly", {})
        weather = pd.DataFrame(hourly)
        if weather.empty:
            rows.append({"game_date": target_date, "home_team": home_team})
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
                "pressure_hpa": serialize_for_json(float(best.get("surface_pressure"))) if pd.notna(best.get("surface_pressure")) else None,
            }
        )
    return pd.DataFrame(rows)


def build_live_candidate_frame(
    dataset_df: pd.DataFrame,
    schedule_games: list[dict[str, Any]],
    *,
    target_date: str,
    hitters_per_team: int = 9,
    active_roster_map: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    forecast_weather = fetch_forecast_weather(
        [game["home_team"] for game in schedule_games],
        target_date=target_date,
    )
    rows: list[dict[str, Any]] = []
    for game in schedule_games:
        matchup_specs = [
            {
                "batting_team": game["away_team"],
                "opponent_team": game["home_team"],
                "is_home": 0,
                "pitcher_id": game["home_pitcher_id"],
                "pitcher_name": game["home_pitcher_name"],
            },
            {
                "batting_team": game["home_team"],
                "opponent_team": game["away_team"],
                "is_home": 1,
                "pitcher_id": game["away_pitcher_id"],
                "pitcher_name": game["away_pitcher_name"],
            },
        ]
        for spec in matchup_specs:
            hitters = select_probable_lineup_hitters(
                dataset_df,
                team_code=spec["batting_team"],
                target_date=target_date,
                hitters_per_team=hitters_per_team,
                active_roster=active_roster_map.get(spec["batting_team"]) if active_roster_map else None,
            )
            pitcher_hand = latest_pitcher_hand(dataset_df, spec["pitcher_id"]) or fetch_player_handedness(spec["pitcher_id"])
            for hitter in hitters:
                rows.append(
                    {
                        "game_pk": int(game["game_pk"]),
                        "game_date": target_date,
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
                        "bat_side": hitter.get("bat_side"),
                        "pitcher_hand": pitcher_hand,
                        "pitch_hand_primary": pitcher_hand,
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
                        "ballpark": PARKS.get(game["home_team"], {}).get("ballpark"),
                    }
                )
    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return candidate_df

    candidate_df["home_team"] = np.where(candidate_df["is_home"].astype(bool), candidate_df["team"], candidate_df["opponent"])
    candidate_df = candidate_df.merge(
        forecast_weather,
        on=["game_date", "home_team"],
        how="left",
        validate="many_to_one",
    )
    candidate_df = candidate_df.drop(columns=["home_team"])
    candidate_df["platoon_advantage"] = np.where(
        candidate_df["bat_side"].notna() & candidate_df["pitch_hand_primary"].notna(),
        (candidate_df["bat_side"] != candidate_df["pitch_hand_primary"]).astype(float),
        np.nan,
    )
    return candidate_df


LIVE_CONTEXT_FEATURES = {"temperature_f", "wind_speed_mph", "humidity_pct", "platoon_advantage"}
LIVE_BATTER_FEATURES = [feature for feature in FEATURE_COLUMNS if feature not in LIVE_CONTEXT_FEATURES and not feature.startswith("pitcher_")]
LIVE_PITCHER_FEATURES = [feature for feature in FEATURE_COLUMNS if feature.startswith("pitcher_")]


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
    batter_snapshot = build_latest_feature_snapshot(
        dataset_df,
        entity_key="batter_id",
        feature_columns=LIVE_BATTER_FEATURES,
    )
    featured = fill_missing_features_from_snapshot(
        featured,
        snapshot_df=batter_snapshot,
        entity_key="batter_id",
        feature_columns=LIVE_BATTER_FEATURES,
        suffix="latest_batter",
    )
    pitcher_snapshot = build_latest_feature_snapshot(
        dataset_df,
        entity_key="pitcher_id",
        feature_columns=LIVE_PITCHER_FEATURES,
    )
    featured = fill_missing_features_from_snapshot(
        featured,
        snapshot_df=pitcher_snapshot,
        entity_key="pitcher_id",
        feature_columns=LIVE_PITCHER_FEATURES,
        suffix="latest_pitcher",
    )
    featured["opponent_team"] = featured["opponent"]
    return featured


def score_live_candidates(
    candidate_df: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    max_picks: int = 20,
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

    ranked = scored.sort_values(["predicted_hr_score", "predicted_hr_probability", "batter_name"], ascending=[False, False, True]).head(max_picks).copy()
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
                "result": "Pending",
            }
        )
    return rows


def settle_pick_records(
    records: list[dict[str, Any]],
    dataset_df: pd.DataFrame,
    *,
    resolved_through_date: str,
) -> list[dict[str, Any]]:
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
            settled.append(row)
            continue
        if not game_date or game_date > resolved_through_date:
            settled.append(row)
            continue

        lookup_value = resolved_lookup.get((game_date, int(batter_id))) if batter_id is not None else None
        resolved_hit = int(lookup_value) if lookup_value is not None else 0
        resolved_label = "HR" if resolved_hit == 1 else "No HR"
        updated = dict(row)
        updated["result"] = resolved_label
        updated["result_label"] = resolved_label
        updated["actual_hit_hr"] = resolved_hit
        settled.append(updated)
    return settled


def write_current_picks(rows: list[dict[str, Any]], path: Path = LIVE_CURRENT_PICKS_PATH) -> None:
    write_json_array(path, rows)


def load_pick_history(path: Path = LIVE_PICK_HISTORY_PATH) -> list[dict[str, Any]]:
    return load_json_array(path)


def write_pick_history(rows: list[dict[str, Any]], path: Path = LIVE_PICK_HISTORY_PATH) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("game_date") or ""),
            -(float(row.get("predicted_hr_score")) if row.get("predicted_hr_score") is not None else -999.0),
            int(row.get("rank") or 999),
            str(row.get("batter_name") or ""),
        ),
    )
    write_json_array(path, ordered)
