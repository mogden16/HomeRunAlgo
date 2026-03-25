"""External data access helpers for Statcast and Meteostat."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import warnings

import numpy as np
import pandas as pd
import requests
from pybaseball import cache, statcast

from config import (
    DATA_DIR,
    DEFAULT_GAME_HOUR_LOCAL,
    PARKS,
    RAW_DATA_DIR,
    PA_ENDING_EVENTS,
    STATCAST_CHUNK_DAYS,
    STATCAST_COLUMNS,
    get_park_info,
    park_factor_cache_path,
    raw_statcast_chunk_path,
    season_from_date_range,
    weather_cache_path,
)

cache.enable()

_SAVANT_PARK_FACTOR_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
_RAW_FEATURE_FAMILIES: dict[str, list[str]] = {
    "pitch_type_matchup": ["pitch_type", "pitch_name"],
    "pitch_quality_context": ["release_spin_rate", "spin_axis", "release_extension"],
    "contact_authority": ["hit_distance_sc"],
    "bat_tracking": ["bat_speed", "attack_angle", "attack_direction", "swing_path_tilt"],
}
_IMPLEMENTED_RAW_FEATURE_FAMILIES = {"pitch_type_matchup", "pitch_quality_context", "contact_authority"}


@dataclass(frozen=True)
class WeatherLookupRow:
    game_date: pd.Timestamp
    home_team: str
    temperature_f: float | None
    humidity_pct: float | None
    wind_speed_mph: float | None
    wind_direction_deg: float | None
    pressure_hpa: float | None


def ensure_directories() -> None:
    """Create data directories used by the pipeline."""
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _chunk_dates(start_date: str, end_date: str, chunk_days: int = STATCAST_CHUNK_DAYS) -> Iterable[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + pd.Timedelta(days=1)


def fetch_statcast_range(start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch a cached Statcast date window and keep only required columns."""
    ensure_directories()
    season = season_from_date_range(start_date, end_date)
    chunk_path = raw_statcast_chunk_path(season, start_date, end_date)
    should_refresh = force_refresh or not chunk_path.exists()
    if not should_refresh:
        df = pd.read_csv(chunk_path, low_memory=False)
        missing_cached = [column for column in STATCAST_COLUMNS if column not in df.columns]
        if missing_cached:
            print(
                f"Refreshing cached Statcast chunk {chunk_path.name} because it is missing columns: {missing_cached}"
            )
            should_refresh = True
    if should_refresh:
        df = statcast(start_dt=start_date, end_dt=end_date)
        if df is None or df.empty:
            raise RuntimeError(f"Statcast returned no data for {start_date} to {end_date}.")
        df.to_csv(chunk_path, index=False)

    missing = [column for column in STATCAST_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(f"Statcast data is missing expected columns: {missing}")

    return df.loc[:, STATCAST_COLUMNS].copy()


def fetch_statcast_season(start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch the full season in manageable cached chunks."""
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _chunk_dates(start_date, end_date):
        frames.append(fetch_statcast_range(chunk_start, chunk_end, force_refresh=force_refresh))
    statcast_df = pd.concat(frames, ignore_index=True)
    statcast_df["game_date"] = pd.to_datetime(statcast_df["game_date"])
    statcast_df = statcast_df[statcast_df["game_type"] == "R"].copy()
    return statcast_df


def build_hr_park_factor_table(statcast_df: pd.DataFrame, season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Build a park-factor lookup, preferring the official Savant source and falling back to local derivation."""
    ensure_directories()
    cache_path = park_factor_cache_path(season)
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path)

    try:
        park_factors = _fetch_savant_hr_park_factors(season)
    except Exception as exc:
        warnings.warn(
            "Official Savant park factors could not be parsed automatically; "
            f"falling back to a local Statcast-derived HR park index. Details: {exc}"
        )
        park_factors = _derive_local_hr_park_factors(statcast_df, season)

    park_factors.to_csv(cache_path, index=False)
    return park_factors


def _fetch_savant_hr_park_factors(season: int) -> pd.DataFrame:
    """Attempt to parse the official Savant Park Factors page if row data are server-rendered."""
    params = {
        "type": "year",
        "year": str(season),
        "batSide": "",
        "stat": "index_HR",
        "condition": "All",
        "rolling": "no",
    }
    response = requests.get(_SAVANT_PARK_FACTOR_URL, params=params, timeout=60)
    response.raise_for_status()

    html = response.text
    if "var data = {}" in html:
        raise RuntimeError("Savant page returned a client-rendered shell without park-factor row data.")

    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("Savant page did not expose any parseable tables.")

    table = tables[0].copy()
    renamed_columns = {str(column).strip().lower(): column for column in table.columns}
    venue_col = renamed_columns.get("venue")
    factor_col = next((column for key, column in renamed_columns.items() if key in {"hr", "index_hr", "index hr"}), None)
    if venue_col is None or factor_col is None:
        raise RuntimeError(f"Could not locate venue/HR columns in Savant table columns: {list(table.columns)}")

    park_factors = table[[venue_col, factor_col]].copy()
    park_factors = park_factors.rename(columns={venue_col: "ballpark", factor_col: "park_factor_hr"})
    park_factors["ballpark"] = park_factors["ballpark"].astype(str).str.strip()
    park_factors["park_factor_hr"] = pd.to_numeric(park_factors["park_factor_hr"], errors="coerce")
    park_factors = park_factors.dropna(subset=["ballpark", "park_factor_hr"]).copy()
    park_factors["season"] = int(season)
    park_factors["source"] = "baseballsavant_official"
    park_factors["home_team"] = park_factors["ballpark"].map(_ballpark_to_home_team_map(season))
    return park_factors[["season", "home_team", "ballpark", "park_factor_hr", "source"]]


def _derive_local_hr_park_factors(statcast_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Fallback HR park index from current-season Statcast events when Savant rows are unavailable."""
    pa_df = statcast_df[statcast_df["events"].isin(PA_ENDING_EVENTS)].copy()
    if pa_df.empty:
        raise RuntimeError("Cannot derive local park factors because no plate-appearance-ending events were found.")

    pa_df["home_team"] = pa_df["home_team"].astype(str)
    pa_df["ballpark"] = [
        get_park_info(str(home_team), season).get("ballpark") if pd.notna(home_team) else np.nan
        for home_team in pa_df["home_team"]
    ]
    pa_df["is_hr"] = pa_df["events"].eq("home_run").astype(int)
    park_summary = pa_df.groupby(["home_team", "ballpark"], dropna=False).agg(
        plate_appearances=("events", "size"),
        hr_count=("is_hr", "sum"),
    ).reset_index()

    league_hr_rate = float(pa_df["is_hr"].mean())
    if league_hr_rate <= 0:
        raise RuntimeError("Cannot derive local park factors because league HR rate is zero.")

    park_summary["park_hr_rate"] = park_summary["hr_count"] / park_summary["plate_appearances"]
    park_summary["park_factor_hr"] = 100.0 * park_summary["park_hr_rate"] / league_hr_rate
    park_summary["park_factor_hr"] = park_summary["park_factor_hr"].round(3)
    park_summary["season"] = int(season)
    park_summary["source"] = "local_statcast_hr_index_fallback"
    return park_summary[["season", "home_team", "ballpark", "park_factor_hr", "source", "plate_appearances", "hr_count"]]


def print_raw_feature_opportunity_audit(start_date: str | None = None, end_date: str | None = None) -> None:
    """Print a coverage audit for raw Statcast feature families available in cached files."""
    if start_date and end_date:
        season = season_from_date_range(start_date, end_date)
        raw_paths = sorted(RAW_DATA_DIR.glob(f"statcast_{season}_*.csv"))
    else:
        raw_paths = sorted(RAW_DATA_DIR.glob("statcast_*_*.csv"))
    if not raw_paths:
        print("\nRaw feature opportunity audit")
        print("-" * 60)
        print("No raw Statcast cache files found.")
        return

    selected_columns = sorted({column for columns in _RAW_FEATURE_FAMILIES.values() for column in columns})
    total_rows = 0
    present_columns: set[str] = set()
    non_null_counts = {column: 0 for column in selected_columns}

    for raw_path in raw_paths:
        header = pd.read_csv(raw_path, nrows=0).columns.tolist()
        available = [column for column in selected_columns if column in header]
        if not available:
            continue
        present_columns.update(available)
        chunk = pd.read_csv(raw_path, usecols=available, low_memory=False)
        total_rows += len(chunk)
        for column in available:
            non_null_counts[column] += int(chunk[column].notna().sum())

    print("\nRaw feature opportunity audit")
    print("-" * 60)
    print(f"Raw cache files scanned: {len(raw_paths)}")
    print(f"Approximate raw rows scanned: {total_rows:,}")
    for family_name, columns in _RAW_FEATURE_FAMILIES.items():
        missing = [column for column in columns if column not in present_columns]
        shares = [non_null_counts[column] / total_rows for column in columns if column in present_columns and total_rows > 0]
        avg_non_null_share = float(sum(shares) / len(shares)) if shares else 0.0
        if family_name in _IMPLEMENTED_RAW_FEATURE_FAMILIES:
            recommendation = "implemented in current schema"
        elif missing:
            recommendation = "low coverage"
        elif avg_non_null_share >= 0.75:
            recommendation = "high-priority next"
        elif avg_non_null_share >= 0.25:
            recommendation = "worth testing"
        else:
            recommendation = "low coverage"
        print(
            f"{family_name}: columns={columns}, present={'yes' if not missing else 'partial'}, "
            f"approx_non_null_share={avg_non_null_share:.1%}, recommendation={recommendation}"
        )
        if missing:
            print(f"  missing columns: {missing}")


def _ballpark_to_home_team_map(season: int) -> dict[str, str]:
    mapping: dict[str, str] = {}
    teams = set(PARKS)
    teams.add("ATH")
    for team in sorted(teams):
        details = get_park_info(team, season)
        ballpark = details.get("ballpark")
        if ballpark and ballpark not in mapping:
            mapping[ballpark] = team
    return mapping


_OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_open_meteo(lat: float, lon: float, start_date: str, end_date: str, tz: str) -> pd.DataFrame:
    """Fetch hourly weather from the Open-Meteo archive (ERA5). No API key required."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": tz,
    }
    resp = requests.get(_OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df["time"]).dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    df = df.drop(columns=["time"])
    return df


def build_weather_table(game_schedule: pd.DataFrame, season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Pull hourly historical weather for each MLB park/date pair used in the dataset."""
    ensure_directories()
    cache_path = weather_cache_path(season)
    if cache_path.exists() and not force_refresh:
        weather_df = pd.read_csv(cache_path, parse_dates=["game_date"])
        return weather_df

    required_cols = {"game_date", "home_team"}
    if not required_cols.issubset(game_schedule.columns):
        raise ValueError(f"game_schedule must include columns {required_cols}")

    rows: list[WeatherLookupRow] = []
    for home_team, park_games in game_schedule.groupby("home_team"):
        park = get_park_info(str(home_team), season)
        if not park:
            raise KeyError(f"No park mapping configured for home team '{home_team}'.")
        tz_name = park["tz"]
        local_dates = pd.to_datetime(park_games["game_date"]).dt.normalize().drop_duplicates().sort_values()
        if local_dates.empty:
            continue

        start_str = local_dates.min().strftime("%Y-%m-%d")
        end_str = local_dates.max().strftime("%Y-%m-%d")
        try:
            weather = _fetch_open_meteo(park["lat"], park["lon"], start_str, end_str, tz_name)
        except Exception as exc:
            warnings.warn(f"Open-Meteo fetch failed for {home_team}: {exc} — filling with nulls.")
            for game_date in local_dates:
                rows.append(WeatherLookupRow(game_date=game_date, home_team=home_team,
                                             temperature_f=None, humidity_pct=None, wind_speed_mph=None,
                                             wind_direction_deg=None, pressure_hpa=None))
            continue

        weather["weather_date"] = weather.index.normalize().tz_localize(None)
        weather["hour_diff"] = abs(weather.index.hour - DEFAULT_GAME_HOUR_LOCAL)

        for game_date in local_dates:
            day_weather = weather[weather["weather_date"] == game_date]
            if day_weather.empty:
                rows.append(WeatherLookupRow(game_date=game_date, home_team=home_team,
                                             temperature_f=None, humidity_pct=None, wind_speed_mph=None,
                                             wind_direction_deg=None, pressure_hpa=None))
                continue
            best = day_weather.sort_values("hour_diff").iloc[0]
            rows.append(
                WeatherLookupRow(
                    game_date=game_date,
                    home_team=home_team,
                    temperature_f=_safe_float(best.get("temperature_2m")),
                    humidity_pct=_safe_float(best.get("relative_humidity_2m")),
                    wind_speed_mph=_safe_float(best.get("wind_speed_10m")),
                    wind_direction_deg=_safe_float(best.get("wind_direction_10m")),
                    pressure_hpa=_safe_float(best.get("surface_pressure")),
                )
            )

    weather_df = pd.DataFrame(rows)
    weather_df.to_csv(cache_path, index=False)
    return weather_df


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
