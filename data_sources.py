"""External data access helpers for Statcast and weather lookups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import time
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
    STATCAST_CHUNK_DAYS,
    STATCAST_COLUMNS,
)
from feature_engineering import append_weather_carry_features
from weather_audit import PRIMARY_WEATHER_FEATURE_COLUMNS, WEATHER_WARNING_NULL_RATE, summarize_weather_feature_coverage

cache.enable()


@dataclass(frozen=True)
class WeatherLookupRow:
    game_date: pd.Timestamp
    home_team: str
    field_bearing_deg: float | None
    temperature_f: float | None
    humidity_pct: float | None
    wind_speed_mph: float | None
    wind_direction_deg: float | None
    pressure_hpa: float | None
    wind_out_to_cf_mph: float | None
    crosswind_mph: float | None
    air_density_index: float | None


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


def _raw_chunk_path(start_date: str, end_date: str) -> Path:
    generic = RAW_DATA_DIR / f"statcast_{start_date}_{end_date}.csv"
    legacy = RAW_DATA_DIR / f"statcast_2024_{start_date}_{end_date}.csv"
    if legacy.exists() and not generic.exists():
        return legacy
    return generic


def _weather_cache_path(game_schedule: pd.DataFrame) -> Path:
    local_dates = pd.to_datetime(game_schedule["game_date"], errors="coerce").dropna()
    if local_dates.empty:
        return RAW_DATA_DIR / "weather_empty.csv"
    start = local_dates.min().strftime("%Y-%m-%d")
    end = local_dates.max().strftime("%Y-%m-%d")
    return RAW_DATA_DIR / f"weather_{start}_{end}.csv"


def _normalize_home_team_code(home_team: object) -> str:
    if pd.isna(home_team):
        return ""
    code = str(home_team).strip().upper()
    return _HOME_TEAM_ALIASES.get(code, code)


def _normalize_weather_schedule(game_schedule: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"game_date", "home_team"}
    if not required_cols.issubset(game_schedule.columns):
        raise ValueError(f"game_schedule must include columns {required_cols}")
    normalized = game_schedule.loc[:, ["game_date", "home_team"]].copy()
    normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.normalize()
    normalized["home_team"] = normalized["home_team"].map(_normalize_home_team_code)
    normalized = normalized.dropna(subset=["game_date"])
    normalized = normalized[normalized["home_team"] != ""].copy()
    return normalized


def _weather_cache_is_healthy(weather_df: pd.DataFrame, normalized_schedule: pd.DataFrame) -> bool:
    required_cols = {"game_date", "home_team", *PRIMARY_WEATHER_FEATURE_COLUMNS}
    if not required_cols.issubset(weather_df.columns):
        return False

    cache_keys = weather_df.loc[:, ["game_date", "home_team"]].drop_duplicates()
    expected_keys = normalized_schedule.drop_duplicates()
    if len(cache_keys) != len(expected_keys):
        return False

    merged = expected_keys.merge(cache_keys, on=["game_date", "home_team"], how="left", indicator=True)
    if not (merged["_merge"] == "both").all():
        return False

    coverage = summarize_weather_feature_coverage(weather_df, feature_columns=list(PRIMARY_WEATHER_FEATURE_COLUMNS))
    if any(not stats["present"] for stats in coverage.values()):
        return False
    if any(
        stats["null_rate"] is not None and float(stats["null_rate"]) > WEATHER_WARNING_NULL_RATE
        for stats in coverage.values()
    ):
        return False
    return True


def _park_field_bearing_deg(home_team: str) -> float | None:
    park = PARKS.get(home_team)
    if not park:
        return None
    return _safe_float(park.get("field_bearing_deg"))


def _augment_weather_carry_fields(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df.copy()
    augmented = weather_df.copy()
    if "field_bearing_deg" not in augmented.columns:
        augmented["field_bearing_deg"] = np.nan
    if augmented["field_bearing_deg"].isna().any():
        augmented["field_bearing_deg"] = augmented["field_bearing_deg"].where(
            augmented["field_bearing_deg"].notna(),
            augmented["home_team"].map(_park_field_bearing_deg),
        )
    return append_weather_carry_features(augmented)


def fetch_statcast_range(start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch a cached Statcast date window and keep only required columns."""
    ensure_directories()
    chunk_path = _raw_chunk_path(start_date, end_date)
    if chunk_path.exists() and not force_refresh:
        df = pd.read_csv(chunk_path, low_memory=False)
    else:
        df = statcast(start_dt=start_date, end_dt=end_date)
        if df is None or df.empty:
            df = pd.DataFrame(columns=STATCAST_COLUMNS)
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


_OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_TIMEOUT_SECONDS = 120
OPEN_METEO_MAX_ATTEMPTS = 3
OPEN_METEO_RETRY_BACKOFF_SECONDS = 2
_HOME_TEAM_ALIASES = {
    "ARI": "AZ",
    "CHW": "CWS",
    "KCR": "KC",
    "OAK": "ATH",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
}


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
    response = None
    session = requests.Session()
    try:
        for attempt in range(1, OPEN_METEO_MAX_ATTEMPTS + 1):
            try:
                response = session.get(_OPEN_METEO_URL, params=params, timeout=OPEN_METEO_TIMEOUT_SECONDS)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == OPEN_METEO_MAX_ATTEMPTS:
                    raise
                warnings.warn(
                    "Open-Meteo archive request timed out or failed "
                    f"for {start_date} to {end_date} (attempt {attempt}/{OPEN_METEO_MAX_ATTEMPTS}); retrying."
                )
                time.sleep(OPEN_METEO_RETRY_BACKOFF_SECONDS ** attempt)
    finally:
        session.close()

    if response is None:
        raise RuntimeError("Open-Meteo archive response was not created after retries.")
    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    # Open-Meteo returns naive local timestamps; use a deterministic DST policy so
    # fall-back windows do not fail when the hourly series contains only one copy
    # of the ambiguous local hour.
    df.index = pd.to_datetime(df["time"]).dt.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
    df = df.drop(columns=["time"])
    return df


def _build_null_weather_row(game_date: pd.Timestamp, home_team: str) -> WeatherLookupRow:
    return WeatherLookupRow(
        game_date=pd.Timestamp(game_date).normalize(),
        home_team=home_team,
        field_bearing_deg=_park_field_bearing_deg(home_team),
        temperature_f=None,
        humidity_pct=None,
        wind_speed_mph=None,
        wind_direction_deg=None,
        pressure_hpa=None,
        wind_out_to_cf_mph=None,
        crosswind_mph=None,
        air_density_index=None,
    )


def _cached_weather_rows_for_team(
    weather_df: pd.DataFrame | None,
    *,
    home_team: str,
    local_dates: pd.Series,
) -> list[WeatherLookupRow]:
    if weather_df is None or weather_df.empty:
        return []
    required_columns = {"game_date", "home_team"}
    if not required_columns.issubset(weather_df.columns):
        return []

    cached = weather_df.copy()
    cached["game_date"] = pd.to_datetime(cached["game_date"], errors="coerce").dt.normalize()
    cached["home_team"] = cached["home_team"].map(_normalize_home_team_code)
    cached = cached[cached["home_team"] == home_team].copy()
    if cached.empty:
        return []

    cached_by_date = {
        pd.Timestamp(row["game_date"]).normalize(): row
        for _, row in cached.dropna(subset=["game_date"]).iterrows()
    }
    recovered_rows: list[WeatherLookupRow] = []
    for game_date in local_dates:
        normalized_game_date = pd.Timestamp(game_date).normalize()
        cached_row = cached_by_date.get(normalized_game_date)
        if cached_row is None:
            continue
        recovered_rows.append(
            WeatherLookupRow(
                game_date=normalized_game_date,
                home_team=home_team,
                field_bearing_deg=_park_field_bearing_deg(home_team),
                temperature_f=_safe_float(cached_row.get("temperature_f")),
                humidity_pct=_safe_float(cached_row.get("humidity_pct")),
                wind_speed_mph=_safe_float(cached_row.get("wind_speed_mph")),
                wind_direction_deg=_safe_float(cached_row.get("wind_direction_deg")),
                pressure_hpa=_safe_float(cached_row.get("pressure_hpa")),
                wind_out_to_cf_mph=_safe_float(cached_row.get("wind_out_to_cf_mph")),
                crosswind_mph=_safe_float(cached_row.get("crosswind_mph")),
                air_density_index=_safe_float(cached_row.get("air_density_index")),
            )
        )
    return recovered_rows


def build_weather_table(game_schedule: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Pull hourly historical weather for each MLB park/date pair used in the dataset."""
    ensure_directories()
    normalized_schedule = _normalize_weather_schedule(game_schedule)
    weather_cache_path = _weather_cache_path(normalized_schedule)
    existing_cached_weather: pd.DataFrame | None = None
    if weather_cache_path.exists() and not force_refresh:
        existing_cached_weather = pd.read_csv(weather_cache_path, parse_dates=["game_date"])
        existing_cached_weather["game_date"] = pd.to_datetime(existing_cached_weather["game_date"], errors="coerce").dt.normalize()
        existing_cached_weather["home_team"] = existing_cached_weather["home_team"].map(_normalize_home_team_code)
        existing_cached_weather = _augment_weather_carry_fields(existing_cached_weather)
        if _weather_cache_is_healthy(existing_cached_weather, normalized_schedule):
            existing_cached_weather.attrs["operational_alerts"] = []
            existing_cached_weather.to_csv(weather_cache_path, index=False)
            return existing_cached_weather
        warnings.warn(
            f"Cached weather table at {weather_cache_path} is incomplete or sparse; rebuilding from source."
        )

    rows: list[WeatherLookupRow] = []
    operational_alerts: list[dict[str, object]] = []
    for home_team, park_games in normalized_schedule.groupby("home_team"):
        if home_team not in PARKS:
            raise KeyError(f"No park mapping configured for home team '{home_team}'.")
        park = PARKS[home_team]
        tz_name = park["tz"]
        local_dates = pd.to_datetime(park_games["game_date"]).dt.normalize().drop_duplicates().sort_values()
        if local_dates.empty:
            continue

        start_str = local_dates.min().strftime("%Y-%m-%d")
        end_str = local_dates.max().strftime("%Y-%m-%d")
        try:
            weather = _fetch_open_meteo(park["lat"], park["lon"], start_str, end_str, tz_name)
        except Exception as exc:
            cached_rows = _cached_weather_rows_for_team(
                existing_cached_weather,
                home_team=home_team,
                local_dates=local_dates,
            )
            recovered_dates = {row.game_date for row in cached_rows}
            missing_dates = [
                pd.Timestamp(game_date).normalize()
                for game_date in local_dates
                if pd.Timestamp(game_date).normalize() not in recovered_dates
            ]
            if cached_rows:
                warnings.warn(
                    f"Open-Meteo fetch failed for {home_team}: {exc}. "
                    f"Reusing cached weather rows for {len(cached_rows)} date(s) and filling {len(missing_dates)} remaining date(s) with nulls."
                )
                operational_alerts.append(
                    {
                        "kind": "warning",
                        "code": "historical_weather_cache_reused",
                        "team": home_team,
                        "reused_dates": [str(row.game_date.date()) for row in cached_rows],
                        "null_dates": [str(game_date.date()) for game_date in missing_dates],
                    }
                )
                rows.extend(cached_rows)
            else:
                warnings.warn(f"Open-Meteo fetch failed for {home_team}: {exc}. Filling with nulls.")
                operational_alerts.append(
                    {
                        "kind": "warning",
                        "code": "historical_weather_null_fallback",
                        "team": home_team,
                        "null_dates": [str(pd.Timestamp(game_date).date()) for game_date in local_dates],
                    }
                )
            for game_date in local_dates:
                normalized_game_date = pd.Timestamp(game_date).normalize()
                if normalized_game_date in recovered_dates:
                    continue
                rows.append(_build_null_weather_row(normalized_game_date, home_team))
            continue

        weather["weather_date"] = weather.index.normalize().tz_localize(None)
        weather["hour_diff"] = abs(weather.index.hour - DEFAULT_GAME_HOUR_LOCAL)

        for game_date in local_dates:
            normalized_game_date = pd.Timestamp(game_date).normalize()
            day_weather = weather[weather["weather_date"] == normalized_game_date]
            if day_weather.empty:
                rows.append(_build_null_weather_row(normalized_game_date, home_team))
                operational_alerts.append(
                    {
                        "kind": "warning",
                        "code": "historical_weather_null_day",
                        "team": home_team,
                        "null_dates": [str(normalized_game_date.date())],
                    }
                )
                continue
            best = day_weather.sort_values("hour_diff").iloc[0]
            rows.append(
                WeatherLookupRow(
                    game_date=normalized_game_date,
                    home_team=home_team,
                    field_bearing_deg=_park_field_bearing_deg(home_team),
                    temperature_f=_safe_float(best.get("temperature_2m")),
                    humidity_pct=_safe_float(best.get("relative_humidity_2m")),
                    wind_speed_mph=_safe_float(best.get("wind_speed_10m")),
                    wind_direction_deg=_safe_float(best.get("wind_direction_10m")),
                    pressure_hpa=_safe_float(best.get("surface_pressure")),
                    wind_out_to_cf_mph=None,
                    crosswind_mph=None,
                    air_density_index=None,
                )
            )

    weather_df = pd.DataFrame(
        rows,
        columns=[
            "game_date",
            "home_team",
            "field_bearing_deg",
            "temperature_f",
            "humidity_pct",
            "wind_speed_mph",
            "wind_direction_deg",
            "pressure_hpa",
            "wind_out_to_cf_mph",
            "crosswind_mph",
            "air_density_index",
        ],
    )
    weather_df = _augment_weather_carry_fields(weather_df)
    weather_df.attrs["operational_alerts"] = operational_alerts
    weather_df.to_csv(weather_cache_path, index=False)
    return weather_df


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
