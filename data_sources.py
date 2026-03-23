"""External data access helpers for Statcast and Meteostat."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from meteostat import hourly, Point
from pybaseball import cache, statcast

from config import (
    DATA_DIR,
    DEFAULT_GAME_HOUR_LOCAL,
    PARKS,
    RAW_DATA_DIR,
    STATCAST_CHUNK_DAYS,
    STATCAST_COLUMNS,
    WEATHER_CACHE_PATH,
)

cache.enable()


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


def _raw_chunk_path(start_date: str, end_date: str) -> Path:
    return RAW_DATA_DIR / f"statcast_2024_{start_date}_{end_date}.csv"


def fetch_statcast_range(start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch a cached Statcast date window and keep only required columns."""
    ensure_directories()
    chunk_path = _raw_chunk_path(start_date, end_date)
    if chunk_path.exists() and not force_refresh:
        df = pd.read_csv(chunk_path, low_memory=False)
    else:
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


def build_weather_table(game_schedule: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """Pull hourly historical weather for each MLB park/date pair used in the dataset."""
    ensure_directories()
    if WEATHER_CACHE_PATH.exists() and not force_refresh:
        weather_df = pd.read_csv(WEATHER_CACHE_PATH, parse_dates=["game_date"])
        return weather_df

    required_cols = {"game_date", "home_team"}
    if not required_cols.issubset(game_schedule.columns):
        raise ValueError(f"game_schedule must include columns {required_cols}")

    rows: list[WeatherLookupRow] = []
    for home_team, park_games in game_schedule.groupby("home_team"):
        if home_team not in PARKS:
            raise KeyError(f"No park mapping configured for home team '{home_team}'.")
        park = PARKS[home_team]
        point = Point(park["lat"], park["lon"], radius=75000)
        tz_name = park["tz"]
        local_dates = pd.to_datetime(park_games["game_date"]).dt.normalize().drop_duplicates().sort_values()
        if local_dates.empty:
            continue

        start_local = local_dates.min().tz_localize(tz_name) - timedelta(hours=2)
        end_local = local_dates.max().tz_localize(tz_name) + timedelta(hours=23)
        hourly_query = hourly(point, start_local.tz_convert("UTC").to_pydatetime().replace(tzinfo=None), end_local.tz_convert("UTC").to_pydatetime().replace(tzinfo=None))
        weather = hourly_query.fetch()
        if weather is None or weather.empty:
            for game_date in local_dates:
                rows.append(WeatherLookupRow(game_date=game_date, home_team=home_team,
                                             temperature_f=None, humidity_pct=None, wind_speed_mph=None,
                                             wind_direction_deg=None, pressure_hpa=None))
            continue

        weather = weather.tz_convert(tz_name)
        weather["weather_date"] = weather.index.normalize()
        weather["hour_diff"] = (weather.index.hour - DEFAULT_GAME_HOUR_LOCAL).abs()

        for game_date in local_dates:
            day_weather = weather[weather["weather_date"] == game_date]
            if day_weather.empty:
                rows.append(WeatherLookupRow(game_date=game_date, home_team=home_team,
                                             temperature_f=None, humidity_pct=None, wind_speed_mph=None,
                                             wind_direction_deg=None, pressure_hpa=None))
                continue
            best = day_weather.sort_values(["hour_diff", "temp"], na_position="last").iloc[0]
            rows.append(
                WeatherLookupRow(
                    game_date=game_date,
                    home_team=home_team,
                    temperature_f=_c_to_f(best.get("temp")),
                    humidity_pct=_safe_float(best.get("rhum")),
                    wind_speed_mph=_kph_to_mph(best.get("wspd")),
                    wind_direction_deg=_safe_float(best.get("wdir")),
                    pressure_hpa=_safe_float(best.get("pres")),
                )
            )

    weather_df = pd.DataFrame(rows)
    weather_df.to_csv(WEATHER_CACHE_PATH, index=False)
    return weather_df


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _c_to_f(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value) * 9 / 5 + 32


def _kph_to_mph(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value) * 0.621371
