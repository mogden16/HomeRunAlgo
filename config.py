"""Central configuration for the MLB home run prediction pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FINAL_DATA_PATH = DATA_DIR / "mlb_player_game_real.csv"

SEASON_START = "2024-03-28"
SEASON_END = "2024-09-30"
DEFAULT_SEASON = int(SEASON_END[:4])
STATCAST_CHUNK_DAYS = 7
DEFAULT_GAME_HOUR_LOCAL = 19
MIN_DATASET_WARNING_ROWS = 1_000
TRAIN_FRACTION = 0.67
TSCV_N_SPLITS = 5
RANDOM_STATE = 42

AB_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "fielders_choice",
    "fielders_choice_out",
    "double_play",
    "triple_play",
    "strikeout",
    "strikeout_double_play",
    "other_out",
    "sac_fly_double_play",
    "field_error",
}

PA_ENDING_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "strikeout",
    "field_out",
    "double_play",
    "force_out",
    "field_error",
    "grounded_into_double_play",
    "hit_by_pitch",
    "fielders_choice",
    "fielders_choice_out",
    "triple_play",
    "sac_fly",
    "sac_bunt",
    "intent_walk",
    "sac_fly_double_play",
    "strikeout_double_play",
    "catcher_interf",
    "other_out",
}

STATCAST_COLUMNS = [
    "game_date",
    "game_pk",
    "game_type",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "home_team",
    "away_team",
    "batter",
    "pitcher",
    "player_name",
    "fielder_2",
    "events",
    "description",
    "stand",
    "p_throws",
    "home_score",
    "away_score",
    "on_1b",
    "on_2b",
    "on_3b",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "bb_type",
    "hc_x",
    "hc_y",
    "release_speed",
    "pitch_type",
    "pitch_name",
    "release_spin_rate",
    "release_extension",
    "spin_axis",
]

PARKS = {
    "AZ": {"ballpark": "Chase Field", "lat": 33.4455, "lon": -112.0667, "tz": "America/Phoenix"},
    "ATL": {"ballpark": "Truist Park", "lat": 33.8907, "lon": -84.4677, "tz": "America/New_York"},
    "BAL": {"ballpark": "Oriole Park at Camden Yards", "lat": 39.2838, "lon": -76.6217, "tz": "America/New_York"},
    "BOS": {"ballpark": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "tz": "America/New_York"},
    "CHC": {"ballpark": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "tz": "America/Chicago"},
    "CWS": {"ballpark": "Guaranteed Rate Field", "lat": 41.83, "lon": -87.6338, "tz": "America/Chicago"},
    "CIN": {"ballpark": "Great American Ball Park", "lat": 39.0979, "lon": -84.5082, "tz": "America/New_York"},
    "CLE": {"ballpark": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "tz": "America/New_York"},
    "COL": {"ballpark": "Coors Field", "lat": 39.7559, "lon": -104.9942, "tz": "America/Denver"},
    "DET": {"ballpark": "Comerica Park", "lat": 42.339, "lon": -83.0485, "tz": "America/New_York"},
    "HOU": {"ballpark": "Daikin Park", "lat": 29.7573, "lon": -95.3555, "tz": "America/Chicago"},
    "KC": {"ballpark": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "tz": "America/Chicago"},
    "LAA": {"ballpark": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "tz": "America/Los_Angeles"},
    "LAD": {"ballpark": "Dodger Stadium", "lat": 34.0739, "lon": -118.24, "tz": "America/Los_Angeles"},
    "MIA": {"ballpark": "loanDepot park", "lat": 25.7781, "lon": -80.2197, "tz": "America/New_York"},
    "MIL": {"ballpark": "American Family Field", "lat": 43.028, "lon": -87.9712, "tz": "America/Chicago"},
    "MIN": {"ballpark": "Target Field", "lat": 44.9817, "lon": -93.2776, "tz": "America/Chicago"},
    "NYM": {"ballpark": "Citi Field", "lat": 40.7571, "lon": -73.8458, "tz": "America/New_York"},
    "NYY": {"ballpark": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "tz": "America/New_York"},
    "ATH": {"ballpark": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005, "tz": "America/Los_Angeles"},
    "OAK": {"ballpark": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005, "tz": "America/Los_Angeles"},
    "PHI": {"ballpark": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "tz": "America/New_York"},
    "PIT": {"ballpark": "PNC Park", "lat": 40.4469, "lon": -80.0057, "tz": "America/New_York"},
    "SD": {"ballpark": "Petco Park", "lat": 32.7073, "lon": -117.1566, "tz": "America/Los_Angeles"},
    "SEA": {"ballpark": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "tz": "America/Los_Angeles"},
    "SF": {"ballpark": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "tz": "America/Los_Angeles"},
    "STL": {"ballpark": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "tz": "America/Chicago"},
    "TB": {"ballpark": "Tropicana Field", "lat": 27.7683, "lon": -82.6534, "tz": "America/New_York"},
    "TEX": {"ballpark": "Globe Life Field", "lat": 32.7473, "lon": -97.0847, "tz": "America/Chicago"},
    "TOR": {"ballpark": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "tz": "America/Toronto"},
    "WSH": {"ballpark": "Nationals Park", "lat": 38.873, "lon": -77.0074, "tz": "America/New_York"},
}

ATH_2025_PARK = {
    "ballpark": "Sutter Health Park",
    "lat": 38.5806,
    "lon": -121.5136,
    "tz": "America/Los_Angeles",
}


def season_from_date(date_value: str) -> int:
    return int(str(date_value)[:4])


def season_from_date_range(start_date: str, end_date: str) -> int:
    start_season = season_from_date(start_date)
    end_season = season_from_date(end_date)
    if start_season != end_season:
        raise ValueError(f"Expected a single MLB season date range, got {start_date} -> {end_date}.")
    return end_season


def weather_cache_path(season: int) -> Path:
    return RAW_DATA_DIR / f"weather_{season}.csv"


def park_factor_cache_path(season: int) -> Path:
    return RAW_DATA_DIR / f"park_factors_{season}.csv"


def raw_statcast_chunk_path(season: int, start_date: str, end_date: str) -> Path:
    return RAW_DATA_DIR / f"statcast_{season}_{start_date}_{end_date}.csv"


def get_park_info(team: str, season: int | None = None) -> dict[str, object]:
    if season is not None and season >= 2025 and team == "ATH":
        return ATH_2025_PARK
    return PARKS.get(team, {})


WEATHER_CACHE_PATH = weather_cache_path(DEFAULT_SEASON)
PARK_FACTOR_CACHE_PATH = park_factor_cache_path(DEFAULT_SEASON)
