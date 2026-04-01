"""Central configuration for the MLB home run prediction pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FINAL_DATA_PATH = DATA_DIR / "mlb_player_game_real.csv"
LIVE_DATA_DIR = DATA_DIR / "live"
LIVE_TRACKING_START_DATE = "2026-03-25"
LIVE_CURRENT_PICKS_PATH = LIVE_DATA_DIR / "current_picks.json"
LIVE_PICK_HISTORY_PATH = LIVE_DATA_DIR / "pick_history.json"
LIVE_MODEL_BUNDLE_PATH = LIVE_DATA_DIR / "model_bundle.pkl"
LIVE_MODEL_METADATA_PATH = LIVE_DATA_DIR / "model_metadata.json"
LIVE_MODEL_DATA_PATH = LIVE_DATA_DIR / "model_training_dataset.csv"
LIVE_DRAFT_PICKS_PATH = LIVE_DATA_DIR / "draft_picks.json"
LIVE_MORNING_BASELINE_PICKS_PATH = LIVE_DATA_DIR / "morning_baseline_picks.json"
LIVE_MODEL_START_DATE = "2024-03-28"

SEASON_START = "2024-03-28"
SEASON_END = "2024-09-30"
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
    "bb_type",
    "hc_x",
    "hc_y",
    "release_speed",
    "pitch_type",
]

PARKS = {
    "AZ": {"ballpark": "Chase Field", "region_abbr": "AZ", "field_bearing_deg": 20.0, "lat": 33.4455, "lon": -112.0667, "tz": "America/Phoenix"},
    "ATL": {"ballpark": "Truist Park", "region_abbr": "GA", "field_bearing_deg": 32.0, "lat": 33.8907, "lon": -84.4677, "tz": "America/New_York"},
    "BAL": {"ballpark": "Oriole Park at Camden Yards", "region_abbr": "MD", "field_bearing_deg": 60.0, "lat": 39.2838, "lon": -76.6217, "tz": "America/New_York"},
    "BOS": {"ballpark": "Fenway Park", "region_abbr": "MA", "field_bearing_deg": 50.0, "lat": 42.3467, "lon": -71.0972, "tz": "America/New_York"},
    "CHC": {"ballpark": "Wrigley Field", "region_abbr": "IL", "field_bearing_deg": 40.0, "lat": 41.9484, "lon": -87.6553, "tz": "America/Chicago"},
    "CWS": {"ballpark": "Guaranteed Rate Field", "region_abbr": "IL", "field_bearing_deg": 30.0, "lat": 41.83, "lon": -87.6338, "tz": "America/Chicago"},
    "CIN": {"ballpark": "Great American Ball Park", "region_abbr": "OH", "field_bearing_deg": 24.0, "lat": 39.0979, "lon": -84.5082, "tz": "America/New_York"},
    "CLE": {"ballpark": "Progressive Field", "region_abbr": "OH", "field_bearing_deg": 20.0, "lat": 41.4962, "lon": -81.6852, "tz": "America/New_York"},
    "COL": {"ballpark": "Coors Field", "region_abbr": "CO", "field_bearing_deg": 30.0, "lat": 39.7559, "lon": -104.9942, "tz": "America/Denver"},
    "DET": {"ballpark": "Comerica Park", "region_abbr": "MI", "field_bearing_deg": 20.0, "lat": 42.339, "lon": -83.0485, "tz": "America/New_York"},
    "HOU": {"ballpark": "Daikin Park", "region_abbr": "TX", "field_bearing_deg": 35.0, "lat": 29.7573, "lon": -95.3555, "tz": "America/Chicago"},
    "KC": {"ballpark": "Kauffman Stadium", "region_abbr": "MO", "field_bearing_deg": 25.0, "lat": 39.0517, "lon": -94.4803, "tz": "America/Chicago"},
    "LAA": {"ballpark": "Angel Stadium", "region_abbr": "CA", "field_bearing_deg": 45.0, "lat": 33.8003, "lon": -117.8827, "tz": "America/Los_Angeles"},
    "LAD": {"ballpark": "Dodger Stadium", "region_abbr": "CA", "field_bearing_deg": 35.0, "lat": 34.0739, "lon": -118.24, "tz": "America/Los_Angeles"},
    "MIA": {"ballpark": "loanDepot park", "region_abbr": "FL", "field_bearing_deg": 37.0, "lat": 25.7781, "lon": -80.2197, "tz": "America/New_York"},
    "MIL": {"ballpark": "American Family Field", "region_abbr": "WI", "field_bearing_deg": 32.0, "lat": 43.028, "lon": -87.9712, "tz": "America/Chicago"},
    "MIN": {"ballpark": "Target Field", "region_abbr": "MN", "field_bearing_deg": 30.0, "lat": 44.9817, "lon": -93.2776, "tz": "America/Chicago"},
    "NYM": {"ballpark": "Citi Field", "region_abbr": "NY", "field_bearing_deg": 40.0, "lat": 40.7571, "lon": -73.8458, "tz": "America/New_York"},
    "NYY": {"ballpark": "Yankee Stadium", "region_abbr": "NY", "field_bearing_deg": 52.0, "lat": 40.8296, "lon": -73.9262, "tz": "America/New_York"},
    "ATH": {"ballpark": "Oakland Coliseum", "region_abbr": "CA", "field_bearing_deg": 50.0, "lat": 37.7516, "lon": -122.2005, "tz": "America/Los_Angeles"},
    "OAK": {"ballpark": "Oakland Coliseum", "region_abbr": "CA", "field_bearing_deg": 50.0, "lat": 37.7516, "lon": -122.2005, "tz": "America/Los_Angeles"},
    "PHI": {"ballpark": "Citizens Bank Park", "region_abbr": "PA", "field_bearing_deg": 28.0, "lat": 39.9061, "lon": -75.1665, "tz": "America/New_York"},
    "PIT": {"ballpark": "PNC Park", "region_abbr": "PA", "field_bearing_deg": 35.0, "lat": 40.4469, "lon": -80.0057, "tz": "America/New_York"},
    "SD": {"ballpark": "Petco Park", "region_abbr": "CA", "field_bearing_deg": 20.0, "lat": 32.7073, "lon": -117.1566, "tz": "America/Los_Angeles"},
    "SEA": {"ballpark": "T-Mobile Park", "region_abbr": "WA", "field_bearing_deg": 18.0, "lat": 47.5914, "lon": -122.3325, "tz": "America/Los_Angeles"},
    "SF": {"ballpark": "Oracle Park", "region_abbr": "CA", "field_bearing_deg": 58.0, "lat": 37.7786, "lon": -122.3893, "tz": "America/Los_Angeles"},
    "STL": {"ballpark": "Busch Stadium", "region_abbr": "MO", "field_bearing_deg": 20.0, "lat": 38.6226, "lon": -90.1928, "tz": "America/Chicago"},
    "TB": {"ballpark": "Tropicana Field", "region_abbr": "FL", "field_bearing_deg": 40.0, "lat": 27.7683, "lon": -82.6534, "tz": "America/New_York"},
    "TEX": {"ballpark": "Globe Life Field", "region_abbr": "TX", "field_bearing_deg": 35.0, "lat": 32.7473, "lon": -97.0847, "tz": "America/Chicago"},
    "TOR": {"ballpark": "Rogers Centre", "region_abbr": "ON", "field_bearing_deg": 30.0, "lat": 43.6414, "lon": -79.3894, "tz": "America/Toronto"},
    "WSH": {"ballpark": "Nationals Park", "region_abbr": "DC", "field_bearing_deg": 30.0, "lat": 38.873, "lon": -77.0074, "tz": "America/New_York"},
}
