"""Feature engineering for a one-row-per-batter-game Statcast dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pkgutil
import re
import string
import pybaseball
import requests

import numpy as np
import pandas as pd

from config import AB_EVENTS, FINAL_DATA_PATH, PA_ENDING_EVENTS, PARKS, STATCAST_COLUMNS

SPRAY_CENTER_X = 125.42
SPRAY_HOME_Y = 198.27
FASTBALL_TYPES = {"FF", "FT", "SI", "FC", "FA"}
BREAKING_BALL_TYPES = {"SL", "CU", "KC", "KN", "SV", "CS"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "SC", "EP"}
GRANULAR_PITCH_TYPES = {
    "four_seam": {"FF", "FA"},
    "sinker": {"SI", "FT"},
    "cutter": {"FC"},
    "slider": {"SL"},
    "curveball": {"CU", "KC", "KN", "SV", "CS"},
    "changeup": {"CH", "FS", "FO", "SC"},
}
PARK_FACTOR_LOOKUP_PATH = Path("data/park_factors_hr.csv")
MAX_PLACEHOLDER_BATTER_NAME_SHARE = 0.01
TEAM_VENUE_KEY_ALIASES = {
    "AZ": "ARI",
    "ARI": "ARI",
    "ARIZONA": "ARI",
    "CWS": "CHW",
    "CHISOX": "CHW",
    "WHITESOX": "CHW",
    "KC": "KCR",
    "KANSASCITY": "KCR",
    "SD": "SDP",
    "SAN DIEGO": "SDP",
    "SF": "SFG",
    "SAN FRANCISCO": "SFG",
    "TB": "TBR",
    "TAMPA": "TBR",
    "WSH": "WSN",
    "WAS": "WSN",
    "NATIONALS": "WSN",
    "OAK": "ATH",
}
NEW_CONTEXT_FEATURE_FAMILIES = {
    "park": ["park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb"],
    "handedness_split": [
        "batter_hr_per_pa_vs_rhp", "batter_hr_per_pa_vs_lhp", "batter_barrels_per_pa_vs_rhp", "batter_barrels_per_pa_vs_lhp",
        "batter_hard_hit_rate_vs_rhp", "batter_hard_hit_rate_vs_lhp", "batter_95plus_ev_rate_vs_rhp", "batter_95plus_ev_rate_vs_lhp",
        "batter_hr_per_pa_vs_pitcher_hand", "batter_barrels_per_pa_vs_pitcher_hand", "batter_hard_hit_rate_vs_pitcher_hand", "batter_95plus_ev_rate_vs_pitcher_hand",
        "pitcher_hr_allowed_per_pa_vs_rhb", "pitcher_hr_allowed_per_pa_vs_lhb", "pitcher_barrels_allowed_per_bbe_vs_rhb", "pitcher_barrels_allowed_per_bbe_vs_lhb",
        "pitcher_hard_hit_allowed_rate_vs_rhb", "pitcher_hard_hit_allowed_rate_vs_lhb", "pitcher_95plus_ev_allowed_rate_vs_rhb", "pitcher_95plus_ev_allowed_rate_vs_lhb",
        "pitcher_hr_allowed_per_pa_vs_batter_hand", "pitcher_barrels_allowed_per_bbe_vs_batter_hand", "pitcher_hard_hit_allowed_rate_vs_batter_hand", "pitcher_95plus_ev_allowed_rate_vs_batter_hand",
        "split_matchup_hr", "split_matchup_barrel", "split_matchup_hard_hit",
    ],
    "pitch_type_matchup": [
        "pitcher_four_seam_pct", "pitcher_sinker_pct", "pitcher_cutter_pct", "pitcher_slider_pct", "pitcher_curveball_pct", "pitcher_changeup_pct",
        "batter_hard_hit_rate_vs_breaking", "batter_barrel_rate_vs_breaking", "batter_contact_rate_vs_breaking",
        "batter_hard_hit_rate_vs_offspeed", "batter_barrel_rate_vs_offspeed", "batter_contact_rate_vs_offspeed",
        "batter_hard_hit_rate_vs_slider", "batter_barrel_rate_vs_slider", "batter_hard_hit_rate_vs_changeup", "batter_barrel_rate_vs_changeup",
        "batter_hard_hit_rate_vs_curveball", "batter_barrel_rate_vs_curveball",
        "breaking_matchup_hard_hit", "breaking_matchup_barrel", "offspeed_matchup_hard_hit", "offspeed_matchup_barrel",
        "slider_matchup_barrel", "changeup_matchup_barrel",
    ],
}
REQUESTED_FEATURE_AUDIT = {
    "park": ["park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb", "park_factor_hr_vs_batter_hand"],
    "handedness": [
        "batter_hr_per_pa_vs_rhp", "batter_hr_per_pa_vs_lhp", "batter_barrels_per_pa_vs_rhp", "batter_barrels_per_pa_vs_lhp",
        "batter_hard_hit_rate_vs_rhp", "batter_hard_hit_rate_vs_lhp", "batter_95plus_ev_rate_vs_rhp", "batter_95plus_ev_rate_vs_lhp",
        "batter_hr_per_pa_vs_pitcher_hand", "batter_barrels_per_pa_vs_pitcher_hand", "batter_hard_hit_rate_vs_pitcher_hand", "batter_95plus_ev_rate_vs_pitcher_hand",
        "pitcher_hr_allowed_per_pa_vs_rhb", "pitcher_hr_allowed_per_pa_vs_lhb", "pitcher_barrels_allowed_per_bbe_vs_rhb", "pitcher_barrels_allowed_per_bbe_vs_lhb",
        "pitcher_hard_hit_allowed_rate_vs_rhb", "pitcher_hard_hit_allowed_rate_vs_lhb", "pitcher_95plus_ev_allowed_rate_vs_rhb", "pitcher_95plus_ev_allowed_rate_vs_lhb",
        "pitcher_hr_allowed_per_pa_vs_batter_hand", "pitcher_barrels_allowed_per_bbe_vs_batter_hand", "pitcher_hard_hit_allowed_rate_vs_batter_hand",
        "pitcher_95plus_ev_allowed_rate_vs_batter_hand", "split_matchup_hr", "split_matchup_barrel", "split_matchup_hard_hit",
    ],
    "pitch_type": [
        "batter_hard_hit_rate_vs_breaking", "batter_barrel_rate_vs_breaking", "batter_contact_rate_vs_breaking",
        "batter_hard_hit_rate_vs_offspeed", "batter_barrel_rate_vs_offspeed", "batter_contact_rate_vs_offspeed",
        "breaking_matchup_hard_hit", "breaking_matchup_barrel", "offspeed_matchup_hard_hit", "offspeed_matchup_barrel",
        "pitcher_four_seam_pct", "pitcher_sinker_pct", "pitcher_slider_pct", "pitcher_curveball_pct", "pitcher_changeup_pct",
        "batter_hard_hit_rate_vs_slider", "batter_barrel_rate_vs_slider", "slider_matchup_barrel", "changeup_matchup_barrel",
    ],
}
CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
SWING_DESCRIPTIONS = CONTACT_DESCRIPTIONS | {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_bunt",
    "missed_bunt",
    "foul_pitchout",
}
MLB_PEOPLE_LOOKUP_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_PEOPLE_BATCH_SIZE = 100
SOURCE_SUMMARY_COLUMNS = [
    "game_pk",
    "batter",
    "pitcher",
    "pitch_type",
    "launch_speed",
    "launch_angle",
    "events",
    "description",
    "stand",
    "p_throws",
]
BATTER_TRAILING_FEATURE_COLUMNS = [
    "hr_per_pa_last_30d",
    "hr_per_pa_last_10d",
    "hr_count_last_30d",
    "hr_count_last_10d",
    "pa_last_30d",
    "pa_last_10d",
    "barrels_per_pa_last_30d",
    "barrels_per_pa_last_10d",
    "barrels_last_30d",
    "barrels_last_10d",
    "hard_hit_rate_last_30d",
    "hard_hit_rate_last_10d",
    "bbe_95plus_ev_rate_last_30d",
    "bbe_95plus_ev_rate_last_10d",
    "avg_exit_velocity_last_10d",
    "max_exit_velocity_last_10d",
]
PITCHER_TRAILING_FEATURE_COLUMNS = [
    "pitcher_hr_allowed_per_pa_last_30d",
    "pitcher_hr_allowed_last_30d",
    "pitcher_barrels_allowed_per_bbe_last_30d",
    "pitcher_barrels_allowed_last_30d",
    "pitcher_hard_hit_allowed_rate_last_30d",
    "pitcher_hard_hit_allowed_last_30d",
    "pitcher_bbe_allowed_last_30d",
    "pitcher_avg_ev_allowed_last_30d",
    "pitcher_95plus_ev_allowed_rate_last_30d",
]
STABLE_BATTER_FEATURE_COLUMNS = [
    "hr_rate_season_to_date",
    "hr_per_pa_last_30d",
    "barrel_rate_last_50_bbe",
    "hard_hit_rate_last_50_bbe",
    "avg_launch_angle_last_50_bbe",
    "avg_exit_velocity_last_50_bbe",
    "fly_ball_rate_last_50_bbe",
    "pull_air_rate_last_50_bbe",
    "batter_k_rate_season_to_date",
    "batter_bb_rate_season_to_date",
    "recent_form_hr_last_7d",
    "recent_form_barrels_last_14d",
    "expected_pa_proxy",
    "days_since_last_game",
]
STABLE_PITCHER_CONTEXT_COLUMNS = [
    "opp_pitcher_id",
    "pitch_hand_primary",
    "pitcher_hr9_season_to_date",
    "pitcher_barrel_rate_allowed_last_50_bbe",
    "pitcher_hard_hit_rate_allowed_last_50_bbe",
    "pitcher_fb_rate_allowed_last_50_bbe",
    "pitcher_k_rate_season_to_date",
    "pitcher_bb_rate_season_to_date",
    "starter_or_bullpen_proxy",
]
STABLE_CONTEXT_FEATURE_COLUMNS = [
    "temperature_f",
    "humidity_pct",
    "wind_speed_mph",
    "wind_direction_deg",
    "pressure_hpa",
    "platoon_advantage",
]
STABLE_ENGINEERED_FEATURE_COLUMNS = [
    *STABLE_BATTER_FEATURE_COLUMNS,
    *STABLE_PITCHER_CONTEXT_COLUMNS,
    *STABLE_CONTEXT_FEATURE_COLUMNS,
]

FINAL_FEATURE_SPECS = {
    "hr_rate_season_to_date": "stable",
    "hr_per_pa_last_30d": "stable",
    "barrel_rate_last_50_bbe": "stable",
    "hard_hit_rate_last_50_bbe": "stable",
    "avg_launch_angle_last_50_bbe": "stable",
    "avg_exit_velocity_last_50_bbe": "stable",
    "fly_ball_rate_last_50_bbe": "stable",
    "pull_air_rate_last_50_bbe": "stable",
    "batter_k_rate_season_to_date": "stable",
    "batter_bb_rate_season_to_date": "stable",
    "recent_form_hr_last_7d": "stable",
    "recent_form_barrels_last_14d": "stable",
    "expected_pa_proxy": "stable",
    "days_since_last_game": "stable",
    "pitcher_hr9_season_to_date": "stable",
    "pitcher_barrel_rate_allowed_last_50_bbe": "stable",
    "pitcher_hard_hit_rate_allowed_last_50_bbe": "stable",
    "pitcher_fb_rate_allowed_last_50_bbe": "stable",
    "pitcher_k_rate_season_to_date": "stable",
    "pitcher_bb_rate_season_to_date": "stable",
    "hr_per_pa_last_30d": "batter_trailing",
    "hr_per_pa_last_10d": "batter_trailing",
    "hr_count_last_30d": "batter_trailing",
    "hr_count_last_10d": "batter_trailing",
    "pa_last_30d": "batter_trailing",
    "pa_last_10d": "batter_trailing",
    "barrels_per_pa_last_30d": "batter_trailing",
    "barrels_per_pa_last_10d": "batter_trailing",
    "barrels_last_30d": "batter_trailing",
    "barrels_last_10d": "batter_trailing",
    "hard_hit_rate_last_30d": "batter_trailing",
    "hard_hit_rate_last_10d": "batter_trailing",
    "bbe_95plus_ev_rate_last_30d": "batter_trailing",
    "bbe_95plus_ev_rate_last_10d": "batter_trailing",
    "avg_exit_velocity_last_10d": "batter_trailing",
    "max_exit_velocity_last_10d": "batter_trailing",
    "pitcher_hr_allowed_per_pa_last_30d": "pitcher_trailing",
    "pitcher_hr_allowed_last_30d": "pitcher_trailing",
    "pitcher_barrels_allowed_per_bbe_last_30d": "pitcher_trailing",
    "pitcher_barrels_allowed_last_30d": "pitcher_trailing",
    "pitcher_hard_hit_allowed_rate_last_30d": "pitcher_trailing",
    "pitcher_hard_hit_allowed_last_30d": "pitcher_trailing",
    "pitcher_bbe_allowed_last_30d": "pitcher_trailing",
    "pitcher_avg_ev_allowed_last_30d": "pitcher_trailing",
    "pitcher_95plus_ev_allowed_rate_last_30d": "pitcher_trailing",
    "pitcher_fastball_pct": "pitch_type",
    "pitcher_breaking_ball_pct": "pitch_type",
    "pitcher_offspeed_pct": "pitch_type",
    "batter_hard_hit_rate_vs_fastballs": "pitch_type",
    "batter_barrel_rate_vs_fastballs": "pitch_type",
    "batter_contact_rate_vs_fastballs": "pitch_type",
    "fastball_matchup_hard_hit": "pitch_type",
    "fastball_matchup_barrel": "pitch_type",
    "park_factor_hr": "park",
    "park_factor_hr_vs_lhb": "park",
    "park_factor_hr_vs_rhb": "park",
    "park_factor_hr_vs_batter_hand": "park",
    "batter_hr_per_pa_vs_rhp": "handedness_split",
    "batter_hr_per_pa_vs_lhp": "handedness_split",
    "batter_barrels_per_pa_vs_rhp": "handedness_split",
    "batter_barrels_per_pa_vs_lhp": "handedness_split",
    "batter_hard_hit_rate_vs_rhp": "handedness_split",
    "batter_hard_hit_rate_vs_lhp": "handedness_split",
    "batter_95plus_ev_rate_vs_rhp": "handedness_split",
    "batter_95plus_ev_rate_vs_lhp": "handedness_split",
    "batter_hr_per_pa_vs_pitcher_hand": "handedness_split",
    "batter_barrels_per_pa_vs_pitcher_hand": "handedness_split",
    "batter_hard_hit_rate_vs_pitcher_hand": "handedness_split",
    "batter_95plus_ev_rate_vs_pitcher_hand": "handedness_split",
    "pitcher_hr_allowed_per_pa_vs_rhb": "handedness_split",
    "pitcher_hr_allowed_per_pa_vs_lhb": "handedness_split",
    "pitcher_barrels_allowed_per_bbe_vs_rhb": "handedness_split",
    "pitcher_barrels_allowed_per_bbe_vs_lhb": "handedness_split",
    "pitcher_hard_hit_allowed_rate_vs_rhb": "handedness_split",
    "pitcher_hard_hit_allowed_rate_vs_lhb": "handedness_split",
    "pitcher_95plus_ev_allowed_rate_vs_rhb": "handedness_split",
    "pitcher_95plus_ev_allowed_rate_vs_lhb": "handedness_split",
    "pitcher_hr_allowed_per_pa_vs_batter_hand": "handedness_split",
    "pitcher_barrels_allowed_per_bbe_vs_batter_hand": "handedness_split",
    "pitcher_hard_hit_allowed_rate_vs_batter_hand": "handedness_split",
    "pitcher_95plus_ev_allowed_rate_vs_batter_hand": "handedness_split",
    "split_matchup_hr": "handedness_split",
    "split_matchup_barrel": "handedness_split",
    "split_matchup_hard_hit": "handedness_split",
    "pitcher_four_seam_pct": "pitch_type",
    "pitcher_sinker_pct": "pitch_type",
    "pitcher_cutter_pct": "pitch_type",
    "pitcher_slider_pct": "pitch_type",
    "pitcher_curveball_pct": "pitch_type",
    "pitcher_changeup_pct": "pitch_type",
    "batter_hard_hit_rate_vs_breaking": "pitch_type",
    "batter_barrel_rate_vs_breaking": "pitch_type",
    "batter_contact_rate_vs_breaking": "pitch_type",
    "batter_hard_hit_rate_vs_offspeed": "pitch_type",
    "batter_barrel_rate_vs_offspeed": "pitch_type",
    "batter_contact_rate_vs_offspeed": "pitch_type",
    "batter_hard_hit_rate_vs_slider": "pitch_type",
    "batter_barrel_rate_vs_slider": "pitch_type",
    "batter_hard_hit_rate_vs_changeup": "pitch_type",
    "batter_barrel_rate_vs_changeup": "pitch_type",
    "batter_hard_hit_rate_vs_curveball": "pitch_type",
    "batter_barrel_rate_vs_curveball": "pitch_type",
    "breaking_matchup_hard_hit": "pitch_type",
    "breaking_matchup_barrel": "pitch_type",
    "offspeed_matchup_hard_hit": "pitch_type",
    "offspeed_matchup_barrel": "pitch_type",
    "slider_matchup_barrel": "pitch_type",
    "changeup_matchup_barrel": "pitch_type",
}


@dataclass
class ExportDecision:
    source_table: str
    included_in_export: bool
    reason: str


def validate_pitch_level_df(df: pd.DataFrame) -> None:
    required = ["game_date", "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Pitch-level Statcast data is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Pitch-level Statcast dataframe is empty.")
    duplicate_pitch_keys = int(df.duplicated(["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"]).sum())
    if duplicate_pitch_keys:
        raise ValueError(f"Pitch-level Statcast dataframe has {duplicate_pitch_keys} duplicate pitch keys.")


def validate_batter_game_df(df: pd.DataFrame) -> None:
    required = ["game_pk", "game_date", "batter_id"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Batter-game dataframe is missing required columns: {missing}")
    duplicates = df[df.duplicated(["batter_id", "game_pk"], keep=False)][["batter_id", "game_pk"]].drop_duplicates()
    if not duplicates.empty:
        raise ValueError(f"Batter-game dataframe must be unique on batter_id + game_pk. Offending keys: {duplicates.head(10).to_dict('records')}")


def validate_batter_identity(df: pd.DataFrame, *, context: str, fail_on_placeholder_share: bool = False) -> dict[str, object]:
    required = ["batter_id", "batter_name"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing batter identity columns: {missing}")
    print(f"=== BATTER IDENTITY AUDIT: {context} ===")
    print(f"row count: {len(df):,}")
    print(f"distinct batter_id count: {df['batter_id'].nunique(dropna=True):,}")
    print(f"distinct batter_name count: {df['batter_name'].nunique(dropna=True):,}")
    grouped = df.dropna(subset=["batter_id", "batter_name"]).groupby("batter_id", dropna=False)["batter_name"].nunique(dropna=True)
    unstable = grouped[grouped > 1].sort_values(ascending=False)
    stable_mapping = unstable.empty
    name_series = df["batter_name"].astype("string")
    placeholder_mask = name_series.str.fullmatch(r"batter_\d+", case=False, na=False)
    null_name_count = int(name_series.isna().sum())
    placeholder_count = int(placeholder_mask.sum())
    placeholder_share = float(placeholder_count / len(df)) if len(df) else 0.0
    print(f"batter_id -> batter_name stable one-to-one: {stable_mapping}")
    print(f"null batter_name rows: {null_name_count:,}")
    print(f"placeholder batter_<id> rows: {placeholder_count:,} ({placeholder_share:.2%})")
    if placeholder_count:
        print("placeholder batter_name sample:")
        print(df.loc[placeholder_mask, ["batter_id", "batter_name", "game_pk", "game_date"]].head(20).to_string(index=False))
    print("top batter_id/batter_name pairs:")
    print(df[["batter_id", "batter_name"]].drop_duplicates().head(20).to_string(index=False))
    if not stable_mapping:
        offenders = unstable.head(20).index.tolist()
        detail = (
            df[df["batter_id"].isin(offenders)][["batter_id", "batter_name", "game_pk", "game_date"]]
            .drop_duplicates()
            .sort_values(["batter_id", "batter_name", "game_date"])
        )
        print("top 20 batter_ids with multiple batter_name values:")
        print(unstable.head(20).to_string())
        print("sample offending rows:")
        print(detail.head(60).to_string(index=False))
    print("==========================================\n")
    if fail_on_placeholder_share and placeholder_share > MAX_PLACEHOLDER_BATTER_NAME_SHARE:
        unresolved_ids = df.loc[placeholder_mask, "batter_id"].dropna().drop_duplicates().head(20).tolist()
        print(f"unresolved batter_id sample (top 20): {unresolved_ids}")
        raise ValueError(
            f"{context}: placeholder batter_<id> names remain too high ({placeholder_share:.2%} > {MAX_PLACEHOLDER_BATTER_NAME_SHARE:.2%})."
        )
    return {
        "stable": stable_mapping,
        "unstable_ids": unstable.head(20).index.tolist(),
        "unstable_count": int(len(unstable)),
        "placeholder_count": placeholder_count,
        "placeholder_share": placeholder_share,
        "null_name_count": null_name_count,
    }


def validate_pitcher_game_df(df: pd.DataFrame) -> None:
    required = ["game_pk", "game_date", "pitcher_id"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Pitcher-game dataframe is missing required columns: {missing}")
    duplicates = df[df.duplicated(["pitcher_id", "game_pk"], keep=False)][["pitcher_id", "game_pk"]].drop_duplicates()
    if not duplicates.empty:
        raise ValueError(f"Pitcher-game dataframe must be unique on pitcher_id + game_pk. Offending keys: {duplicates.head(10).to_dict('records')}")


def validate_final_model_df(df: pd.DataFrame) -> None:
    validate_batter_game_df(df.rename(columns={"player_id": "batter_id"}) if "player_id" in df.columns and "batter_id" not in df.columns else df)


def print_source_summary(df: pd.DataFrame, source_type: str) -> None:
    game_dates = pd.to_datetime(df.get("game_date"), errors="coerce") if "game_date" in df.columns else pd.Series(dtype="datetime64[ns]")
    date_min = game_dates.min() if not game_dates.empty else pd.NaT
    date_max = game_dates.max() if not game_dates.empty else pd.NaT
    print("\n=== SOURCE SUMMARY ===")
    print(f"source_type: {source_type}")
    print(f"row_count: {len(df):,}")
    print(f"date_range: {date_min} -> {date_max}")
    for column in SOURCE_SUMMARY_COLUMNS:
        print(f"has_{column}: {column in df.columns}")
    print(f"supports_batter_game_features: {all(column in df.columns for column in ['game_pk', 'batter', 'events'])}")
    print(f"supports_pitcher_recent_contact: {all(column in df.columns for column in ['game_pk', 'pitcher', 'launch_speed'])}")
    print(f"supports_pitch_type_matchups_from_raw: {all(column in df.columns for column in ['pitch_type', 'description', 'launch_speed'])}")
    print("======================\n")


def build_player_game_dataset(statcast_df: pd.DataFrame, debug_feature_audit: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    del debug_feature_audit
    raw_df = statcast_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"], errors="coerce")
    validate_pitch_level_df(raw_df)
    print_source_summary(raw_df, "pybaseball.statcast pitch-level Statcast")

    pa_df = extract_plate_appearances(raw_df)
    batter_name_lookup, batter_lookup_status = build_batter_name_lookup(raw_df, pa_df)
    batter_game_df = aggregate_batter_games(pa_df, batter_name_lookup=batter_name_lookup)
    pitcher_game_df = aggregate_pitcher_games(pa_df)
    primary_pitchers = select_primary_opposing_pitchers(pitcher_game_df)
    batter_game_df = merge_with_diagnostics(
        batter_game_df,
        primary_pitchers,
        on=["game_pk", "team", "opponent"],
        how="left",
        step_name="attach primary opposing pitcher context",
        validate="many_to_one",
    )
    for column in ["opp_pitcher_id", "opp_pitcher_name", "pitch_hand_primary", "opp_pitcher_bf"]:
        right_column = f"{column}_y"
        left_column = f"{column}_x"
        if right_column in batter_game_df.columns and left_column in batter_game_df.columns:
            batter_game_df[column] = batter_game_df[right_column].combine_first(batter_game_df[left_column])
    duplicate_merge_columns = [column for column in batter_game_df.columns if column.endswith("_x") or column.endswith("_y")]
    if duplicate_merge_columns:
        batter_game_df = batter_game_df.drop(columns=duplicate_merge_columns)
    validate_batter_game_df(batter_game_df)
    validate_pitcher_game_df(pitcher_game_df)

    print("=== PLATE APPEARANCE DIAGNOSTICS ===")
    print(f"total pitch rows: {len(raw_df):,}")
    print(f"total inferred plate appearances: {len(pa_df):,}")
    print(f"total batter-game rows: {len(batter_game_df):,}")
    print(f"average PA per batter-game: {batter_game_df['pa_count'].mean():.3f}")
    print(f"raw Statcast player_name usable for batter lookup: {batter_lookup_status.get('player_name_usable')}")
    print(f"reverse lookup needed: {batter_lookup_status.get('reverse_lookup_needed')}")
    print(f"placeholder share after lookup resolution: {batter_lookup_status.get('final_placeholder_share', 0.0):.2%}")
    print(f"real batter names present after lookup resolution: {batter_lookup_status.get('real_names_ready')}")
    print("====================================\n")
    return batter_game_df, pitcher_game_df


def extract_plate_appearances(statcast_df: pd.DataFrame) -> pd.DataFrame:
    df = statcast_df.copy()
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    pa_end_mask = df["events"].notna() & df["events"].isin(PA_ENDING_EVENTS)
    if not pa_end_mask.any():
        raise RuntimeError("Could not infer any plate appearances from pitch-level Statcast events.")

    pa_df = df.loc[pa_end_mask].drop_duplicates(["game_pk", "at_bat_number"], keep="last").copy()
    pa_df["batting_team"] = np.where(pa_df["inning_topbot"].eq("Top"), pa_df["away_team"], pa_df["home_team"])
    pa_df["fielding_team"] = np.where(pa_df["inning_topbot"].eq("Top"), pa_df["home_team"], pa_df["away_team"])
    pa_df["is_home"] = pa_df["batting_team"].eq(pa_df["home_team"]).astype(int)
    pa_df["plate_appearance"] = 1
    pa_df["pa_count"] = 1
    pa_df["ab_count"] = pa_df["events"].isin(AB_EVENTS).astype(int)
    pa_df["hr_count"] = pa_df["events"].eq("home_run").astype(int)
    pa_df["is_bb"] = pa_df["events"].isin(["walk", "intent_walk"]).astype(int)
    pa_df["is_k"] = pa_df["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pa_df["bbe_count"] = pa_df["launch_speed"].notna().astype(int)
    pa_df["barrel_count"] = is_barrel(pa_df["launch_speed"], pa_df["launch_angle"]).astype(int)
    pa_df["hard_hit_bbe_count"] = ((pd.to_numeric(pa_df["launch_speed"], errors="coerce") >= 95) & pa_df["launch_speed"].notna()).astype(int)
    pa_df["ev_95plus_bbe_count"] = pa_df["hard_hit_bbe_count"]
    pa_df["fly_ball_bbe_count"] = pa_df["bb_type"].isin(["fly_ball", "popup"]).astype(int)
    pa_df["spray_angle"] = np.degrees(np.arctan2(pa_df["hc_x"] - SPRAY_CENTER_X, SPRAY_HOME_Y - pa_df["hc_y"]))
    pa_df["pull_air_bbe_count"] = is_pull_air(pa_df).astype(int)
    pa_df["contact_event_count"] = pa_df["description"].isin(CONTACT_DESCRIPTIONS).astype(int)
    pa_df["swing_event_count"] = pa_df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    pa_df["pitch_type_bucket"] = pa_df["pitch_type"].map(classify_pitch_type_bucket) if "pitch_type" in pa_df.columns else np.nan
    pa_df["inferred_batter_name"] = pa_df["des"].map(infer_batter_name_from_description) if "des" in pa_df.columns else np.nan
    pa_df["batting_order"] = pa_df.groupby(["game_pk", "batting_team"])["at_bat_number"].rank(method="dense").clip(upper=9)
    return pa_df


def infer_batter_name_from_description(description: object) -> str | float:
    if pd.isna(description):
        return np.nan
    text = str(description).strip()
    if not text:
        return np.nan

    intentional_walk_match = re.search(r"intentionally walks ([^\\.]+)", text, flags=re.IGNORECASE)
    if intentional_walk_match:
        return intentional_walk_match.group(1).strip()

    pattern = re.compile(
        r"^([A-Za-zÀ-ÖØ-öø-ÿ'., -]+?) "
        r"(?:singles|doubles|triples|homers|walks|strikes out|grounds|flies|lines|"
        r"pops|reaches|called out|out on|hits|bunts|sacrifices|hit by pitch)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return canonicalize_batter_name(match.group(1).strip())
    return np.nan


def canonicalize_batter_name(name: object) -> str | float:
    if pd.isna(name):
        return np.nan
    cleaned = str(name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\b(bunt|ground|fly|line|pop)\b$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or np.nan


def _fallback_player_reverse_lookup(player_ids: list[int]) -> pd.Series:
    if not player_ids:
        return pd.Series(dtype="string")
    resolved_frames: list[pd.DataFrame] = []
    for start in range(0, len(player_ids), MLB_PEOPLE_BATCH_SIZE):
        batch = [str(pid) for pid in player_ids[start : start + MLB_PEOPLE_BATCH_SIZE]]
        try:
            response = requests.get(
                MLB_PEOPLE_LOOKUP_URL,
                params={"personIds": ",".join(batch)},
                timeout=60,
            )
            response.raise_for_status()
            people = response.json().get("people", [])
        except Exception as exc:
            print(f"WARNING: MLB Stats API batter-name fallback failed for batch starting at {start}: {exc}")
            people = []
        if not people:
            continue
        frame = pd.DataFrame(people)
        if frame.empty or "id" not in frame.columns or "fullName" not in frame.columns:
            continue
        frame = frame.rename(columns={"id": "key_mlbam", "fullName": "batter_name_candidate"})
        frame["batter_name_candidate"] = frame["batter_name_candidate"].map(canonicalize_batter_name)
        frame["key_mlbam"] = pd.to_numeric(frame["key_mlbam"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["key_mlbam", "batter_name_candidate"]).drop_duplicates(subset=["key_mlbam"])
        if not frame.empty:
            resolved_frames.append(frame[["key_mlbam", "batter_name_candidate"]])
    if resolved_frames:
        resolved = pd.concat(resolved_frames, ignore_index=True).drop_duplicates(subset=["key_mlbam"], keep="first")
        return resolved.set_index("key_mlbam")["batter_name_candidate"]
    try:
        lookup_df = pybaseball.playerid_reverse_lookup([str(pid) for pid in player_ids], key_type="mlbam")
    except Exception as exc:
        print(f"WARNING: pybaseball.playerid_reverse_lookup fallback failed: {exc}")
        return pd.Series(dtype="string")
    if lookup_df.empty or "key_mlbam" not in lookup_df.columns:
        return pd.Series(dtype="string")
    first_name_col = "name_first" if "name_first" in lookup_df.columns else None
    last_name_col = "name_last" if "name_last" in lookup_df.columns else None
    if not first_name_col or not last_name_col:
        return pd.Series(dtype="string")
    lookup_df["batter_name_candidate"] = (
        lookup_df[first_name_col].fillna("").astype(str).str.strip() + " " + lookup_df[last_name_col].fillna("").astype(str).str.strip()
    ).str.strip()
    lookup_df["batter_name_candidate"] = lookup_df["batter_name_candidate"].map(canonicalize_batter_name)
    lookup_df["key_mlbam"] = pd.to_numeric(lookup_df["key_mlbam"], errors="coerce").astype("Int64")
    resolved = lookup_df.dropna(subset=["key_mlbam", "batter_name_candidate"]).drop_duplicates(subset=["key_mlbam"])
    return resolved.set_index("key_mlbam")["batter_name_candidate"]


def _fallback_existing_dataset_lookup(player_ids: list[int]) -> pd.Series:
    if not player_ids or not FINAL_DATA_PATH.exists():
        return pd.Series(dtype="string")
    try:
        existing_df = pd.read_csv(FINAL_DATA_PATH, usecols=["batter_id", "batter_name"])
    except Exception as exc:
        print(f"WARNING: existing dataset batter-name fallback failed: {exc}")
        return pd.Series(dtype="string")
    if existing_df.empty:
        return pd.Series(dtype="string")
    existing_df["batter_name"] = existing_df["batter_name"].map(canonicalize_batter_name)
    existing_df["batter_id"] = pd.to_numeric(existing_df["batter_id"], errors="coerce").astype("Int64")
    existing_df = existing_df.dropna(subset=["batter_id", "batter_name"])
    existing_df = existing_df[existing_df["batter_id"].isin(player_ids)]
    if existing_df.empty:
        return pd.Series(dtype="string")
    resolved = (
        existing_df.groupby(["batter_id", "batter_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["batter_id", "count"], ascending=[True, False])
        .drop_duplicates(subset=["batter_id"], keep="first")
    )
    return resolved.set_index("batter_id")["batter_name"]


def build_batter_name_lookup(raw_df: pd.DataFrame, pa_df: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    candidate_frames: list[pd.DataFrame] = []
    status: dict[str, object] = {"player_name_usable": False, "reverse_lookup_needed": False, "final_placeholder_share": 1.0, "real_names_ready": False}
    all_batter_ids = (
        pd.to_numeric(raw_df["batter"], errors="coerce").dropna().astype("Int64").unique().tolist()
        if "batter" in raw_df.columns
        else []
    )

    def finalize_lookup(lookup: pd.Series, candidate_rows: int) -> tuple[pd.Series, dict[str, object]]:
        lookup = lookup.dropna().astype("string")
        unresolved_local = sorted(
            set(all_batter_ids) - set(pd.to_numeric(pd.Series(lookup.index), errors="coerce").dropna().astype("Int64").tolist())
        )
        if unresolved_local:
            status["reverse_lookup_needed"] = True
            reverse_lookup_series = _fallback_player_reverse_lookup(unresolved_local)
            print("=== BATTER REVERSE LOOKUP AUDIT ===")
            print(f"unresolved IDs before reverse lookup: {len(unresolved_local):,}")
            print(f"IDs resolved by reverse lookup: {reverse_lookup_series.index.nunique():,}")
            unresolved_after_reverse = sorted(
                set(unresolved_local)
                - set(pd.to_numeric(pd.Series(reverse_lookup_series.index), errors="coerce").dropna().astype("Int64").tolist())
            )
            print(f"IDs unresolved after reverse lookup: {len(unresolved_after_reverse):,}")
            if not reverse_lookup_series.empty:
                print(f"reverse lookup sample: {reverse_lookup_series.head(20).to_dict()}")
                lookup = pd.concat([lookup, reverse_lookup_series[~reverse_lookup_series.index.isin(lookup.index)]])
            print("===================================\n")
            unresolved_local = sorted(
                set(all_batter_ids) - set(pd.to_numeric(pd.Series(lookup.index), errors="coerce").dropna().astype("Int64").tolist())
            )
        if unresolved_local:
            existing_dataset_series = _fallback_existing_dataset_lookup(unresolved_local)
            print("=== EXISTING DATASET NAME FALLBACK AUDIT ===")
            print(f"IDs unresolved before dataset fallback: {len(unresolved_local):,}")
            print(f"IDs resolved from existing engineered dataset: {existing_dataset_series.index.nunique():,}")
            if not existing_dataset_series.empty:
                print(f"existing dataset fallback sample: {existing_dataset_series.head(20).to_dict()}")
                lookup = pd.concat([lookup, existing_dataset_series[~existing_dataset_series.index.isin(lookup.index)]])
            print("===========================================\n")
        print("=== BATTER NAME LOOKUP BUILD AUDIT ===")
        print(f"lookup source columns from raw statcast: {name_columns if name_columns else 'none; parsed descriptions used'}")
        print(f"lookup candidate rows: {candidate_rows:,}")
        print(f"lookup distinct batter_ids: {lookup.index.nunique():,}")
        print(f"lookup sample: {lookup.head(20).to_dict()}")
        print("======================================\n")
        status["real_names_ready"] = bool(len(lookup) > 0)
        status["final_placeholder_share"] = 0.0 if status["real_names_ready"] else 1.0
        return lookup, status

    if "inferred_batter_name" in pa_df.columns:
        candidate_frames.append(
            pa_df[["batter", "inferred_batter_name"]].rename(columns={"batter": "batter_id", "inferred_batter_name": "batter_name_candidate"})
        )
    name_columns = [col for col in ["batter_name", "batter_fullname", "hitter_name"] if col in raw_df.columns]
    for col in name_columns:
        candidate_frames.append(raw_df[["batter", col]].rename(columns={"batter": "batter_id", col: "batter_name_candidate"}))
    if {"batter", "player_name"}.issubset(raw_df.columns):
        player_name_audit = (
            raw_df.dropna(subset=["batter", "player_name"])
            .groupby("batter", dropna=False)["player_name"]
            .nunique(dropna=True)
        )
        player_name_usable = bool((player_name_audit <= 2).mean() >= 0.80) if len(player_name_audit) else False
        status["player_name_usable"] = player_name_usable
        print("=== RAW player_name USABILITY AUDIT ===")
        print(f"batter->player_name stability share (<=2 names): {(player_name_audit <= 2).mean() if len(player_name_audit) else 0.0:.3f}")
        print(f"raw player_name usable for batter lookup: {player_name_usable}")
        sample_counts = player_name_audit.sort_values(ascending=False).head(20)
        print("sample distinct player_name count per batter_id:")
        print(sample_counts.to_string())
        print("=======================================\n")
        if player_name_usable:
            candidate_frames.append(raw_df[["batter", "player_name"]].rename(columns={"batter": "batter_id", "player_name": "batter_name_candidate"}))
    if "des" in raw_df.columns:
        parsed = raw_df[["batter", "des"]].copy()
        parsed["batter_name_candidate"] = parsed["des"].map(infer_batter_name_from_description)
        candidate_frames.append(parsed[["batter", "batter_name_candidate"]].rename(columns={"batter": "batter_id"}))

    if not candidate_frames:
        return finalize_lookup(pd.Series(dtype="string"), 0)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates["batter_name_candidate"] = candidates["batter_name_candidate"].map(canonicalize_batter_name)
    candidates = candidates.dropna(subset=["batter_id", "batter_name_candidate"])
    candidates = candidates[~candidates["batter_name_candidate"].astype("string").str.fullmatch(r"batter_\d+", case=False, na=False)]
    if candidates.empty:
        return finalize_lookup(pd.Series(dtype="string"), 0)
    lookup = (
        candidates.groupby(["batter_id", "batter_name_candidate"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["batter_id", "count"], ascending=[True, False])
        .drop_duplicates(subset=["batter_id"], keep="first")
        .set_index("batter_id")["batter_name_candidate"]
    )
    unresolved = sorted(set(all_batter_ids) - set(pd.to_numeric(pd.Series(lookup.index), errors="coerce").dropna().astype("Int64").tolist()))
    reverse_lookup_series = pd.Series(dtype="string")
    if unresolved:
        status["reverse_lookup_needed"] = True
        reverse_lookup_series = _fallback_player_reverse_lookup(unresolved)
        print("=== BATTER REVERSE LOOKUP AUDIT ===")
        print(f"unresolved IDs before reverse lookup: {len(unresolved):,}")
        print(f"IDs resolved by reverse lookup: {reverse_lookup_series.index.nunique():,}")
        unresolved_after_reverse = sorted(set(unresolved) - set(pd.to_numeric(pd.Series(reverse_lookup_series.index), errors='coerce').dropna().astype('Int64').tolist()))
        print(f"IDs unresolved after reverse lookup: {len(unresolved_after_reverse):,}")
        if not reverse_lookup_series.empty:
            print(f"reverse lookup sample: {reverse_lookup_series.head(20).to_dict()}")
        print("===================================\n")
        if not reverse_lookup_series.empty:
            lookup = pd.concat([lookup, reverse_lookup_series[~reverse_lookup_series.index.isin(lookup.index)]])
    return finalize_lookup(lookup, len(candidates))


def aggregate_batter_games(pa_df: pd.DataFrame, batter_name_lookup: pd.Series | None = None) -> pd.DataFrame:
    # Assumption: opponent_pitcher_id is the pitcher faced in the batter's terminal PA row.
    # We keep the most frequent pitcher seen by the batter in that game as the opposing pitcher proxy.
    batter_pitcher_map = (
        pa_df.groupby(["game_pk", "batter", "pitcher"], dropna=False)
        .agg(pa_vs_pitcher=("plate_appearance", "sum"), opp_pitcher_name=("player_name", "first"))
        .reset_index()
        .sort_values(["game_pk", "batter", "pa_vs_pitcher", "pitcher"], ascending=[True, True, False, True])
        .drop_duplicates(["game_pk", "batter"], keep="first")
        .rename(columns={"pitcher": "pitcher_id"})
    )

    batter_game_df = (
        pa_df.groupby(["game_pk", "game_date", "batter"], dropna=False)
        .agg(
            batter_name=("inferred_batter_name", "first"),
            team=("batting_team", "first"),
            opponent=("fielding_team", "first"),
            is_home=("is_home", "max"),
            bat_side=("stand", "first"),
            pitcher_hand=("p_throws", "first"),
            batting_order=("batting_order", "min"),
            pa_count=("pa_count", "sum"),
            ab_count=("ab_count", "sum"),
            hr_count=("hr_count", "sum"),
            bbe_count=("bbe_count", "sum"),
            barrel_count=("barrel_count", "sum"),
            hard_hit_bbe_count=("hard_hit_bbe_count", "sum"),
            batter_k_count=("is_k", "sum"),
            batter_bb_count=("is_bb", "sum"),
            avg_launch_angle=("launch_angle", "mean"),
            avg_exit_velocity=("launch_speed", "mean"),
            max_exit_velocity=("launch_speed", "max"),
            ev_95plus_bbe_count=("ev_95plus_bbe_count", "sum"),
            fly_ball_bbe_count=("fly_ball_bbe_count", "sum"),
            pull_air_bbe_count=("pull_air_bbe_count", "sum"),
        )
        .reset_index()
        .rename(columns={"batter": "batter_id"})
    )
    batter_game_df["hit_hr"] = (batter_game_df["hr_count"] > 0).astype(int)
    batter_game_df = merge_with_diagnostics(
        batter_game_df,
        batter_pitcher_map[["game_pk", "batter", "pitcher_id", "opp_pitcher_name"]].rename(columns={"batter": "batter_id"}),
        left_on=["game_pk", "batter_id"],
        right_on=["game_pk", "batter_id"],
        how="left",
        step_name="attach primary opposing pitcher to batter-game table",
        validate="one_to_one",
    )
    source_from_pa = int(batter_game_df["batter_name"].notna().sum())
    source_from_lookup = 0
    source_from_placeholder = 0
    if batter_name_lookup is not None and not batter_name_lookup.empty:
        before_lookup_non_null = int(batter_game_df["batter_name"].notna().sum())
        batter_game_df["batter_name"] = batter_game_df["batter_name"].where(
            batter_game_df["batter_name"].notna(),
            batter_game_df["batter_id"].map(batter_name_lookup),
        )
        source_from_lookup = int(batter_game_df["batter_name"].notna().sum()) - before_lookup_non_null

    missing_batter_names = batter_game_df["batter_name"].isna()
    if missing_batter_names.any():
        preferred_name = (
            batter_game_df.dropna(subset=["batter_name"])
            .groupby("batter_id", dropna=False)["batter_name"]
            .agg(lambda s: s.astype(str).value_counts().index[0])
        )
        batter_game_df["batter_name"] = batter_game_df["batter_name"].astype("string")
        batter_game_df.loc[missing_batter_names, "batter_name"] = batter_game_df.loc[missing_batter_names, "batter_id"].map(preferred_name)
        missing_batter_names = batter_game_df["batter_name"].isna()
        batter_game_df.loc[missing_batter_names, "batter_name"] = (
            "batter_" + batter_game_df.loc[missing_batter_names, "batter_id"].astype("Int64").astype(str)
        )
        source_from_placeholder = int(missing_batter_names.sum())
    batter_game_df["batter_name"] = batter_game_df["batter_name"].map(canonicalize_batter_name)
    batter_game_df["player_id"] = batter_game_df["batter_id"]
    batter_game_df["player_name"] = batter_game_df["batter_name"]
    batter_game_df["opp_pitcher_id"] = batter_game_df["pitcher_id"]
    batter_game_df["pitch_hand_primary"] = batter_game_df["pitcher_hand"]
    batter_game_df["opp_pitcher_bf"] = np.nan
    batter_game_df["ballpark"] = np.where(
        batter_game_df["is_home"].astype(bool),
        batter_game_df["team"].map(lambda team: PARKS.get(team, {}).get("ballpark")),
        batter_game_df["opponent"].map(lambda team: PARKS.get(team, {}).get("ballpark")),
    )
    batter_game_df["park_factor_hr"] = np.nan
    batter_game_df = batter_game_df.sort_values(["batter_id", "game_date", "game_pk"]).reset_index(drop=True)
    print("batter_name source in batter-game construction:")
    print(f"  from PA parsed batter-side text: {source_from_pa:,}")
    print(f"  from raw-data lookup merge: {source_from_lookup:,}")
    print(f"  fallback placeholder batter_<id>: {source_from_placeholder:,}")
    validate_batter_identity(batter_game_df, context="aggregate_batter_games output")
    return batter_game_df


def aggregate_pitcher_games(pa_df: pd.DataFrame) -> pd.DataFrame:
    pa_enriched = pa_df.copy()
    pa_enriched["is_rhb"] = pa_enriched["stand"].eq("R").astype(int)
    pa_enriched["is_lhb"] = pa_enriched["stand"].eq("L").astype(int)
    for side in ["rhb", "lhb"]:
        mask = pa_enriched[f"is_{side}"]
        pa_enriched[f"pa_against_{side}"] = pa_enriched["pa_count"] * mask
        pa_enriched[f"hr_allowed_{side}"] = pa_enriched["hr_count"] * mask
        pa_enriched[f"bbe_allowed_{side}"] = pa_enriched["bbe_count"] * mask
        pa_enriched[f"barrels_allowed_{side}"] = pa_enriched["barrel_count"] * mask
        pa_enriched[f"hard_hit_bbe_allowed_{side}"] = pa_enriched["hard_hit_bbe_count"] * mask
        pa_enriched[f"ev_95plus_bbe_allowed_{side}"] = pa_enriched["ev_95plus_bbe_count"] * mask
    pitcher_game_df = (
        pa_enriched.groupby(["game_pk", "game_date", "batting_team", "fielding_team", "pitcher"], dropna=False)
        .agg(
            pitcher_name=("player_name", "first"),
            p_throws=("p_throws", "first"),
            pa_against=("pa_count", "sum"),
            hr_allowed=("hr_count", "sum"),
            bbe_allowed=("bbe_count", "sum"),
            barrels_allowed=("barrel_count", "sum"),
            hard_hit_bbe_allowed=("hard_hit_bbe_count", "sum"),
            fb_allowed=("fly_ball_bbe_count", "sum"),
            pitcher_k_count=("is_k", "sum"),
            pitcher_bb_count=("is_bb", "sum"),
            batters_faced=("plate_appearance", "sum"),
            outs_recorded=("ab_count", "sum"),
            avg_ev_allowed=("launch_speed", "mean"),
            max_ev_allowed=("launch_speed", "max"),
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
        .rename(columns={"batting_team": "team", "fielding_team": "opponent", "pitcher": "pitcher_id"})
    )
    pitcher_game_df["innings_pitched_est"] = pitcher_game_df["outs_recorded"] / 3.0
    pitcher_game_df = pitcher_game_df.sort_values(["pitcher_id", "game_date", "game_pk"]).reset_index(drop=True)
    return pitcher_game_df


def select_primary_opposing_pitchers(pitcher_game_df: pd.DataFrame) -> pd.DataFrame:
    primary = (
        pitcher_game_df.sort_values(["game_pk", "team", "batters_faced", "pitcher_id"], ascending=[True, True, False, True])
        .drop_duplicates(["game_pk", "team"], keep="first")
        .rename(
            columns={
                "pitcher_id": "opp_pitcher_id",
                "pitcher_name": "opp_pitcher_name",
                "p_throws": "pitch_hand_primary",
                "batters_faced": "opp_pitcher_bf",
            }
        )
    )
    return primary[["game_pk", "team", "opponent", "opp_pitcher_id", "opp_pitcher_name", "pitch_hand_primary", "opp_pitcher_bf"]]


def add_leakage_safe_features(
    player_game_df: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    statcast_df: pd.DataFrame | None = None,
    debug_feature_audit: bool = False,
    missingness_threshold: float = 0.95,
) -> pd.DataFrame:
    del debug_feature_audit
    batter_df = player_game_df.copy()
    pitcher_df = pitcher_game_df.copy()
    batter_df["game_date"] = pd.to_datetime(batter_df["game_date"])
    pitcher_df["game_date"] = pd.to_datetime(pitcher_df["game_date"])
    validate_batter_game_df(batter_df)
    validate_pitcher_game_df(pitcher_df)
    raw_field_status = validate_required_raw_fields(statcast_df)

    batter_features = compute_batter_trailing_features(batter_df)
    pitcher_features = compute_pitcher_trailing_features(pitcher_df)
    stable_batter_features = compute_stable_batter_features(batter_df)
    stable_pitcher_features = compute_stable_pitcher_features(pitcher_df).rename(columns={"pitcher_id": "opp_pitcher_id"})
    batter_hand_features = compute_batter_handedness_split_features(batter_df)
    pitcher_hand_features = compute_pitcher_handedness_split_features(pitcher_df)

    print_pre_merge_feature_table_diagnostics(
        table_name="batter trailing feature table",
        feature_df=batter_features,
        feature_columns=BATTER_TRAILING_FEATURE_COLUMNS,
        key_columns=["batter_id", "game_pk"],
        left_df=batter_df,
        left_keys=["batter_id", "game_pk"],
    )
    print_pre_merge_feature_table_diagnostics(
        table_name="pitcher trailing feature table",
        feature_df=pitcher_features,
        feature_columns=PITCHER_TRAILING_FEATURE_COLUMNS,
        key_columns=["pitcher_id", "game_pk"],
        left_df=batter_df,
        left_keys=["pitcher_id", "game_pk"],
    )

    batter_before_counts = count_non_nulls(batter_features, BATTER_TRAILING_FEATURE_COLUMNS)
    pitcher_before_counts = count_non_nulls(pitcher_features, PITCHER_TRAILING_FEATURE_COLUMNS)

    dataset = merge_with_diagnostics(
        batter_df,
        batter_features,
        on=["batter_id", "game_pk"],
        how="left",
        step_name="merge batter trailing features",
        validate="one_to_one",
        tracked_feature_columns=BATTER_TRAILING_FEATURE_COLUMNS,
    )
    dataset = merge_with_diagnostics(
        dataset,
        pitcher_features,
        left_on=["pitcher_id", "game_pk"],
        right_on=["pitcher_id", "game_pk"],
        how="left",
        step_name="merge pitcher trailing features",
        validate="many_to_one",
        tracked_feature_columns=PITCHER_TRAILING_FEATURE_COLUMNS,
    )
    dataset = merge_with_diagnostics(
        dataset,
        stable_batter_features,
        on=["batter_id", "game_pk"],
        how="left",
        step_name="merge stable batter features",
        validate="one_to_one",
        tracked_feature_columns=STABLE_BATTER_FEATURE_COLUMNS + ["bbe_count_last_50"],
    )
    for column in ["hr_per_pa_last_30d"]:
        left_column = f"{column}_x"
        right_column = f"{column}_y"
        if left_column in dataset.columns and right_column in dataset.columns:
            dataset[column] = dataset[right_column].combine_first(dataset[left_column])
            dataset = dataset.drop(columns=[left_column, right_column])
    dataset = merge_with_diagnostics(
        dataset,
        stable_pitcher_features,
        on=["opp_pitcher_id", "game_pk"],
        how="left",
        step_name="merge stable pitcher features",
        validate="many_to_one",
        tracked_feature_columns=[
            "pitcher_hr9_season_to_date",
            "pitcher_barrel_rate_allowed_last_50_bbe",
            "pitcher_hard_hit_rate_allowed_last_50_bbe",
            "pitcher_fb_rate_allowed_last_50_bbe",
            "pitcher_k_rate_season_to_date",
            "pitcher_bb_rate_season_to_date",
        ],
    )

    print_before_after_trailing_report(
        batter_before_counts=batter_before_counts,
        batter_after_counts=count_non_nulls(dataset, BATTER_TRAILING_FEATURE_COLUMNS),
        pitcher_before_counts=pitcher_before_counts,
        pitcher_after_counts=count_non_nulls(dataset, PITCHER_TRAILING_FEATURE_COLUMNS),
    )

    pitch_type_status = {"included": False, "reason": "pitch-type features were not attempted"}
    park_status = {"included": False, "reason": "park factors were not attempted"}
    hand_status = {"included": False, "reason": "handedness split features were not attempted"}
    if statcast_df is not None:
        dataset, park_status = add_park_factor_features(dataset, statcast_df=statcast_df)
        pitch_type_features, pitch_type_status = build_pitch_type_matchup_features(statcast_df, dataset)
        if not pitch_type_features.empty:
            dataset = merge_with_diagnostics(
                dataset,
                pitch_type_features,
                on=["batter_id", "game_pk"],
                how="left",
                step_name="merge pitch-type matchup features",
                validate="one_to_one",
            )
    else:
        dataset, park_status = add_park_factor_features(dataset, statcast_df=None)

    dataset = merge_with_diagnostics(
        dataset,
        batter_hand_features,
        on=["batter_id", "game_pk"],
        how="left",
        step_name="merge batter handedness split features",
        validate="one_to_one",
    )
    dataset = merge_with_diagnostics(
        dataset,
        pitcher_hand_features,
        left_on=["pitcher_id", "game_pk"],
        right_on=["pitcher_id", "game_pk"],
        how="left",
        step_name="merge pitcher handedness split features",
        validate="many_to_one",
    )
    dataset = build_matchup_selected_handedness_features(dataset)
    hand_status = {"included": True, "reason": "batter/pitcher handedness split features were computed from prior games"}
    identity_suffix_columns = [col for col in dataset.columns if col in {"batter_name_x", "batter_name_y", "pitcher_name_x", "pitcher_name_y"}]
    batter_merge_overwrite_detected = len(identity_suffix_columns) > 0
    print(f"identity suffix columns after feature merges: {identity_suffix_columns if identity_suffix_columns else 'none'}")
    print(f"batter_name merge overwrite detected: {batter_merge_overwrite_detected}")
    batter_identity_status = validate_batter_identity(dataset, context="post feature merges", fail_on_placeholder_share=True)

    assert_no_duplicate_batter_game_rows(dataset, context="post feature merges")

    dataset["platoon_advantage"] = np.where(
        dataset["bat_side"].notna() & dataset["pitch_hand_primary"].notna(),
        (dataset["bat_side"] != dataset["pitch_hand_primary"]).astype(float),
        np.nan,
    )
    dataset["starter_or_bullpen_proxy"] = np.where(dataset["opp_pitcher_bf"] >= 12, "starter_like", "bullpen_like")

    decisions = finalize_feature_export(dataset, missingness_threshold=missingness_threshold)
    model_features = load_model_feature_list()
    if "park_factor_hr" in model_features and ("park_factor_hr" not in dataset.columns or dataset["park_factor_hr"].notna().sum() == 0):
        raise ValueError("park_factor_hr is in the model feature list but remains missing/100%-null after engineering.")
    final_columns = base_export_columns(dataset) + [feature for feature, decision in decisions.items() if decision.included_in_export]
    dataset = dataset.loc[:, [column for column in final_columns if column in dataset.columns]].copy()
    dataset = dataset.sort_values(["game_date", "game_pk", "batter_id"]).reset_index(drop=True)
    dataset["player_id"] = dataset["batter_id"]
    validate_final_model_df(dataset)
    print_final_feature_quality_summary(dataset, decisions)
    summarize_new_context_feature_quality(dataset, decisions)
    print_requested_feature_audit(dataset, decisions)
    print_focused_pass_summary(
        dataset,
        decisions,
        park_status=park_status,
        pybaseball_audit=park_status.get("pybaseball_audit"),
        batter_identity_status=batter_identity_status,
        batter_merge_overwrite_detected=batter_merge_overwrite_detected,
    )
    print_rerun_verdict(
        dataset,
        pitch_type_status=pitch_type_status,
        park_status=park_status,
        hand_status=hand_status,
        raw_field_status=raw_field_status,
    )
    return dataset


def compute_batter_trailing_features(batter_game_df: pd.DataFrame) -> pd.DataFrame:
    print("=== BATTER TRAILING FEATURE DIAGNOSTICS ===")
    print(f"input row count: {len(batter_game_df):,}")
    print(f"distinct batter count: {batter_game_df['batter_id'].nunique(dropna=True):,}")
    print(f"date range: {batter_game_df['game_date'].min()} -> {batter_game_df['game_date'].max()}")
    print(f"input grain one row per batter-game: {not batter_game_df.duplicated(['batter_id', 'game_pk']).any()}")
    print("sorting keys: ['batter_id', 'game_date', 'game_pk']")
    print("grouping keys: ['batter_id']")

    feature_frames: list[pd.DataFrame] = []
    rows_with_prior_1 = 0
    rows_with_prior_2 = 0
    sample_frames: list[pd.DataFrame] = []
    for sample_idx, (batter_id, group) in enumerate(batter_game_df.groupby("batter_id", sort=False), start=1):
        grp = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
        if grp[["game_date", "game_pk"]].duplicated().any():
            raise ValueError(f"Batter {batter_id} has duplicate game_date/game_pk rows after aggregation.")

        prior_games = np.arange(len(grp))
        rows_with_prior_1 += int((prior_games >= 1).sum())
        rows_with_prior_2 += int((prior_games >= 2).sum())

        # Root cause fixed here: the prior implementation built rolling Series with a reset 0..n-1 index
        # but assigned them back onto grouped frames that still carried their original row index labels.
        # Pandas aligned on index labels during assignment, which made the trailing columns appear all-null
        # after computation for most groups. Resetting per-group index before rolling keeps the results aligned.
        grp = _append_date_window_features(
            grp,
            entity_id=batter_id,
            entity_label="batter",
            configs=[("30D", "30d"), ("10D", "10d")],
        )
        feature_frames.append(grp[["batter_id", "game_pk"] + BATTER_TRAILING_FEATURE_COLUMNS])

        if sample_idx <= 3:
            sample_frames.append(grp[[
                "batter_id",
                "game_pk",
                "game_date",
                "pa_count",
                "hr_count",
                "barrel_count",
                "hard_hit_bbe_count",
                "hr_per_pa_last_30d",
                "hr_per_pa_last_10d",
                "barrels_per_pa_last_30d",
                "hard_hit_rate_last_10d",
            ]].head(6))

    feature_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(columns=["batter_id", "game_pk"] + BATTER_TRAILING_FEATURE_COLUMNS)
    print(f"rows with at least 1 prior game: {rows_with_prior_1:,}")
    print(f"rows with at least 2 prior games: {rows_with_prior_2:,}")
    print("valid non-null counts before merge-back:")
    for column, count in count_non_nulls(feature_df, BATTER_TRAILING_FEATURE_COLUMNS).items():
        print(f"  {column}: {count:,}")
    for idx, sample in enumerate(sample_frames, start=1):
        print(f"example batter sample {idx}:")
        print(sample.to_string(index=False))
    print("==========================================\n")
    assert_trailing_feature_table_has_signal(feature_df, BATTER_TRAILING_FEATURE_COLUMNS, "batter")
    return feature_df


def compute_pitcher_trailing_features(pitcher_game_df: pd.DataFrame) -> pd.DataFrame:
    print("=== PITCHER TRAILING FEATURE DIAGNOSTICS ===")
    print(f"input row count: {len(pitcher_game_df):,}")
    print(f"distinct pitcher count: {pitcher_game_df['pitcher_id'].nunique(dropna=True):,}")
    print(f"date range: {pitcher_game_df['game_date'].min()} -> {pitcher_game_df['game_date'].max()}")
    print(f"input grain one row per pitcher-game: {not pitcher_game_df.duplicated(['pitcher_id', 'game_pk']).any()}")
    print("grouping keys: ['pitcher_id']")
    print("sorting keys: ['pitcher_id', 'game_date', 'game_pk']")

    feature_frames: list[pd.DataFrame] = []
    rows_with_prior_history = 0
    sample_frames: list[pd.DataFrame] = []
    for sample_idx, (pitcher_id, group) in enumerate(pitcher_game_df.groupby("pitcher_id", sort=False), start=1):
        grp = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
        rows_with_prior_history += int((np.arange(len(grp)) >= 1).sum())

        rolled = grp.set_index("game_date")
        hr_last_30 = rolled["hr_allowed"].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        pa_last_30 = rolled["pa_against"].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        bbe_last_30 = rolled["bbe_allowed"].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        barrels_last_30 = rolled["barrels_allowed"].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        hard_hit_last_30 = rolled["hard_hit_bbe_allowed"].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        ev95_last_30 = rolled["ev_95plus_bbe_allowed"].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        ev_num_last_30 = (rolled["avg_ev_allowed"].fillna(0) * rolled["bbe_allowed"].fillna(0)).rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        grp["pitcher_hr_allowed_last_30d"] = hr_last_30.to_numpy()
        grp["pitcher_bbe_allowed_last_30d"] = bbe_last_30.to_numpy()
        grp["pitcher_barrels_allowed_last_30d"] = barrels_last_30.to_numpy()
        grp["pitcher_hard_hit_allowed_last_30d"] = hard_hit_last_30.to_numpy()
        grp["pitcher_hr_allowed_per_pa_last_30d"] = safe_rate(hr_last_30, pa_last_30).to_numpy()
        grp["pitcher_barrels_allowed_per_bbe_last_30d"] = safe_rate(barrels_last_30, bbe_last_30).to_numpy()
        grp["pitcher_hard_hit_allowed_rate_last_30d"] = safe_rate(hard_hit_last_30, bbe_last_30).to_numpy()
        grp["pitcher_avg_ev_allowed_last_30d"] = safe_rate(ev_num_last_30, bbe_last_30).to_numpy()
        grp["pitcher_95plus_ev_allowed_rate_last_30d"] = safe_rate(ev95_last_30, bbe_last_30).to_numpy()
        feature_frames.append(grp[["pitcher_id", "game_pk"] + PITCHER_TRAILING_FEATURE_COLUMNS])

        if sample_idx <= 3:
            sample_frames.append(grp[[
                "pitcher_id",
                "game_pk",
                "game_date",
                "pa_against",
                "hr_allowed",
                "barrels_allowed",
                "hard_hit_bbe_allowed",
                "pitcher_hr_allowed_per_pa_last_30d",
                "pitcher_barrels_allowed_per_bbe_last_30d",
                "pitcher_hard_hit_allowed_rate_last_30d",
            ]].head(6))

    feature_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(columns=["pitcher_id", "game_pk"] + PITCHER_TRAILING_FEATURE_COLUMNS)
    print(f"rows with prior history: {rows_with_prior_history:,}")
    print("valid non-null counts before merge-back:")
    for column, count in count_non_nulls(feature_df, PITCHER_TRAILING_FEATURE_COLUMNS).items():
        print(f"  {column}: {count:,}")
    for idx, sample in enumerate(sample_frames, start=1):
        print(f"example pitcher sample {idx}:")
        print(sample.to_string(index=False))
    print("===========================================\n")
    assert_trailing_feature_table_has_signal(feature_df, PITCHER_TRAILING_FEATURE_COLUMNS, "pitcher")
    return feature_df


def shifted_cumulative_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return safe_rate(numerator.cumsum().shift(1), denominator.cumsum().shift(1))


def rolling_day_sum(grp: pd.DataFrame, value_col: str, window: str) -> pd.Series:
    return grp.set_index("game_date")[value_col].rolling(window=window, closed="left", min_periods=1).sum().reset_index(drop=True)


def rolling_day_rate(grp: pd.DataFrame, numerator_col: str, denominator_col: str, window: str) -> pd.Series:
    numerator = rolling_day_sum(grp, numerator_col, window)
    denominator = rolling_day_sum(grp, denominator_col, window)
    return safe_rate(numerator, denominator)


def expected_pa_fallback(grp: pd.DataFrame) -> pd.Series:
    prior_batting_order = grp["batting_order"].shift(1)
    current_batting_order = grp["batting_order"]
    order = prior_batting_order.where(prior_batting_order.notna(), current_batting_order)
    proxy = 4.65 - 0.08 * (order.fillna(9) - 1)
    proxy = proxy.clip(lower=3.6, upper=4.8)
    proxy = proxy.where(order.notna(), np.nan)
    return proxy.astype(float)


def count_window_features(
    grp: pd.DataFrame,
    *,
    count_col: str,
    numerators: dict[str, str],
    weighted_means: dict[str, str],
    window_size: int,
) -> pd.DataFrame:
    counts = pd.to_numeric(grp[count_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    numerator_values = {
        feature_name: pd.to_numeric(grp[source_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for feature_name, source_col in numerators.items()
    }
    weighted_mean_values = {
        feature_name: pd.to_numeric(grp[source_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for feature_name, source_col in weighted_means.items()
    }

    results = {feature_name: np.full(len(grp), np.nan, dtype=float) for feature_name in [*numerators.keys(), *weighted_means.keys()]}
    count_history = np.full(len(grp), np.nan, dtype=float)

    for row_idx in range(len(grp)):
        remaining = float(window_size)
        denominator = 0.0
        numerator_sums = {feature_name: 0.0 for feature_name in numerators}
        weighted_sums = {feature_name: 0.0 for feature_name in weighted_means}
        history_idx = row_idx - 1

        while history_idx >= 0 and remaining > 0:
            available = counts[history_idx]
            if available > 0:
                take = min(available, remaining)
                share = take / available
                denominator += take
                remaining -= take
                for feature_name, source_values in numerator_values.items():
                    numerator_sums[feature_name] += source_values[history_idx] * share
                for feature_name, source_values in weighted_mean_values.items():
                    weighted_sums[feature_name] += source_values[history_idx] * share
            history_idx -= 1

        count_history[row_idx] = denominator if denominator > 0 else np.nan
        if denominator > 0:
            for feature_name in numerators:
                results[feature_name][row_idx] = numerator_sums[feature_name] / denominator
            for feature_name in weighted_means:
                results[feature_name][row_idx] = weighted_sums[feature_name] / denominator

    results[count_col.replace("count", f"count_last_{window_size}")] = count_history
    return pd.DataFrame(results, index=grp.index)


def compute_stable_batter_features(batter_game_df: pd.DataFrame) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    for batter_id, group in batter_game_df.groupby("batter_id", sort=False):
        grp = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
        if grp[["game_date", "game_pk"]].duplicated().any():
            raise ValueError(f"Stable batter features require unique batter_id/game_pk rows; found duplicates for batter {batter_id}.")

        grp["avg_exit_velocity_num"] = grp["avg_exit_velocity"].fillna(0.0) * grp["bbe_count"].fillna(0.0)
        grp["avg_launch_angle_num"] = grp["avg_launch_angle"].fillna(0.0) * grp["bbe_count"].fillna(0.0)
        grp["hr_rate_season_to_date"] = shifted_cumulative_rate(grp["hr_count"], grp["pa_count"])
        grp["hr_per_pa_last_30d"] = rolling_day_rate(grp, numerator_col="hr_count", denominator_col="pa_count", window="30D")
        grp["batter_k_rate_season_to_date"] = shifted_cumulative_rate(grp["batter_k_count"], grp["pa_count"])
        grp["batter_bb_rate_season_to_date"] = shifted_cumulative_rate(grp["batter_bb_count"], grp["pa_count"])
        grp["recent_form_hr_last_7d"] = rolling_day_sum(grp, value_col="hr_count", window="7D")
        grp["recent_form_barrels_last_14d"] = rolling_day_sum(grp, value_col="barrel_count", window="14D")
        expected_pa_raw = rolling_day_sum(grp, value_col="pa_count", window="14D")
        grp["expected_pa_proxy"] = expected_pa_raw.where(expected_pa_raw.notna(), expected_pa_fallback(grp))
        grp["days_since_last_game"] = grp["game_date"].diff().dt.days.astype(float)

        last_50_bbe = count_window_features(
            grp,
            count_col="bbe_count",
            numerators={
                "barrel_rate_last_50_bbe": "barrel_count",
                "hard_hit_rate_last_50_bbe": "hard_hit_bbe_count",
                "fly_ball_rate_last_50_bbe": "fly_ball_bbe_count",
                "pull_air_rate_last_50_bbe": "pull_air_bbe_count",
            },
            weighted_means={
                "avg_exit_velocity_last_50_bbe": "avg_exit_velocity_num",
                "avg_launch_angle_last_50_bbe": "avg_launch_angle_num",
            },
            window_size=50,
        ).rename(columns={"bbe_count_last_50": "bbe_count_last_50"})
        grp = pd.concat([grp, last_50_bbe], axis=1)
        if "bbe_count_last_50" not in grp.columns:
            grp["bbe_count_last_50"] = np.nan

        feature_frames.append(
            grp[
                [
                    "batter_id",
                    "game_pk",
                    "hr_rate_season_to_date",
                    "hr_per_pa_last_30d",
                    "barrel_rate_last_50_bbe",
                    "hard_hit_rate_last_50_bbe",
                    "avg_launch_angle_last_50_bbe",
                    "avg_exit_velocity_last_50_bbe",
                    "fly_ball_rate_last_50_bbe",
                    "pull_air_rate_last_50_bbe",
                    "batter_k_rate_season_to_date",
                    "batter_bb_rate_season_to_date",
                    "recent_form_hr_last_7d",
                    "recent_form_barrels_last_14d",
                    "expected_pa_proxy",
                    "days_since_last_game",
                    "bbe_count_last_50",
                ]
            ]
        )
    return pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(
        columns=[
            "batter_id",
            "game_pk",
            "hr_rate_season_to_date",
            "hr_per_pa_last_30d",
            "barrel_rate_last_50_bbe",
            "hard_hit_rate_last_50_bbe",
            "avg_launch_angle_last_50_bbe",
            "avg_exit_velocity_last_50_bbe",
            "fly_ball_rate_last_50_bbe",
            "pull_air_rate_last_50_bbe",
            "batter_k_rate_season_to_date",
            "batter_bb_rate_season_to_date",
            "recent_form_hr_last_7d",
            "recent_form_barrels_last_14d",
            "expected_pa_proxy",
            "days_since_last_game",
            "bbe_count_last_50",
        ]
    )


def compute_stable_pitcher_features(pitcher_game_df: pd.DataFrame) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    for pitcher_id, group in pitcher_game_df.groupby("pitcher_id", sort=False):
        grp = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
        grp["pitcher_hr9_season_to_date"] = np.where(
            grp["innings_pitched_est"].cumsum().shift(1) > 0,
            grp["hr_allowed"].cumsum().shift(1) * 9.0 / grp["innings_pitched_est"].cumsum().shift(1),
            np.nan,
        )
        grp["pitcher_k_rate_season_to_date"] = shifted_cumulative_rate(grp["pitcher_k_count"], grp["batters_faced"])
        grp["pitcher_bb_rate_season_to_date"] = shifted_cumulative_rate(grp["pitcher_bb_count"], grp["batters_faced"])
        last_50_bbe = count_window_features(
            grp,
            count_col="bbe_allowed",
            numerators={
                "pitcher_barrel_rate_allowed_last_50_bbe": "barrels_allowed",
                "pitcher_hard_hit_rate_allowed_last_50_bbe": "hard_hit_bbe_allowed",
                "pitcher_fb_rate_allowed_last_50_bbe": "fb_allowed",
            },
            weighted_means={},
            window_size=50,
        )
        grp = pd.concat([grp, last_50_bbe], axis=1)
        feature_frames.append(
            grp[
                [
                    "pitcher_id",
                    "game_pk",
                    "pitcher_hr9_season_to_date",
                    "pitcher_barrel_rate_allowed_last_50_bbe",
                    "pitcher_hard_hit_rate_allowed_last_50_bbe",
                    "pitcher_fb_rate_allowed_last_50_bbe",
                    "pitcher_k_rate_season_to_date",
                    "pitcher_bb_rate_season_to_date",
                ]
            ]
        )
    return pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(
        columns=[
            "pitcher_id",
            "game_pk",
            "pitcher_hr9_season_to_date",
            "pitcher_barrel_rate_allowed_last_50_bbe",
            "pitcher_hard_hit_rate_allowed_last_50_bbe",
            "pitcher_fb_rate_allowed_last_50_bbe",
            "pitcher_k_rate_season_to_date",
            "pitcher_bb_rate_season_to_date",
        ]
    )


def validate_required_raw_fields(statcast_df: pd.DataFrame | None) -> dict[str, bool]:
    required = {
        "park_mapping_fields": {"game_pk", "game_date", "home_team", "away_team"},
        "handedness_fields": {"stand", "p_throws", "batter", "pitcher"},
        "pitch_type_fields": {"pitch_type", "description", "batter", "pitcher"},
        "key_fields": {"game_pk", "game_date", "batter", "pitcher"},
    }
    status: dict[str, bool] = {}
    if statcast_df is None:
        print("WARNING: Raw Statcast dataframe not supplied; skipping raw-field validation checks.")
        for label in required:
            status[label] = False
        return status
    columns = set(statcast_df.columns)
    print("\n=== RAW FIELD SUPPORT CHECKS ===")
    for label, fields in required.items():
        missing = sorted(fields - columns)
        supported = len(missing) == 0
        status[label] = supported
        if supported:
            print(f"{label}: supported")
        else:
            print(f"{label}: missing -> {missing}")
    print("================================\n")
    return status


def _mlb_team_to_bref_code(team: object) -> str | None:
    if pd.isna(team):
        return None
    team = str(team).upper()
    mapping = {"AZ": "ARI", "CWS": "CHW", "KC": "KCR", "SD": "SDP", "SF": "SFG", "TB": "TBR", "WSH": "WSN", "OAK": "ATH"}
    return mapping.get(team, team)


def normalize_venue_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    return text or None


def normalize_team_venue_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isnumeric():
        text = str(int(float(text)))
    normalized = re.sub(rf"[{re.escape(string.punctuation)}\s]", "", text.upper())
    normalized = TEAM_VENUE_KEY_ALIASES.get(normalized, normalized)
    normalized = _mlb_team_to_bref_code(normalized)
    return normalized


def audit_pybaseball_park_factor_support() -> dict[str, object]:
    keywords = ["park", "stadium", "venue", "factor"]
    hits: list[str] = []
    modules_checked = 0
    for module_info in pkgutil.walk_packages(pybaseball.__path__, prefix="pybaseball."):
        modules_checked += 1
        module_name = module_info.name.lower()
        if any(keyword in module_name for keyword in keywords):
            hits.append(module_info.name)
    usable = len(hits) > 0
    print("=== PYBASEBALL PARK-FACTOR AUDIT ===")
    print(f"modules_scanned: {modules_checked}")
    if usable:
        print(f"candidate_modules_with_park_factor_terms: {hits[:10]}")
    else:
        print("No direct pybaseball park-factor source found; using fallback park-factor source")
    print("====================================\n")
    return {"usable": usable, "hits": hits}


def load_local_park_factor_lookup(path: Path = PARK_FACTOR_LOOKUP_PATH) -> pd.DataFrame | None:
    """Load maintainable local park-factor table (100 = league average) if available."""
    if not path.exists():
        print(f"WARNING: local park-factor file not found at {path}; trying empirical fallback.")
        return None

    lookup = pd.read_csv(path)
    required = {"venue_key", "park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb"}
    missing = sorted(required - set(lookup.columns))
    if missing:
        print(f"WARNING: local park-factor file missing required columns {missing}; trying empirical fallback.")
        return None

    lookup = lookup.rename(columns={"venue_key": "park_team_code"})
    lookup["park_mapping_key"] = lookup["park_team_code"].map(normalize_team_venue_key)
    for feature in ["park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb"]:
        lookup[feature] = pd.to_numeric(lookup[feature], errors="coerce")
    return lookup[["park_team_code", "park_mapping_key", "park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb"]].drop_duplicates(
        subset=["park_mapping_key"]
    )


def derive_empirical_park_factors_from_statcast(statcast_df: pd.DataFrame) -> pd.DataFrame:
    pa_df = extract_plate_appearances(statcast_df.copy())
    pa_df["home_team_for_park"] = pa_df["home_team"]
    pa_df["bbe"] = pa_df["bbe_count"].fillna(0)
    pa_df["hr"] = pa_df["hr_count"].fillna(0)
    league_hr_rate = safe_scalar_rate(pa_df["hr"].sum(), pa_df["bbe"].sum())
    if pd.isna(league_hr_rate) or league_hr_rate <= 0:
        raise ValueError("Unable to derive empirical park factors because league HR-on-contact rate is unavailable.")

    park = pa_df.groupby("home_team_for_park", dropna=False).agg(
        hr=("hr", "sum"),
        bbe=("bbe", "sum"),
        hr_lhb=("hr", lambda s: pa_df.loc[s.index, "hr"].where(pa_df.loc[s.index, "stand"].eq("L"), 0).sum()),
        bbe_lhb=("bbe", lambda s: pa_df.loc[s.index, "bbe"].where(pa_df.loc[s.index, "stand"].eq("L"), 0).sum()),
        hr_rhb=("hr", lambda s: pa_df.loc[s.index, "hr"].where(pa_df.loc[s.index, "stand"].eq("R"), 0).sum()),
        bbe_rhb=("bbe", lambda s: pa_df.loc[s.index, "bbe"].where(pa_df.loc[s.index, "stand"].eq("R"), 0).sum()),
    ).reset_index()
    prior_weight = 50.0
    park["park_factor_hr"] = (
        ((park["hr"] + prior_weight * league_hr_rate) / (park["bbe"] + prior_weight)) / league_hr_rate
    ) * 100.0
    park["park_factor_hr_vs_lhb"] = (
        ((park["hr_lhb"] + prior_weight * league_hr_rate) / (park["bbe_lhb"] + prior_weight)) / league_hr_rate
    ) * 100.0
    park["park_factor_hr_vs_rhb"] = (
        ((park["hr_rhb"] + prior_weight * league_hr_rate) / (park["bbe_rhb"] + prior_weight)) / league_hr_rate
    ) * 100.0
    park = park.rename(columns={"home_team_for_park": "park_team_code"})
    park["park_mapping_key"] = park["park_team_code"].map(normalize_team_venue_key)
    return park[["park_team_code", "park_mapping_key", "park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb"]]


def add_park_factor_features(model_df: pd.DataFrame, statcast_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, object]]:
    df = model_df.copy()
    df = df.drop(columns=["park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb"], errors="ignore")
    df["home_team_for_park"] = np.where(df["is_home"].astype(bool), df["team"], df["opponent"])
    df["park_source"] = np.where(df["ballpark"].notna(), "ballpark_name", "home_team_fallback")
    df["park_team_code"] = df["home_team_for_park"].map(_mlb_team_to_bref_code)
    df["park_mapping_key"] = df["park_team_code"].map(normalize_team_venue_key)
    df["normalized_ballpark_key"] = df["ballpark"].map(normalize_venue_key) if "ballpark" in df.columns else np.nan
    print("=== PARK-FACTOR SOURCE AUDIT ===")
    print(f"local_park_factor_lookup_exists: {PARK_FACTOR_LOOKUP_PATH.exists()} ({PARK_FACTOR_LOOKUP_PATH})")
    venue_fields = [col for col in ["venue_id", "venue_name", "home_team", "away_team", "ballpark", "game_pk", "game_date"] if col in df.columns]
    print(f"available_venue_game_fields_in_batter_game_table: {venue_fields}")
    print("park_factor canonical mapping key: park_mapping_key (normalized home-team venue code)")
    print("================================\n")

    pybaseball_audit = audit_pybaseball_park_factor_support()
    if pybaseball_audit["usable"]:
        print("WARNING: pybaseball park-factor candidates were found but no stable documented API was identified; continuing with fallback.")
    park_table = load_local_park_factor_lookup()
    source = "local_lookup_csv" if park_table is not None else ""
    if park_table is None:
        if statcast_df is not None:
            park_table = derive_empirical_park_factors_from_statcast(statcast_df)
            source = "empirical_statcast_fallback"
        else:
            raise ValueError("No local park-factor lookup file and no Statcast dataframe available for empirical fallback.")
    print(f"park_lookup_key_count: {park_table['park_mapping_key'].nunique(dropna=True):,}")
    print(f"park_lookup_key_sample: {park_table['park_mapping_key'].dropna().astype(str).head(10).tolist()}")
    print(f"park-factor source row count: {len(park_table):,}")
    print(f"available key columns in park-factor table: {[c for c in park_table.columns if 'park' in c or 'key' in c or 'venue' in c]}")
    print(f"available venue/game/team columns in batter-game table: {[c for c in df.columns if c in {'game_pk','game_date','team','opponent','home_team_for_park','park_team_code','park_mapping_key','ballpark','normalized_ballpark_key'}]}")
    print("exact merge keys chosen: left=['park_mapping_key'] right=['park_mapping_key']")
    print(f"merge key dtype left park_mapping_key={df['park_mapping_key'].dtype} | right park_mapping_key={park_table['park_mapping_key'].dtype}")
    print(f"unique merge keys left={df['park_mapping_key'].nunique(dropna=True):,} | right={park_table['park_mapping_key'].nunique(dropna=True):,}")
    print(f"sample left keys (20): {df['park_mapping_key'].dropna().astype(str).drop_duplicates().head(20).tolist()}")
    print(f"sample right keys (20): {park_table['park_mapping_key'].dropna().astype(str).drop_duplicates().head(20).tolist()}")
    df = merge_with_diagnostics(
        df,
        park_table,
        on=["park_mapping_key"],
        how="left",
        step_name="attach park factor lookup",
        validate="many_to_one",
    )
    print("=== PARK FACTOR DIAGNOSTICS ===")
    print(f"park-factor source used: {source}")
    print(f"venue/team fields available in model df: {sorted([col for col in ['game_pk', 'game_date', 'team', 'opponent', 'is_home', 'ballpark'] if col in df.columns])}")
    print(f"park identified via ballpark field rows: {int((df['park_source'] == 'ballpark_name').sum()):,}")
    print(f"park identified via home_team fallback rows: {int((df['park_source'] == 'home_team_fallback').sum()):,}")
    print("venue key used for mapping: park_mapping_key")
    matched = int(df["park_factor_hr"].notna().sum()) if "park_factor_hr" in df.columns else 0
    unmatched = int(df["park_factor_hr"].isna().sum()) if "park_factor_hr" in df.columns else len(df)
    print(f"matched park factor rows: {matched:,} / {len(df):,}")
    print(f"unmatched park factor rows: {unmatched:,} / {len(df):,} ({(unmatched / len(df) * 100 if len(df) else 0):.2f}%)")
    unmatched_keys = df.loc[df["park_factor_hr"].isna(), "park_mapping_key"].dropna().astype(str).unique().tolist()
    matched_keys = df.loc[df["park_factor_hr"].notna(), "park_mapping_key"].dropna().astype(str).unique().tolist()
    print(f"sample unmatched venue keys (20): {unmatched_keys[:20]}")
    print(f"sample matched venue keys (20): {matched_keys[:20]}")
    for feature in ["park_factor_hr", "park_factor_hr_vs_lhb", "park_factor_hr_vs_rhb"]:
        non_null = int(df[feature].notna().sum()) if feature in df.columns else 0
        miss_pct = float(df[feature].isna().mean() * 100) if feature in df.columns else 100.0
        print(f"{feature}: non_null={non_null:,}, missing_pct={miss_pct:.2f}%")
    df["park_factor_hr_vs_batter_hand"] = np.where(df["bat_side"].eq("L"), df["park_factor_hr_vs_lhb"], df["park_factor_hr_vs_rhb"])
    non_null = int(df["park_factor_hr_vs_batter_hand"].notna().sum())
    miss_pct = float(df["park_factor_hr_vs_batter_hand"].isna().mean() * 100) if len(df) else 100.0
    print(f"park_factor_hr_vs_batter_hand: non_null={non_null:,}, missing_pct={miss_pct:.2f}%")
    if df["park_factor_hr"].notna().sum() == 0:
        print('WARNING: park_factor_hr excluded because venue mapping produced zero usable values')
        raise ValueError("park_factor_hr is 100% missing after park-factor mapping; excluded because venue mapping produced zero usable values.")
    else:
        status = {
            "included": True,
            "reason": f"park factors populated using {source}",
            "source": source,
            "merge_keys": "park_mapping_key <- park_mapping_key",
            "matched_rows": matched,
            "unmatched_rows": unmatched,
            "pybaseball_audit": pybaseball_audit,
        }
    print("===============================\n")
    return df.drop(columns=["home_team_for_park", "park_team_code", "park_mapping_key", "park_source", "normalized_ballpark_key"], errors="ignore"), status


def _split_causal_rate(
    grp: pd.DataFrame,
    split_col: str,
    split_value: str,
    numerator_col: str,
    denominator_col: str,
) -> pd.Series:
    mask = grp[split_col].eq(split_value).astype(float)
    numerator = grp[numerator_col].fillna(0.0) * mask
    denominator = grp[denominator_col].fillna(0.0) * mask
    return numerator.cumsum().shift(1) / denominator.cumsum().shift(1).replace({0.0: np.nan})


def compute_batter_handedness_split_features(batter_game_df: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for _, group in batter_game_df.groupby("batter_id", sort=False):
        grp = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
        for hand, suffix in [("R", "rhp"), ("L", "lhp")]:
            grp[f"batter_hr_per_pa_vs_{suffix}"] = _split_causal_rate(grp, "pitcher_hand", hand, "hr_count", "pa_count")
            grp[f"batter_barrels_per_pa_vs_{suffix}"] = _split_causal_rate(grp, "pitcher_hand", hand, "barrel_count", "pa_count")
            grp[f"batter_hard_hit_rate_vs_{suffix}"] = _split_causal_rate(grp, "pitcher_hand", hand, "hard_hit_bbe_count", "bbe_count")
            grp[f"batter_95plus_ev_rate_vs_{suffix}"] = _split_causal_rate(grp, "pitcher_hand", hand, "ev_95plus_bbe_count", "bbe_count")
        records.append(grp[[
            "batter_id", "game_pk", "batter_hr_per_pa_vs_rhp", "batter_hr_per_pa_vs_lhp", "batter_barrels_per_pa_vs_rhp",
            "batter_barrels_per_pa_vs_lhp", "batter_hard_hit_rate_vs_rhp", "batter_hard_hit_rate_vs_lhp",
            "batter_95plus_ev_rate_vs_rhp", "batter_95plus_ev_rate_vs_lhp",
        ]])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["batter_id", "game_pk"])


def compute_pitcher_handedness_split_features(pitcher_game_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "pa_against_rhb", "pa_against_lhb", "hr_allowed_rhb", "hr_allowed_lhb", "bbe_allowed_rhb", "bbe_allowed_lhb",
        "barrels_allowed_rhb", "barrels_allowed_lhb", "hard_hit_bbe_allowed_rhb", "hard_hit_bbe_allowed_lhb",
        "ev_95plus_bbe_allowed_rhb", "ev_95plus_bbe_allowed_lhb",
    ]
    if not all(col in pitcher_game_df.columns for col in required):
        print("WARNING: Skipping pitcher handedness split features; aggregate pitcher splits were unavailable.")
        return pd.DataFrame(columns=["pitcher_id", "game_pk"])

    records: list[pd.DataFrame] = []
    for _, group in pitcher_game_df.groupby("pitcher_id", sort=False):
        grp = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
        prior = grp[required].fillna(0.0).cumsum().shift(1)
        grp["pitcher_hr_allowed_per_pa_vs_rhb"] = safe_rate(prior["hr_allowed_rhb"], prior["pa_against_rhb"])
        grp["pitcher_hr_allowed_per_pa_vs_lhb"] = safe_rate(prior["hr_allowed_lhb"], prior["pa_against_lhb"])
        grp["pitcher_barrels_allowed_per_bbe_vs_rhb"] = safe_rate(prior["barrels_allowed_rhb"], prior["bbe_allowed_rhb"])
        grp["pitcher_barrels_allowed_per_bbe_vs_lhb"] = safe_rate(prior["barrels_allowed_lhb"], prior["bbe_allowed_lhb"])
        grp["pitcher_hard_hit_allowed_rate_vs_rhb"] = safe_rate(prior["hard_hit_bbe_allowed_rhb"], prior["bbe_allowed_rhb"])
        grp["pitcher_hard_hit_allowed_rate_vs_lhb"] = safe_rate(prior["hard_hit_bbe_allowed_lhb"], prior["bbe_allowed_lhb"])
        grp["pitcher_95plus_ev_allowed_rate_vs_rhb"] = safe_rate(prior["ev_95plus_bbe_allowed_rhb"], prior["bbe_allowed_rhb"])
        grp["pitcher_95plus_ev_allowed_rate_vs_lhb"] = safe_rate(prior["ev_95plus_bbe_allowed_lhb"], prior["bbe_allowed_lhb"])
        records.append(grp[[
            "pitcher_id", "game_pk", "pitcher_hr_allowed_per_pa_vs_rhb", "pitcher_hr_allowed_per_pa_vs_lhb",
            "pitcher_barrels_allowed_per_bbe_vs_rhb", "pitcher_barrels_allowed_per_bbe_vs_lhb",
            "pitcher_hard_hit_allowed_rate_vs_rhb", "pitcher_hard_hit_allowed_rate_vs_lhb",
            "pitcher_95plus_ev_allowed_rate_vs_rhb", "pitcher_95plus_ev_allowed_rate_vs_lhb",
        ]])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["pitcher_id", "game_pk"])


def build_matchup_selected_handedness_features(df: pd.DataFrame) -> pd.DataFrame:
    dataset = df.copy()
    dataset["batter_hr_per_pa_vs_pitcher_hand"] = np.where(dataset["pitcher_hand"].eq("L"), dataset.get("batter_hr_per_pa_vs_lhp"), dataset.get("batter_hr_per_pa_vs_rhp"))
    dataset["batter_barrels_per_pa_vs_pitcher_hand"] = np.where(dataset["pitcher_hand"].eq("L"), dataset.get("batter_barrels_per_pa_vs_lhp"), dataset.get("batter_barrels_per_pa_vs_rhp"))
    dataset["batter_hard_hit_rate_vs_pitcher_hand"] = np.where(dataset["pitcher_hand"].eq("L"), dataset.get("batter_hard_hit_rate_vs_lhp"), dataset.get("batter_hard_hit_rate_vs_rhp"))
    dataset["batter_95plus_ev_rate_vs_pitcher_hand"] = np.where(dataset["pitcher_hand"].eq("L"), dataset.get("batter_95plus_ev_rate_vs_lhp"), dataset.get("batter_95plus_ev_rate_vs_rhp"))

    dataset["pitcher_hr_allowed_per_pa_vs_batter_hand"] = np.where(dataset["bat_side"].eq("L"), dataset.get("pitcher_hr_allowed_per_pa_vs_lhb"), dataset.get("pitcher_hr_allowed_per_pa_vs_rhb"))
    dataset["pitcher_barrels_allowed_per_bbe_vs_batter_hand"] = np.where(dataset["bat_side"].eq("L"), dataset.get("pitcher_barrels_allowed_per_bbe_vs_lhb"), dataset.get("pitcher_barrels_allowed_per_bbe_vs_rhb"))
    dataset["pitcher_hard_hit_allowed_rate_vs_batter_hand"] = np.where(dataset["bat_side"].eq("L"), dataset.get("pitcher_hard_hit_allowed_rate_vs_lhb"), dataset.get("pitcher_hard_hit_allowed_rate_vs_rhb"))
    dataset["pitcher_95plus_ev_allowed_rate_vs_batter_hand"] = np.where(dataset["bat_side"].eq("L"), dataset.get("pitcher_95plus_ev_allowed_rate_vs_lhb"), dataset.get("pitcher_95plus_ev_allowed_rate_vs_rhb"))

    dataset["split_matchup_hr"] = dataset["batter_hr_per_pa_vs_pitcher_hand"] * dataset["pitcher_hr_allowed_per_pa_vs_batter_hand"]
    dataset["split_matchup_barrel"] = dataset["batter_barrels_per_pa_vs_pitcher_hand"] * dataset["pitcher_barrels_allowed_per_bbe_vs_batter_hand"]
    dataset["split_matchup_hard_hit"] = dataset["batter_hard_hit_rate_vs_pitcher_hand"] * dataset["pitcher_hard_hit_allowed_rate_vs_batter_hand"]
    return dataset


def compute_expanded_pitch_mix_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    pitch_df = raw_df[["game_pk", "game_date", "pitcher", "pitch_type"]].copy()
    pitch_df["game_date"] = pd.to_datetime(pitch_df["game_date"], errors="coerce")
    pitch_df = pitch_df.dropna(subset=["game_pk", "pitcher", "pitch_type"])
    for name, pitch_types in GRANULAR_PITCH_TYPES.items():
        pitch_df[f"is_{name}"] = pitch_df["pitch_type"].isin(pitch_types).astype(int)
    pitch_df["is_fastball"] = pitch_df["pitch_type"].isin(FASTBALL_TYPES).astype(int)
    pitch_df["is_breaking_ball"] = pitch_df["pitch_type"].isin(BREAKING_BALL_TYPES).astype(int)
    pitch_df["is_offspeed"] = pitch_df["pitch_type"].isin(OFFSPEED_TYPES).astype(int)
    pitcher_game = pitch_df.groupby(["pitcher", "game_date", "game_pk"], dropna=False).agg(
        total_pitches=("pitch_type", "size"),
        fastball=("is_fastball", "sum"),
        breaking_ball=("is_breaking_ball", "sum"),
        offspeed=("is_offspeed", "sum"),
        four_seam=("is_four_seam", "sum"),
        sinker=("is_sinker", "sum"),
        cutter=("is_cutter", "sum"),
        slider=("is_slider", "sum"),
        curveball=("is_curveball", "sum"),
        changeup=("is_changeup", "sum"),
    ).reset_index().rename(columns={"pitcher": "pitcher_id"}).sort_values(["pitcher_id", "game_date", "game_pk"])
    out_frames: list[pd.DataFrame] = []
    for _, group in pitcher_game.groupby("pitcher_id", sort=False):
        grp = group.copy()
        rolled = grp.set_index("game_date")
        total_prior = rolled["total_pitches"].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
        for col in ["fastball", "breaking_ball", "offspeed", "four_seam", "sinker", "cutter", "slider", "curveball", "changeup"]:
            prior = rolled[col].rolling("30D", closed="left", min_periods=1).sum().reset_index(drop=True)
            grp[f"pitcher_{col}_pct"] = safe_rate(prior, total_prior).to_numpy()
        out_frames.append(grp[[
            "pitcher_id", "game_pk", "pitcher_fastball_pct", "pitcher_breaking_ball_pct", "pitcher_offspeed_pct",
            "pitcher_four_seam_pct", "pitcher_sinker_pct", "pitcher_cutter_pct", "pitcher_slider_pct", "pitcher_curveball_pct", "pitcher_changeup_pct",
        ]])
    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(columns=["pitcher_id", "game_pk"])


def compute_batter_pitch_type_split_features(pa_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    pa = pa_df.copy()
    pa["granular_pitch_bucket"] = pa["pitch_type"].apply(classify_granular_pitch_type)
    bucket_rows: list[pd.DataFrame] = []
    specs = [
        ("fastball", "fastballs"),
        ("breaking_ball", "breaking"),
        ("offspeed", "offspeed"),
        ("slider", "slider"),
        ("changeup", "changeup"),
        ("curveball", "curveball"),
    ]
    for bucket_value, bucket_label in specs:
        sub = pa[pa["pitch_type_bucket"].eq(bucket_value) | pa["granular_pitch_bucket"].eq(bucket_value)].copy()
        if sub.empty:
            continue
        agg = sub.groupby(["batter", "game_date", "game_pk"], dropna=False).agg(
            bbe=("bbe_count", "sum"),
            hard_hit=("hard_hit_bbe_count", "sum"),
            barrel=("barrel_count", "sum"),
            contact=("contact_event_count", "sum"),
            swing=("swing_event_count", "sum"),
        ).reset_index().sort_values(["batter", "game_date", "game_pk"])
        out = []
        for batter_id, group in agg.groupby("batter", sort=False):
            g = group.copy()
            prior = g[["bbe", "hard_hit", "barrel", "contact", "swing"]].cumsum().shift(1)
            g[f"batter_hard_hit_rate_vs_{bucket_label}"] = safe_rate(prior["hard_hit"], prior["bbe"])
            g[f"batter_barrel_rate_vs_{bucket_label}"] = safe_rate(prior["barrel"], prior["bbe"])
            if bucket_label in {"fastballs", "breaking", "offspeed"}:
                g[f"batter_contact_rate_vs_{bucket_label}"] = safe_rate(prior["contact"], prior["swing"])
            out.append(g)
        joined = pd.concat(out, ignore_index=True)
        cols = [c for c in joined.columns if c.startswith("batter_")]
        bucket_rows.append(joined[["batter", "game_pk"] + cols].rename(columns={"batter": "batter_id"}))
    if not bucket_rows:
        return pd.DataFrame(columns=["batter_id", "game_pk"])
    merged = bucket_rows[0]
    for extra in bucket_rows[1:]:
        merged = merge_with_diagnostics(merged, extra, on=["batter_id", "game_pk"], how="outer", step_name="combine batter pitch-type split blocks", validate="one_to_one")
    return merge_with_diagnostics(
        model_df[["batter_id", "game_pk"]],
        merged,
        on=["batter_id", "game_pk"],
        how="left",
        step_name="attach batter causal pitch-type split features",
        validate="one_to_one",
    )


def build_expanded_pitch_matchup_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nan_series = pd.Series(np.nan, index=out.index)
    out["breaking_matchup_hard_hit"] = out.get("pitcher_breaking_ball_pct", nan_series) * out.get("batter_hard_hit_rate_vs_breaking", nan_series)
    out["breaking_matchup_barrel"] = out.get("pitcher_breaking_ball_pct", nan_series) * out.get("batter_barrel_rate_vs_breaking", nan_series)
    out["offspeed_matchup_hard_hit"] = out.get("pitcher_offspeed_pct", nan_series) * out.get("batter_hard_hit_rate_vs_offspeed", nan_series)
    out["offspeed_matchup_barrel"] = out.get("pitcher_offspeed_pct", nan_series) * out.get("batter_barrel_rate_vs_offspeed", nan_series)
    out["slider_matchup_barrel"] = out.get("pitcher_slider_pct", nan_series) * out.get("batter_barrel_rate_vs_slider", nan_series)
    out["changeup_matchup_barrel"] = out.get("pitcher_changeup_pct", nan_series) * out.get("batter_barrel_rate_vs_changeup", nan_series)
    return out


def build_pitch_type_matchup_features(statcast_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = statcast_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"], errors="coerce")
    required_raw = {"pitch_type", "description", "batter", "pitcher", "game_pk", "game_date"}
    if not required_raw.issubset(raw_df.columns):
        warning = f"Skipping pitch-type features because raw Statcast is missing columns: {sorted(required_raw - set(raw_df.columns))}"
        print(f"WARNING: {warning}")
        return pd.DataFrame(), {"included": False, "reason": warning}

    pa_df = extract_plate_appearances(raw_df)
    if pa_df.empty:
        warning = "Skipping pitch-type features because no plate appearances were derivable."
        print(f"WARNING: {warning}")
        return pd.DataFrame(), {"included": False, "reason": warning}

    pitcher_mix = compute_expanded_pitch_mix_features(raw_df)
    batter_splits = compute_batter_pitch_type_split_features(pa_df, model_df)
    features = merge_with_diagnostics(
        model_df[["batter_id", "game_pk", "pitcher_id"]],
        pitcher_mix.drop(columns=["batter_id"], errors="ignore"),
        on=["game_pk", "pitcher_id"],
        how="left",
        step_name="attach expanded pitcher pitch mix onto model rows",
        validate="many_to_one",
    )
    features = merge_with_diagnostics(
        features,
        batter_splits,
        on=["batter_id", "game_pk"],
        how="left",
        step_name="attach batter expanded pitch-type splits onto model rows",
        validate="one_to_one",
    )
    features = build_expanded_pitch_matchup_interactions(features)
    features["fastball_matchup_hard_hit"] = features["pitcher_fastball_pct"] * features["batter_hard_hit_rate_vs_fastballs"]
    features["fastball_matchup_barrel"] = features["pitcher_fastball_pct"] * features["batter_barrel_rate_vs_fastballs"]
    return features.drop(columns=["pitcher_id"]), {"included": True, "reason": "causal pitch-type splits and pitcher mix features were computed"}


def assert_no_duplicate_batter_game_rows(df: pd.DataFrame, context: str) -> None:
    if df.duplicated(["batter_id", "game_pk"]).any():
        offenders = df[df.duplicated(["batter_id", "game_pk"], keep=False)][["batter_id", "game_pk"]].drop_duplicates().head(10)
        raise ValueError(f"Duplicate batter-game rows detected {context}. Offending keys: {offenders.to_dict('records')}")


def merge_with_diagnostics(left_df: pd.DataFrame, right_df: pd.DataFrame, *, step_name: str, tracked_feature_columns: Iterable[str] | None = None, **merge_kwargs) -> pd.DataFrame:
    tracked_feature_columns = list(tracked_feature_columns or [])
    left_keys = merge_kwargs.get("on") or merge_kwargs.get("left_on")
    right_keys = merge_kwargs.get("on") or merge_kwargs.get("right_on")
    if isinstance(left_keys, str):
        left_keys = [left_keys]
    if isinstance(right_keys, str):
        right_keys = [right_keys]
    left_keys = list(left_keys or [])
    right_keys = list(right_keys or [])
    print(f"\n=== MERGE DIAGNOSTICS: {step_name} ===")
    print(f"left row count: {len(left_df):,}")
    print(f"right row count: {len(right_df):,}")
    if left_keys:
        left_dupes = int(left_df.duplicated(left_keys).sum())
        right_dupes = int(right_df.duplicated(right_keys).sum())
        print(f"left duplicate key rows: {left_dupes:,}")
        print(f"right duplicate key rows: {right_dupes:,}")
    matched_probe = left_df.merge(
        right_df[right_keys].drop_duplicates() if right_keys else right_df.copy(),
        how="left",
        left_on=left_keys if merge_kwargs.get("left_on") is not None else None,
        right_on=right_keys if merge_kwargs.get("right_on") is not None else None,
        on=merge_kwargs.get("on"),
        indicator=True,
    )
    matched = int((matched_probe["_merge"] == "both").sum())
    unmatched = int((matched_probe["_merge"] != "both").sum())
    print(f"matched row count: {matched:,}")
    print(f"unmatched row count: {unmatched:,}")
    identity_cols = ["batter_name", "pitcher_name", "batter_id", "pitcher_id"]
    left_identity_presence = {col: col in left_df.columns for col in identity_cols}
    right_identity_presence = {col: col in right_df.columns for col in identity_cols}
    print(f"identity columns before merge (left): {left_identity_presence}")
    print(f"identity columns before merge (right): {right_identity_presence}")
    shared_identity = [col for col in identity_cols if col in left_df.columns and col in right_df.columns]
    if shared_identity:
        print(f"WARNING: right table also contains identity columns that could overwrite/suffix: {shared_identity}")
    merged = left_df.merge(right_df, indicator=True, **merge_kwargs)
    print(f"row count after merge: {len(merged):,}")
    for col in identity_cols:
        if col in merged.columns:
            continue
        left_suffix_col = f"{col}_x"
        right_suffix_col = f"{col}_y"
        if left_suffix_col in merged.columns or right_suffix_col in merged.columns:
            print(f"WARNING: merge produced suffixed identity columns for {col}; preserving left-side hitter/pitcher identity columns is required.")
    for feature in tracked_feature_columns:
        if feature in merged.columns:
            print(f"post-merge non-null {feature}: {int(merged[feature].notna().sum()):,}")
    many_to_many = len(merged) > len(left_df) and merge_kwargs.get("how", "left") in {"left", "inner"}
    if many_to_many:
        print("WARNING: merge increased row count; inspect keys for many-to-many behavior.")
    print("==========================================\n")
    return merged.drop(columns=["_merge"])




def count_non_nulls(df: pd.DataFrame, feature_columns: Iterable[str]) -> dict[str, int]:
    return {column: int(df[column].notna().sum()) for column in feature_columns if column in df.columns}


def assert_trailing_feature_table_has_signal(feature_df: pd.DataFrame, feature_columns: Iterable[str], label: str) -> None:
    if feature_df.empty:
        raise ValueError(f"{label.title()} trailing feature table is empty before merge-back.")
    counts = count_non_nulls(feature_df, feature_columns)
    if counts and max(counts.values()) == 0:
        raise ValueError(f"{label.title()} trailing feature table was computed but every tracked trailing feature is null before merge-back.")


def print_pre_merge_feature_table_diagnostics(
    *,
    table_name: str,
    feature_df: pd.DataFrame,
    feature_columns: Iterable[str],
    key_columns: list[str],
    left_df: pd.DataFrame,
    left_keys: list[str],
) -> None:
    print(f"=== PRE-MERGE DIAGNOSTICS: {table_name} ===")
    print(f"row count: {len(feature_df):,}")
    print(f"duplicate key count: {int(feature_df.duplicated(key_columns).sum()):,}")
    print("non-null counts:")
    for column, count in count_non_nulls(feature_df, feature_columns).items():
        print(f"  {column}: {count:,}")
    print(f"merge keys left: {left_keys}")
    print(f"merge keys right: {key_columns}")
    for left_key, right_key in zip(left_keys, key_columns):
        print(f"  dtype {left_key} (left)={left_df[left_key].dtype} | {right_key} (right)={feature_df[right_key].dtype}")
    print("===========================================\n")


def print_before_after_trailing_report(
    *,
    batter_before_counts: dict[str, int],
    batter_after_counts: dict[str, int],
    pitcher_before_counts: dict[str, int],
    pitcher_after_counts: dict[str, int],
) -> None:
    print("=== TRAILING FEATURE BEFORE/AFTER COUNTS ===")
    print("batter trailing features:")
    for column in BATTER_TRAILING_FEATURE_COLUMNS:
        print(f"  {column}: before={batter_before_counts.get(column, 0):,}, after={batter_after_counts.get(column, 0):,}")
    print("pitcher trailing features:")
    for column in PITCHER_TRAILING_FEATURE_COLUMNS:
        print(f"  {column}: before={pitcher_before_counts.get(column, 0):,}, after={pitcher_after_counts.get(column, 0):,}")
    print("============================================\n")

def finalize_feature_export(df: pd.DataFrame, missingness_threshold: float = 0.95) -> dict[str, ExportDecision]:
    decisions: dict[str, ExportDecision] = {}
    family_thresholds = {
        "park": 0.50,
        "handedness_split": 0.70,
        "pitch_type": 0.70,
    }
    for feature, source_table in FINAL_FEATURE_SPECS.items():
        if feature not in df.columns:
            print(f"WARNING: Excluding {feature} because the source table did not produce it.")
            decisions[feature] = ExportDecision(source_table, False, "feature not produced")
            continue
        missing_pct = float(df[feature].isna().mean())
        if missing_pct >= 1.0:
            print(f"WARNING: Excluding {feature} because it is 100% missing.")
            decisions[feature] = ExportDecision(source_table, False, "100% missing")
            continue
        threshold = family_thresholds.get(source_table, missingness_threshold)
        if missing_pct > threshold:
            print(f"WARNING: Excluding {feature} because missingness {missing_pct:.1%} exceeded threshold {threshold:.1%}.")
            decisions[feature] = ExportDecision(source_table, False, f"missingness {missing_pct:.1%} above {threshold:.1%} threshold")
            continue
        decisions[feature] = ExportDecision(source_table, True, "feature passed coverage checks")
    return decisions


def base_export_columns(df: pd.DataFrame) -> list[str]:
    ordered = [
        "game_pk", "game_date", "batter_id", "player_id", "batter_name", "player_name", "pitcher_id", "opp_pitcher_id",
        "pitcher_name", "opp_pitcher_name", "team", "opponent", "is_home", "bat_side", "pitcher_hand", "pitch_hand_primary",
        "pa_count", "hr_count", "hit_hr", "bbe_count", "barrel_count", "hard_hit_bbe_count", "avg_exit_velocity", "max_exit_velocity",
        "ev_95plus_bbe_count", "fly_ball_bbe_count", "pull_air_bbe_count", "ballpark", "park_factor_hr", "platoon_advantage",
        "starter_or_bullpen_proxy",
    ]
    return [column for column in ordered if column in df.columns]


def print_final_feature_quality_summary(df: pd.DataFrame, decisions: dict[str, ExportDecision]) -> None:
    print("\n=== FINAL FEATURE QUALITY SUMMARY ===")
    print("feature_name | non_null_count | missing_pct | source_table | included_in_export | reason")
    for feature, decision in decisions.items():
        raw = df.loc[:, feature] if feature in df.columns else pd.Series(dtype=float)
        series = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
        non_null = int(series.notna().sum()) if feature in df.columns else 0
        missing_pct = float(series.isna().mean() * 100) if feature in df.columns and len(df) else 100.0
        print(f"{feature} | {non_null:,} | {missing_pct:.2f}% | {decision.source_table} | {'yes' if decision.included_in_export else 'no'} | {decision.reason}")
    print("=====================================\n")


def summarize_new_context_feature_quality(df: pd.DataFrame, decisions: dict[str, ExportDecision]) -> None:
    print("=== NEW CONTEXT FEATURE QUALITY SUMMARY ===")
    print("feature_name | non_null_count | missing_pct | source_family | included_in_export | reason_if_excluded")
    for family, features in NEW_CONTEXT_FEATURE_FAMILIES.items():
        for feature in features:
            if feature not in df.columns and feature not in decisions:
                continue
            raw = df.loc[:, feature] if feature in df.columns else pd.Series(dtype=float)
            series = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
            non_null = int(series.notna().sum()) if feature in df.columns else 0
            missing_pct = float(series.isna().mean() * 100) if feature in df.columns and len(df) else 100.0
            decision = decisions.get(feature)
            included = decision.included_in_export if decision else False
            reason = decision.reason if decision else "feature not produced"
            excluded_reason = "" if included else reason
            print(f"{feature} | {non_null:,} | {missing_pct:.2f}% | {family} | {'yes' if included else 'no'} | {excluded_reason}")
    print("===========================================\n")


def load_model_feature_list() -> set[str]:
    try:
        from train_model import FEATURE_COLUMNS, OPTIONAL_SECONDARY_FEATURES
        return set(FEATURE_COLUMNS) | set(OPTIONAL_SECONDARY_FEATURES)
    except Exception as exc:
        print(f"WARNING: could not import train_model feature lists for audit: {exc}")
        return set()


def print_requested_feature_audit(df: pd.DataFrame, decisions: dict[str, ExportDecision]) -> None:
    model_features = load_model_feature_list()
    print("=== REQUESTED FEATURE AUDIT TABLE ===")
    print("feature_name | family | status | non_null_count | missing_pct | included_in_model | reason_if_missing_or_excluded")
    for family, features in REQUESTED_FEATURE_AUDIT.items():
        for feature in features:
            present = feature in df.columns
            if present:
                raw = df.loc[:, feature]
                series = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
                non_null = int(series.notna().sum())
                missing_pct = float(series.isna().mean() * 100) if len(df) else 100.0
            else:
                non_null = 0
                missing_pct = 100.0
            decision = decisions.get(feature)
            included_in_export = bool(decision.included_in_export) if decision else False
            included_in_model_bool = feature in model_features
            if not present:
                status = "MISSING"
                reason = "feature not produced"
            elif present and not included_in_model_bool:
                status = "PRESENT_BUT_NOT_EXPORTED"
                reason = "present but not in model list"
            elif present and included_in_model_bool and not included_in_export:
                status = "PRESENT_BUT_NOT_EXPORTED"
                reason = decision.reason if decision else "excluded from engineered export"
            else:
                status = "PRESENT_AND_EXPORTED"
                reason = ""
            included_in_model = "yes" if included_in_model_bool else "no"
            print(f"{feature} | {family} | {status} | {non_null:,} | {missing_pct:.2f}% | {included_in_model} | {reason}")
    print("====================================\n")


def print_focused_pass_summary(
    df: pd.DataFrame,
    decisions: dict[str, ExportDecision],
    park_status: dict[str, object],
    pybaseball_audit: dict[str, object] | None,
    batter_identity_status: dict[str, object],
    batter_merge_overwrite_detected: bool,
) -> None:
    model_features = load_model_feature_list()
    requested = [feature for features in REQUESTED_FEATURE_AUDIT.values() for feature in features]
    newly_implemented = [feature for feature in requested if feature in df.columns]
    present_but_omitted = [feature for feature in requested if feature in df.columns and feature not in model_features]
    now_included = [feature for feature in requested if feature in df.columns and feature in model_features]
    skipped = [feature for feature in requested if feature not in df.columns]
    dead_features = [
        feature
        for feature in requested
        if feature in df.columns and feature in decisions and not decisions[feature].included_in_export
    ]

    pybaseball_has_source = bool(pybaseball_audit["usable"]) if pybaseball_audit else False
    park_series = df.loc[:, "park_factor_hr"] if "park_factor_hr" in df.columns else pd.Series(dtype=float)
    if isinstance(park_series, pd.DataFrame):
        park_series = park_series.iloc[:, 0]
    park_populated = bool(park_series.notna().any()) if "park_factor_hr" in df.columns else False

    print("=== FINAL VALIDATION SUMMARY ===")
    print("[A] Park-factor status")
    print(f"pybaseball_park_factor_source_found: {pybaseball_has_source}")
    print(f"park_factor_source_used: {park_status.get('source', park_status.get('reason', 'unknown'))}")
    print(f"park_factor_merge_keys_used: {park_status.get('merge_keys', 'park_mapping_key <- park_mapping_key')}")
    print(f"park_factor_matched_rows: {park_status.get('matched_rows', 'unknown')}")
    print(f"park_factor_unmatched_rows: {park_status.get('unmatched_rows', 'unknown')}")
    print(f"park_factor_hr_non_null_count: {int(park_series.notna().sum()) if 'park_factor_hr' in df.columns else 0}")
    print(f"park_factor_hr_populated: {park_populated}")
    print(f"park_factor_hr_in_model_feature_list: {'park_factor_hr' in model_features}")
    print("\n[B] Batter identity status")
    print(f"batter_id_to_batter_name_stable: {batter_identity_status.get('stable')}")
    print(f"batter_name_overwritten_by_merges: {batter_merge_overwrite_detected}")
    print("ranked_output_candidate_names_confirmed_hitters: evaluated in train_model.validate_ranked_output_identity")
    print("\n[C] Requested feature audit")
    print(f"requested_features_present_in_engineered_dataset: {len(newly_implemented)}")
    print(f"requested_features_present_but_omitted_from_model_list: {present_but_omitted if present_but_omitted else 'none'}")
    if present_but_omitted:
        for feature in present_but_omitted:
            print(f"Feature existed but was omitted from model list; now included: {feature}")
    print(f"requested_features_present_and_in_model_list: {len(now_included)}")
    print(f"requested_features_still_missing: {skipped if skipped else 'none'}")
    print(f"requested_features_excluded_from_export_due_to_quality: {dead_features if dead_features else 'none'}")
    print("================================\n")


def print_rerun_verdict(
    df: pd.DataFrame,
    *,
    pitch_type_status: dict[str, object],
    park_status: dict[str, object],
    hand_status: dict[str, object],
    raw_field_status: dict[str, bool],
) -> None:
    hr_features_ready = all(feature in df.columns and df[feature].notna().any() for feature in ["hr_per_pa_last_30d", "hr_per_pa_last_10d"])
    pitcher_features_ready = all(
        feature in df.columns
        for feature in [
            "pitcher_hr_allowed_per_pa_last_30d",
            "pitcher_barrels_allowed_per_bbe_last_30d",
            "pitcher_hard_hit_allowed_rate_last_30d",
        ]
    ) and all(feature in df.columns and df[feature].notna().any() for feature in [
            "pitcher_hr_allowed_per_pa_last_30d",
            "pitcher_barrels_allowed_per_bbe_last_30d",
            "pitcher_hard_hit_allowed_rate_last_30d",
        ])
    print("=== RERUN VERDICT ===")
    print(f"dataset_valid_for_train_model: {not df.duplicated(['batter_id', 'game_pk']).any()}")
    print(f"park_features_included: {park_status.get('included', False)}")
    print(f"park_features_reason: {park_status.get('reason')}")
    print(f"handedness_features_included: {hand_status.get('included', False)}")
    print(f"handedness_features_reason: {hand_status.get('reason')}")
    print(f"pitch_type_features_included: {pitch_type_status.get('included', False)}")
    print(f"pitch_type_features_reason: {pitch_type_status.get('reason')}")
    print(f"raw_field_support_status: {raw_field_status}")
    print(f"batter_trailing_hr_features_included: {hr_features_ready}")
    print(f"pitcher_recent_contact_features_included: {pitcher_features_ready}")
    print("=====================\n")


def validate_dataset(df: pd.DataFrame) -> list[str]:
    validate_final_model_df(df)
    warnings: list[str] = []
    if "hr_per_pa_last_30d" in df.columns:
        first_game = df.sort_values(["batter_id", "game_date", "game_pk"]).groupby("batter_id").head(1)
        if first_game["hr_per_pa_last_30d"].notna().any():
            warnings.append("hr_per_pa_last_30d should be NaN for each batter's first observed game.")
    return warnings


def audit_existing_engineered_dataset(df: pd.DataFrame, debug_feature_audit: bool = False, missingness_threshold: float = 0.95) -> pd.DataFrame:
    del debug_feature_audit, missingness_threshold
    validate_final_model_df(df)
    print("Input already appears to be engineered. Running final validation only.")
    return df


def _append_date_window_features(grp: pd.DataFrame, entity_id: object, entity_label: str, configs: list[tuple[str, str]]) -> pd.DataFrame:
    grp = grp.copy()
    rolled = grp.set_index("game_date")
    for window, suffix in configs:
        hr_sum = rolled["hr_count"].rolling(window, closed="left", min_periods=1).sum().reset_index(drop=True)
        pa_sum = rolled["pa_count"].rolling(window, closed="left", min_periods=1).sum().reset_index(drop=True)
        barrel_sum = rolled["barrel_count"].rolling(window, closed="left", min_periods=1).sum().reset_index(drop=True)
        hard_hit_sum = rolled["hard_hit_bbe_count"].rolling(window, closed="left", min_periods=1).sum().reset_index(drop=True)
        bbe_sum = rolled["bbe_count"].rolling(window, closed="left", min_periods=1).sum().reset_index(drop=True)
        ev95_sum = rolled["ev_95plus_bbe_count"].rolling(window, closed="left", min_periods=1).sum().reset_index(drop=True)
        avg_ev_num = (rolled["avg_exit_velocity"].fillna(0) * rolled["bbe_count"].fillna(0)).rolling(window, closed="left", min_periods=1).sum().reset_index(drop=True)
        max_ev = rolled["max_exit_velocity"].rolling(window, closed="left", min_periods=1).max().reset_index(drop=True)
        grp[f"hr_count_last_{suffix}"] = hr_sum.to_numpy()
        grp[f"pa_last_{suffix}"] = pa_sum.to_numpy()
        grp[f"barrels_last_{suffix}"] = barrel_sum.to_numpy()
        grp[f"hr_per_pa_last_{suffix}"] = safe_rate(hr_sum, pa_sum).to_numpy()
        grp[f"barrels_per_pa_last_{suffix}"] = safe_rate(barrel_sum, pa_sum).to_numpy()
        grp[f"hard_hit_rate_last_{suffix}"] = safe_rate(hard_hit_sum, bbe_sum).to_numpy()
        grp[f"bbe_95plus_ev_rate_last_{suffix}"] = safe_rate(ev95_sum, bbe_sum).to_numpy()
        if suffix == "10d":
            grp["avg_exit_velocity_last_10d"] = safe_rate(avg_ev_num, bbe_sum).to_numpy()
            grp["max_exit_velocity_last_10d"] = max_ev.to_numpy()
    if grp.sort_values(["game_date", "game_pk"]).index.tolist() != grp.index.tolist():
        raise ValueError(f"{entity_label} {entity_id} trailing window input must remain sorted by game_date and game_pk.")
    return grp


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace({0: np.nan})


def safe_scalar_rate(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def is_barrel(launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    ev = pd.to_numeric(launch_speed, errors="coerce")
    la = pd.to_numeric(launch_angle, errors="coerce")
    min_angle = 26 - ((ev - 98).clip(lower=0) * 0.5)
    max_angle = 30 + ((ev - 98).clip(lower=0) * 0.5)
    return ev.ge(98) & la.ge(min_angle) & la.le(max_angle)


def is_pull_air(pa_df: pd.DataFrame) -> pd.Series:
    air = pa_df["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    right_pull = pa_df["stand"].eq("R") & pa_df["spray_angle"].lt(-15)
    left_pull = pa_df["stand"].eq("L") & pa_df["spray_angle"].gt(15)
    return air & (right_pull | left_pull)


def classify_pitch_type_bucket(pitch_type: object) -> str | float:
    if pd.isna(pitch_type):
        return np.nan
    pitch = str(pitch_type).upper()
    if pitch in FASTBALL_TYPES:
        return "fastball"
    if pitch in BREAKING_BALL_TYPES:
        return "breaking_ball"
    if pitch in OFFSPEED_TYPES:
        return "offspeed"
    return np.nan


def classify_granular_pitch_type(pitch_type: object) -> str | float:
    if pd.isna(pitch_type):
        return np.nan
    pitch = str(pitch_type).upper()
    for label, pitch_types in GRANULAR_PITCH_TYPES.items():
        if pitch in pitch_types:
            return label
    return np.nan


def _classify_input_dataframe(df: pd.DataFrame) -> tuple[str, list[str]]:
    raw_markers = [column for column in ["at_bat_number", "pitch_number", "events", "inning_topbot"] if column in df.columns]
    engineered_markers = [column for column in ["game_pk", "batter_id", "hit_hr"] if column in df.columns]
    if len(raw_markers) == 4:
        return "raw_statcast", raw_markers
    if len(engineered_markers) == 3:
        return "engineered_dataset", engineered_markers
    return "unknown", raw_markers + engineered_markers


def _load_input_dataframe(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path, parse_dates=["game_date"], low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", nargs="?", default=str(FINAL_DATA_PATH))
    parser.add_argument("--output", default=str(FINAL_DATA_PATH))
    parser.add_argument("--mode", choices=["auto", "rebuild", "audit"], default="auto")
    args = parser.parse_args()

    input_df = _load_input_dataframe(args.input_path)
    dataset_kind, trigger_columns = _classify_input_dataframe(input_df)
    print(f"Reading input file: {args.input_path}")
    print(f"Requested mode: {args.mode}")
    print(f"Classified input as: {dataset_kind} via {trigger_columns}")

    if args.mode == "audit" or (args.mode == "auto" and dataset_kind == "engineered_dataset"):
        dataset = audit_existing_engineered_dataset(input_df)
    else:
        if dataset_kind != "raw_statcast":
            missing_statcast = sorted(set(STATCAST_COLUMNS) - set(input_df.columns))
            raise ValueError(f"Expected raw Statcast pitch-level input. Missing columns include: {missing_statcast[:10]}")
        batter_game_df, pitcher_game_df = build_player_game_dataset(input_df)
        dataset = add_leakage_safe_features(batter_game_df, pitcher_game_df, statcast_df=input_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
