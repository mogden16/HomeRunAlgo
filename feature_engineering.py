"""Feature engineering for a one-row-per-batter-game Statcast dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import AB_EVENTS, FINAL_DATA_PATH, PA_ENDING_EVENTS, PARKS, STATCAST_COLUMNS

SPRAY_CENTER_X = 125.42
SPRAY_HOME_Y = 198.27
FASTBALL_TYPES = {"FF", "FT", "SI", "FC", "FA"}
BREAKING_BALL_TYPES = {"SL", "CU", "KC", "KN", "SV", "CS"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "SC", "EP"}
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

FINAL_FEATURE_SPECS = {
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
    batter_game_df = aggregate_batter_games(pa_df)
    pitcher_game_df = aggregate_pitcher_games(pa_df)
    validate_batter_game_df(batter_game_df)
    validate_pitcher_game_df(pitcher_game_df)

    print("=== PLATE APPEARANCE DIAGNOSTICS ===")
    print(f"total pitch rows: {len(raw_df):,}")
    print(f"total inferred plate appearances: {len(pa_df):,}")
    print(f"total batter-game rows: {len(batter_game_df):,}")
    print(f"average PA per batter-game: {batter_game_df['pa_count'].mean():.3f}")
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
    return pa_df


def aggregate_batter_games(pa_df: pd.DataFrame) -> pd.DataFrame:
    # Assumption: opponent_pitcher_id is the pitcher faced in the batter's terminal PA row.
    # We keep the most frequent pitcher seen by the batter in that game as the opposing pitcher proxy.
    batter_pitcher_map = (
        pa_df.groupby(["game_pk", "batter", "pitcher"], dropna=False)
        .size()
        .reset_index(name="pa_vs_pitcher")
        .sort_values(["game_pk", "batter", "pa_vs_pitcher", "pitcher"], ascending=[True, True, False, True])
        .drop_duplicates(["game_pk", "batter"], keep="first")
        .rename(columns={"pitcher": "pitcher_id"})
    )

    batter_game_df = (
        pa_df.groupby(["game_pk", "game_date", "batter"], dropna=False)
        .agg(
            batter_name=("player_name", "first"),
            team=("batting_team", "first"),
            opponent=("fielding_team", "first"),
            is_home=("is_home", "max"),
            bat_side=("stand", "first"),
            pitcher_hand=("p_throws", "first"),
            pa_count=("pa_count", "sum"),
            hr_count=("hr_count", "sum"),
            bbe_count=("bbe_count", "sum"),
            barrel_count=("barrel_count", "sum"),
            hard_hit_bbe_count=("hard_hit_bbe_count", "sum"),
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
        batter_pitcher_map[["game_pk", "batter", "pitcher_id"]].rename(columns={"batter": "batter_id"}),
        left_on=["game_pk", "batter_id"],
        right_on=["game_pk", "batter_id"],
        how="left",
        step_name="attach primary opposing pitcher to batter-game table",
        validate="one_to_one",
    )
    batter_game_df["player_id"] = batter_game_df["batter_id"]
    batter_game_df["player_name"] = batter_game_df["batter_name"]
    batter_game_df["opp_pitcher_id"] = batter_game_df["pitcher_id"]
    batter_game_df["opp_pitcher_name"] = np.nan
    batter_game_df["pitch_hand_primary"] = batter_game_df["pitcher_hand"]
    batter_game_df["opp_pitcher_bf"] = np.nan
    batter_game_df["ballpark"] = np.where(
        batter_game_df["is_home"].astype(bool),
        batter_game_df["team"].map(lambda team: PARKS.get(team, {}).get("ballpark")),
        batter_game_df["opponent"].map(lambda team: PARKS.get(team, {}).get("ballpark")),
    )
    batter_game_df["park_factor_hr"] = np.nan
    batter_game_df = batter_game_df.sort_values(["batter_id", "game_date", "game_pk"]).reset_index(drop=True)
    return batter_game_df


def aggregate_pitcher_games(pa_df: pd.DataFrame) -> pd.DataFrame:
    pitcher_game_df = (
        pa_df.groupby(["game_pk", "game_date", "pitcher"], dropna=False)
        .agg(
            pitcher_name=("player_name", "first"),
            p_throws=("p_throws", "first"),
            pa_against=("pa_count", "sum"),
            hr_allowed=("hr_count", "sum"),
            bbe_allowed=("bbe_count", "sum"),
            barrels_allowed=("barrel_count", "sum"),
            hard_hit_bbe_allowed=("hard_hit_bbe_count", "sum"),
            avg_ev_allowed=("launch_speed", "mean"),
            max_ev_allowed=("launch_speed", "max"),
            ev_95plus_bbe_allowed=("ev_95plus_bbe_count", "sum"),
        )
        .reset_index()
        .rename(columns={"pitcher": "pitcher_id"})
    )
    pitcher_game_df = pitcher_game_df.sort_values(["pitcher_id", "game_date", "game_pk"]).reset_index(drop=True)
    return pitcher_game_df


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

    batter_features = compute_batter_trailing_features(batter_df)
    pitcher_features = compute_pitcher_trailing_features(pitcher_df)

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

    print_before_after_trailing_report(
        batter_before_counts=batter_before_counts,
        batter_after_counts=count_non_nulls(dataset, BATTER_TRAILING_FEATURE_COLUMNS),
        pitcher_before_counts=pitcher_before_counts,
        pitcher_after_counts=count_non_nulls(dataset, PITCHER_TRAILING_FEATURE_COLUMNS),
    )

    pitch_type_status = {"included": False, "reason": "pitch-type features were not attempted"}
    if statcast_df is not None:
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

    dataset["platoon_advantage"] = np.where(
        dataset["bat_side"].notna() & dataset["pitch_hand_primary"].notna(),
        (dataset["bat_side"] != dataset["pitch_hand_primary"]).astype(float),
        np.nan,
    )
    dataset["starter_or_bullpen_proxy"] = np.where(dataset["pa_count"] >= 3, "starter_like", "bullpen_like")

    decisions = finalize_feature_export(dataset, missingness_threshold=missingness_threshold)
    final_columns = base_export_columns(dataset) + [feature for feature, decision in decisions.items() if decision.included_in_export]
    dataset = dataset.loc[:, [column for column in final_columns if column in dataset.columns]].copy()
    dataset = dataset.sort_values(["game_date", "game_pk", "batter_id"]).reset_index(drop=True)
    dataset["player_id"] = dataset["batter_id"]
    validate_final_model_df(dataset)
    print_final_feature_quality_summary(dataset, decisions)
    print_rerun_verdict(dataset, pitch_type_status)
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


def build_pitch_type_matchup_features(statcast_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = statcast_df.copy()
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"], errors="coerce")
    required_raw = {"pitch_type", "description", "batter", "pitcher", "game_pk"}
    if not required_raw.issubset(raw_df.columns):
        warning = f"Skipping pitch-type features because raw Statcast is missing columns: {sorted(required_raw - set(raw_df.columns))}"
        print(f"WARNING: {warning}")
        return pd.DataFrame(), {"included": False, "reason": warning}

    print("Pitch-type feature path selected: Path A (raw pitch-level Statcast with pitch_type support).")
    pa_df = extract_plate_appearances(raw_df)
    fastball_pa = pa_df[pa_df["pitch_type_bucket"] == "fastball"].copy()
    if fastball_pa.empty:
        warning = "Skipping pitch-type features because no fastball-classified terminal PA rows were available."
        print(f"WARNING: {warning}")
        return pd.DataFrame(), {"included": False, "reason": warning}

    batter_fastball = (
        fastball_pa.groupby("batter", dropna=False)
        .agg(
            batter_hard_hit_rate_vs_fastballs=("hard_hit_bbe_count", "mean"),
            batter_barrel_rate_vs_fastballs=("barrel_count", "mean"),
            batter_contact_rate_vs_fastballs=("contact_event_count", lambda s: safe_scalar_rate(s.sum(), fastball_pa.loc[s.index, "swing_event_count"].sum())),
        )
        .reset_index()
        .rename(columns={"batter": "batter_id"})
    )
    pitcher_share = (
        raw_df.assign(pitch_type_bucket=raw_df["pitch_type"].map(classify_pitch_type_bucket))
        .dropna(subset=["pitcher", "pitch_type_bucket"])
        .groupby(["game_pk", "pitcher", "pitch_type_bucket"], dropna=False)
        .size()
        .rename("pitch_count")
        .reset_index()
    )
    if pitcher_share.empty:
        warning = "Skipping pitch-type features because raw Statcast pitch_type counts could not be computed."
        print(f"WARNING: {warning}")
        return pd.DataFrame(), {"included": False, "reason": warning}

    totals = pitcher_share.groupby(["game_pk", "pitcher"], dropna=False)["pitch_count"].sum().rename("total_pitches").reset_index()
    pitcher_share = pitcher_share.merge(totals, on=["game_pk", "pitcher"], how="left", validate="many_to_one")
    pitcher_share["pitch_pct"] = pitcher_share["pitch_count"] / pitcher_share["total_pitches"].replace({0: np.nan})
    pitcher_game_share = (
        pitcher_share.pivot_table(index=["game_pk", "pitcher"], columns="pitch_type_bucket", values="pitch_pct", aggfunc="first")
        .reset_index()
        .rename(columns={"pitcher": "pitcher_id", "fastball": "pitcher_fastball_pct", "breaking_ball": "pitcher_breaking_ball_pct", "offspeed": "pitcher_offspeed_pct"})
    )
    for column in ["pitcher_fastball_pct", "pitcher_breaking_ball_pct", "pitcher_offspeed_pct"]:
        if column not in pitcher_game_share.columns:
            pitcher_game_share[column] = np.nan

    features = merge_with_diagnostics(
        model_df[["batter_id", "game_pk", "pitcher_id"]],
        pitcher_game_share,
        on=["game_pk", "pitcher_id"],
        how="left",
        step_name="attach pitcher game-level pitch-type shares",
        validate="many_to_one",
    )
    features = merge_with_diagnostics(
        features,
        batter_fastball,
        on=["batter_id"],
        how="left",
        step_name="attach batter fastball outcome splits",
        validate="many_to_one",
    )
    features["fastball_matchup_hard_hit"] = features["pitcher_fastball_pct"] * features["batter_hard_hit_rate_vs_fastballs"]
    features["fastball_matchup_barrel"] = features["pitcher_fastball_pct"] * features["batter_barrel_rate_vs_fastballs"]
    return features.drop(columns=["pitcher_id"]), {"included": True, "reason": "raw Statcast pitch_type was available"}


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
    merged = left_df.merge(right_df, indicator=True, **merge_kwargs)
    print(f"row count after merge: {len(merged):,}")
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
        if missing_pct > missingness_threshold:
            print(f"WARNING: Excluding {feature} because missingness {missing_pct:.1%} exceeded threshold {missingness_threshold:.1%}.")
            decisions[feature] = ExportDecision(source_table, False, f"missingness {missing_pct:.1%} above threshold")
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
        series = df[feature] if feature in df.columns else pd.Series(dtype=float)
        non_null = int(series.notna().sum()) if feature in df.columns else 0
        missing_pct = float(series.isna().mean() * 100) if feature in df.columns and len(df) else 100.0
        print(f"{feature} | {non_null:,} | {missing_pct:.2f}% | {decision.source_table} | {'yes' if decision.included_in_export else 'no'} | {decision.reason}")
    print("=====================================\n")


def print_rerun_verdict(df: pd.DataFrame, pitch_type_status: dict[str, object]) -> None:
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
    print(f"pitch_type_features_included: {pitch_type_status.get('included', False)}")
    print(f"pitch_type_features_reason: {pitch_type_status.get('reason')}")
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
