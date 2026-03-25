"""Leakage-safe aggregation and rolling feature engineering for player-game HR prediction."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd

from config import AB_EVENTS, PA_ENDING_EVENTS, get_park_info

SPRAY_CENTER_X = 125.42
SPRAY_HOME_Y = 198.27
PITCH_FAMILY_CODES = {
    "fastball": {"FF", "FA", "FT", "SI", "FC"},
    "breaking": {"SL", "ST", "CU", "KC", "SV", "SC", "CS"},
    "offspeed": {"CH", "FS", "FO", "EP"},
}
PITCH_FAMILY_ORDER = tuple(PITCH_FAMILY_CODES.keys())
PITCH_TYPE_MATCHUP_FEATURE_COLUMNS = [
    "pitcher_fastball_share_season_to_date",
    "pitcher_breaking_share_season_to_date",
    "matchup_expected_barrel_rate_vs_pitch_mix",
    "matchup_expected_hard_hit_rate_vs_pitch_mix",
    "matchup_expected_fly_ball_rate_vs_pitch_mix",
]
PITCH_QUALITY_CONTEXT_FEATURE_COLUMNS = [
    "pitcher_release_spin_rate_season_to_date",
    "pitcher_release_extension_season_to_date",
    "pitcher_spin_axis_sin_season_to_date",
    "pitcher_spin_axis_cos_season_to_date",
]
PITCH_STYLE_FEATURE_COLUMNS = PITCH_TYPE_MATCHUP_FEATURE_COLUMNS + PITCH_QUALITY_CONTEXT_FEATURE_COLUMNS
LONG_CONTACT_DISTANCE_FT = 380.0
CONTACT_AUTHORITY_FEATURE_COLUMNS = [
    "avg_hit_distance_last_50_bbe",
    "long_contact_rate_last_50_bbe",
    "pitcher_avg_hit_distance_allowed_last_50_bbe",
]

LEGACY_ENGINEERED_FEATURE_COLUMNS = [
    "hr_per_pa_last_30d",
    "expected_pa_proxy",
    "recent_form_hr_last_7d",
    "recent_form_barrels_last_14d",
]

CURRENT_ENGINEERED_FEATURE_COLUMNS = [
    "hr_rate_season_to_date",
    "barrel_rate_last_50_bbe",
    "hard_hit_rate_last_50_bbe",
    "avg_launch_angle_last_50_bbe",
    "avg_exit_velocity_last_50_bbe",
    "fly_ball_rate_last_50_bbe",
    "pull_air_rate_last_50_bbe",
    "batter_k_rate_season_to_date",
    "batter_bb_rate_season_to_date",
    "days_since_last_game",
    "bbe_count_last_50",
    "avg_hit_distance_last_50_bbe",
    "long_contact_rate_last_50_bbe",
    "pitcher_hr9_season_to_date",
    "pitcher_barrel_rate_allowed_last_50_bbe",
    "pitcher_hard_hit_rate_allowed_last_50_bbe",
    "pitcher_fb_rate_allowed_last_50_bbe",
    "pitcher_avg_hit_distance_allowed_last_50_bbe",
    "pitcher_k_rate_season_to_date",
    "pitcher_bb_rate_season_to_date",
    "pitcher_fastball_share_season_to_date",
    "pitcher_breaking_share_season_to_date",
    "matchup_expected_barrel_rate_vs_pitch_mix",
    "matchup_expected_hard_hit_rate_vs_pitch_mix",
    "matchup_expected_fly_ball_rate_vs_pitch_mix",
    "pitcher_release_spin_rate_season_to_date",
    "pitcher_release_extension_season_to_date",
    "pitcher_spin_axis_sin_season_to_date",
    "pitcher_spin_axis_cos_season_to_date",
    "park_factor_hr",
    "platoon_advantage",
    "starter_or_bullpen_proxy",
    "temperature_f",
    "humidity_pct",
    "wind_speed_mph",
    "wind_direction_deg",
    "pressure_hpa",
]

CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS = [
    "hr_rate_season_to_date",
    "barrel_rate_last_50_bbe",
    "hard_hit_rate_last_50_bbe",
    "avg_launch_angle_last_50_bbe",
    "avg_exit_velocity_last_50_bbe",
    "fly_ball_rate_last_50_bbe",
    "pull_air_rate_last_50_bbe",
    "batter_k_rate_season_to_date",
    "batter_bb_rate_season_to_date",
    "days_since_last_game",
    "bbe_count_last_50",
    "avg_hit_distance_last_50_bbe",
    "long_contact_rate_last_50_bbe",
    "pitcher_hr9_season_to_date",
    "pitcher_barrel_rate_allowed_last_50_bbe",
    "pitcher_hard_hit_rate_allowed_last_50_bbe",
    "pitcher_fb_rate_allowed_last_50_bbe",
    "pitcher_avg_hit_distance_allowed_last_50_bbe",
    "pitcher_k_rate_season_to_date",
    "pitcher_bb_rate_season_to_date",
    "pitcher_fastball_share_season_to_date",
    "pitcher_breaking_share_season_to_date",
    "matchup_expected_barrel_rate_vs_pitch_mix",
    "matchup_expected_hard_hit_rate_vs_pitch_mix",
    "matchup_expected_fly_ball_rate_vs_pitch_mix",
    "pitcher_release_spin_rate_season_to_date",
    "pitcher_release_extension_season_to_date",
    "pitcher_spin_axis_sin_season_to_date",
    "pitcher_spin_axis_cos_season_to_date",
    "park_factor_hr",
    "platoon_advantage",
    "temperature_f",
    "humidity_pct",
    "wind_speed_mph",
    "wind_direction_deg",
    "pressure_hpa",
]


def build_player_game_dataset(statcast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Statcast pitch-level data into one batter-game row."""
    pa_df = _extract_plate_appearances(statcast_df)
    if pa_df.empty:
        raise RuntimeError("No plate appearances were derived from Statcast input.")

    player_game = _aggregate_batter_games(pa_df)
    pitcher_game = _aggregate_pitcher_games(pa_df)
    primary_pitchers = _select_primary_pitchers(pitcher_game)

    dataset = player_game.merge(
        primary_pitchers,
        how="left",
        on=["game_pk", "team", "opponent"],
        validate="many_to_one",
    )

    dataset["ballpark"] = _map_ballparks(dataset["opponent"], dataset["game_date"])
    home_mask = dataset["is_home"].astype(bool)
    dataset.loc[home_mask, "ballpark"] = _map_ballparks(
        dataset.loc[home_mask, "team"],
        dataset.loc[home_mask, "game_date"],
    )
    dataset = dataset.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
    return dataset, pitcher_game


def attach_park_factors(player_game_df: pd.DataFrame, park_factor_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach season-level park factors to player-game rows using the resolved home park."""
    dataset = player_game_df.copy()
    dataset["home_team"] = np.where(dataset["is_home"].astype(bool), dataset["team"], dataset["opponent"])
    dataset["ballpark"] = _map_ballparks(dataset["home_team"], dataset["game_date"])

    park_factor_lookup = park_factor_df.copy()
    if "home_team" not in park_factor_lookup.columns:
        park_factor_lookup["home_team"] = np.nan
    park_factor_lookup["park_factor_hr"] = pd.to_numeric(park_factor_lookup["park_factor_hr"], errors="coerce")

    dataset = dataset.merge(
        park_factor_lookup[["home_team", "park_factor_hr", "source"]].rename(columns={"source": "park_factor_source"}),
        on="home_team",
        how="left",
        validate="many_to_one",
    )

    matched_rows = int(dataset["park_factor_hr"].notna().sum())
    unmatched_rows = int(dataset["park_factor_hr"].isna().sum())
    matched_parks = sorted(dataset.loc[dataset["park_factor_hr"].notna(), "ballpark"].dropna().unique().tolist())
    unmatched_parks = sorted(dataset.loc[dataset["park_factor_hr"].isna(), "ballpark"].dropna().unique().tolist())
    source_values = sorted(dataset["park_factor_source"].dropna().astype(str).unique().tolist())
    audit = {
        "source": source_values[0] if len(source_values) == 1 else ",".join(source_values) if source_values else "unknown",
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "non_null_share": float(matched_rows / len(dataset)) if len(dataset) else 0.0,
        "matched_parks": matched_parks,
        "unmatched_parks": unmatched_parks,
    }
    return dataset.drop(columns=["home_team"]), audit


def _map_ballparks(team_series: pd.Series, game_date_series: pd.Series) -> pd.Series:
    values = [
        get_park_info(str(team), season=pd.Timestamp(game_date).year).get("ballpark")
        if pd.notna(team) and pd.notna(game_date)
        else np.nan
        for team, game_date in zip(team_series, game_date_series)
    ]
    return pd.Series(values, index=team_series.index)


def add_leakage_safe_features(
    player_game_df: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    statcast_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add rolling batter and pitcher features using only pre-game information."""
    df = player_game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["opp_pitcher_id"] = pd.to_numeric(df["opp_pitcher_id"], errors="coerce").astype("Int64")
    pitcher_game_df = pitcher_game_df.copy()
    pitcher_game_df["game_date"] = pd.to_datetime(pitcher_game_df["game_date"])
    pitch_history_df, pitch_history_audit = _prepare_pitch_history(statcast_df)

    batter_daily = _build_batter_daily(df)
    batter_features = _compute_batter_daily_features(batter_daily)
    df = df.merge(batter_features, on=["player_id", "game_date"], how="left", validate="many_to_one")

    pitcher_daily = _build_pitcher_daily(pitcher_game_df)
    pitcher_features = _compute_pitcher_daily_features(pitcher_daily)
    df = df.merge(
        pitcher_features,
        left_on=["opp_pitcher_id", "game_date"],
        right_on=["pitcher_id", "game_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["pitcher_id"], errors="ignore")

    batter_pitch_matchup = _compute_batter_pitch_matchup_features(pitch_history_df)
    df = df.merge(batter_pitch_matchup, on=["player_id", "game_date"], how="left", validate="many_to_one")

    pitcher_pitch_context = _compute_pitcher_pitch_context_features(pitch_history_df)
    df = df.merge(
        pitcher_pitch_context,
        left_on=["opp_pitcher_id", "game_date"],
        right_on=["pitcher_id", "game_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["pitcher_id"], errors="ignore")

    for outcome_name in ("barrel", "hard_hit", "fly_ball"):
        batter_cols = [f"batter_{outcome_name}_rate_vs_{family}_season_to_date" for family in PITCH_FAMILY_ORDER]
        share_cols = [f"pitcher_{family}_share_season_to_date" for family in PITCH_FAMILY_ORDER]
        df[f"matchup_expected_{outcome_name}_rate_vs_pitch_mix"] = _weighted_share_average(df, batter_cols, share_cols)

    df["platoon_advantage"] = np.where(
        df["bat_side"].notna() & df["pitch_hand_primary"].notna(),
        (df["bat_side"] != df["pitch_hand_primary"]).astype(float),
        np.nan,
    )
    df["starter_or_bullpen_proxy"] = np.where(df["opp_pitcher_bf"] >= 12, "starter_like", "bullpen_like")
    feature_audit = _build_feature_family_audit(pitch_history_audit, df)
    intermediate_pitch_columns = [
        *(f"batter_{outcome}_rate_vs_{family}_season_to_date" for outcome in ("barrel", "hard_hit", "fly_ball") for family in PITCH_FAMILY_ORDER),
        "pitcher_offspeed_share_season_to_date",
    ]
    df = df.drop(columns=list(intermediate_pitch_columns), errors="ignore")
    return df, feature_audit


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return sanity-check warnings or raise on hard validation failures."""
    required = ["game_date", "game_pk", "player_id", "hit_hr"]
    missing_required = df[required].isna().any()
    if missing_required.any():
        bad_cols = list(missing_required[missing_required].index)
        raise ValueError(f"Dataset has missing required values in columns: {bad_cols}")

    duplicate_count = int(df.duplicated(["game_pk", "player_id"]).sum())
    if duplicate_count:
        raise ValueError(f"Dataset has {duplicate_count} duplicate batter-game rows.")

    warnings: list[str] = []
    # Leakage check: shifted season-to-date rates should be NaN for each player's first
    # appearance, because there is no prior-game history available yet.
    first_game_date_per_player = df.groupby("player_id")["game_date"].transform("min")
    is_first_game = df["game_date"] == first_game_date_per_player
    if df.loc[is_first_game, "hr_rate_season_to_date"].notna().any():
        warnings.append(
            "hr_rate_season_to_date is non-NaN for a player's first game appearance; "
            "the shifted cumulative season-to-date calculation may be leaking same-day data."
        )
    if "matchup_expected_barrel_rate_vs_pitch_mix" in df.columns and df.loc[is_first_game, "matchup_expected_barrel_rate_vs_pitch_mix"].notna().any():
        warnings.append(
            "matchup_expected_barrel_rate_vs_pitch_mix is non-NaN for a batter's first game appearance; "
            "pitch-type matchup features may be leaking same-day data."
        )
    if "avg_hit_distance_last_50_bbe" in df.columns and df.loc[is_first_game, "avg_hit_distance_last_50_bbe"].notna().any():
        warnings.append(
            "avg_hit_distance_last_50_bbe is non-NaN for a batter's first game appearance; "
            "contact-authority features may be leaking same-day data."
        )
    return warnings


def _extract_plate_appearances(statcast_df: pd.DataFrame) -> pd.DataFrame:
    df = statcast_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])

    pa_end_mask = df["events"].notna() & df["events"].isin(PA_ENDING_EVENTS)
    if not pa_end_mask.any():
        raise RuntimeError("Unable to identify plate-appearance ending rows from Statcast events.")

    pa_df = df.loc[pa_end_mask].drop_duplicates(["game_pk", "at_bat_number"], keep="last").copy()
    pa_df["team"] = np.where(pa_df["inning_topbot"].eq("Top"), pa_df["away_team"], pa_df["home_team"])
    pa_df["opponent"] = np.where(pa_df["inning_topbot"].eq("Top"), pa_df["home_team"], pa_df["away_team"])
    pa_df["is_home"] = pa_df["team"].eq(pa_df["home_team"]).astype(int)
    pa_df["plate_appearance"] = 1
    pa_df["at_bat"] = pa_df["events"].isin(AB_EVENTS).astype(int)
    pa_df["is_hr"] = pa_df["events"].eq("home_run").astype(int)
    pa_df["is_bb"] = pa_df["events"].isin(["walk", "intent_walk"]).astype(int)
    pa_df["is_k"] = pa_df["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    pa_df["is_bbe"] = pa_df["launch_speed"].notna().astype(int)
    pa_df["is_hard_hit"] = (pa_df["launch_speed"] >= 95).fillna(False).astype(int)
    pa_df["is_barrel"] = _is_barrel(pa_df["launch_speed"], pa_df["launch_angle"]).fillna(False).astype(int)
    pa_df["is_fly_ball"] = pa_df["bb_type"].isin(["fly_ball", "popup"]).astype(int)
    if "hit_distance_sc" not in pa_df.columns:
        pa_df["hit_distance_sc"] = np.nan
    pa_df["hit_distance_sc"] = pd.to_numeric(pa_df["hit_distance_sc"], errors="coerce")
    pa_df["contact_distance_sum"] = pa_df["hit_distance_sc"].fillna(0.0)
    pa_df["is_long_contact"] = (
        pa_df["hit_distance_sc"].ge(LONG_CONTACT_DISTANCE_FT).fillna(False) & pa_df["is_bbe"].eq(1)
    ).astype(int)
    pa_df["spray_angle"] = np.degrees(np.arctan2(pa_df["hc_x"] - SPRAY_CENTER_X, SPRAY_HOME_Y - pa_df["hc_y"]))
    pa_df["is_pull_air"] = _is_pull_air(pa_df)
    pa_df["batting_order"] = pa_df.groupby(["game_pk", "team"])["at_bat_number"].rank(method="dense").clip(upper=9)
    return pa_df


def _aggregate_batter_games(pa_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "game_date",
        "game_pk",
        "batter",
        "team",
        "opponent",
        "is_home",
    ]
    agg = pa_df.groupby(group_cols, dropna=False).agg(
        player_name=("player_name", "first"),
        plate_appearances=("plate_appearance", "sum"),
        at_bats=("at_bat", "sum"),
        hr_count=("is_hr", "sum"),
        bbe_count=("is_bbe", "sum"),
        barrel_count=("is_barrel", "sum"),
        hard_hit_count=("is_hard_hit", "sum"),
        fly_ball_count=("is_fly_ball", "sum"),
        pull_air_count=("is_pull_air", "sum"),
        contact_distance_sum=("contact_distance_sum", "sum"),
        long_contact_count=("is_long_contact", "sum"),
        avg_launch_angle=("launch_angle", "mean"),
        avg_exit_velocity=("launch_speed", "mean"),
        batter_k_count=("is_k", "sum"),
        batter_bb_count=("is_bb", "sum"),
        batting_order=("batting_order", "min"),
        bat_side=("stand", lambda x: x.mode().iloc[0] if len(x) > 0 else None),
    ).reset_index()
    agg = agg.rename(columns={"batter": "player_id"})
    agg["hit_hr"] = (agg["hr_count"] > 0).astype(int)
    agg["batting_order"] = agg["batting_order"].where(agg["batting_order"] <= 9)
    return agg


def _aggregate_pitcher_games(pa_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["game_date", "game_pk", "team", "opponent", "pitcher", "p_throws"]
    agg = pa_df.groupby(group_cols, dropna=False).agg(
        batters_faced=("plate_appearance", "sum"),
        pitcher_hr_allowed=("is_hr", "sum"),
        pitcher_bbe_allowed=("is_bbe", "sum"),
        pitcher_barrel_allowed=("is_barrel", "sum"),
        pitcher_hard_hit_allowed=("is_hard_hit", "sum"),
        pitcher_fb_allowed=("is_fly_ball", "sum"),
        pitcher_contact_distance_allowed_sum=("contact_distance_sum", "sum"),
        pitcher_k_count=("is_k", "sum"),
        pitcher_bb_count=("is_bb", "sum"),
        outs_recorded=("at_bat", "sum"),
    ).reset_index()
    agg["innings_pitched_est"] = agg["outs_recorded"] / 3.0
    return agg


def _select_primary_pitchers(pitcher_game_df: pd.DataFrame) -> pd.DataFrame:
    df = pitcher_game_df.sort_values(["game_pk", "team", "batters_faced"], ascending=[True, True, False]).copy()
    primary = df.drop_duplicates(["game_pk", "team"], keep="first").copy()
    primary = primary.rename(
        columns={
            "team": "team",
            "opponent": "opponent",
            "pitcher": "opp_pitcher_id",
            "p_throws": "pitch_hand_primary",
            "batters_faced": "opp_pitcher_bf",
        }
    )
    primary["opp_pitcher_name"] = np.nan
    return primary[["game_pk", "team", "opponent", "opp_pitcher_id", "opp_pitcher_name", "pitch_hand_primary", "opp_pitcher_bf"]]


def _build_batter_daily(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["avg_exit_velocity_num"] = enriched["avg_exit_velocity"] * enriched["bbe_count"]
    enriched["avg_launch_angle_num"] = enriched["avg_launch_angle"] * enriched["bbe_count"]
    daily = enriched.groupby(["player_id", "game_date"], dropna=False).agg(
        plate_appearances=("plate_appearances", "sum"),
        hr_count=("hr_count", "sum"),
        bbe_count=("bbe_count", "sum"),
        barrel_count=("barrel_count", "sum"),
        hard_hit_count=("hard_hit_count", "sum"),
        fly_ball_count=("fly_ball_count", "sum"),
        pull_air_count=("pull_air_count", "sum"),
        contact_distance_sum=("contact_distance_sum", "sum"),
        long_contact_count=("long_contact_count", "sum"),
        batter_k_count=("batter_k_count", "sum"),
        batter_bb_count=("batter_bb_count", "sum"),
        avg_exit_velocity_num=("avg_exit_velocity_num", "sum"),
        avg_launch_angle_num=("avg_launch_angle_num", "sum"),
    ).reset_index()
    return daily.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def _compute_batter_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    features: list[pd.DataFrame] = []
    for _, group in daily.groupby("player_id", sort=False):
        grp = group.sort_values("game_date").copy()
        grp["hr_rate_season_to_date"] = _shifted_cumulative_rate(grp["hr_count"], grp["plate_appearances"])
        grp["batter_k_rate_season_to_date"] = _shifted_cumulative_rate(grp["batter_k_count"], grp["plate_appearances"])
        grp["batter_bb_rate_season_to_date"] = _shifted_cumulative_rate(grp["batter_bb_count"], grp["plate_appearances"])
        grp["days_since_last_game"] = grp["game_date"].diff().dt.days.astype(float)
        grp[[
            "barrel_rate_last_50_bbe",
            "hard_hit_rate_last_50_bbe",
            "fly_ball_rate_last_50_bbe",
            "pull_air_rate_last_50_bbe",
            "avg_exit_velocity_last_50_bbe",
            "avg_launch_angle_last_50_bbe",
            "avg_hit_distance_last_50_bbe",
            "long_contact_rate_last_50_bbe",
            "bbe_count_last_50",
        ]] = _count_window_features(
            grp,
            count_col="bbe_count",
            numerators={
                "barrel_rate_last_50_bbe": "barrel_count",
                "hard_hit_rate_last_50_bbe": "hard_hit_count",
                "fly_ball_rate_last_50_bbe": "fly_ball_count",
                "pull_air_rate_last_50_bbe": "pull_air_count",
                "long_contact_rate_last_50_bbe": "long_contact_count",
            },
            weighted_means={
                "avg_exit_velocity_last_50_bbe": "avg_exit_velocity_num",
                "avg_launch_angle_last_50_bbe": "avg_launch_angle_num",
                "avg_hit_distance_last_50_bbe": "contact_distance_sum",
            },
            window_size=50,
        )
        features.append(grp[[
            "player_id",
            "game_date",
            "hr_rate_season_to_date",
            "barrel_rate_last_50_bbe",
            "hard_hit_rate_last_50_bbe",
            "avg_launch_angle_last_50_bbe",
            "avg_exit_velocity_last_50_bbe",
            "fly_ball_rate_last_50_bbe",
            "pull_air_rate_last_50_bbe",
            "batter_k_rate_season_to_date",
            "batter_bb_rate_season_to_date",
            "days_since_last_game",
            "bbe_count_last_50",
            "avg_hit_distance_last_50_bbe",
            "long_contact_rate_last_50_bbe",
        ]])
    return pd.concat(features, ignore_index=True)


def _build_pitcher_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(["pitcher", "game_date"], dropna=False).agg(
        innings_pitched_est=("innings_pitched_est", "sum"),
        pitcher_hr_allowed=("pitcher_hr_allowed", "sum"),
        pitcher_bbe_allowed=("pitcher_bbe_allowed", "sum"),
        pitcher_barrel_allowed=("pitcher_barrel_allowed", "sum"),
        pitcher_hard_hit_allowed=("pitcher_hard_hit_allowed", "sum"),
        pitcher_fb_allowed=("pitcher_fb_allowed", "sum"),
        pitcher_contact_distance_allowed_sum=("pitcher_contact_distance_allowed_sum", "sum"),
        pitcher_k_count=("pitcher_k_count", "sum"),
        pitcher_bb_count=("pitcher_bb_count", "sum"),
        batters_faced=("batters_faced", "sum"),
    ).reset_index().rename(columns={"pitcher": "pitcher_id"})
    return daily.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def _compute_pitcher_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    features: list[pd.DataFrame] = []
    for _, group in daily.groupby("pitcher_id", sort=False):
        grp = group.sort_values("game_date").copy()
        shifted_ip = grp["innings_pitched_est"].cumsum().shift(1)
        shifted_hr = grp["pitcher_hr_allowed"].cumsum().shift(1)
        grp["pitcher_hr9_season_to_date"] = np.where(shifted_ip > 0, shifted_hr * 9 / shifted_ip, np.nan)
        grp["pitcher_k_rate_season_to_date"] = _shifted_cumulative_rate(grp["pitcher_k_count"], grp["batters_faced"])
        grp["pitcher_bb_rate_season_to_date"] = _shifted_cumulative_rate(grp["pitcher_bb_count"], grp["batters_faced"])
        grp[[
            "pitcher_barrel_rate_allowed_last_50_bbe",
            "pitcher_hard_hit_rate_allowed_last_50_bbe",
            "pitcher_fb_rate_allowed_last_50_bbe",
            "pitcher_avg_hit_distance_allowed_last_50_bbe",
            "pitcher_bbe_allowed_last_50",
        ]] = _count_window_features(
            grp,
            count_col="pitcher_bbe_allowed",
            numerators={
                "pitcher_barrel_rate_allowed_last_50_bbe": "pitcher_barrel_allowed",
                "pitcher_hard_hit_rate_allowed_last_50_bbe": "pitcher_hard_hit_allowed",
                "pitcher_fb_rate_allowed_last_50_bbe": "pitcher_fb_allowed",
            },
            weighted_means={
                "pitcher_avg_hit_distance_allowed_last_50_bbe": "pitcher_contact_distance_allowed_sum",
            },
            window_size=50,
        )
        features.append(grp[[
            "pitcher_id",
            "game_date",
            "pitcher_hr9_season_to_date",
            "pitcher_barrel_rate_allowed_last_50_bbe",
            "pitcher_hard_hit_rate_allowed_last_50_bbe",
            "pitcher_fb_rate_allowed_last_50_bbe",
            "pitcher_avg_hit_distance_allowed_last_50_bbe",
            "pitcher_k_rate_season_to_date",
            "pitcher_bb_rate_season_to_date",
        ]])
    return pd.concat(features, ignore_index=True)


def _map_pitch_family(pitch_type: pd.Series) -> pd.Series:
    family_lookup = {
        code: family_name
        for family_name, codes in PITCH_FAMILY_CODES.items()
        for code in codes
    }
    normalized = pitch_type.astype(str).str.upper().str.strip()
    mapped = normalized.map(family_lookup)
    mapped = mapped.where(pitch_type.notna(), np.nan)
    return mapped


def _prepare_pitch_history(statcast_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    df = statcast_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitch_family"] = _map_pitch_family(df["pitch_type"]) if "pitch_type" in df.columns else np.nan
    df["is_bbe"] = df["launch_speed"].notna().astype(int)
    df["is_hard_hit"] = (pd.to_numeric(df["launch_speed"], errors="coerce") >= 95).fillna(False).astype(int)
    df["is_barrel"] = _is_barrel(df["launch_speed"], df["launch_angle"]).fillna(False).astype(int)
    df["is_fly_ball"] = df["bb_type"].isin(["fly_ball", "popup"]).astype(int)
    spin_axis_deg = pd.to_numeric(df["spin_axis"], errors="coerce")
    spin_axis_rad = np.deg2rad(spin_axis_deg)
    df["spin_axis_sin"] = np.where(spin_axis_deg.notna(), np.sin(spin_axis_rad), np.nan)
    df["spin_axis_cos"] = np.where(spin_axis_deg.notna(), np.cos(spin_axis_rad), np.nan)
    df["release_spin_rate"] = pd.to_numeric(df["release_spin_rate"], errors="coerce")
    df["release_extension"] = pd.to_numeric(df["release_extension"], errors="coerce")
    audit = {
        "raw_pitch_rows": int(len(df)),
        "mapped_pitch_rows": int(df["pitch_family"].notna().sum()),
        "mapped_pitch_share": float(df["pitch_family"].notna().mean()) if len(df) else 0.0,
    }
    return df, audit


def _compute_batter_pitch_matchup_features(pitch_history_df: pd.DataFrame) -> pd.DataFrame:
    required = pitch_history_df["pitch_family"].notna() & pitch_history_df["batter"].notna()
    pitch_df = pitch_history_df.loc[required].copy()
    if pitch_df.empty:
        return pd.DataFrame(columns=["player_id", "game_date"])

    grouped = (
        pitch_df.groupby(["batter", "game_date", "pitch_family"], dropna=False)
        .agg(
            family_bbe_count=("is_bbe", "sum"),
            family_barrel_count=("is_barrel", "sum"),
            family_hard_hit_count=("is_hard_hit", "sum"),
            family_fly_ball_count=("is_fly_ball", "sum"),
        )
        .reset_index()
    )
    pivoted = grouped.pivot(index=["batter", "game_date"], columns="pitch_family")
    pivoted = pivoted.sort_index(axis=1)
    pivoted.columns = [f"{metric}_{family}" for metric, family in pivoted.columns]
    pivoted = pivoted.reset_index().rename(columns={"batter": "player_id"})
    pivoted["player_id"] = pd.to_numeric(pivoted["player_id"], errors="coerce").astype("Int64")
    for family in PITCH_FAMILY_ORDER:
        for metric_name in ("family_bbe_count", "family_barrel_count", "family_hard_hit_count", "family_fly_ball_count"):
            column_name = f"{metric_name}_{family}"
            if column_name not in pivoted.columns:
                pivoted[column_name] = 0.0

    pivoted = pivoted.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    for family in PITCH_FAMILY_ORDER:
        den_col = f"family_bbe_count_{family}"
        for outcome in ("barrel", "hard_hit", "fly_ball"):
            num_col = f"family_{outcome}_count_{family}"
            feature_col = f"batter_{outcome}_rate_vs_{family}_season_to_date"
            pivoted[feature_col] = _group_shifted_cumulative_rate(pivoted, "player_id", num_col, den_col)

    feature_columns = [
        f"batter_{outcome}_rate_vs_{family}_season_to_date"
        for outcome in ("barrel", "hard_hit", "fly_ball")
        for family in PITCH_FAMILY_ORDER
    ]
    return pivoted[["player_id", "game_date", *feature_columns]]


def _compute_pitcher_pitch_context_features(pitch_history_df: pd.DataFrame) -> pd.DataFrame:
    required = pitch_history_df["pitch_family"].notna() & pitch_history_df["pitcher"].notna()
    pitch_df = pitch_history_df.loc[required].copy()
    if pitch_df.empty:
        return pd.DataFrame(columns=["pitcher_id", "game_date"])

    for family in PITCH_FAMILY_ORDER:
        pitch_df[f"family_count_{family}"] = pitch_df["pitch_family"].eq(family).astype(int)

    for metric in ("release_spin_rate", "release_extension", "spin_axis_sin", "spin_axis_cos"):
        valid_mask = pitch_df[metric].notna().astype(int)
        pitch_df[f"{metric}_sum"] = pitch_df[metric].fillna(0.0)
        pitch_df[f"{metric}_count"] = valid_mask

    agg_spec: dict[str, tuple[str, str]] = {"total_family_pitch_count": ("pitch_family", "size")}
    for family in PITCH_FAMILY_ORDER:
        agg_spec[f"family_count_{family}"] = (f"family_count_{family}", "sum")
    for metric in ("release_spin_rate", "release_extension", "spin_axis_sin", "spin_axis_cos"):
        agg_spec[f"{metric}_sum"] = (f"{metric}_sum", "sum")
        agg_spec[f"{metric}_count"] = (f"{metric}_count", "sum")

    daily = (
        pitch_df.groupby(["pitcher", "game_date"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .rename(columns={"pitcher": "pitcher_id"})
        .sort_values(["pitcher_id", "game_date"])
        .reset_index(drop=True)
    )
    daily["pitcher_id"] = pd.to_numeric(daily["pitcher_id"], errors="coerce").astype("Int64")

    total_den = "total_family_pitch_count"
    daily["pitcher_fastball_share_season_to_date"] = _group_shifted_cumulative_rate(
        daily, "pitcher_id", "family_count_fastball", total_den
    )
    daily["pitcher_breaking_share_season_to_date"] = _group_shifted_cumulative_rate(
        daily, "pitcher_id", "family_count_breaking", total_den
    )
    daily["pitcher_offspeed_share_season_to_date"] = _group_shifted_cumulative_rate(
        daily, "pitcher_id", "family_count_offspeed", total_den
    )
    for metric in ("release_spin_rate", "release_extension", "spin_axis_sin", "spin_axis_cos"):
        feature_col = f"pitcher_{metric}_season_to_date"
        daily[feature_col] = _group_shifted_cumulative_rate(
            daily,
            "pitcher_id",
            f"{metric}_sum",
            f"{metric}_count",
        )

    return daily[[
        "pitcher_id",
        "game_date",
        "pitcher_fastball_share_season_to_date",
        "pitcher_breaking_share_season_to_date",
        "pitcher_offspeed_share_season_to_date",
        "pitcher_release_spin_rate_season_to_date",
        "pitcher_release_extension_season_to_date",
        "pitcher_spin_axis_sin_season_to_date",
        "pitcher_spin_axis_cos_season_to_date",
    ]]


def _group_shifted_cumulative_rate(df: pd.DataFrame, group_col: str, numerator_col: str, denominator_col: str) -> pd.Series:
    grouped = df.groupby(group_col, sort=False)
    cumulative_num = grouped[numerator_col].cumsum()
    cumulative_den = grouped[denominator_col].cumsum()
    shifted_num = cumulative_num.groupby(df[group_col], sort=False).shift(1)
    shifted_den = cumulative_den.groupby(df[group_col], sort=False).shift(1)
    return shifted_num / shifted_den.replace({0: np.nan})


def _weighted_share_average(df: pd.DataFrame, value_columns: list[str], share_columns: list[str]) -> pd.Series:
    value_frame = df[value_columns].astype(float)
    share_frame = df[share_columns].astype(float)
    available = value_frame.notna().to_numpy() & share_frame.notna().to_numpy()
    value_values = value_frame.to_numpy(dtype=float)
    share_values = share_frame.to_numpy(dtype=float)
    weighted_sum = np.where(available, value_values * share_values, 0.0).sum(axis=1)
    available_share = np.where(available, share_values, 0.0).sum(axis=1)
    return pd.Series(weighted_sum / np.where(available_share > 0, available_share, np.nan), index=df.index)


def _build_feature_family_audit(pitch_history_audit: dict[str, object], dataset: pd.DataFrame) -> dict[str, object]:
    pitch_feature_non_null_share = {
        feature_name: float(dataset[feature_name].notna().mean()) if feature_name in dataset.columns and len(dataset) else 0.0
        for feature_name in PITCH_STYLE_FEATURE_COLUMNS
    }
    contact_feature_non_null_share = {
        feature_name: float(dataset[feature_name].notna().mean()) if feature_name in dataset.columns and len(dataset) else 0.0
        for feature_name in CONTACT_AUTHORITY_FEATURE_COLUMNS
    }
    pitch_history_audit.update(
        {
            "feature_non_null_share": pitch_feature_non_null_share,
            "near_all_null_features": [name for name, share in pitch_feature_non_null_share.items() if share <= 0.05],
            "all_null_features": [name for name, share in pitch_feature_non_null_share.items() if share == 0.0],
            "contact_feature_non_null_share": contact_feature_non_null_share,
            "contact_near_all_null_features": [name for name, share in contact_feature_non_null_share.items() if share <= 0.05],
            "contact_all_null_features": [name for name, share in contact_feature_non_null_share.items() if share == 0.0],
        }
    )
    return pitch_history_audit


def _shifted_cumulative_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    cumulative_num = numerator.cumsum().shift(1)
    cumulative_den = denominator.cumsum().shift(1)
    return cumulative_num / cumulative_den.replace({0: np.nan})


def _count_window_features(
    grp: pd.DataFrame,
    count_col: str,
    numerators: dict[str, str],
    weighted_means: dict[str, str],
    window_size: int,
) -> pd.DataFrame:
    counts = grp[count_col].fillna(0).astype(float).tolist()
    numerator_values = {name: grp[column].fillna(0).astype(float).tolist() for name, column in numerators.items()}
    weighted_values = {name: grp[column].fillna(0).astype(float).tolist() for name, column in weighted_means.items()}

    queue: deque[tuple[float, dict[str, float], dict[str, float]]] = deque()
    running_count = 0.0
    running_num = {name: 0.0 for name in numerators}
    running_weighted = {name: 0.0 for name in weighted_means}
    rows: list[dict[str, float | None]] = []

    for idx, count in enumerate(counts):
        row: dict[str, float | None] = {"bbe_count_last_50" if count_col == "bbe_count" else "pitcher_bbe_allowed_last_50": running_count}
        for name in numerators:
            row[name] = running_num[name] / running_count if running_count > 0 else np.nan
        for name in weighted_means:
            row[name] = running_weighted[name] / running_count if running_count > 0 else np.nan
        rows.append(row)

        current_num = {name: numerator_values[name][idx] for name in numerators}
        current_weighted = {name: weighted_values[name][idx] for name in weighted_means}
        queue.append((count, current_num, current_weighted))
        running_count += count
        for name in numerators:
            running_num[name] += current_num[name]
        for name in weighted_means:
            running_weighted[name] += current_weighted[name]

        while running_count > window_size and queue:
            oldest_count, oldest_num, oldest_weighted = queue[0]
            overflow = running_count - window_size
            trim = min(oldest_count, overflow)
            if oldest_count <= trim + 1e-9:
                queue.popleft()
            else:
                remaining_fraction = (oldest_count - trim) / oldest_count if oldest_count else 0.0
                queue[0] = (
                    oldest_count - trim,
                    {name: value * remaining_fraction for name, value in oldest_num.items()},
                    {name: value * remaining_fraction for name, value in oldest_weighted.items()},
                )
            running_count -= trim
            trim_fraction = trim / oldest_count if oldest_count else 0.0
            for name in numerators:
                running_num[name] -= oldest_num[name] * trim_fraction
            for name in weighted_means:
                running_weighted[name] -= oldest_weighted[name] * trim_fraction

    return pd.DataFrame(rows, index=grp.index)


def _is_barrel(launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    ev = pd.to_numeric(launch_speed, errors="coerce")
    la = pd.to_numeric(launch_angle, errors="coerce")
    min_angle = 26 - ((ev - 98).clip(lower=0) * 0.5)
    max_angle = 30 + ((ev - 98).clip(lower=0) * 0.5)
    return ev.ge(98) & la.ge(min_angle) & la.le(max_angle)


def _is_pull_air(pa_df: pd.DataFrame) -> pd.Series:
    air = pa_df["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    right_pull = pa_df["stand"].eq("R") & pa_df["spray_angle"].lt(-15)
    left_pull = pa_df["stand"].eq("L") & pa_df["spray_angle"].gt(15)
    return (air & (right_pull | left_pull)).fillna(False).astype(int)


def print_current_feature_schema() -> None:
    print("Current engineered feature schema")
    print("-" * 60)
    print(f"Engineered features ({len(CURRENT_ENGINEERED_FEATURE_COLUMNS)}): {CURRENT_ENGINEERED_FEATURE_COLUMNS}")
    print(f"Model candidate features ({len(CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS)}): {CURRENT_MODEL_CANDIDATE_FEATURE_COLUMNS}")
    print(f"Legacy engineered features removed ({len(LEGACY_ENGINEERED_FEATURE_COLUMNS)}): {LEGACY_ENGINEERED_FEATURE_COLUMNS}")


if __name__ == "__main__":
    print_current_feature_schema()
