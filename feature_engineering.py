"""Leakage-safe aggregation and rolling feature engineering for player-game HR prediction."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd

from config import AB_EVENTS, PA_ENDING_EVENTS, PARKS

SPRAY_CENTER_X = 125.42
SPRAY_HOME_Y = 198.27


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

    dataset["ballpark"] = dataset["opponent"].map(lambda team: PARKS.get(team, {}).get("ballpark"))
    home_mask = dataset["is_home"].astype(bool)
    dataset.loc[home_mask, "ballpark"] = dataset.loc[home_mask, "team"].map(lambda team: PARKS.get(team, {}).get("ballpark"))
    dataset["park_factor_hr"] = np.nan
    dataset = dataset.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
    return dataset, pitcher_game


def add_leakage_safe_features(player_game_df: pd.DataFrame, pitcher_game_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling batter and pitcher features using only pre-game information."""
    df = player_game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    pitcher_game_df = pitcher_game_df.copy()
    pitcher_game_df["game_date"] = pd.to_datetime(pitcher_game_df["game_date"])

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

    df["platoon_advantage"] = np.where(
        df["bat_side"].notna() & df["pitch_hand_primary"].notna(),
        (df["bat_side"] != df["pitch_hand_primary"]).astype(float),
        np.nan,
    )
    df["starter_or_bullpen_proxy"] = np.where(df["opp_pitcher_bf"] >= 12, "starter_like", "bullpen_like")
    return df


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
    # Leakage check: for each player's very first game appearance, the 30-day rolling
    # window uses closed="left", so it must have seen zero prior games and should be NaN.
    # If it is non-NaN for any player's first game, same-day data leaked into the window.
    first_game_date_per_player = df.groupby("player_id")["game_date"].transform("min")
    is_first_game = df["game_date"] == first_game_date_per_player
    if df.loc[is_first_game, "hr_per_pa_last_30d"].notna().any():
        warnings.append(
            "hr_per_pa_last_30d is non-NaN for a player's first game appearance; "
            "the closed='left' rolling window may be including same-day data (leakage)."
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
    pa_df["spray_angle"] = np.degrees(np.arctan2(pa_df["hc_x"] - SPRAY_CENTER_X, SPRAY_HOME_Y - pa_df["hc_y"]))
    pa_df["is_pull_air"] = _is_pull_air(pa_df)
    pa_df["batting_order"] = pa_df.groupby(["game_pk", "team"])["at_bat_number"].rank(method="dense").clip(upper=9)
    return pa_df


def _aggregate_batter_games(pa_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "game_date",
        "game_pk",
        "batter",
        "player_name",
        "team",
        "opponent",
        "is_home",
    ]
    agg = pa_df.groupby(group_cols, dropna=False).agg(
        plate_appearances=("plate_appearance", "sum"),
        at_bats=("at_bat", "sum"),
        hr_count=("is_hr", "sum"),
        bbe_count=("is_bbe", "sum"),
        barrel_count=("is_barrel", "sum"),
        hard_hit_count=("is_hard_hit", "sum"),
        fly_ball_count=("is_fly_ball", "sum"),
        pull_air_count=("is_pull_air", "sum"),
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
        grp["hr_per_pa_last_30d"] = _rolling_day_rate(grp, numerator_col="hr_count", denominator_col="plate_appearances", window="30D")
        grp["recent_form_hr_last_7d"] = _rolling_day_sum(grp, value_col="hr_count", window="7D")
        grp["recent_form_barrels_last_14d"] = _rolling_day_sum(grp, value_col="barrel_count", window="14D")
        grp["expected_pa_proxy"] = _rolling_day_rate(grp, numerator_col="plate_appearances", denominator_col=None, window="14D")
        grp["days_since_last_game"] = grp["game_date"].diff().dt.days.astype(float)
        grp[[
            "barrel_rate_last_50_bbe",
            "hard_hit_rate_last_50_bbe",
            "fly_ball_rate_last_50_bbe",
            "pull_air_rate_last_50_bbe",
            "avg_exit_velocity_last_50_bbe",
            "avg_launch_angle_last_50_bbe",
            "bbe_count_last_50",
        ]] = _count_window_features(
            grp,
            count_col="bbe_count",
            numerators={
                "barrel_rate_last_50_bbe": "barrel_count",
                "hard_hit_rate_last_50_bbe": "hard_hit_count",
                "fly_ball_rate_last_50_bbe": "fly_ball_count",
                "pull_air_rate_last_50_bbe": "pull_air_count",
            },
            weighted_means={
                "avg_exit_velocity_last_50_bbe": "avg_exit_velocity_num",
                "avg_launch_angle_last_50_bbe": "avg_launch_angle_num",
            },
            window_size=50,
        )
        features.append(grp[[
            "player_id",
            "game_date",
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
            "expected_pa_proxy",
            "days_since_last_game",
            "recent_form_hr_last_7d",
            "recent_form_barrels_last_14d",
            "bbe_count_last_50",
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
            "pitcher_bbe_allowed_last_50",
        ]] = _count_window_features(
            grp,
            count_col="pitcher_bbe_allowed",
            numerators={
                "pitcher_barrel_rate_allowed_last_50_bbe": "pitcher_barrel_allowed",
                "pitcher_hard_hit_rate_allowed_last_50_bbe": "pitcher_hard_hit_allowed",
                "pitcher_fb_rate_allowed_last_50_bbe": "pitcher_fb_allowed",
            },
            weighted_means={},
            window_size=50,
        )
        features.append(grp[[
            "pitcher_id",
            "game_date",
            "pitcher_hr9_season_to_date",
            "pitcher_barrel_rate_allowed_last_50_bbe",
            "pitcher_hard_hit_rate_allowed_last_50_bbe",
            "pitcher_fb_rate_allowed_last_50_bbe",
            "pitcher_k_rate_season_to_date",
            "pitcher_bb_rate_season_to_date",
        ]])
    return pd.concat(features, ignore_index=True)


def _rolling_day_sum(grp: pd.DataFrame, value_col: str, window: str) -> pd.Series:
    temp = grp.set_index("game_date")
    return temp[value_col].rolling(window=window, closed="left").sum().reset_index(drop=True)


def _rolling_day_rate(grp: pd.DataFrame, numerator_col: str, denominator_col: str | None, window: str) -> pd.Series:
    temp = grp.set_index("game_date")
    numerator = temp[numerator_col].rolling(window=window, closed="left").sum()
    if denominator_col is None:
        return numerator.reset_index(drop=True)
    denominator = temp[denominator_col].rolling(window=window, closed="left").sum()
    return (numerator / denominator.replace({0: np.nan})).reset_index(drop=True)


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
