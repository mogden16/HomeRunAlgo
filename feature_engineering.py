"""Leakage-safe aggregation and rolling feature engineering for player-game HR prediction."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import AB_EVENTS, FINAL_DATA_PATH, PA_ENDING_EVENTS, PARKS, STATCAST_COLUMNS

SPRAY_CENTER_X = 125.42
SPRAY_HOME_Y = 198.27
PARK_FACTOR_FEATURE = "park_factor_hr"
AUTO_DROP_MISSINGNESS_THRESHOLD = 0.50
DEFAULT_FAIL_MISSINGNESS_THRESHOLD = 0.50
FLAGGED_FEATURES = [
    "expected_pa_proxy",
    "hr_per_pa_last_30d",
    "recent_form_barrels_last_14d",
    "recent_form_hr_last_7d",
]
FEATURE_LINEAGE = {
    "expected_pa_proxy": {
        "sources": ["plate_appearances", "batting_order"],
        "transformation": "Primary path: 14-day shifted rolling sum of prior plate_appearances by player_id. Fallback path: batting-order-based deterministic proxy when rolling history is unavailable.",
        "grouping_keys": ["player_id"],
        "merge_keys": ["player_id", "game_date"],
        "null_rules": "Null before repair when player had no prior game in the last 14 days. Fallback remains null only if both prior batting_order and current batting_order are missing.",
    },
    "hr_per_pa_last_30d": {
        "sources": ["hr_count", "plate_appearances"],
        "transformation": "30-day shifted rolling HR / PA rate using prior daily batter rows only.",
        "grouping_keys": ["player_id"],
        "merge_keys": ["player_id", "game_date"],
        "null_rules": "Null when there is no prior plate appearance in the previous 30 days or denominator is zero.",
    },
    "recent_form_barrels_last_14d": {
        "sources": ["barrel_count"],
        "transformation": "14-day shifted rolling sum of prior barrel_count values.",
        "grouping_keys": ["player_id"],
        "merge_keys": ["player_id", "game_date"],
        "null_rules": "Null when there is no prior game in the previous 14 days.",
    },
    "recent_form_hr_last_7d": {
        "sources": ["hr_count"],
        "transformation": "7-day shifted rolling sum of prior hr_count values.",
        "grouping_keys": ["player_id"],
        "merge_keys": ["player_id", "game_date"],
        "null_rules": "Null when there is no prior game in the previous 7 days.",
    },
}


NEW_FEATURE_MISSINGNESS_THRESHOLD = 0.90
PITCH_TYPE_GROUPS = {
    "fastball": {"FF", "FA", "FT", "FC", "FS", "SI", "SF"},
    "breaking": {"SL", "CU", "KC", "SV", "CS", "KN"},
    "offspeed": {"CH", "FO", "SC", "EP"},
}
PITCH_TYPE_GROUP_LABELS = {
    "fastball": "fastball = FF/FA/FT/FC/FS/SI/SF; breaking = SL/CU/KC/SV/CS/KN; offspeed = CH/FO/SC/EP",
    "breaking": "fastball = FF/FA/FT/FC/FS/SI/SF; breaking = SL/CU/KC/SV/CS/KN; offspeed = CH/FO/SC/EP",
    "offspeed": "fastball = FF/FA/FT/FC/FS/SI/SF; breaking = SL/CU/KC/SV/CS/KN; offspeed = CH/FO/SC/EP",
}
NEW_FEATURE_NOTES = {
    "hr_per_pa_last_30d": "Trailing 30-day batter HR per PA using prior batter-game rows only.",
    "hr_per_pa_last_10d": "Trailing 10-day batter HR per PA using prior batter-game rows only.",
    "hr_count_last_30d": "Trailing 30-day batter HR count using prior batter-game rows only.",
    "hr_count_last_10d": "Trailing 10-day batter HR count using prior batter-game rows only.",
    "pa_last_30d": "Trailing 30-day batter PA total using prior batter-game rows only.",
    "pa_last_10d": "Trailing 10-day batter PA total using prior batter-game rows only.",
    "barrels_per_pa_last_10d": "Trailing 10-day batter barrels per PA from prior games only.",
    "barrels_per_pa_last_30d": "Trailing 30-day batter barrels per PA from prior games only.",
    "barrels_last_10d": "Trailing 10-day batter barrel count from prior games only.",
    "barrels_last_30d": "Trailing 30-day batter barrel count from prior games only.",
    "hard_hit_rate_last_10d": "Trailing 10-day hard-hit / BBE from prior games only.",
    "hard_hit_rate_last_30d": "Trailing 30-day hard-hit / BBE from prior games only.",
    "hard_hit_bbe_last_10d": "Trailing 10-day hard-hit BBE count from prior games only.",
    "hard_hit_bbe_last_30d": "Trailing 30-day hard-hit BBE count from prior games only.",
    "bbe_last_10d": "Trailing 10-day BBE count from prior games only.",
    "bbe_last_30d": "Trailing 30-day BBE count from prior games only.",
    "avg_exit_velocity_last_10d": "Trailing 10-day mean exit velocity over prior BBE only.",
    "avg_exit_velocity_last_30d": "Trailing 30-day mean exit velocity over prior BBE only.",
    "max_exit_velocity_last_10d": "Trailing 10-day max exit velocity over prior BBE only.",
    "bbe_95plus_ev_rate_last_10d": "Trailing 10-day share of prior BBE hit 95+ mph.",
    "bbe_95plus_ev_rate_last_30d": "Trailing 30-day share of prior BBE hit 95+ mph.",
    "pitcher_hr_allowed_per_pa_last_30d": "Trailing 30-day pitcher HR allowed per PA against prior batters only.",
    "pitcher_hr_allowed_last_30d": "Trailing 30-day pitcher HR allowed count against prior batters only.",
    "pitcher_barrels_allowed_per_bbe_last_30d": "Trailing 30-day pitcher barrels allowed per BBE against prior batters only.",
    "pitcher_barrels_allowed_last_30d": "Trailing 30-day pitcher barrels allowed count against prior batters only.",
    "pitcher_hard_hit_allowed_rate_last_30d": "Trailing 30-day pitcher hard-hit allowed / BBE against prior batters only.",
    "pitcher_hard_hit_allowed_last_30d": "Trailing 30-day pitcher hard-hit allowed count against prior batters only.",
    "pitcher_bbe_allowed_last_30d": "Trailing 30-day pitcher BBE allowed count against prior batters only.",
    "pitcher_avg_ev_allowed_last_30d": "Trailing 30-day pitcher average EV allowed over prior BBE only.",
    "pitcher_95plus_ev_allowed_rate_last_30d": "Trailing 30-day pitcher 95+ mph BBE allowed rate over prior BBE only.",
    "pitcher_fastball_pct": "Shifted season-to-date pitcher fastball share from prior pitch-level usage.",
    "pitcher_breaking_ball_pct": "Shifted season-to-date pitcher breaking-ball share from prior pitch-level usage.",
    "pitcher_offspeed_pct": "Shifted season-to-date pitcher offspeed share from prior pitch-level usage.",
    "pitcher_four_seam_pct": "Shifted season-to-date pitcher FF share from prior pitch-level usage.",
    "pitcher_sinker_pct": "Shifted season-to-date pitcher SI share from prior pitch-level usage.",
    "pitcher_slider_pct": "Shifted season-to-date pitcher SL share from prior pitch-level usage.",
    "pitcher_curveball_pct": "Shifted season-to-date pitcher CU/KC share from prior pitch-level usage.",
    "pitcher_changeup_pct": "Shifted season-to-date pitcher CH share from prior pitch-level usage.",
    "batter_contact_rate_vs_fastballs": "Shifted season-to-date batter contact / swings versus fastballs from prior pitch-level events.",
    "batter_hard_hit_rate_vs_fastballs": "Shifted season-to-date batter hard-hit / BBE versus fastballs from prior pitch-level events.",
    "batter_barrel_rate_vs_fastballs": "Shifted season-to-date batter barrels / BBE versus fastballs from prior pitch-level events.",
    "batter_slugging_vs_fastballs": "Shifted season-to-date batter total bases / AB-ending fastballs from prior pitch-level events.",
    "batter_avg_ev_vs_fastballs": "Shifted season-to-date batter average EV on BBE versus fastballs from prior pitch-level events.",
    "fastball_matchup_hard_hit": "Pitcher fastball usage multiplied by batter hard-hit rate versus fastballs.",
    "fastball_matchup_barrel": "Pitcher fastball usage multiplied by batter barrel rate versus fastballs.",
    "fastball_matchup_contact": "Pitcher fastball usage multiplied by batter contact rate versus fastballs.",
}
EXPORT_CANDIDATE_NEW_FEATURES = list(NEW_FEATURE_NOTES.keys())


@dataclass
class NewFeatureRegistry:
    coverage_threshold: float = NEW_FEATURE_MISSINGNESS_THRESHOLD

    def __post_init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.warnings: list[str] = []

    def register(self, feature_names: Iterable[str], note: str | None = None) -> None:
        for feature in feature_names:
            self.records.setdefault(feature, {"note": note or NEW_FEATURE_NOTES.get(feature, ""), "included": False, "reason": "not evaluated"})

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[NEW-FEATURES][WARN] {message}")

    def evaluate(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        included: list[str] = []
        skipped: list[str] = []
        total_rows = len(df)
        for feature, meta in self.records.items():
            if feature not in df.columns:
                meta.update({"included": False, "reason": "not created"})
                skipped.append(feature)
                continue
            non_null = int(df[feature].notna().sum())
            missing_pct = float(100.0 * (1 - non_null / total_rows)) if total_rows else 100.0
            meta["non_null"] = non_null
            meta["missing_pct"] = missing_pct
            if missing_pct >= self.coverage_threshold * 100:
                meta.update({"included": False, "reason": f"skipped due to {missing_pct:.2f}% missingness"})
                skipped.append(feature)
            else:
                meta.update({"included": True, "reason": "coverage acceptable"})
                included.append(feature)
        return included, skipped

def build_player_game_dataset(
    statcast_df: pd.DataFrame,
    debug_feature_audit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate Statcast pitch-level data into one batter-game row."""
    _print_feature_lineage(debug_feature_audit)
    audit_ctx = FeatureAuditContext(enabled=debug_feature_audit, mode="rebuild")
    _summarize_feature_missingness(statcast_df, FLAGGED_FEATURES, "raw source load", audit_ctx, batter_col="batter", pitcher_col="pitcher")

    pa_df = _extract_plate_appearances(statcast_df)
    if pa_df.empty:
        raise RuntimeError("No plate appearances were derived from Statcast input.")
    _summarize_feature_missingness(pa_df, FLAGGED_FEATURES, "cleaning / normalization of player identifiers and dates", audit_ctx, batter_col="batter", pitcher_col="pitcher")

    player_game = _aggregate_batter_games(pa_df)
    player_game = _merge_optional_event_level_features(player_game, _aggregate_batter_pitch_type_game_features(statcast_df), ["game_pk", "player_id"], "batter pitch-type game features")
    _summarize_feature_missingness(player_game, FLAGGED_FEATURES, "grouped batter-game aggregation", audit_ctx, batter_col="player_id")

    pitcher_game = _aggregate_pitcher_games(pa_df)
    pitcher_game = _merge_optional_event_level_features(pitcher_game, _aggregate_pitcher_pitch_type_game_features(statcast_df), ["game_pk", "pitcher"], "pitcher pitch-type game features")
    primary_pitchers = _select_primary_pitchers(pitcher_game)
    _summarize_feature_missingness(primary_pitchers, FLAGGED_FEATURES, "grouped pitcher aggregation", audit_ctx, pitcher_col="opp_pitcher_id")

    dataset = _merge_with_audit(
        player_game,
        primary_pitchers,
        how="left",
        on=["game_pk", "team", "opponent"],
        validate="many_to_one",
        step_name="merge primary pitcher onto batter-game table",
        audit_ctx=audit_ctx,
        uniqueness_expected=True,
    )

    dataset["ballpark"] = dataset["opponent"].map(lambda team: PARKS.get(team, {}).get("ballpark"))
    home_mask = dataset["is_home"].astype(bool)
    dataset.loc[home_mask, "ballpark"] = dataset.loc[home_mask, "team"].map(lambda team: PARKS.get(team, {}).get("ballpark"))
    dataset["park_factor_hr"] = np.nan
    dataset = dataset.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
    _summarize_feature_missingness(dataset, FLAGGED_FEATURES, "post primary-pitcher join batter-game table", audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")
    return dataset, pitcher_game


class FeatureAuditContext:
    def __init__(self, enabled: bool = False, mode: str = "auto"):
        self.enabled = enabled
        self.mode = mode
        self.feature_statuses = {
            feature: {"status": "pending", "explanation": "Not yet audited.", "final_missing_pct": None}
            for feature in FLAGGED_FEATURES
        }
        self.lineage_events: dict[str, list[str]] = {feature: [] for feature in FLAGGED_FEATURES}

    def log(self, message: str) -> None:
        if self.enabled:
            print(message)

    def note(self, feature: str, message: str) -> None:
        self.lineage_events.setdefault(feature, []).append(message)
        self.log(f"[FEATURE-AUDIT] {feature}: {message}")

    def set_status(self, feature: str, status: str, explanation: str, final_missing_pct: float | None = None) -> None:
        self.feature_statuses[feature] = {
            "status": status,
            "explanation": explanation,
            "final_missing_pct": final_missing_pct,
        }


def add_leakage_safe_features(
    player_game_df: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    debug_feature_audit: bool = False,
    missingness_threshold: float = AUTO_DROP_MISSINGNESS_THRESHOLD,
) -> pd.DataFrame:
    """Add rolling batter and pitcher features using only pre-game information."""
    audit_ctx = FeatureAuditContext(enabled=debug_feature_audit, mode="rebuild")
    registry = NewFeatureRegistry()
    registry.register(EXPORT_CANDIDATE_NEW_FEATURES)
    df = player_game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    pitcher_game_df = pitcher_game_df.copy()
    pitcher_game_df["game_date"] = pd.to_datetime(pitcher_game_df["game_date"])
    _summarize_feature_missingness(df, FLAGGED_FEATURES, "feature pipeline start", audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")
    _audit_identifier_columns(df, audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")

    batter_daily = _build_batter_daily(df)
    _summarize_feature_missingness(batter_daily, FLAGGED_FEATURES, "daily batter rollup before rolling stats", audit_ctx, batter_col="player_id")
    batter_features = _compute_batter_daily_features(batter_daily, audit_ctx=audit_ctx)
    _summarize_feature_missingness(batter_features, FLAGGED_FEATURES, "rolling-stat calculations before merge-back", audit_ctx, batter_col="player_id")
    df = _merge_with_audit(
        df,
        batter_features,
        how="left",
        on=["player_id", "game_date"],
        validate="many_to_one",
        step_name="merge batter rolling features back to batter-game table",
        audit_ctx=audit_ctx,
        tracked_features=FLAGGED_FEATURES,
        uniqueness_expected=True,
    )
    df = df.drop(columns=["expected_pa_proxy_raw", "expected_pa_proxy_fallback"], errors="ignore")
    _summarize_feature_missingness(df, FLAGGED_FEATURES, "post batter rolling-feature merge", audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")

    pitcher_daily = _build_pitcher_daily(pitcher_game_df)
    pitcher_features = _compute_pitcher_daily_features(pitcher_daily)
    _summarize_feature_missingness(pitcher_features, FLAGGED_FEATURES, "pitcher rolling-stat calculations", audit_ctx, pitcher_col="pitcher_id")
    df = _merge_with_audit(
        df,
        pitcher_features,
        how="left",
        left_on=["opp_pitcher_id", "game_date"],
        right_on=["pitcher_id", "game_date"],
        validate="many_to_one",
        step_name="merge pitcher rolling features back to batter-game table",
        audit_ctx=audit_ctx,
        tracked_features=[],
        uniqueness_expected=True,
    ).drop(columns=["pitcher_id"], errors="ignore")

    df["platoon_advantage"] = np.where(
        df["bat_side"].notna() & df["pitch_hand_primary"].notna(),
        (df["bat_side"] != df["pitch_hand_primary"]).astype(float),
        np.nan,
    )
    df["starter_or_bullpen_proxy"] = np.where(df["opp_pitcher_bf"] >= 12, "starter_like", "bullpen_like")
    df = compute_pitch_matchup_interactions(df, registry)

    included_new_features, skipped_new_features = summarize_new_feature_quality(df, registry)
    if skipped_new_features:
        df = df.drop(columns=[feature for feature in skipped_new_features if feature in df.columns], errors="ignore")

    df, dropped_features = _drop_unreliable_flagged_features(df, audit_ctx, missingness_threshold=missingness_threshold)
    _summarize_feature_missingness(df, FLAGGED_FEATURES, "final selected feature output", audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")
    _final_feature_quality_report(df, audit_ctx, dropped_features)
    print("\n[NEW-FEATURES] Verdict:")
    print(f"[NEW-FEATURES] added={included_new_features}")
    print(f"[NEW-FEATURES] skipped={skipped_new_features}")
    ready = "yes" if included_new_features else "no"
    print(f"[NEW-FEATURES] engineered dataset ready to rerun train_model.py: {ready}")
    return df




def audit_existing_engineered_dataset(
    df: pd.DataFrame,
    debug_feature_audit: bool = False,
    missingness_threshold: float = AUTO_DROP_MISSINGNESS_THRESHOLD,
) -> pd.DataFrame:
    """Audit an already-engineered batter-game dataset without rebuilding from raw Statcast rows."""
    audit_ctx = FeatureAuditContext(enabled=debug_feature_audit, mode="audit")
    if "game_date" in df.columns:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    _print_feature_lineage(debug_feature_audit)
    audit_ctx.log("[FEATURE-AUDIT] Input looks like an already-engineered batter-game dataset; skipping raw Statcast aggregation and running dataset-level audits only.")
    _summarize_feature_missingness(df, FLAGGED_FEATURES, "engineered dataset input", audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")
    _audit_identifier_columns(df, audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")
    df, dropped_features = _drop_unreliable_flagged_features(df, audit_ctx, missingness_threshold=missingness_threshold)
    _summarize_feature_missingness(df, FLAGGED_FEATURES, "final selected feature output", audit_ctx, batter_col="player_id", pitcher_col="opp_pitcher_id")
    _final_feature_quality_report(df, audit_ctx, dropped_features)
    return df


def _classify_input_dataframe(df: pd.DataFrame) -> tuple[str, list[str]]:
    raw_markers = [column for column in ["at_bat_number", "pitch_number", "events", "inning_topbot"] if column in df.columns]
    engineered_markers = [column for column in ["game_pk", "player_id", "hit_hr"] if column in df.columns]
    if len(raw_markers) == 4:
        return "raw_statcast", raw_markers
    if len(engineered_markers) == 3:
        return "engineered_dataset", engineered_markers
    return "unknown", raw_markers + engineered_markers


def _print_input_classification(input_path: str, dataset_kind: str, trigger_columns: list[str], mode: str) -> None:
    print(f"[FEATURE-AUDIT] Reading input file: {input_path}")
    print(f"[FEATURE-AUDIT] Requested mode: {mode}")
    print(f"[FEATURE-AUDIT] Classified input as: {dataset_kind}")
    print(f"[FEATURE-AUDIT] Trigger columns for classification: {trigger_columns}")


def _assert_missingness_threshold(df: pd.DataFrame, threshold: float) -> None:
    feature_columns = [column for column in FLAGGED_FEATURES + [PARK_FACTOR_FEATURE] if column in df.columns]
    offenders = []
    for feature in feature_columns:
        missing_pct = float(df[feature].isna().mean() * 100)
        if missing_pct > threshold * 100:
            offenders.append(f"{feature}={missing_pct:.2f}%")
    if offenders:
        raise ValueError(
            f"Missingness threshold exceeded ({threshold:.0%}) for selected features: {', '.join(offenders)}"
        )


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
    first_game_date_per_player = df.groupby("player_id")["game_date"].transform("min")
    is_first_game = df["game_date"] == first_game_date_per_player
    if "hr_per_pa_last_30d" in df.columns and df.loc[is_first_game, "hr_per_pa_last_30d"].notna().any():
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
    pa_df["is_barrel"] = _is_barrel(pa_df["launch_speed"], pa_df["launch_angle"]).astype(int)
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
        "stand",
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
        max_exit_velocity_game=("launch_speed", "max"),
        batter_k_count=("is_k", "sum"),
        batter_bb_count=("is_bb", "sum"),
        batting_order=("batting_order", "min"),
    ).reset_index()
    agg = agg.rename(columns={"batter": "player_id", "stand": "bat_side"})
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
        pitcher_avg_ev_allowed_game=("launch_speed", "mean"),
        pitcher_95plus_ev_allowed=("is_hard_hit", "sum"),
    ).reset_index()
    agg["innings_pitched_est"] = agg["outs_recorded"] / 3.0
    return agg


def _merge_optional_event_level_features(base_df: pd.DataFrame, extra_df: pd.DataFrame, keys: list[str], label: str) -> pd.DataFrame:
    if extra_df.empty:
        print(f"[NEW-FEATURES][WARN] No {label} were created from source data; continuing without them.")
        return base_df
    overlap = [column for column in extra_df.columns if column in base_df.columns and column not in keys]
    if overlap:
        extra_df = extra_df.drop(columns=overlap)
    return base_df.merge(extra_df, how="left", on=keys, validate="one_to_one")


def _find_pitch_type_column(df: pd.DataFrame) -> str | None:
    for column in ["pitch_type", "pitch_name"]:
        if column in df.columns:
            return column
    return None


def _normalize_pitch_classes(series: pd.Series) -> pd.Series:
    codes = series.astype(str).str.upper().str.strip()
    pitch_class = pd.Series(pd.NA, index=series.index, dtype="object")
    for label, codes_set in PITCH_TYPE_GROUPS.items():
        pitch_class = pitch_class.where(~codes.isin(codes_set), label)
    return pitch_class


def _is_swing(description: pd.Series) -> pd.Series:
    desc = description.fillna("").astype(str)
    swing_tokens = ["swinging_strike", "foul", "hit_into_play", "foul_tip", "missed_bunt", "foul_bunt"]
    return desc.apply(lambda value: any(token in value for token in swing_tokens))


def _is_contact(description: pd.Series) -> pd.Series:
    desc = description.fillna("").astype(str)
    contact_tokens = ["foul", "hit_into_play", "foul_tip", "foul_bunt"]
    return desc.apply(lambda value: any(token in value for token in contact_tokens))


def _total_bases_from_events(events: pd.Series) -> pd.Series:
    mapping = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    return events.map(mapping).fillna(0).astype(float)


def validate_pitch_type_feature_availability(statcast_df: pd.DataFrame) -> dict[str, object]:
    pitch_type_col = _find_pitch_type_column(statcast_df)
    availability = {
        "pitch_type_col": pitch_type_col,
        "has_pitch_type": pitch_type_col is not None,
        "has_batter_id": "batter" in statcast_df.columns,
        "has_pitcher_id": "pitcher" in statcast_df.columns,
        "has_description": "description" in statcast_df.columns,
        "has_events": "events" in statcast_df.columns,
        "has_ev": "launch_speed" in statcast_df.columns,
        "has_launch_angle": "launch_angle" in statcast_df.columns,
    }
    availability["supported"] = all(availability[key] for key in ["has_pitch_type", "has_batter_id", "has_pitcher_id", "has_description", "has_events", "has_ev", "has_launch_angle"])
    if not availability["supported"]:
        missing = [key for key, value in availability.items() if key.startswith("has_") and not value]
        print(f"[NEW-FEATURES][WARN] Pitch-type features skipped because required raw fields are unavailable: {missing}")
    else:
        print(f"[NEW-FEATURES] Pitch-type feature source detected via column '{pitch_type_col}'.")
    return availability


def _aggregate_batter_pitch_type_game_features(statcast_df: pd.DataFrame) -> pd.DataFrame:
    availability = validate_pitch_type_feature_availability(statcast_df)
    if not availability["supported"]:
        return pd.DataFrame()
    pitch_type_col = availability["pitch_type_col"]
    pitch_df = statcast_df.copy()
    pitch_df["pitch_class"] = _normalize_pitch_classes(pitch_df[pitch_type_col])
    pitch_df = pitch_df[pitch_df["pitch_class"].notna()].copy()
    if pitch_df.empty:
        print("[NEW-FEATURES][WARN] Pitch-type column exists, but no rows mapped into fastball/breaking/offspeed buckets.")
        return pd.DataFrame()
    pitch_df["is_swing"] = _is_swing(pitch_df["description"]).astype(int)
    pitch_df["is_contact"] = _is_contact(pitch_df["description"]).astype(int)
    pitch_df["is_bbe"] = pitch_df["launch_speed"].notna().astype(int)
    pitch_df["is_hard_hit"] = (pitch_df["launch_speed"] >= 95).fillna(False).astype(int)
    pitch_df["is_barrel"] = _is_barrel(pitch_df["launch_speed"], pitch_df["launch_angle"]).astype(int)
    pitch_df["ab_ending_fastball"] = ((pitch_df["events"].isin(AB_EVENTS)) & (pitch_df["pitch_class"] == "fastball")).astype(int)
    pitch_df["tb_fastball"] = np.where(pitch_df["pitch_class"] == "fastball", _total_bases_from_events(pitch_df["events"]), 0.0)
    pitch_df["fastball_bbe_ev_sum"] = np.where((pitch_df["pitch_class"] == "fastball") & pitch_df["launch_speed"].notna(), pitch_df["launch_speed"], 0.0)
    out = pitch_df.groupby(["game_date", "game_pk", "batter"], dropna=False).agg(
        fastball_swings=("is_swing", lambda s: float(s[pitch_df.loc[s.index, "pitch_class"] == "fastball"].sum())),
        fastball_contact=("is_contact", lambda s: float(s[pitch_df.loc[s.index, "pitch_class"] == "fastball"].sum())),
        fastball_bbe=("is_bbe", lambda s: float(s[pitch_df.loc[s.index, "pitch_class"] == "fastball"].sum())),
        fastball_hard_hit=("is_hard_hit", lambda s: float(s[pitch_df.loc[s.index, "pitch_class"] == "fastball"].sum())),
        fastball_barrels=("is_barrel", lambda s: float(s[pitch_df.loc[s.index, "pitch_class"] == "fastball"].sum())),
        fastball_total_bases=("tb_fastball", "sum"),
        fastball_ab=("ab_ending_fastball", "sum"),
        fastball_ev_sum=("fastball_bbe_ev_sum", "sum"),
    ).reset_index().rename(columns={"batter": "player_id"})
    return out


def _aggregate_pitcher_pitch_type_game_features(statcast_df: pd.DataFrame) -> pd.DataFrame:
    availability = validate_pitch_type_feature_availability(statcast_df)
    if not availability["supported"]:
        return pd.DataFrame()
    pitch_type_col = availability["pitch_type_col"]
    pitch_df = statcast_df.copy()
    pitch_df["pitch_class"] = _normalize_pitch_classes(pitch_df[pitch_type_col])
    pitch_df = pitch_df[pitch_df["pitch_class"].notna()].copy()
    if pitch_df.empty:
        return pd.DataFrame()
    pitch_df["pitch_count"] = 1
    pitch_df["is_four_seam"] = pitch_df[pitch_type_col].astype(str).str.upper().eq("FF").astype(int)
    pitch_df["is_sinker"] = pitch_df[pitch_type_col].astype(str).str.upper().eq("SI").astype(int)
    pitch_df["is_slider"] = pitch_df[pitch_type_col].astype(str).str.upper().eq("SL").astype(int)
    pitch_df["is_curveball"] = pitch_df[pitch_type_col].astype(str).str.upper().isin(["CU", "KC"]).astype(int)
    pitch_df["is_changeup"] = pitch_df[pitch_type_col].astype(str).str.upper().eq("CH").astype(int)
    out = pitch_df.groupby(["game_date", "game_pk", "pitcher"], dropna=False).agg(
        pitch_count=("pitch_count", "sum"),
        fastball_pitch_count=("pitch_class", lambda s: float((s == "fastball").sum())),
        breaking_pitch_count=("pitch_class", lambda s: float((s == "breaking").sum())),
        offspeed_pitch_count=("pitch_class", lambda s: float((s == "offspeed").sum())),
        four_seam_pitch_count=("is_four_seam", "sum"),
        sinker_pitch_count=("is_sinker", "sum"),
        slider_pitch_count=("is_slider", "sum"),
        curveball_pitch_count=("is_curveball", "sum"),
        changeup_pitch_count=("is_changeup", "sum"),
    ).reset_index()
    return out


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
    optional_sum_columns = [
        "hard_hit_95plus_count",
        "fastball_swings",
        "fastball_contact",
        "fastball_bbe",
        "fastball_hard_hit",
        "fastball_barrels",
        "fastball_total_bases",
        "fastball_ab",
        "fastball_ev_sum",
    ]
    for column in optional_sum_columns:
        if column not in enriched.columns:
            enriched[column] = 0.0
    daily = enriched.groupby(["player_id", "game_date"], dropna=False).agg(
        plate_appearances=("plate_appearances", "sum"),
        hr_count=("hr_count", "sum"),
        bbe_count=("bbe_count", "sum"),
        barrel_count=("barrel_count", "sum"),
        hard_hit_count=("hard_hit_count", "sum"),
        hard_hit_95plus_count=("hard_hit_count", "sum"),
        fly_ball_count=("fly_ball_count", "sum"),
        pull_air_count=("pull_air_count", "sum"),
        batter_k_count=("batter_k_count", "sum"),
        batter_bb_count=("batter_bb_count", "sum"),
        avg_exit_velocity_num=("avg_exit_velocity_num", "sum"),
        avg_launch_angle_num=("avg_launch_angle_num", "sum"),
        batting_order=("batting_order", "min"),
        fastball_swings=("fastball_swings", "sum"),
        fastball_contact=("fastball_contact", "sum"),
        fastball_bbe=("fastball_bbe", "sum"),
        fastball_hard_hit=("fastball_hard_hit", "sum"),
        fastball_barrels=("fastball_barrels", "sum"),
        fastball_total_bases=("fastball_total_bases", "sum"),
        fastball_ab=("fastball_ab", "sum"),
        fastball_ev_sum=("fastball_ev_sum", "sum"),
        max_exit_velocity_game=("max_exit_velocity_game", "max"),
    ).reset_index()
    return daily.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def compute_batter_trailing_hr_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp["hr_count_last_30d"] = _rolling_day_sum(grp, value_col="hr_count", window="30D")
    grp["hr_count_last_10d"] = _rolling_day_sum(grp, value_col="hr_count", window="10D")
    grp["pa_last_30d"] = _rolling_day_sum(grp, value_col="plate_appearances", window="30D")
    grp["pa_last_10d"] = _rolling_day_sum(grp, value_col="plate_appearances", window="10D")
    grp["hr_per_pa_last_30d"] = grp["hr_count_last_30d"] / grp["pa_last_30d"].replace({0: np.nan})
    grp["hr_per_pa_last_10d"] = grp["hr_count_last_10d"] / grp["pa_last_10d"].replace({0: np.nan})
    return grp


def compute_batter_recent_contact_quality_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp["barrels_last_10d"] = _rolling_day_sum(grp, value_col="barrel_count", window="10D")
    grp["barrels_last_30d"] = _rolling_day_sum(grp, value_col="barrel_count", window="30D")
    grp["bbe_last_10d"] = _rolling_day_sum(grp, value_col="bbe_count", window="10D")
    grp["bbe_last_30d"] = _rolling_day_sum(grp, value_col="bbe_count", window="30D")
    grp["hard_hit_bbe_last_10d"] = _rolling_day_sum(grp, value_col="hard_hit_count", window="10D")
    grp["hard_hit_bbe_last_30d"] = _rolling_day_sum(grp, value_col="hard_hit_count", window="30D")
    grp["barrels_per_pa_last_10d"] = grp["barrels_last_10d"] / grp["pa_last_10d"].replace({0: np.nan})
    grp["barrels_per_pa_last_30d"] = grp["barrels_last_30d"] / grp["pa_last_30d"].replace({0: np.nan})
    grp["hard_hit_rate_last_10d"] = grp["hard_hit_bbe_last_10d"] / grp["bbe_last_10d"].replace({0: np.nan})
    grp["hard_hit_rate_last_30d"] = grp["hard_hit_bbe_last_30d"] / grp["bbe_last_30d"].replace({0: np.nan})
    grp["avg_exit_velocity_last_10d"] = _rolling_day_weighted_mean(grp, value_num_col="avg_exit_velocity_num", weight_col="bbe_count", window="10D")
    grp["avg_exit_velocity_last_30d"] = _rolling_day_weighted_mean(grp, value_num_col="avg_exit_velocity_num", weight_col="bbe_count", window="30D")
    grp["max_exit_velocity_last_10d"] = _rolling_day_max(grp, value_col="max_exit_velocity_game", window="10D")
    grp["bbe_95plus_ev_rate_last_10d"] = grp["hard_hit_bbe_last_10d"] / grp["bbe_last_10d"].replace({0: np.nan})
    grp["bbe_95plus_ev_rate_last_30d"] = grp["hard_hit_bbe_last_30d"] / grp["bbe_last_30d"].replace({0: np.nan})
    return grp


def compute_batter_pitch_type_split_features(grp: pd.DataFrame) -> pd.DataFrame:
    fastball_cols = ["fastball_swings", "fastball_contact", "fastball_bbe", "fastball_hard_hit", "fastball_barrels", "fastball_total_bases", "fastball_ab", "fastball_ev_sum"]
    if not any(column in grp.columns for column in fastball_cols):
        return grp
    for column in fastball_cols:
        if column not in grp.columns:
            grp[column] = 0.0
    grp["batter_contact_rate_vs_fastballs"] = _shifted_cumulative_rate(grp["fastball_contact"], grp["fastball_swings"])
    grp["batter_hard_hit_rate_vs_fastballs"] = _shifted_cumulative_rate(grp["fastball_hard_hit"], grp["fastball_bbe"])
    grp["batter_barrel_rate_vs_fastballs"] = _shifted_cumulative_rate(grp["fastball_barrels"], grp["fastball_bbe"])
    grp["batter_slugging_vs_fastballs"] = _shifted_cumulative_rate(grp["fastball_total_bases"], grp["fastball_ab"])
    grp["batter_avg_ev_vs_fastballs"] = _shifted_cumulative_rate(grp["fastball_ev_sum"], grp["fastball_bbe"])
    return grp


def _compute_batter_daily_features(daily: pd.DataFrame, audit_ctx: FeatureAuditContext | None = None) -> pd.DataFrame:
    features: list[pd.DataFrame] = []
    audit_ctx = audit_ctx or FeatureAuditContext(enabled=False)
    for player_id, group in daily.groupby("player_id", sort=False):
        grp = group.sort_values("game_date").copy()
        if not grp["game_date"].is_monotonic_increasing:
            raise ValueError(f"Batter rolling features require sorted game_date per player_id; found unsorted rows for player {player_id}.")

        grp["hr_rate_season_to_date"] = _shifted_cumulative_rate(grp["hr_count"], grp["plate_appearances"])
        grp["batter_k_rate_season_to_date"] = _shifted_cumulative_rate(grp["batter_k_count"], grp["plate_appearances"])
        grp["batter_bb_rate_season_to_date"] = _shifted_cumulative_rate(grp["batter_bb_count"], grp["plate_appearances"])
        grp["hr_per_pa_last_30d"] = _rolling_day_rate(grp, numerator_col="hr_count", denominator_col="plate_appearances", window="30D")
        grp["recent_form_hr_last_7d"] = _rolling_day_sum(grp, value_col="hr_count", window="7D")
        grp["recent_form_barrels_last_14d"] = _rolling_day_sum(grp, value_col="barrel_count", window="14D")
        grp["expected_pa_proxy_raw"] = _rolling_day_sum(grp, value_col="plate_appearances", window="14D")
        grp["expected_pa_proxy_fallback"] = _expected_pa_fallback(grp)
        grp["expected_pa_proxy"] = grp["expected_pa_proxy_raw"].where(grp["expected_pa_proxy_raw"].notna(), grp["expected_pa_proxy_fallback"])
        grp["days_since_last_game"] = grp["game_date"].diff().dt.days.astype(float)

        grp = compute_batter_trailing_hr_features(grp)
        grp = compute_batter_recent_contact_quality_features(grp)
        grp = compute_batter_pitch_type_split_features(grp)

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
        features.append(grp)
    feature_df = pd.concat(features, ignore_index=True)
    keep_columns = [
        "player_id", "game_date", "hr_rate_season_to_date", "hr_per_pa_last_30d", "barrel_rate_last_50_bbe",
        "hard_hit_rate_last_50_bbe", "avg_launch_angle_last_50_bbe", "avg_exit_velocity_last_50_bbe",
        "fly_ball_rate_last_50_bbe", "pull_air_rate_last_50_bbe", "batter_k_rate_season_to_date",
        "batter_bb_rate_season_to_date", "expected_pa_proxy_raw", "expected_pa_proxy_fallback",
        "expected_pa_proxy", "days_since_last_game", "recent_form_hr_last_7d", "recent_form_barrels_last_14d",
        "bbe_count_last_50"
    ] + [feature for feature in EXPORT_CANDIDATE_NEW_FEATURES if feature in feature_df.columns]
    keep_columns = list(dict.fromkeys(keep_columns))
    _audit_flagged_rolling_features(daily, feature_df, audit_ctx)
    return feature_df[keep_columns]


def _build_pitcher_daily(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "pitcher_avg_ev_allowed_game" not in enriched.columns:
        enriched["pitcher_avg_ev_allowed_game"] = np.nan
    enriched["pitcher_ev_allowed_sum"] = enriched["pitcher_avg_ev_allowed_game"] * enriched["pitcher_bbe_allowed"].fillna(0)
    optional_sum_columns = [
        "pitch_count", "fastball_pitch_count", "breaking_pitch_count", "offspeed_pitch_count",
        "four_seam_pitch_count", "sinker_pitch_count", "slider_pitch_count", "curveball_pitch_count", "changeup_pitch_count",
        "pitcher_95plus_ev_allowed",
    ]
    for column in optional_sum_columns:
        if column not in enriched.columns:
            enriched[column] = 0.0
    daily = enriched.groupby(["pitcher", "game_date"], dropna=False).agg(
        innings_pitched_est=("innings_pitched_est", "sum"),
        pitcher_hr_allowed=("pitcher_hr_allowed", "sum"),
        pitcher_bbe_allowed=("pitcher_bbe_allowed", "sum"),
        pitcher_barrel_allowed=("pitcher_barrel_allowed", "sum"),
        pitcher_hard_hit_allowed=("pitcher_hard_hit_allowed", "sum"),
        pitcher_fb_allowed=("pitcher_fb_allowed", "sum"),
        pitcher_k_count=("pitcher_k_count", "sum"),
        pitcher_bb_count=("pitcher_bb_count", "sum"),
        batters_faced=("batters_faced", "sum"),
        pitch_count=("pitch_count", "sum"),
        fastball_pitch_count=("fastball_pitch_count", "sum"),
        breaking_pitch_count=("breaking_pitch_count", "sum"),
        offspeed_pitch_count=("offspeed_pitch_count", "sum"),
        four_seam_pitch_count=("four_seam_pitch_count", "sum"),
        sinker_pitch_count=("sinker_pitch_count", "sum"),
        slider_pitch_count=("slider_pitch_count", "sum"),
        curveball_pitch_count=("curveball_pitch_count", "sum"),
        changeup_pitch_count=("changeup_pitch_count", "sum"),
        pitcher_ev_allowed_sum=("pitcher_ev_allowed_sum", "sum"),
        pitcher_95plus_ev_allowed=("pitcher_95plus_ev_allowed", "sum"),
    ).reset_index().rename(columns={"pitcher": "pitcher_id"})
    return daily.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def compute_pitcher_recent_contact_allowed_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp["pitcher_hr_allowed_last_30d"] = _rolling_day_sum(grp, value_col="pitcher_hr_allowed", window="30D")
    grp["pitcher_bbe_allowed_last_30d"] = _rolling_day_sum(grp, value_col="pitcher_bbe_allowed", window="30D")
    grp["pitcher_barrels_allowed_last_30d"] = _rolling_day_sum(grp, value_col="pitcher_barrel_allowed", window="30D")
    grp["pitcher_hard_hit_allowed_last_30d"] = _rolling_day_sum(grp, value_col="pitcher_hard_hit_allowed", window="30D")
    grp["pitcher_pa_allowed_last_30d"] = _rolling_day_sum(grp, value_col="batters_faced", window="30D")
    grp["pitcher_hr_allowed_per_pa_last_30d"] = grp["pitcher_hr_allowed_last_30d"] / grp["pitcher_pa_allowed_last_30d"].replace({0: np.nan})
    grp["pitcher_barrels_allowed_per_bbe_last_30d"] = grp["pitcher_barrels_allowed_last_30d"] / grp["pitcher_bbe_allowed_last_30d"].replace({0: np.nan})
    grp["pitcher_hard_hit_allowed_rate_last_30d"] = grp["pitcher_hard_hit_allowed_last_30d"] / grp["pitcher_bbe_allowed_last_30d"].replace({0: np.nan})
    grp["pitcher_avg_ev_allowed_last_30d"] = _rolling_day_weighted_mean(grp, value_num_col="pitcher_ev_allowed_sum", weight_col="pitcher_bbe_allowed", window="30D")
    grp["pitcher_95plus_ev_allowed_rate_last_30d"] = _rolling_day_sum(grp, value_col="pitcher_95plus_ev_allowed", window="30D") / grp["pitcher_bbe_allowed_last_30d"].replace({0: np.nan})
    return grp


def compute_pitcher_pitch_mix_features(grp: pd.DataFrame) -> pd.DataFrame:
    if "pitch_count" not in grp.columns:
        return grp
    grp["pitcher_fastball_pct"] = _shifted_cumulative_rate(grp["fastball_pitch_count"], grp["pitch_count"])
    grp["pitcher_breaking_ball_pct"] = _shifted_cumulative_rate(grp["breaking_pitch_count"], grp["pitch_count"])
    grp["pitcher_offspeed_pct"] = _shifted_cumulative_rate(grp["offspeed_pitch_count"], grp["pitch_count"])
    grp["pitcher_four_seam_pct"] = _shifted_cumulative_rate(grp["four_seam_pitch_count"], grp["pitch_count"])
    grp["pitcher_sinker_pct"] = _shifted_cumulative_rate(grp["sinker_pitch_count"], grp["pitch_count"])
    grp["pitcher_slider_pct"] = _shifted_cumulative_rate(grp["slider_pitch_count"], grp["pitch_count"])
    grp["pitcher_curveball_pct"] = _shifted_cumulative_rate(grp["curveball_pitch_count"], grp["pitch_count"])
    grp["pitcher_changeup_pct"] = _shifted_cumulative_rate(grp["changeup_pitch_count"], grp["pitch_count"])
    return grp


def _compute_pitcher_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    features: list[pd.DataFrame] = []
    for _, group in daily.groupby("pitcher_id", sort=False):
        grp = group.sort_values("game_date").copy()
        shifted_ip = grp["innings_pitched_est"].cumsum().shift(1)
        shifted_hr = grp["pitcher_hr_allowed"].cumsum().shift(1)
        grp["pitcher_hr9_season_to_date"] = np.where(shifted_ip > 0, shifted_hr * 9 / shifted_ip, np.nan)
        grp["pitcher_k_rate_season_to_date"] = _shifted_cumulative_rate(grp["pitcher_k_count"], grp["batters_faced"])
        grp["pitcher_bb_rate_season_to_date"] = _shifted_cumulative_rate(grp["pitcher_bb_count"], grp["batters_faced"])
        grp = compute_pitcher_recent_contact_allowed_features(grp)
        grp = compute_pitcher_pitch_mix_features(grp)
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
        features.append(grp)
    feature_df = pd.concat(features, ignore_index=True)
    keep_columns = [
        "pitcher_id", "game_date", "pitcher_hr9_season_to_date", "pitcher_barrel_rate_allowed_last_50_bbe",
        "pitcher_hard_hit_rate_allowed_last_50_bbe", "pitcher_fb_rate_allowed_last_50_bbe",
        "pitcher_k_rate_season_to_date", "pitcher_bb_rate_season_to_date"
    ] + [feature for feature in EXPORT_CANDIDATE_NEW_FEATURES if feature in feature_df.columns]
    keep_columns = list(dict.fromkeys(keep_columns))
    return feature_df[keep_columns]


def _rolling_day_weighted_mean(grp: pd.DataFrame, value_num_col: str, weight_col: str, window: str) -> pd.Series:
    temp = grp.set_index("game_date")
    numerator = temp[value_num_col].rolling(window=window, closed="left", min_periods=1).sum()
    denominator = temp[weight_col].rolling(window=window, closed="left", min_periods=1).sum()
    return (numerator / denominator.replace({0: np.nan})).reset_index(drop=True)


def _rolling_day_max(grp: pd.DataFrame, value_col: str, window: str) -> pd.Series:
    temp = grp.set_index("game_date")
    return temp[value_col].rolling(window=window, closed="left", min_periods=1).max().reset_index(drop=True)


def compute_pitch_matchup_interactions(df: pd.DataFrame, registry: NewFeatureRegistry | None = None) -> pd.DataFrame:
    interaction_specs = {
        "fastball_matchup_hard_hit": ("pitcher_fastball_pct", "batter_hard_hit_rate_vs_fastballs"),
        "fastball_matchup_barrel": ("pitcher_fastball_pct", "batter_barrel_rate_vs_fastballs"),
        "fastball_matchup_contact": ("pitcher_fastball_pct", "batter_contact_rate_vs_fastballs"),
    }
    for feature, (left_col, right_col) in interaction_specs.items():
        if left_col in df.columns and right_col in df.columns:
            df[feature] = df[left_col] * df[right_col]
        elif registry is not None:
            registry.warn(f"Skipped {feature} because {left_col} and/or {right_col} were unavailable.")
    return df


def summarize_new_feature_quality(df: pd.DataFrame, registry: NewFeatureRegistry) -> tuple[list[str], list[str]]:
    included, skipped = registry.evaluate(df)
    print("\n[NEW-FEATURES] Focused feature-quality summary")
    print("-" * 120)
    print(f"{'feature':<42} {'non_null':>10} {'missing_%':>10} {'included':>10} note")
    for feature in registry.records:
        meta = registry.records[feature]
        print(
            f"{feature:<42} {int(meta.get('non_null', 0)):>10} {float(meta.get('missing_pct', 100.0)):>10.2f} {str(bool(meta.get('included', False))):>10} {meta.get('note', '')}"
        )
    return included, skipped


def _rolling_day_sum(grp: pd.DataFrame, value_col: str, window: str) -> pd.Series:
    temp = grp.set_index("game_date")
    return temp[value_col].rolling(window=window, closed="left", min_periods=1).sum().reset_index(drop=True)


def _rolling_day_rate(grp: pd.DataFrame, numerator_col: str, denominator_col: str | None, window: str) -> pd.Series:
    temp = grp.set_index("game_date")
    numerator = temp[numerator_col].rolling(window=window, closed="left", min_periods=1).sum()
    if denominator_col is None:
        return numerator.reset_index(drop=True)
    denominator = temp[denominator_col].rolling(window=window, closed="left", min_periods=1).sum()
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


def _expected_pa_fallback(grp: pd.DataFrame) -> pd.Series:
    prior_batting_order = grp["batting_order"].shift(1)
    current_batting_order = grp["batting_order"]
    order = prior_batting_order.where(prior_batting_order.notna(), current_batting_order)
    proxy = 4.65 - 0.08 * (order.fillna(9) - 1)
    proxy = proxy.clip(lower=3.6, upper=4.8)
    proxy = proxy.where(order.notna(), np.nan)
    return proxy.astype(float)


def _audit_identifier_columns(df: pd.DataFrame, audit_ctx: FeatureAuditContext, batter_col: str, pitcher_col: str | None = None) -> None:
    if not audit_ctx.enabled:
        return
    audit_ctx.log("[FEATURE-AUDIT] Join key audit: checking identifier normalization, dtypes, and date alignment.")
    if batter_col in df.columns:
        audit_ctx.log(
            f"[FEATURE-AUDIT] batter key={batter_col} dtype={df[batter_col].dtype} missing={int(df[batter_col].isna().sum())} distinct={df[batter_col].nunique(dropna=True)}"
        )
    if pitcher_col and pitcher_col in df.columns:
        audit_ctx.log(
            f"[FEATURE-AUDIT] pitcher key={pitcher_col} dtype={df[pitcher_col].dtype} missing={int(df[pitcher_col].isna().sum())} distinct={df[pitcher_col].nunique(dropna=True)}"
        )
    if "player_name" in df.columns:
        normalized_name_count = int(df["player_name"].astype(str).str.strip().str.lower().nunique(dropna=True))
        audit_ctx.log(f"[FEATURE-AUDIT] player_name normalization audit: normalized distinct names={normalized_name_count}. Name-based joins are not used for flagged features.")
    if "game_date" in df.columns:
        audit_ctx.log(f"[FEATURE-AUDIT] game_date dtype={df['game_date'].dtype}; flagged-feature joins use game_date only (no separate event_date key).")


def _audit_flagged_rolling_features(daily: pd.DataFrame, feature_df: pd.DataFrame, audit_ctx: FeatureAuditContext) -> None:
    if not audit_ctx.enabled:
        return
    joined = daily.merge(feature_df[["player_id", "game_date"] + FLAGGED_FEATURES + ["expected_pa_proxy_raw", "expected_pa_proxy_fallback"]], on=["player_id", "game_date"], how="left", validate="one_to_one")
    total_rows = len(joined)
    specs = {
        "hr_per_pa_last_30d": {"window": 30, "source_cols": ["hr_count", "plate_appearances"], "empty_window": "no prior PA in 30-day lookback or denominator zero"},
        "recent_form_hr_last_7d": {"window": 7, "source_cols": ["hr_count"], "empty_window": "no prior game in 7-day lookback"},
        "recent_form_barrels_last_14d": {"window": 14, "source_cols": ["barrel_count"], "empty_window": "no prior game in 14-day lookback"},
        "expected_pa_proxy": {"window": 14, "source_cols": ["plate_appearances", "batting_order"], "empty_window": "no prior PA in 14-day lookback before batting-order fallback"},
    }
    for feature, spec in specs.items():
        groups = daily.sort_values(["player_id", "game_date"]).groupby("player_id", sort=False)
        eligible = 0
        source_missing = 0
        insufficient_history = 0
        for _, grp in groups:
            dates = grp["game_date"]
            for idx in range(len(grp)):
                prior = grp.iloc[:idx]
                if prior.empty:
                    insufficient_history += 1
                    continue
                days = (dates.iloc[idx] - prior["game_date"]).dt.days
                within_window = prior.loc[(days >= 0) & (days <= spec["window"])].copy()
                if within_window.empty:
                    insufficient_history += 1
                    continue
                source_available = within_window[spec["source_cols"]].notna().all(axis=1).any()
                if source_available:
                    eligible += 1
                else:
                    source_missing += 1
        valid = int(joined[feature].notna().sum()) if feature in joined.columns else 0
        merge_failure = max(eligible - valid, 0) if feature != "expected_pa_proxy" else 0
        audit_ctx.note(
            feature,
            f"Rolling audit -> eligible={eligible:,}, valid={valid:,}, insufficient_history={insufficient_history:,}, source_missing={source_missing:,}, merge_failure_after_calc={merge_failure:,}.",
        )
        if feature == "expected_pa_proxy":
            fallback_count = int((joined["expected_pa_proxy_raw"].isna() & joined["expected_pa_proxy_fallback"].notna()).sum())
            audit_ctx.note(feature, f"expected_pa_proxy fallback rows supplied by batting-order proxy={fallback_count:,}.")
            audit_ctx.note(feature, "expected_pa_proxy lineage: raw source does not include projected lineups, so the repair uses prior 14-day PA totals first and a batting-order proxy fallback second.")


def _merge_with_audit(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    how: str,
    step_name: str,
    audit_ctx: FeatureAuditContext,
    tracked_features: Iterable[str] | None = None,
    uniqueness_expected: bool = False,
    **merge_kwargs,
) -> pd.DataFrame:
    tracked_features = list(tracked_features or [])
    if not audit_ctx.enabled:
        return left_df.merge(right_df, how=how, **merge_kwargs)

    merge_keys = merge_kwargs.get("on") or list(zip(merge_kwargs.get("left_on", []), merge_kwargs.get("right_on", [])))
    audit_ctx.log(f"[FEATURE-AUDIT] Merge audit: {step_name}")
    audit_ctx.log(f"[FEATURE-AUDIT] merge keys={merge_keys}")
    audit_ctx.log(f"[FEATURE-AUDIT] left rows={len(left_df):,}, right rows={len(right_df):,}")

    on = merge_kwargs.get("on")
    if on is not None:
        left_keys = on
        right_keys = on
    else:
        left_keys = merge_kwargs.get("left_on", [])
        right_keys = merge_kwargs.get("right_on", [])
    for l_key, r_key in zip(left_keys, right_keys):
        if l_key in left_df.columns and r_key in right_df.columns and left_df[l_key].dtype != right_df[r_key].dtype:
            audit_ctx.log(f"[FEATURE-AUDIT][WARN] dtype mismatch for merge key {l_key}/{r_key}: left={left_df[l_key].dtype}, right={right_df[r_key].dtype}")
    if uniqueness_expected:
        dupes = int(right_df.duplicated(right_keys).sum()) if right_keys else 0
        if dupes:
            audit_ctx.log(f"[FEATURE-AUDIT][WARN] {dupes:,} duplicate right-side merge keys detected where uniqueness was expected.")

    matched_pairs = left_df.merge(
        right_df[right_keys].drop_duplicates(),
        how="left",
        left_on=left_keys,
        right_on=right_keys,
        indicator=True,
    )["_merge"]
    matched = int((matched_pairs == "both").sum())
    unmatched = int((matched_pairs != "both").sum())
    unmatched_rate = unmatched / len(left_df) if len(left_df) else 0.0
    audit_ctx.log(f"[FEATURE-AUDIT] matched rows={matched:,}, unmatched rows={unmatched:,} ({unmatched_rate:.2%})")
    if unmatched_rate > 0.05:
        audit_ctx.log(f"[FEATURE-AUDIT][WARN] High unmatched rate in {step_name}: {unmatched_rate:.2%}")

    merged = left_df.merge(right_df, how=how, indicator=True, **merge_kwargs)
    if tracked_features:
        for feature in tracked_features:
            if feature in merged.columns:
                valid = int(merged[feature].notna().sum())
                audit_ctx.log(f"[FEATURE-AUDIT] {feature} non-null after {step_name}: {valid:,} / {len(merged):,}")
    merged = merged.drop(columns=["_merge"])
    return merged


def _drop_unreliable_flagged_features(
    df: pd.DataFrame,
    audit_ctx: FeatureAuditContext,
    missingness_threshold: float = AUTO_DROP_MISSINGNESS_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    for feature in FLAGGED_FEATURES:
        if feature not in df.columns:
            audit_ctx.set_status(feature, "dropped", "Feature is absent from the engineered dataset.", 100.0)
            dropped.append(feature)
            continue
        missing_pct = float(df[feature].isna().mean() * 100)
        if feature == "expected_pa_proxy":
            if missing_pct > missingness_threshold * 100:
                df = df.drop(columns=[feature])
                dropped.append(feature)
                audit_ctx.set_status(
                    feature,
                    "dropped",
                    f"Dropped because the two-stage proxy still left {missing_pct:.2f}% missingness, above the allowed threshold.",
                    missing_pct,
                )
                audit_ctx.log(f"[FEATURE-AUDIT][WARN] Dropping {feature} because the repaired proxy still left {missing_pct:.2f}% missingness.")
            else:
                audit_ctx.set_status(
                    feature,
                    "simplified",
                    "Repaired with a two-stage proxy: prior 14-day plate appearances, then batting-order fallback when recent history is unavailable.",
                    missing_pct,
                )
            continue
        audit_ctx.set_status(
            feature,
            "repaired",
            "Shifted rolling window retained; min_periods=1 now preserves valid prior-history windows without introducing leakage.",
            missing_pct,
        )
        if missing_pct > missingness_threshold * 100:
            df = df.drop(columns=[feature])
            dropped.append(feature)
            audit_ctx.set_status(
                feature,
                "dropped",
                f"Dropped because missingness remained {missing_pct:.2f}% after audit, indicating unreliable source coverage or merge alignment.",
                missing_pct,
            )
            audit_ctx.log(f"[FEATURE-AUDIT][WARN] Dropping {feature} from final dataset because missingness remained {missing_pct:.2f}%.")

    if PARK_FACTOR_FEATURE in df.columns:
        park_missing_pct = float(df[PARK_FACTOR_FEATURE].isna().mean() * 100)
        if park_missing_pct >= 100.0:
            df = df.drop(columns=[PARK_FACTOR_FEATURE])
            dropped.append(PARK_FACTOR_FEATURE)
            audit_ctx.log(f"[FEATURE-AUDIT][WARN] Dropping {PARK_FACTOR_FEATURE} because it is 100% missing and no source-backed rebuild exists in this pipeline.")
    return df, dropped


def _summarize_feature_missingness(
    df: pd.DataFrame,
    feature_names: Iterable[str],
    step_name: str,
    audit_ctx: FeatureAuditContext,
    batter_col: str | None = None,
    pitcher_col: str | None = None,
) -> None:
    if not audit_ctx.enabled:
        return
    row_count = len(df)
    batter_count = df[batter_col].nunique(dropna=True) if batter_col and batter_col in df.columns else "n/a"
    pitcher_count = df[pitcher_col].nunique(dropna=True) if pitcher_col and pitcher_col in df.columns else "n/a"
    audit_ctx.log(f"\n[FEATURE-AUDIT] Step: {step_name}")
    audit_ctx.log(f"[FEATURE-AUDIT] rows={row_count:,}, distinct batters={batter_count}, distinct pitchers={pitcher_count}")
    for feature in feature_names:
        if feature in df.columns:
            non_null = int(df[feature].notna().sum())
            missing_pct = (1 - non_null / row_count) * 100 if row_count else 0.0
            audit_ctx.log(f"[FEATURE-AUDIT] {feature}: non_null={non_null:,}, missing_pct={missing_pct:.2f}%")
        else:
            audit_ctx.log(f"[FEATURE-AUDIT] {feature}: column not present at this step")


def _print_feature_lineage(enabled: bool) -> None:
    if not enabled:
        return
    print("[FEATURE-AUDIT] Flagged feature lineage overview:")
    for feature, lineage in FEATURE_LINEAGE.items():
        print(f"  - {feature}: sources={lineage['sources']}; grouping_keys={lineage['grouping_keys']}; merge_keys={lineage['merge_keys']}")
        print(f"    transformation={lineage['transformation']}")
        print(f"    null-risk={lineage['null_rules']}")


def _final_feature_quality_report(df: pd.DataFrame, audit_ctx: FeatureAuditContext, dropped_features: list[str]) -> None:
    if not audit_ctx.enabled:
        return
    numeric_or_engineered = [col for col in df.columns if col not in {"player_name", "opp_pitcher_name", "starter_or_bullpen_proxy", "ballpark"}]
    report_rows = []
    for col in numeric_or_engineered:
        non_null = int(df[col].notna().sum())
        missing_pct = float(df[col].isna().mean() * 100)
        source_note = "flagged feature" if col in FLAGGED_FEATURES else ""
        report_rows.append((col, non_null, missing_pct, source_note))
    report_rows.sort(key=lambda row: (-row[2], row[0]))
    audit_ctx.log("\n[FEATURE-AUDIT] Final feature quality report (worst missingness first):")
    for feature, non_null, missing_pct, source_note in report_rows:
        audit_ctx.log(f"[FEATURE-AUDIT] {feature}: non_null={non_null:,}, missing_pct={missing_pct:.2f}% {source_note}".rstrip())

    audit_ctx.log("\n[FEATURE-AUDIT] Focused flagged-feature verdict:")
    for feature in FLAGGED_FEATURES:
        status = audit_ctx.feature_statuses.get(feature, {})
        final_missing_pct = status.get("final_missing_pct")
        if feature in df.columns:
            final_missing_pct = float(df[feature].isna().mean() * 100)
        audit_ctx.log(
            f"[FEATURE-AUDIT] {feature}: status={status.get('status', 'unknown')}, final_missing_pct={final_missing_pct if final_missing_pct is not None else 'n/a'}, explanation={status.get('explanation', 'n/a')}"
        )
    fixed = [f for f, meta in audit_ctx.feature_statuses.items() if meta["status"] in {"repaired", "simplified"}]
    dropped = dropped_features
    still_high = [f for f in FLAGGED_FEATURES if f in df.columns and df[f].isna().mean() > AUTO_DROP_MISSINGNESS_THRESHOLD]
    verdict = "suitable" if not still_high and not dropped else "needs review"
    audit_ctx.log("\n[FEATURE-AUDIT] Final verdict:")
    audit_ctx.log(f"[FEATURE-AUDIT] fixed_or_simplified={fixed}")
    audit_ctx.log(f"[FEATURE-AUDIT] dropped={dropped}")
    audit_ctx.log(f"[FEATURE-AUDIT] still_high_missingness={still_high}")
    audit_ctx.log(f"[FEATURE-AUDIT] engineered dataset verdict={verdict}; {'re-run train_model.py next.' if verdict == 'suitable' else 'resolve remaining flagged-feature coverage before more tuning.'}")


def _is_barrel(launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    ev = pd.to_numeric(launch_speed, errors="coerce")
    la = pd.to_numeric(launch_angle, errors="coerce")
    min_angle = 26 - ((ev - 98).clip(lower=0) * 0.5)
    max_angle = 30 + ((ev - 98).clip(lower=0) * 0.5)
    is_barrel = ev.ge(98) & la.ge(min_angle) & la.le(max_angle)
    return is_barrel.fillna(False)


def _is_pull_air(pa_df: pd.DataFrame) -> pd.Series:
    air = pa_df["bb_type"].isin(["fly_ball", "line_drive", "popup"])
    right_pull = pa_df["stand"].eq("R") & pa_df["spray_angle"].lt(-15)
    left_pull = pa_df["stand"].eq("L") & pa_df["spray_angle"].gt(15)
    return (air & (right_pull | left_pull)).astype(int)


def _load_input_dataframe(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path, parse_dates=["game_date"], low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", nargs="?", default=str(FINAL_DATA_PATH), help="Path to either a raw Statcast CSV file or an already-engineered batter-game dataset. Defaults to the configured final dataset path.")
    parser.add_argument("--output", default=str(FINAL_DATA_PATH), help="Optional path for writing the engineered dataset or audited engineered copy.")
    parser.add_argument("--mode", choices=["auto", "rebuild", "audit"], default="auto", help="Execution mode: rebuild from raw data, audit an existing engineered dataset, or auto-detect.")
    parser.add_argument("--debug-feature-audit", action="store_true", help="Print step-by-step lineage, merge, and missingness diagnostics for the four flagged features.")
    parser.add_argument("--fail-on-high-missingness", action="store_true", help="Fail if selected tracked features exceed the configured missingness threshold after rebuild/audit.")
    parser.add_argument("--missingness-threshold", type=float, default=DEFAULT_FAIL_MISSINGNESS_THRESHOLD, help="Missingness threshold used by --fail-on-high-missingness and rebuild-time feature dropping. Expressed as a 0-1 fraction.")
    args = parser.parse_args()

    input_df = _load_input_dataframe(args.input_path)
    dataset_kind, trigger_columns = _classify_input_dataframe(input_df)
    _print_input_classification(args.input_path, dataset_kind, trigger_columns, args.mode)

    if args.mode == "rebuild":
        if dataset_kind != "raw_statcast":
            raise ValueError(
                "Rebuild mode requires raw Statcast-style pitch-level input. "
                f"Received {dataset_kind} based on columns {trigger_columns}."
            )
        print("[FEATURE-AUDIT] Executing rebuild branch: raw Statcast -> plate appearances -> batter-game -> rolling features -> final export.")
        player_game_df, pitcher_game_df = build_player_game_dataset(input_df, debug_feature_audit=args.debug_feature_audit)
        dataset = add_leakage_safe_features(
            player_game_df,
            pitcher_game_df,
            debug_feature_audit=args.debug_feature_audit,
            missingness_threshold=args.missingness_threshold,
        )
        dataset_type = "rebuilt engineered dataset"
    elif args.mode == "audit":
        if dataset_kind != "engineered_dataset":
            raise ValueError(
                "Audit mode requires an already-engineered batter-game dataset. "
                f"Received {dataset_kind} based on columns {trigger_columns}."
            )
        print("[FEATURE-AUDIT] Executing audit branch: existing engineered dataset only (no raw rebuild performed).")
        dataset = audit_existing_engineered_dataset(
            input_df,
            debug_feature_audit=args.debug_feature_audit,
            missingness_threshold=args.missingness_threshold,
        )
        dataset_type = "audited engineered dataset"
    else:
        if dataset_kind == "raw_statcast":
            print("[FEATURE-AUDIT] Auto mode selected rebuild branch because raw Statcast markers were found.")
            player_game_df, pitcher_game_df = build_player_game_dataset(input_df, debug_feature_audit=args.debug_feature_audit)
            dataset = add_leakage_safe_features(
                player_game_df,
                pitcher_game_df,
                debug_feature_audit=args.debug_feature_audit,
                missingness_threshold=args.missingness_threshold,
            )
            dataset_type = "rebuilt engineered dataset"
        elif dataset_kind == "engineered_dataset":
            print("[FEATURE-AUDIT] Auto mode selected audit branch because engineered-dataset markers were found and raw event-level columns were absent.")
            dataset = audit_existing_engineered_dataset(
                input_df,
                debug_feature_audit=args.debug_feature_audit,
                missingness_threshold=args.missingness_threshold,
            )
            dataset_type = "audited engineered dataset"
        else:
            missing_statcast = sorted(set(STATCAST_COLUMNS) - set(input_df.columns))
            raise ValueError(
                "Input file is neither recognized raw Statcast data nor a batter-game engineered dataset. "
                f"Missing raw Statcast columns include: {missing_statcast[:8]}"
            )

    if args.fail_on_high_missingness:
        _assert_missingness_threshold(dataset, args.missingness_threshold)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Saved {dataset_type} to {args.output} ({len(dataset):,} rows).")


if __name__ == "__main__":
    main()
