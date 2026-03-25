"""Build a real 2024 MLB batter-game home run dataset from Statcast and Meteostat."""

from __future__ import annotations

import argparse

import pandas as pd
from pybaseball import playerid_reverse_lookup

from config import FINAL_DATA_PATH, MIN_DATASET_WARNING_ROWS, SEASON_END, SEASON_START
from data_sources import (
    build_hr_park_factor_table,
    build_weather_table,
    fetch_statcast_season,
    print_raw_feature_opportunity_audit,
)
from feature_engineering import (
    CURRENT_ENGINEERED_FEATURE_COLUMNS,
    LEGACY_ENGINEERED_FEATURE_COLUMNS,
    add_leakage_safe_features,
    attach_park_factors,
    build_player_game_dataset,
    validate_dataset,
)

MAX_PLACEHOLDER_SHARE = 0.20


def _print_engineered_schema_summary(dataset: pd.DataFrame) -> None:
    present_current = [column for column in CURRENT_ENGINEERED_FEATURE_COLUMNS if column in dataset.columns]
    missing_current = [column for column in CURRENT_ENGINEERED_FEATURE_COLUMNS if column not in dataset.columns]
    legacy_present = [column for column in LEGACY_ENGINEERED_FEATURE_COLUMNS if column in dataset.columns]

    print("\nEngineered schema export audit")
    print("-" * 60)
    print(f"Current engineered feature count: {len(CURRENT_ENGINEERED_FEATURE_COLUMNS)}")
    print(f"Current engineered features present: {len(present_current)}")
    print(f"Current engineered features missing: {missing_current if missing_current else 'None'}")
    print(f"Legacy engineered features still present: {legacy_present if legacy_present else 'None'}")


def _print_park_factor_audit(audit: dict[str, object]) -> None:
    print("\nPark-factor audit")
    print("-" * 60)
    print(f"Park-factor source used: {audit['source']}")
    print(f"Matched rows: {audit['matched_rows']:,}")
    print(f"Unmatched rows: {audit['unmatched_rows']:,}")
    print(f"park_factor_hr non-null share: {audit['non_null_share']:.2%}")
    print(f"Matched parks ({len(audit['matched_parks'])}): {audit['matched_parks']}")
    print(f"Unmatched parks ({len(audit['unmatched_parks'])}): {audit['unmatched_parks'] if audit['unmatched_parks'] else 'None'}")


def _resolve_batter_name_lookup(statcast_df: pd.DataFrame) -> tuple[dict[int, str], set[int], set[int], set[int]]:
    has_player_name = "player_name" in statcast_df.columns
    non_null_player_name = int(statcast_df["player_name"].notna().sum()) if has_player_name else 0
    batter_ids = set(statcast_df["batter"].dropna().astype(int).tolist())
    raw_lookup: dict[int, str] = {}

    stable_ids: set[int] = set()
    if has_player_name:
        raw_pairs = statcast_df[["batter", "player_name"]].dropna().copy()
        raw_pairs["batter"] = raw_pairs["batter"].astype(int)
        raw_pairs["player_name"] = raw_pairs["player_name"].astype(str).str.strip()
        raw_pairs = raw_pairs[raw_pairs["player_name"] != ""]
        if not raw_pairs.empty:
            stable_name_counts = raw_pairs.groupby("batter")["player_name"].nunique(dropna=True)
            stable_ids = set(stable_name_counts[stable_name_counts == 1].index.astype(int).tolist())
            raw_lookup = (
                raw_pairs[raw_pairs["batter"].isin(stable_ids)]
                .drop_duplicates(subset=["batter"], keep="first")
                .set_index("batter")["player_name"]
                .to_dict()
            )

    unresolved_ids = sorted(batter_ids - set(raw_lookup))
    reverse_lookup: dict[int, str] = {}
    if unresolved_ids:
        reverse_df = playerid_reverse_lookup(unresolved_ids, key_type="mlbam")
        if reverse_df is not None and not reverse_df.empty:
            reverse_df["key_mlbam"] = pd.to_numeric(reverse_df["key_mlbam"], errors="coerce").astype("Int64")
            reverse_df = reverse_df.dropna(subset=["key_mlbam"]).copy()
            reverse_df["full_name"] = (
                reverse_df["name_first"].fillna("").astype(str).str.strip()
                + " "
                + reverse_df["name_last"].fillna("").astype(str).str.strip()
            ).str.strip()
            reverse_df = reverse_df[reverse_df["full_name"] != ""]
            reverse_lookup = (
                reverse_df.drop_duplicates(subset=["key_mlbam"], keep="first")
                .set_index("key_mlbam")["full_name"]
                .to_dict()
            )

    resolved_lookup = dict(raw_lookup)
    reverse_ids: set[int] = set()
    for batter_id in unresolved_ids:
        reverse_name = reverse_lookup.get(batter_id)
        if reverse_name:
            resolved_lookup[batter_id] = reverse_name
            reverse_ids.add(batter_id)

    placeholder_ids = batter_ids - set(resolved_lookup)

    print("Batter-name source audit")
    print("-" * 60)
    print(f"Raw Statcast has player_name column: {has_player_name}")
    print(f"Raw Statcast non-null player_name rows: {non_null_player_name:,}")
    print(f"Raw Statcast stable batter_id -> player_name mappings: {len(stable_ids):,}")
    print(f"Source count (raw Statcast lookup): {len(raw_lookup):,} batter_ids")
    print(f"Source count (reverse lookup fallback): {len(reverse_ids):,} batter_ids")
    print(f"Source count (placeholder fallback): {len(placeholder_ids):,} batter_ids")

    return resolved_lookup, set(raw_lookup), reverse_ids, placeholder_ids


def _print_batter_identity_validation(dataset: pd.DataFrame, stage: str) -> tuple[int, float, list[int]]:
    placeholder_mask = dataset["batter_name"].fillna("").str.match(r"^batter_\d+$")
    placeholder_rows = int(placeholder_mask.sum())
    placeholder_share = float(placeholder_rows / len(dataset)) if len(dataset) else 0.0
    unresolved_ids = sorted(dataset.loc[placeholder_mask, "player_id"].dropna().astype(int).unique().tolist())
    sample_pairs = dataset[["player_id", "batter_name"]].drop_duplicates().head(10)

    print(f"\nBatter identity validation: {stage}")
    print("-" * 60)
    print(f"Rows: {len(dataset):,}")
    print(f"Distinct batter_id count: {dataset['player_id'].nunique(dropna=True):,}")
    print(f"Distinct batter_name count: {dataset['batter_name'].nunique(dropna=True):,}")
    print(f"Placeholder batter_<id> rows: {placeholder_rows:,}")
    print(f"Placeholder share %: {placeholder_share * 100:.2f}%")
    print("Sample batter_id / batter_name pairs:")
    print(sample_pairs.to_string(index=False))
    return placeholder_rows, placeholder_share, unresolved_ids


def generate_mlb_dataset(
    output_path: str = str(FINAL_DATA_PATH),
    start_date: str = SEASON_START,
    end_date: str = SEASON_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Generate a real player-game dataset with leakage-safe historical features."""
    statcast_df = fetch_statcast_season(start_date=start_date, end_date=end_date, force_refresh=force_refresh)
    print_raw_feature_opportunity_audit()
    batter_name_lookup, raw_source_ids, reverse_source_ids, _ = _resolve_batter_name_lookup(statcast_df)
    park_factor_df = build_hr_park_factor_table(statcast_df, season=pd.Timestamp(end_date).year, force_refresh=force_refresh)

    player_game_df, pitcher_game_df = build_player_game_dataset(statcast_df)
    player_game_df, park_factor_audit = attach_park_factors(player_game_df, park_factor_df)
    player_game_df["player_id"] = pd.to_numeric(player_game_df["player_id"], errors="coerce").astype("Int64")
    player_game_df["batter_name"] = player_game_df["player_id"].map(batter_name_lookup)
    missing_batter_name = player_game_df["batter_name"].isna() & player_game_df["player_id"].notna()
    player_game_df.loc[missing_batter_name, "batter_name"] = (
        "batter_" + player_game_df.loc[missing_batter_name, "player_id"].astype(int).astype(str)
    )
    player_game_df["batter_name_source"] = player_game_df["player_id"].map(
        lambda batter_id: (
            "raw_statcast"
            if pd.notna(batter_id) and int(batter_id) in raw_source_ids
            else "reverse_lookup"
            if pd.notna(batter_id) and int(batter_id) in reverse_source_ids
            else "placeholder"
        )
    )

    pitcher_name_lookup = {}
    if "player_name" in statcast_df.columns:
        pitcher_pairs = statcast_df[["pitcher", "player_name"]].dropna().copy()
        pitcher_pairs["pitcher"] = pitcher_pairs["pitcher"].astype(int)
        stable_pitcher_counts = pitcher_pairs.groupby("pitcher")["player_name"].nunique(dropna=True)
        stable_pitcher_ids = set(stable_pitcher_counts[stable_pitcher_counts == 1].index.astype(int).tolist())
        pitcher_name_lookup = (
            pitcher_pairs[pitcher_pairs["pitcher"].isin(stable_pitcher_ids)]
            .drop_duplicates(subset=["pitcher"], keep="first")
            .set_index("pitcher")["player_name"]
            .to_dict()
        )
    player_game_df["pitcher_name"] = player_game_df["opp_pitcher_id"].map(pitcher_name_lookup)

    _, first_share, _ = _print_batter_identity_validation(player_game_df, stage="post batter_name assignment")

    dataset = add_leakage_safe_features(player_game_df, pitcher_game_df)

    schedule = dataset[["game_date", "team", "opponent", "is_home"]].copy()
    schedule["home_team"] = schedule.apply(lambda row: row["team"] if row["is_home"] else row["opponent"], axis=1)
    weather_df = build_weather_table(schedule[["game_date", "home_team"]].drop_duplicates(), force_refresh=force_refresh)
    dataset["home_team"] = dataset.apply(lambda row: row["team"] if row["is_home"] else row["opponent"], axis=1)
    dataset = dataset.merge(weather_df, on=["game_date", "home_team"], how="left", validate="many_to_one")
    dataset = dataset.drop(columns=["home_team"])

    _, final_share, unresolved_ids = _print_batter_identity_validation(dataset, stage="post all feature merges")
    if final_share > MAX_PLACEHOLDER_SHARE:
        print(f"Unresolved batter_ids sample (up to 50): {unresolved_ids[:50]}")
        raise RuntimeError(
            f"Placeholder batter-name share {final_share:.2%} exceeds allowed {MAX_PLACEHOLDER_SHARE:.2%} "
            "after raw + reverse lookup resolution."
        )
    if first_share > MAX_PLACEHOLDER_SHARE:
        print("WARNING: placeholder share exceeded threshold after initial assignment but recovered after merges.")

    warnings = validate_dataset(dataset)
    dataset = dataset.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
    _print_park_factor_audit(park_factor_audit)
    _print_engineered_schema_summary(dataset)
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Saved real MLB player-game dataset to {output_path} ({len(dataset):,} rows).")
    if len(dataset) < MIN_DATASET_WARNING_ROWS:
        print("WARNING: dataset is smaller than expected; source availability may be limited.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(FINAL_DATA_PATH), help="Output CSV path.")
    parser.add_argument("--start-date", default=SEASON_START, help="Inclusive Statcast start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=SEASON_END, help="Inclusive Statcast end date (YYYY-MM-DD).")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached raw files and re-pull remote data.")
    args = parser.parse_args()
    generate_mlb_dataset(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        force_refresh=args.force_refresh,
    )
