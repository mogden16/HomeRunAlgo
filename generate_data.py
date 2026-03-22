"""Build a real 2024 MLB batter-game home run dataset from Statcast and Meteostat."""

from __future__ import annotations

import argparse

import pandas as pd

from config import FINAL_DATA_PATH, MIN_DATASET_WARNING_ROWS, SEASON_END, SEASON_START
from data_sources import build_weather_table, fetch_statcast_season
from feature_engineering import add_leakage_safe_features, build_player_game_dataset, validate_dataset


def generate_mlb_dataset(
    output_path: str = str(FINAL_DATA_PATH),
    start_date: str = SEASON_START,
    end_date: str = SEASON_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Generate a real player-game dataset with leakage-safe historical features."""
    statcast_df = fetch_statcast_season(start_date=start_date, end_date=end_date, force_refresh=force_refresh)
    player_game_df, pitcher_game_df = build_player_game_dataset(statcast_df)
    dataset = add_leakage_safe_features(player_game_df, pitcher_game_df)

    schedule = dataset[["game_date", "team", "opponent", "is_home"]].copy()
    schedule["home_team"] = schedule.apply(lambda row: row["team"] if row["is_home"] else row["opponent"], axis=1)
    weather_df = build_weather_table(schedule[["game_date", "home_team"]].drop_duplicates(), force_refresh=force_refresh)
    dataset["home_team"] = dataset.apply(lambda row: row["team"] if row["is_home"] else row["opponent"], axis=1)
    dataset = dataset.merge(weather_df, on=["game_date", "home_team"], how="left", validate="many_to_one")
    dataset = dataset.drop(columns=["home_team"])

    warnings = validate_dataset(dataset)
    dataset = dataset.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
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
